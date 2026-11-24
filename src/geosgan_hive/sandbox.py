from __future__ import annotations

import fnmatch
import hashlib
import os
import stat
from dataclasses import dataclass
from typing import Any

from .errors import SandboxViolationError

#: Never expose these names through file tools (credential files, checkpoint DBs,
#: VCS internals). Matched against every path component of the resolved path.
DEFAULT_DENY_PATTERNS = (".env", ".env.*", "*.db", "*.db-wal", "*.db-shm", ".git",
                         ".ssh", "id_rsa", "*.pem", "*.key")

#: Hash files only up to this size for fs_metadata.
MAX_HASH_BYTES = 10 * 1024 * 1024


@dataclass
class FileInfo:
    path: str
    size: int
    is_dir: bool
    mtime: float
    sha256: str | None


class WorkspaceSandbox:
    """Workspace isolation for all file tools.

    Guarantees (fail closed):
    - only paths under the configured workspace root are reachable (absolute paths
      outside the root, `..` traversal and symlink/junction escapes are rejected via
      realpath + commonpath containment);
    - denied name patterns (credentials, checkpoint DBs, VCS dirs) are rejected on
      EVERY path component;
    - only regular files are opened (no devices/special files), with O_NOFOLLOW
      where the platform supports it, and the opened fd is re-verified afterwards
      (race-resistant against concurrent path replacement: on Linux the real path
      of /proc/self/fd/N is checked; elsewhere fstat regular-file + containment of
      the pre-open realpath is the fallback);
    - size caps are enforced before reading.

    Platform notes: O_NOFOLLOW and /proc/self/fd verification are POSIX; on Windows,
    realpath resolves junctions/reparse points and containment is enforced, but
    fd-level re-verification is not available — documented limitation.
    """

    def __init__(self, root: str, deny_patterns: list[str] | None = None) -> None:
        if not root or "\x00" in root:
            raise SandboxViolationError("invalid workspace root")
        self.root = os.path.realpath(root)
        os.makedirs(self.root, exist_ok=True)
        self.deny_patterns = tuple(deny_patterns or DEFAULT_DENY_PATTERNS)

    # ---- path resolution ----
    def _deny_check(self, resolved: str) -> None:
        rel = os.path.relpath(resolved, self.root)
        parts = rel.split(os.sep)
        for part in parts:
            for pattern in self.deny_patterns:
                if fnmatch.fnmatch(part, pattern) or fnmatch.fnmatch(part.lower(), pattern.lower()):
                    raise SandboxViolationError(
                        f"access to '{part}' is denied by workspace policy"
                    )

    def resolve(self, relpath: str) -> str:
        """Resolve a user-supplied relative path to an absolute path inside the
        workspace, rejecting traversal/absolute-outside/symlink escapes/denied names."""
        if not isinstance(relpath, str) or not relpath.strip() or "\x00" in relpath:
            raise SandboxViolationError("invalid path")
        if os.path.isabs(relpath) or (os.name == "nt" and relpath[1:3] == ":\\"):
            # absolute paths are only allowed if they already point inside the root
            candidate = os.path.realpath(relpath)
            if os.path.commonpath([candidate, self.root]) != self.root:
                raise SandboxViolationError("absolute path outside the workspace root")
            resolved = candidate
        else:
            joined = os.path.normpath(os.path.join(self.root, relpath))
            resolved = os.path.realpath(joined)
            if os.path.commonpath([resolved, self.root]) != self.root:
                raise SandboxViolationError(
                    f"path escapes the workspace root (traversal or symlink): {relpath!r}"
                )
        self._deny_check(resolved)
        return resolved

    # ---- race-resistant read ----
    def open_read(self, relpath: str, *, max_bytes: int) -> tuple[int, str]:
        """Open a file for reading race-resistently. Returns (fd, resolved_path).
        Raises SandboxViolationError on any violation (fail closed)."""
        resolved = self.resolve(relpath)
        # pre-check BEFORE opening: reject non-regular files (fifos/devices would
        # otherwise block or misbehave on open)
        try:
            pre = os.stat(resolved)
        except OSError as exc:
            raise SandboxViolationError(f"cannot stat {relpath!r}: {exc}") from exc
        if not stat.S_ISREG(pre.st_mode):
            raise SandboxViolationError(f"{relpath!r} is not a regular file")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
        try:
            fd = os.open(resolved, flags)
        except OSError as exc:
            raise SandboxViolationError(f"cannot open {relpath!r}: {exc.strerror or exc}") from exc
        try:
            st = os.fstat(fd)
            if not stat.S_ISREG(st.st_mode):
                raise SandboxViolationError(f"{relpath!r} is not a regular file")
            if st.st_size > max_bytes:
                raise SandboxViolationError(
                    f"{relpath!r} is {st.st_size} bytes > limit {max_bytes}"
                )
            # re-verify the OPENED file (TOCTOU): on Linux check the fd's real path
            if os.path.isdir("/proc/self/fd"):
                fd_target = os.path.realpath(f"/proc/self/fd/{fd}")
                if os.path.commonpath([fd_target, self.root]) != self.root:
                    raise SandboxViolationError("opened file escaped the workspace (race)")
                self._deny_check(fd_target)
            return fd, resolved
        except BaseException:
            os.close(fd)
            raise

    def read_text(self, relpath: str, *, max_bytes: int) -> str:
        fd, _ = self.open_read(relpath, max_bytes=max_bytes)
        try:
            data = os.read(fd, max_bytes + 1)
        finally:
            os.close(fd)
        if len(data) > max_bytes:
            raise SandboxViolationError(f"{relpath!r} exceeds read limit {max_bytes}")
        return data.decode("utf-8", errors="replace")

    def metadata(self, relpath: str) -> FileInfo:
        resolved = self.resolve(relpath)
        try:
            st = os.stat(resolved, follow_symlinks=False)
        except OSError as exc:
            raise SandboxViolationError(f"cannot stat {relpath!r}: {exc}") from exc
        if stat.S_ISLNK(st.st_mode):
            # resolve() already maps symlinks inside the root; a symlink here means it
            # points outside (escape) — resolve() would have rejected it, so fail closed
            raise SandboxViolationError(f"{relpath!r} is a symlink (escape attempt)")
        sha = None
        if stat.S_ISREG(st.st_mode) and st.st_size <= MAX_HASH_BYTES:
            h = hashlib.sha256()
            fd, _ = self.open_read(relpath, max_bytes=MAX_HASH_BYTES)
            try:
                while True:
                    chunk = os.read(fd, 65_536)
                    if not chunk:
                        break
                    h.update(chunk)
            finally:
                os.close(fd)
            sha = h.hexdigest()
        return FileInfo(
            path=os.path.relpath(resolved, self.root),
            size=st.st_size,
            is_dir=stat.S_ISDIR(st.st_mode),
            mtime=st.st_mtime,
            sha256=sha,
        )

    def list_dir(self, relpath: str, *, max_entries: int) -> list[dict[str, Any]]:
        resolved = self.resolve(relpath)
        if not os.path.isdir(resolved):
            raise SandboxViolationError(f"{relpath!r} is not a directory")
        try:
            names = sorted(os.listdir(resolved))
        except OSError as exc:
            raise SandboxViolationError(f"cannot list {relpath!r}: {exc}") from exc
        out: list[dict[str, Any]] = []
        for name in names[:max_entries]:
            full = os.path.join(resolved, name)
            try:
                st = os.lstat(full)
                out.append({
                    "name": name,
                    "type": ("dir" if stat.S_ISDIR(st.st_mode)
                             else "symlink" if stat.S_ISLNK(st.st_mode)
                             else "file" if stat.S_ISREG(st.st_mode) else "other"),
                    "size": st.st_size,
                })
            except OSError:
                out.append({"name": name, "type": "unknown", "size": 0})
        truncated = len(names) > max_entries
        if truncated:
            out.append({"name": f"...[{len(names) - max_entries} more entries truncated]",
                        "type": "truncated", "size": 0})
        return out
