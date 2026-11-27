from __future__ import annotations

import fnmatch
import hashlib
import os
import stat
import time
import uuid
from dataclasses import dataclass
from typing import Any

from .errors import SandboxViolationError

#: Never expose these names through file tools (credential files, checkpoint DBs,
#: VCS internals). Matched against every path component of the resolved path.
DEFAULT_DENY_PATTERNS = (".env", ".env.*", "*.db", "*.db-wal", "*.db-shm", ".git",
                         ".ssh", "id_rsa", "*.pem", "*.key")

#: Hash files only up to this size for fs_metadata.
MAX_HASH_BYTES = 10 * 1024 * 1024

#: Hard cap on a single written file (fs_write/fs_update), independent of the
#: gateway input cap.
MAX_WRITE_BYTES = 1024 * 1024


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

    Platform notes: race-resistant containment requires O_NOFOLLOW plus post-open
    fd re-verification via /proc/self/fd (Linux). On platforms where that mechanism
    is unavailable (Windows, macOS, ...), realpath containment alone is NOT
    sufficient — therefore ALL file operations (fs_read/fs_list/fs_metadata/
    fs_write/fs_update) are DISABLED BY DEFAULT on such platforms (fail closed,
    tools are not even registered) unless the operator explicitly passes
    allow_insecure_fs=True, accepting the documented residual TOCTOU risk.
    """

    def __init__(self, root: str, deny_patterns: list[str] | None = None,
                 write_extensions: list[str] | None = None, *,
                 platform: str | None = None, fd_reverify: bool | None = None,
                 allow_insecure_fs: bool = False) -> None:
        if not root or "\x00" in root:
            raise SandboxViolationError("invalid workspace root")
        self.root = os.path.realpath(root)
        os.makedirs(self.root, exist_ok=True)
        self.deny_patterns = tuple(deny_patterns or DEFAULT_DENY_PATTERNS)
        self.write_extensions = tuple(
            (e.lower() for e in write_extensions) if write_extensions is not None
            else (".txt", ".md")
        )
        self.platform = platform or os.name
        if fd_reverify is None:
            fd_reverify = self.platform == "posix" and os.path.isdir("/proc/self/fd")
        #: race-resistant containment is only claimed when post-open fd
        #: re-verification is available (Linux /proc/self/fd)
        self.fd_reverify = bool(fd_reverify)
        self.race_resistant = self.fd_reverify
        self.allow_insecure_fs = bool(allow_insecure_fs)

    def _fs_guard(self) -> None:
        """Fail closed when race-resistant containment cannot be guaranteed on this
        platform (no post-open fd re-verification), unless the operator explicitly
        accepted the residual TOCTOU risk with allow_insecure_fs=True."""
        if not self.race_resistant and not self.allow_insecure_fs:
            raise SandboxViolationError(
                f"file operations are disabled by default on platform {self.platform!r}: "
                "race-resistant containment requires post-open fd re-verification "
                "(/proc/self/fd), which is unavailable here; realpath containment alone "
                "is not sufficient (fail closed)"
            )

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
        self._fs_guard()
        resolved = self.resolve(relpath)
        # pre-check BEFORE opening: reject non-regular files (fifos/devices would
        # otherwise block or misbehave on open)
        try:
            pre = os.stat(resolved)
        except OSError as exc:
            raise SandboxViolationError(f"cannot stat {relpath!r}: {exc}") from exc
        if not stat.S_ISREG(pre.st_mode):
            raise SandboxViolationError(f"{relpath!r} is not a regular file")
        if pre.st_nlink > 1:
            # a hard link means another name exists for this inode, possibly OUTSIDE
            # the workspace (path containment cannot catch hard links)
            raise SandboxViolationError(
                f"{relpath!r} has {pre.st_nlink} hard links — refusing (possible link "
                "outside the workspace)"
            )
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
        try:
            fd = os.open(resolved, flags)
        except OSError as exc:
            raise SandboxViolationError(f"cannot open {relpath!r}: {exc.strerror or exc}") from exc
        try:
            st = os.fstat(fd)
            if not stat.S_ISREG(st.st_mode):
                raise SandboxViolationError(f"{relpath!r} is not a regular file")
            if st.st_nlink > 1:
                raise SandboxViolationError(
                    f"{relpath!r} has {st.st_nlink} hard links — refusing (possible link "
                    "outside the workspace)"
                )
            if st.st_size > max_bytes:
                raise SandboxViolationError(
                    f"{relpath!r} is {st.st_size} bytes > limit {max_bytes}"
                )
            # re-verify the OPENED file (TOCTOU): check the fd's real path via /proc
            if self.fd_reverify:
                fd_target = os.path.realpath(f"/proc/self/fd/{fd}")
                if os.path.commonpath([fd_target, self.root]) != self.root:
                    raise SandboxViolationError("opened file escaped the workspace (race)")
                self._deny_check(fd_target)
            return fd, resolved
        except BaseException:
            os.close(fd)
            raise

    def read_text(self, relpath: str, *, max_bytes: int) -> str:
        self._fs_guard()
        fd, _ = self.open_read(relpath, max_bytes=max_bytes)
        try:
            data = os.read(fd, max_bytes + 1)
        finally:
            os.close(fd)
        if len(data) > max_bytes:
            raise SandboxViolationError(f"{relpath!r} exceeds read limit {max_bytes}")
        return data.decode("utf-8", errors="replace")

    def metadata(self, relpath: str) -> FileInfo:
        self._fs_guard()
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

    # ---- writes (atomic replace; approval-bound preconditions) ----
    def sha256_file(self, relpath: str) -> str:
        self._fs_guard()
        fd, _ = self.open_read(relpath, max_bytes=MAX_HASH_BYTES)
        h = hashlib.sha256()
        try:
            while True:
                chunk = os.read(fd, 65_536)
                if not chunk:
                    break
                h.update(chunk)
        finally:
            os.close(fd)
        return h.hexdigest()

    def write_text(self, relpath: str, content: str, *, expected_sha256: str | None = None,
                   require_exists: bool = False,
                   max_bytes: int = MAX_WRITE_BYTES) -> str:
        """Atomically create (require_exists=False: target must NOT exist) or update
        (require_exists=True: target must exist and match expected_sha256) a plaintext
        file inside the workspace. Returns the sha256 of the written content.

        Fail closed: containment + deny patterns (resolve), extension allowlist,
        final component must not be a symlink, parent must be a directory inside the
        root, size cap, and for updates the CURRENT content hash must still equal the
        approved expected_sha256 (TOCTOU: if the file changed after approval, the
        write is rejected and a fresh approval is required). The replace itself is
        atomic (tmp file in the same directory + os.replace), so readers never see a
        partially written file."""
        self._fs_guard()
        resolved = self.resolve(relpath)
        data = content.encode("utf-8")
        if len(data) > max_bytes:
            raise SandboxViolationError(
                f"content is {len(data)} bytes > write limit {max_bytes}"
            )
        ext = os.path.splitext(resolved)[1].lower()
        if self.write_extensions and ext not in self.write_extensions:
            raise SandboxViolationError(
                f"extension {ext!r} is not in the write allowlist "
                f"{sorted(self.write_extensions)}"
            )
        lexical = (os.path.normpath(relpath) if os.path.isabs(relpath)
                   else os.path.normpath(os.path.join(self.root, relpath)))
        if os.path.islink(lexical):
            raise SandboxViolationError(
                f"{relpath!r} is a symlink \u2014 refusing to write through it"
            )
        parent = os.path.dirname(resolved)
        try:
            pst = os.stat(parent)
        except OSError as exc:
            raise SandboxViolationError(f"parent directory missing: {exc}") from exc
        if not stat.S_ISDIR(pst.st_mode) or os.path.commonpath([parent, self.root]) != self.root:
            raise SandboxViolationError("parent directory is not inside the workspace")
        exists = os.path.lexists(resolved)
        if exists:
            st = os.lstat(resolved)
            if not stat.S_ISREG(st.st_mode):
                raise SandboxViolationError(f"{relpath!r} is not a regular file")
            if st.st_nlink > 1:
                raise SandboxViolationError(
                    f"{relpath!r} has {st.st_nlink} hard links — refusing to write a "
                    "file that is also linked outside the workspace"
                )
            if not require_exists:
                raise SandboxViolationError(
                    f"{relpath!r} already exists \u2014 use fs_update with expected_sha256"
                )
            if expected_sha256 is not None:
                current = self.sha256_file(relpath)
                if current != expected_sha256:
                    raise SandboxViolationError(
                        f"{relpath!r} changed after approval (sha256 mismatch) \u2014 "
                        "fresh approval required"
                    )
        elif require_exists:
            raise SandboxViolationError(
                f"{relpath!r} does not exist \u2014 use fs_write to create it"
            )
        # atomic replace: tmp file in the SAME directory (same filesystem)
        tmp = os.path.join(parent, f".{os.path.basename(resolved)}.tmp-{os.getpid()}-"
                           f"{uuid.uuid4().hex}")
        flags = (os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
                 | getattr(os, "O_CLOEXEC", 0))
        try:
            fd = os.open(tmp, flags, 0o600)
            try:
                view = memoryview(data)
                while view:
                    n = os.write(fd, view)
                    view = view[n:]
                os.fsync(fd)
            finally:
                os.close(fd)
            os.replace(tmp, resolved)  # atomic on the same filesystem
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        return hashlib.sha256(data).hexdigest()

    # ---- W5: secure staging + descriptor-relative atomic binary publication ----
    #
    # Staging area for exact-byte-approved artifacts lives INSIDE the workspace at
    # .hive-staging/ (0700, files 0600). Staged bytes are sensitive: they are
    # covered by the same containment/deny rules, are never persisted to SQLite,
    # and are swept by age (sweep_staging). Publication never trusts os.replace
    # alone: the parent directory is opened by handle, the staged file's fd is
    # re-verified via /proc/self/fd, the rename is descriptor-relative
    # (src_dir_fd/dst_dir_fd), and the published file is re-opened (O_NOFOLLOW)
    # and re-hashed afterwards. Requires fd re-verification — fail closed on
    # platforms without it (never a timeout-only/allow_insecure path).

    STAGING_DIRNAME = ".hive-staging"

    def _staging_root(self) -> str:
        self._fs_guard()
        if not self.fd_reverify:
            raise SandboxViolationError(
                "staging requires post-open fd re-verification (/proc/self/fd) — "
                "fail closed on this platform")
        root = self.resolve(self.STAGING_DIRNAME)
        os.makedirs(root, exist_ok=True)
        try:
            os.chmod(root, 0o700)
        except OSError:
            pass
        return root

    def stage_bytes(self, data: bytes, *, max_bytes: int) -> tuple[str, str]:
        """Write bytes to a fresh 0600 staging file; returns (relpath, sha256).
        The relpath is workspace-relative and always under .hive-staging/."""
        root = self._staging_root()
        if not isinstance(data, (bytes, bytearray)):
            raise SandboxViolationError("staged content must be bytes")
        if len(data) > max_bytes:
            raise SandboxViolationError(f"staged content {len(data)}B > cap {max_bytes}B")
        name = f"artifact-{uuid.uuid4().hex}"
        full = os.path.join(root, name)
        flags = (os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
                 | getattr(os, "O_CLOEXEC", 0))
        try:
            fd = os.open(full, flags, 0o600)
            try:
                view = memoryview(bytes(data))
                while view:
                    n = os.write(fd, view)
                    view = view[n:]
                os.fsync(fd)
            finally:
                os.close(fd)
            # post-write path verification (guards against path replacement races)
            if os.path.realpath(full) != os.path.join(os.path.realpath(root), name):
                raise SandboxViolationError("staging path changed during write")
        except BaseException:
            try:
                os.unlink(full)
            except OSError:
                pass
            raise
        return f"{self.STAGING_DIRNAME}/{name}", hashlib.sha256(bytes(data)).hexdigest()

    def read_staged(self, staging_relpath: str, *, max_bytes: int) -> bytes:
        """Read a staged artifact (must be directly under .hive-staging/)."""
        self._fs_guard()
        if not self.fd_reverify:
            raise SandboxViolationError("staged reads require fd re-verification — fail closed")
        root = self._staging_root()
        if not os.path.basename(staging_relpath).startswith("artifact-") \
                or os.path.dirname(staging_relpath) != self.STAGING_DIRNAME:
            raise SandboxViolationError("invalid staging path")
        resolved = self.resolve(staging_relpath)
        if os.path.dirname(resolved) != root:
            raise SandboxViolationError("staging path escapes the staging dir")
        fd, _ = self.open_read(staging_relpath, max_bytes=max_bytes)
        try:
            chunks = []
            remaining = max_bytes + 1
            while remaining > 0:
                chunk = os.read(fd, min(1 << 20, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            data = b"".join(chunks)
        finally:
            os.close(fd)
        if len(data) > max_bytes:
            raise SandboxViolationError("staged artifact exceeds cap")
        return data

    def discard_staged(self, staging_relpath: str) -> None:
        self._fs_guard()
        root = self._staging_root()
        resolved = self.resolve(staging_relpath)
        if os.path.dirname(resolved) != root:
            raise SandboxViolationError("staging path escapes the staging dir")
        try:
            os.unlink(resolved)
        except OSError:
            pass

    def sweep_staging(self, *, max_age_seconds: float = 3600.0) -> int:
        """Delete staged artifacts older than max_age_seconds (orphan/crash cleanup)."""
        self._fs_guard()
        root = self._staging_root()
        now = time.time()
        n = 0
        for name in os.listdir(root):
            full = os.path.join(root, name)
            try:
                if now - os.path.getmtime(full) > max_age_seconds:
                    os.unlink(full)
                    n += 1
            except OSError:
                pass
        return n

    def write_bytes_atomic(self, relpath: str, data: bytes, *,
                           expected_sha256: str | None = None,
                           require_exists: bool = False,
                           max_bytes: int = MAX_WRITE_BYTES) -> str:
        """W5 secure binary publication: containment + deny patterns + extension
        allowlist + symlink/hardlink/special-file rejection + destination
        precondition (expected current hash or expected absence) + size cap +
        staging in the same directory + PARENT DIRECTORY HANDLE + staged-fd
        re-verification + descriptor-relative rename + POST-PUBLISH re-open and
        hash verification. os.replace alone is never trusted. Requires fd
        re-verification — fail closed elsewhere. Returns sha256 of published bytes."""
        self._fs_guard()
        if not self.fd_reverify:
            raise SandboxViolationError(
                "atomic binary publication requires post-open fd re-verification "
                "(/proc/self/fd) — fail closed on this platform")
        resolved = self.resolve(relpath)
        if not isinstance(data, (bytes, bytearray)):
            raise SandboxViolationError("content must be bytes")
        data = bytes(data)
        if len(data) > max_bytes:
            raise SandboxViolationError(f"content {len(data)}B > write limit {max_bytes}B")
        ext = os.path.splitext(resolved)[1].lower()
        if self.write_extensions and ext not in self.write_extensions:
            raise SandboxViolationError(
                f"extension {ext!r} not in write allowlist {sorted(self.write_extensions)}")
        lexical = os.path.normpath(relpath) if os.path.isabs(relpath) \
            else os.path.normpath(os.path.join(self.root, relpath))
        if os.path.islink(lexical):
            raise SandboxViolationError(f"{relpath!r} is a symlink — refusing")
        parent = os.path.dirname(resolved)
        base = os.path.basename(resolved)
        pst = os.stat(parent)
        if not stat.S_ISDIR(pst.st_mode) \
                or os.path.commonpath([parent, self.root]) != self.root:
            raise SandboxViolationError("parent directory is not inside the workspace")
        exists = os.path.lexists(resolved)
        if exists:
            st = os.lstat(resolved)
            if not stat.S_ISREG(st.st_mode):
                raise SandboxViolationError(f"{relpath!r} is not a regular file")
            if st.st_nlink > 1:
                raise SandboxViolationError(
                    f"{relpath!r} has {st.st_nlink} hard links — refusing")
            if not require_exists:
                raise SandboxViolationError(f"{relpath!r} already exists — approval must bind "
                                             "the expected current hash (fs_update semantics)")
            if expected_sha256 is not None:
                current = self.sha256_file(relpath)
                if current != expected_sha256:
                    raise SandboxViolationError(
                        f"{relpath!r} changed after approval (sha256 mismatch) — fresh approval")
        elif require_exists:
            raise SandboxViolationError(f"{relpath!r} does not exist")
        digest = hashlib.sha256(data).hexdigest()
        tmpname = f".{base}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
        dirfd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                        | getattr(os, "O_CLOEXEC", 0))
        try:
            flags = (os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
                     | getattr(os, "O_CLOEXEC", 0))
            fd = os.open(tmpname, flags, 0o600, dir_fd=dirfd)
            try:
                # staged fd re-verification: this fd is the tmp file inside the
                # directory we hold by handle (defeats parent-dir replacement)
                real = os.readlink(f"/proc/self/fd/{fd}")
                if os.path.dirname(os.path.realpath(real)) != os.path.realpath(parent):
                    raise SandboxViolationError("staged fd is not inside the destination dir")
                view = memoryview(data)
                while view:
                    n = os.write(fd, view)
                    view = view[n:]
                os.fsync(fd)
            finally:
                os.close(fd)
            os.rename(tmpname, base, src_dir_fd=dirfd, dst_dir_fd=dirfd)
        except BaseException:
            try:
                os.unlink(tmpname, dir_fd=dirfd)
            except OSError:
                pass
            raise
        finally:
            os.close(dirfd)
        # post-publish verification: re-open the destination, verify regular file,
        # single link, and that the published bytes hash to the approved digest
        pfd = os.open(resolved, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                      | getattr(os, "O_CLOEXEC", 0))
        try:
            st = os.fstat(pfd)
            if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
                raise SandboxViolationError("post-publish verification failed (type/links)")
            h = hashlib.sha256()
            while True:
                chunk = os.read(pfd, 1 << 20)
                if not chunk:
                    break
                h.update(chunk)
        finally:
            os.close(pfd)
        if h.hexdigest() != digest:
            raise SandboxViolationError("post-publish hash mismatch — artifact corrupted")
        return digest

    def list_dir(self, relpath: str, *, max_entries: int) -> list[dict[str, Any]]:
        self._fs_guard()
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
