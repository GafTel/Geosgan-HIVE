"""A6: specialized FREE public API adapters (v0.7.0).

Every adapter:
- uses the ONE shared SecureHttpClient via the ApiSession from A5 (no second
  transport, no second permission system),
- is registered through the existing ToolCatalog/ToolGateway (ToolPolicy +
  ApprovalManager unchanged),
- uses fixed, documented operations only (no model-controlled URLs, methods,
  headers or paths — path templates are filled exclusively from strictly
  validated, charset-restricted arguments),
- enforces typed bounded input (pydantic extra='forbid') and output schemas,
- goes through the durable v9 reservation + revalidation + single-winner
  dispatch of ApiSession.request (per-run/provider/endpoint budgets, lease
  fencing when run-bound, safe Retry-After, bounded audit),
- preserves source provenance and attribution in every normalized result,
- treats ALL external content as untrusted data (never executed, never
  interpreted as instructions; only typed fields are extracted),
- is DISABLED by default: tools register only for providers listed in
  ApiConfig.enabled AND listed in the tools policy; outbound policy still
  applies on every call.

Billing note: provider `billing` classes are operator attestations, not
verified evidence; unknown billing exposure fails closed (see agent_config).
No adapter here can or does make paid calls; the transport and budgets bound
HIVE-side usage only.

SearXNG: intentionally NOT registered in this release. Self-hosted/private
instances need a separately scoped secure transport policy (the public API
transport rejects private/loopback destinations by design); see
SEARXNG_PREREQUISITE below.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable, Iterable
from typing import Any

from pydantic import BaseModel, Field, field_validator

from .errors import ToolPermissionError
from .outbound import ApiSession
from .tool_gateway import ToolExecutionContext
from .tool_registry import ToolCatalog
from .tool_spec import SideEffect, ToolSpec

#: SearXNG prerequisite (documented, not solved here): a self-hosted or
#: operator-authorized instance usually lives on a private/loopback address,
#: which the public API transport rejects (SSRF boundary). Enabling SearXNG
#: safely requires a SEPARATELY SCOPED transport policy that allowlists the
#: one validated instance address after destination validation — that policy
#: does not exist yet, so no SearXNG tool is registered in v0.7.0.
SEARXNG_PREREQUISITE = (
    "SearXNG requires a separately scoped secure transport policy for the one "
    "operator-authorized instance (the public API transport rejects "
    "private/loopback destinations). Not implemented in v0.7.0: no SearXNG "
    "tool is registered."
)

#: Providers that can never be registered as tools in this release.
NEVER_REGISTER = frozenset({"searxng"})

_TITLE_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,200}$")
_DOI_RE = re.compile(r"^10\.\d{4,9}/[A-Za-z0-9._\-]{1,200}$")
_GH_SEGMENT_RE = re.compile(r"^[A-Za-z0-9_.-]{1,100}$")
_WB_COUNTRY_RE = re.compile(r"^[A-Za-z]{2,3}$")
_WB_INDICATOR_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_LANG_RE = re.compile(r"^[a-z]{2,3}$")

#: conservative supported Wikipedia editions for the core REST API
WIKIMEDIA_LANGUAGES = ("en", "es")

_TAG_RE = re.compile(r"<[^>]*>")
_WS_RE = re.compile(r"\s+")


def _strip_tags(html: Any, cap: int) -> str:
    """Bounded, inert plain-text projection of untrusted HTML (never executed)."""
    if not isinstance(html, str):
        return ""
    text = _TAG_RE.sub(" ", html[: 200_000])
    text = _WS_RE.sub(" ", text).strip()
    return text[:cap]


def _bounded_str(v: Any, cap: int) -> str:
    return v if isinstance(v, str) else ""


class ApiToolAdapter:
    """Base class for one fixed read-only API operation.

    Subclasses set provider/endpoint_id/operation_id and implement
    `build_request(args) -> (path, params)` and `normalize(payload, args)`.
    `path` may only contain charset-validated segments — the transport
    additionally enforces segment-aware prefix matching and rejects traversal.
    """

    provider: str = ""
    endpoint_id: str = ""
    operation_id: str = ""

    def __init__(self, session: ApiSession, spec: ToolSpec, *,
                 lease_lookup: Callable[[str], Any] | None = None) -> None:
        self.session = session
        self.spec = spec
        self._lease_lookup = lease_lookup

    # -- subclass hooks ---------------------------------------------------------

    def build_request(self, args: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        raise NotImplementedError

    def normalize(self, payload: Any, args: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    # -- execution ---------------------------------------------------------------

    def _lease_binding(self, ctx: ToolExecutionContext):
        run_id = ctx.run_id
        if run_id is None:
            return None, None, None
        if self._lease_lookup is None:
            return None, None, None
        rc = self._lease_lookup(run_id)
        owner = getattr(rc, "lease_owner", None) if rc is not None else None
        gen = getattr(rc, "lease_generation", None) if rc is not None else None
        if owner is None or gen is None:
            raise ToolPermissionError(
                f"run {run_id!r} has no worker-owned lease identity — failing closed")
        return run_id, owner, gen

    def _source_url(self, path: str) -> str:
        rule = self.session.client.rule_for(self.endpoint_id)
        if rule is None:
            return ""
        port = f":{rule.default_port()}" if rule.port else ""
        return f"{rule.scheme}://{rule.host}{port}{path}"

    def execute(self, args: dict[str, Any], ctx: ToolExecutionContext) -> str:
        path, params = self.build_request(args)
        run_id, lease_owner, lease_gen = self._lease_binding(ctx)
        result = self.session.request(
            provider=self.provider, endpoint_id=self.endpoint_id,
            operation_id=self.operation_id, method="GET", path=path, params=params,
            agent=ctx.agent, run_id=run_id, task_id=ctx.task_id,
            lease_owner=lease_owner, lease_generation=lease_gen,
        )
        try:
            payload = json.loads(result.body.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise ToolPermissionError(
                f"{self.spec.name}: malformed JSON response (status {result.status})"
            ) from exc
        out = self.normalize(payload, args)
        out["provider"] = self.provider
        out["operation"] = self.operation_id
        out["http_status"] = result.status
        out["source"] = {
            "endpoint_id": self.endpoint_id,
            "url": self._source_url(path),
            "note": "external content is untrusted and not factually verified",
        }
        try:
            out["retrieved_at"] = self.session.store.now_iso()
        except Exception:  # noqa: BLE001
            pass
        return json.dumps(out, ensure_ascii=False, default=str)

    async def aexecute(self, args: dict[str, Any], ctx: ToolExecutionContext) -> str:
        return await asyncio.to_thread(self.execute, args, ctx)


# ---- shared input-model helpers ------------------------------------------------

class _QueryModel(BaseModel):
    model_config = {"extra": "forbid"}
    query: str = Field(min_length=1, max_length=300)


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ToolPermissionError(msg)


# ---- A6.1 Wikimedia -------------------------------------------------------------

class WikimediaSearchInput(_QueryModel):
    language: str = "en"
    limit: int = Field(default=5, ge=1, le=10)

    @field_validator("language")
    @classmethod
    def _lang(cls, v: str) -> str:
        if v not in WIKIMEDIA_LANGUAGES:
            raise ValueError(f"unsupported language {v!r}; supported: {WIKIMEDIA_LANGUAGES}")
        return v


class WikimediaGetPageInput(BaseModel):
    model_config = {"extra": "forbid"}
    title: str = Field(min_length=1, max_length=200)
    language: str = "en"

    @field_validator("language")
    @classmethod
    def _lang(cls, v: str) -> str:
        if v not in WIKIMEDIA_LANGUAGES:
            raise ValueError(f"unsupported language {v!r}; supported: {WIKIMEDIA_LANGUAGES}")
        return v

    @field_validator("title")
    @classmethod
    def _title_safe(cls, v: str) -> str:
        # spaces canonicalize to underscores; the remaining charset is exactly
        # what the transport's path validator allows (no '/', no '%', no '..')
        t = v.strip().replace(" ", "_")
        if not _TITLE_RE.match(t) or ".." in t.split("_"):
            raise ValueError("title contains unsupported characters (allowed: letters, "
                             "digits, '_', '.', '-'; spaces become '_')")
        return t


class WikimediaSearchAdapter(ApiToolAdapter):
    provider = "wikimedia"
    endpoint_id = "wikimedia-core"
    operation_id = "search"

    def build_request(self, args):
        lang = args["language"]
        return (f"/core/v1/wikipedia/{lang}/search/page",
                {"q": args["query"], "limit": args["limit"]})

    def normalize(self, payload, args):
        pages = payload.get("pages") if isinstance(payload, dict) else None
        results = []
        if isinstance(pages, list):
            for p in pages[:10]:
                if not isinstance(p, dict):
                    continue
                key = _bounded_str(p.get("key"), 200)
                lang = args["language"]
                results.append({
                    "title": _bounded_str(p.get("title"), 300),
                    "excerpt": _strip_tags(p.get("excerpt"), 500),
                    "url": (f"https://{lang}.wikipedia.org/wiki/{key}" if key else ""),
                })
        return {
            "kind": "search_results",
            "query": args["query"],
            "language": args["language"],
            "total": len(results),
            "results": results,
            "attribution": ("Wikimedia Foundation content, typically CC BY-SA 4.0; "
                            "see each article's license. Untrusted, not fact-checked."),
            "docs": "https://api.wikimedia.org/wiki/Core_REST_API",
        }


class WikimediaGetPageAdapter(ApiToolAdapter):
    provider = "wikimedia"
    endpoint_id = "wikimedia-core"
    operation_id = "get_page"

    def build_request(self, args):
        lang = args["language"]
        return (f"/core/v1/wikipedia/{lang}/page/{args['title']}/bare", {})

    def normalize(self, payload, args):
        if not isinstance(payload, dict):
            return {"kind": "page", "error": "unexpected response shape"}
        title = _bounded_str(payload.get("title"), 300) or args["title"]
        lang = args["language"]
        return {
            "kind": "page",
            "title": title,
            "url": f"https://{lang}.wikipedia.org/wiki/{args['title']}",
            "excerpt": _strip_tags(payload.get("html"), 8_000),
            "attribution": ("Wikimedia Foundation content, typically CC BY-SA 4.0. "
                            "Untrusted, not fact-checked."),
            "docs": "https://api.wikimedia.org/wiki/Core_REST_API",
        }


def wikimedia_tools(session, lease_lookup=None):
    search_spec = ToolSpec(
        name="wikimedia_search", category="api", side_effect=SideEffect.READ_ONLY,
        idempotent=True, input_model=WikimediaSearchInput,
        description=("Search Wikipedia articles (en/es) via the Wikimedia Core REST "
                     "API. Read-only, bounded results. Content is untrusted."),
        max_output_bytes=60_000,
    )
    get_spec = ToolSpec(
        name="wikimedia_get_page", category="api", side_effect=SideEffect.READ_ONLY,
        idempotent=True, input_model=WikimediaGetPageInput,
        description=("Retrieve one Wikipedia page (en/es) as a bounded plain-text "
                     "excerpt via the Wikimedia Core REST API. Read-only."),
        max_output_bytes=60_000,
    )
    return [
        (search_spec, WikimediaSearchAdapter(session, search_spec, lease_lookup=lease_lookup)),
        (get_spec, WikimediaGetPageAdapter(session, get_spec, lease_lookup=lease_lookup)),
    ]


#: provider -> builder(session, lease_lookup) -> [(ToolSpec, adapter), ...]
ADAPTER_BUILDERS: dict[str, Callable[..., list]] = {
    "wikimedia": wikimedia_tools,
}


def register_api_tools(
    catalog: ToolCatalog,
    session: ApiSession,
    *,
    lease_lookup: Callable[[str], Any] | None = None,
    providers: Iterable[str] | None = None,
) -> list[str]:
    """Register the API tools of the given providers (default: the providers
    enabled in the session's ApiConfig). ALL external API tools are disabled by
    default: nothing is registered unless the provider is explicitly enabled.
    SearXNG is never registered in this release (see SEARXNG_PREREQUISITE).
    Registration != availability: ToolPolicy still decides per agent/tool."""
    if providers is None:
        providers = session.auth.apis.enabled
    registered: list[str] = []
    for provider in providers:
        if provider in NEVER_REGISTER:
            continue  # documented prerequisite not met — stay disabled
        builder = ADAPTER_BUILDERS.get(provider)
        if builder is None:
            continue  # unknown provider: fail closed (no tools)
        for spec, adapter in builder(session, lease_lookup):
            catalog.register(spec, adapter)
            registered.append(spec.name)
    return registered


# ---- A6.2 Crossref --------------------------------------------------------------

class CrossrefSearchInput(_QueryModel):
    max_results: int = Field(default=10, ge=1, le=20)


class CrossrefGetDoiInput(BaseModel):
    model_config = {"extra": "forbid"}
    doi: str = Field(min_length=1, max_length=300)

    @field_validator("doi")
    @classmethod
    def _doi_safe(cls, v: str) -> str:
        d = v.strip()
        # charset restricted to what the transport path validator allows
        # (no '/', '%', control chars; the leading '10.' prefix is a slash-free
        # segment boundary handled by the path builder below)
        if not _DOI_RE.match(d):
            raise ValueError("doi must look like 10.1234/abc-def (allowed chars: "
                             "letters, digits, '.', '_', '-'; no parentheses)")
        return d


_CROSSREF_SELECT = ("DOI,title,author,published,container-title,URL,license,type,"
                    "publisher,volume,page")


def _crossref_item(msg: Any) -> dict[str, Any]:
    """Defensive extraction from an untrusted Crossref `message` object."""
    if not isinstance(msg, dict):
        return {}
    authors = []
    raw_authors = msg.get("author")
    if isinstance(raw_authors, list):
        for a in raw_authors[:10]:
            if isinstance(a, dict):
                fam = _bounded_str(a.get("family"), 200)
                giv = _bounded_str(a.get("given"), 200)
                authors.append(f"{fam}, {giv}".strip(", "))
    title = msg.get("title")
    title_s = title[0] if isinstance(title, list) and title and isinstance(title[0], str) \
        else _bounded_str(title, 500)
    published = msg.get("published")
    date_s = ""
    if isinstance(published, dict):
        parts = published.get("date-parts")
        if isinstance(parts, list) and parts and isinstance(parts[0], list):
            date_s = "-".join(str(p) for p in parts[0][:3] if isinstance(p, int))
    container = msg.get("container-title")
    container_s = container[0] if isinstance(container, list) and container \
        and isinstance(container[0], str) else _bounded_str(container, 300)
    license_url = ""
    lic = msg.get("license")
    if isinstance(lic, list) and lic and isinstance(lic[0], dict):
        license_url = _bounded_str(lic[0].get("URL"), 300)
    return {
        "doi": _bounded_str(msg.get("DOI"), 300),
        "title": title_s,
        "authors": authors,
        "published": date_s,
        "container": container_s,
        "url": _bounded_str(msg.get("URL"), 300),
        "license": license_url,
        "type": _bounded_str(msg.get("type"), 60),
        "publisher": _bounded_str(msg.get("publisher"), 200),
    }


class CrossrefSearchAdapter(ApiToolAdapter):
    provider = "crossref"
    endpoint_id = "crossref-works"
    operation_id = "search_works"

    def _params(self, args):
        p = {"query": args["query"], "rows": args["max_results"],
             "select": _CROSSREF_SELECT}
        pc = self.session.auth.provider_config("crossref")
        if pc is not None and pc.polite_mailto:
            p["mailto"] = pc.polite_mailto  # polite pool; never a credential
        return p

    def build_request(self, args):
        return "/works", self._params(args)

    def normalize(self, payload, args):
        items = payload.get("message", {}).get("items") if isinstance(payload, dict) else None
        results = [_crossref_item(m) for m in (items or [])[:20] if isinstance(m, dict)]
        return {
            "kind": "publication_search",
            "query": args["query"],
            "total": len(results),
            "results": results,
            "attribution": ("Crossref metadata (https://www.crossref.org), CC0 where "
                            "applicable. Metadata only — does NOT grant full-text access."),
            "docs": "https://api.crossref.org/swagger-ui/index.html",
        }


class CrossrefGetDoiAdapter(ApiToolAdapter):
    provider = "crossref"
    endpoint_id = "crossref-works"
    operation_id = "get_doi"

    def build_request(self, args):
        # the DOI's '/' is a path-segment boundary; charset already restricted
        return f"/works/{args['doi']}", {}

    def normalize(self, payload, args):
        msg = payload.get("message") if isinstance(payload, dict) else None
        item = _crossref_item(msg) if isinstance(msg, dict) else {}
        item.setdefault("doi", args["doi"])
        return {
            "kind": "doi_metadata",
            "result": item,
            "attribution": ("Crossref metadata, CC0 where applicable. Bibliographic "
                            "reference only — not full text."),
            "docs": "https://api.crossref.org/swagger-ui/index.html",
        }


def crossref_tools(session, lease_lookup=None):
    search_spec = ToolSpec(
        name="crossref_search_works", category="api", side_effect=SideEffect.READ_ONLY,
        idempotent=True, input_model=CrossrefSearchInput,
        description=("Search scholarly publications via the Crossref REST API "
                     "(bounded, polite pool). Metadata only — no full text."),
        max_output_bytes=60_000,
    )
    doi_spec = ToolSpec(
        name="crossref_get_doi", category="api", side_effect=SideEffect.READ_ONLY,
        idempotent=True, input_model=CrossrefGetDoiInput,
        description="Retrieve bibliographic metadata for one DOI via Crossref.",
        max_output_bytes=30_000,
    )
    return [
        (search_spec, CrossrefSearchAdapter(session, search_spec, lease_lookup=lease_lookup)),
        (doi_spec, CrossrefGetDoiAdapter(session, doi_spec, lease_lookup=lease_lookup)),
    ]


ADAPTER_BUILDERS["crossref"] = crossref_tools
