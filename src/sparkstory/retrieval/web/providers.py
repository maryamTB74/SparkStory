"""Searching the web, behind one seam.

One ``search_web`` normalising whatever a provider returns into
``list[WebResult]``. There is one provider today, and the seam still earns its
place -- adding a second should be one function and one settings value, the same
argument that put ``get_chat_model`` and ``get_embedder`` behind single factories.

**The URL that comes out of here is asserted by the model, not observed.**
Perplexity reads the web and writes a synthesised answer; the per-source URL
arrives in a structured-output field the model filled in. There is no citation
metadata to parse -- the model is prompted to segment its own answer by source.
That cannot be trusted here, because this project already found the same defect
once: an early spike had the model fill ``source`` with a chunk id, and a
plausible fabrication would have survived any check that only looked for *a*
source. So a ``WebResult`` is ``verified=False`` by construction and stays that
way until ``verify`` has fetched the page.

**Tavily was considered and dropped.** Its URLs come structurally from the search
API's response rather than from a model, which would make fabrication impossible
rather than merely detectable. But the fetch catches a fabrication anyway, so
Tavily bought cheapness -- skipping the fetch -- not safety, and the fetch is
being paid for regardless. The dispatch shape below is what makes adding Tavily
later cheap if that trade ever changes.
"""

from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, Field

from sparkstory.config import settings
from sparkstory.entities.exceptions import ConfigurationError
from sparkstory.retrieval.web.ledger import Evidence
from sparkstory.utils.logging_utils import get_logger

logger = get_logger(__name__)

# A provider takes a query and returns raw rows. Typed as a callable rather than
# a class so the test seam is a function -- there is no state to hold, and a
# Protocol for one implementation is the abstraction this codebase defers until a
# second one exists.
WebSearchProvider = Callable[[str], Awaitable[list[dict[str, Any]]]]


# Not prompt text: a WebResult is built by us from a provider's response, never
# bound as an output schema. The thing that *does* reach a model is `text` and
# the ledger id, rendered by the search tool.
class WebResult(BaseModel):
    """One candidate source, before anything has checked it."""

    url: str = Field(description="Where the provider says this came from.")
    title: str = Field(description="What the page is called.")
    text: str = Field(description="The passage the claim would be drawn from.")
    query: str = Field(description="The search that surfaced it.")
    # Always False here, and there is a test pinning that. Only a fetch may set
    # it, and only `verify` performs one.
    verified: bool = Field(
        default=False,
        description="Whether a fetched page was found to support the claim.",
    )
    # True when the URL came out of a search API's response rather than out of a
    # model's structured output. It cannot then be fabricated -- which is what
    # makes it acceptable, and *only* then, when the fetcher itself is the thing
    # that is unavailable. Perplexity's URLs are never structural.
    url_is_structural: bool = Field(
        default=False,
        description="Whether the URL came from an API response, not from a model.",
    )
    # Set by `verify`, carried into the ledger. Kept here rather than only on
    # WebSource so the two never disagree about the same source.
    evidence: Evidence = Field(
        default=Evidence.NONE,
        description="What kind of check cleared this source.",
    )


# Instructs the model to attribute each passage to exactly one page, which is
# what makes per-source URLs possible at all. The one-source-per-section rule is
# the load-bearing line: a merged answer cannot be checked against any single
# page, so it could never be verified afterwards.
WEB_SEARCH_PROMPT = """\
Question: {query}

Answer the question above using what you find on the web.

Organise the answer into sections, one per source. A section must draw on \
exactly one page -- never combine two sources in a section, because each answer \
is checked against its own page afterwards and a combined one cannot be.

Prefer official and reference sources over blogs and opinion. Write plainly \
enough that the answer could be explained to a young child, and keep each \
section under 200 words.

Return a list of objects, each with:
- url: the address of the page this section came from
- title: what the page is called
- answer: what that page says about the question"""


async def search_web(
    query: str,
    provider: WebSearchProvider | None = None,
    fallback: WebSearchProvider | None = None,
) -> list[WebResult]:
    """Search the web and return candidate sources, none of them yet verified.

    Args:
        query: What to look up, in plain words.
        provider: How to search. Defaults to Perplexity; tests inject a fake, so
            nothing here constructs a client or reads a key unless asked to.
        fallback: Used **only if ``provider`` raises**. Defaults to Tavily when a
            key is configured, and to nothing when it is not.

    Returns:
        Zero or more :class:`WebResult`. An empty list is a legitimate answer and
        is treated as one -- the same rule the corpus tools follow, so the
        Researcher sees one vocabulary rather than two.

    **An empty primary result does not trigger the fallback**, and that is a
    deliberate refusal rather than an omission. Most premises have nothing to
    look up, the Researcher's prompt says so explicitly, and retrying "nothing
    found" on a second provider is pressure to invent. That lever has already
    been pulled too far in the other direction once, when an instruction meant to
    stop invention stopped grounding entirely. Only an *unusable* provider falls
    through.

    A malformed row costs that row rather than the run, following the split this
    codebase draws between raising on the impossible and returning for the
    recoverable. One bad row among three should still ground the story in two.
    """
    search = provider or _perplexity_provider()
    structural = False
    try:
        rows = await search(query)
    except Exception as primary_error:
        second = fallback if fallback is not None else _tavily_provider()
        if second is None:
            raise
        logger.warning("web search failed (%s); falling back", primary_error)
        try:
            rows = await second(query)
        except Exception:
            # Both gone. Raise the *first* error: "no web results" and "the web
            # is broken" are different outcomes, and the second must not arrive
            # looking like an empty search.
            raise primary_error from None
        structural = True

    results: list[WebResult] = []
    for row in rows:
        url = str(row.get("url") or "").strip()
        text = str(row.get("answer") or "").strip()
        if not url or not text:
            logger.warning(
                "dropping a web result with no %s", "url" if not url else "text"
            )
            continue
        results.append(
            WebResult(
                url=url,
                title=str(row.get("title") or url).strip(),
                text=text,
                query=query,
                url_is_structural=structural,
            )
        )

    logger.info("web search %r -> %d candidate(s)", query, len(results))
    return results


def _tavily_provider() -> WebSearchProvider | None:
    """Build the fallback, or ``None`` when no key is configured.

    Returning ``None`` rather than a provider that raises matters: without a key
    the behaviour must be exactly what it was before the fallback existed, so an
    absent fallback cannot become a new failure mode of its own.

    The important difference from Perplexity is structural rather than stylistic
    -- ``result["url"]`` comes out of the API's own response, so it is not
    something a model could invent.
    """
    if settings.api_key_for("TAVILY_API_KEY") is None:
        return None

    async def search(query: str) -> list[dict[str, Any]]:
        from tavily import AsyncTavilyClient

        client = AsyncTavilyClient(api_key=settings.api_key_for("TAVILY_API_KEY"))
        response = await client.search(
            query=query,
            search_depth="advanced",
            max_results=settings.retrieval_top_k,
            include_answer=False,
            include_raw_content=False,
        )
        return [
            {
                "url": row.get("url", ""),
                "title": row.get("title", ""),
                "answer": row.get("content", ""),
            }
            for row in response.get("results", [])
        ]

    return search


def _perplexity_provider() -> WebSearchProvider:
    """Build the real provider.

    Imported lazily and constructed only when actually called, so that at
    ``MAX_WEB_SEARCHES=0`` -- the default -- no client exists, no key is read and
    the dependency is never imported. That is what keeps the offline suite
    offline, and it is why this is a factory rather than a module-level object.

    Everything in here is exactly what a unit test cannot check, which is the
    same reason ``build_researcher_agent`` is separate from ``ResearcherNode``.
    """

    async def search(query: str) -> list[dict[str, Any]]:
        from langchain.chat_models import init_chat_model
        from pydantic import BaseModel as _BaseModel

        class _Source(_BaseModel):
            url: str
            title: str = ""
            answer: str

        class _Response(_BaseModel):
            sources: list[_Source]

        api_key = settings.api_key_for("PERPLEXITY_API_KEY")
        if not api_key:
            # ConfigurationError rather than a new ProviderError: an exception
            # class arrives when something raises it, not in advance. `_retry_on`
            # already excludes this class, so a missing key fails once instead of
            # being retried and printing three tracebacks for a problem whose fix
            # is one line in `.env`. A *transient* network failure is a different
            # thing and should retry, which it does: `default_retry_on` returns
            # True for the client's own exceptions.
            raise ConfigurationError(
                "PERPLEXITY_API_KEY is not set, but MAX_WEB_SEARCHES is above 0. "
                "Set the key or set MAX_WEB_SEARCHES=0 to disable web search."
            )

        model = init_chat_model(
            "perplexity:sonar-pro", api_key=api_key
        ).with_structured_output(_Response)
        response = await model.ainvoke(WEB_SEARCH_PROMPT.format(query=query))
        return [source.model_dump() for source in response.sources]

    return search
