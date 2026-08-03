"""The two tools the Researcher can call.

**Docstrings in this module are prompt text.** A tool's description is how the
agent decides whether to call it at all, so these are written for a children's
book researcher and not for us: they say what kind of question the tool answers,
never how the search works underneath. "Searches the vector index using hybrid
BM25 and cosine similarity" would spend the model's attention on machinery it
cannot act on.

**Two tools rather than one with an index argument.** Lesson 9 describes an agent
choosing between ``search_incident_runbooks`` and ``search_marketing_materials``,
and the reason to follow it here is verification: "did it consult craft for a
premise with no factual spine?" becomes a question about which tool appears in the
transcript, rather than a judgement about a paragraph. Lesson 11's single unified
tool is the multimodal case, which is not ours.

**Every candidate carries its source.** The task 1 spike's canned tool returned
only ``[id] text``, and the model filled ``GroundedFact.source`` with the id --
``"moon#1"`` where the design promises ``"NASA -- Earth's Moon"``. Nothing failed
and the attribution was simply wrong. A model can only quote what it was given.
"""

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from langchain_core.tools import BaseTool, tool

from sparkstory.config import settings
from sparkstory.retrieval.chunks import SourceKind
from sparkstory.retrieval.hybrid import HybridIndex
from sparkstory.retrieval.store import SearchHit

# The ledger is plain data with no client and no key, so importing it eagerly
# costs nothing and keeps the web tool's wiring readable. `providers` and
# `verify` are imported lazily inside the tool: they pull in the web
# dependencies, and at MAX_WEB_SEARCHES=0 nothing should load them.
from sparkstory.retrieval.web.ledger import WebLedger, WebSource
from sparkstory.utils.logging_utils import get_logger

if TYPE_CHECKING:
    from sparkstory.retrieval.web.providers import WebResult

WebSearcher = Callable[[str], Awaitable[list["WebResult"]]]
WebVerifier = Callable[["WebResult"], Awaitable["WebResult | None"]]

logger = get_logger(__name__)

#: What a tool returns when it has nothing. Phrased as an answer rather than an
#: error, because for most premises it is the correct answer: the Researcher's
#: prompt authorises returning no facts, and a message that reads like a failure
#: would push it to try again or to invent something.
NOTHING_FOUND = "Nothing in the collection matches that. There may be nothing to find."


def _render(hits: list[SearchHit]) -> str:
    """Format candidates so each field the agent must copy is labelled.

    One labelled block per candidate rather than a single blob: merged text invites
    a fact whose claim comes from one chunk and whose id comes from another, and
    that mistake is invisible afterwards.

    Rank is shown; the similarity number is not. The number is an RRF score, which
    is not comparable across queries, and showing it would invite the agent to
    reason about a quantity that does not mean what it looks like.
    """
    if not hits:
        return NOTHING_FOUND

    blocks = []
    for rank, hit in enumerate(hits, start=1):
        blocks.append(
            f"{rank}. id: {hit.chunk.chunk_id}\n"
            f"   source: {hit.chunk.source}\n"
            f"   text: {hit.chunk.text}"
        )
    return "\n\n".join(blocks)


def build_retrieval_tools(
    index: HybridIndex,
    ledger: WebLedger | None = None,
    searcher: WebSearcher | None = None,
    verifier: WebVerifier | None = None,
) -> list[BaseTool]:
    """Build the retrieval tools over a given index.

    A factory taking the index rather than module-level tools reading a global,
    because a ``@tool`` function's arguments belong to the *model* -- the index is
    not something an agent should be able to choose. Closing over it keeps the
    injection principle intact and lets a test point the tools at ``tmp_path``.

    Args:
        index: The local corpus.
        ledger: Enables the web tool when given. **Absent means the tool is not
            built at all**, rather than built and refusing: at
            ``MAX_WEB_SEARCHES=0`` no client should exist and no key should be
            read, which is what keeps the test suite offline.
        searcher: How to search the web. Injected in tests.
        verifier: How to check a result. Injected in tests.
    """

    @tool
    def search_facts(query: str, top_k: int = settings.retrieval_top_k) -> str:
        """Look up true things about the real world: space, animals, weather, plants.

        Use this when the story involves something real that it could get wrong --
        the Moon, a fox, rain, a seed. Search for the thing itself, in a few plain
        words, rather than for the whole story idea.

        Many stories need nothing from here. A story about a lost toy, a birthday
        or a feeling has nothing to get factually wrong, and finding nothing is a
        perfectly good outcome.

        Each result is labelled with an id, where it came from, and the text
        itself. Copy the id and the source exactly if you keep it.
        """
        hits = index.search(query, source_kind=SourceKind.FACT, top_k=top_k)
        logger.info("search_facts(%r) -> %d hit(s)", query, len(hits))
        return _render(hits)

    @tool
    def search_craft(query: str, top_k: int = settings.retrieval_top_k) -> str:
        """Look up techniques that make a story a pleasure to read aloud.

        Use this to find a device the story could be built around -- a line that
        repeats, a pattern of three, words chosen for their sound, a question that
        carries a reader across the page turn. Search for the effect you want
        rather than for the story's subject.

        Almost every story can use one of these, but only if it genuinely fits.
        Two are plenty and one is often better.

        Each result is labelled with an id, where it came from, and the text
        itself. Copy the id and the source exactly if you keep it.
        """
        hits = index.search(query, source_kind=SourceKind.CRAFT, top_k=top_k)
        logger.info("search_craft(%r) -> %d hit(s)", query, len(hits))
        return _render(hits)

    tools: list[BaseTool] = [search_facts, search_craft]
    if ledger is None:
        return tools

    @tool
    async def search_web(query: str) -> str:
        """Look up something true that the collection does not cover.

        Use this **only after** searching the collection and finding nothing. The
        collection is chosen and checked; this is neither, so it is a last resort
        rather than a second opinion. Most stories never need it.

        A page is read before anything from it is offered to you, and anything
        that could not be read or did not say what it claimed has already been
        thrown away -- so what comes back is what survived. Sometimes that is
        nothing, and nothing is a perfectly good outcome.

        Each result is labelled with an id, where it came from, and the text
        itself. Copy the id and the source exactly if you keep it.
        """
        results = await (searcher or _default_searcher())(query)

        kept: list[str] = []
        for result in results:
            checked = result if result.verified else await _check(result, verifier)
            if checked is None:
                continue
            source_id = ledger.add(
                WebSource(
                    url=checked.url,
                    title=checked.title,
                    text=checked.text,
                    query=checked.query,
                    verified=checked.verified,
                    evidence=checked.evidence,
                )
            )
            # Rendered in the same shape as a corpus chunk, deliberately: the
            # Researcher's existing "copy each identifier exactly" instruction
            # then covers web results with no prompt change.
            kept.append(
                f"{len(kept) + 1}. id: {source_id}\n"
                f"   source: {checked.url}\n"
                f"   text: {checked.text}"
            )

        logger.info("search_web(%r) -> %d verified source(s)", query, len(kept))
        return "\n\n".join(kept) if kept else NOTHING_FOUND

    tools.append(search_web)
    return tools


async def _check(result: WebResult, verifier: WebVerifier | None):
    """Verify one result, using the injected verifier or the real one."""
    if verifier is not None:
        return await verifier(result)
    from sparkstory.retrieval.web.verify import verify_result

    return await verify_result(result)


def _default_searcher() -> WebSearcher:
    """The real search, imported lazily so the default path never loads it."""
    from sparkstory.retrieval.web.providers import search_web as _search

    async def search(query: str):
        return await _search(query)

    return search
