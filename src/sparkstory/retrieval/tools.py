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

from langchain_core.tools import BaseTool, tool

from sparkstory.config import settings
from sparkstory.retrieval.chunks import SourceKind
from sparkstory.retrieval.hybrid import HybridIndex
from sparkstory.retrieval.store import SearchHit
from sparkstory.utils.logging_utils import get_logger

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


def build_retrieval_tools(index: HybridIndex) -> list[BaseTool]:
    """Build the two tools over a given index.

    A factory taking the index rather than two module-level tools reading a global,
    because a ``@tool`` function's arguments belong to the *model* -- the index is
    not something an agent should be able to choose. Closing over it keeps the
    injection principle intact and lets a test point the tools at ``tmp_path``.
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

    return [search_facts, search_craft]
