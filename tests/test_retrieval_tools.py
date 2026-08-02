"""The two tools the Researcher chooses between.

The most important assertion here is
``TestPayload::test_names_the_source_for_every_candidate``. In the task 1 spike the
canned tool returned only ``[id] text``, and the model filled ``GroundedFact.source``
with the id -- ``"moon#1"`` where the design promises ``"NASA -- Earth's Moon"``.
Nothing failed; the attribution was simply wrong, in the one feature whose purpose
is being able to say where a claim came from. A tool can only be quoted on what it
actually returns.

Two tools rather than one with an ``index=`` argument, so *which* index the agent
consulted is visible in the transcript rather than being a judgement about a
paragraph.
"""

from pathlib import Path

from sparkstory.retrieval.chunks import Chunk, SourceKind
from sparkstory.retrieval.embed import FakeEmbedder
from sparkstory.retrieval.hybrid import HybridIndex
from sparkstory.retrieval.store import LocalVectorStore
from sparkstory.retrieval.tools import NOTHING_FOUND, build_retrieval_tools

CHUNKS = [
    ("moon#1", "The Moon has no air, so nothing can flutter.", SourceKind.FACT),
    ("moon#2", "The Moon's gravity is weaker than Earth's.", SourceKind.FACT),
    (
        "goose#1",
        "Refrain: one line comes back unchanged every time round.",
        SourceKind.CRAFT,
    ),
    ("goose#2", "Three attempts, the third one working.", SourceKind.CRAFT),
]


def tools_over(root: Path, chunks: list[Chunk] | None = None) -> dict:
    store = LocalVectorStore(root=root, embedder=FakeEmbedder(dimensions=256))
    store.save(
        chunks
        if chunks is not None
        else [
            Chunk(
                chunk_id=cid,
                text=text,
                title="Moon" if kind is SourceKind.FACT else "Mother Goose",
                source="NASA -- Earth's Moon"
                if kind is SourceKind.FACT
                else "Mother Goose (Project Gutenberg)",
                licence="public domain",
                source_kind=kind,
            )
            for cid, text, kind in CHUNKS
        ]
    )
    return {tool.name: tool for tool in build_retrieval_tools(HybridIndex(store=store))}


class TestToolShape:
    def test_builds_exactly_two_tools(self, tmp_path: Path) -> None:
        assert set(tools_over(tmp_path)) == {"search_facts", "search_craft"}

    def test_each_tool_describes_when_to_use_it(self, tmp_path: Path) -> None:
        """The description is prompt text: it is how the agent decides which index
        a premise needs, and an unhelpful one turns index selection into a coin
        flip."""
        for tool in tools_over(tmp_path).values():
            assert tool.description
            assert len(tool.description) > 80, tool.name

    def test_descriptions_do_not_leak_our_machinery(self, tmp_path: Path) -> None:
        """Same audit the node prompts get. "Search the vector store" tells a
        children's-book researcher nothing and spends its attention."""
        banned = (
            "langchain",
            "pydantic",
            "json",
            "schema",
            "workflow",
            "pipeline",
            "vector",
            "embedding",
            "bm25",
            "rubric",
        )
        for tool in tools_over(tmp_path).values():
            lowered = tool.description.lower()
            for term in banned:
                assert term not in lowered, f"{tool.name} leaks {term!r}"


class TestPayload:
    def test_names_the_source_for_every_candidate(self, tmp_path: Path) -> None:
        """The spike's defect. Without this the model has no attribution to copy
        and uses the id, producing `source: "moon#1"`."""
        result = tools_over(tmp_path)["search_facts"].invoke(
            {"query": "moon air flutter"}
        )
        assert "NASA -- Earth's Moon" in result

    def test_names_the_chunk_id_for_every_candidate(self, tmp_path: Path) -> None:
        """Provenance: a fact that cannot name its chunk cannot be checked."""
        result = tools_over(tmp_path)["search_facts"].invoke({"query": "moon"})
        assert "moon#1" in result or "moon#2" in result

    def test_returns_the_text_itself(self, tmp_path: Path) -> None:
        result = tools_over(tmp_path)["search_facts"].invoke(
            {"query": "moon air flutter"}
        )
        assert "nothing can flutter" in result

    def test_labels_each_candidate_so_they_cannot_run_together(
        self, tmp_path: Path
    ) -> None:
        """Two facts merged into one blob invite a fact whose claim comes from one
        chunk and whose id comes from another."""
        result = tools_over(tmp_path)["search_facts"].invoke({"query": "moon"})
        assert result.count("id:") >= 2


class TestEachToolPinsItsOwnIndex:
    def test_facts_never_returns_craft(self, tmp_path: Path) -> None:
        """A rhyme returned by the fact tool would be cited as a fact about the
        world."""
        result = tools_over(tmp_path)["search_facts"].invoke(
            {"query": "refrain repeats line"}
        )
        assert "goose#" not in result

    def test_craft_never_returns_facts(self, tmp_path: Path) -> None:
        result = tools_over(tmp_path)["search_craft"].invoke({"query": "moon air"})
        assert "moon#" not in result


class TestNothingFound:
    def test_an_empty_index_says_so_rather_than_raising(self, tmp_path: Path) -> None:
        """Fail open all the way to the tool: the agent is told there is nothing,
        and its prompt authorises returning no facts."""
        store = LocalVectorStore(root=tmp_path / "nope", embedder=FakeEmbedder())
        tools = {t.name: t for t in build_retrieval_tools(HybridIndex(store=store))}
        assert tools["search_facts"].invoke({"query": "the moon"}) == NOTHING_FOUND

    def test_a_kind_with_no_chunks_says_so(self, tmp_path: Path) -> None:
        facts_only = [
            Chunk(
                chunk_id=cid,
                text=text,
                title="Moon",
                source="NASA",
                licence="public domain",
                source_kind=kind,
            )
            for cid, text, kind in CHUNKS
            if kind is SourceKind.FACT
        ]
        tools = tools_over(tmp_path, facts_only)
        assert tools["search_craft"].invoke({"query": "refrain"}) == NOTHING_FOUND


class TestTopK:
    def test_defaults_to_the_configured_top_k(self, tmp_path: Path) -> None:
        """Not hardcoded: RETRIEVAL_TOP_K decides how many candidates the agent
        gets to rerank, which is the one knob that trades tokens for recall."""
        from sparkstory.config import settings

        tools = tools_over(tmp_path)
        result = tools["search_facts"].invoke({"query": "moon"})
        assert result.count("id:") <= settings.retrieval_top_k

    def test_the_agent_may_ask_for_fewer(self, tmp_path: Path) -> None:
        result = tools_over(tmp_path)["search_facts"].invoke(
            {"query": "moon", "top_k": 1}
        )
        assert result.count("id:") == 1
