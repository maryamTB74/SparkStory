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
from sparkstory.retrieval.web.ledger import WebLedger
from sparkstory.retrieval.web.providers import WebResult

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


def web_tools_over(root: Path, ledger: WebLedger, searcher=None, verifier=None) -> dict:
    """Tools with the web one enabled, everything injected so nothing goes out."""
    store = LocalVectorStore(root=root, embedder=FakeEmbedder(dimensions=256))
    store.save(
        [
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
    return {
        tool.name: tool
        for tool in build_retrieval_tools(
            HybridIndex(store=store),
            ledger=ledger,
            searcher=searcher,
            verifier=verifier,
        )
    }


async def _one_verified_source(query: str):
    return [
        WebResult(
            url="https://example.org/submarines",
            title="How submarines work",
            text="A submarine sinks by letting water into its ballast tanks.",
            query=query,
            verified=True,
        )
    ]


class TestToolShape:
    def test_builds_exactly_two_tools(self, tmp_path: Path) -> None:
        assert set(tools_over(tmp_path)) == {"search_facts", "search_craft"}

    def test_the_web_tool_is_absent_without_a_ledger(self, tmp_path: Path) -> None:
        """No ledger means the feature is off, and off must mean *not built*.

        MAX_WEB_SEARCHES=0 is what keeps the suite offline, and a tool that
        exists but refuses at call time would still have constructed a client.
        """
        assert "search_web" not in tools_over(tmp_path)

    def test_the_web_tool_appears_when_enabled(self, tmp_path: Path) -> None:
        tools = web_tools_over(tmp_path, WebLedger(), searcher=_one_verified_source)
        assert "search_web" in tools

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


class TestWebTool:
    """The web tool, with search and verification injected.

    Nothing here reaches the network. The live half -- that Perplexity answers in
    this shape and that Firecrawl can read the pages it names -- is what task 12's
    runs are for.
    """

    async def test_a_verified_source_enters_the_ledger(self, tmp_path: Path) -> None:
        ledger = WebLedger()
        tools = web_tools_over(tmp_path, ledger, searcher=_one_verified_source)
        await tools["search_web"].ainvoke({"query": "how does a submarine sink"})
        assert len(ledger) == 1
        assert ledger.sources[0].url == "https://example.org/submarines"

    async def test_the_result_shows_the_id_the_agent_must_copy(
        self, tmp_path: Path
    ) -> None:
        """Rendered in the same `id:` shape as a corpus chunk, deliberately.

        The Researcher's existing "copy each identifier exactly" instruction then
        covers web results with no prompt change, and `web:1` cannot be confused
        with `moon#1` by a reader or by a prefix test.
        """
        tools = web_tools_over(tmp_path, WebLedger(), searcher=_one_verified_source)
        result = await tools["search_web"].ainvoke({"query": "submarines"})
        assert "web:1" in result

    async def test_an_unverified_source_is_not_offered(self, tmp_path: Path) -> None:
        """The point of the whole feature.

        A source that failed the fetch-and-check never reaches the agent, so it
        cannot be cited -- rather than reaching it and being dropped later, which
        would spend the agent's attention on something already known to be bad.
        """

        async def unverifiable(query: str):
            return []

        ledger = WebLedger()
        tools = web_tools_over(tmp_path, ledger, searcher=unverifiable)
        result = await tools["search_web"].ainvoke({"query": "submarines"})
        assert result == NOTHING_FOUND
        assert len(ledger) == 0

    async def test_finding_nothing_uses_the_same_words_as_the_local_tools(
        self, tmp_path: Path
    ) -> None:
        """One vocabulary for "nothing found", so the agent needs one rule."""

        async def nothing(query: str):
            return []

        tools = web_tools_over(tmp_path, WebLedger(), searcher=nothing)
        assert await tools["search_web"].ainvoke({"query": "x"}) == NOTHING_FOUND

    def test_the_description_does_not_leak_our_machinery(self, tmp_path: Path) -> None:
        """Same audit the other tools get, plus the words this feature adds.

        "ledger" and "provenance" are ours, not the researcher's business, and
        naming a vendor tells a children's-book researcher nothing while spending
        its attention.
        """
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
            "ledger",
            "provenance",
            "firecrawl",
            "perplexity",
        )
        tools = web_tools_over(tmp_path, WebLedger(), searcher=_one_verified_source)
        lowered = tools["search_web"].description.lower()
        for term in banned:
            assert term not in lowered, f"search_web leaks {term!r}"

    async def test_ids_keep_counting_across_calls(self, tmp_path: Path) -> None:
        """Two searches in one run must not both mint web:1."""
        ledger = WebLedger()
        tools = web_tools_over(tmp_path, ledger, searcher=_one_verified_source)
        first = await tools["search_web"].ainvoke({"query": "a"})
        second = await tools["search_web"].ainvoke({"query": "b"})
        assert "web:1" in first
        assert "web:2" in second
