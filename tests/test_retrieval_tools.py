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

from conftest import FakeChunkStore

from sparkstory.retrieval.chunks import Chunk, SourceKind
from sparkstory.retrieval.tools import NOTHING_FOUND, build_retrieval_tools
from sparkstory.retrieval.types import SearchHit
from sparkstory.retrieval.web.ledger import WebLedger
from sparkstory.retrieval.web.providers import WebResult

CHUNKS = [
    ("moon#1", "The Moon has no air, so nothing can flutter.", SourceKind.FACT),
    ("moon#2", "The Moon's gravity is weaker than Earth's.", SourceKind.FACT),
]


def tools_over(root: Path, chunks: list[Chunk] | None = None) -> dict:
    store = FakeChunkStore()
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
    return {tool.name: tool for tool in build_retrieval_tools(store)}


def web_tools_over(root: Path, ledger: WebLedger, searcher=None, verifier=None) -> dict:
    """Tools with the web one enabled, everything injected so nothing goes out."""
    store = FakeChunkStore()
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
            store,
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
        assert set(tools_over(tmp_path)) == {"search_facts"}

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


# `TestEachToolPinsItsOwnIndex` stood here. Both its tests asserted that
# search_facts never returned a craft chunk and search_craft never returned a
# fact -- the guarantee that made the two-tool split meaningful. With one tool
# and one kind there is nothing left to pin apart.


class TestNothingFound:
    def test_an_empty_index_says_so_rather_than_raising(self, tmp_path: Path) -> None:
        """Fail open all the way to the tool: the agent is told there is nothing,
        and its prompt authorises returning no facts."""
        store = FakeChunkStore()
        tools = {t.name: t for t in build_retrieval_tools(store)}
        assert tools["search_facts"].invoke({"query": "the moon"}) == NOTHING_FOUND

    # `test_a_kind_with_no_chunks_says_so` stood here: it built a facts-only
    # corpus and asserted search_craft reported NOTHING_FOUND rather than raising.
    # With one tool and one kind there is no second index to be empty, and the
    # empty-store case above already covers the fail-open behaviour.


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


class TestAThinResultIsDistinguishableFromAStrongOne:
    """The Researcher is told to search again when a search comes back weak, and
    it can only do that if a weak search *looks* different from a good one.

    Before this, one hit and five hits rendered as the same shape -- labelled
    blocks, no total -- so "notice that the collection barely covers this" was an
    instruction with nothing to act on.
    """

    @staticmethod
    def _hits(n: int) -> list[SearchHit]:
        return [
            SearchHit(
                chunk=Chunk(
                    chunk_id=f"moon#{i}",
                    text="The Moon has no air, so nothing can flutter.",
                    title="Moon",
                    source="NASA -- Earth's Moon",
                    licence="public domain",
                    source_kind=SourceKind.FACT,
                ),
                similarity=0.5,
            )
            for i in range(1, n + 1)
        ]

    def test_the_count_is_stated(self) -> None:
        from sparkstory.retrieval.tools import _render

        assert _render(self._hits(3)).startswith("3 candidates found.")

    def test_one_hit_says_candidate_singular(self) -> None:
        """Pedantic on purpose: "1 candidates found" reads as a bug in the tool,
        and an agent that distrusts its instrument reasons around it."""
        from sparkstory.retrieval.tools import _render

        assert _render(self._hits(1)).startswith("1 candidate found.")

    def test_every_candidate_still_renders_its_own_labelled_block(self) -> None:
        """The count is added *beside* the blocks, not instead of one. Merging
        candidates into a summary is what invites a fact whose claim comes from
        one chunk and whose id comes from another.
        """
        from sparkstory.retrieval.tools import _render

        rendered = _render(self._hits(2))

        assert "id: moon#1" in rendered
        assert "id: moon#2" in rendered

    def test_nothing_found_is_still_phrased_as_an_answer(self) -> None:
        """Guards the change that invited a second search from breaking the
        wording that stopped over-searching.

        An earlier instruction here said coming back with nothing was "a good
        outcome and a common one", which stopped invention and also stopped
        grounding: a story about visiting the Moon came back with zero facts. The
        fix was a discriminating question rather than a blanket permission. This
        message must keep reading as a result rather than a failure, or the
        Researcher will treat an empty collection as something to try harder at.
        """
        assert "There may be nothing to find" in NOTHING_FOUND
        for word in ("error", "failed", "unavailable", "try again"):
            assert word not in NOTHING_FOUND.lower()
