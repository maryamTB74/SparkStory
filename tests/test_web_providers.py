"""Web search, behind a seam.

Offline throughout: the provider is injected, so these tests never construct a
client and never read a key. The live half -- that Perplexity actually answers in
this shape -- is left to a run against the real endpoint, exactly as
``build_researcher_agent`` was.

**The invariant these tests exist to protect** is that a URL coming out of here
is *model-asserted*. Perplexity returns a synthesised answer and the URL arrives
in a structured-output field the model filled in, so a plausible fabrication is
indistinguishable from a real citation until something fetches it. That is the
same defect that made this project overwrite a fact's ``source`` from the store
rather than trust the agent, and it is why ``verify`` exists.
"""

import pytest

from sparkstory.retrieval.web.providers import WebResult, search_web


class FakeWebSearch:
    """Stands in for a search provider: records the query, returns canned rows."""

    def __init__(self, rows: list[dict] | Exception) -> None:
        self._rows = rows
        self.queries: list[str] = []

    async def __call__(self, query: str) -> list[dict]:
        self.queries.append(query)
        if isinstance(self._rows, Exception):
            raise self._rows
        return self._rows


def a_row(**overrides: object) -> dict:
    payload: dict = {
        "url": "https://example.org/submarines",
        "title": "How submarines work",
        "answer": "A submarine sinks by letting water into its ballast tanks.",
    }
    payload.update(overrides)
    return payload


class TestSearchWeb:
    async def test_returns_one_result_per_source(self) -> None:
        provider = FakeWebSearch([a_row(), a_row(url="https://example.org/2")])
        results = await search_web("how does a submarine sink", provider=provider)
        assert [r.url for r in results] == [
            "https://example.org/submarines",
            "https://example.org/2",
        ]

    async def test_carries_the_query_that_found_it(self) -> None:
        """The ledger records it, and a run artifact reader needs to know what was
        asked to judge whether the answer is on point."""
        provider = FakeWebSearch([a_row()])
        results = await search_web("how does a submarine sink", provider=provider)
        assert results[0].query == "how does a submarine sink"

    async def test_an_empty_result_is_legitimate_not_an_error(self) -> None:
        """Same rule the corpus tools follow, so the Researcher sees one
        vocabulary: finding nothing is an answer, not a failure."""
        assert await search_web("nonsense", provider=FakeWebSearch([])) == []

    async def test_a_row_missing_a_url_is_dropped_not_raised(self) -> None:
        """A malformed row costs that row, not the run.

        Same split this codebase draws everywhere else: `validation.py` raises on
        the impossible, `reviews.py` and `provenance.py` return for the
        recoverable. A provider that returns one bad row among three should still
        ground the story in the other two.
        """
        provider = FakeWebSearch([a_row(), {"title": "no url", "answer": "x"}])
        results = await search_web("q", provider=provider)
        assert len(results) == 1

    async def test_a_row_missing_text_is_dropped(self) -> None:
        provider = FakeWebSearch([a_row(answer="")])
        assert await search_web("q", provider=provider) == []


class TestTheTrustBoundary:
    """The invariant a later edit is most likely to forget."""

    def test_a_result_is_unverified_when_it_arrives(self) -> None:
        """A URL out of search is an assertion, not provenance.

        Nothing between the provider and the ledger may mark it otherwise --
        only a fetch can. If this ever defaults to True, a fabricated URL reaches
        a book with a citation that was never checked.
        """
        assert WebResult(url="u", title="t", text="x", query="q").verified is False

    def test_the_module_states_that_the_url_is_model_asserted(self) -> None:
        """A comment is not enforcement, but this one is load-bearing enough that
        its removal should fail something. The whole design of `verify` rests on
        nobody downstream treating this URL as a source."""
        import sparkstory.retrieval.web.providers as providers

        assert "asserted" in (providers.__doc__ or "").lower()


class TestTavilyFallback:
    """Tavily stands in only when the primary path is *unusable*.

    Never because a search legitimately found nothing. This project already has
    a scar there: an "empty is fine" instruction meant to stop invention stopped
    grounding instead, and a story about the Moon came back with zero facts.
    Retrying an empty result on a second provider
    is the same lever pushed the other way -- it would make "nothing found",
    which is the correct answer for most premises, into a failure to route
    around.
    """

    async def test_tavily_is_not_called_when_perplexity_works(self) -> None:
        """No wasted call, and no second opinion. The fallback is for outages."""
        primary = FakeWebSearch([a_row()])
        fallback = FakeWebSearch([a_row(url="https://tavily.test/x")])
        results = await search_web("q", provider=primary, fallback=fallback)
        assert [r.url for r in results] == ["https://example.org/submarines"]
        assert fallback.queries == []

    async def test_an_empty_primary_result_does_not_trigger_the_fallback(
        self,
    ) -> None:
        """The assertion this class exists for.

        Finding nothing is a legitimate answer that the Researcher's prompt
        explicitly authorises. Retrying it elsewhere is pressure to invent.
        """
        primary = FakeWebSearch([])
        fallback = FakeWebSearch([a_row(url="https://tavily.test/x")])
        assert await search_web("q", provider=primary, fallback=fallback) == []
        assert fallback.queries == []

    async def test_a_failing_primary_falls_back(self) -> None:
        primary = FakeWebSearch(RuntimeError("perplexity is down"))
        fallback = FakeWebSearch([a_row(url="https://tavily.test/x")])
        results = await search_web("q", provider=primary, fallback=fallback)
        assert [r.url for r in results] == ["https://tavily.test/x"]

    async def test_a_fallback_result_is_marked_as_coming_from_a_search_api(
        self,
    ) -> None:
        """Tavily's URL is structural, and that difference has to be recorded.

        It is what lets the fetcher-unavailable path accept it: a URL from an
        API response cannot be fabricated, unlike one a model wrote.
        """
        primary = FakeWebSearch(RuntimeError("down"))
        fallback = FakeWebSearch([a_row(url="https://tavily.test/x")])
        results = await search_web("q", provider=primary, fallback=fallback)
        assert results[0].url_is_structural is True

    async def test_a_primary_result_is_not_structural(self) -> None:
        results = await search_web("q", provider=FakeWebSearch([a_row()]))
        assert results[0].url_is_structural is False

    async def test_both_failing_raises_the_original_error(self) -> None:
        """ "No web results" and "the web is broken" are different, and the
        second must stay visible rather than looking like an empty search."""
        primary = FakeWebSearch(RuntimeError("perplexity is down"))
        fallback = FakeWebSearch(RuntimeError("tavily is down too"))
        with pytest.raises(RuntimeError, match="perplexity is down"):
            await search_web("q", provider=primary, fallback=fallback)

    async def test_no_fallback_configured_lets_the_primary_error_through(
        self,
    ) -> None:
        """Without a Tavily key the behaviour is exactly what it was before this
        feature existed -- a fallback that is absent must not become a new
        failure mode of its own."""
        with pytest.raises(RuntimeError, match="down"):
            await search_web("q", provider=FakeWebSearch(RuntimeError("down")))
