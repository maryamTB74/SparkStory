"""Fetching a cited page, and checking that it says what was claimed.

**The load-bearing module of the web feature.** With Tavily dropped, this is the
only thing standing between a URL the model typed and a fact in a child's book.

Offline throughout: both the fetcher and the judge are injected.

The judge is a model judging a model, so the playbook's question was asked before
the prompt was written -- *what is the laziest thing that satisfies this?* For
"is this claim supported by this page?" the laziest answer is **yes**. So a
verdict alone is not accepted: the judge must quote the sentence that supports
the claim, and **code, not the model, checks that the quote is really in the
page**. That converts an agreeable "yes" into something falsifiable, which is the
same move `drop_unprovenanced` makes by overwriting `source` from the store
rather than trusting what the agent wrote.
"""

from sparkstory.retrieval.web.ledger import Evidence
from sparkstory.retrieval.web.providers import WebResult
from sparkstory.retrieval.web.verify import (
    ClaimVerdict,
    fetch_page,
    verify_result,
)

PAGE = (
    "# How submarines work\n\n"
    "A submarine sinks by letting water into its ballast tanks. "
    "To rise, compressed air pushes the water back out."
)


class FakeFetcher:
    """Stands in for Firecrawl: returns canned markdown, or None for a failure."""

    def __init__(self, pages: dict[str, str | None]) -> None:
        self._pages = pages
        self.requested: list[str] = []

    async def __call__(self, url: str) -> str | None:
        self.requested.append(url)
        return self._pages.get(url)


class FakeJudge:
    """Stands in for the verdict model."""

    def __init__(self, verdict: ClaimVerdict) -> None:
        self._verdict = verdict
        self.calls = 0

    async def __call__(self, claim: str, page: str) -> ClaimVerdict:
        self.calls += 1
        return self._verdict


def a_result(**overrides: object) -> WebResult:
    payload: dict = {
        "url": "https://example.org/submarines",
        "title": "How submarines work",
        "text": "A submarine sinks by letting water into its ballast tanks.",
        "query": "how does a submarine sink",
    }
    payload.update(overrides)
    return WebResult(**payload)


class TestFetchPage:
    async def test_returns_the_markdown(self) -> None:
        fetcher = FakeFetcher({"https://example.org/submarines": PAGE})
        assert await fetch_page(a_result().url, fetcher=fetcher) == PAGE

    async def test_a_failed_fetch_returns_none_rather_than_raising(self) -> None:
        """A dead link costs the fact, not the book.

        This is the whole reason a fabricated URL is safe to receive: it 404s,
        the fetch returns None, and the fact is dropped by the caller. If this
        raised, one hallucinated URL would kill a run.
        """
        fetcher = FakeFetcher({"https://example.org/submarines": None})
        assert await fetch_page(a_result().url, fetcher=fetcher) is None

    async def test_an_empty_page_is_treated_as_a_failure(self) -> None:
        fetcher = FakeFetcher({"https://example.org/submarines": "   "})
        assert await fetch_page(a_result().url, fetcher=fetcher) is None


class TestVerifyResult:
    async def test_a_supported_claim_with_a_real_quote_passes(self) -> None:
        judge = FakeJudge(
            ClaimVerdict(
                supported=True,
                quote="A submarine sinks by letting water into its ballast tanks.",
            )
        )
        verified = await verify_result(
            a_result(),
            fetcher=FakeFetcher({"https://example.org/submarines": PAGE}),
            judge=judge,
        )
        assert verified is not None
        assert verified.verified is True

    async def test_a_fabricated_quote_fails_even_when_the_verdict_says_yes(
        self,
    ) -> None:
        """The assertion this whole module exists for.

        The judge is a model and the laziest answer to "is this supported?" is
        yes. Requiring a quote and checking it *in code* is what makes the
        verdict falsifiable -- a model that agrees but cannot point at the
        sentence has not verified anything.
        """
        judge = FakeJudge(
            ClaimVerdict(supported=True, quote="Submarines are powered by dolphins.")
        )
        assert (
            await verify_result(
                a_result(),
                fetcher=FakeFetcher({"https://example.org/submarines": PAGE}),
                judge=judge,
            )
            is None
        )

    async def test_an_unsupported_verdict_fails(self) -> None:
        judge = FakeJudge(ClaimVerdict(supported=False, quote=""))
        assert (
            await verify_result(
                a_result(),
                fetcher=FakeFetcher({"https://example.org/submarines": PAGE}),
                judge=judge,
            )
            is None
        )

    async def test_an_unfetchable_page_fails_without_calling_the_judge(self) -> None:
        """A dead URL is settled before any model call. Cheaper, and it means a
        fabricated URL never reaches a judge that might be talked into it."""
        judge = FakeJudge(ClaimVerdict(supported=True, quote="anything"))
        result = await verify_result(
            a_result(),
            fetcher=FakeFetcher({"https://example.org/submarines": None}),
            judge=judge,
        )
        assert result is None
        assert judge.calls == 0

    async def test_quote_matching_ignores_whitespace_and_case(self) -> None:
        """A page's line wrapping is not a reason to reject a real quote.

        Deliberately forgiving on formatting and strict on content: the failure
        being guarded against is an invented sentence, not a reflowed one.
        """
        judge = FakeJudge(
            ClaimVerdict(
                supported=True,
                quote="A  SUBMARINE   sinks\nby letting water into its ballast tanks.",
            )
        )
        verified = await verify_result(
            a_result(),
            fetcher=FakeFetcher({"https://example.org/submarines": PAGE}),
            judge=judge,
        )
        assert verified is not None

    async def test_verification_can_be_skipped_and_says_so(self) -> None:
        """The skip must be visible in the record, never silently equivalent.

        `verify_web_claims=False` exists so a test can switch the network off. A
        source that skipped the check is kept, but marked unverified -- and
        `drop_unprovenanced` drops unverified sources, so nothing reaches a book
        on the strength of a check that never ran.
        """
        judge = FakeJudge(ClaimVerdict(supported=True, quote="unused"))
        fetcher = FakeFetcher({})
        kept = await verify_result(
            a_result(), fetcher=fetcher, judge=judge, verify=False
        )
        assert kept is not None
        assert kept.verified is False
        assert judge.calls == 0
        assert fetcher.requested == []


class TestFetcherUnavailable:
    """When Firecrawl itself is the thing that is down.

    Tavily can replace Perplexity -- same job, different vendor -- but it cannot
    replace Firecrawl, because searching and fetching are different jobs. So the
    degraded path accepts a *structural* URL on the strength of where it came
    from rather than on the strength of a page having been read.

    That is genuinely weaker and is recorded as such: `Evidence.SEARCH_API`
    rather than `Evidence.FETCHED`. A boolean could not carry the difference, and
    a run during an outage would then be indistinguishable from a healthy one.
    """

    def _structural(self) -> WebResult:
        return a_result().model_copy(update={"url_is_structural": True})

    async def test_a_structural_url_is_accepted_when_there_is_no_fetcher(
        self,
    ) -> None:
        kept = await verify_result(self._structural(), fetcher=None, judge=None)
        assert kept is not None
        assert kept.verified is True

    async def test_and_it_is_recorded_as_the_weaker_evidence(self) -> None:
        """The whole point of the third state."""
        kept = await verify_result(self._structural(), fetcher=None, judge=None)
        assert kept is not None
        assert kept.evidence is Evidence.SEARCH_API

    async def test_a_model_asserted_url_is_not_accepted_without_a_fetch(
        self,
    ) -> None:
        """The line that must not move.

        Perplexity's URL is written by a model. With no fetcher there is nothing
        at all standing behind it, so it is dropped -- an outage may degrade the
        guarantee, never remove it.
        """
        assert await verify_result(a_result(), fetcher=None, judge=None) is None

    async def test_a_working_fetcher_still_gives_the_full_guarantee(self) -> None:
        """A structural URL is not a shortcut when the fetcher works."""
        judge = FakeJudge(
            ClaimVerdict(
                supported=True,
                quote="A submarine sinks by letting water into its ballast tanks.",
            )
        )
        kept = await verify_result(
            self._structural(),
            fetcher=FakeFetcher({"https://example.org/submarines": PAGE}),
            judge=judge,
        )
        assert kept is not None
        assert kept.evidence is Evidence.FETCHED

    async def test_a_structural_url_that_fails_a_real_check_is_still_dropped(
        self,
    ) -> None:
        """Structural provenance is about the URL, not about the content. If the
        page was read and did not support the claim, that is a real answer."""
        judge = FakeJudge(ClaimVerdict(supported=False, quote=""))
        assert (
            await verify_result(
                self._structural(),
                fetcher=FakeFetcher({"https://example.org/submarines": PAGE}),
                judge=judge,
            )
            is None
        )
