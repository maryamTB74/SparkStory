"""The web half of retrieval, measured.

**Marked ``web``, so it is excluded from ``make test`` and ``ci-local``.** These
particular tests inject their candidate rather than searching, so they need no key
and touch no network -- but the marker is on the file because the searching tests
that belong beside them do, and splitting one concern across two markers would
make "is the web half measured?" a question about which file you ran.

Why this file exists at all: ``search_web`` joins the Researcher's tool list
whenever ``MAX_WEB_SEARCHES > 0``, and until now nothing exercised it. That left
the half of retrieval covering every subject the corpus lacks -- which is most
subjects a parent might ask about -- completely unmeasured.

**The rejection is tested before the acceptance, deliberately.** The one live web
run this project has done accepted a page whose captured text was navigation
chrome: "Uploaded by / AI-enhanced title and description / Share this document /
Footer menu / About / Support". The URL was real; the page was furniture. Nothing
in that path could have rejected it, because rejecting it needs a page fetch that
was unavailable. A check that has never rejected anything is unfalsified rather
than proven, so the first thing measured here is a refusal.
"""

import pytest

from sparkstory.retrieval.web.ledger import Evidence
from sparkstory.retrieval.web.providers import WebResult
from sparkstory.retrieval.web.verify import verify_result

pytestmark = pytest.mark.web


def _candidate(*, url: str, structural: bool) -> WebResult:
    """A search result as it arrives: unverified, whatever its origin."""
    return WebResult(
        url=url,
        title="How many hearts does an octopus have?",
        text="An octopus has three hearts and blue blood.",
        query="how many hearts does an octopus have",
        url_is_structural=structural,
    )


class TestARejectionActuallyHappens:
    """The falsification. Without a case that is refused, the check is untested."""

    async def test_a_model_written_url_is_dropped_when_no_page_can_be_fetched(
        self,
    ) -> None:
        """With no fetcher available, a claim survives only if its URL came out of
        a search API's own response. A URL a model wrote cannot be checked at all
        -- not even for existing -- so it is dropped rather than trusted.

        This is the case the whole verification design exists for: the alternative
        is accepting an address because it looks like an address.
        """
        rejected = await verify_result(
            _candidate(url="https://example.invalid/not-a-real-page", structural=False),
            fetcher=None,
        )

        assert rejected is None, (
            "a model-written URL was accepted with no page fetched; nothing in "
            "this path could tell a real source from a fabricated one"
        )

    async def test_the_run_survives_the_rejection(self) -> None:
        """Dropping a claim must not raise. A story with one fewer fact is a
        poorer story; a story that failed to generate is a failure, and the two
        must not be confused by a check that kills the run when it refuses.
        """
        for structural in (True, False):
            outcome = await verify_result(
                _candidate(url="https://example.invalid/page", structural=structural),
                fetcher=None,
            )
            assert outcome is None or outcome.verified


class TestAcceptanceRecordsHowItWasEarned:
    async def test_a_structural_url_is_accepted_and_says_the_page_was_not_read(
        self,
    ) -> None:
        """The approving case, and it must record *how* it approved.

        A boolean would say ``verified: true`` here and lose the distinction that
        matters: the URL is real because a search API returned it, and nobody read
        the page. So this run is not the same as one where a judge quoted a
        supporting sentence, and a reader months later must be able to tell.
        """
        accepted = await verify_result(
            _candidate(
                url="https://oceanservice.noaa.gov/facts/octopus.html",
                structural=True,
            ),
            fetcher=None,
        )

        assert accepted is not None
        assert accepted.verified is True
        assert accepted.evidence is Evidence.SEARCH_API, (
            "a source accepted on its URL alone must not be recorded as though a "
            f"page was fetched and checked; got {accepted.evidence}"
        )
