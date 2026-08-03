"""The run-scoped ledger of web sources.

Pure data: no network, no model, no key. Built before anything that reaches the
web precisely so the rest of the feature has something offline to be tested
against.

The ledger is the web half of what ``LocalVectorStore`` is for the corpus -- the
thing an id resolves against, and the authority on attribution. Its ids are
run-scoped rather than global because a web source has no stable identity across
runs: the same query tomorrow returns different pages, so ``web:1`` means "the
first source *this run* accepted" and nothing more.
"""

from sparkstory.retrieval.web.ledger import WebLedger, WebSource


def a_source(**overrides: object) -> dict:
    payload: dict = {
        "url": "https://example.org/submarines",
        "title": "How submarines work",
        "text": "A submarine sinks by letting water into its ballast tanks.",
        "query": "how does a submarine sink",
        "verified": True,
    }
    payload.update(overrides)
    return payload


class TestWebSource:
    def test_carries_everything_needed_to_attribute_a_claim(self) -> None:
        source = WebSource(**a_source())
        assert source.url == "https://example.org/submarines"
        assert source.title == "How submarines work"
        assert source.query == "how does a submarine sink"
        assert source.verified is True

    def test_verified_defaults_to_false(self) -> None:
        """Unverified is the safe default.

        A source is only verified once a page has been fetched and found to
        support the claim. Defaulting to True would mean a construction site that
        forgot the flag silently produces provenance it never earned.
        """
        payload = a_source()
        del payload["verified"]
        assert WebSource(**payload).verified is False


class TestWebLedger:
    def test_mints_ids_in_order(self) -> None:
        ledger = WebLedger()
        assert ledger.add(WebSource(**a_source())) == "web:1"
        assert ledger.add(WebSource(**a_source(url="https://example.org/2"))) == "web:2"

    def test_a_minted_id_round_trips(self) -> None:
        ledger = WebLedger()
        source_id = ledger.add(WebSource(**a_source()))
        fetched = ledger.get(source_id)
        assert fetched is not None
        assert fetched.url == "https://example.org/submarines"

    def test_an_unknown_id_returns_none(self) -> None:
        assert WebLedger().get("web:99") is None

    def test_a_corpus_id_returns_none(self) -> None:
        """The ledger must not answer for the store.

        `drop_unprovenanced` will consult both, and an id belonging to one that
        resolved against the other would attribute a corpus fact to a web page or
        the reverse.
        """
        ledger = WebLedger()
        ledger.add(WebSource(**a_source()))
        assert ledger.get("moon#1") is None

    def test_ids_are_run_scoped(self) -> None:
        """Two ledgers share no state, and both start at web:1.

        A web source has no identity across runs -- the same query tomorrow
        returns different pages -- so `web:1` means "the first source this run
        accepted". A module-level counter would make run 2's `web:1` resolve
        against run 1's page, which is the worst kind of wrong: plausible.
        """
        first, second = WebLedger(), WebLedger()
        assert first.add(WebSource(**a_source())) == "web:1"
        assert (
            second.add(WebSource(**a_source(url="https://elsewhere.test"))) == "web:1"
        )
        kept = second.get("web:1")
        assert kept is not None
        assert kept.url == "https://elsewhere.test"

    def test_sources_are_listable_for_the_run_artifact(self) -> None:
        """The run writes the ledger to disk so a reader can see what was
        fetched and what was rejected. Session 9's finding M: a run whose
        artifacts do not record what happened is not evidence."""
        ledger = WebLedger()
        ledger.add(WebSource(**a_source()))
        ledger.add(WebSource(**a_source(url="https://example.org/2", verified=False)))
        assert [s.url for s in ledger.sources] == [
            "https://example.org/submarines",
            "https://example.org/2",
        ]

    def test_an_empty_ledger_is_falsy_and_lists_nothing(self) -> None:
        """The usual case: most premises never reach the web at all."""
        ledger = WebLedger()
        assert ledger.sources == []
        assert not ledger
