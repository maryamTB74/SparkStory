"""Keeping only what the corpus actually supports.

Two jobs, and the second one was not in the original design.

**Drop what we cannot find.** A fact citing a chunk id we never stored is a fact we
cannot stand behind, so it goes -- and it goes *quietly*, as a dropped review does,
because research is enrichment and must never kill a book.

**Correct the attribution rather than trusting it.** ``source`` is not the model's
to state: the store knows it. Overwriting makes the task 1 spike's defect --
``source: "moon#1"`` where the design promises ``"NASA -- Earth's Moon"`` --
impossible rather than merely visible in a test.
"""

from pathlib import Path

from sparkstory.entities.grounding import CraftDevice, GroundedFact, StoryGrounding
from sparkstory.retrieval.chunks import Chunk, SourceKind
from sparkstory.retrieval.embed import FakeEmbedder
from sparkstory.retrieval.provenance import drop_unprovenanced
from sparkstory.retrieval.store import LocalVectorStore
from sparkstory.retrieval.web.ledger import WebLedger, WebSource


def built_store(root: Path) -> LocalVectorStore:
    store = LocalVectorStore(root=root, embedder=FakeEmbedder(dimensions=256))
    store.save(
        [
            Chunk(
                chunk_id="moon#1",
                text="The Moon has no air.",
                title="The Moon",
                source="NASA -- Earth's Moon",
                licence="public domain",
                source_kind=SourceKind.FACT,
            ),
            Chunk(
                chunk_id="goose#1",
                text="Refrain: one line comes back unchanged.",
                title="Nursery rhymes",
                source="Mother Goose (Project Gutenberg)",
                licence="public domain",
                source_kind=SourceKind.CRAFT,
            ),
        ]
    )
    return store


def fact(
    chunk_id: str = "moon#1", source: str = "NASA -- Earth's Moon"
) -> GroundedFact:
    return GroundedFact(
        claim="The Moon has no air.",
        story_note="Nothing outdoors can flutter or make a sound.",
        source=source,
        chunk_id=chunk_id,
    )


def device(chunk_id: str = "goose#1") -> CraftDevice:
    return CraftDevice(
        device="refrain",
        how_to_use="Repeat one line at each of the three attempts.",
        chunk_id=chunk_id,
    )


class TestKeepsWhatIsSupported:
    def test_a_fact_citing_a_real_chunk_survives(self, tmp_path: Path) -> None:
        kept = drop_unprovenanced(
            StoryGrounding(facts=[fact()], craft_devices=[]), built_store(tmp_path)
        )
        assert len(kept.facts) == 1
        assert kept.facts[0].claim == "The Moon has no air."

    def test_the_model_s_own_words_are_left_alone(self, tmp_path: Path) -> None:
        """`claim` and `story_note` are the agent's contribution and the
        whole value of the step -- only attribution is overwritten."""
        kept = drop_unprovenanced(
            StoryGrounding(facts=[fact()], craft_devices=[]), built_store(tmp_path)
        )
        assert kept.facts[0].story_note == (
            "Nothing outdoors can flutter or make a sound."
        )

    def test_a_craft_device_citing_a_real_chunk_survives(self, tmp_path: Path) -> None:
        kept = drop_unprovenanced(
            StoryGrounding(facts=[], craft_devices=[device()]), built_store(tmp_path)
        )
        assert len(kept.craft_devices) == 1


class TestCorrectsAttribution:
    def test_overwrites_an_invented_source(self, tmp_path: Path) -> None:
        """The spike's failure, made unreachable. The store is authoritative about
        where a chunk came from."""
        kept = drop_unprovenanced(
            StoryGrounding(facts=[fact(source="moon#1")], craft_devices=[]),
            built_store(tmp_path),
        )
        assert kept.facts[0].source == "NASA -- Earth's Moon"

    def test_overwrites_a_plausible_but_wrong_source(self, tmp_path: Path) -> None:
        """The dangerous case: a citation that looks right. A fabricated
        "Encyclopaedia Britannica" would survive every other check."""
        kept = drop_unprovenanced(
            StoryGrounding(
                facts=[fact(source="Encyclopaedia Britannica, 2019")], craft_devices=[]
            ),
            built_store(tmp_path),
        )
        assert kept.facts[0].source == "NASA -- Earth's Moon"


class TestDropsWhatIsNot:
    def test_an_invented_chunk_id_is_dropped(self, tmp_path: Path) -> None:
        kept = drop_unprovenanced(
            StoryGrounding(facts=[fact(chunk_id="atlantis#7")], craft_devices=[]),
            built_store(tmp_path),
        )
        assert kept.facts == []

    def test_dropping_everything_yields_empty_grounding_not_an_error(
        self, tmp_path: Path
    ) -> None:
        """Fail open. An agent that invented all three facts costs us the facts,
        not the book."""
        kept = drop_unprovenanced(
            StoryGrounding(
                facts=[fact(chunk_id=f"nope#{i}") for i in range(3)],
                craft_devices=[device(chunk_id="nope#9")],
            ),
            built_store(tmp_path),
        )
        assert kept.facts == []
        assert kept.craft_devices == []

    def test_a_fact_citing_a_craft_chunk_is_dropped(self, tmp_path: Path) -> None:
        """A category error, not a typo: a nursery rhyme cited as a fact about the
        world would ground the story in something that is not true of it."""
        kept = drop_unprovenanced(
            StoryGrounding(facts=[fact(chunk_id="goose#1")], craft_devices=[]),
            built_store(tmp_path),
        )
        assert kept.facts == []

    def test_a_craft_device_citing_a_fact_chunk_is_dropped(
        self, tmp_path: Path
    ) -> None:
        kept = drop_unprovenanced(
            StoryGrounding(facts=[], craft_devices=[device(chunk_id="moon#1")]),
            built_store(tmp_path),
        )
        assert kept.craft_devices == []

    def test_mixed_input_keeps_only_the_good_ones(self, tmp_path: Path) -> None:
        kept = drop_unprovenanced(
            StoryGrounding(
                facts=[fact(), fact(chunk_id="invented#1")], craft_devices=[device()]
            ),
            built_store(tmp_path),
        )
        assert [f.chunk_id for f in kept.facts] == ["moon#1"]
        assert len(kept.craft_devices) == 1


class TestNoIndexAtAll:
    def test_everything_is_dropped(self, tmp_path: Path) -> None:
        """If nothing can be verified, nothing survives. With no index the tools
        returned nothing, so any fact here was invented by definition."""
        store = LocalVectorStore(root=tmp_path / "nope", embedder=FakeEmbedder())
        kept = drop_unprovenanced(
            StoryGrounding(facts=[fact()], craft_devices=[device()]), store
        )
        assert kept.facts == []
        assert kept.craft_devices == []


class TestDoesNotMutateItsInput:
    def test_the_original_is_unchanged(self, tmp_path: Path) -> None:
        """The run artifact should record what the agent actually returned, so a
        later reader can see what was dropped and why."""
        original = StoryGrounding(facts=[fact(source="wrong")], craft_devices=[])
        drop_unprovenanced(original, built_store(tmp_path))
        assert original.facts[0].source == "wrong"


class TestWebProvenance:
    """A web source is kept or dropped by the same rule as a corpus chunk.

    That symmetry is the point. `drop_unprovenanced` already makes a fabricated
    corpus citation unreachable rather than merely detectable, by resolving the
    id and overwriting `source` from the store. A web fact gets the identical
    treatment against the ledger -- so "where did this come from?" has one answer
    with one strength, rather than a strong answer for the corpus and a weaker
    one for the web.
    """

    def _ledger(self, verified: bool = True) -> WebLedger:
        ledger = WebLedger()
        ledger.add(
            WebSource(
                url="https://example.org/submarines",
                title="How submarines work",
                text="A submarine sinks by letting water into its ballast tanks.",
                query="how does a submarine sink",
                verified=verified,
            )
        )
        return ledger

    def test_a_verified_web_fact_is_kept(self, tmp_path: Path) -> None:
        kept = drop_unprovenanced(
            StoryGrounding(facts=[fact(chunk_id="web:1")]),
            built_store(tmp_path),
            ledger=self._ledger(),
        )
        assert len(kept.facts) == 1

    def test_attribution_is_overwritten_from_the_ledger(self, tmp_path: Path) -> None:
        """Same rule as the store, for the same reason: `source` is not the
        model's to state. A plausible fabrication would survive any check that
        only looked for *a* source, so the record is authoritative."""
        kept = drop_unprovenanced(
            StoryGrounding(
                facts=[fact(chunk_id="web:1", source="Encyclopaedia Britannica, 2019")]
            ),
            built_store(tmp_path),
            ledger=self._ledger(),
        )
        assert kept.facts[0].source == "https://example.org/submarines"

    def test_an_unknown_web_id_is_dropped(self, tmp_path: Path) -> None:
        kept = drop_unprovenanced(
            StoryGrounding(facts=[fact(chunk_id="web:9")]),
            built_store(tmp_path),
            ledger=self._ledger(),
        )
        assert kept.facts == []

    def test_an_unverified_source_is_dropped(self, tmp_path: Path) -> None:
        """The load-bearing one.

        A source in the ledger but never checked -- because VERIFY_WEB_CLAIMS was
        off, or because the fetch was skipped -- must not ground a book. This is
        what makes the skip flag safe to exist: an unverified source is recorded
        honestly and then refused here.
        """
        kept = drop_unprovenanced(
            StoryGrounding(facts=[fact(chunk_id="web:1")]),
            built_store(tmp_path),
            ledger=self._ledger(verified=False),
        )
        assert kept.facts == []

    def test_a_web_id_is_dropped_when_there_is_no_ledger(self, tmp_path: Path) -> None:
        """The web tool was off, so nothing can vouch for a web id. An agent that
        invented one gets it dropped rather than trusted."""
        kept = drop_unprovenanced(
            StoryGrounding(facts=[fact(chunk_id="web:1")]), built_store(tmp_path)
        )
        assert kept.facts == []

    def test_a_craft_device_may_not_cite_the_web(self, tmp_path: Path) -> None:
        """Same category error the fact/craft split already rejects. Craft comes
        from the curated collection; a web page is not a read-aloud technique."""
        kept = drop_unprovenanced(
            StoryGrounding(craft_devices=[device(chunk_id="web:1")]),
            built_store(tmp_path),
            ledger=self._ledger(),
        )
        assert kept.craft_devices == []

    def test_corpus_facts_still_resolve_when_a_ledger_is_present(
        self, tmp_path: Path
    ) -> None:
        """The two must not interfere: a run with a ledger still grounds normally
        in the corpus."""
        kept = drop_unprovenanced(
            StoryGrounding(facts=[fact(chunk_id="moon#1")]),
            built_store(tmp_path),
            ledger=self._ledger(),
        )
        assert len(kept.facts) == 1
        assert kept.facts[0].source == "NASA -- Earth's Moon"
