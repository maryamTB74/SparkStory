"""The local vector store.

The distinction this module exists to pin down: **absence is not corruption.** An
index that was never built returns nothing, because research must never be able to
kill a book (spec section 11). An index whose files disagree with each other raises,
because that is a bug and answering anyway would mean answering wrongly.
"""

from pathlib import Path

import numpy as np
import pytest

from sparkstory.retrieval.chunks import Chunk, SourceKind
from sparkstory.retrieval.embed import FakeEmbedder
from sparkstory.retrieval.store import LocalVectorStore

FACTS = [
    ("moon#1", "The Moon has no air, so there is no wind and no sound."),
    ("moon#2", "The Moon's gravity is weaker than Earth's, so you would bounce."),
    ("penguin#1", "Penguins cannot fly, but they swim fast using their wings."),
]
CRAFT = [
    ("goose#1", "Hickory dickory dock, the mouse ran up the clock."),
]


def corpus() -> list[Chunk]:
    chunks = [
        Chunk(
            chunk_id=cid,
            text=text,
            title=cid.split("#")[0].title(),
            source="Test corpus",
            licence="public domain",
            source_kind=SourceKind.FACT,
        )
        for cid, text in FACTS
    ]
    chunks += [
        Chunk(
            chunk_id=cid,
            text=text,
            title="Mother Goose",
            source="Test corpus",
            licence="public domain",
            source_kind=SourceKind.CRAFT,
        )
        for cid, text in CRAFT
    ]
    return chunks


def built_store(root: Path) -> LocalVectorStore:
    store = LocalVectorStore(root=root, embedder=FakeEmbedder(dimensions=256))
    store.save(corpus())
    return store


class TestRoundTrip:
    def test_saves_and_reloads_every_chunk(self, tmp_path: Path) -> None:
        built_store(tmp_path)
        reopened = LocalVectorStore(
            root=tmp_path, embedder=FakeEmbedder(dimensions=256)
        )
        assert {chunk.chunk_id for chunk in reopened.chunks} == {
            "moon#1",
            "moon#2",
            "penguin#1",
            "goose#1",
        }

    def test_writes_exactly_three_files(self, tmp_path: Path) -> None:
        built_store(tmp_path)
        assert sorted(path.name for path in tmp_path.iterdir()) == [
            "chunks.json",
            "meta.json",
            "vectors.npy",
        ]

    def test_records_which_embedder_built_it(self, tmp_path: Path) -> None:
        """So an index built by one model and searched by another is a named
        error rather than a set of plausible wrong answers."""
        built_store(tmp_path)
        import json

        meta = json.loads((tmp_path / "meta.json").read_text())
        assert meta["dimensions"] == 256
        assert meta["chunk_count"] == 4


class TestSearch:
    def test_ranks_the_relevant_chunk_first(self, tmp_path: Path) -> None:
        hits = built_store(tmp_path).search("moon air wind", top_k=3)
        assert hits[0].chunk.chunk_id == "moon#1"

    def test_respects_top_k(self, tmp_path: Path) -> None:
        assert len(built_store(tmp_path).search("moon", top_k=2)) == 2

    def test_similarity_scores_descend(self, tmp_path: Path) -> None:
        hits = built_store(tmp_path).search("moon air", top_k=4)
        scores = [hit.similarity for hit in hits]
        assert scores == sorted(scores, reverse=True)

    def test_filters_by_source_kind(self, tmp_path: Path) -> None:
        """Each tool pins one kind, so a leak here would let the fact tool return
        nursery rhymes -- which the agent would dutifully cite as a fact."""
        hits = built_store(tmp_path).search("mouse clock", source_kind=SourceKind.CRAFT)
        assert {hit.chunk.chunk_id for hit in hits} == {"goose#1"}

    def test_a_kind_with_no_chunks_returns_nothing(self, tmp_path: Path) -> None:
        store = LocalVectorStore(root=tmp_path, embedder=FakeEmbedder(dimensions=256))
        facts_only = [c for c in corpus() if c.source_kind is SourceKind.FACT]
        store.save(facts_only)
        assert store.search("anything", source_kind=SourceKind.CRAFT) == []


class TestAbsenceIsNotAnError:
    def test_a_missing_index_returns_no_results(self, tmp_path: Path) -> None:
        """Fail open. Research is enrichment, so an index nobody built must not
        raise -- the book simply gets planned ungrounded."""
        store = LocalVectorStore(
            root=tmp_path / "never-built", embedder=FakeEmbedder(dimensions=256)
        )
        assert store.search("the moon") == []
        assert store.chunks == []

    def test_a_missing_index_reports_itself_as_empty(self, tmp_path: Path) -> None:
        store = LocalVectorStore(root=tmp_path / "nope", embedder=FakeEmbedder())
        assert not store.is_built


class TestCorruptionIsAnError:
    def test_mismatched_lengths_raise(self, tmp_path: Path) -> None:
        """Half an index is a bug. Answering from it would mean pairing a chunk
        with another chunk's vector, which returns confident nonsense."""
        built_store(tmp_path)
        np.save(tmp_path / "vectors.npy", np.zeros((2, 256), dtype=np.float32))
        store = LocalVectorStore(root=tmp_path, embedder=FakeEmbedder(dimensions=256))
        with pytest.raises(ValueError, match="disagree"):
            store.search("moon")

    def test_a_dimension_mismatch_names_the_fix(self, tmp_path: Path) -> None:
        """Swapping EMBEDDING_MODEL without re-ingesting is an easy mistake and
        produces meaningless similarities rather than an obvious failure."""
        built_store(tmp_path)
        store = LocalVectorStore(root=tmp_path, embedder=FakeEmbedder(dimensions=64))
        with pytest.raises(ValueError, match="ingest_knowledge"):
            store.search("moon")


class TestLookupByChunkId:
    def test_finds_a_stored_chunk(self, tmp_path: Path) -> None:
        """This is what provenance filtering is built on: a fact naming a chunk
        we never stored is a fact we cannot stand behind."""
        assert built_store(tmp_path).get("moon#2") is not None

    def test_returns_none_for_an_unknown_id(self, tmp_path: Path) -> None:
        assert built_store(tmp_path).get("invented#99") is None
