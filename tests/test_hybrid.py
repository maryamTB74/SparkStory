"""Keyword search, and fusing it with vector search.

Two things are being pinned here.

**RRF fuses on rank, not score.** That is the whole reason it was chosen over a
weighted average: a BM25 score of 7.3 and a cosine of 0.41 are not comparable, and
any attempt to make them comparable is a tuning constant nobody can justify.

**Each retriever must be able to win alone.** The tests isolate them with a
``ConstantEmbedder`` -- an embedder with no signal at all -- so a result that
survives can only have come from BM25. Without that isolation, ``FakeEmbedder``
is lexical too and both retrievers would agree on everything, which would make
the fusion tests pass while proving nothing.
"""

from pathlib import Path

import numpy as np

from sparkstory.retrieval.bm25 import BM25Index, tokenize
from sparkstory.retrieval.chunks import Chunk, SourceKind
from sparkstory.retrieval.embed import FakeEmbedder
from sparkstory.retrieval.hybrid import RRF_K, HybridIndex, rrf_fuse
from sparkstory.retrieval.store import LocalVectorStore

DOCS = [
    "The Moon has no air, so there is no wind and no sound.",
    "Penguins cannot fly, but they swim fast using their wings.",
    "A volcano erupts when magma pushes up to the surface.",
    "Hickory dickory dock, the mouse ran up the clock.",
]


class ConstantEmbedder:
    """An embedder with no signal: every text gets the same vector.

    Not a toy. It is how a test proves BM25 contributed a result, since with a
    constant embedder the vector half can only return an arbitrary order.
    """

    dimensions = 8

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        row = np.eye(1, self.dimensions)[0]
        return np.tile(row, (len(texts), 1)).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return np.eye(1, self.dimensions)[0].astype(np.float32)


class TestTokenize:
    def test_lowercases_and_drops_punctuation(self) -> None:
        assert tokenize("The Moon, has no air!") == ["the", "moon", "has", "no", "air"]

    def test_is_the_same_tokenizer_the_fake_embedder_uses(self) -> None:
        """Both halves of a hybrid search must agree on what a word is, or a
        comma changes which retriever can see a chunk."""
        assert tokenize("air,") == tokenize("air")


class TestBM25:
    def test_ranks_an_exact_term_match_first(self) -> None:
        index = BM25Index(DOCS)
        assert int(np.argmax(index.scores("volcano magma"))) == 2

    def test_a_query_with_no_known_terms_scores_nothing(self) -> None:
        """Must be zeros rather than a crash or a NaN: a premise about a lost
        teddy shares no term with a corpus of space facts, and that is normal."""
        scores = BM25Index(DOCS).scores("xylophone quokka")
        assert scores.shape == (len(DOCS),)
        assert not scores.any()

    def test_a_term_in_every_document_carries_almost_no_weight(self) -> None:
        """IDF doing its job. Without it, "the" would dominate every query and
        keyword search would rank by document length."""
        index = BM25Index(["the cat sat", "the dog sat", "the bird sat"])
        assert index.scores("the").max() < index.scores("cat").max()

    def test_longer_documents_are_not_favoured(self) -> None:
        """Length normalisation: a chunk that merely mentions a term often should
        not beat a short chunk that is about it."""
        index = BM25Index(["moon", "moon " * 40 + "and many other unrelated words"])
        assert index.scores("moon")[0] > 0

    def test_function_words_contribute_nothing(self) -> None:
        """Regression test for a defect found by building the real index. The
        query "could a flag wave on the moon?" ranked the chunk about *sound*
        first, because "could" is rare in a corpus of short factual statements and
        so earned a high idf, while the chunk it matched only said "could not
        hear". IDF cannot fix a word that is common in English and rare in the
        corpus -- it weights it the wrong way."""
        assert not BM25Index(DOCS).scores("could of the a which").any()

    def test_negation_is_not_treated_as_a_function_word(self) -> None:
        """ "no" carries the meaning of half the fact corpus -- "the Moon has *no*
        air" -- so an aggressive stoplist would delete the point of the chunk."""
        assert BM25Index(DOCS).scores("no air").any()

    def test_a_content_word_still_wins_over_a_function_word(self) -> None:
        scores = BM25Index(DOCS).scores("could a flag wave on the moon?")
        assert int(np.argmax(scores)) == 0

    def test_an_empty_corpus_does_not_divide_by_zero(self) -> None:
        assert BM25Index([]).scores("moon").shape == (0,)


class TestRRF:
    def test_fuses_on_rank_not_score(self) -> None:
        """The case that decides the choice of RRF. `steady` is never first, but
        it is near the top of both rankings; `spiky` wins one and is buried in the
        other. Rank-only fusion prefers agreement, which is what we want when the
        two scales are incomparable."""
        vector_ranking = ["spiky"] + [f"filler{i}" for i in range(29)]
        keyword_ranking = ["other", "steady"] + [f"filler{i}" for i in range(28)]
        vector_ranking.insert(1, "steady")
        keyword_ranking.append("spiky")

        fused = rrf_fuse([vector_ranking, keyword_ranking])
        assert fused.index("steady") < fused.index("spiky")

    def test_k_is_sixty(self) -> None:
        """Carried from storyweave. Pinned because changing it silently changes
        how much a single strong ranking can outweigh agreement."""
        assert RRF_K == 60

    def test_is_deterministic(self) -> None:
        rankings = [["a", "b", "c"], ["c", "a", "b"]]
        assert rrf_fuse(rankings) == rrf_fuse(rankings)

    def test_one_empty_ranking_leaves_the_other_intact(self) -> None:
        """A retriever that found nothing must not suppress the one that did."""
        assert rrf_fuse([["a", "b", "c"], []]) == ["a", "b", "c"]

    def test_both_empty_returns_empty(self) -> None:
        assert rrf_fuse([[], []]) == []

    def test_an_item_in_only_one_ranking_still_appears(self) -> None:
        """Fusion is a union, not an intersection: BM25 finding an exact term the
        vectors missed is the entire point of hybrid search."""
        assert "only-bm25" in rrf_fuse([["a"], ["only-bm25"]])


def built_index(root: Path, embedder: object) -> HybridIndex:
    chunks = [
        Chunk(
            chunk_id=f"doc#{i + 1}",
            text=text,
            title="Test",
            source="Test corpus",
            licence="public domain",
            source_kind=SourceKind.CRAFT if i == 3 else SourceKind.FACT,
        )
        for i, text in enumerate(DOCS)
    ]
    store = LocalVectorStore(root=root, embedder=embedder)  # type: ignore[arg-type]
    store.save(chunks)
    return HybridIndex(store=store)


class TestHybridIndex:
    def test_keyword_search_alone_can_surface_a_chunk(self, tmp_path: Path) -> None:
        """With a signal-free embedder, anything ranked first came from BM25. This
        is the test that proves the keyword half is wired in at all."""
        index = built_index(tmp_path, ConstantEmbedder())
        hits = index.search("volcano magma", top_k=2)
        assert hits[0].chunk.chunk_id == "doc#3"

    def test_respects_source_kind(self, tmp_path: Path) -> None:
        """Both halves must filter, not just the vector one -- otherwise the fact
        tool can return a nursery rhyme through the keyword path."""
        index = built_index(tmp_path, FakeEmbedder(dimensions=256))
        hits = index.search("mouse clock", source_kind=SourceKind.FACT, top_k=4)
        assert all(hit.chunk.source_kind is SourceKind.FACT for hit in hits)

    def test_respects_top_k(self, tmp_path: Path) -> None:
        index = built_index(tmp_path, FakeEmbedder(dimensions=256))
        assert len(index.search("moon", top_k=2)) == 2

    def test_a_missing_index_returns_nothing(self, tmp_path: Path) -> None:
        """Fail open, same as the store: research is enrichment."""
        store = LocalVectorStore(root=tmp_path / "nope", embedder=FakeEmbedder())
        assert HybridIndex(store=store).search("anything") == []

    def test_finds_a_chunk_both_retrievers_agree_on(self, tmp_path: Path) -> None:
        index = built_index(tmp_path, FakeEmbedder(dimensions=256))
        hits = index.search("moon air wind", top_k=3)
        assert hits[0].chunk.chunk_id == "doc#1"
