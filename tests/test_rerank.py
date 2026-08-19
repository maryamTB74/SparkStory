"""The reranker seam and its LLM implementation.

Nothing here reaches a provider: the model is faked, because what is under test is
the mapping from an answer to a reordered list rather than any model's judgement.
The judgement is measured live in ``test_retrieval_eval.py``, where it can be
compared against fusion on the labelled set.

The three tests that matter are the ones about *invented* and *omitted* ids. Those
are failures a reranker can cause that plain fusion cannot, because fusion only
ever reorders things it retrieved, while a model writing ids can write one that
does not exist.
"""

import pytest

from sparkstory.retrieval.chunks import Chunk, SourceKind
from sparkstory.retrieval.rerank import identity_reranker
from sparkstory.retrieval.rerankers.llm import RankedIds, build_llm_reranker
from sparkstory.retrieval.types import SearchHit


def _hit(chunk_id: str, text: str = "") -> SearchHit:
    return SearchHit(
        chunk=Chunk(
            chunk_id=chunk_id,
            text=text or f"text for {chunk_id}",
            title="Test",
            source="test",
            licence="test",
            source_kind=SourceKind.FACT,
        ),
        similarity=0.5,
    )


class _FakeModel:
    """Answers with whatever ids it was constructed with."""

    def __init__(self, ids: list[str]) -> None:
        self._ids = ids

    async def ainvoke(self, _messages: object) -> RankedIds:
        return RankedIds(chunk_ids=self._ids)


class TestIdentityReranker:
    """The control. A comparison needs a baseline that runs the same plumbing."""

    async def test_it_preserves_fusion_order(self) -> None:
        hits = [_hit("a#1"), _hit("b#1"), _hit("c#1")]
        result = await identity_reranker("q", hits, 3)
        assert [h.chunk.chunk_id for h in result] == ["a#1", "b#1", "c#1"]

    async def test_it_truncates_to_top_k(self) -> None:
        """Truncation is part of the seam's contract, not the caller's job: two
        rerankers handed the same candidates must return the same number, or a
        comparison measures how many they returned rather than how well they
        ranked.
        """
        hits = [_hit("a#1"), _hit("b#1"), _hit("c#1")]
        assert len(await identity_reranker("q", hits, 2)) == 2

    async def test_no_candidates_is_not_an_error(self) -> None:
        assert await identity_reranker("q", [], 3) == []


class TestTheModelReordersButCannotInvent:
    async def test_it_applies_the_models_order(self) -> None:
        hits = [_hit("a#1"), _hit("b#1"), _hit("c#1")]
        rerank = build_llm_reranker(_FakeModel(["c#1", "a#1", "b#1"]))

        result = await rerank("q", hits, 3)

        assert [h.chunk.chunk_id for h in result] == ["c#1", "a#1", "b#1"]

    async def test_an_id_that_was_never_a_candidate_is_dropped(self) -> None:
        """The one failure a reranker can cause that fusion cannot.

        The model writes ids, so it can write one that does not exist, and a
        grounded fact built from an invented id would cite a chunk nobody can look
        up -- undetectable afterwards, which is exactly the property that made the
        store overwrite `source` rather than trust it.
        """
        hits = [_hit("a#1"), _hit("b#1")]
        rerank = build_llm_reranker(_FakeModel(["moon#99", "b#1", "a#1"]))

        result = await rerank("q", hits, 3)

        assert [h.chunk.chunk_id for h in result] == ["b#1", "a#1"]

    async def test_an_omitted_candidate_is_demoted_rather_than_deleted(self) -> None:
        """A candidate the model leaves out falls behind what it chose and stays.

        Deleting would let one model call shrink the result below top_k, which
        turns a ranking into a filter the caller did not ask for -- and the caller
        here is retrieval, which has its own reasons for the number it requested.
        """
        hits = [_hit("a#1"), _hit("b#1"), _hit("c#1")]
        rerank = build_llm_reranker(_FakeModel(["c#1"]))

        result = await rerank("q", hits, 3)

        assert [h.chunk.chunk_id for h in result] == ["c#1", "a#1", "b#1"]

    async def test_an_empty_answer_falls_back_to_fusion_order(self) -> None:
        """A model that chooses nothing must not empty the results. Fusion's
        ranking is a worse answer than a good rerank and a much better one than
        no answer, so it is what remains.
        """
        hits = [_hit("a#1"), _hit("b#1")]
        rerank = build_llm_reranker(_FakeModel([]))

        result = await rerank("q", hits, 2)

        assert [h.chunk.chunk_id for h in result] == ["a#1", "b#1"]

    async def test_it_never_calls_the_model_with_no_candidates(self) -> None:
        """Nothing to rank is not a question worth paying for."""

        class _Exploding:
            async def ainvoke(self, _messages: object) -> RankedIds:
                raise AssertionError("the model was called with no candidates")

        assert await build_llm_reranker(_Exploding())("q", [], 3) == []


@pytest.mark.parametrize("top_k", [1, 2, 3])
async def test_both_rerankers_return_the_same_count(top_k: int) -> None:
    """The property the comparison depends on. If two rerankers return different
    numbers of hits for the same request, their hit-rates are not comparable and
    the eval harness is measuring the wrong thing.
    """
    hits = [_hit("a#1"), _hit("b#1"), _hit("c#1")]
    llm = build_llm_reranker(_FakeModel(["c#1", "b#1", "a#1"]))

    assert len(await identity_reranker("q", hits, top_k)) == len(
        await llm("q", hits, top_k)
    )
