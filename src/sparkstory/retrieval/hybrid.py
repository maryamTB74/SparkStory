"""Vector search and keyword search, fused on rank.

**Fusion is Reciprocal Rank Fusion at k=60**, carried over from storyweave, and
the reason is scale rather than sophistication: a BM25 score of 7.3 and a cosine of
0.41 cannot be averaged without inventing a weight nobody can justify. RRF throws
the scores away and keeps only the ranks, so the two retrievers vote instead of
being reconciled. Agreement between them beats a single strong opinion, which is
the behaviour you want when neither retriever is trustworthy alone.
"""

import numpy as np

from sparkstory.retrieval.bm25 import BM25Index
from sparkstory.retrieval.chunks import SourceKind
from sparkstory.retrieval.store import LocalVectorStore, SearchHit
from sparkstory.utils.logging_utils import get_logger

#: The RRF constant. Larger values flatten the difference between ranks, so a
#: first place counts for less relative to broad agreement. 60 is the value from
#: the original paper and from storyweave; pinned and tested, because changing it
#: silently changes how much one retriever can outweigh the other.
logger = get_logger(__name__)

RRF_K = 60

#: How many candidates each retriever contributes before fusion. Fusion needs
#: material: if both halves returned exactly `top_k`, a chunk ranked just outside
#: one retriever's cut could never be rescued by the other, which is the case
#: hybrid search exists to handle.
_CANDIDATE_MULTIPLIER = 4
_MINIMUM_CANDIDATES = 10

#: Below this many distinct known query terms, keyword search abstains entirely.
#: One term that appears everywhere ranks by length, which is noise, and RRF cannot
#: distinguish a noisy vote from an informed one.
_MINIMUM_MATCHED_TERMS = 2


def rrf_fuse(rankings: list[list[str]]) -> list[str]:
    """Fuse ranked id lists into one, by reciprocal rank.

    Each item scores ``1 / (RRF_K + rank)`` in every ranking it appears in, with
    rank 1-based, and the scores are summed. An item missing from a ranking simply
    earns nothing there -- so this is a union, not an intersection, which is what
    lets BM25 rescue an exact-term match the vectors never surfaced.

    Ties break on first appearance, making the result stable for identical input.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for position, identifier in enumerate(ranking, start=1):
            scores[identifier] = scores.get(identifier, 0.0) + 1.0 / (RRF_K + position)
    return sorted(scores, key=lambda identifier: -scores[identifier])


class HybridIndex:
    """A store, a keyword index over the same chunks, and RRF over both.

    Wraps ``LocalVectorStore`` rather than extending it, so the vector store stays
    usable on its own -- which is what the retrieval eval set needs in order to
    show whether fusion actually helps.

    The BM25 index is built lazily from the store's chunks and cached, because
    tokenising the corpus per query would cost more than the search.
    """

    def __init__(self, store: LocalVectorStore) -> None:
        self.store = store
        self._bm25: BM25Index | None = None
        self._bm25_ids: list[str] = []

    def _keyword_index(self) -> tuple[BM25Index, list[str]]:
        if self._bm25 is None:
            chunks = self.store.chunks
            # Indexes `embed_text`, not `text`, so both retrievers see the same
            # words. Otherwise a query naming the source title ("moon") could be
            # found by vectors and missed by keywords.
            self._bm25 = BM25Index([chunk.embed_text for chunk in chunks])
            self._bm25_ids = [chunk.chunk_id for chunk in chunks]
        return self._bm25, self._bm25_ids

    def search(
        self,
        query: str,
        source_kind: SourceKind | None = None,
        top_k: int = 5,
    ) -> list[SearchHit]:
        """Search both halves and return the fused top ``top_k``.

        ``SearchHit.similarity`` on a fused result is the **RRF score**, not a
        cosine. It is deliberately not comparable across queries and exists only
        to preserve the ordering; nothing shows it to a parent, and the retrieval
        tools present rank rather than score for exactly this reason.
        """
        chunks = self.store.chunks
        if not chunks:
            return []

        by_id = {
            chunk.chunk_id: chunk
            for chunk in chunks
            if source_kind is None or chunk.source_kind is source_kind
        }
        if not by_id:
            return []

        candidates = max(top_k * _CANDIDATE_MULTIPLIER, _MINIMUM_CANDIDATES)

        vector_ranking = [
            hit.chunk.chunk_id
            for hit in self.store.search(
                query, source_kind=source_kind, top_k=candidates
            )
        ]

        bm25, ids = self._keyword_index()
        # A keyword ranking built from a single generic term is not a second
        # opinion -- BM25 falls back to ranking by document length, and RRF has no
        # way to tell that vote from a real one. Measured, not assumed: fusing it
        # cost one hit on the labelled set. See BM25Index.matched_terms.
        if bm25.matched_terms(query) < _MINIMUM_MATCHED_TERMS:
            logger.debug("keyword search abstains on %r: too few known terms", query)
            scores = np.zeros(len(ids), dtype=np.float32)
        else:
            scores = bm25.scores(query)
        # Filtering here rather than building a per-kind BM25 index: the corpus is
        # small, and two indexes would have to be kept in step with the store's
        # chunk order. A zero score is dropped so a chunk nothing matched cannot
        # enter the ranking purely by position.
        keyword_ranking = [
            ids[int(index)]
            for index in np.argsort(scores)[::-1]
            if ids[int(index)] in by_id and scores[int(index)] > 0
        ][:candidates]

        fused = rrf_fuse([vector_ranking, keyword_ranking])
        ranked = [identifier for identifier in fused if identifier in by_id][:top_k]

        position = {identifier: rank for rank, identifier in enumerate(fused, start=1)}
        return [
            SearchHit(
                chunk=by_id[identifier],
                similarity=1.0 / (RRF_K + position[identifier]),
            )
            for identifier in ranked
        ]
