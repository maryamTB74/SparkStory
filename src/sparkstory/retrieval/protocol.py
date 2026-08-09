"""The interface a chunk store satisfies, declared rather than implied.

Until now there was exactly one store, and ``store.py``'s docstring recorded the
intent that a future Postgres implementation would be its peer -- *"the interface
is ``search`` and ``get``, neither of which mentions numpy"*. That intent was
real but never written down as a type: ``HybridIndex.__init__`` and
``drop_unprovenanced`` both annotate the concrete ``LocalVectorStore``, so nothing
could be substituted for it without editing every caller.

This module closes that gap, and it does so at the moment the project's own rule
allows: **a Protocol is added when a second implementation exists**, not in
advance. Before ``PgVectorStore`` it would have been an abstraction over one
thing.

``Protocol`` rather than an ABC, deliberately. Structural typing means an
implementation conforms by having the right methods, so neither store has to
inherit from anything and neither has to know this module exists. That matters
here because the two implementations have nothing in common underneath -- one is
numpy over three files, the other is SQL -- and an ABC would invite sharing code
between them that has no business being shared.
"""

from typing import Protocol, runtime_checkable

from sparkstory.retrieval.chunks import Chunk, SourceKind
from sparkstory.retrieval.types import SearchHit


@runtime_checkable
class ChunkStore(Protocol):
    """Somewhere chunks live and can be searched.

    Five members, which is exactly what ``LocalVectorStore`` already exposes. No
    member is added here that no caller uses: Rule 3 applies to interfaces as much
    as to config, and a Protocol is a promise every implementation has to keep.

    ``runtime_checkable`` so a test can assert conformance with ``isinstance``.
    Note this only checks that the *methods exist*, never their signatures -- so
    it catches a missing method and not a wrong one. The real check is that the
    same test suite runs against both implementations.
    """

    def save(self, chunks: list[Chunk]) -> None:
        """Replace the stored corpus with ``chunks``.

        Whole-corpus replacement rather than an upsert, because ingestion is an
        offline batch step (lesson 9's offline phase) and a partial write is the
        one outcome neither implementation should be able to produce.
        """
        ...

    @property
    def is_built(self) -> bool:
        """Whether anything has been stored yet.

        A property, not a method, because ``LocalVectorStore`` defines it as one
        and callers write ``if store.is_built``. Worth stating because
        ``runtime_checkable`` compares *names* only -- it would happily accept an
        implementation where this was a method, and the mismatch would surface as
        a truthy bound method that is always ``True``.

        Distinct from "is empty" on purpose: a store that was never built and a
        store built from an empty corpus are different problems, and only the
        first is a setup mistake.
        """
        ...

    @property
    def chunks(self) -> list[Chunk]:
        """Every stored chunk.

        **The awkward member, and it is here under protest.** Returning the entire
        corpus is natural for an in-memory store and is a full table scan for a
        database one. It survives because two real callers need it --
        ``provenance.py`` and the corpus quality tests -- and because 58 rows cost
        nothing.

        It is the first thing to remove if the corpus grows: a Protocol member
        that fetches every row is a scaling trap wearing an interface. Replacing
        it means giving ``provenance.py`` a bulk ``get_many(ids)`` instead, which
        is what it actually wants.
        """
        ...

    def get(self, chunk_id: str) -> Chunk | None:
        """Look one chunk up by id, or ``None`` if it was never stored.

        The trust gate. ``drop_unprovenanced`` resolves every ``chunk_id`` a model
        wrote against this method and overwrites ``source`` from the result, which
        is what makes a fabricated citation *unreachable* rather than merely
        detectable. Any implementation returning a chunk for an id it did not
        store breaks the provenance guarantee outright.
        """
        ...

    def search(
        self,
        query: str,
        source_kind: SourceKind | None = None,
        top_k: int = 5,
    ) -> list[SearchHit]:
        """Rank stored chunks against ``query``, best first.

        ``source_kind`` filters **before** scoring, not after -- lesson 9's
        metadata filtering. Filtering afterwards spends the top-k budget on chunks
        the caller cannot use, and can return fewer than ``top_k`` results while
        relevant ones exist.

        ``SearchHit.similarity`` is deliberately *not* specified to be a cosine.
        The local store returns one; a fused ranking returns an RRF score, which
        is not comparable across queries. Callers order by it and must not
        threshold on it.
        """
        ...
