"""Types shared by every retrieval implementation and by their callers.

``SearchHit`` lived in ``store.py`` while there was exactly one store. It moved
here when a second implementation arrived, because the alternative was worse in
both directions: the Protocol would import from the concrete module it abstracts,
and ``tools.py`` -- which only ever wanted the return type -- would depend on a
particular store to name it.

Nothing here imports a store, an embedder or numpy, so this module is a leaf.
"""

from dataclasses import dataclass

from sparkstory.retrieval.chunks import Chunk


@dataclass(frozen=True)
class SearchHit:
    """One retrieved chunk and how similar it was.

    A dataclass rather than a Pydantic model: it never crosses a process boundary
    and is never validated against model output, so validation would be overhead
    with no reader.

    ``similarity`` means whatever the implementation that produced it means. A
    plain vector search returns a cosine; a fused ranking returns an RRF score,
    which exists only to preserve ordering and is not comparable across queries.
    Callers order by it; nothing may threshold on it.
    """

    chunk: Chunk
    similarity: float
