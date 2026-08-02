"""A vector index that is three files on disk.

    chunks.json    the chunks, in index order
    vectors.npy    one row per chunk, unit length, in the same order
    meta.json      which embedder built it, and how wide the vectors are

Rule 2 is satisfied without an argument -- delete the whole thing and the app still
writes stories -- but so is the product requirement, because a served vector
database would buy nothing at this size. Phase B's migration to pgvector stays
open: the interface is ``search`` and ``get``, neither of which mentions numpy.

**Absence and corruption are treated differently, and that is the design.** An
index that was never built returns nothing, because research is enrichment and
must never be able to destroy a book. An index whose files disagree raises,
because pairing chunk *i* with vector *j* returns confident nonsense, and a
retrieval layer that answers wrongly is worse than one that answers not at all.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sparkstory.retrieval.chunks import Chunk, SourceKind
from sparkstory.retrieval.embed import Embedder
from sparkstory.utils.logging_utils import get_logger

logger = get_logger(__name__)

_CHUNKS_FILE = "chunks.json"
_VECTORS_FILE = "vectors.npy"
_META_FILE = "meta.json"


@dataclass(frozen=True)
class SearchHit:
    """One retrieved chunk and how similar it was.

    A dataclass rather than a Pydantic model: it never crosses a process boundary
    and is never validated against model output, so validation would be overhead
    with no reader.
    """

    chunk: Chunk
    similarity: float


class LocalVectorStore:
    """Chunks, their vectors, and cosine search over them.

    The root is injected rather than read from settings, so a test points it at
    ``tmp_path`` and callers pass ``settings.knowledge_root``. Same shape the canon
    store was going to use.

    Loading is lazy and cached: the MCP server must start without touching disk,
    and a per-request reload would re-parse the whole corpus on the cheapest path
    in the system.
    """

    def __init__(self, root: Path, embedder: Embedder) -> None:
        self.root = Path(root)
        self.embedder = embedder
        self._chunks: list[Chunk] | None = None
        self._vectors: np.ndarray | None = None

    # --- Writing ---------------------------------------------------------
    def save(self, chunks: list[Chunk]) -> None:
        """Embed every chunk and write the index, replacing whatever was there.

        Called by ingestion only, never on a request path. Embedding happens here
        rather than in the caller so that the vectors and the chunk order cannot
        be assembled separately and get out of step.
        """
        self.root.mkdir(parents=True, exist_ok=True)
        vectors = self.embedder.embed_texts([chunk.embed_text for chunk in chunks])

        (self.root / _CHUNKS_FILE).write_text(
            json.dumps([chunk.model_dump() for chunk in chunks], indent=2) + "\n",
            encoding="utf-8",
        )
        np.save(self.root / _VECTORS_FILE, vectors)
        (self.root / _META_FILE).write_text(
            json.dumps(
                {
                    "dimensions": int(self.embedder.dimensions),
                    "chunk_count": len(chunks),
                    "kinds": sorted({chunk.source_kind.value for chunk in chunks}),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        logger.info("Wrote %d chunks to %s", len(chunks), self.root)

    # --- Reading ---------------------------------------------------------
    @property
    def is_built(self) -> bool:
        """Whether an index exists at all. Absence is a normal state."""
        return (self.root / _CHUNKS_FILE).is_file() and (
            self.root / _VECTORS_FILE
        ).is_file()

    def _load(self) -> tuple[list[Chunk], np.ndarray]:
        if self._chunks is not None and self._vectors is not None:
            return self._chunks, self._vectors

        if not self.is_built:
            logger.info(
                "No knowledge index at %s -- retrieval returns nothing", self.root
            )
            self._chunks, self._vectors = (
                [],
                np.zeros((0, self.embedder.dimensions), dtype=np.float32),
            )
            return self._chunks, self._vectors

        raw = json.loads((self.root / _CHUNKS_FILE).read_text(encoding="utf-8"))
        chunks = [Chunk.model_validate(item) for item in raw]
        vectors = np.load(self.root / _VECTORS_FILE)

        if len(chunks) != vectors.shape[0]:
            raise ValueError(
                f"Index at {self.root} is corrupt: chunks.json and vectors.npy "
                f"disagree ({len(chunks)} chunks, {vectors.shape[0]} vectors). "
                "Rebuild it with: uv run python scripts/ingest_knowledge.py"
            )
        if vectors.shape[0] and vectors.shape[1] != self.embedder.dimensions:
            raise ValueError(
                f"Index at {self.root} was built with {vectors.shape[1]}-dimensional "
                f"vectors but the configured embedder produces "
                f"{self.embedder.dimensions}. Rebuild it with: "
                "uv run python scripts/ingest_knowledge.py"
            )

        self._chunks, self._vectors = chunks, vectors
        return chunks, vectors

    @property
    def chunks(self) -> list[Chunk]:
        """Every stored chunk, in index order."""
        return self._load()[0]

    def get(self, chunk_id: str) -> Chunk | None:
        """Look one chunk up by id.

        What provenance filtering is built on, and what a later session's
        ``validate_grounding`` will call: a fact citing an id we never stored is a
        fact we cannot stand behind.
        """
        return next((c for c in self.chunks if c.chunk_id == chunk_id), None)

    def search(
        self,
        query: str,
        source_kind: SourceKind | None = None,
        top_k: int = 5,
    ) -> list[SearchHit]:
        """Cosine search, optionally restricted to one kind of source.

        Filtering happens *before* scoring rather than after, which is lesson 9's
        metadata filtering: filtering afterwards would spend the top-k budget on
        chunks the caller cannot use and could return fewer than ``top_k`` results
        while relevant ones existed.
        """
        chunks, vectors = self._load()
        if not chunks:
            return []

        indices = [
            i
            for i, chunk in enumerate(chunks)
            if source_kind is None or chunk.source_kind is source_kind
        ]
        if not indices:
            return []

        # Vectors are unit length by construction (see embed.py), so a dot product
        # is already the cosine and there is nothing to normalise here.
        query_vector = self.embedder.embed_query(query)
        scores = vectors[indices] @ query_vector

        ranked = np.argsort(scores)[::-1][:top_k]
        return [
            SearchHit(chunk=chunks[indices[int(i)]], similarity=float(scores[int(i)]))
            for i in ranked
        ]
