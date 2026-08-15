"""The corpus and its vectors in Postgres, searched by one SQL statement.

Satisfies :class:`~sparkstory.retrieval.protocol.ChunkStore`, so the retrieval
tools and ``drop_unprovenanced`` cannot tell it from the store it replaces.

**The whole hybrid search is one query**, which is the difference between this and
the numpy store it replaces. There, a vector search ran in numpy and a BM25 index
ran in Python and ``HybridIndex`` fused the two rankings. Here pgvector's ``<=>``
and Postgres's ``ts_rank`` each produce a ranking inside the database and RRF
fuses them in a CTE. Nothing loads the corpus into memory to answer a query.

**``ts_rank`` is not BM25, and that was the risk.** It has no ``k1``/``b``
saturation parameters, and the ``english`` configuration drops ``no`` -- which the
hand-picked stoplist deliberately *kept*, because it carries the meaning of half
the fact corpus ("the Moon has *no* air"). Postgres-native BM25 exists only
through an extension such as VectorChord, not through stock ``tsvector``.

So it was measured before being specced, on the real corpus and the real 20
labelled queries:

    hit-rate@1        0.85 -> 0.85    identical
    fusion vs vectors 17-16 -> 17-16  identical; the keyword half still earns its place
    top-2             19/20 -> 18/20  ONE HIT WORSE, and that is a real regression
    hit-rate@3        1.00 -> 1.00    saturated; it cannot fail, so it proves nothing

The top-2 loss is recorded rather than rounded away. It is one of only two
measurements with room to move, and it moved the wrong way.

**Sync, not async.** ``search_facts`` and ``search_craft`` are sync ``@tool``
functions invoked from inside LangGraph's event loop. An async store would need
``asyncio.run`` there, which raises inside a running loop, so async here would
force those signatures to change too.
"""

from typing import Any

from sqlalchemy import Engine, MetaData, Table, create_engine, text

from sparkstory.db.models import chunks_table
from sparkstory.entities.exceptions import ConfigurationError
from sparkstory.retrieval.chunks import Chunk, SourceKind
from sparkstory.retrieval.embed import Embedder
from sparkstory.retrieval.types import SearchHit
from sparkstory.utils.logging_utils import get_logger

logger = get_logger(__name__)

#: Reciprocal Rank Fusion constant, unchanged from the numpy implementation. A
#: `ts_rank` score and a cosine distance cannot be averaged without inventing a
#: weight, so fusion stays rank-only -- the original argument, still true.
RRF_K = 60

#: Fetch more candidates per half than are returned, so a chunk ranked modestly by
#: both retrievers can still win on fusion. Mirrors the numpy implementation.
_CANDIDATE_MULTIPLIER = 4
_MINIMUM_CANDIDATES = 20


class PgVectorStore:
    """Chunks, their vectors and a keyword index, all in one Postgres table."""

    def __init__(
        self,
        database_url: str,
        embedder: Embedder,
        embedding_model: str,
        engine: Engine | None = None,
    ) -> None:
        """Build a store bound to one table.

        Args:
            database_url: A ``postgresql+psycopg://`` URL.
            embedder: Used for ``save`` and to embed a query. Injected rather than
                built here so a test can supply ``FakeEmbedder``.
            embedding_model: The registry entry name. Together with the embedder's
                dimensionality it picks the table, so a store can only ever read
                vectors made by the embedder it was given.
            engine: Injected in tests. Defaults to one created from ``database_url``.
        """
        self.embedder = embedder
        self.embedding_model = embedding_model
        self._engine = engine or create_engine(database_url)
        # A fresh MetaData rather than Base.metadata: constructing two stores for
        # the same embedder in one process (a test does exactly this) would
        # otherwise raise on a duplicate table name.
        self.table: Table = chunks_table(
            embedding_model, embedder.dimensions, MetaData()
        )

    # --- Writing ---------------------------------------------------------
    def save(self, chunks: list[Chunk]) -> None:
        """Embed every chunk and replace the table's contents.

        Whole-corpus replacement inside one transaction, not an upsert. Ingestion
        is an offline step that runs on the whole corpus at once, so a partial
        write is the one outcome that should be impossible: a half-written index
        answers queries confidently and wrongly.

        Embedding happens here rather than in the caller so vectors and rows
        cannot be assembled separately and drift out of step.
        """
        vectors = self.embedder.embed_texts([chunk.embed_text for chunk in chunks])

        with self._engine.begin() as conn:
            conn.execute(self.table.delete())
            if chunks:
                conn.execute(
                    self.table.insert(),
                    [
                        {
                            "chunk_id": chunk.chunk_id,
                            "text": chunk.text,
                            "title": chunk.title,
                            "embed_text": chunk.embed_text,
                            "source": chunk.source,
                            "licence": chunk.licence,
                            "url": chunk.url,
                            "source_kind": chunk.source_kind.value,
                            "content_sha256": chunk.content_sha256,
                            # pgvector accepts its literal form, "[0.1,0.2,...]",
                            # which is what str(list) produces. Avoids depending on
                            # the `pgvector` package for an adapter we would use
                            # in exactly one place.
                            "embedding": str(vector.tolist()),
                        }
                        for chunk, vector in zip(chunks, vectors, strict=True)
                    ],
                )
        logger.info("Wrote %d chunks to %s", len(chunks), self.table.name)

    # --- Reading ---------------------------------------------------------
    @property
    def is_built(self) -> bool:
        """Whether the table exists and holds anything.

        A property rather than a method, matching the store this replaces: callers
        write ``if store.is_built``, and a bound method would be truthy always --
        an unbuilt store would look built.

        A missing table is a normal, reportable state rather than an error, because
        the corpus is ingested by a separate offline step and forgetting it is a
        setup mistake an operator should be told about plainly.
        """
        try:
            with self._engine.connect() as conn:
                count = conn.execute(
                    text(f"SELECT count(*) FROM {self.table.name}")  # noqa: S608
                ).scalar_one()
        except Exception:  # noqa: BLE001 -- any connection or missing-table error
            return False
        return bool(count)

    @property
    def chunks(self) -> list[Chunk]:
        """Every stored chunk.

        A full table scan, and it is here because ``provenance.py`` and the corpus
        quality tests need it -- see the Protocol's note. Fine at 58 rows; the
        first thing to replace with a bulk ``get_many(ids)`` if the corpus grows.
        """
        with self._engine.connect() as conn:
            rows = conn.execute(
                self.table.select().order_by(self.table.c.chunk_id)
            ).mappings()
            return [_chunk_from_row(row) for row in rows]

    def get(self, chunk_id: str) -> Chunk | None:
        """Look one chunk up by id, or ``None``.

        The trust gate. ``drop_unprovenanced`` resolves every id a model wrote
        through this and rewrites ``source`` from the row, which is what makes a
        fabricated citation unreachable rather than merely detectable.
        """
        with self._engine.connect() as conn:
            row = (
                conn.execute(
                    self.table.select().where(self.table.c.chunk_id == chunk_id)
                )
                .mappings()
                .first()
            )
        return _chunk_from_row(row) if row else None

    def search_vectors_only(
        self,
        query: str,
        source_kind: SourceKind | None = None,
        top_k: int = 5,
    ) -> list[SearchHit]:
        """Rank by vector similarity alone, with no keyword half and no fusion.

        **Exists for the measurement, not for the application**, and nothing in the
        pipeline calls it. Fusion's entire justification is that it beats
        vector-only retrieval -- 17 to 16 at top-1 on the labelled set when that
        was last measured -- and a claim like that has to stay checkable. Without
        this, "is the keyword half earning its place?" would be unanswerable
        without hand-writing SQL in a test.

        The previous implementation got this for free: ``HybridIndex`` wrapped a
        vector store, so a test could reach through to ``index.store.search``.
        Fusing inside one SQL statement removes that seam, so the comparison
        becomes an explicit method rather than an accident of composition.
        """
        statement = text(f"""
            SELECT *
            FROM {self.table.name}
            WHERE (
                CAST(:kind AS text) IS NULL
                OR source_kind = CAST(:kind AS text)
            )
            ORDER BY embedding <=> CAST(:qvec AS vector)
            LIMIT :top_k
        """)  # noqa: S608 -- identifier only, see `search`
        params = {
            "qvec": str(self.embedder.embed_query(query).tolist()),
            "kind": source_kind.value if source_kind else None,
            "top_k": top_k,
        }
        with self._engine.connect() as conn:
            rows = conn.execute(statement, params).mappings().all()
        # `similarity` is ordering-only by contract and this method exists to be
        # ranked rather than scored, so it is left at zero rather than converting a
        # cosine *distance* into something that would look comparable to an RRF
        # score and is not.
        return [SearchHit(chunk=_chunk_from_row(row), similarity=0.0) for row in rows]

    def search(
        self,
        query: str,
        source_kind: SourceKind | None = None,
        top_k: int = 5,
    ) -> list[SearchHit]:
        """Hybrid search: pgvector and ts_rank, fused by RRF, in one statement.

        ``source_kind`` filters *before* scoring rather than after. Filtering
        afterwards spends the top-k budget on rows the caller cannot use, and can
        return fewer than ``top_k`` results while relevant ones exist.

        The keyword half **abstains structurally**. ``hybrid.py`` counted matched
        terms and skipped BM25 below two, because BM25 falls back to ranking by
        document length and RRF cannot tell that vote from a real one. In SQL the
        ``@@`` operator returns no rows when nothing matches, so a query with no
        known terms simply casts no vote. Same protection, expressed by the shape
        of the query rather than by a threshold -- and this is the one behaviour
        that is imitated rather than ported.

        Returns:
            Up to ``top_k`` hits. ``similarity`` is the **RRF score**, not a
            cosine: it preserves ordering and is not comparable across queries.
            Nothing may threshold on it.
        """
        candidates = max(top_k * _CANDIDATE_MULTIPLIER, _MINIMUM_CANDIDATES)
        query_vector = str(self.embedder.embed_query(query).tolist())

        # The table name is interpolated because an identifier cannot be a bound
        # parameter. It is not user input: it comes from `table_name_for`, which
        # collapses everything outside [0-9a-zA-Z] to underscores, so a registry
        # entry cannot inject SQL. Every actual *value* below is bound.
        statement = text(f"""
            WITH vec AS (
                SELECT
                    chunk_id,
                    ROW_NUMBER() OVER (
                        ORDER BY embedding <=> CAST(:qvec AS vector)
                    ) AS rank
                FROM {self.table.name}
                WHERE (
                    CAST(:kind AS text) IS NULL
                    OR source_kind = CAST(:kind AS text)
                )
                ORDER BY embedding <=> CAST(:qvec AS vector)
                LIMIT :candidates
            ),
            kw AS (
                SELECT
                    chunk_id,
                    ROW_NUMBER() OVER (
                        ORDER BY ts_rank(
                            tsv, websearch_to_tsquery('english', :q)
                        ) DESC
                    ) AS rank
                FROM {self.table.name}
                WHERE (
                    CAST(:kind AS text) IS NULL
                    OR source_kind = CAST(:kind AS text)
                )
                  AND tsv @@ websearch_to_tsquery('english', :q)
                LIMIT :candidates
            )
            SELECT chunk_id, SUM(score) AS rrf FROM (
                SELECT chunk_id, 1.0 / (:rrf_k + rank) AS score FROM vec
                UNION ALL
                SELECT chunk_id, 1.0 / (:rrf_k + rank) AS score FROM kw
            ) fused
            GROUP BY chunk_id
            ORDER BY rrf DESC, chunk_id
            LIMIT :top_k
        """)  # noqa: S608 -- identifier only; see the comment above

        params: dict[str, Any] = {
            "qvec": query_vector,
            "q": query,
            "kind": source_kind.value if source_kind else None,
            "candidates": candidates,
            "rrf_k": RRF_K,
            "top_k": top_k,
        }

        with self._engine.connect() as conn:
            ranked = conn.execute(statement, params).fetchall()
            if not ranked:
                return []

            ids = [row[0] for row in ranked]
            rows = (
                conn.execute(self.table.select().where(self.table.c.chunk_id.in_(ids)))
                .mappings()
                .all()
            )

        # Re-order to match the fused ranking: an IN () query returns rows in
        # whatever order Postgres likes, and the ordering IS the result here.
        by_id = {row["chunk_id"]: _chunk_from_row(row) for row in rows}
        return [
            SearchHit(chunk=by_id[chunk_id], similarity=float(score))
            for chunk_id, score in ranked
            if chunk_id in by_id
        ]


def _chunk_from_row(row: Any) -> Chunk:
    """Rebuild a ``Chunk`` from a table row.

    ``content_sha256`` and ``embed_text`` are columns but *properties* on
    ``Chunk``, derived from ``text`` and ``title``. They are stored so the database
    is queryable on its own terms and so a rewritten chunk under a reused
    positional id stays detectable -- but they are not passed to the constructor,
    which would fail on a property with no setter.
    """
    return Chunk(
        chunk_id=row["chunk_id"],
        text=row["text"],
        title=row["title"],
        source=row["source"],
        licence=row["licence"],
        url=row["url"],
        source_kind=SourceKind(row["source_kind"]),
    )


def build_store(
    database_url: str | None, embedder: Embedder, embedding_model: str
) -> PgVectorStore:
    """Build a store, or fail with a message naming what to set.

    Raises:
        ConfigurationError: ``DATABASE_URL`` is unset. Deliberately *not* a bare
            ``ValueError``: a built-in exception type must never stand for a
            domain condition, because the tool layer translates
            ``ConfigurationError`` into something an operator can act on and lets
            everything else surface as the bug it is.
    """
    if not database_url:
        raise ConfigurationError(
            "DATABASE_URL is not set, so there is no corpus to search. "
            "Start one with `docker compose up -d postgres`, then run "
            "`make migrate && make ingest`."
        )
    return PgVectorStore(
        database_url=database_url, embedder=embedder, embedding_model=embedding_model
    )
