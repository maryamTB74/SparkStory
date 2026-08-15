"""Per-child memory in Postgres: append-only writes, exact reads.

**Reads are SQL equality, never similarity, and that is the whole design.** The
Writer must see the same character facts on every run: a near-miss here *is* the
Finn/Kit defect, where one premise produced a fox called Finn in one run and Kit
in another. So ``fetch`` takes no query and no ``top_k`` -- it returns everything
this child has, ordered deterministically.

The episodic tier is the exception and is searched by vector, because "what
stories has this child had" is genuinely fuzzy and a near-miss there is harmless.

**Nothing is ever updated in place.** A contradiction adds a row; a resolution
adds a pointer. The books on the shelf stay true to what was written when they
were made.

**Sync, not async**, matching ``PgVectorStore`` and for the same reason: the
callers are LangGraph ``@task`` bodies already inside a running event loop, where
``asyncio.run`` raises.
"""

from sqlalchemy import Engine, MetaData, Table, create_engine, select

from sparkstory.config import settings
from sparkstory.db.models import memories_table
from sparkstory.entities.exceptions import ConfigurationError
from sparkstory.memory.types import ChildId, MemoryKind, MemoryRecord
from sparkstory.utils.logging_utils import get_logger

logger = get_logger(__name__)


class PgMemoryStore:
    """Everything one child's stories have taught us."""

    def __init__(
        self,
        database_url: str | None,
        engine: Engine | None = None,
        embedding_dimensions: int | None = None,
    ) -> None:
        """Build a store.

        Args:
            database_url: A ``postgresql+psycopg://`` URL. Ignored when ``engine``
                is supplied.
            engine: Injected in tests, matching ``PgVectorStore``'s seam. The
                offline suite passes SQLite.
            embedding_dimensions: Episodic vector width, or ``None`` to omit the
                column. ``None`` on SQLite, which has no ``vector`` type.

        Raises:
            ConfigurationError: no engine and no ``DATABASE_URL``. Never a bare
                ``ValueError``, because the tool layer translates this into
                something an operator can act on.
        """
        if engine is None:
            if not database_url:
                raise ConfigurationError(
                    "DATABASE_URL is not set, so there is no memory to read. "
                    "Start one with `docker compose up -d postgres`, then run "
                    "`make migrate`."
                )
            engine = create_engine(database_url)
        self._engine = engine
        # A fresh MetaData, as PgVectorStore does: constructing two stores in one
        # process would otherwise raise on a duplicate table name.
        self.table: Table = memories_table(
            metadata=MetaData(), embedding_dimensions=embedding_dimensions
        )

    # --- Writing ---------------------------------------------------------
    def save(self, records: list[MemoryRecord]) -> None:
        """Append records. Never updates, never deletes.

        Returns early on an empty list rather than opening a transaction: a book
        that established nothing is a normal outcome, not a write of nothing.
        """
        if not records:
            return
        with self._engine.begin() as conn:
            conn.execute(
                self.table.insert(),
                [
                    {
                        "memory_id": record.memory_id,
                        "child_id": record.child_id,
                        "kind": record.kind.value,
                        "text": record.text,
                        "subject": record.subject,
                        "source_request_id": record.source_request_id,
                        "created_at": record.created_at,
                        "superseded_by": record.superseded_by,
                    }
                    for record in records
                ],
            )
        logger.info("Wrote %d memories for %s", len(records), records[0].child_id)

    def supersede(self, memory_id: str, by: str) -> None:
        """Retire a record without deleting it.

        The only mutation this store performs, and it touches one nullable
        pointer. The retired row keeps its text, so a book generated while it was
        live remains explicable.
        """
        with self._engine.begin() as conn:
            conn.execute(
                self.table.update()
                .where(self.table.c.memory_id == memory_id)
                .values(superseded_by=by)
            )

    # --- Reading ---------------------------------------------------------
    def fetch(
        self, child_id: ChildId, kind: MemoryKind | None = None
    ) -> list[MemoryRecord]:
        """Every live record for this child, oldest first.

        No query, no ``top_k``, no similarity -- see the module docstring.

        Ordering is by ``created_at`` then ``memory_id`` so that two rows written
        in the same transaction, which share a timestamp, cannot swap places
        between calls. Without the tiebreak an identical fetch could return the
        same rows in a different order, and the determinism this exists for would
        hold only by luck.
        """
        stmt = select(self.table).where(
            self.table.c.child_id == child_id,
            self.table.c.superseded_by.is_(None),
        )
        if kind is not None:
            stmt = stmt.where(self.table.c.kind == kind.value)
        stmt = stmt.order_by(self.table.c.created_at, self.table.c.memory_id)

        with self._engine.connect() as conn:
            rows = conn.execute(stmt).mappings().all()

        return [
            MemoryRecord(
                memory_id=row["memory_id"],
                child_id=row["child_id"],
                kind=MemoryKind(row["kind"]),
                text=row["text"],
                subject=row["subject"],
                source_request_id=row["source_request_id"],
                created_at=row["created_at"],
                superseded_by=row["superseded_by"],
            )
            for row in rows
        ]


def build_memory_store() -> PgMemoryStore:
    """Build a store from settings.

    Lives here rather than in either workflow because *both* use it -- the outline
    pipeline reads, the story pipeline writes -- and ``workflows/retries.py``
    exists precisely so that neither workflow has to import the other. It is also
    the single seam a test patches to intercept memory, matching
    ``build_research_context``.
    """
    config = settings.embedding_configs[settings.embedding_model]
    return PgMemoryStore(
        database_url=settings.database_url,
        embedding_dimensions=int(config["dimensions"]),
    )
