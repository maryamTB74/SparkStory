"""The chunks table, one per embedder.

**A factory rather than a declarative class, and the reason is the table name.**
It is derived from the embedding registry entry and its dimensionality --
``chunks_potion_base_8m`` -- because two embedders producing different-width
vectors cannot share a fixed-width ``vector(n)`` column. A ``__tablename__`` has
to be known when the class body executes, and this one is not known until
settings are read.

The alternative was a declarative class per embedder, which works but makes
adding an embedder a code change rather than a registry entry, and the registry
exists precisely so that swapping models stays configuration.

**Why one table per embedder rather than one table migrated between widths.**
Both are queryable at the same time, so "is the hosted embedder actually better?"
is a direct A/B over the same corpus and the same 20 labelled queries. Under a
single migrated column, adding a hosted embedder would change dimensions, storage
and ranking at once, and a movement in hit-rate would be unattributable. This
project has already made that mistake once elsewhere, attributing an effect to a
change that later runs showed was not the cause.

**What this module honestly does not buy.** SQLAlchemy cannot express a pgvector
column: ``postgresql.base.ischema_names`` contains ``tsvector`` but not
``vector``. Combined with the ``GENERATED ALWAYS`` tsvector column, that means the
migration is hand-written with ``op.execute`` and Alembic's autogenerate is not
usable for this table. So the value here is a typed, single description of the
schema that the store and the migration are both checked against -- not code
generation.
"""

import re

from sqlalchemy import (
    Column,
    Computed,
    DateTime,
    Index,
    MetaData,
    String,
    Table,
    Text,
)
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.types import UserDefinedType

from sparkstory.db.base import Base


class Vector(UserDefinedType):
    """The pgvector ``vector(n)`` column type.

    SQLAlchemy has no built-in for it and the ``pgvector`` Python package is not a
    dependency: it supplies adapters we do not need, because a vector goes in as a
    string literal (``str(v.tolist())``) and comes back only as an ordering. Not a
    dependency until something needs it.
    """

    cache_ok = True

    def __init__(self, dimensions: int) -> None:
        self.dimensions = dimensions

    def get_col_spec(self, **_: object) -> str:
        return f"vector({self.dimensions})"


def table_name_for(embedding_model: str, dimensions: int) -> str:
    """Build the table name for an embedder, e.g. ``chunks_potion_base_8m_256``.

    The dimensionality is part of the name rather than only the column type, so a
    registry entry that silently changes width cannot quietly reuse a table built
    for the old one -- it looks for a table that does not exist, which is a clear
    failure instead of a dimension mismatch at query time.

    Non-identifier characters collapse to underscores because registry names are
    written for humans (``potion-base-8M``) and table names are not.
    """
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", embedding_model).strip("_").lower()
    return f"chunks_{slug}_{dimensions}"


def chunks_table(
    embedding_model: str, dimensions: int, metadata: MetaData | None = None
) -> Table:
    """Describe the chunks table for one embedder.

    Every column mirrors a field of ``retrieval.chunks.Chunk`` except ``tsv``,
    which is derived.

    Args:
        embedding_model: A key of ``settings.embedding_configs``.
        dimensions: That entry's vector width.
        metadata: Where to register the table. Defaults to ``Base.metadata``.
            Passing a fresh ``MetaData`` lets a test build the same table twice
            without SQLAlchemy raising on a duplicate name.
    """
    name = table_name_for(embedding_model, dimensions)
    return Table(
        name,
        metadata if metadata is not None else Base.metadata,
        # The positional id -- `moon#1`, the third chunk of corpus/facts/moon.md.
        # Primary key because provenance resolves against it: drop_unprovenanced
        # looks up every chunk_id a model wrote and overwrites `source` from the
        # row, which is what makes a fabricated citation unreachable rather than
        # merely detectable.
        Column("chunk_id", String, primary_key=True),
        Column("text", Text, nullable=False),
        Column("title", Text, nullable=False),
        # Context-enriched: "It has no air" embeds poorly alone and well as
        # "The Moon: It has no air". Deliberately distinct from `text`,
        # which is what the agent is shown.
        Column("embed_text", Text, nullable=False),
        # Authoritative attribution. Never model-written -- the store overwrites
        # whatever a model claimed.
        Column("source", Text, nullable=False),
        Column("licence", Text, nullable=False),
        # Nullable on purpose: in a feature whose whole point is factual accuracy,
        # an absent URL is better than a fabricated one.
        Column("url", Text, nullable=True),
        Column("source_kind", String, nullable=False),
        # Lets a rewritten chunk under a reused positional id be detected.
        Column("content_sha256", String, nullable=False),
        Column("embedding", Vector(dimensions), nullable=False),
        # GENERATED ALWAYS rather than written by the ingest step. A column the
        # application populates is a column it can forget to repopulate, and the
        # failure would be silent under-retrieval: rows that simply never match.
        # Indexes embed_text, not text, so both retrievers see the same words --
        # otherwise a query naming the source title ("moon") is findable by
        # vectors and missed by keywords.
        Column(
            "tsv",
            TSVECTOR,
            Computed("to_tsvector('english', embed_text)", persisted=True),
            nullable=True,
        ),
        Index(f"ix_{name}_tsv", "tsv", postgresql_using="gin"),
        Index(f"ix_{name}_source_kind", "source_kind"),
        # No ivfflat/hnsw index on `embedding`, deliberately. At 58 rows a
        # sequential scan beats an approximate index, and ANN trades recall for
        # speed -- the wrong trade when recall is the measured property. Add one
        # when the corpus demands it, and re-measure the labelled set when doing
        # so.
    )


def memories_table(
    metadata: MetaData | None = None, embedding_dimensions: int | None = None
) -> Table:
    """Describe the per-child memories table.

    **One table for both tiers, not one per embedder.** ``chunks_table`` is split
    by embedder because every row there carries a fixed-width vector. Here only
    *episodic* rows are embedded and ``embedding`` is nullable, so changing
    embedder re-embeds some rows rather than invalidating the table. Splitting
    would also scatter one child's memory across tables, which is the opposite of
    what this store is for.

    Args:
        metadata: Where to register the table. Pass a fresh ``MetaData`` to build
            it twice in one process, as a test does.
        embedding_dimensions: Width of the episodic vector column, or ``None`` to
            omit the column entirely. ``None`` is what lets the semantic tier be
            tested on SQLite, which has no ``vector`` type -- and the semantic
            tier is the one carrying this package's guarantees.
    """
    columns = [
        Column("memory_id", String, primary_key=True),
        # Indexed and present in every read: this is the scope, and no query
        # omits it. A leaked scope is one child's memory in another child's book.
        Column("child_id", String, nullable=False, index=True),
        Column("kind", String, nullable=False),
        Column("text", Text, nullable=False),
        # Nullable: episodic records are about a book, not a subject.
        Column("subject", Text, nullable=True),
        Column("source_request_id", String, nullable=False),
        Column("created_at", DateTime(timezone=True), nullable=False),
        # A resolution adds a pointer; it never deletes. Deliberately NOT a
        # ForeignKey even though it references this table: a record may be
        # superseded by one written in a later run, and the constraint would
        # order inserts for no gain.
        Column("superseded_by", String, nullable=True),
    ]
    if embedding_dimensions is not None:
        columns.append(Column("embedding", Vector(embedding_dimensions), nullable=True))

    return Table(
        "memories",
        metadata if metadata is not None else Base.metadata,
        *columns,
        # The exact-fetch access path: every read is (child_id, kind).
        Index("ix_memories_child_kind", "child_id", "kind"),
    )
