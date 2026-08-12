"""add memories table

Revision ID: 425780473a85
Revises: 1832964e2bc0
Create Date: 2026-08-11 23:02:46.875744

Creates the per-child memories table: what SparkStory remembers between books.

**Calls the same factory the store uses**, as the chunks revision does, rather
than restating the columns here where they could silently drift from the model.

**One table for both tiers, unlike chunks.** ``chunks_table`` is parameterised by
embedder because every row there carries a fixed-width ``vector(n)``. Here only
*episodic* rows are embedded and the column is nullable, so a change of embedder
re-embeds some rows rather than invalidating the table -- and one child's memory
stays in one place, which is the point of the store.

The vector width still has to come from settings, because the column is
``vector(n)``. A database migrated under one embedder and then queried under a
wider one would find the column too narrow for its episodes; the semantic tier,
which carries this feature's guarantees, is unaffected either way.
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op
from sparkstory.config import settings
from sparkstory.db.models import memories_table

# revision identifiers, used by Alembic.
revision: str = "425780473a85"
down_revision: str | Sequence[str] | None = "1832964e2bc0"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _table() -> sa.Table:
    """Build the table description for the configured embedder.

    A fresh ``MetaData`` rather than ``Base.metadata``, for the same reason the
    chunks revision uses one: registering the same table name twice raises.
    """
    config = settings.embedding_configs[settings.embedding_model]
    return memories_table(
        metadata=sa.MetaData(), embedding_dimensions=int(config["dimensions"])
    )


def upgrade() -> None:
    """Create the memories table.

    No ``CREATE EXTENSION`` here: the chunks revision runs first and installs
    pgvector, and this revision declares that dependency through
    ``down_revision``. Repeating it would be harmless but would imply this
    migration stands alone, which it does not -- the ``vector`` column below
    needs what that one created.
    """
    # `checkfirst=False`: this revision is the thing that creates the table, so a
    # table already present means the migration state and the database disagree,
    # and failing loudly is the correct outcome.
    _table().create(op.get_bind(), checkfirst=False)


def downgrade() -> None:
    """Drop the table, and deliberately not the extension.

    The chunks table needs `vector` too, and it is not this revision's to remove.
    """
    _table().drop(op.get_bind(), checkfirst=False)
