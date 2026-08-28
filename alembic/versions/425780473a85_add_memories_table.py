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

**The vector width is a literal, and that is a correction.** This revision used to
read ``settings.embedding_model``, so the column it built was 256 or 768 wide
depending on what the operator happened to have configured -- measured, not
inferred: the same revision was run twice against a clean database and produced
both widths. A migration has to produce the same schema wherever it runs, which
is why the course's eight migrations import no settings at all, and it is the
same defect that made revision 1832964e2bc0 collide with 9c4a1f7b2d10.

Pinned to ``potion-base-8M`` because that was the default when this revision was
authored, so it is the width every already-migrated database holds. The factory
import stays: it is what keeps this column and ``memories_table`` from drifting.

**What pinning does not solve, stated because it is a real limit.** Unlike chunks,
there is one ``memories`` table rather than one per embedder, and ``build_memory_store``
still asks for the width of whichever embedder is configured. So a run under a
768-dim embedder will find this column too narrow for its episodes. That is a
pre-existing property of the one-table design, not something this change
introduced -- and it now fails the same way on every machine instead of depending
on who ran the migration. The semantic tier, which carries this feature's
guarantees, uses no vector column and is unaffected either way.
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


#: The registry key this revision takes its vector width from. A literal, so the
#: schema does not depend on ``EMBEDDING_MODEL``. Both chunks revisions name
#: their embedder the same way, for the same reason.
_MODEL_ID = "potion-base-8M"


def _table() -> sa.Table:
    """Build the table description at this revision's pinned width.

    A fresh ``MetaData`` rather than ``Base.metadata``, for the same reason the
    chunks revision uses one: registering the same table name twice raises.
    """
    config = settings.embedding_configs[_MODEL_ID]
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
