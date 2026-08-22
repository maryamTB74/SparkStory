"""add the chunks table for the hosted Gemini embedder

Revision ID: 9c4a1f7b2d10
Revises: 425780473a85
Create Date: 2026-08-19

Creates ``chunks_gemini_embedding_768`` beside the existing local-embedder table,
which is left untouched. This is the "adding a second embedder means a new
revision calling the same factory" case the first chunks migration anticipated.

**This revision names its embedder literally, unlike revision 1832964e2bc0.**
That earlier migration reads ``settings.embedding_model``, so the table it builds
depends on an environment variable at the moment it runs. That was defensible
when there was one registry entry and it was the default; it is not now, because
the default has moved. Replaying that migration today would build *this* table
rather than the local one it originally created, and a migration that produces a
different schema on a different machine is not a migration.

So the registry entry is read by key here. The width still comes from
``embedding_configs`` rather than being restated, because ``dimensions`` is also
what the table *name* is built from -- taking them from two places is how the
name and the column silently disagree.

**If the spike changes the width**, this revision is the thing to edit, and the
edit is safe as long as it has not yet run anywhere: change ``dimensions`` in the
registry and the table name follows automatically. Once it has run, the honest
move is a new revision creating the new-width table, because the old one holds
vectors that cannot be converted.
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op
from sparkstory.config import settings
from sparkstory.db.models import chunks_table

# revision identifiers, used by Alembic.
revision: str = "9c4a1f7b2d10"
down_revision: str | Sequence[str] | None = "425780473a85"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

#: The registry key this revision creates a table for. A literal, so the schema
#: this migration produces does not depend on `EMBEDDING_MODEL`.
_MODEL_ID = "gemini-embedding"


def _table() -> sa.Table:
    """Build the table description for the hosted embedder.

    A fresh ``MetaData`` for the reason the first chunks revision gives: this
    module is imported once per migration run, and registering the same table
    name on shared metadata twice raises.
    """
    config = settings.embedding_configs[_MODEL_ID]
    return chunks_table(_MODEL_ID, int(config["dimensions"]), sa.MetaData())


def upgrade() -> None:
    """Create the table. The extension is already present from revision 1."""
    # IF NOT EXISTS rather than assuming: this revision can be the first to run
    # on a database restored from a dump that predates the extension, and the
    # cost of the guard is nothing.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    table = _table()
    # `checkfirst=False` for the reason revision 1 gives: this revision is what
    # creates the table, so finding one already there means the migration state
    # and the database disagree, and that should fail rather than pass quietly.
    table.create(op.get_bind(), checkfirst=False)


def downgrade() -> None:
    """Drop this embedder's table only.

    Not the extension, and not the local embedder's table. Both are shared: the
    extension by every chunks table and the memories table, and the local table
    by anyone who has set ``EMBEDDING_MODEL=potion-base-8M`` to get retrieval
    working while Google is unavailable.
    """
    _table().drop(op.get_bind(), checkfirst=False)
