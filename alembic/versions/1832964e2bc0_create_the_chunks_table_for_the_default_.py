"""create the chunks table for the default embedder

Revision ID: 1832964e2bc0
Revises:
Create Date: 2026-08-09 12:32:41.467588

Creates the pgvector extension and one chunks table, for whichever embedder
``EMBEDDING_MODEL`` names when this migration runs.

**This migration is parameterised by settings, which is unusual and deliberate.**
The table name embeds the embedder and its dimensionality
(``chunks_potion_base_8m_256``) because a ``vector(n)`` column is fixed width and
two embedders cannot share one. So the schema depends on configuration in a way
Alembic's autogenerate cannot see -- hence a hand-written revision that calls the
same ``chunks_table()`` factory the store uses, rather than restating the columns
here where they could silently drift from the model.

Adding a second embedder means a new revision calling the same factory with the
new registry entry. Both tables then coexist, which is what makes "is the hosted
embedder better?" a direct A/B over one corpus rather than a before-and-after.
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op
from sparkstory.config import settings
from sparkstory.db.models import chunks_table

# revision identifiers, used by Alembic.
revision: str = "1832964e2bc0"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _table() -> sa.Table:
    """Build the table description for the configured embedder.

    A fresh ``MetaData`` rather than ``Base.metadata``: this module is imported
    once per migration run, and registering the same table name on the shared
    metadata twice raises.
    """
    config = settings.embedding_configs[settings.embedding_model]
    return chunks_table(
        settings.embedding_model, int(config["dimensions"]), sa.MetaData()
    )


def upgrade() -> None:
    """Create the extension, then the table."""
    # The one thing here that cannot come from the model: pgvector is an
    # extension, and `vector(n)` is not a type until it is installed. IF NOT
    # EXISTS so a database that already has it (a shared instance, a second
    # embedder's migration) is not an error.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    table = _table()
    # `checkfirst=False`: this revision is the thing that creates the table, so a
    # table already present means the migration state and the database disagree,
    # and failing loudly is the correct outcome.
    table.create(op.get_bind(), checkfirst=False)


def downgrade() -> None:
    """Drop the table, and deliberately not the extension.

    Another table may depend on `vector`, including one for a different embedder
    created by a later revision. Dropping a shared extension on the way down would
    take those with it.
    """
    _table().drop(op.get_bind(), checkfirst=False)
