"""create the chunks table for the default embedder

Revision ID: 1832964e2bc0
Revises:
Create Date: 2026-08-09 12:32:41.467588

Creates the pgvector extension and one chunks table, for whichever embedder
``EMBEDDING_MODEL`` names when this migration runs.

The table name embeds the embedder and its dimensionality
(``chunks_potion_base_8m_256``) because a ``vector(n)`` column is fixed width and
two embedders cannot share one. So the schema depends on the registry in a way
Alembic's autogenerate cannot see -- hence a hand-written revision that calls the
same ``chunks_table()`` factory the store uses, rather than restating the columns
here where they could silently drift from the model.

**The embedder is named literally, and that is a correction.** This revision used
to read ``settings.embedding_model``, so the table it built depended on an
environment variable at the moment it ran. That was invisible while there was one
registry entry and it was the default. When the default moved to
``gemini-embedding``, this revision began building *that* table on any machine
with no ``EMBEDDING_MODEL`` set -- so a fresh database got no local table at all,
and revision 9c4a1f7b2d10 then failed with ``DuplicateTable`` because the table it
creates already existed. CI caught it; a laptop with the variable set never would.

A migration is a record of one schema change and has to produce the same schema
wherever it runs, which is why the course's eight migrations import no settings at
all. Importing the *factory* is a different thing from reading the *environment*:
the factory is what keeps the column width and the table name from disagreeing,
and it is pinned to a key here so the result cannot vary.

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


#: The registry key this revision creates a table for. A literal, so the schema
#: this migration produces does not depend on ``EMBEDDING_MODEL``. Revision
#: 9c4a1f7b2d10 names its own embedder the same way, for the same reason.
_MODEL_ID = "potion-base-8M"


def _table() -> sa.Table:
    """Build the table description for the local embedder.

    A fresh ``MetaData`` rather than ``Base.metadata``: this module is imported
    once per migration run, and registering the same table name on the shared
    metadata twice raises.
    """
    config = settings.embedding_configs[_MODEL_ID]
    return chunks_table(_MODEL_ID, int(config["dimensions"]), sa.MetaData())


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
