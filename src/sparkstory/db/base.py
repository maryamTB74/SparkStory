"""The declarative base every table definition hangs off.

Nine lines, and what they exist for is ``Base.metadata``: Alembic needs a single
``MetaData`` to point ``target_metadata`` at.

**There is deliberately no session module beside this one.** A managed-session
layer -- a per-event-loop cache of engines and connectors -- solves a real
problem when a hosted connector binds to the event loop at creation time and the
server handles requests in more than one loop context. SparkStory has none of
that: no hosted connector, no second loop context, and no chosen deployment
host. Building it would import the solution to a problem this project does not
have. Connections are made where they are used, from ``PgVectorStore``,
synchronously.
"""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Base class for SparkStory's SQLAlchemy models."""
