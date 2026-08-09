"""The declarative base every table definition hangs off.

Nine lines, taken essentially verbatim from ``research_agent_part_3/src/db/base.py``
-- one of the few parts of the course's ``db/`` package that ports without
argument. What it exists for is ``Base.metadata``: Alembic needs a single
``MetaData`` to point ``target_metadata`` at.

**What did NOT port, and why it matters that it did not.** The course's
``session.py`` is 160 lines maintaining a per-event-loop cache of engines and
Cloud SQL connectors. Its own comments explain the reason: the Cloud SQL Python
Connector binds to the event loop at creation time, and FastMCP may handle UI
routes and MCP protocol handlers in different event loop contexts on Cloud Run.
SparkStory has no Cloud SQL, no UI routes, and cloud deployment is deferred with
no host chosen -- so copying it would have imported the solution to a problem
this project does not have. Connections here are made where they are used, from
``PgVectorStore``, synchronously.
"""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Base class for SparkStory's SQLAlchemy models."""
