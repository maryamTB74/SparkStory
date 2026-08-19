"""Storage for the serial model: approve once, then chapters in order.

Two guarantees live here and nowhere else, so both are enforced in code rather
than described in a docstring:

**A chapter cannot exist in a world nobody approved.** ``queue_chapter`` refuses
a ``DRAFT`` or ``RETIRED`` world. This is the moved approval gate
(``entities/serial`` explains why it moved); if it can be routed around, the
parent approved nothing.

**Chapter ordinals are contiguous and per-season.** They are assigned by the
store from what is already there, never supplied by a caller, and the database
carries a unique constraint behind that. A child counts chapters, and two
chapter fours is a defect that is invisible in review and obvious on a shelf.

**Sync, not async**, matching ``PgMemoryStore`` and ``PgVectorStore``, and for
the same reason: the callers are LangGraph ``@task`` bodies already inside a
running event loop.
"""

import json
from datetime import UTC, datetime

from sqlalchemy import (
    Engine,
    MetaData,
    Row,
    Table,
    create_engine,
    func,
    select,
)
from sqlalchemy.sql.elements import ColumnElement

from sparkstory.config import settings
from sparkstory.db.models import chapters_table, seasons_table, worlds_table
from sparkstory.entities.exceptions import SparkStoryError
from sparkstory.entities.serial import (
    Chapter,
    ChapterState,
    Season,
    World,
    WorldStatus,
)
from sparkstory.entities.stories import Tone, WorldRules
from sparkstory.memory.types import ChildId
from sparkstory.utils.logging_utils import get_logger

logger = get_logger(__name__)


class WorldNotApproved(SparkStoryError):
    """A chapter was queued into a world no parent has approved.

    Its own type rather than a ``ValueError`` because the tool layer translates
    it: the operator-facing message is a prompt to approve the world, not a
    stack trace.
    """


class UnknownWorld(SparkStoryError):
    """A world id that no row matches."""


class PgSerialStore:
    """A child's shows, their seasons, and the chapters in them."""

    def __init__(self, database_url: str | None, engine: Engine | None = None) -> None:
        """Build a store.

        Args:
            database_url: A ``postgresql+psycopg://`` URL. Ignored when ``engine``
                is supplied.
            engine: Injected in tests, matching ``PgMemoryStore``'s seam. The
                offline suite passes SQLite, which every table here supports.

        Raises:
            ConfigurationError: no engine and no ``DATABASE_URL``.
        """
        if engine is None:
            if not database_url:
                from sparkstory.entities.exceptions import ConfigurationError

                raise ConfigurationError(
                    "DATABASE_URL is not set, so there are no worlds to read. "
                    "Start one with `docker compose up -d postgres`, then run "
                    "`make migrate`."
                )
            engine = create_engine(database_url)
        self._engine = engine
        # One fresh MetaData for all three, as PgMemoryStore does: two stores in
        # one process would otherwise raise on a duplicate table name.
        metadata = MetaData()
        self.worlds: Table = worlds_table(metadata=metadata)
        self.seasons: Table = seasons_table(metadata=metadata)
        self.chapters: Table = chapters_table(metadata=metadata)

    # --- Worlds ----------------------------------------------------------
    def create_world(self, world: World) -> World:
        """Insert a world exactly as given, approved or not.

        Deliberately does *not* force ``DRAFT``. A test, a migration and a
        seeded demo all have legitimate reasons to write an approved world, and
        a store that silently rewrites what it was handed is harder to reason
        about than one that stores it.
        """
        with self._engine.begin() as conn:
            conn.execute(self.worlds.insert(), [_world_row(world)])
        return world

    def approve_world(self, world_id: str) -> World:
        """Move a world to ``APPROVED`` and stamp when.

        Idempotent: approving an approved world returns it unchanged rather than
        re-stamping, so a double-tapped button does not rewrite the audit trail.
        """
        world = self.get_world(world_id)
        if world.status is WorldStatus.APPROVED:
            return world

        approved = world.approved()
        with self._engine.begin() as conn:
            conn.execute(
                self.worlds.update()
                .where(self.worlds.c.world_id == world_id)
                .values(status=approved.status.value, approved_at=approved.approved_at)
            )
        logger.info("world approved", extra={"world_id": world_id})
        return approved

    def retire_world(self, world_id: str) -> World:
        """Stop new chapters. Everything already written stays readable."""
        world = self.get_world(world_id)
        retired = world.model_copy(update={"status": WorldStatus.RETIRED})
        with self._engine.begin() as conn:
            conn.execute(
                self.worlds.update()
                .where(self.worlds.c.world_id == world_id)
                .values(status=retired.status.value)
            )
        return retired

    def get_world(self, world_id: str) -> World:
        """Fetch one world.

        Raises:
            UnknownWorld: no row matches.
        """
        with self._engine.begin() as conn:
            row = conn.execute(
                select(self.worlds).where(self.worlds.c.world_id == world_id)
            ).one_or_none()
        if row is None:
            raise UnknownWorld(f"no world with id {world_id!r}")
        return _world_from(row)

    def worlds_for(self, child_id: ChildId) -> list[World]:
        """Every world this child has, oldest first.

        Ordered by ``created_at`` then ``world_id`` -- the tiebreak matters
        because two worlds made in the same transaction share a timestamp, and
        an unstable shelf order is the kind of thing a child notices.
        """
        with self._engine.begin() as conn:
            rows = conn.execute(
                select(self.worlds)
                .where(self.worlds.c.child_id == child_id)
                .order_by(self.worlds.c.created_at, self.worlds.c.world_id)
            ).all()
        return [_world_from(r) for r in rows]

    # --- Seasons ---------------------------------------------------------
    def open_season(self, world_id: str, title: str) -> Season:
        """Start the next season of a world.

        The ordinal is computed here rather than passed in, for the same reason
        chapter ordinals are: a caller that has to count is a caller that can
        miscount.

        Raises:
            WorldNotApproved: the world is not approved.
        """
        world = self.get_world(world_id)
        if not world.accepts_chapters:
            raise WorldNotApproved(
                f"world {world_id!r} is {world.status.value}, so no season can "
                "open in it. A parent approves the world once, and every "
                "chapter is written inside what they approved."
            )

        season = Season(
            world_id=world_id,
            ordinal=self._next_season_ordinal(world_id),
            title=title,
        )
        with self._engine.begin() as conn:
            conn.execute(self.seasons.insert(), [_season_row(season)])
        return season

    def seasons_in(self, world_id: str) -> list[Season]:
        with self._engine.begin() as conn:
            rows = conn.execute(
                select(self.seasons)
                .where(self.seasons.c.world_id == world_id)
                .order_by(self.seasons.c.ordinal)
            ).all()
        return [_season_from(r) for r in rows]

    # --- Chapters --------------------------------------------------------
    def queue_chapter(self, season_id: str, world_id: str, seed: str) -> Chapter:
        """Add the next chapter of a season, in the ``QUEUED`` state.

        This is the moved approval gate. If a chapter can be queued into an
        unapproved world, the parent approved nothing.

        Raises:
            WorldNotApproved: the world is ``DRAFT`` or ``RETIRED``.
        """
        world = self.get_world(world_id)
        if not world.accepts_chapters:
            raise WorldNotApproved(
                f"world {world_id!r} is {world.status.value}, so no chapter can "
                "be queued in it."
            )

        chapter = Chapter(
            world_id=world_id,
            season_id=season_id,
            ordinal=self._next_chapter_ordinal(season_id),
            seed=seed,
        )
        with self._engine.begin() as conn:
            conn.execute(self.chapters.insert(), [_chapter_row(chapter)])
        return chapter

    def advance_chapter(
        self,
        chapter_id: str,
        state: ChapterState,
        request_id: str | None = None,
    ) -> None:
        """Record where a chapter has got to.

        ``ready_at`` is stamped when the chapter first becomes readable, and only
        then -- a chapter that later gains narration keeps the moment it was
        finished, not the moment it was read.
        """
        values: dict[str, object] = {"state": state.value}
        if request_id is not None:
            values["request_id"] = request_id
        if state is ChapterState.READY:
            values["ready_at"] = datetime.now(UTC)

        with self._engine.begin() as conn:
            conn.execute(
                self.chapters.update()
                .where(self.chapters.c.chapter_id == chapter_id)
                .values(**values)
            )

    def withdraw_chapter(self, chapter_id: str) -> None:
        """Pull a chapter from the review feed.

        A state change, never a delete. The child may already have read it, and
        a shelf that silently loses a book is worse than one that admits it.
        """
        self.advance_chapter(chapter_id, ChapterState.WITHDRAWN)

    def chapters_in(self, season_id: str) -> list[Chapter]:
        """Every chapter of a season in reading order, withdrawn ones included.

        Filtering here would make the review feed and the child's shelf two
        different queries over two different truths. The caller decides what to
        show.
        """
        with self._engine.begin() as conn:
            rows = conn.execute(
                select(self.chapters)
                .where(self.chapters.c.season_id == season_id)
                .order_by(self.chapters.c.ordinal)
            ).all()
        return [_chapter_from(r) for r in rows]

    def readable_chapters(self, season_id: str) -> list[Chapter]:
        """What the child is allowed to see: ready or already narrated."""
        return [
            c
            for c in self.chapters_in(season_id)
            if c.state in (ChapterState.READY, ChapterState.NARRATED)
        ]

    # --- Ordinals --------------------------------------------------------
    def _next_season_ordinal(self, world_id: str) -> int:
        return self._next_ordinal(
            self.seasons, self.seasons.c.world_id == world_id, self.seasons.c.ordinal
        )

    def _next_chapter_ordinal(self, season_id: str) -> int:
        return self._next_ordinal(
            self.chapters,
            self.chapters.c.season_id == season_id,
            self.chapters.c.ordinal,
        )

    def _next_ordinal(
        self,
        table: Table,
        where: ColumnElement[bool],
        column: ColumnElement[int],
    ) -> int:
        """One past the highest ordinal present, or 1.

        Max rather than count, deliberately. Counting would reuse the number of a
        withdrawn chapter, and the unique constraint would then reject the
        insert -- turning a parent's retraction into a failure to write the next
        chapter.
        """
        with self._engine.begin() as conn:
            highest = conn.execute(
                select(func.max(column)).select_from(table).where(where)
            ).scalar()
        return 1 if highest is None else int(highest) + 1


# --- Row mapping ---------------------------------------------------------
# Explicit rather than reflective. Every one of these is a place a rename could
# silently drop a field, and a test asserts the round trip rather than the map.


def _utc(value: datetime | None) -> datetime | None:
    """Re-attach UTC to a timestamp that lost it in the database.

    Postgres ``timestamptz`` round-trips an aware datetime; SQLite has no such
    type and hands back a naive one. Normalising here rather than in the caller
    means the store behaves identically on both, which is the point of testing
    on SQLite at all.

    Left unfixed this is not a cosmetic difference: a naive datetime compared
    against an aware one raises ``TypeError``, so the failure would surface far
    from here, in whatever first tried to order a shelf by date.
    """
    if value is None:
        return None
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _world_row(world: World) -> dict[str, object]:
    return {
        "world_id": world.world_id,
        "child_id": world.child_id,
        "title": world.title,
        "premise": world.premise,
        "tone": world.tone.value,
        "world_rules": world.world_rules.value,
        "avoid": json.dumps(world.avoid),
        "must_include": json.dumps(world.must_include),
        "status": world.status.value,
        "created_at": world.created_at,
        "approved_at": world.approved_at,
    }


def _world_from(row: Row) -> World:
    return World(
        world_id=row.world_id,
        child_id=row.child_id,
        title=row.title,
        premise=row.premise,
        tone=Tone(row.tone),
        world_rules=WorldRules(row.world_rules),
        avoid=json.loads(row.avoid),
        must_include=json.loads(row.must_include),
        status=WorldStatus(row.status),
        created_at=_utc(row.created_at),
        approved_at=_utc(row.approved_at),
    )


def _season_row(season: Season) -> dict[str, object]:
    return {
        "season_id": season.season_id,
        "world_id": season.world_id,
        "ordinal": season.ordinal,
        "title": season.title,
        "chapter_target": season.chapter_target,
        "created_at": season.created_at,
        "completed_at": season.completed_at,
    }


def _season_from(row: Row) -> Season:
    return Season(
        season_id=row.season_id,
        world_id=row.world_id,
        ordinal=row.ordinal,
        title=row.title,
        chapter_target=row.chapter_target,
        created_at=_utc(row.created_at),
        completed_at=_utc(row.completed_at),
    )


def _chapter_row(chapter: Chapter) -> dict[str, object]:
    return {
        "chapter_id": chapter.chapter_id,
        "season_id": chapter.season_id,
        "world_id": chapter.world_id,
        "ordinal": chapter.ordinal,
        "seed": chapter.seed,
        "state": chapter.state.value,
        "request_id": chapter.request_id,
        "created_at": chapter.created_at,
        "ready_at": chapter.ready_at,
    }


def _chapter_from(row: Row) -> Chapter:
    return Chapter(
        chapter_id=row.chapter_id,
        season_id=row.season_id,
        world_id=row.world_id,
        ordinal=row.ordinal,
        seed=row.seed,
        state=ChapterState(row.state),
        request_id=row.request_id,
        created_at=_utc(row.created_at),
        ready_at=_utc(row.ready_at),
    )


def build_serial_store() -> PgSerialStore:
    """The store the application uses, wired from settings."""
    return PgSerialStore(database_url=settings.database_url)
