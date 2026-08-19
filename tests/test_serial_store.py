"""The serial store: approve once, then chapters in contiguous order.

Runs on the SQLite `serial_engine` fixture, so these execute under a plain
`make test` with no database. Nothing here needs Postgres -- see the fixture.
"""

import pytest
from sqlalchemy import Engine

from sparkstory.entities.serial import ChapterState, World, WorldStatus
from sparkstory.serial.store import PgSerialStore, UnknownWorld, WorldNotApproved


def _store(engine: Engine) -> PgSerialStore:
    return PgSerialStore(database_url=None, engine=engine)


def _world(child_id: str = "maryam", **overrides: object) -> World:
    base: dict[str, object] = {
        "child_id": child_id,
        "title": "Kit and the Moon",
        "premise": "a fox who wants to visit the Moon",
        "avoid": ["spiders"],
    }
    base.update(overrides)
    return World(**base)  # type: ignore[arg-type]


def _approved_world_with_season(store: PgSerialStore) -> tuple[str, str]:
    world = store.create_world(_world())
    store.approve_world(world.world_id)
    season = store.open_season(world.world_id, "Season One")
    return world.world_id, season.season_id


# --- The moved approval gate --------------------------------------------
def test_an_unapproved_world_takes_no_chapters(serial_engine: Engine) -> None:
    """If this can be routed around, the parent approved nothing."""
    store = _store(serial_engine)
    world = store.create_world(_world())

    with pytest.raises(WorldNotApproved):
        store.queue_chapter("any-season", world.world_id, "Kit finds a hat")


def test_an_unapproved_world_opens_no_season(serial_engine: Engine) -> None:
    store = _store(serial_engine)
    world = store.create_world(_world())

    with pytest.raises(WorldNotApproved):
        store.open_season(world.world_id, "Season One")


def test_a_retired_world_takes_no_new_chapters(serial_engine: Engine) -> None:
    store = _store(serial_engine)
    world_id, season_id = _approved_world_with_season(store)
    store.queue_chapter(season_id, world_id, "chapter one")

    store.retire_world(world_id)

    with pytest.raises(WorldNotApproved):
        store.queue_chapter(season_id, world_id, "chapter two")


def test_retiring_leaves_what_was_written_readable(serial_engine: Engine) -> None:
    """A child's finished books are not the parent's to un-write."""
    store = _store(serial_engine)
    world_id, season_id = _approved_world_with_season(store)
    chapter = store.queue_chapter(season_id, world_id, "chapter one")
    store.advance_chapter(chapter.chapter_id, ChapterState.READY)

    store.retire_world(world_id)

    assert [c.chapter_id for c in store.readable_chapters(season_id)] == [
        chapter.chapter_id
    ]


def test_approving_twice_does_not_rewrite_the_audit_trail(
    serial_engine: Engine,
) -> None:
    """A double-tapped button must not move the timestamp."""
    store = _store(serial_engine)
    world = store.create_world(_world())

    first = store.approve_world(world.world_id)
    second = store.approve_world(world.world_id)

    assert first.approved_at == second.approved_at
    assert store.get_world(world.world_id).status is WorldStatus.APPROVED


# --- Ordinals ------------------------------------------------------------
def test_chapter_ordinals_are_contiguous_and_start_at_one(
    serial_engine: Engine,
) -> None:
    store = _store(serial_engine)
    world_id, season_id = _approved_world_with_season(store)

    for i in range(4):
        store.queue_chapter(season_id, world_id, f"seed {i}")

    assert [c.ordinal for c in store.chapters_in(season_id)] == [1, 2, 3, 4]


def test_a_withdrawn_chapter_does_not_free_its_number(
    serial_engine: Engine,
) -> None:
    """The reason ordinals come from max and not from count.

    Counting would reuse a withdrawn chapter's number, the unique constraint
    would reject the insert, and a parent's retraction would surface as a
    failure to write the next chapter.
    """
    store = _store(serial_engine)
    world_id, season_id = _approved_world_with_season(store)
    first = store.queue_chapter(season_id, world_id, "one")
    store.queue_chapter(season_id, world_id, "two")

    store.withdraw_chapter(first.chapter_id)
    third = store.queue_chapter(season_id, world_id, "three")

    assert third.ordinal == 3


def test_two_seasons_number_their_chapters_independently(
    serial_engine: Engine,
) -> None:
    store = _store(serial_engine)
    world_id, first_season = _approved_world_with_season(store)
    second_season = store.open_season(world_id, "Season Two").season_id

    store.queue_chapter(first_season, world_id, "s1c1")
    store.queue_chapter(second_season, world_id, "s2c1")

    assert store.chapters_in(first_season)[0].ordinal == 1
    assert store.chapters_in(second_season)[0].ordinal == 1


def test_season_ordinals_count_up_per_world(serial_engine: Engine) -> None:
    store = _store(serial_engine)
    world_id, _ = _approved_world_with_season(store)

    second = store.open_season(world_id, "Season Two")

    assert second.ordinal == 2
    assert [s.ordinal for s in store.seasons_in(world_id)] == [1, 2]


# --- Scope ---------------------------------------------------------------
def test_one_child_never_sees_another_child_s_worlds(
    serial_engine: Engine,
) -> None:
    store = _store(serial_engine)
    store.create_world(_world("maryam", title="Kit and the Moon"))
    store.create_world(_world("sam", title="Ted and the Boat"))

    assert [w.title for w in store.worlds_for("maryam")] == ["Kit and the Moon"]


def test_an_unknown_world_is_an_error_not_an_empty_result(
    serial_engine: Engine,
) -> None:
    store = _store(serial_engine)

    with pytest.raises(UnknownWorld):
        store.get_world("no-such-world")


# --- Round trips ---------------------------------------------------------
def test_a_world_survives_the_round_trip_intact(serial_engine: Engine) -> None:
    """The row mapping is written by hand, so a dropped field is a real risk."""
    store = _store(serial_engine)
    world = _world(must_include=["a paper rocket"], avoid=["spiders", "thunder"])
    store.create_world(world)

    loaded = store.get_world(world.world_id)

    assert loaded.model_dump() == world.model_dump()


def test_a_chapter_survives_the_round_trip_intact(serial_engine: Engine) -> None:
    store = _store(serial_engine)
    world_id, season_id = _approved_world_with_season(store)
    chapter = store.queue_chapter(season_id, world_id, "Kit finds a hat")

    loaded = store.chapters_in(season_id)[0]

    assert loaded.model_dump() == chapter.model_dump()


# --- Progress ------------------------------------------------------------
def test_becoming_ready_stamps_when_and_narration_does_not_move_it(
    serial_engine: Engine,
) -> None:
    """A chapter keeps the moment it was finished, not the moment it was read."""
    store = _store(serial_engine)
    world_id, season_id = _approved_world_with_season(store)
    chapter = store.queue_chapter(season_id, world_id, "one")

    store.advance_chapter(chapter.chapter_id, ChapterState.READY)
    ready_at = store.chapters_in(season_id)[0].ready_at
    store.advance_chapter(chapter.chapter_id, ChapterState.NARRATED)
    after = store.chapters_in(season_id)[0]

    assert ready_at is not None
    assert after.ready_at == ready_at
    assert after.state is ChapterState.NARRATED


def test_the_pipeline_run_is_recorded_against_the_chapter(
    serial_engine: Engine,
) -> None:
    """Ties a chapter to the memories that run wrote."""
    store = _store(serial_engine)
    world_id, season_id = _approved_world_with_season(store)
    chapter = store.queue_chapter(season_id, world_id, "one")

    store.advance_chapter(
        chapter.chapter_id, ChapterState.GENERATING, request_id="req-42"
    )

    assert store.chapters_in(season_id)[0].request_id == "req-42"


def test_the_child_shelf_hides_what_the_review_feed_shows(
    serial_engine: Engine,
) -> None:
    """Two views, one query. The caller decides what to show, not the store."""
    store = _store(serial_engine)
    world_id, season_id = _approved_world_with_season(store)
    ready = store.queue_chapter(season_id, world_id, "one")
    pulled = store.queue_chapter(season_id, world_id, "two")
    store.queue_chapter(season_id, world_id, "three")

    store.advance_chapter(ready.chapter_id, ChapterState.READY)
    store.withdraw_chapter(pulled.chapter_id)

    assert len(store.chapters_in(season_id)) == 3
    assert [c.chapter_id for c in store.readable_chapters(season_id)] == [
        ready.chapter_id
    ]


def test_timestamps_come_back_timezone_aware(serial_engine: Engine) -> None:
    """The regression guard for a real defect the round-trip tests exposed.

    SQLite has no ``timestamptz`` and returns a naive datetime where Postgres
    returns an aware one. Unfixed, the failure would not surface here -- it
    would surface wherever something first compared two timestamps, because
    naive against aware raises ``TypeError``.
    """
    store = _store(serial_engine)
    world_id, season_id = _approved_world_with_season(store)
    chapter = store.queue_chapter(season_id, world_id, "one")
    store.advance_chapter(chapter.chapter_id, ChapterState.READY)

    world = store.get_world(world_id)
    season = store.seasons_in(world_id)[0]
    loaded = store.chapters_in(season_id)[0]

    assert world.created_at.tzinfo is not None
    assert world.approved_at is not None and world.approved_at.tzinfo is not None
    assert season.created_at.tzinfo is not None
    assert loaded.created_at.tzinfo is not None
    assert loaded.ready_at is not None and loaded.ready_at.tzinfo is not None
    # The comparison that would have raised.
    assert loaded.ready_at >= loaded.created_at
