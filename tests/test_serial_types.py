"""Worlds, seasons and chapters as values: approval, and chapter shape."""

import pytest
from pydantic import ValidationError

from sparkstory.entities.serial import (
    Chapter,
    ChapterState,
    World,
    WorldStatus,
    pages_for,
)
from sparkstory.entities.stories import ChildProfile, ReadingLevel, Tone, WorldRules


def _world(**overrides: object) -> World:
    base: dict[str, object] = {
        "child_id": "maryam",
        "title": "Kit and the Moon",
        "premise": "a fox who wants to visit the Moon",
        "avoid": ["spiders"],
        "must_include": ["a paper rocket"],
        "tone": Tone.MAGICAL,
        "world_rules": WorldRules.IMAGINATIVE,
    }
    base.update(overrides)
    return World(**base)  # type: ignore[arg-type]


def test_a_new_world_is_draft_and_takes_no_chapters() -> None:
    """Approval is opt-in. A world nobody looked at is not a space of stories."""
    world = _world()

    assert world.status is WorldStatus.DRAFT
    assert world.accepts_chapters is False
    assert world.approved_at is None


def test_approving_stamps_when_and_does_not_mutate() -> None:
    world = _world()
    approved = world.approved()

    assert approved.accepts_chapters is True
    assert approved.approved_at is not None
    # The original is untouched: approval returns a new value, so a caller
    # holding the draft cannot be surprised by it changing underneath them.
    assert world.status is WorldStatus.DRAFT
    assert world.world_id == approved.world_id


def test_the_brief_carries_everything_the_parent_approved() -> None:
    """The seam that makes approval mean something.

    Nothing but the world knows its ``avoid`` list, so a caller assembling its
    own ``StoryBrief`` cannot route around what the parent agreed to.
    """
    world = _world().approved()
    child = ChildProfile(
        name="Ada",
        age=6,
        reading_level=ReadingLevel.DEVELOPING,
        child_id="maryam",
    )

    brief = world.brief_for(child, "Kit builds a paper rocket")

    assert brief.avoid == ["spiders"]
    assert brief.must_include == ["a paper rocket"]
    assert brief.tone is Tone.MAGICAL
    assert brief.world_rules is WorldRules.IMAGINATIVE
    assert world.premise in brief.premise
    assert "Kit builds a paper rocket" in brief.premise


def test_the_brief_copies_the_lists_rather_than_sharing_them() -> None:
    """Two chapters must not be able to edit each other's constraints."""
    world = _world().approved()
    child = ChildProfile(name="Ada", age=6, child_id="maryam")

    brief = world.brief_for(child, "a first chapter")
    brief.avoid.append("thunder")

    assert world.avoid == ["spiders"]


@pytest.mark.parametrize(
    ("level", "pages"),
    [
        (ReadingLevel.PRE_READER, 6),
        (ReadingLevel.EARLY_READER, 8),
        (ReadingLevel.DEVELOPING, 10),
        (ReadingLevel.CONFIDENT, 12),
    ],
)
def test_chapter_length_follows_reading_level(level: ReadingLevel, pages: int) -> None:
    assert pages_for(level) == pages


def test_every_reading_level_has_a_length() -> None:
    """Total by construction. A partial mapping is one more way to be wrong."""
    for level in ReadingLevel:
        assert pages_for(level) >= 4


def test_chapter_length_never_leaves_what_the_brief_accepts() -> None:
    """``StoryBrief.page_count`` is bounded 4--24; every level must land inside."""
    for level in ReadingLevel:
        assert 4 <= pages_for(level) <= 24


def test_a_chapter_is_unfinished_until_it_is_narrated_or_withdrawn() -> None:
    """A chapter the pipeline finished but nobody read is not done."""
    chapter = Chapter(world_id="w", season_id="s", ordinal=1, seed="Kit finds a hat")

    assert chapter.state is ChapterState.QUEUED
    assert chapter.is_finished is False

    assert chapter.model_copy(update={"state": ChapterState.READY}).is_finished is False
    assert (
        chapter.model_copy(update={"state": ChapterState.NARRATED}).is_finished is True
    )
    assert (
        chapter.model_copy(update={"state": ChapterState.WITHDRAWN}).is_finished is True
    )


def test_ordinals_start_at_one() -> None:
    with pytest.raises(ValidationError):
        Chapter(world_id="w", season_id="s", ordinal=0, seed="nope")


def test_a_child_id_that_could_be_spelled_two_ways_is_refused() -> None:
    """Inherited from ``ChildId``, and asserted here because it is the scope key.

    Two spellings of one child would split a shelf in half.
    """
    with pytest.raises(ValidationError):
        _world(child_id="Maryam--B")
