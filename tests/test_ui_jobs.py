"""The job registry: state, transitions, and the guard against double-approve."""

import pytest

from sparkstory.entities.stories import ChildProfile, StoryBrief
from sparkstory.mcp.ui.jobs import JobRegistry, JobState


@pytest.fixture
def brief() -> StoryBrief:
    return StoryBrief(
        child=ChildProfile(name="Maryam", age=5),
        premise="a fox who wants to visit the moon",
    )


def test_create_returns_a_planning_job_with_a_unique_id(brief: StoryBrief) -> None:
    registry = JobRegistry()
    first = registry.create(brief)
    second = registry.create(brief)

    assert first.state is JobState.PLANNING
    assert first.id != second.id
    assert first.brief == brief


def test_create_records_the_original_premise(brief: StoryBrief) -> None:
    # Revise amends the premise; the form and the book still show what was asked.
    registry = JobRegistry()
    job = registry.create(brief)

    assert job.original_premise == "a fox who wants to visit the moon"


def test_get_returns_none_for_an_unknown_id(brief: StoryBrief) -> None:
    registry = JobRegistry()

    assert registry.get("not-a-job") is None


def test_update_replaces_fields_and_leaves_others_alone(brief: StoryBrief) -> None:
    registry = JobRegistry()
    job = registry.create(brief)

    updated = registry.update(job.id, detail="critiquing the outline")

    assert updated.detail == "critiquing the outline"
    assert updated.state is JobState.PLANNING
    assert updated.brief == brief


def test_the_job_record_is_frozen(brief: StoryBrief) -> None:
    # Mutation must go through the registry, or two writers race silently.
    registry = JobRegistry()
    job = registry.create(brief)

    with pytest.raises(AttributeError):
        job.state = JobState.COMPLETE  # type: ignore[misc]


def test_transition_succeeds_from_the_expected_state(brief: StoryBrief) -> None:
    registry = JobRegistry()
    job = registry.create(brief)

    moved = registry.transition(job.id, JobState.PLANNING, JobState.WRITING)

    assert moved is not None
    assert moved.state is JobState.WRITING


def test_transition_returns_none_from_any_other_state(brief: StoryBrief) -> None:
    # This is the double-approve guard. Two POSTs to /approve must not both
    # start write_story against the same run directory.
    registry = JobRegistry()
    job = registry.create(brief)
    registry.transition(job.id, JobState.PLANNING, JobState.WRITING)

    again = registry.transition(job.id, JobState.PLANNING, JobState.WRITING)

    assert again is None
    assert registry.get(job.id).state is JobState.WRITING


def test_transition_returns_none_for_an_unknown_id(brief: StoryBrief) -> None:
    registry = JobRegistry()

    assert registry.transition("nope", JobState.PLANNING, JobState.WRITING) is None


def test_transition_applies_changes_atomically(brief: StoryBrief) -> None:
    registry = JobRegistry()
    job = registry.create(brief)

    moved = registry.transition(
        job.id, JobState.PLANNING, JobState.FAILED, error="no API key"
    )

    assert moved.state is JobState.FAILED
    assert moved.error == "no API key"


def test_a_job_id_is_not_derived_from_the_premise(brief: StoryBrief) -> None:
    # A guessable id is a way to read someone else's book.
    registry = JobRegistry()
    job = registry.create(brief)

    assert "fox" not in job.id
    assert "moon" not in job.id
