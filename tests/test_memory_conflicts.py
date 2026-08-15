"""Detecting a fact that disagrees with a stored one."""

from sparkstory.memory.conflicts import find_conflicts
from sparkstory.memory.types import MemoryKind, MemoryRecord


def _record(text: str, subject: str = "Kit") -> MemoryRecord:
    return MemoryRecord(
        child_id="maryam",
        kind=MemoryKind.SEMANTIC,
        text=text,
        subject=subject,
        source_request_id="req-1",
    )


def test_a_genuine_contradiction_is_flagged() -> None:
    conflicts = find_conflicts(
        new=[_record("Kit has a bushy red tail.")],
        stored=[_record("Kit has a white-tipped tail.")],
    )
    assert len(conflicts) == 1
    assert conflicts[0].subject == "Kit"
    assert conflicts[0].stored_text == "Kit has a white-tipped tail."
    assert conflicts[0].new_text == "Kit has a bushy red tail."


def test_an_identical_restatement_is_not_a_conflict() -> None:
    """A conflict test where nothing can conflict proves nothing, so the negative
    direction is asserted too."""
    conflicts = find_conflicts(
        new=[_record("Kit has a white-tipped tail.")],
        stored=[_record("Kit has a white-tipped tail.")],
    )
    assert conflicts == []


def test_restatement_ignores_case_and_surrounding_whitespace() -> None:
    conflicts = find_conflicts(
        new=[_record("  kit has a white-tipped tail.  ")],
        stored=[_record("Kit has a white-tipped tail.")],
    )
    assert conflicts == []


def test_different_subjects_never_conflict() -> None:
    conflicts = find_conflicts(
        new=[_record("Ted is a bear.", subject="Ted")],
        stored=[_record("Kit is a fox.", subject="Kit")],
    )
    assert conflicts == []


def test_a_new_subject_is_not_a_conflict() -> None:
    """The first book about a character must not report a conflict with nothing."""
    assert find_conflicts(new=[_record("Kit is a fox.")], stored=[]) == []


def test_episodic_records_never_conflict() -> None:
    """Two books about the moon is a repetition to avoid, not a contradiction."""
    episode = MemoryRecord(
        child_id="maryam",
        kind=MemoryKind.EPISODIC,
        text="Story 2: Kit reached the moon.",
        source_request_id="req-2",
    )
    stored = MemoryRecord(
        child_id="maryam",
        kind=MemoryKind.EPISODIC,
        text="Story 1: Kit reached the moon.",
        source_request_id="req-1",
    )
    assert find_conflicts(new=[episode], stored=[stored]) == []


def test_every_disagreeing_stored_fact_is_reported() -> None:
    """Two stored descriptions and one new one is two conflicts, not one.

    Append-only means a subject accumulates rows, so by the third book there may
    be several stored facts to disagree with. Reporting only the first would hide
    the history the parent needs to choose between.
    """
    conflicts = find_conflicts(
        new=[_record("Kit has a bushy red tail.")],
        stored=[
            _record("Kit has a white-tipped tail."),
            _record("Kit has a short grey tail."),
        ],
    )
    assert len(conflicts) == 2
