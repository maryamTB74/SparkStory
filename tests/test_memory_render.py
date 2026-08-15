"""Turning stored memory into prompt text."""

from sparkstory.memory.render import render_memory
from sparkstory.memory.types import MemoryKind, MemoryRecord


def _semantic(text: str, subject: str) -> MemoryRecord:
    return MemoryRecord(
        child_id="maryam",
        kind=MemoryKind.SEMANTIC,
        text=text,
        subject=subject,
        source_request_id="req-1",
    )


def _episodic(text: str) -> MemoryRecord:
    return MemoryRecord(
        child_id="maryam",
        kind=MemoryKind.EPISODIC,
        text=text,
        source_request_id="req-1",
    )


def test_no_memory_renders_nothing() -> None:
    """A first book must send no memory section at all, not an empty heading."""
    assert render_memory([]) == ""


def test_a_character_fact_is_rendered_with_its_subject() -> None:
    out = render_memory([_semantic("A fox with a white-tipped tail.", "Kit")])
    assert "Kit" in out
    assert "white-tipped tail" in out


def test_episodes_are_framed_as_what_to_avoid_repeating() -> None:
    out = render_memory([_episodic("Kit built a paper rocket and reached the moon.")])
    assert "paper rocket" in out
    assert "already" in out.lower() or "again" in out.lower()


def test_the_two_tiers_are_framed_oppositely() -> None:
    """Facts are to obey, episodes are to avoid.

    Rendering both as 'here is what happened before' would invite the planner to
    reuse the plot it was shown, which is the opposite of why episodes are kept.
    """
    facts_only = render_memory([_semantic("A fox.", "Kit")])
    episodes_only = render_memory([_episodic("Kit reached the moon.")])

    assert "keep them exactly" in facts_only.lower()
    assert "do not tell" in episodes_only.lower()


def test_no_internal_vocabulary_reaches_the_model() -> None:
    """Prompt text reaches the model, so it must carry no engineering terms."""
    out = render_memory(
        [_semantic("A fox.", "Kit"), _episodic("Kit reached the moon.")]
    )
    lowered = out.lower()
    for banned in (
        "semantic",
        "episodic",
        "procedural",
        "memory_id",
        "child_id",
        "record",
        "database",
        "postgres",
        "superseded",
        "tier",
    ):
        assert banned not in lowered, f"{banned!r} leaked into prompt text"
