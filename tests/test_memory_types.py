"""Types for long-term memory.

The `child_id` tests are the load-bearing ones here. The caller is an LLM agent,
so the value scoping every read must be rejected by the *type* rather than by a
store remembering to sanitise -- see `memory/types.py` for the argument.
"""

import pytest
from pydantic import BaseModel, ValidationError

from sparkstory.memory.types import ChildId, MemoryKind, MemoryRecord


class _Holder(BaseModel):
    """A model whose only job is to apply the ChildId constraints."""

    child_id: ChildId


@pytest.mark.parametrize(
    "bad",
    [
        "../../etc/passwd",  # path traversal
        "Kit Smith",  # space
        "UPPER",  # case
        "",  # empty
        "x" * 65,  # too long
        "child--1",  # doubled separator
        "-leading",
        "trailing-",
    ],
)
def test_child_id_rejects_unsafe_values(bad: str) -> None:
    """The caller is an LLM agent, so the type is the guard -- not the store."""
    with pytest.raises(ValidationError):
        _Holder(child_id=bad)


@pytest.mark.parametrize("good", ["maryam", "maryam-5", "a", "child-2026-08"])
def test_child_id_accepts_slugs(good: str) -> None:
    assert _Holder(child_id=good).child_id == good


def test_procedural_is_declared_but_unused() -> None:
    """Dropped deliberately: adding it later is a row, not a migration."""
    assert MemoryKind.PROCEDURAL.value == "procedural"


def test_record_defaults_are_generated() -> None:
    a = MemoryRecord(
        child_id="maryam",
        kind=MemoryKind.SEMANTIC,
        text="Kit is a fox with a white-tipped tail.",
        subject="Kit",
        source_request_id="req-1",
    )
    b = MemoryRecord(
        child_id="maryam",
        kind=MemoryKind.SEMANTIC,
        text="Kit is a fox with a white-tipped tail.",
        subject="Kit",
        source_request_id="req-1",
    )
    assert a.memory_id != b.memory_id, "each record needs its own id"
    assert a.created_at.tzinfo is not None, "timestamps must be tz-aware"
