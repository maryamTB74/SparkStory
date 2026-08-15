"""What a memory is, and the key it is scoped by.

**Two tiers are built, not three.** ``PROCEDURAL`` is declared so adding it later
is a row rather than a migration, and nothing writes or reads it. It was dropped
because it would derive guidance from eval scorecards, and judging the same books
twice at temperature 0 moved a dimension by up to 0.25 -- a noise floor larger
than the effects such a note would claim to detect.
"""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated
from uuid import uuid4

from pydantic import BaseModel, Field, StringConstraints


class MemoryKind(StrEnum):
    """Which tier a record belongs to."""

    #: A durable fact about a character or the child's world. Fetched exactly.
    SEMANTIC = "semantic"
    #: One summary of one finished book. Searched by similarity.
    EPISODIC = "episodic"
    #: Declared, never written. See the module docstring.
    PROCEDURAL = "procedural"


# The scope key, validated by the type rather than by the store.
#
# The caller is an LLM agent, so "the store remembers to sanitise" is not a guard.
# Nothing here builds a filesystem path -- it is a SQL parameter -- but the value
# scopes every read, and a scope key that can be spoofed leaks one child's memory
# into another's book.
#
# The pattern forbids leading, trailing and doubled separators so that two
# spellings cannot denote one child.
ChildId = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=64,
        pattern=r"^[a-z0-9]+(-[a-z0-9]+)*$",
    ),
]


class MemoryRecord(BaseModel):
    """One remembered thing.

    Append-only: no field is ever updated in place. A superseded record keeps its
    row and gains a pointer, so the books on the shelf stay true to what was
    written when they were made.
    """

    child_id: ChildId
    kind: MemoryKind
    text: str = Field(min_length=1, max_length=500)
    #: The grouping key for exact fetch, e.g. "Kit". None for episodic records,
    #: which are about a whole book rather than a subject.
    subject: str | None = Field(default=None, max_length=80)
    #: Which run wrote this, tying a memory back to the book it came from.
    source_request_id: str
    memory_id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    #: Set when a later record supersedes this one. Never a delete.
    superseded_by: str | None = None


class MemoryConflict(BaseModel):
    """A new fact that disagrees with a stored one.

    Surfaced in the outline preview so a parent decides. Not an error: two
    descriptions of one fox is a question for a human, not a failed run.
    """

    subject: str
    stored_text: str
    new_text: str
