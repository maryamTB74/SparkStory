"""Worlds, seasons and chapters: what makes a book into a series.

The engine writes one book. A child's app needs a *show* -- recurring characters
in a world that persists, so that chapter nine is constrained by chapter one.
This module is the shape of that, and nothing more: no generation, no storage,
no policy about when a chapter may run.

**The parent approves the world, not every chapter.** The existing gate sits
between ``plan_story`` and ``write_story``, and it is kept -- but it is moved.
A child waiting on an adult to approve each chapter has no loop, so approval
binds once, to the space of stories: premise, tone, world rules, and the
``avoid`` list. Chapters then generate freely inside that space.

That narrows what approval means, and it is worth saying plainly rather than
discovering later: the parent is approving a *space of stories*, not each story.
The per-chapter safety rubric and the fail-closed ``avoid`` gate are untouched.
Approval was never what made a chapter safe.

**Nothing is deleted.** A withdrawn chapter keeps its row and gains a state, in
the same spirit as ``MemoryRecord.superseded_by``: a child may have already read
it, and a shelf that silently loses a book is worse than one that admits it.
"""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated
from uuid import uuid4

from pydantic import BaseModel, Field, StringConstraints

from sparkstory.entities.stories import (
    ChildProfile,
    ReadingLevel,
    StoryBrief,
    Tone,
    WorldRules,
)
from sparkstory.memory.types import ChildId

#: Server-minted identifiers. Unlike ``ChildId`` these are never supplied by a
#: caller, so the constraint is a shape assertion rather than a guard.
SerialId = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=64)
]


def _new_id() -> str:
    return str(uuid4())


def _now() -> datetime:
    return datetime.now(UTC)


class WorldStatus(StrEnum):
    """Where a world stands with the parent."""

    #: Described but not yet approved. No chapter may be queued.
    DRAFT = "draft"
    #: Approved. Chapters generate inside the approved constraints.
    APPROVED = "approved"
    #: No new chapters. Everything already written stays readable, because a
    #: child's finished books are not the parent's to un-write.
    RETIRED = "retired"


class ChapterState(StrEnum):
    """Where one chapter has got to.

    Six states rather than a flag and a payload, for the reason ``JobState``
    gives: a boolean cannot tell *running* from *failed*, and that collapse is
    what hides a broken pipeline behind an empty shelf.
    """

    QUEUED = "queued"
    GENERATING = "generating"
    #: Written and illustrated; nobody has read it aloud yet.
    READY = "ready"
    #: The child has recorded their narration. This is the completed state --
    #: a chapter nobody narrated is not finished, whatever the pipeline did.
    NARRATED = "narrated"
    FAILED = "failed"
    #: Pulled by the parent from the review feed. Kept, not deleted.
    WITHDRAWN = "withdrawn"


#: States from which nothing further happens.
TERMINAL_STATES = frozenset({ChapterState.NARRATED, ChapterState.WITHDRAWN})


class World(BaseModel):
    """One child's show: the constraints every chapter in it obeys."""

    child_id: ChildId
    title: str = Field(min_length=1, max_length=80)
    premise: str = Field(
        min_length=3,
        max_length=500,
        description="What this show is about, in the parent's own words.",
    )
    tone: Tone = Tone.GENTLE
    world_rules: WorldRules = WorldRules.IMAGINATIVE
    #: Copied onto every chapter brief. The hard constraint, unchanged.
    avoid: list[str] = Field(default_factory=list, max_length=20)
    #: Threaded into every chapter so a running joke can survive the season.
    must_include: list[str] = Field(default_factory=list, max_length=10)
    status: WorldStatus = WorldStatus.DRAFT
    world_id: SerialId = Field(default_factory=_new_id)
    created_at: datetime = Field(default_factory=_now)
    approved_at: datetime | None = None

    @property
    def accepts_chapters(self) -> bool:
        return self.status is WorldStatus.APPROVED

    def approved(self) -> World:
        """A copy in the approved state. Never mutates in place."""
        return self.model_copy(
            update={"status": WorldStatus.APPROVED, "approved_at": _now()}
        )

    def brief_for(self, child: ChildProfile, seed: str) -> StoryBrief:
        """Build the brief for one chapter of this world.

        This is the seam between the serial model and the engine: everything the
        parent approved is applied here, and a caller cannot route around it by
        assembling its own ``StoryBrief`` -- because nothing else knows the
        world's ``avoid`` list.

        Args:
            child: Whose show this is. Supplies name, pronouns, reading level.
            seed: The one thing the *child* chose about this chapter.
        """
        return StoryBrief(
            child=child,
            premise=f"{self.premise} In this chapter: {seed}",
            tone=self.tone,
            world_rules=self.world_rules,
            page_count=pages_for(child.reading_level),
            must_include=list(self.must_include),
            avoid=list(self.avoid),
        )


class Season(BaseModel):
    """Ten chapters that become one book."""

    world_id: SerialId
    #: 1-based, per world. Season two is a second book on the same shelf.
    ordinal: int = Field(ge=1)
    title: str = Field(min_length=1, max_length=80)
    #: How many chapters before the season is collected into a book. Ten is a
    #: guess about attention, not a technical limit.
    chapter_target: int = Field(default=10, ge=1, le=30)
    season_id: SerialId = Field(default_factory=_new_id)
    created_at: datetime = Field(default_factory=_now)
    completed_at: datetime | None = None


class Chapter(BaseModel):
    """One episode: a ``Story`` plus where it sits and how far it has got."""

    world_id: SerialId
    season_id: SerialId
    #: 1-based, per season, contiguous. The store assigns it.
    ordinal: int = Field(ge=1)
    #: The one thing the child chose, in their words or from an offered seed.
    seed: str = Field(min_length=1, max_length=300)
    state: ChapterState = ChapterState.QUEUED
    #: Ties this chapter to a pipeline run, and so to the memories it wrote.
    request_id: str | None = None
    chapter_id: SerialId = Field(default_factory=_new_id)
    created_at: datetime = Field(default_factory=_now)
    ready_at: datetime | None = None

    @property
    def is_finished(self) -> bool:
        return self.state in TERMINAL_STATES


def pages_for(reading_level: ReadingLevel) -> int:
    """How long a chapter is, for a reader at this level.

    Chapter length is the app's decision, not the engine's: ``StoryBrief``
    already accepts 4--24 pages, so this is configuration rather than a schema
    change. The numbers come from how long one sitting lasts at each level --
    roughly three, five and eight minutes read aloud.

    ``PRE_READER`` is below the app's 5--8 band and is never offered, but the
    mapping is total anyway. A partial function here would be one more way for a
    caller to be wrong, and the correct behaviour for a younger child is a
    shorter book, not an exception.
    """
    return {
        ReadingLevel.PRE_READER: 6,
        ReadingLevel.EARLY_READER: 8,
        ReadingLevel.DEVELOPING: 10,
        ReadingLevel.CONFIDENT: 12,
    }[reading_level]
