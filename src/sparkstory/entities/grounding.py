"""What research found before the story was planned.

Same three rules as ``entities/stories.py``: docstrings and field descriptions
here are **prompt text** (they become the JSON schema the researcher is bound to),
enums constrain the model rather than only the code, and validation constraints on
model output are load-bearing.

Two decisions in this module are worth understanding before editing it.

**Neither list has a ``min_length``.** An empty ``facts`` list is the correct
answer for most premises -- a lost teddy, a birthday, a feeling have nothing to get
factually wrong -- so it must be representable. Same shape as the review loops one
level down: there the empty list is a loop's stop signal, here it is a legitimate
result. Requiring one fact would not produce a schema error, it
would produce invented facts, which is worse because it looks like success.

**A fact is carried as a note, not as material.** ``claim`` records what was found
and is *never rendered into the planner's prompt*; only ``story_note`` is. The
planner prompt already forbids a character reciting facts, and the laziest way to
satisfy "use what research found" is to have one recite it. Splitting the two
fields means the thing the planner sees cannot be pasted into a story.

The field is ``story_note`` rather than ``story_constraint`` because it is only a
constraint in one of the two world-rule modes. Under ``IMAGINATIVE`` the same value
is a usable detail the premise may break, so the old name was wrong for half its
uses -- and since field names and descriptions are prompt text, that wrong name was
being sent to the model writing into it.
"""

from pydantic import BaseModel, Field


class GroundedFact(BaseModel):
    """Something true about the real world, and what it means for the story."""

    claim: str = Field(
        min_length=3,
        max_length=200,
        description="The fact itself, in one plain sentence a child could hear.",
    )
    story_note: str = Field(
        min_length=3,
        max_length=200,
        description=(
            "What this fact means for the story, written as something true of "
            "its world rather than a line anyone says. 'The Moon has no air' "
            "becomes 'nothing outdoors can flutter, drift or make a sound'. "
            "Never write a sentence that could appear in the finished story."
        ),
    )
    source: str = Field(
        min_length=1,
        max_length=120,
        description=(
            "Where this came from, copied exactly as it was given to you, so the "
            "fact can be checked later."
        ),
    )
    # Provenance, and the reason it exists is a step ahead of today's design: once
    # research is its own MCP tool, a client threads this object back and nothing else
    # distinguishes a fact we supplied from one a client invented. An id we can
    # look up turns that into a check.
    chunk_id: str = Field(
        min_length=1,
        max_length=120,
        description=(
            "The identifier shown in square brackets beside the text you found, "
            "copied exactly. Do not invent one."
        ),
    )


class StoryGrounding(BaseModel):
    """What is worth knowing before this story is planned."""

    # No `min_length` on either list -- see the module docstring. `max_length` is
    # deliberately tiny rather than a runaway guard: unlike the review models,
    # where the real cap is a number in the prompt, here a small hard cap is the
    # point. Four live runs show this planner treating any budget as a target --
    # offered six beats it used six, every time -- so three facts is three facts.
    facts: list[GroundedFact] = Field(
        default_factory=list,
        max_length=3,
        description=(
            "At most three facts this story must not get wrong, most important "
            "first. Return an empty list when the premise has nothing factual to "
            "get wrong, which is the usual case. Never add a fact to fill space."
        ),
    )
