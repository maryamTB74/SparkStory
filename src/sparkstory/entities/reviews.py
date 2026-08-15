"""What a critic says about a draft.

**Two models per artifact, and the split matters.** The ``*Output`` model is what
the critic is bound to produce. The unsuffixed model is what the node returns and
a generator consumes, and it carries the draft the reviews describe -- so a review
list can never be paired with the wrong draft. Binding the fat one instead would
pay the critic to echo the whole outline back on every pass.

Docstrings and field descriptions on the ``*Output`` models are **prompt text**:
they are sent to the critic as JSON schema. Engineering rationale goes in ``#``
comments, which never reach the model.
"""

from enum import StrEnum

from pydantic import BaseModel, Field

from sparkstory.entities.stories import StoryOutline, StoryProse


# Two rubrics, and both come from a failure actually
# read in outputs/20260730-124002-*: the resolution arrived by coincidence
# ("Maryam gathered fallen stars from the grass"), and the want belonged to the
# fox while the child only helped.
#
# `theme` (does not moralise) and `brief_adherence` (must_include / avoid / tone)
# were considered and left out. The reviewed run had no `avoid` violations and
# nobody has caught it moralising, so both are hypothesised defects -- and a
# rubric for a defect that has never been observed is the same mistake as config
# for a feature that does not exist. Add a third when a real book fails in a
# third way.
class OutlineRubric(StrEnum):
    """What a story plan is judged against."""

    PROTAGONIST = "protagonist"
    EARNED_RESOLUTION = "earned_resolution"


class OutlineReview(BaseModel):
    """One problem found in the story plan."""

    rubric: OutlineRubric = Field(
        description="Which requirement this plan fails.",
    )
    beat_position: int | None = Field(
        default=None,
        ge=1,
        description=(
            "The position of the beat this is about. Leave empty when it is "
            "about the story as a whole."
        ),
    )
    comment: str = Field(
        min_length=10,
        max_length=600,
        description=(
            "What is wrong and why it matters, addressed to the person who will "
            "fix it. Say why, not only what. Do not rewrite the plan yourself."
        ),
    )


class OutlineReviewsOutput(BaseModel):
    """Every problem found in the story plan."""

    # No `min_length`: an empty list is the signal that the plan is good and the
    # revision loop should stop, so it has to be reachable.
    #
    # `max_length` is a runaway guard, not the cap. The real limit is a number in
    # the critic's prompt, so the model drops its own least important findings
    # rather than having whichever ones came first kept by truncation.
    reviews: list[OutlineReview] = Field(
        default_factory=list,
        max_length=20,
        description=(
            "Every problem worth fixing, most important first. Return nothing at "
            "all if the plan meets every requirement."
        ),
    )


# Assembled in code, never returned by a model, so -- unlike everything above --
# nothing here is prompt text and no field needs a description.
class OutlineReviews(BaseModel):
    """A story plan and what is wrong with it."""

    outline: StoryOutline
    reviews: list[OutlineReview]


# Four rubrics from failures read in a real run -- the prose copied the plan a
# page late, six of eight pages opened with a character's name, the interiority
# the plan called for never arrived, and nobody has yet judged reading level.
#
# `safety` is the one exception to growing rubrics from evidence, and it is
# deliberate. It is not a quality rubric but a guardrail: the cost of missing an
# `avoid` item once, in a book written for a named five-year-old, is
# categorically different from the cost of a flat sentence. `avoid` is currently
# unenforced anywhere else in the system, and a kids' product wants its
# guardrails fail-closed.
class ProseRubric(StrEnum):
    """What the finished words are judged against."""

    PLAN_FIDELITY = "plan_fidelity"
    READ_ALOUD = "read_aloud"
    INTERIORITY = "interiority"
    READING_LEVEL = "reading_level"
    SAFETY = "safety"


class ProseReview(BaseModel):
    """One problem found in the words of the book."""

    rubric: ProseRubric = Field(
        description="Which requirement these words fail.",
    )
    page_number: int | None = Field(
        default=None,
        ge=1,
        description=(
            "The page this is about. Leave empty when it is about the book as a "
            "whole, such as every page beginning the same way."
        ),
    )
    comment: str = Field(
        min_length=10,
        max_length=600,
        description=(
            "What is wrong and why it matters to a child hearing this read "
            "aloud. Say why, not only what. Do not rewrite the page yourself."
        ),
    )


class ProseReviewsOutput(BaseModel):
    """Every problem found in the words of the book."""

    # Same two constraints as OutlineReviewsOutput, for the same two reasons: no
    # `min_length`, because empty is the stop signal; `max_length` as a runaway
    # guard only, with the real cap stated as a number in the critic's prompt so
    # the model drops its own least important findings rather than having
    # truncation keep whichever came first.
    reviews: list[ProseReview] = Field(
        default_factory=list,
        max_length=20,
        description=(
            "Every problem worth fixing, most important first. Return nothing at "
            "all if the words meet every requirement."
        ),
    )


# Assembled in code, never returned by a model, so nothing here is prompt text.
class ProseReviews(BaseModel):
    """The words of a book and what is wrong with them."""

    prose: StoryProse
    reviews: list[ProseReview]
