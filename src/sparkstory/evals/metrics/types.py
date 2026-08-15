"""Score types for a book's scorecard.

Two rules are encoded structurally rather than left to convention.

**Binary per page, averaged per dimension.** ``CriterionScore.score`` is an int
bounded to 0-1, so a dimension's book-level value is the *share of pages that
passed* and cannot quietly become anything else.

**Computed and judged values never mix.** They live in two models on
``BookScorecard``, with no field summing across them and no overall score. A
single blended number would rank a book-wide defect equal to a local one -- the
mistake ``draft_score`` already makes, where a book shipped carrying a whole-book
finding because one is fewer than three -- and whatever gets averaged becomes the
thing a loop optimises.
"""

from typing import Annotated, ClassVar

from annotated_types import Ge, Le
from pydantic import BaseModel, Field


class CriterionScore(BaseModel):
    """One dimension's verdict on one page."""

    score: Annotated[int, Ge(0), Le(1)] = Field(
        description="1 if this page satisfies the criterion, 0 if it does not.",
    )
    reason: str = Field(
        min_length=1,
        description="Why this page earned that score, in one sentence.",
    )


class PageScore(BaseModel):
    """Every judged dimension, for one page."""

    page_number: int = Field(
        ge=1,
        description="The page this scores, matching the page number in the book.",
    )
    delight: CriterionScore = Field(
        description="Would a child ask for this page to be read again?",
    )
    showing: CriterionScore = Field(
        description=(
            "Is feeling conveyed through action, image or speech rather than "
            "named outright?"
        ),
    )
    momentum: CriterionScore = Field(
        description="Does this page make a listener want to turn it?",
    )


class BookScores(BaseModel):
    """Scores for every page of one book."""

    # A ClassVar, so it is not a schema field and never reaches the model as an
    # instruction. The aggregation loop reads this, so a dimension added to
    # `PageScore` and not here would be silently unaggregated -- there is a test
    # asserting the two agree.
    #: Judged dimension names, in report order.
    JUDGED_DIMENSIONS: ClassVar[tuple[str, ...]] = ("delight", "showing", "momentum")

    pages: list[PageScore] = Field(
        min_length=1,
        description="One entry per page of the book, in page order.",
    )


# Not bound as any model's output schema, so these docstrings are documentation
# rather than prompt text.
class DeterministicScores(BaseModel):
    """What can be counted about a book without a model.

    Every value is raw. No thresholds and no pass/fail: the right threshold is
    unknowable before the distribution is, and a number invented now would be
    obeyed later as though it had been measured.
    """

    distinct_opener_ratio: float
    question_ending_ratio: float
    words_per_page: float
    # Kept alongside words_per_page, which cannot see what this sees: two books can
    # spend the same page budget on few long sentences or many short ones, and only
    # the second reads as a list of declaratives. It earned its place by tracking a
    # human's judgement of delight better than the LLM judge did.
    words_per_sentence: float
    beats_per_page: float
    # Absolute word counts, not runs normalised by note length. Measured against
    # two real runs, normalising scores a genuine recital (6 shared words out of a
    # 19-word note) at 0.30 and an innocent 3-word overlap from a 12-word note at
    # 0.25 -- so it compresses the one distinction the metric exists to draw, and
    # would have ranked the clean book worse than the defective one.
    #
    # `None` rather than 0 when the run carried no grounding notes: nothing was
    # recited because there was nothing to recite, which is the absence of a
    # measurement rather than a good score.
    fact_recital_beats: int | None = None
    fact_recital_prose: int | None = None


class JudgedScores(BaseModel):
    """The share of pages passing, per judged dimension, with the reasons.

    ``reasons`` is kept although only the scores are averaged. It is what makes a
    dropped dimension readable rather than merely visible, and it is how a judge
    that agrees rather than judges would be noticed -- three 1.0s with generic
    reasoning look identical to three 1.0s that were earned.
    """

    delight: float
    showing: float
    momentum: float
    reasons: dict[str, list[str]] = Field(default_factory=dict)


class BookScorecard(BaseModel):
    """Everything measured about one book.

    Deliberately has no overall score. Eight numbers, no ninth summarising them.
    """

    name: str
    page_count: int
    deterministic: DeterministicScores
    # None when the judge failed or was skipped, and the computed numbers survive
    # either way: losing a whole measurement to one failed model call is what the
    # per-sample error fallback exists to avoid.
    judged: JudgedScores | None = None
    #: Why `judged` is absent, so a null is never silent.
    judge_error: str | None = None
    # The raw verdicts, kept alongside their average rather than instead of it.
    # Measuring a judge against a human label is a per-page comparison, and two
    # book-level means can agree while every page disagrees -- an error on one page
    # cancelling an opposite error on another. Averaging first destroyed exactly the
    # data the alignment score needs, which is why the 2026-08-04 baseline can only
    # ever be compared average-to-average.
    #
    # `None` whenever `judged` is None, so the two halves of a judge's answer are
    # present or absent together.
    judged_pages: BookScores | None = None
