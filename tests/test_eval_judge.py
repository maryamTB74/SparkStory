"""Tests for the book judge and its score types."""

from collections.abc import Callable

import pytest
from pydantic import ValidationError

from sparkstory.entities.stories import Story
from sparkstory.evals.metrics.judge import BookJudge, aggregate_page_scores
from sparkstory.evals.metrics.types import (
    BookScorecard,
    BookScores,
    CriterionScore,
    PageScore,
)
from sparkstory.models.fake_model import FakeModel


def _page(
    number: int, delight: int = 1, showing: int = 1, momentum: int = 1
) -> PageScore:
    """A PageScore with every dimension set, for tests that vary only one."""
    return PageScore(
        page_number=number,
        delight=CriterionScore(score=delight, reason="r"),
        showing=CriterionScore(score=showing, reason="r"),
        momentum=CriterionScore(score=momentum, reason="r"),
    )


def test_criterion_score_accepts_zero_and_one() -> None:
    assert CriterionScore(score=0, reason="r").score == 0
    assert CriterionScore(score=1, reason="r").score == 1


@pytest.mark.parametrize("bad", [-1, 2])
def test_criterion_score_rejects_non_binary(bad: int) -> None:
    # Binary is the whole basis of "share of pages that passed". A 2 would make a
    # dimension's mean uninterpretable rather than merely wrong.
    with pytest.raises(ValidationError):
        CriterionScore(score=bad, reason="r")


def test_judged_dimensions_match_page_score_fields() -> None:
    # The aggregation loop iterates JUDGED_DIMENSIONS, so a dimension added to
    # PageScore and not here would never be aggregated and nothing would fail.
    scored = set(PageScore.model_fields) - {"page_number"}
    assert set(BookScores.JUDGED_DIMENSIONS) == scored


def test_scorecard_has_no_overall_score() -> None:
    # Rule 18: the moment there is one number, that number becomes the target.
    for forbidden in ("overall", "total", "mean", "score"):
        assert forbidden not in BookScorecard.model_fields


def test_aggregate_averages_each_dimension_separately() -> None:
    scores = BookScores(
        pages=[
            _page(1, delight=1, showing=0, momentum=1),
            _page(2, delight=0, showing=0, momentum=1),
        ]
    )
    judged = aggregate_page_scores(scores, page_count=2)
    assert judged.delight == 0.5
    assert judged.showing == 0.0
    assert judged.momentum == 1.0


def test_aggregate_keeps_reasons_per_dimension() -> None:
    judged = aggregate_page_scores(BookScores(pages=[_page(1)]), page_count=1)
    assert judged.reasons["delight"] == ["p1: r"]


def test_aggregate_raises_on_missing_page() -> None:
    # Averaging over 1 page when the book has 2 would score a truncated answer
    # higher than a complete one.
    with pytest.raises(ValueError, match="page"):
        aggregate_page_scores(BookScores(pages=[_page(1)]), page_count=2)


def test_aggregate_raises_on_duplicate_page() -> None:
    with pytest.raises(ValueError, match="page"):
        aggregate_page_scores(BookScores(pages=[_page(1), _page(1)]), page_count=2)


async def test_judge_binds_book_scores_and_sends_the_prose(
    book_factory: Callable[..., Story],
) -> None:
    build = book_factory
    story = build(["The fox ran.", "Kit slept.", "Rain fell.", "Dawn came."])
    model = FakeModel(BookScores(pages=[_page(n) for n in range(1, 5)]))
    result = await BookJudge(model, story=story).ainvoke()
    assert model.bound_schema is BookScores
    assert "The fox ran." in str(model.messages)
    assert len(result.pages) == 4


async def test_judge_does_not_send_the_plan(book_factory: Callable[..., Story]) -> None:
    build = book_factory
    # A page is judged on what a child would hear. Sending the plan would invite
    # the judge to score fidelity to notes, which the prose critic already does.
    story = build(["The fox ran.", "Kit slept.", "Rain fell.", "Dawn came."])
    model = FakeModel(BookScores(pages=[_page(n) for n in range(1, 5)]))
    await BookJudge(model, story=story).ainvoke()
    sent = str(model.messages)
    assert "Kit looks up, eyes wide" not in sent  # a visual_action from the plan
    assert "curiosity" not in sent  # an emotional_shift from the plan


async def test_judge_prompt_leaks_no_internal_terms(
    book_factory: Callable[..., Story],
) -> None:
    build = book_factory
    # Rule 1: docstrings and Field descriptions are prompt text, and an earlier
    # version of this project shipped "the Canon Agent" to a model as its task.
    story = build(["The fox ran.", "Kit slept.", "Rain fell.", "Dawn came."])
    model = FakeModel(BookScores(pages=[_page(n) for n in range(1, 5)]))
    await BookJudge(model, story=story).ainvoke()
    sent = str(model.messages).lower()
    for term in ("rubric", "regression", "harness", "metric", "scorecard", "eval"):
        assert term not in sent
