"""Tests for assembling a book's scorecard."""

from collections.abc import Callable
from typing import Any

from sparkstory.entities.stories import Story
from sparkstory.evals.metrics.judge import BookJudge
from sparkstory.evals.metrics.types import BookScores, CriterionScore, PageScore
from sparkstory.evals.scorecard import score_book
from sparkstory.models.fake_model import FakeModel

FOUR_PAGES = ["The fox ran.", "The bird sang.", "The moon rose.", "Kit slept."]


def _page(number: int) -> PageScore:
    return PageScore(
        page_number=number,
        delight=CriterionScore(score=1, reason="r"),
        showing=CriterionScore(score=1, reason="r"),
        momentum=CriterionScore(score=1, reason="r"),
    )


async def test_score_book_without_a_judge_reports_computed_only(
    book_factory: Callable[..., Story],
) -> None:
    card = await score_book(book_factory(FOUR_PAGES), name="b", notes=[])
    assert card.deterministic.distinct_opener_ratio == 0.5
    assert card.judged is None
    assert card.judge_error is None


async def test_score_book_with_a_judge_reports_both_halves(
    book_factory: Callable[..., Story],
) -> None:
    story = book_factory(FOUR_PAGES)
    judge = BookJudge(
        FakeModel(BookScores(pages=[_page(n) for n in range(1, 5)])), story=story
    )
    card = await score_book(story, name="b", notes=[], judge=judge)
    assert card.deterministic.distinct_opener_ratio == 0.5
    assert card.judged is not None
    assert card.judged.delight == 1.0


async def test_judge_failure_keeps_the_computed_numbers(
    book_factory: Callable[..., Story],
) -> None:
    # The whole point of two separate fields: one failed model call must not
    # destroy the free half of a measurement that is already paid for.
    class Failing:
        def with_structured_output(self, schema: Any, **_: Any) -> Failing:
            return self

        async def ainvoke(self, messages: Any, **_: Any) -> None:
            raise RuntimeError("provider exploded")

    story = book_factory(FOUR_PAGES)
    card = await score_book(
        story, name="b", notes=[], judge=BookJudge(Failing(), story=story)
    )
    assert card.deterministic.words_per_page == 2.75
    assert card.judged is None
    assert "provider exploded" in card.judge_error


async def test_malformed_judge_response_is_recorded_not_raised(
    book_factory: Callable[..., Story],
) -> None:
    # A judge that scored 1 page of 4 would otherwise average over the wrong
    # denominator and report a higher number for a worse answer.
    story = book_factory(FOUR_PAGES)
    judge = BookJudge(FakeModel(BookScores(pages=[_page(1)])), story=story)
    card = await score_book(story, name="b", notes=[], judge=judge)
    assert card.judged is None
    assert "expected each of 1..4" in card.judge_error


async def test_notes_produce_recital_numbers(
    book_factory: Callable[..., Story],
) -> None:
    story = book_factory(
        ["His wings found empty space.", "b", "c", "d"],
        beat_summaries=["Wings need air to push against, he learns."] * 4,
    )
    card = await score_book(story, name="b", notes=["wings need air to push against"])
    assert card.deterministic.fact_recital_beats == 6


def test_fixture_capture_collects_story_notes_from_the_research_task() -> None:
    """The callback `--fixtures` uses to recover grounding the pipeline drops.

    Tested because the first baseline run reported "-" for both recital columns on
    all five books: `generate_and_score` passed `notes=[]`, so the metric that
    found the recital defect could not fire at all. A column that is always empty
    is not a measurement.
    """
    from sparkstory.entities.grounding import GroundedFact, StoryGrounding

    notes: list[str] = []

    def capture(task_name: str, value: object) -> None:
        if task_name != "research":
            return
        notes.extend(fact.story_note for fact in getattr(value, "facts", None) or [])

    grounding = StoryGrounding(
        facts=[
            GroundedFact(
                claim="The Moon has no air.",
                story_note="nothing outdoors can flutter, drift or make a sound",
                source="moon#1",
                chunk_id="moon#1",
            )
        ],
        devices=[],
    )
    capture("plan_outline", grounding)  # ignored: wrong task
    assert notes == []
    capture("research", grounding)
    assert notes == ["nothing outdoors can flutter, drift or make a sound"]
