"""The outline pipeline: plan, critique, revise, keep the best.

Moved out of ``test_workflow.py`` when planning became ``plan_story``'s job
rather than a stage inside ``write_story``. Faked by intercepting
``get_chat_model`` in the outline workflow module -- a *different* patch target
from the story workflow's, which is exactly why the split needed its own file.
A patch aimed at the wrong module does not fail; it silently reaches a real
model factory, and the test fails for a reason that looks nothing like the cause.
"""

from collections.abc import Callable
from typing import Any

import pytest

from sparkstory.entities.reviews import (
    OutlineReview,
    OutlineReviewsOutput,
    OutlineRubric,
)
from sparkstory.entities.stories import StoryBrief, StoryOutline
from sparkstory.mcp.tools.plan_story import plan_story_tool
from sparkstory.models.fake_model import FakeModel
from sparkstory.workflows.plan_outline import run_outline_pipeline

OUTLINE_FACTORY = "sparkstory.workflows.plan_outline.get_chat_model"


@pytest.fixture
def outline_fakes(
    monkeypatch: pytest.MonkeyPatch, outline: StoryOutline
) -> Callable[..., dict[type, FakeModel]]:
    """Build the planner and critic fakes with scripted critic verdicts.

    A factory rather than a fixture: a revision loop's behaviour is a function of
    what the critic says on each pass, and that has to be per-test. FakeModel
    repeats its final response once exhausted, so a one-element verdict list
    models a critic that never approves.
    """

    def build(
        verdicts: list[OutlineReviewsOutput] | None = None,
    ) -> dict[type, FakeModel]:
        by_schema: dict[type, FakeModel] = {
            StoryOutline: FakeModel(outline),
            OutlineReviewsOutput: FakeModel(
                *(verdicts or [OutlineReviewsOutput(reviews=[])])
            ),
        }

        class Dispatcher:
            """Stands in for an unbound model; hands the right fake over once
            bound. Reads `by_schema` at call time, so a test may replace an
            entry after the fixture has built it."""

            def with_structured_output(self, schema: type, **_: Any) -> FakeModel:
                return by_schema[schema].with_structured_output(schema)

        monkeypatch.setattr(OUTLINE_FACTORY, lambda *_a, **_k: Dispatcher())
        return by_schema

    return build


def _finding() -> OutlineReview:
    return OutlineReview(
        rubric=OutlineRubric.PROTAGONIST,
        beat_position=None,
        comment="The want belongs to the fox, not to the child.",
    )


class TestRunOutlinePipeline:
    async def test_returns_an_outline(
        self,
        outline_fakes: Callable[..., dict[type, FakeModel]],
        brief: StoryBrief,
        outline: StoryOutline,
    ) -> None:
        outline_fakes()
        assert await run_outline_pipeline(brief) == outline

    async def test_an_approving_critic_costs_one_extra_call(
        self,
        outline_fakes: Callable[..., dict[type, FakeModel]],
        brief: StoryBrief,
    ) -> None:
        """Empty on the first pass means no revision at all: the loop's whole
        cost on a good plan is one critic call."""
        fakes = outline_fakes()
        await run_outline_pipeline(brief)
        assert len(fakes[StoryOutline].calls) == 1
        assert len(fakes[OutlineReviewsOutput].calls) == 1

    async def test_a_finding_triggers_exactly_one_revision(
        self,
        outline_fakes: Callable[..., dict[type, FakeModel]],
        brief: StoryBrief,
    ) -> None:
        fakes = outline_fakes(
            [
                OutlineReviewsOutput(reviews=[_finding()]),
                OutlineReviewsOutput(reviews=[]),
            ]
        )
        await run_outline_pipeline(brief)
        assert len(fakes[StoryOutline].calls) == 2

    async def test_a_critic_that_never_approves_stops_at_the_cap(
        self,
        outline_fakes: Callable[..., dict[type, FakeModel]],
        brief: StoryBrief,
    ) -> None:
        """Hitting the cap returns a plan anyway. It is not an error: a plan the
        critic still dislikes beats no book at all."""
        fakes = outline_fakes([OutlineReviewsOutput(reviews=[_finding()])])
        result = await run_outline_pipeline(brief)
        assert isinstance(result, StoryOutline)
        # One first pass plus max_outline_revisions (default 2).
        assert len(fakes[StoryOutline].calls) == 3

    async def test_the_finding_reaches_the_planner(
        self,
        outline_fakes: Callable[..., dict[type, FakeModel]],
        brief: StoryBrief,
    ) -> None:
        """The test that catches a loop which runs but feeds nothing forward --
        the failure mode that passes every other test in this file."""
        fakes = outline_fakes(
            [
                OutlineReviewsOutput(reviews=[_finding()]),
                OutlineReviewsOutput(reviews=[]),
            ]
        )
        await run_outline_pipeline(brief)
        revision = fakes[StoryOutline].calls[1]
        assert len(revision) == 4
        assert _finding().comment in revision[3].content

    async def test_the_critic_sees_the_revised_outline_not_the_first_draft(
        self,
        outline_fakes: Callable[..., dict[type, FakeModel]],
        brief: StoryBrief,
    ) -> None:
        """Re-reviewing the draft that was already criticised would loop forever
        on a finding that had in fact been fixed."""
        fakes = outline_fakes(
            [
                OutlineReviewsOutput(reviews=[_finding()]),
                OutlineReviewsOutput(reviews=[]),
            ]
        )
        await run_outline_pipeline(brief)
        assert len(fakes[OutlineReviewsOutput].calls) == 2

    async def test_the_loop_returns_the_best_outline_not_the_last(
        self,
        outline_fakes: Callable[..., dict[type, FakeModel]],
        brief: StoryBrief,
        outline: StoryOutline,
    ) -> None:
        """A later revision can be worse and the loop cannot tell -- it
        re-critiques and sees a different set of findings, not a better plan."""
        better = outline.model_copy(update={"title": "The Better Plan"})
        worse = outline.model_copy(update={"title": "The Worse Plan"})
        fakes = outline_fakes(
            [
                OutlineReviewsOutput(reviews=[_finding(), _finding()]),
                OutlineReviewsOutput(reviews=[_finding()]),
                OutlineReviewsOutput(reviews=[_finding(), _finding()]),
            ]
        )
        fakes[StoryOutline] = FakeModel(outline, better, worse)

        result = await run_outline_pipeline(brief)

        assert result.title == "The Better Plan"

    async def test_every_completed_task_is_reported(
        self,
        outline_fakes: Callable[..., dict[type, FakeModel]],
        brief: StoryBrief,
    ) -> None:
        """The debug script numbers artifacts from this callback.

        Also pins the stream filter: ``plan_outline`` returns a ``StoryOutline``
        and so does the entrypoint, so the entrypoint's own result is identified
        by task name rather than by type. Get that wrong and the final outline
        arrives here as if it were a task result.
        """
        outline_fakes()
        seen: list[str] = []
        await run_outline_pipeline(brief, on_task_result=lambda n, _v: seen.append(n))
        assert seen == ["plan_outline", "critique_outline"]


class TestPlanStoryToolRunsTheCritic:
    """The preview is the plan the book is built from, so it must be critiqued.

    Before this, ``plan_story`` was one uncritiqued call and ``write_story``
    planned again -- so a parent approved a plan that was never used. See
    docs/superpowers/specs/2026-07-31-plan-carries-forward-design.md.
    """

    async def test_the_tool_revises_when_the_critic_objects(
        self,
        outline_fakes: Callable[..., dict[type, FakeModel]],
        brief: StoryBrief,
    ) -> None:
        fakes = outline_fakes(
            [
                OutlineReviewsOutput(reviews=[_finding()]),
                OutlineReviewsOutput(reviews=[]),
            ]
        )
        await plan_story_tool(brief)
        assert len(fakes[StoryOutline].calls) == 2, (
            "the preview must revise, or a parent approves an uncritiqued plan"
        )
