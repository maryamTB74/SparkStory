"""The story engine end to end, with no network.

Every stage's model is faked by intercepting ``get_chat_model`` inside the
workflow module -- the one place the workflow builds models -- and returning a
different queued response per stage.

What only this file can catch: **wiring**. Each node can be perfect while the
workflow hands stage 3 the wrong stage-2 output, and no per-node test would notice.
"""

from typing import Any

import pytest
from fastmcp.exceptions import ToolError

from sparkstory.entities.exceptions import StoryStructureError
from sparkstory.entities.stories import (
    PagePlan,
    Story,
    StoryBrief,
    StoryOutline,
    StoryProse,
)
from sparkstory.mcp.tools.write_story import write_story_tool
from sparkstory.models.exceptions import MissingAPIKeyError
from sparkstory.models.fake_model import FakeModel
from sparkstory.workflows.write_story import _retry_on, run_story_pipeline

WORKFLOW_FACTORY = "sparkstory.workflows.write_story.get_chat_model"


@pytest.fixture
def fakes(
    monkeypatch: pytest.MonkeyPatch,
    outline: StoryOutline,
    page_plan: PagePlan,
    prose: StoryProse,
) -> dict[type, FakeModel]:
    """Give each stage its own FakeModel, keyed by the schema that stage binds.

    Keying on the bound schema rather than call order means a test still reads
    correctly if the pipeline's stages are ever reordered.
    """
    by_schema = {
        StoryOutline: FakeModel(outline),
        PagePlan: FakeModel(page_plan),
        StoryProse: FakeModel(prose),
    }

    class Dispatcher:
        """Stands in for an unbound model; hands over the right fake once bound."""

        def with_structured_output(self, schema: type, **_: Any) -> FakeModel:
            return by_schema[schema].with_structured_output(schema)

    monkeypatch.setattr(WORKFLOW_FACTORY, lambda *_a, **_k: Dispatcher())
    return by_schema  # type: ignore[return-value]


class TestRunStoryPipeline:
    async def test_returns_a_story_carrying_the_whole_provenance_chain(
        self,
        fakes: dict[type, FakeModel],
        brief: StoryBrief,
        outline: StoryOutline,
        page_plan: PagePlan,
        prose: StoryProse,
    ) -> None:
        story = await run_story_pipeline(brief)

        assert isinstance(story, Story)
        assert story.outline == outline
        assert story.page_plan == page_plan
        assert story.pages == prose.pages

    async def test_each_stage_receives_the_previous_stage_output(
        self,
        fakes: dict[type, FakeModel],
        brief: StoryBrief,
        outline: StoryOutline,
        page_plan: PagePlan,
    ) -> None:
        """The wiring assertion: a per-node test cannot catch a misrouted value."""
        await run_story_pipeline(brief)

        plot_prompt = fakes[PagePlan].messages[1].content
        assert outline.title in plot_prompt
        assert outline.beats[0].summary in plot_prompt

        writer_prompt = fakes[StoryProse].messages[1].content
        assert outline.title in writer_prompt
        assert page_plan.pages[0].visual_action in writer_prompt

    async def test_every_stage_is_called_exactly_once(
        self, fakes: dict[type, FakeModel], brief: StoryBrief
    ) -> None:
        """One call per stage: the writer writes the whole book in one pass."""
        await run_story_pipeline(brief)
        for schema, fake in fakes.items():
            assert len(fake.calls) == 1, f"{schema.__name__} stage called twice"

    async def test_a_structurally_wrong_plan_fails_loudly(
        self,
        monkeypatch: pytest.MonkeyPatch,
        brief: StoryBrief,
        outline: StoryOutline,
        page_plan: PagePlan,
        prose: StoryProse,
    ) -> None:
        """A plan that drops a beat must stop the run, not produce a thin book."""
        broken = PagePlan(pages=page_plan.pages[:-1])  # 9 pages, brief asks 10

        by_schema = {
            StoryOutline: FakeModel(outline),
            PagePlan: FakeModel(broken),
            StoryProse: FakeModel(prose),
        }

        class Dispatcher:
            def with_structured_output(self, schema: type, **_: Any) -> FakeModel:
                return by_schema[schema].with_structured_output(schema)

        monkeypatch.setattr(WORKFLOW_FACTORY, lambda *_a, **_k: Dispatcher())

        with pytest.raises(StoryStructureError, match="9 pages"):
            await run_story_pipeline(brief)

        # And the expensive stage never ran.
        assert by_schema[StoryProse].calls == []


class TestRetryPolicy:
    def test_structural_errors_are_not_retried(self) -> None:
        """Retrying an identical prompt only re-rolls the dice, and hides the rate."""
        assert _retry_on(StoryStructureError("wrong page count")) is False

    def test_validation_errors_are_not_retried(self) -> None:
        """LangGraph's default already declines ValueError, which ValidationError is."""

        class Boom(ValueError):
            pass

        assert _retry_on(Boom("bad shape")) is False

    def test_configuration_errors_are_not_retried(self) -> None:
        """Regression test for a defect found by running it.

        A missing GOOGLE_API_KEY was retried three times, printing three
        tracebacks for a problem whose fix is one line in .env. LangGraph's
        default_retry_on returns True for exception types it does not recognise,
        which includes all of ours.
        """
        assert _retry_on(MissingAPIKeyError("no key")) is False

    def test_transient_failures_are_retried(self) -> None:
        assert _retry_on(ConnectionError("provider hiccup")) is True


class TestToolErrorTranslation:
    async def test_missing_api_key_becomes_tool_error(
        self, monkeypatch: pytest.MonkeyPatch, brief: StoryBrief
    ) -> None:
        def raise_missing_key(*_: Any, **__: Any) -> None:
            raise MissingAPIKeyError(
                "Model 'gemini-3.5-flash' requires GOOGLE_API_KEY, which is not set."
            )

        monkeypatch.setattr(WORKFLOW_FACTORY, raise_missing_key)
        with pytest.raises(ToolError, match="GOOGLE_API_KEY"):
            await write_story_tool(brief)

    async def test_structure_errors_are_not_dressed_up_as_config_errors(
        self,
        monkeypatch: pytest.MonkeyPatch,
        brief: StoryBrief,
        outline: StoryOutline,
        page_plan: PagePlan,
        prose: StoryProse,
    ) -> None:
        """No operator can fix malformed output by editing .env."""
        broken = PagePlan(pages=page_plan.pages[:-1])
        by_schema = {
            StoryOutline: FakeModel(outline),
            PagePlan: FakeModel(broken),
            StoryProse: FakeModel(prose),
        }

        class Dispatcher:
            def with_structured_output(self, schema: type, **_: Any) -> FakeModel:
                return by_schema[schema].with_structured_output(schema)

        monkeypatch.setattr(WORKFLOW_FACTORY, lambda *_a, **_k: Dispatcher())

        with pytest.raises(StoryStructureError):
            await write_story_tool(brief)
