"""Story Planner behaviour, and the tool layer's error translation.

No network. Node tests pass a ``FakeModel`` to the constructor rather than
patching a module attribute by string path -- a patch target is a string, so it
rots silently on rename and the test then passes for the wrong reason.

``plan_story`` still chooses the model from settings, so the few tests that care
about *which* model is chosen do intercept the factory.
"""

from typing import Any

import pytest
from fastmcp.exceptions import ToolError

from sparkstory.config import settings
from sparkstory.entities.exceptions import ConfigurationError, SparkStoryError
from sparkstory.entities.stories import StoryBrief, StoryOutline
from sparkstory.mcp.tools.plan_story import plan_story_tool
from sparkstory.models.exceptions import MissingAPIKeyError, UnknownModelError
from sparkstory.models.fake_model import FakeModel
from sparkstory.nodes.story_planner import StoryPlannerNode, plan_story

PLANNER_FACTORY = "sparkstory.nodes.story_planner.get_chat_model"


class TestStoryPlannerNode:
    async def test_returns_the_models_outline(
        self, brief: StoryBrief, outline: StoryOutline
    ) -> None:
        node = StoryPlannerNode(model=FakeModel(outline), brief=brief)
        assert await node.ainvoke() is outline

    async def test_binds_its_own_output_schema(
        self, brief: StoryBrief, outline: StoryOutline
    ) -> None:
        """The schema must be bound to the model, not parsed out afterwards."""
        fake = FakeModel(outline)
        StoryPlannerNode(model=fake, brief=brief)
        assert fake.bound_schema is StoryOutline

    async def test_sends_a_system_and_a_human_message(
        self, brief: StoryBrief, outline: StoryOutline
    ) -> None:
        fake = FakeModel(outline)
        await StoryPlannerNode(model=fake, brief=brief).ainvoke()

        assert len(fake.messages) == 2
        system, human = fake.messages
        assert system.type == "system"
        assert human.type == "human"

    async def test_brief_constraints_reach_the_model(
        self, brief: StoryBrief, outline: StoryOutline
    ) -> None:
        """An `avoid` entry that never reaches the prompt is a safety failure."""
        fake = FakeModel(outline)
        await StoryPlannerNode(model=fake, brief=brief).ainvoke()

        human = fake.messages[1].content
        assert brief.premise in human
        assert brief.child.name in human
        assert brief.child.pronouns.value in human
        for avoided in brief.avoid:
            assert avoided in human
        for required in brief.must_include:
            assert required in human


class TestPlanStory:
    """The thin orchestrator that picks a model and runs the node."""

    async def test_uses_the_configured_planner_model(
        self, monkeypatch: pytest.MonkeyPatch, brief: StoryBrief, outline: StoryOutline
    ) -> None:
        seen: dict[str, Any] = {}

        def factory(model_id: str) -> FakeModel:
            seen["model_id"] = model_id
            return FakeModel(outline)

        monkeypatch.setattr(PLANNER_FACTORY, factory)
        assert await plan_story(brief) is outline
        assert seen["model_id"] == settings.planner_model


class TestToolErrorTranslation:
    """Internal exceptions must become actionable client-facing errors."""

    async def test_missing_api_key_becomes_tool_error(
        self, monkeypatch: pytest.MonkeyPatch, brief: StoryBrief
    ) -> None:
        def raise_missing_key(*_: Any, **__: Any) -> None:
            raise MissingAPIKeyError(
                "Model 'gemini-3.5-flash' requires GOOGLE_API_KEY, which is not set."
            )

        monkeypatch.setattr(PLANNER_FACTORY, raise_missing_key)
        with pytest.raises(ToolError, match="GOOGLE_API_KEY"):
            await plan_story_tool(brief)

    async def test_unknown_model_becomes_tool_error(
        self, monkeypatch: pytest.MonkeyPatch, brief: StoryBrief
    ) -> None:
        def raise_unknown(*_: Any, **__: Any) -> None:
            raise UnknownModelError(
                "Unknown model_id 'nope'. Known models: gemini-3.5-flash."
            )

        monkeypatch.setattr(PLANNER_FACTORY, raise_unknown)
        with pytest.raises(ToolError, match="Known models"):
            await plan_story_tool(brief)

    async def test_unexpected_errors_are_not_swallowed(
        self, monkeypatch: pytest.MonkeyPatch, brief: StoryBrief
    ) -> None:
        """Blanket-catching would turn a genuine bug into a polite message."""

        def raise_bug(*_: Any, **__: Any) -> None:
            raise ZeroDivisionError("a real bug")

        monkeypatch.setattr(PLANNER_FACTORY, raise_bug)
        with pytest.raises(ZeroDivisionError):
            await plan_story_tool(brief)

    @pytest.mark.parametrize(
        "builtin_error", [RuntimeError("unrelated"), KeyError("unrelated")]
    )
    async def test_builtin_errors_are_not_mistaken_for_config_errors(
        self,
        monkeypatch: pytest.MonkeyPatch,
        brief: StoryBrief,
        builtin_error: Exception,
    ) -> None:
        """Regression test for a real defect.

        This layer used to catch bare ``RuntimeError`` and ``KeyError`` to mean
        "missing API key" and "unknown model". Any unrelated error of those types
        -- from LangChain, the transport, asyncio -- was therefore reported to the
        client as a configuration problem, sending debugging to the wrong layer.
        Only our own ConfigurationError may be translated.
        """

        def raise_builtin(*_: Any, **__: Any) -> None:
            raise builtin_error

        monkeypatch.setattr(PLANNER_FACTORY, raise_builtin)
        with pytest.raises(type(builtin_error)):
            await plan_story_tool(brief)


class TestExceptionTaxonomy:
    def test_config_errors_share_a_translatable_base(self) -> None:
        """The tool layer catches ConfigurationError, so both must inherit it."""
        assert issubclass(UnknownModelError, ConfigurationError)
        assert issubclass(MissingAPIKeyError, ConfigurationError)

    def test_all_errors_share_a_package_base(self) -> None:
        """Lets a caller distinguish our failures from arbitrary exceptions."""
        assert issubclass(ConfigurationError, SparkStoryError)
