"""Prompt assembly.

Prompt text is behaviour, not presentation: an omitted constraint changes what
the model produces. These tests treat the rendered prompt as an output worth
asserting on.
"""

from sparkstory.entities.guidelines import READING_LEVEL_GUIDANCE
from sparkstory.entities.stories import ChildProfile, ReadingLevel, StoryBrief
from sparkstory.nodes.story_planner import (
    STORY_PLANNER_SYSTEM_PROMPT,
    render_story_brief,
)


class TestReadingLevelGuidance:
    def test_every_level_has_guidance(self) -> None:
        """Adding a ReadingLevel must not silently miss the prompt."""
        missing = [
            level for level in ReadingLevel if level not in READING_LEVEL_GUIDANCE
        ]
        assert not missing, f"no guidance for {missing}"


class TestRenderStoryBrief:
    def test_includes_child_details(self, brief: StoryBrief) -> None:
        rendered = render_story_brief(brief)
        assert brief.child.name in rendered
        assert str(brief.child.age) in rendered
        assert brief.child.pronouns.value in rendered

    def test_includes_reading_level_guidance(self, brief: StoryBrief) -> None:
        assert READING_LEVEL_GUIDANCE[brief.child.reading_level] in render_story_brief(
            brief
        )

    def test_includes_hard_constraints(self, brief: StoryBrief) -> None:
        rendered = render_story_brief(brief)
        for avoided in brief.avoid:
            assert avoided in rendered

    def test_omits_empty_optional_sections(self, child: ChildProfile) -> None:
        """Empty headings are wasted tokens and invite the model to fill them."""
        minimal = StoryBrief(
            child=child.model_copy(update={"interests": []}), premise="a lost hat"
        )
        rendered = render_story_brief(minimal)
        assert "Interests:" not in rendered
        assert "Must include:" not in rendered
        assert "Must avoid entirely:" not in rendered


class TestSystemPrompt:
    def test_describes_craft_not_output_format(self) -> None:
        """Structured output enforces shape; restating it wastes tokens and drifts."""
        lowered = STORY_PLANNER_SYSTEM_PROMPT.lower()
        for forbidden in ("json", "schema", "```", "field"):
            assert forbidden not in lowered, (
                f"prompt describes output format: {forbidden!r}"
            )

    def test_states_the_non_negotiable_rules(self) -> None:
        lowered = STORY_PLANNER_SYSTEM_PROMPT.lower()
        assert "pronouns" in lowered
        assert "avoid" in lowered
        assert "moralise" in lowered
