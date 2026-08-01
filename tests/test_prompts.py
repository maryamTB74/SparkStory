"""Prompt assembly.

Prompt text is behaviour, not presentation: an omitted constraint changes what
the model produces. These tests treat the rendered prompt as an output worth
asserting on.
"""

from sparkstory.entities.guidelines import READING_LEVEL_GUIDANCE
from sparkstory.entities.stories import (
    ChildProfile,
    PagePlan,
    ReadingLevel,
    StoryBrief,
    StoryOutline,
)
from sparkstory.nodes.plot_planner import PLOT_PLANNER_SYSTEM_PROMPT
from sparkstory.nodes.story_planner import (
    STORY_PLANNER_SYSTEM_PROMPT,
    render_story_brief,
)
from sparkstory.nodes.writer import WRITER_SYSTEM_PROMPT, render_prose_request


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

    def test_caps_beats_by_the_page_count(self) -> None:
        """A 6-beat outline for a 5-page book is unbuildable, so say so up front.

        Only the prompt can prevent this; validation can merely reject it after a
        model call has been paid for.
        """
        assert "never plan more beats than the book has pages" in (
            STORY_PLANNER_SYSTEM_PROMPT.lower()
        )

    def test_the_beat_limit_is_rendered_as_a_number(self, brief: StoryBrief) -> None:
        """ "Target length" alone left the limit to be inferred, and it was not."""
        assert f"at most {brief.page_count} beats" in render_story_brief(brief)


class TestPlotPlannerSystemPrompt:
    def test_demands_notes_not_narration(self) -> None:
        """The schema description alone is what failed before: the planner
        emitted finished sentences and the Writer paraphrased them."""
        assert "notes, not narration" in PLOT_PLANNER_SYSTEM_PROMPT.lower()

    def test_names_all_three_note_fields(self) -> None:
        """A prompt describing only two leaves the third to the schema, which
        is the situation this change exists to leave."""
        lowered = PLOT_PLANNER_SYSTEM_PROMPT.lower()
        assert "what the picture shows" in lowered
        assert "what changes inside" in lowered
        assert "page turn" in lowered

    def test_describes_craft_not_output_format(self) -> None:
        lowered = PLOT_PLANNER_SYSTEM_PROMPT.lower()
        for forbidden in ("json", "schema", "```"):
            assert forbidden not in lowered, (
                f"prompt describes output format: {forbidden!r}"
            )


class TestRenderProseRequest:
    def test_renders_all_three_notes_per_page(
        self, brief: StoryBrief, outline: StoryOutline, page_plan: PagePlan
    ) -> None:
        rendered = render_prose_request(brief, outline, page_plan)
        first = page_plan.pages[0]
        assert first.visual_action in rendered
        assert first.emotional_shift in rendered
        assert first.page_turn_hook in rendered

    def test_omits_the_hook_on_the_final_page(
        self, brief: StoryBrief, outline: StoryOutline, page_plan: PagePlan
    ) -> None:
        """A 'leaves open:' line with nothing after it invites an invented
        cliffhanger on the page where the book ends."""
        rendered = render_prose_request(brief, outline, page_plan)
        last_block = rendered.rsplit("Page ", 1)[1]
        assert "leaves open:" not in last_block


class TestWriterSystemPrompt:
    def test_forbids_copying_the_notes(self) -> None:
        """Finding #1: four of eight pages were the plan with the tense changed."""
        assert "never copy their wording" in WRITER_SYSTEM_PROMPT.lower()

    def test_requires_all_three_notes_to_land(self) -> None:
        """Without this, 'do not copy' is satisfiable by ignoring the notes --
        which is the page-drift half of the same defect."""
        assert "all three" in WRITER_SYSTEM_PROMPT.lower()
