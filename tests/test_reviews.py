"""Session 4 schemas: the reshaped scene plan, and the review models.

These are Pydantic models bound with `with_structured_output`, so their
docstrings and field descriptions are prompt text. The tests here guard both
the shape and what that text says.
"""

from sparkstory.entities.stories import PagePlan, ScenePlan


class TestScenePlan:
    def test_splits_the_summary_into_three_fields(self) -> None:
        """One prose-shaped sentence is what the Writer paraphrased. Three
        orthogonal notes cannot be concatenated into a page."""
        page = ScenePlan(
            page_number=1,
            beat_position=1,
            setting="the garden at night",
            visual_action="Maryam at the window, moon low over the fence",
            emotional_shift="wonder tips into wanting",
            page_turn_hook="how could anyone get up there?",
            characters_present=["Maryam"],
        )
        assert page.visual_action
        assert page.emotional_shift
        assert page.page_turn_hook

    def test_scene_summary_is_gone(self) -> None:
        """A leftover field would let the Plot Planner keep emitting prose."""
        assert "scene_summary" not in ScenePlan.model_fields

    def test_page_turn_hook_is_optional(self) -> None:
        """The last page answers rather than asks."""
        page = ScenePlan(
            page_number=8,
            beat_position=4,
            setting="the garden at night",
            visual_action="Maryam and Pip asleep under the window",
            emotional_shift="wanting settles into having",
            characters_present=["Maryam", "Pip"],
        )
        assert page.page_turn_hook is None

    def test_descriptions_forbid_finished_prose(self) -> None:
        """The old description said 'one or two sentences', which is an
        instruction to write prose. These must not."""
        schema = ScenePlan.model_json_schema()
        for field in ("visual_action", "emotional_shift"):
            description = schema["properties"][field]["description"].lower()
            assert "sentence" not in description or "not" in description

    def test_page_plan_still_bounds_page_count(self) -> None:
        assert PagePlan.model_fields["pages"].metadata
