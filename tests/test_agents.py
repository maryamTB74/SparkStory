"""The Plot Planner and Writer nodes.

No network: each node is constructed with a ``FakeModel``. The assertions are
mostly about what reaches the model, because a constraint that never arrives in
the prompt cannot be honoured -- and for `avoid`, that is a safety failure rather
than a quality one.
"""

from sparkstory.entities.guidelines import READING_LEVEL_GUIDANCE
from sparkstory.entities.stories import (
    PagePlan,
    StoryBrief,
    StoryOutline,
    StoryProse,
)
from sparkstory.models.fake_model import FakeModel
from sparkstory.nodes.plot_planner import PLOT_PLANNER_SYSTEM_PROMPT, PlotPlannerNode
from sparkstory.nodes.writer import WRITER_SYSTEM_PROMPT, WriterNode


class TestPlotPlannerNode:
    async def test_returns_the_models_plan(
        self,
        brief: StoryBrief,
        outline: StoryOutline,
        page_plan: PagePlan,
    ) -> None:
        node = PlotPlannerNode(model=FakeModel(page_plan), brief=brief, outline=outline)
        assert await node.ainvoke() is page_plan

    async def test_binds_its_own_output_schema(
        self, brief: StoryBrief, outline: StoryOutline, page_plan: PagePlan
    ) -> None:
        fake = FakeModel(page_plan)
        PlotPlannerNode(model=fake, brief=brief, outline=outline)
        assert fake.bound_schema is PagePlan

    async def test_the_target_page_count_reaches_the_model(
        self, brief: StoryBrief, outline: StoryOutline, page_plan: PagePlan
    ) -> None:
        """Without the count, the plan cannot be right except by luck."""
        fake = FakeModel(page_plan)
        await PlotPlannerNode(model=fake, brief=brief, outline=outline).ainvoke()
        assert f"exactly {brief.page_count} pages" in fake.messages[1].content

    async def test_every_beat_reaches_the_model(
        self, brief: StoryBrief, outline: StoryOutline, page_plan: PagePlan
    ) -> None:
        fake = FakeModel(page_plan)
        await PlotPlannerNode(model=fake, brief=brief, outline=outline).ainvoke()

        human = fake.messages[1].content
        for beat in outline.beats:
            assert beat.title in human
            assert beat.summary in human
            assert f"Beat {beat.position}" in human

    async def test_avoid_list_reaches_the_model(
        self, brief: StoryBrief, outline: StoryOutline, page_plan: PagePlan
    ) -> None:
        fake = FakeModel(page_plan)
        await PlotPlannerNode(model=fake, brief=brief, outline=outline).ainvoke()
        for avoided in brief.avoid:
            assert avoided in fake.messages[1].content

    def test_prompt_forbids_prose_and_appearance(self) -> None:
        """Its two most common failure modes, both expensive downstream."""
        lowered = PLOT_PLANNER_SYSTEM_PROMPT.lower()
        assert "do not write the story" in lowered
        assert "one page is one moment" in lowered
        assert "colours" in lowered


class TestWriterNode:
    async def test_returns_the_models_prose(
        self,
        brief: StoryBrief,
        outline: StoryOutline,
        page_plan: PagePlan,
        prose: StoryProse,
    ) -> None:
        node = WriterNode(
            model=FakeModel(prose),
            brief=brief,
            outline=outline,
            page_plan=page_plan,
        )
        assert await node.ainvoke() is prose

    async def test_binds_its_own_output_schema(
        self,
        brief: StoryBrief,
        outline: StoryOutline,
        page_plan: PagePlan,
        prose: StoryProse,
    ) -> None:
        fake = FakeModel(prose)
        WriterNode(model=fake, brief=brief, outline=outline, page_plan=page_plan)
        assert fake.bound_schema is StoryProse

    async def test_every_page_of_the_plan_reaches_the_model(
        self,
        brief: StoryBrief,
        outline: StoryOutline,
        page_plan: PagePlan,
        prose: StoryProse,
    ) -> None:
        fake = FakeModel(prose)
        await WriterNode(
            model=fake, brief=brief, outline=outline, page_plan=page_plan
        ).ainvoke()

        human = fake.messages[1].content
        for page in page_plan.pages:
            assert f"Page {page.page_number}" in human
            # Both notes, not just one: a renderer that dropped emotional_shift
            # would silently undo finding #4's structural fix.
            assert page.visual_action in human
            assert page.emotional_shift in human

    async def test_reading_level_guidance_reaches_the_model(
        self,
        brief: StoryBrief,
        outline: StoryOutline,
        page_plan: PagePlan,
        prose: StoryProse,
    ) -> None:
        """The bare enum value would leave the model to invent our definition."""
        fake = FakeModel(prose)
        await WriterNode(
            model=fake, brief=brief, outline=outline, page_plan=page_plan
        ).ainvoke()
        assert (
            READING_LEVEL_GUIDANCE[brief.child.reading_level]
            in fake.messages[1].content
        )

    async def test_child_and_safety_constraints_reach_the_model(
        self,
        brief: StoryBrief,
        outline: StoryOutline,
        page_plan: PagePlan,
        prose: StoryProse,
    ) -> None:
        fake = FakeModel(prose)
        await WriterNode(
            model=fake, brief=brief, outline=outline, page_plan=page_plan
        ).ainvoke()

        human = fake.messages[1].content
        assert brief.child.name in human
        assert brief.child.pronouns.value in human
        for avoided in brief.avoid:
            assert avoided in human
        for required in brief.must_include:
            assert required in human

    def test_prompt_forbids_moralising_and_appearance(self) -> None:
        lowered = WRITER_SYSTEM_PROMPT.lower()
        assert "never state the theme" in lowered
        assert "lesson" in lowered
        assert "colours" in lowered


class TestPromptsDoNotLeakInternalTerms:
    """Docstrings and prompts are model-facing; our vocabulary is not.

    Session 1 shipped "the Canon Agent" and "spend tokens" to the model as part of
    its task. With prompt text now living on each node, the audit is this test
    rather than a single file to read.
    """

    def test_no_internal_vocabulary_in_any_system_prompt(self) -> None:
        forbidden = (
            "node",
            "beat_position",
            "page_count",
            "schema",
            "json",
            "token",
            "pydantic",
            "langgraph",
            "agent",
        )
        for name, prompt in (
            ("plot planner", PLOT_PLANNER_SYSTEM_PROMPT),
            ("writer", WRITER_SYSTEM_PROMPT),
        ):
            lowered = prompt.lower()
            for term in forbidden:
                assert term not in lowered, f"{name} prompt leaks {term!r}"
