"""The Plot Planner and Writer nodes.

No network: each node is constructed with a ``FakeModel``. The assertions are
mostly about what reaches the model, because a constraint that never arrives in
the prompt cannot be honoured -- and for `avoid`, that is a safety failure rather
than a quality one.
"""

from sparkstory.entities.grounding import GroundedFact, StoryGrounding
from sparkstory.entities.guidelines import READING_LEVEL_GUIDANCE
from sparkstory.entities.reviews import ProseReview, ProseReviews, ProseRubric
from sparkstory.entities.stories import (
    PagePlan,
    StoryBrief,
    StoryOutline,
    StoryProse,
    WorldRules,
)
from sparkstory.models.fake_model import FakeModel
from sparkstory.nodes.plot_planner import PLOT_PLANNER_SYSTEM_PROMPT, PlotPlannerNode
from sparkstory.nodes.writer import (
    WRITER_SYSTEM_PROMPT,
    WriterNode,
    render_prose_grounding,
    render_prose_request,
)


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


def _grounding() -> StoryGrounding:
    """One fact with every field populated, so the leak assertions have something
    to catch: a claim that must never render, and a source and chunk_id that must
    never reach a model."""
    return StoryGrounding(
        facts=[
            GroundedFact(
                claim="The Moon has no air.",
                story_note="Nothing outdoors can flutter, drift or make a sound.",
                source="NASA -- Moon Facts",
                chunk_id="moon#1",
            )
        ]
    )


def _grounded(outline: StoryOutline) -> StoryOutline:
    """The fixture outline with grounding attached, through validation.

    ``model_validate`` rather than ``model_copy(update=...)``, which skips
    validation and would set the attribute whether or not the field existed. That
    is a check with no room to fail, and it made an earlier version of a schema
    test pass before the field was added.
    """
    return StoryOutline.model_validate(
        outline.model_dump() | {"grounding": _grounding().model_dump()}
    )


class TestRenderProseGrounding:
    """The Writer's own grounding renderer.

    Deliberately not ``render_grounding`` reused. The planner is told to *shape a
    story that obeys this*; the Writer is told *this is already true -- write what
    it causes*. Opposite instructions over the same data, and one shared renderer
    would drift toward whichever caller was edited last.
    """

    def test_empty_grounding_renders_nothing(self) -> None:
        """Byte-identical prompts for an ungrounded run: it keeps every existing
        writer test valid and makes the A/B control arm real rather than nominal."""
        assert (
            render_prose_grounding(StoryGrounding(facts=[]), WorldRules.IMAGINATIVE)
            == ""
        )

    def test_the_story_note_is_rendered(self) -> None:
        rendered = render_prose_grounding(_grounding(), WorldRules.REALISTIC)
        assert "Nothing outdoors can flutter, drift or make a sound." in rendered

    def test_the_claim_is_never_rendered(self) -> None:
        """The whole reason ``claim`` and ``story_note`` are separate fields. A
        claim is directly recitable, and prose is the stage that writes the
        sentences a child reads aloud -- so this is where handing one over costs
        the most."""
        for rules in (WorldRules.REALISTIC, WorldRules.IMAGINATIVE):
            assert "The Moon has no air" not in render_prose_grounding(
                _grounding(), rules
            )

    def test_neither_source_nor_chunk_id_is_rendered(self) -> None:
        for rules in (WorldRules.REALISTIC, WorldRules.IMAGINATIVE):
            rendered = render_prose_grounding(_grounding(), rules)
            assert "NASA" not in rendered
            assert "moon#1" not in rendered

    def test_both_world_rules_forbid_stating_the_fact(self) -> None:
        """The same literal sentence in both branches, and the same one the
        planner uses -- not a paraphrase. A paraphrase of a prohibition is how a
        prohibition quietly weakens, and this sentence is what stands between a
        grounded book and a character reciting a fact."""
        prohibition = "Do not state them, explain them, or have anyone mention them"
        for rules in (WorldRules.REALISTIC, WorldRules.IMAGINATIVE):
            assert prohibition in render_prose_grounding(_grounding(), rules)

    def test_the_two_modes_differ(self) -> None:
        """If they rendered identically the argument would be dead weight, and a
        vacuous A/B would read as a real one, which has happened twice on live
        runs."""
        assert render_prose_grounding(
            _grounding(), WorldRules.REALISTIC
        ) != render_prose_grounding(_grounding(), WorldRules.IMAGINATIVE)


class TestWriterReceivesGrounding:
    """Task 4: the renderer wired into the prompt the Writer actually sends."""

    async def test_the_story_note_reaches_the_writer(
        self,
        brief: StoryBrief,
        outline: StoryOutline,
        page_plan: PagePlan,
        prose: StoryProse,
    ) -> None:
        model = FakeModel(prose)
        await WriterNode(
            model=model,
            brief=brief,
            outline=_grounded(outline),
            page_plan=page_plan,
        ).ainvoke()
        sent = "\n".join(str(m.content) for m in model.messages)
        assert "Nothing outdoors can flutter, drift or make a sound." in sent

    async def test_no_provenance_reaches_the_writer(
        self,
        brief: StoryBrief,
        outline: StoryOutline,
        page_plan: PagePlan,
        prose: StoryProse,
    ) -> None:
        model = FakeModel(prose)
        await WriterNode(
            model=model,
            brief=brief,
            outline=_grounded(outline),
            page_plan=page_plan,
        ).ainvoke()
        sent = "\n".join(str(m.content) for m in model.messages)
        assert "moon#1" not in sent
        assert "NASA" not in sent
        assert "The Moon has no air" not in sent

    def test_an_ungrounded_outline_renders_the_prompt_unchanged(
        self, brief: StoryBrief, outline: StoryOutline, page_plan: PagePlan
    ) -> None:
        """Acceptance test 4, byte-identical rather than merely similar. This is
        what keeps the provider-side prompt-cache prefix intact and what makes the
        ungrounded control arm a real comparison."""
        assert outline.grounding is None
        with_explicit_none = StoryOutline.model_validate(
            outline.model_dump() | {"grounding": None}
        )
        assert render_prose_request(brief, outline, page_plan) == render_prose_request(
            brief, with_explicit_none, page_plan
        )

    async def test_grounding_survives_a_revision(
        self,
        brief: StoryBrief,
        outline: StoryOutline,
        page_plan: PagePlan,
        prose: StoryProse,
    ) -> None:
        """The verification question most likely to be missed.

        The prose loop replays the previous draft, so a constraint present on
        draft 1 and absent on draft 2 would look like success on any run that
        converged immediately -- and the loop would be "fixing" a grounded page
        into an ungrounded one while its finding count improved.
        """
        model = FakeModel(prose)
        await WriterNode(
            model=model,
            brief=brief,
            outline=_grounded(outline),
            page_plan=page_plan,
            reviews=ProseReviews(
                prose=prose,
                reviews=[
                    ProseReview(
                        rubric=ProseRubric.READ_ALOUD,
                        page_number=1,
                        comment="Four pages open with the same word.",
                    )
                ],
            ),
        ).ainvoke()
        sent = "\n".join(str(m.content) for m in model.messages)
        assert "Nothing outdoors can flutter, drift or make a sound." in sent


class TestPromptsDoNotLeakInternalTerms:
    """Docstrings and prompts are model-facing; our vocabulary is not.

    An early version shipped "the Canon Agent" and "spend tokens" to the model as
    part of its task. With prompt text now living on each node, the audit is this
    test rather than a single file to read.
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
