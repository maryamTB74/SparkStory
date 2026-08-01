"""Critic nodes, with no network.

A critic is a Node like any other: an injected model, its inputs, and the prompt
that joins them. These tests assert what it *sent* and which schema it *bound*,
which is what FakeModel exists for.
"""

from sparkstory.entities.guidelines import READING_LEVEL_GUIDANCE
from sparkstory.entities.reviews import (
    OutlineReview,
    OutlineReviewsOutput,
    OutlineRubric,
    ProseReviewsOutput,
    ProseRubric,
)
from sparkstory.entities.stories import (
    PagePlan,
    StoryBrief,
    StoryOutline,
    StoryPage,
    StoryProse,
)
from sparkstory.models.fake_model import FakeModel
from sparkstory.nodes.outline_critic import (
    OUTLINE_CRITIC_SYSTEM_PROMPT,
    OutlineCriticNode,
    render_outline_review_request,
)
from sparkstory.nodes.prose_critic import (
    PROSE_CRITIC_SYSTEM_PROMPT,
    ProseCriticNode,
    render_prose_review_request,
)


class TestOutlineCriticNode:
    async def test_binds_the_thin_output_schema(
        self, brief: StoryBrief, outline: StoryOutline
    ) -> None:
        """Binding the wrapper would pay the critic to echo the outline back."""
        model = FakeModel(OutlineReviewsOutput(reviews=[]))
        await OutlineCriticNode(model=model, brief=brief, outline=outline).ainvoke()
        assert model.bound_schema is OutlineReviewsOutput

    async def test_returns_the_wrapper_carrying_the_outline(
        self, brief: StoryBrief, outline: StoryOutline
    ) -> None:
        """The planner needs the draft and its reviews as one object, or the two
        can be paired wrongly."""
        review = OutlineReview(
            rubric=OutlineRubric.PROTAGONIST,
            comment="The want belongs to Pip; Maryam only helps him get it.",
        )
        model = FakeModel(OutlineReviewsOutput(reviews=[review]))
        result = await OutlineCriticNode(
            model=model, brief=brief, outline=outline
        ).ainvoke()
        assert result.outline is outline
        assert result.reviews == [review]

    async def test_an_approving_critic_returns_no_reviews(
        self, brief: StoryBrief, outline: StoryOutline
    ) -> None:
        """Empty is the stop signal, so it has to survive the wrapper. If this
        ever broke, the loop could not terminate early and would look like a
        critic that never approves."""
        model = FakeModel(OutlineReviewsOutput(reviews=[]))
        result = await OutlineCriticNode(
            model=model, brief=brief, outline=outline
        ).ainvoke()
        assert result.reviews == []


class TestOutlineCriticPrompt:
    def test_names_every_rubric(self) -> None:
        """Adding a rubric to the enum without writing its criteria into the
        prompt lets the model emit a judgement it was never told how to make."""
        lowered = OUTLINE_CRITIC_SYSTEM_PROMPT.lower()
        for rubric in OutlineRubric:
            assert rubric.value.replace("_", " ") in lowered

    def test_instructs_returning_nothing_when_the_plan_is_good(self) -> None:
        """Without this a critic finds something every pass and the empty-list
        stop signal never fires."""
        assert "return nothing" in OUTLINE_CRITIC_SYSTEM_PROMPT.lower()

    def test_forbids_rewriting_the_plan(self) -> None:
        """A critic that returns a fixed outline makes the planner redundant and
        routes around validation."""
        assert "do not rewrite the plan" in OUTLINE_CRITIC_SYSTEM_PROMPT.lower()

    def test_request_states_the_cap_as_a_number(
        self, brief: StoryBrief, outline: StoryOutline
    ) -> None:
        rendered = render_outline_review_request(brief, outline, max_reviews=5)
        assert "at most 5" in rendered

    def test_request_carries_the_child_and_the_beats(
        self, brief: StoryBrief, outline: StoryOutline
    ) -> None:
        rendered = render_outline_review_request(brief, outline, max_reviews=5)
        assert brief.child.name in rendered
        for beat in outline.beats:
            assert beat.title in rendered
            assert beat.summary in rendered

    def test_describes_craft_not_output_format(self) -> None:
        lowered = OUTLINE_CRITIC_SYSTEM_PROMPT.lower()
        for forbidden in ("json", "schema", "```"):
            assert forbidden not in lowered, (
                f"prompt describes output format: {forbidden!r}"
            )


class TestProseCriticNode:
    async def test_binds_the_thin_output_schema(
        self, brief: StoryBrief, page_plan: PagePlan, prose: StoryProse
    ) -> None:
        model = FakeModel(ProseReviewsOutput(reviews=[]))
        await ProseCriticNode(
            model=model, brief=brief, page_plan=page_plan, prose=prose
        ).ainvoke()
        assert model.bound_schema is ProseReviewsOutput

    async def test_returns_the_wrapper_carrying_the_prose(
        self, brief: StoryBrief, page_plan: PagePlan, prose: StoryProse
    ) -> None:
        model = FakeModel(ProseReviewsOutput(reviews=[]))
        result = await ProseCriticNode(
            model=model, brief=brief, page_plan=page_plan, prose=prose
        ).ainvoke()
        assert result.prose is prose

    async def test_an_approving_critic_returns_no_reviews(
        self, brief: StoryBrief, page_plan: PagePlan, prose: StoryProse
    ) -> None:
        model = FakeModel(ProseReviewsOutput(reviews=[]))
        result = await ProseCriticNode(
            model=model, brief=brief, page_plan=page_plan, prose=prose
        ).ainvoke()
        assert result.reviews == []


class TestProseCriticPrompt:
    def test_names_every_rubric(self) -> None:
        lowered = PROSE_CRITIC_SYSTEM_PROMPT.lower()
        for rubric in ProseRubric:
            assert rubric.value.replace("_", " ") in lowered

    def test_instructs_returning_nothing_when_the_words_are_good(self) -> None:
        assert "return nothing" in PROSE_CRITIC_SYSTEM_PROMPT.lower()

    def test_warns_that_a_wrong_safety_finding_is_costly(self) -> None:
        """Safety fails closed, so a false positive destroys a finished book.
        The critic has to know that before it raises one."""
        assert "whole book" in PROSE_CRITIC_SYSTEM_PROMPT.lower()

    def test_forbids_rewriting_the_words(self) -> None:
        assert "do not rewrite the words" in PROSE_CRITIC_SYSTEM_PROMPT.lower()

    def test_request_pairs_each_page_with_its_plan(
        self, brief: StoryBrief, page_plan: PagePlan, prose: StoryProse
    ) -> None:
        """plan_fidelity is unjudgeable unless the two sit side by side, which is
        how a one-page drift went unnoticed for a whole session."""
        rendered = render_prose_review_request(brief, page_plan, prose, max_reviews=5)
        assert page_plan.pages[0].visual_action in rendered
        assert page_plan.pages[0].emotional_shift in rendered
        assert prose.pages[0].text in rendered

    def test_request_carries_the_avoid_list(
        self, brief: StoryBrief, page_plan: PagePlan, prose: StoryProse
    ) -> None:
        """The safety rubric is the first thing in the system that enforces
        `avoid`, so the list has to actually reach the critic."""
        rendered = render_prose_review_request(brief, page_plan, prose, max_reviews=5)
        for avoided in brief.avoid:
            assert avoided in rendered

    def test_request_carries_the_reading_level_guidance(
        self, brief: StoryBrief, page_plan: PagePlan, prose: StoryProse
    ) -> None:
        """The bare enum value would leave the critic to invent our definition."""
        rendered = render_prose_review_request(brief, page_plan, prose, max_reviews=5)
        assert READING_LEVEL_GUIDANCE[brief.child.reading_level] in rendered

    def test_request_omits_the_outline(
        self, brief: StoryBrief, page_plan: PagePlan, prose: StoryProse
    ) -> None:
        """A critic that cannot see an artifact cannot invent findings about it."""
        rendered = render_prose_review_request(brief, page_plan, prose, max_reviews=5)
        assert "Beat " not in rendered

    def test_a_missing_page_is_marked_not_silently_skipped(
        self, brief: StoryBrief, page_plan: PagePlan
    ) -> None:
        """validate_prose catches this first, but the critic may run on prose a
        revision pass has just produced -- and a silently absent page reads to
        the critic as a page with no problems."""
        short = StoryProse(
            pages=[
                StoryPage(page_number=p.page_number, text="Words.")
                for p in page_plan.pages[:-1]
            ]
        )
        rendered = render_prose_review_request(brief, page_plan, short, max_reviews=5)
        assert "(missing)" in rendered

    def test_naming_a_feeling_counts_as_an_interiority_failure(self) -> None:
        """Live-run regression. The first version ranked a named feeling above
        an absent one, so the Writer's cheapest fix was to state it -- and page
        7 went from "Up it climbs, light and quick" to "Joy surges through",
        which is the emotional_shift note pasted in."""
        lowered = PROSE_CRITIC_SYSTEM_PROMPT.lower()
        assert "just as much a failure as leaving the feeling out" in lowered

    def test_forbids_quoting_the_notes_back(self) -> None:
        """The critic's own comment supplied the words that got pasted: "The
        notes say wonder replaces doubt and joy surges" became the page."""
        assert "never quote the notes back" in PROSE_CRITIC_SYSTEM_PROMPT.lower()


class TestProtagonistRubricScope:
    def test_it_checks_more_than_the_beats(self) -> None:
        """Live-run regression (outputs/20260730-232426-*). The critic approved
        a plan whose beats said Maryam wanted it while the logline, theme and
        character descriptions all still said she was helping. It had that
        evidence in front of it; it has to be told to use it."""
        lowered = OUTLINE_CRITIC_SYSTEM_PROMPT.lower()
        assert "logline" in lowered
        assert "theme" in lowered
        assert "character description" in lowered
