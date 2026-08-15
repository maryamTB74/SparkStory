"""Generators editing their own drafts.

There is no editor node: the generator is rebuilt with the
reviews attached. What these tests guard is the *message shape*, because a loop
that runs the right number of times while feeding nothing forward passes every
other test in the suite and looks exactly like success.
"""

from langchain_core.messages import AIMessage

from sparkstory.entities.grounding import GroundedFact, StoryGrounding
from sparkstory.entities.reviews import (
    OutlineReview,
    OutlineReviews,
    OutlineRubric,
    ProseReview,
    ProseReviews,
    ProseRubric,
)
from sparkstory.entities.stories import (
    PagePlan,
    StoryBrief,
    StoryOutline,
    StoryProse,
)
from sparkstory.models.fake_model import FakeModel
from sparkstory.nodes.story_planner import StoryPlannerNode
from sparkstory.nodes.writer import WriterNode


def _reviews(outline: StoryOutline) -> OutlineReviews:
    return OutlineReviews(
        outline=outline,
        reviews=[
            OutlineReview(
                rubric=OutlineRubric.PROTAGONIST,
                beat_position=2,
                comment="The want belongs to Pip; Maryam only helps him get it.",
            )
        ],
    )


class TestStoryPlannerRevision:
    async def test_first_pass_sends_two_messages(
        self, brief: StoryBrief, outline: StoryOutline
    ) -> None:
        """The non-revision path stays untouched, which is also what keeps the
        provider-side prompt-cache prefix intact."""
        model = FakeModel(outline)
        await StoryPlannerNode(model=model, brief=brief).ainvoke()
        assert len(model.messages) == 2

    async def test_revision_sends_four_messages(
        self, brief: StoryBrief, outline: StoryOutline
    ) -> None:
        model = FakeModel(outline)
        await StoryPlannerNode(
            model=model, brief=brief, reviews=_reviews(outline)
        ).ainvoke()
        assert len(model.messages) == 4

    async def test_the_previous_draft_is_replayed_as_the_models_own_turn(
        self, brief: StoryBrief, outline: StoryOutline
    ) -> None:
        """The model then edits something it owns rather than critiquing a
        stranger's work.
        """
        model = FakeModel(outline)
        await StoryPlannerNode(
            model=model, brief=brief, reviews=_reviews(outline)
        ).ainvoke()
        replay = model.messages[2]
        assert isinstance(replay, AIMessage)
        assert outline.title in replay.content

    async def test_the_replayed_draft_carries_no_provenance(
        self, brief: StoryBrief, outline: StoryOutline
    ) -> None:
        """Schema text reaches the model, in the one place a whole outline is
        serialised to one.

        The replay dumps the previous draft with ``model_dump_json``, so once the
        outline carries grounding it would hand the planner every ``chunk_id`` and
        ``source``. Worse than the general case: the planner is the one node that
        both sees this schema and could *fill the field itself*, inventing an id
        that ``drop_unprovenanced`` then drops silently -- a real fact lost with no
        error anywhere.

        The draft itself must still be replayed. Excluding the whole message would
        "fix" the leak by removing the mechanism it protects.
        """
        grounded = StoryOutline.model_validate(
            outline.model_dump()
            | {
                "grounding": StoryGrounding(
                    facts=[
                        GroundedFact(
                            claim="The Moon has no air.",
                            story_note="Nothing outdoors can flutter or make a sound.",
                            source="NASA -- Moon Facts",
                            chunk_id="moon#1",
                        )
                    ]
                ).model_dump()
            }
        )
        model = FakeModel(outline)
        await StoryPlannerNode(
            model=model, brief=brief, reviews=_reviews(grounded)
        ).ainvoke()
        replay = model.messages[2]
        assert isinstance(replay, AIMessage)
        assert grounded.title in replay.content
        assert "moon#1" not in replay.content
        assert "NASA" not in replay.content

    async def test_the_review_comment_reaches_the_model(
        self, brief: StoryBrief, outline: StoryOutline
    ) -> None:
        """The test that catches a loop feeding nothing forward."""
        reviews = _reviews(outline)
        model = FakeModel(outline)
        await StoryPlannerNode(model=model, brief=brief, reviews=reviews).ainvoke()
        assert reviews.reviews[0].comment in model.messages[3].content

    async def test_the_rubric_and_beat_reach_the_model(
        self, brief: StoryBrief, outline: StoryOutline
    ) -> None:
        """A comment with no anchor makes the planner guess which beat to fix,
        and a wrong guess changes something nobody complained about."""
        model = FakeModel(outline)
        await StoryPlannerNode(
            model=model, brief=brief, reviews=_reviews(outline)
        ).ainvoke()
        sent = model.messages[3].content
        assert OutlineRubric.PROTAGONIST.value in sent
        assert "beat 2" in sent.lower()

    async def test_the_planner_is_told_to_keep_what_was_not_criticised(
        self, brief: StoryBrief, outline: StoryOutline
    ) -> None:
        """The whole plan is regenerated, so beats nobody criticised must be
        told to survive or the revision churns the parts that worked."""
        model = FakeModel(outline)
        await StoryPlannerNode(
            model=model, brief=brief, reviews=_reviews(outline)
        ).ainvoke()
        assert "keep everything that was not criticised" in (
            model.messages[3].content.lower()
        )


def _prose_reviews(prose: StoryProse) -> ProseReviews:
    return ProseReviews(
        prose=prose,
        reviews=[
            ProseReview(
                rubric=ProseRubric.INTERIORITY,
                page_number=4,
                comment="Page four is all action; the disappointment never arrives.",
            )
        ],
    )


class TestWriterRevision:
    async def test_first_pass_sends_two_messages(
        self,
        brief: StoryBrief,
        outline: StoryOutline,
        page_plan: PagePlan,
        prose: StoryProse,
    ) -> None:
        model = FakeModel(prose)
        await WriterNode(
            model=model, brief=brief, outline=outline, page_plan=page_plan
        ).ainvoke()
        assert len(model.messages) == 2

    async def test_revision_replays_the_draft_as_the_models_own_turn(
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
            outline=outline,
            page_plan=page_plan,
            reviews=_prose_reviews(prose),
        ).ainvoke()
        assert len(model.messages) == 4
        assert isinstance(model.messages[2], AIMessage)
        assert prose.pages[0].text in model.messages[2].content

    async def test_the_review_comment_and_page_reach_the_model(
        self,
        brief: StoryBrief,
        outline: StoryOutline,
        page_plan: PagePlan,
        prose: StoryProse,
    ) -> None:
        """The test that catches a loop feeding nothing forward."""
        reviews = _prose_reviews(prose)
        model = FakeModel(prose)
        await WriterNode(
            model=model,
            brief=brief,
            outline=outline,
            page_plan=page_plan,
            reviews=reviews,
        ).ainvoke()
        sent = model.messages[3].content
        assert reviews.reviews[0].comment in sent
        assert "page 4" in sent.lower()

    async def test_the_writer_is_told_to_leave_good_pages_alone(
        self,
        brief: StoryBrief,
        outline: StoryOutline,
        page_plan: PagePlan,
        prose: StoryProse,
    ) -> None:
        """The whole book is regenerated, so pages nobody criticised must be told
        to survive -- otherwise voice drifts where no finding existed, which is
        the damage the loop exists to prevent."""
        model = FakeModel(prose)
        await WriterNode(
            model=model,
            brief=brief,
            outline=outline,
            page_plan=page_plan,
            reviews=_prose_reviews(prose),
        ).ainvoke()
        assert "unchanged" in model.messages[3].content.lower()

    async def test_a_book_level_review_is_not_anchored_to_a_page(
        self,
        brief: StoryBrief,
        outline: StoryOutline,
        page_plan: PagePlan,
        prose: StoryProse,
    ) -> None:
        """Inventing a page number for a book-wide finding would send the fix to
        one page instead of all of them."""
        reviews = ProseReviews(
            prose=prose,
            reviews=[
                ProseReview(
                    rubric=ProseRubric.READ_ALOUD,
                    comment="Every page opens with the same word.",
                )
            ],
        )
        model = FakeModel(prose)
        await WriterNode(
            model=model,
            brief=brief,
            outline=outline,
            page_plan=page_plan,
            reviews=reviews,
        ).ainvoke()
        assert "the book as a whole" in model.messages[3].content.lower()
