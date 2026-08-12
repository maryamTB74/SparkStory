"""The story engine end to end, with no network.

Every stage's model is faked by intercepting ``get_chat_model`` inside the
workflow module -- the one place the workflow builds models -- and returning a
different queued response per stage.

What only this file can catch: **wiring**. Each node can be perfect while the
workflow hands stage 3 the wrong stage-2 output, and no per-node test would notice.
"""

from collections.abc import Callable
from typing import Any

import pytest
from fastmcp.exceptions import ToolError

from sparkstory.entities.exceptions import (
    ConfigurationError,
    SparkStoryError,
    StoryStructureError,
    UnsafeContentError,
)
from sparkstory.entities.grounding import GroundedFact, StoryGrounding
from sparkstory.entities.reviews import (
    ProseReview,
    ProseReviewsOutput,
    ProseRubric,
)
from sparkstory.entities.stories import (
    PagePlan,
    Story,
    StoryBrief,
    StoryOutline,
    StoryPage,
    StoryProse,
)
from sparkstory.mcp.tools.write_story import write_story_tool
from sparkstory.models.exceptions import MissingAPIKeyError
from sparkstory.models.fake_model import FakeModel
from sparkstory.workflows import write_story as write_story_module
from sparkstory.workflows.retries import _retry_on
from sparkstory.workflows.write_story import run_story_pipeline

WORKFLOW_FACTORY = "sparkstory.workflows.write_story.get_chat_model"


@pytest.fixture
def looping_fakes(
    monkeypatch: pytest.MonkeyPatch,
    page_plan: PagePlan,
    prose: StoryProse,
) -> Callable[..., dict[type, FakeModel]]:
    """Build the stage fakes with a scripted sequence of critic verdicts.

    No planner or outline-critic fake: this workflow no longer plans. The
    outline is a caller-supplied argument, so tests pass the ``outline``
    fixture directly and the outline loop is covered in
    ``test_outline_workflow.py``.

    A factory rather than a fixture, because a revision loop's behaviour is a
    function of what the critic says on each pass, and that has to be per-test.

    FakeModel repeats its final response once exhausted, so a one-element verdict
    list models a critic that never approves, and the loop must stop at its cap.
    """

    def build(
        prose_verdicts: list[ProseReviewsOutput] | None = None,
    ) -> dict[type, FakeModel]:
        by_schema: dict[type, FakeModel] = {
            PagePlan: FakeModel(page_plan),
            StoryProse: FakeModel(prose),
            ProseReviewsOutput: FakeModel(
                *(prose_verdicts or [ProseReviewsOutput(reviews=[])])
            ),
        }

        class Dispatcher:
            """Stands in for an unbound model; hands the right fake over once
            bound. Reads `by_schema` at call time, so a test may add or replace
            an entry after the fixture has built it."""

            def with_structured_output(self, schema: type, **_: Any) -> FakeModel:
                return by_schema[schema].with_structured_output(schema)

        monkeypatch.setattr(WORKFLOW_FACTORY, lambda *_a, **_k: Dispatcher())
        return by_schema

    return build


@pytest.fixture
def fakes(
    looping_fakes: Callable[..., dict[type, FakeModel]],
) -> dict[type, FakeModel]:
    """Each stage's FakeModel, keyed by the schema that stage binds, with a
    critic that approves on the first pass.

    Keying on the bound schema rather than call order means a test still reads
    correctly if the pipeline's stages are ever reordered. The approving critic
    keeps these wiring tests measuring wiring: one call per stage, no revisions.
    """
    return looping_fakes()


class TestCallerGroundingIsVerified:
    """The outline arrives from a caller, so its grounding does too.

    A fabricated ``source: "NASA"`` is a provenance lie in the one feature whose
    purpose is factual accuracy, so every ``chunk_id`` is resolved against the
    store and ``source`` is overwritten from it -- the same move
    ``drop_unprovenanced`` already makes on the planning side, applied where the
    data stops being ours.

    It **drops** rather than raising, matching ``drop_unroutable_prose_reviews``: a
    brief whose whole grounding is fabricated becomes an ungrounded run, not a
    failed one.
    """

    async def test_a_fabricated_chunk_id_is_dropped(
        self,
        fakes: dict[type, FakeModel],
        brief: StoryBrief,
        outline: StoryOutline,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Acceptance test 5, forced rather than left unfalsified. An unfalsified
        check is not a proven one -- finding M is the worked example."""
        monkeypatch.setattr(
            write_story_module, "build_store", lambda *a, **k: _EmptyStore()
        )
        grounded = StoryOutline.model_validate(
            outline.model_dump()
            | {
                "grounding": StoryGrounding(
                    facts=[
                        GroundedFact(
                            claim="Invented.",
                            story_note="An invented note.",
                            source="Nowhere at all",
                            chunk_id="moon#9999",
                        )
                    ]
                ).model_dump()
            }
        )

        story = await run_story_pipeline(brief, grounded)

        assert story.outline.grounding is not None
        assert story.outline.grounding.facts == []

    async def test_an_ungrounded_outline_needs_no_database(
        self,
        fakes: dict[type, FakeModel],
        brief: StoryBrief,
        outline: StoryOutline,
    ) -> None:
        """`build_store` raises `ConfigurationError` when DATABASE_URL is unset, so
        verifying unconditionally would make an ungrounded `write_story` newly
        require Postgres -- a real behaviour change for every caller that never
        researched. Nothing is stubbed here deliberately: this test would fail if
        the store were built when there is nothing to verify."""
        assert outline.grounding is None
        story = await run_story_pipeline(brief, outline)
        assert story.outline.grounding is None


class _EmptyStore:
    """A store that vouches for nothing, so every cited id is unprovenanced.

    Not a `PgVectorStore`: `drop_unprovenanced` takes the `ChunkStore` protocol, so
    the only methods needed are the ones it calls. A real store here would need a
    database and would put this module's offline guarantee at risk.
    """

    def get(self, chunk_id: str) -> None:
        return None


class TestRunStoryPipeline:
    async def test_returns_a_story_carrying_the_whole_provenance_chain(
        self,
        fakes: dict[type, FakeModel],
        brief: StoryBrief,
        outline: StoryOutline,
        page_plan: PagePlan,
        prose: StoryProse,
    ) -> None:
        story = await run_story_pipeline(brief, outline)

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
        await run_story_pipeline(brief, outline)

        plot_prompt = fakes[PagePlan].messages[1].content
        assert outline.title in plot_prompt
        assert outline.beats[0].summary in plot_prompt

        writer_prompt = fakes[StoryProse].messages[1].content
        assert outline.title in writer_prompt
        assert page_plan.pages[0].visual_action in writer_prompt

    async def test_every_stage_is_called_exactly_once(
        self, fakes: dict[type, FakeModel], brief: StoryBrief, outline: StoryOutline
    ) -> None:
        """One call per stage: the writer writes the whole book in one pass."""
        await run_story_pipeline(brief, outline)
        for schema, fake in fakes.items():
            assert len(fake.calls) == 1, f"{schema.__name__} stage called twice"

    async def test_a_structurally_wrong_plan_fails_loudly(
        self,
        looping_fakes: Callable[..., dict[type, FakeModel]],
        brief: StoryBrief,
        outline: StoryOutline,
        page_plan: PagePlan,
    ) -> None:
        """A plan that drops a beat must stop the run, not produce a thin book."""
        fakes = looping_fakes()
        # 9 pages where the brief asks for 10. Replaced after the fixture built
        # the dispatcher, which reads the dict at call time.
        fakes[PagePlan] = FakeModel(PagePlan(pages=page_plan.pages[:-1]))

        with pytest.raises(StoryStructureError, match="9 pages"):
            await run_story_pipeline(brief, outline)

        # And the expensive stage never ran.
        assert fakes[StoryProse].calls == []

    async def test_an_outline_that_does_not_fit_the_brief_fails_loudly(
        self,
        fakes: dict[type, FakeModel],
        brief: StoryBrief,
        outline: StoryOutline,
    ) -> None:
        """The outline is caller-supplied now, so this is a boundary check.

        A beat needs a page of its own. brief.page_count is 10 and StoryOutline
        permits up to 8 beats, so a too-many-beats outline needs a smaller brief
        rather than a bigger outline.
        """
        cramped = brief.model_copy(update={"page_count": 4})
        five_beats = outline.model_copy(
            update={"beats": [*outline.beats, outline.beats[-1]]}
        )

        with pytest.raises(StoryStructureError, match="beats"):
            await run_story_pipeline(cramped, five_beats)

        # Rejected before a single model call was paid for.
        assert fakes[PagePlan].calls == []


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
        self, monkeypatch: pytest.MonkeyPatch, brief: StoryBrief, outline: StoryOutline
    ) -> None:
        def raise_missing_key(*_: Any, **__: Any) -> None:
            raise MissingAPIKeyError(
                "Model 'gemini-3.5-flash' requires GOOGLE_API_KEY, which is not set."
            )

        monkeypatch.setattr(WORKFLOW_FACTORY, raise_missing_key)
        with pytest.raises(ToolError, match="GOOGLE_API_KEY"):
            await write_story_tool(brief, outline)

    async def test_structure_errors_are_not_dressed_up_as_config_errors(
        self,
        looping_fakes: Callable[..., dict[type, FakeModel]],
        brief: StoryBrief,
        outline: StoryOutline,
        page_plan: PagePlan,
    ) -> None:
        """No operator can fix malformed output by editing .env."""
        fakes = looping_fakes()
        fakes[PagePlan] = FakeModel(PagePlan(pages=page_plan.pages[:-1]))

        with pytest.raises(StoryStructureError):
            await write_story_tool(brief, outline)

    async def test_a_mismatched_outline_is_a_client_error_not_a_bug(
        self,
        fakes: dict[type, FakeModel],
        brief: StoryBrief,
        outline: StoryOutline,
    ) -> None:
        """The outline comes from an LLM client, so a mismatch is its mistake.

        Deliberately narrower than the test above, which must keep passing: a
        StoryStructureError raised *inside* the pipeline still means our own
        agent produced nonsense, and that is a bug the client cannot act on.
        """
        cramped = brief.model_copy(update={"page_count": 4})
        five_beats = outline.model_copy(
            update={"beats": [*outline.beats, outline.beats[-1]]}
        )

        with pytest.raises(ToolError, match="beats"):
            await write_story_tool(cramped, five_beats)


class TestUnsafeContentClassification:
    def test_unsafe_content_is_not_retried(self) -> None:
        """LangGraph's default_retry_on returns True for types it does not
        recognise, which is every exception of ours -- so a new class that is
        not classified here silently gets three attempts."""
        assert _retry_on(UnsafeContentError("a spider on page four")) is False

    def test_unsafe_content_is_not_a_configuration_error(self) -> None:
        """No operator fixes this by editing .env, so it must not be dressed up
        as something an operator can act on."""
        assert not isinstance(UnsafeContentError("x"), ConfigurationError)

    def test_unsafe_content_is_one_of_ours(self) -> None:
        """A caller distinguishing our failures from arbitrary Python needs it
        under the shared base."""
        assert isinstance(UnsafeContentError("x"), SparkStoryError)


def _prose_finding(rubric: ProseRubric = ProseRubric.INTERIORITY) -> ProseReview:
    return ProseReview(
        rubric=rubric,
        page_number=4,
        comment="Page four is all action; the disappointment never arrives.",
    )


class TestProseLoop:
    async def test_an_approving_critic_costs_one_extra_call(
        self,
        brief: StoryBrief,
        outline: StoryOutline,
        looping_fakes: Callable[..., dict[type, FakeModel]],
    ) -> None:
        fakes = looping_fakes(prose_verdicts=[ProseReviewsOutput(reviews=[])])
        await run_story_pipeline(brief, outline)
        assert len(fakes[StoryProse].calls) == 1
        assert len(fakes[ProseReviewsOutput].calls) == 1

    async def test_a_finding_triggers_a_rewrite(
        self,
        brief: StoryBrief,
        outline: StoryOutline,
        looping_fakes: Callable[..., dict[type, FakeModel]],
    ) -> None:
        fakes = looping_fakes(
            prose_verdicts=[
                ProseReviewsOutput(reviews=[_prose_finding()]),
                ProseReviewsOutput(reviews=[]),
            ]
        )
        await run_story_pipeline(brief, outline)
        assert len(fakes[StoryProse].calls) == 2

    async def test_the_finding_reaches_the_writer(
        self,
        brief: StoryBrief,
        outline: StoryOutline,
        looping_fakes: Callable[..., dict[type, FakeModel]],
    ) -> None:
        """The test that catches a loop feeding nothing forward."""
        fakes = looping_fakes(
            prose_verdicts=[
                ProseReviewsOutput(reviews=[_prose_finding()]),
                ProseReviewsOutput(reviews=[]),
            ]
        )
        await run_story_pipeline(brief, outline)
        rewrite = fakes[StoryProse].calls[1]
        assert len(rewrite) == 4
        assert _prose_finding().comment in rewrite[3].content

    async def test_the_loop_always_ends_on_a_critique(
        self,
        brief: StoryBrief,
        outline: StoryOutline,
        looping_fakes: Callable[..., dict[type, FakeModel]],
    ) -> None:
        """N rewrites but N+1 critiques. Ending on an unreviewed rewrite would
        make the safety gate judge a draft that no longer exists."""
        fakes = looping_fakes(
            prose_verdicts=[ProseReviewsOutput(reviews=[_prose_finding()])]
        )
        await run_story_pipeline(brief, outline)
        # max_prose_revisions defaults to 2.
        assert len(fakes[StoryProse].calls) == 3
        assert len(fakes[ProseReviewsOutput].calls) == 3

    async def test_a_surviving_craft_finding_still_returns_the_book(
        self,
        brief: StoryBrief,
        outline: StoryOutline,
        looping_fakes: Callable[..., dict[type, FakeModel]],
    ) -> None:
        """Only safety fails closed. A flat page is not worth losing a book."""
        looping_fakes(
            prose_verdicts=[
                ProseReviewsOutput(reviews=[_prose_finding(ProseRubric.READ_ALOUD)])
            ]
        )
        assert isinstance(await run_story_pipeline(brief, outline), Story)

    async def test_a_surviving_safety_finding_fails_closed(
        self,
        brief: StoryBrief,
        outline: StoryOutline,
        looping_fakes: Callable[..., dict[type, FakeModel]],
    ) -> None:
        """Returning a book with a known safety finding is worse than none."""
        looping_fakes(
            prose_verdicts=[
                ProseReviewsOutput(reviews=[_prose_finding(ProseRubric.SAFETY)])
            ]
        )
        with pytest.raises(UnsafeContentError, match="safety"):
            await run_story_pipeline(brief, outline)

    async def test_a_safety_finding_fixed_in_time_does_not_fail_closed(
        self,
        brief: StoryBrief,
        outline: StoryOutline,
        looping_fakes: Callable[..., dict[type, FakeModel]],
    ) -> None:
        """The gate judges the final draft, not the one that was criticised."""
        looping_fakes(
            prose_verdicts=[
                ProseReviewsOutput(reviews=[_prose_finding(ProseRubric.SAFETY)]),
                ProseReviewsOutput(reviews=[]),
            ]
        )
        assert isinstance(await run_story_pipeline(brief, outline), Story)

    async def test_deterministic_findings_merge_with_the_critics(
        self,
        brief: StoryBrief,
        outline: StoryOutline,
        looping_fakes: Callable[..., dict[type, FakeModel]],
    ) -> None:
        """A counted finding must trigger a rewrite on its own, with no critic
        finding at all -- otherwise the merge is decorative and #2 never gets
        fixed unless the LLM happens to notice it too."""
        fakes = looping_fakes(prose_verdicts=[ProseReviewsOutput(reviews=[])])
        droning = StoryProse(
            pages=[
                StoryPage(page_number=n, text=f"Maryam did thing {n}.")
                for n in range(1, 11)
            ]
        )
        fakes[StoryProse] = FakeModel(droning)

        await run_story_pipeline(brief, outline)

        assert len(fakes[StoryProse].calls) > 1, "counted finding did not loop"
        assert "begin with the same word" in fakes[StoryProse].calls[1][3].content


class TestUnsafeContentTranslation:
    async def test_the_client_is_told_rather_than_shown_a_traceback(
        self,
        brief: StoryBrief,
        outline: StoryOutline,
        looping_fakes: Callable[..., dict[type, FakeModel]],
    ) -> None:
        """Unlike StoryStructureError this is not a bug: it means the system
        worked and the answer is no, which the caller needs to hear. The caller
        may be an LLM agent, which can act on a sentence and not on a stack."""
        looping_fakes(
            prose_verdicts=[
                ProseReviewsOutput(reviews=[_prose_finding(ProseRubric.SAFETY)])
            ]
        )
        with pytest.raises(ToolError) as caught:
            await write_story_tool(brief, outline)
        assert "safety" in str(caught.value).lower()

    async def test_the_comment_reaches_the_client(
        self,
        brief: StoryBrief,
        outline: StoryOutline,
        looping_fakes: Callable[..., dict[type, FakeModel]],
    ) -> None:
        """'We could not do it' without saying why leaves the parent unable to
        adjust the brief and try again."""
        looping_fakes(
            prose_verdicts=[
                ProseReviewsOutput(reviews=[_prose_finding(ProseRubric.SAFETY)])
            ]
        )
        with pytest.raises(ToolError) as caught:
            await write_story_tool(brief, outline)
        assert _prose_finding().comment in str(caught.value)


class TestTaskResultCallback:
    async def test_every_completed_task_is_reported(
        self,
        brief: StoryBrief,
        outline: StoryOutline,
        looping_fakes: Callable[..., dict[type, FakeModel]],
    ) -> None:
        """The loops run inside the entrypoint, so a returned Story shows only
        the drafts that survived. Without this hook a bad convergence is
        indistinguishable from a clean first pass."""
        looping_fakes()
        seen: list[str] = []
        await run_story_pipeline(
            brief, outline, on_task_result=lambda n, _v: seen.append(n)
        )
        assert {"plan_pages", "write_prose", "critique_prose"} <= set(seen)

    async def test_each_revision_is_reported_separately(
        self,
        brief: StoryBrief,
        outline: StoryOutline,
        looping_fakes: Callable[..., dict[type, FakeModel]],
    ) -> None:
        """One entry per call, not one per task name -- otherwise the artifact
        for a three-pass run looks the same as for a one-pass run."""
        looping_fakes(
            prose_verdicts=[
                ProseReviewsOutput(reviews=[_prose_finding()]),
                ProseReviewsOutput(reviews=[]),
            ]
        )
        seen: list[str] = []
        await run_story_pipeline(
            brief, outline, on_task_result=lambda n, _v: seen.append(n)
        )
        assert seen.count("write_prose") == 2

    async def test_the_callback_is_optional(
        self,
        brief: StoryBrief,
        outline: StoryOutline,
        looping_fakes: Callable[..., dict[type, FakeModel]],
    ) -> None:
        """The MCP tool passes one argument and must keep working."""
        looping_fakes()
        assert isinstance(await run_story_pipeline(brief, outline), Story)

    async def test_the_story_is_still_returned_when_streaming(
        self,
        brief: StoryBrief,
        outline: StoryOutline,
        looping_fakes: Callable[..., dict[type, FakeModel]],
    ) -> None:
        """Streaming replaced ainvoke, so the return value has to survive it."""
        looping_fakes()
        story = await run_story_pipeline(
            brief, outline, on_task_result=lambda _n, _v: None
        )
        assert isinstance(story, Story)
        assert story.outline.title


class TestLoopsKeepTheBestDraft:
    """Live-run regression (outputs/20260730-232426-*).

    The prose loop oscillated 5 -> 3 -> 3 findings and hit its cap, and pages 3
    and 6 were better in draft 2 than in draft 3. Returning the last draft
    shipped the worse one. The loop now returns the best draft it saw.
    """

    async def test_the_prose_loop_returns_the_draft_with_fewest_findings(
        self,
        brief: StoryBrief,
        outline: StoryOutline,
        looping_fakes: Callable[..., dict[type, FakeModel]],
        page_plan: PagePlan,
    ) -> None:
        good = StoryProse(
            pages=[
                StoryPage(
                    page_number=p.page_number, text=f"Draft two page {p.page_number}."
                )
                for p in page_plan.pages
            ]
        )
        worse = StoryProse(
            pages=[
                StoryPage(
                    page_number=p.page_number, text=f"Draft three page {p.page_number}."
                )
                for p in page_plan.pages
            ]
        )
        fakes = looping_fakes(
            prose_verdicts=[
                ProseReviewsOutput(reviews=[_prose_finding(), _prose_finding()]),
                ProseReviewsOutput(reviews=[_prose_finding()]),
                ProseReviewsOutput(reviews=[_prose_finding(), _prose_finding()]),
            ]
        )
        fakes[StoryProse] = FakeModel(
            StoryProse(
                pages=[
                    StoryPage(page_number=p.page_number, text="one.")
                    for p in page_plan.pages
                ]
            ),
            good,
            worse,
        )

        story = await run_story_pipeline(brief, outline)

        assert story.pages == good.pages, "the loop shipped a worse later draft"

    async def test_a_tie_keeps_the_earlier_draft(
        self,
        brief: StoryBrief,
        outline: StoryOutline,
        looping_fakes: Callable[..., dict[type, FakeModel]],
        page_plan: PagePlan,
    ) -> None:
        """Equal findings means the extra rewrite bought nothing, and every
        rewrite is another chance to damage a page nobody complained about."""
        first = StoryProse(
            pages=[
                StoryPage(page_number=p.page_number, text="first.")
                for p in page_plan.pages
            ]
        )
        second = StoryProse(
            pages=[
                StoryPage(page_number=p.page_number, text="second.")
                for p in page_plan.pages
            ]
        )
        fakes = looping_fakes(
            prose_verdicts=[ProseReviewsOutput(reviews=[_prose_finding()])]
        )
        fakes[StoryProse] = FakeModel(first, second)

        story = await run_story_pipeline(brief, outline)

        assert story.pages == first.pages

    async def test_a_safe_draft_beats_a_smaller_unsafe_one(
        self,
        brief: StoryBrief,
        outline: StoryOutline,
        looping_fakes: Callable[..., dict[type, FakeModel]],
        page_plan: PagePlan,
    ) -> None:
        """Fewest-findings alone would keep a draft with a safety finding over a
        safe one. For a guardrail that is the wrong way round."""
        unsafe = StoryProse(
            pages=[
                StoryPage(page_number=p.page_number, text="unsafe.")
                for p in page_plan.pages
            ]
        )
        safe = StoryProse(
            pages=[
                StoryPage(page_number=p.page_number, text="safe.")
                for p in page_plan.pages
            ]
        )
        fakes = looping_fakes(
            prose_verdicts=[
                ProseReviewsOutput(reviews=[_prose_finding(ProseRubric.SAFETY)]),
                ProseReviewsOutput(
                    reviews=[_prose_finding(), _prose_finding(), _prose_finding()]
                ),
            ]
        )
        fakes[StoryProse] = FakeModel(unsafe, safe)

        story = await run_story_pipeline(brief, outline)

        assert story.pages == safe.pages
