"""The story engine end to end, with no network.

Every stage's model is faked by intercepting ``get_chat_model`` inside the
workflow module -- the one place the workflow builds models -- and returning a
different queued response per stage.

What only this file can catch: **wiring**. Each node can be perfect while the
workflow hands stage 3 the wrong stage-2 output, and no per-node test would notice.
"""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from fastmcp.exceptions import ToolError

from sparkstory.entities.exceptions import (
    ConfigurationError,
    SparkStoryError,
    StoryStructureError,
    UnsafeContentError,
    VideoConfigurationError,
    VideoGenerationError,
)
from sparkstory.entities.grounding import GroundedFact, StoryGrounding
from sparkstory.entities.illustration import ArtItem, ArtStatus, StoryArt
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
from sparkstory.mcp.tools import destinations as destinations_module
from sparkstory.mcp.tools import pdf as pdf_module
from sparkstory.mcp.tools.pdf import render_pdf_beside
from sparkstory.mcp.tools.write_story import write_story_tool
from sparkstory.memory.extract import ExtractedFact, ExtractedMemories
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
        """Forced rather than left unfalsified: a check that has never rejected
        anything is unfalsified, not proven."""
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
        self,
        monkeypatch: pytest.MonkeyPatch,
        brief: StoryBrief,
        outline: StoryOutline,
        tmp_path: Path,
    ) -> None:
        def raise_missing_key(*_: Any, **__: Any) -> None:
            raise MissingAPIKeyError(
                "Model 'gemini-3.5-flash' requires GOOGLE_API_KEY, which is not set."
            )

        monkeypatch.setattr(WORKFLOW_FACTORY, raise_missing_key)
        with pytest.raises(ToolError, match="GOOGLE_API_KEY"):
            await write_story_tool(brief, outline, str(tmp_path))

    async def test_structure_errors_are_not_dressed_up_as_config_errors(
        self,
        looping_fakes: Callable[..., dict[type, FakeModel]],
        brief: StoryBrief,
        outline: StoryOutline,
        page_plan: PagePlan,
        tmp_path: Path,
    ) -> None:
        """No operator can fix malformed output by editing .env."""
        fakes = looping_fakes()
        fakes[PagePlan] = FakeModel(PagePlan(pages=page_plan.pages[:-1]))

        with pytest.raises(StoryStructureError):
            await write_story_tool(brief, outline, str(tmp_path))

    async def test_a_mismatched_outline_is_a_client_error_not_a_bug(
        self,
        fakes: dict[type, FakeModel],
        brief: StoryBrief,
        outline: StoryOutline,
        tmp_path: Path,
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
            await write_story_tool(cramped, five_beats, str(tmp_path))


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
        tmp_path: Path,
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
            await write_story_tool(brief, outline, str(tmp_path))
        assert "safety" in str(caught.value).lower()

    async def test_the_comment_reaches_the_client(
        self,
        brief: StoryBrief,
        outline: StoryOutline,
        looping_fakes: Callable[..., dict[type, FakeModel]],
        tmp_path: Path,
    ) -> None:
        """'We could not do it' without saying why leaves the parent unable to
        adjust the brief and try again."""
        looping_fakes(
            prose_verdicts=[
                ProseReviewsOutput(reviews=[_prose_finding(ProseRubric.SAFETY)])
            ]
        )
        with pytest.raises(ToolError) as caught:
            await write_story_tool(brief, outline, str(tmp_path))
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


class TestMemoryIsWrittenAfterTheBook:
    """Memory's write half: only a finished book, and never at the book's expense.

    Patched at ``build_memory_store`` in the write_story module -- a different
    patch target from the outline workflow's, which is the same trap that split
    these two test files in the first place.
    """

    @staticmethod
    def _with_child_id(brief: StoryBrief) -> StoryBrief:
        payload = brief.model_dump()
        payload["child"]["child_id"] = "maryam-5"
        return StoryBrief.model_validate(payload)

    @staticmethod
    def _extracted() -> ExtractedMemories:
        return ExtractedMemories(
            facts=[ExtractedFact(subject="Kit", text="A fox with a white tail.")],
            episode="Maryam sent Kit to the moon.",
        )

    class _RecordingStore:
        def __init__(self) -> None:
            self.saved: list[Any] = []

        def save(self, records: list[Any]) -> None:
            self.saved.extend(records)

    async def test_no_child_id_never_writes(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fakes: dict[type, FakeModel],
        brief: StoryBrief,
        outline: StoryOutline,
    ) -> None:
        """Opt-in, and the overwhelmingly common path."""

        def _fail() -> Any:
            raise AssertionError("memory must not be written without a child_id")

        monkeypatch.setattr(write_story_module, "build_memory_store", _fail)
        await run_story_pipeline(brief, outline)

    async def test_a_finished_book_is_remembered(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fakes: dict[type, FakeModel],
        brief: StoryBrief,
        outline: StoryOutline,
    ) -> None:
        store = self._RecordingStore()
        monkeypatch.setattr(write_story_module, "build_memory_store", lambda: store)
        fakes[ExtractedMemories] = FakeModel(self._extracted())

        await run_story_pipeline(self._with_child_id(brief), outline)

        texts = [r.text for r in store.saved]
        assert "A fox with a white tail." in texts
        assert "Maryam sent Kit to the moon." in texts

    async def test_a_failed_write_still_returns_the_book(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fakes: dict[type, FakeModel],
        brief: StoryBrief,
        outline: StoryOutline,
    ) -> None:
        """Fail open. The parent asked for a book, not for a side effect."""

        def _boom() -> Any:
            raise RuntimeError("postgres is down")

        monkeypatch.setattr(write_story_module, "build_memory_store", _boom)
        fakes[ExtractedMemories] = FakeModel(self._extracted())

        story = await run_story_pipeline(self._with_child_id(brief), outline)
        assert story.pages, "the book must survive a memory failure"

    async def test_an_unsafe_book_is_never_remembered(
        self,
        monkeypatch: pytest.MonkeyPatch,
        looping_fakes: Callable[..., dict[type, FakeModel]],
        brief: StoryBrief,
        outline: StoryOutline,
    ) -> None:
        """A book that fails the safety gate must leave no trace.

        It never reaches the write at all: the gate raises inside the entrypoint,
        and `remember_story` is called after the pipeline returns.
        """
        store = self._RecordingStore()
        monkeypatch.setattr(write_story_module, "build_memory_store", lambda: store)
        fakes = looping_fakes(
            [
                ProseReviewsOutput(
                    reviews=[
                        ProseReview(
                            rubric=ProseRubric.SAFETY,
                            page_number=1,
                            comment="Unsafe content on this page.",
                        )
                    ]
                )
            ]
        )
        fakes[ExtractedMemories] = FakeModel(self._extracted())

        with pytest.raises(UnsafeContentError):
            await run_story_pipeline(self._with_child_id(brief), outline)

        assert store.saved == [], "an unsafe book must leave nothing behind"


class TestVideoErrorClassification:
    """Retry classification for the two newest exceptions.

    ``default_retry_on`` returns True for every type it does not recognise, which
    is all of ours -- so a class that is not classified here silently gets three
    attempts. Both are asserted rather than assumed, exactly as the image and
    audio pairs are.
    """

    def test_a_missing_ffmpeg_is_not_retried(self) -> None:
        """Retrying cannot make a binary appear.

        The same shape as the missing GOOGLE_API_KEY that was retried three
        times, printing three tracebacks for a one-line fix.
        """
        assert _retry_on(VideoConfigurationError("ffmpeg is not installed")) is False

    def test_a_clip_failure_is_retried(self) -> None:
        """A killed subprocess or a full disk is transient, like a 503."""
        assert _retry_on(VideoGenerationError("ffmpeg exited 1")) is True

    def test_video_configuration_error_is_a_configuration_error(self) -> None:
        """The tool layer translates only ConfigurationError into a message
        naming what to fix, and inheriting picks up the retry exclusion."""
        assert issubclass(VideoConfigurationError, ConfigurationError)

    def test_video_generation_error_is_a_sibling_not_a_child(self) -> None:
        """No operator fixes a broken encode by editing .env."""
        assert issubclass(VideoGenerationError, SparkStoryError)
        assert not issubclass(VideoGenerationError, ConfigurationError)


class TestTheBookIsSavedToDisk:
    """``write_story`` writes the finished book, and says where it went.

    Before this, the prose was the one artifact that existed only as a tool
    result: a client reported "your story is ready" with no path to give, and
    closing the session lost the book. The two media tools already took an
    ``output_directory`` and wrote files, so the surprising part was that the
    book itself did not.
    """

    async def test_the_story_lands_in_the_directory(
        self,
        fakes: dict[type, FakeModel],
        brief: StoryBrief,
        outline: StoryOutline,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(destinations_module, "_OUTPUT_ROOT", tmp_path)

        story = await write_story_tool(brief, outline, "a-book")

        saved = tmp_path / "a-book" / "story.json"
        assert saved.is_file()
        # The directory, not the file: the folder now holds two artifacts, and a
        # client sends this same string back as `illustrate_story`'s
        # `output_directory` so a book and its pictures stay together.
        assert story.saved_to == str(tmp_path / "a-book")

    async def test_what_was_written_is_what_was_returned(
        self,
        fakes: dict[type, FakeModel],
        brief: StoryBrief,
        outline: StoryOutline,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A file that does not round-trip is worse than no file: it looks like
        a book until something tries to read it."""
        monkeypatch.setattr(destinations_module, "_OUTPUT_ROOT", tmp_path)

        story = await write_story_tool(brief, outline, "a-book")

        reloaded = Story.model_validate_json(
            (tmp_path / "a-book" / "story.json").read_text()
        )
        assert reloaded.pages == story.pages
        assert reloaded.outline == story.outline

    async def test_a_missing_directory_is_created(
        self,
        fakes: dict[type, FakeModel],
        brief: StoryBrief,
        outline: StoryOutline,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The caller is an LLM client naming a directory per book, so requiring
        it to exist already would fail on the common path, not an edge case."""
        monkeypatch.setattr(destinations_module, "_OUTPUT_ROOT", tmp_path)

        story = await write_story_tool(brief, outline, "run-1/book")

        assert (tmp_path / "run-1" / "book" / "story.json").is_file()
        assert story.saved_to is not None

    async def test_an_unwritable_directory_is_a_client_error(
        self,
        fakes: dict[type, FakeModel],
        brief: StoryBrief,
        outline: StoryOutline,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The caller chose the path and can choose a better one -- the same
        test this tool applies to a mismatched outline. It must arrive as a
        sentence, because the book has already been paid for by then."""
        monkeypatch.setattr(destinations_module, "_OUTPUT_ROOT", tmp_path)
        blocker = tmp_path / "story-output"
        blocker.write_text("not a directory")

        with pytest.raises(ToolError):
            await write_story_tool(brief, outline, "story-output")


class TestTheDestinationIsConfined:
    """``output_directory`` is chosen by an LLM client, so it is a name inside
    ``outputs/`` rather than a path anywhere on disk.

    Observed live: told to use ``outputs/<name>``, the model passed
    ``tara_star_river`` and the book landed in the repo root. The prompt asks;
    it cannot enforce. This is the same argument that made ``ChildId`` a type
    rather than a sanitising call in the store -- when the caller is an agent,
    the guard belongs in code.
    """

    async def test_a_bare_name_lands_under_outputs(
        self,
        fakes: dict[type, FakeModel],
        brief: StoryBrief,
        outline: StoryOutline,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(destinations_module, "_OUTPUT_ROOT", tmp_path)

        story = await write_story_tool(brief, outline, "tara-star-river")

        assert (tmp_path / "tara-star-river" / "story.json").is_file()
        assert story.saved_to is not None
        assert "tara-star-river" in story.saved_to

    async def test_an_outputs_prefix_is_not_doubled(
        self,
        fakes: dict[type, FakeModel],
        brief: StoryBrief,
        outline: StoryOutline,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A client that follows the prompt writes `outputs/<name>`; one that
        does not writes `<name>`. Both must reach the same directory, or the
        obedient client is the one that gets a surprising path."""
        monkeypatch.setattr(destinations_module, "_OUTPUT_ROOT", tmp_path)

        await write_story_tool(brief, outline, "outputs/tara-star-river")

        assert (tmp_path / "tara-star-river" / "story.json").is_file()

    @pytest.mark.parametrize(
        "escape", ["../elsewhere", "/etc/sparkstory", "book/../../escape"]
    )
    async def test_an_escaping_path_is_refused(
        self,
        fakes: dict[type, FakeModel],
        brief: StoryBrief,
        outline: StoryOutline,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        escape: str,
    ) -> None:
        """Refused rather than silently rewritten: a client that asked for a
        path it cannot have should be told, not quietly redirected."""
        monkeypatch.setattr(destinations_module, "_OUTPUT_ROOT", tmp_path)

        with pytest.raises(ToolError):
            await write_story_tool(brief, outline, escape)


def _one_pixel_jpeg() -> bytes:
    """A real, decodable JPEG.

    `conftest`'s image fixture writes a plausible JPEG *header* and no image,
    which is right for asserting a path was recorded and useless here: an
    undecodable file takes `_draw_illustration`'s exception path, leaves the
    frame blank, and produces exactly the text-only PDF this test is trying to
    tell apart from the illustrated one. Rule 33's shape -- a fake that is wrong
    in a plausible direction passes the weaker assertion.

    Encoded rather than pasted as a base64 constant, because a hand-split
    constant has silently lost bytes in this repository before.
    """
    from io import BytesIO

    from PIL import Image

    buffer = BytesIO()
    Image.new("RGB", (64, 48), (200, 120, 60)).save(buffer, format="JPEG")
    return buffer.getvalue()


def _pdf_pages(raw: bytes) -> bytes:
    """A PDF's content, minus the parts that differ between identical renders.

    reportlab stamps a creation date and a document id derived from the path and
    the clock, so two renders of the same story never compare equal byte for
    byte. Everything before the trailer is the drawn content.
    """
    return raw.split(b"trailer")[0]


class TestTheBookIsAlsoRenderedAsAPdf:
    """``write_story`` writes ``story.pdf`` beside ``story.json``.

    The JSON is what every other stage reads -- illustration, narration, the
    evals, ``scripts/build_pdf.py``. The PDF is the one artifact a *parent* can
    open, and before this it existed only for runs made through
    ``scripts/write_one_story.py``: a book made through MCP was a file no reader
    could read.
    """

    async def test_a_pdf_lands_beside_the_json(
        self,
        fakes: dict[type, FakeModel],
        brief: StoryBrief,
        outline: StoryOutline,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(destinations_module, "_OUTPUT_ROOT", tmp_path)

        story = await write_story_tool(brief, outline, "a-book")

        pdf = tmp_path / "a-book" / "story.pdf"
        assert pdf.is_file()
        assert story.pdf_saved_to == str(pdf)
        # Asserting the magic bytes rather than only the path, because an empty
        # or truncated file would satisfy `is_file` and satisfy nothing else --
        # the same reason the JSON has a round-trip test rather than an
        # existence check.
        assert pdf.read_bytes().startswith(b"%PDF")

    async def test_illustrating_re_renders_the_pdf_with_the_pictures_in_it(
        self,
        story: Story,
        tmp_path: Path,
    ) -> None:
        """The illustrated PDF, and the reason this is not covered by the renderer's
        own tests.

        `render_pdf` has always accepted a `StoryArt` and placed its images, and
        `scripts/write_one_story.py` has always passed one. What did not work was
        the MCP path: `write_story` renders before any picture exists and
        `illustrate_story` had no re-render, so a book made through the tools got
        a text-only PDF for ever and the images sat beside it unused. A real run
        produced exactly that -- six JPEGs, a `story.json`, and no illustrated
        book -- and every test passed throughout, because each half worked.

        Asserted by *size against the text-only render of the same story*, not by
        `is_file` or the `%PDF` magic bytes. Both of those pass on the text-only
        book, so they could not fail if the art were dropped -- which is the one
        thing this test exists to catch.
        """
        image = tmp_path / "page-01.jpg"
        image.write_bytes(_one_pixel_jpeg())

        art = StoryArt(
            style_bible="soft watercolour",
            pages=[
                ArtItem(
                    key=str(story.pages[0].page_number),
                    status=ArtStatus.CONDITIONED,
                    path=image,
                    detail="conditioned on: Kim",
                )
            ],
        )

        plain_dir = tmp_path / "plain"
        plain_dir.mkdir()
        illustrated_dir = tmp_path / "illustrated"
        illustrated_dir.mkdir()

        assert render_pdf_beside(story, plain_dir) is not None
        result = render_pdf_beside(story, illustrated_dir, art)

        assert result == str(illustrated_dir / "story.pdf")
        plain = (plain_dir / "story.pdf").stat().st_size
        illustrated = (illustrated_dir / "story.pdf").stat().st_size
        assert illustrated > plain, (
            f"the illustrated PDF ({illustrated} bytes) is no larger than the "
            f"text-only one ({plain}), so the image was not embedded"
        )

    async def test_an_all_failed_art_run_renders_the_text_only_book(
        self,
        story: Story,
        tmp_path: Path,
    ) -> None:
        """A run where every image failed must render byte-for-byte as text-only.

        `illustrate_story` passes its `StoryArt` unconditionally rather than
        checking whether anything drew, on the strength of `render_pdf`'s
        documented promise that `None` and an all-failed `StoryArt` behave
        identically. That promise is what makes the unconditional call safe, so
        it is asserted here rather than trusted -- and a byte comparison is the
        only assertion that can fail if a blank frame ever starts drawing
        something.
        """
        art = StoryArt(
            style_bible="soft watercolour",
            pages=[
                ArtItem(
                    key=str(page.page_number),
                    status=ArtStatus.FAILED,
                    path=None,
                    detail="resource-exhausted",
                )
                for page in story.pages
            ],
        )

        plain_dir = tmp_path / "plain"
        plain_dir.mkdir()
        failed_dir = tmp_path / "failed"
        failed_dir.mkdir()

        render_pdf_beside(story, plain_dir)
        render_pdf_beside(story, failed_dir, art)

        # Excluding the trailing PDF id, which reportlab derives from the file
        # path and the clock, so two renders of identical content differ there.
        assert _pdf_pages((plain_dir / "story.pdf").read_bytes()) == _pdf_pages(
            (failed_dir / "story.pdf").read_bytes()
        )

    async def test_a_failed_pdf_does_not_lose_the_book(
        self,
        fakes: dict[type, FakeModel],
        brief: StoryBrief,
        outline: StoryOutline,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The one soft failure in this tool, and the reason it is soft: the
        book is already written and saved by the time a PDF can fail, and
        ``scripts/build_pdf.py`` rebuilds the PDF from that JSON alone. Raising
        here would discard a correct book over a rendering.
        """
        monkeypatch.setattr(destinations_module, "_OUTPUT_ROOT", tmp_path)

        def refuse_to_render(*_: object, **__: object) -> None:
            raise StoryStructureError("page 3 does not fit its frame")

        monkeypatch.setattr(pdf_module, "render_pdf", refuse_to_render)

        story = await write_story_tool(brief, outline, "a-book")

        assert (tmp_path / "a-book" / "story.json").is_file()
        assert story.saved_to == str(tmp_path / "a-book")
        # The field is what a client acts on. Without it, a client reading only
        # `saved_to` would tell a parent the PDF is in that folder, and be
        # wrong -- which is the failure the field exists to prevent.
        assert story.pdf_saved_to is None

    async def test_a_failed_pdf_is_logged_loudly(
        self,
        fakes: dict[type, FakeModel],
        brief: StoryBrief,
        outline: StoryOutline,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """An absent field says *that* the PDF is missing; only the log says
        why. A soft failure with no log is the shape of finding CC, where a
        stage failed open and the run looked completely normal.
        """
        monkeypatch.setattr(destinations_module, "_OUTPUT_ROOT", tmp_path)

        def refuse_to_render(*_: object, **__: object) -> None:
            raise OSError("no space left on device")

        monkeypatch.setattr(pdf_module, "render_pdf", refuse_to_render)

        with caplog.at_level(logging.ERROR):
            await write_story_tool(brief, outline, "a-book")

        assert "no space left on device" in caplog.text
