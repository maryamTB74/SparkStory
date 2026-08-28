"""The outline pipeline: plan, critique, revise, keep the best.

Moved out of ``test_workflow.py`` when planning became ``plan_story``'s job
rather than a stage inside ``write_story``. Faked by intercepting
``get_chat_model`` in the outline workflow module -- a *different* patch target
from the story workflow's, which is exactly why the split needed its own file.
A patch aimed at the wrong module does not fail; it silently reaches a real
model factory, and the test fails for a reason that looks nothing like the cause.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from sparkstory.config import settings
from sparkstory.entities.grounding import GroundedFact, StoryGrounding
from sparkstory.entities.reviews import (
    OutlineReview,
    OutlineReviewsOutput,
    OutlineRubric,
)
from sparkstory.entities.stories import StoryBrief, StoryOutline
from sparkstory.mcp.tools.plan_story import plan_story_tool
from sparkstory.memory.types import MemoryKind, MemoryRecord
from sparkstory.models.fake_model import FakeModel
from sparkstory.retrieval.chunks import Chunk, SourceKind
from sparkstory.workflows import plan_outline as _plan_outline
from sparkstory.workflows.plan_outline import run_outline_pipeline

OUTLINE_FACTORY = "sparkstory.workflows.plan_outline.get_chat_model"


@pytest.fixture
def outline_fakes(
    monkeypatch: pytest.MonkeyPatch, outline: StoryOutline
) -> Callable[..., dict[type, FakeModel]]:
    """Build the planner and critic fakes with scripted critic verdicts.

    A factory rather than a fixture: a revision loop's behaviour is a function of
    what the critic says on each pass, and that has to be per-test. FakeModel
    repeats its final response once exhausted, so a one-element verdict list
    models a critic that never approves.
    """

    def build(
        verdicts: list[OutlineReviewsOutput] | None = None,
    ) -> dict[type, FakeModel]:
        by_schema: dict[type, FakeModel] = {
            StoryOutline: FakeModel(outline),
            OutlineReviewsOutput: FakeModel(
                *(verdicts or [OutlineReviewsOutput(reviews=[])])
            ),
        }

        class Dispatcher:
            """Stands in for an unbound model; hands the right fake over once
            bound. Reads `by_schema` at call time, so a test may replace an
            entry after the fixture has built it."""

            def with_structured_output(self, schema: type, **_: Any) -> FakeModel:
                return by_schema[schema].with_structured_output(schema)

        monkeypatch.setattr(OUTLINE_FACTORY, lambda *_a, **_k: Dispatcher())
        return by_schema

    return build


def _finding() -> OutlineReview:
    return OutlineReview(
        rubric=OutlineRubric.PROTAGONIST,
        beat_position=None,
        comment="The want belongs to the fox, not to the child.",
    )


class TestRunOutlinePipeline:
    async def test_returns_an_outline(
        self,
        outline_fakes: Callable[..., dict[type, FakeModel]],
        brief: StoryBrief,
        outline: StoryOutline,
    ) -> None:
        outline_fakes()
        result = await run_outline_pipeline(brief)
        # Compared field by field rather than as a whole object: the returned
        # outline now also carries the run's grounding, so whole-object equality
        # against the fixture would fail on a field this test is not about.
        assert result.model_dump(exclude={"grounding"}) == outline.model_dump(
            exclude={"grounding"}
        )

    async def test_an_approving_critic_costs_one_extra_call(
        self,
        outline_fakes: Callable[..., dict[type, FakeModel]],
        brief: StoryBrief,
    ) -> None:
        """Empty on the first pass means no revision at all: the loop's whole
        cost on a good plan is one critic call."""
        fakes = outline_fakes()
        await run_outline_pipeline(brief)
        assert len(fakes[StoryOutline].calls) == 1
        assert len(fakes[OutlineReviewsOutput].calls) == 1

    async def test_a_finding_triggers_exactly_one_revision(
        self,
        outline_fakes: Callable[..., dict[type, FakeModel]],
        brief: StoryBrief,
    ) -> None:
        fakes = outline_fakes(
            [
                OutlineReviewsOutput(reviews=[_finding()]),
                OutlineReviewsOutput(reviews=[]),
            ]
        )
        await run_outline_pipeline(brief)
        assert len(fakes[StoryOutline].calls) == 2

    async def test_a_critic_that_never_approves_stops_at_the_cap(
        self,
        outline_fakes: Callable[..., dict[type, FakeModel]],
        brief: StoryBrief,
    ) -> None:
        """Hitting the cap returns a plan anyway. It is not an error: a plan the
        critic still dislikes beats no book at all."""
        fakes = outline_fakes([OutlineReviewsOutput(reviews=[_finding()])])
        result = await run_outline_pipeline(brief)
        assert isinstance(result, StoryOutline)
        # One first pass plus max_outline_revisions (default 2).
        assert len(fakes[StoryOutline].calls) == 3

    async def test_the_finding_reaches_the_planner(
        self,
        outline_fakes: Callable[..., dict[type, FakeModel]],
        brief: StoryBrief,
    ) -> None:
        """The test that catches a loop which runs but feeds nothing forward --
        the failure mode that passes every other test in this file."""
        fakes = outline_fakes(
            [
                OutlineReviewsOutput(reviews=[_finding()]),
                OutlineReviewsOutput(reviews=[]),
            ]
        )
        await run_outline_pipeline(brief)
        revision = fakes[StoryOutline].calls[1]
        assert len(revision) == 4
        assert _finding().comment in revision[3].content

    async def test_the_critic_sees_the_revised_outline_not_the_first_draft(
        self,
        outline_fakes: Callable[..., dict[type, FakeModel]],
        brief: StoryBrief,
    ) -> None:
        """Re-reviewing the draft that was already criticised would loop forever
        on a finding that had in fact been fixed."""
        fakes = outline_fakes(
            [
                OutlineReviewsOutput(reviews=[_finding()]),
                OutlineReviewsOutput(reviews=[]),
            ]
        )
        await run_outline_pipeline(brief)
        assert len(fakes[OutlineReviewsOutput].calls) == 2

    async def test_the_loop_returns_the_best_outline_not_the_last(
        self,
        outline_fakes: Callable[..., dict[type, FakeModel]],
        brief: StoryBrief,
        outline: StoryOutline,
    ) -> None:
        """A later revision can be worse and the loop cannot tell -- it
        re-critiques and sees a different set of findings, not a better plan."""
        better = outline.model_copy(update={"title": "The Better Plan"})
        worse = outline.model_copy(update={"title": "The Worse Plan"})
        fakes = outline_fakes(
            [
                OutlineReviewsOutput(reviews=[_finding(), _finding()]),
                OutlineReviewsOutput(reviews=[_finding()]),
                OutlineReviewsOutput(reviews=[_finding(), _finding()]),
            ]
        )
        fakes[StoryOutline] = FakeModel(outline, better, worse)

        result = await run_outline_pipeline(brief)

        assert result.title == "The Better Plan"

    async def test_every_completed_task_is_reported(
        self,
        outline_fakes: Callable[..., dict[type, FakeModel]],
        brief: StoryBrief,
    ) -> None:
        """The debug script numbers artifacts from this callback.

        Also pins the stream filter: ``plan_outline`` returns a ``StoryOutline``
        and so does the entrypoint, so the entrypoint's own result is identified
        by task name rather than by type. Get that wrong and the final outline
        arrives here as if it were a task result.
        """
        outline_fakes()
        seen: list[str] = []
        await run_outline_pipeline(brief, on_task_result=lambda n, _v: seen.append(n))
        assert seen == ["research", "plan_outline", "critique_outline"]


class TestPlanStoryToolRunsTheCritic:
    """The preview is the plan the book is built from, so it must be critiqued.

    Before this, ``plan_story`` was one uncritiqued call and ``write_story``
    planned again -- so a parent approved a plan that was never used. See
    docs/superpowers/specs/2026-07-31-plan-carries-forward-design.md.
    """

    async def test_the_tool_revises_when_the_critic_objects(
        self,
        outline_fakes: Callable[..., dict[type, FakeModel]],
        brief: StoryBrief,
    ) -> None:
        fakes = outline_fakes(
            [
                OutlineReviewsOutput(reviews=[_finding()]),
                OutlineReviewsOutput(reviews=[]),
            ]
        )
        await plan_story_tool(brief)
        assert len(fakes[StoryOutline].calls) == 2, (
            "the preview must revise, or a parent approves an uncritiqued plan"
        )


# --- Research ------------------------------------------------------------
RESEARCH_CONTEXT = "sparkstory.workflows.plan_outline.build_research_context"

# Captured before any fixture can replace it, so the construction tests below
# can call the genuine builder without depending on autouse ordering.
REAL_BUILD_CONTEXT = _plan_outline.build_research_context


class _StubStore:
    """Answers ``get`` and nothing else, which is all provenance filtering uses."""

    def __init__(self, *chunk_ids: str) -> None:
        self._chunks = {
            chunk_id: Chunk(
                chunk_id=chunk_id,
                text="The Moon has no air.",
                title="The Moon",
                source="NASA -- Earth's Moon",
                licence="public domain",
                source_kind=SourceKind.FACT,
            )
            for chunk_id in chunk_ids
        }

    def get(self, chunk_id: str) -> Chunk | None:
        return self._chunks.get(chunk_id)


class _StubResearchAgent:
    def __init__(self, result: Any) -> None:
        self._result = result

    async def ainvoke(self, payload: dict, **_: Any) -> dict:
        if isinstance(self._result, Exception):
            raise self._result
        return {"structured_response": self._result, "messages": []}


def _grounded_fact(chunk_id: str = "moon#1") -> GroundedFact:
    return GroundedFact(
        claim="The Moon has no air.",
        story_note="Nothing outdoors can flutter or make a sound.",
        source="NASA -- Earth's Moon",
        chunk_id=chunk_id,
    )


@pytest.fixture(autouse=True)
def _stub_research(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Stub the research seam for *every* test in this module.

    Autouse, and that is a correctness requirement rather than convenience. Research
    now runs before planning on the default path, so a test that only faked
    ``get_chat_model`` would reach the real ``build_research_context`` -- loading
    embedding weights from disk and attempting a live provider call, then failing
    open so the test still *looked* fine. One such test took 25 seconds and quietly
    broke the offline guarantee this suite has held from the start.

    Patched once, reading a mutable holder, so ``fake_research`` can change the
    result without a second ``setattr`` whose ordering against this one would depend
    on fixture resolution order.
    """
    holder: dict[str, Any] = {
        "result": StoryGrounding(),
        "chunks": ("moon#1",),
        # None means the web tool was never built, which is the default and what
        # every test in this module should see unless it says otherwise.
        "ledger": None,
    }
    monkeypatch.setattr(
        RESEARCH_CONTEXT,
        lambda: (
            _StubResearchAgent(holder["result"]),
            _StubStore(*holder["chunks"]),
            holder["ledger"],
        ),
    )
    return holder


@pytest.fixture
def fake_research(_stub_research: dict[str, Any]) -> Callable[..., None]:
    """Override what the stubbed researcher returns for one test."""

    def build(result: Any, known_chunks: tuple[str, ...] = ("moon#1",)) -> None:
        _stub_research["result"] = result
        _stub_research["chunks"] = known_chunks

    return build


class TestResearchRunsBeforePlanning:
    async def test_grounding_is_reported_as_a_completed_task(
        self,
        outline_fakes: Callable[..., dict[type, FakeModel]],
        fake_research: Callable[..., None],
        brief: StoryBrief,
    ) -> None:
        """So the debug script writes research-1.json with no extra code."""
        outline_fakes()
        fake_research(StoryGrounding(facts=[_grounded_fact()]))

        seen: dict[str, Any] = {}
        await run_outline_pipeline(
            brief, on_task_result=lambda n, v: seen.setdefault(n, v)
        )

        assert isinstance(seen["research"], StoryGrounding)
        assert seen["research"].facts[0].chunk_id == "moon#1"

    async def test_research_runs_first(
        self,
        outline_fakes: Callable[..., dict[type, FakeModel]],
        fake_research: Callable[..., None],
        brief: StoryBrief,
    ) -> None:
        """Order is the point of the whole session: a constraint discovered after
        the plan exists cannot shape it."""
        outline_fakes()
        fake_research(StoryGrounding())

        order: list[str] = []
        await run_outline_pipeline(brief, on_task_result=lambda n, _v: order.append(n))
        assert order[0] == "research"

    async def test_unprovenanced_facts_are_dropped_before_planning(
        self,
        outline_fakes: Callable[..., dict[type, FakeModel]],
        fake_research: Callable[..., None],
        brief: StoryBrief,
    ) -> None:
        """An invented chunk id never reaches the planner, so a fabricated fact
        cannot shape a story even once."""
        outline_fakes()
        fake_research(
            StoryGrounding(
                facts=[_grounded_fact("moon#1"), _grounded_fact("atlantis#9")]
            )
        )

        seen: dict[str, Any] = {}
        await run_outline_pipeline(
            brief, on_task_result=lambda n, v: seen.setdefault(n, v)
        )
        assert [f.chunk_id for f in seen["research"].facts] == ["moon#1"]


class TestResearchCanBeSwitchedOff:
    async def test_zero_steps_skips_research_entirely(
        self,
        outline_fakes: Callable[..., dict[type, FakeModel]],
        fake_research: Callable[..., None],
        brief: StoryBrief,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """MAX_RESEARCH_STEPS=0 is what makes the grounded/ungrounded A/B possible
        with no code change -- and it must skip the step, not run it with no
        budget."""
        outline_fakes()
        fake_research(StoryGrounding(facts=[_grounded_fact()]))
        monkeypatch.setattr(settings, "max_research_steps", 0)

        seen: list[str] = []
        await run_outline_pipeline(brief, on_task_result=lambda n, _v: seen.append(n))
        assert "research" not in seen

    async def test_zero_steps_still_produces_an_outline(
        self,
        outline_fakes: Callable[..., dict[type, FakeModel]],
        brief: StoryBrief,
        outline: StoryOutline,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        outline_fakes()
        monkeypatch.setattr(settings, "max_research_steps", 0)
        assert await run_outline_pipeline(brief) == outline


class TestGroundingIsAttachedToTheReturnedOutline:
    """The point of the whole feature: grounding survives ``plan_story``.

    It used to be computed, planned from, and dropped when the pipeline returned a
    bare outline -- so the Writer had never once seen a fact, and a craft device
    could only ever be *described* in a beat summary -- a planner told to repeat a
    phrase wrote "they repeat the phrase" instead of repeating one.

    Attached to the outline rather than returned beside it, so the pair cannot come
    apart. ``world_rules`` lives on the brief, so a genre change re-runs retrieval
    and re-frames the result: a grounding paired with the wrong brief is not stale,
    it is wrong.
    """

    async def test_the_returned_outline_carries_the_runs_grounding(
        self,
        outline_fakes: Callable[..., dict[type, FakeModel]],
        fake_research: Callable[..., None],
        brief: StoryBrief,
    ) -> None:
        outline_fakes()
        fake_research(StoryGrounding(facts=[_grounded_fact()]))

        result = await run_outline_pipeline(brief)

        assert result.grounding is not None
        assert result.grounding.facts[0].chunk_id == "moon#1"

    async def test_no_research_means_no_grounding(
        self,
        outline_fakes: Callable[..., dict[type, FakeModel]],
        brief: StoryBrief,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The control arm of the A/B has to stay representable, and an outline
        with no grounding has to stay usable."""
        outline_fakes()
        monkeypatch.setattr(settings, "max_research_steps", 0)

        result = await run_outline_pipeline(brief)

        assert result.grounding is None

    async def test_empty_grounding_is_attached_as_empty_not_dropped(
        self,
        outline_fakes: Callable[..., dict[type, FakeModel]],
        fake_research: Callable[..., None],
        brief: StoryBrief,
    ) -> None:
        """Research ran and found nothing -- a correct and common answer for a
        premise with no factual spine.

        Distinguishable from "research never ran", which is why the fact count has
        to be checked before comparing two runs: a run that retrieved nothing
        renders identically in both world-rule modes, so a comparison against it is
        vacuous. That mistake has been made twice on live runs.
        """
        outline_fakes()
        fake_research(StoryGrounding(facts=[]))

        result = await run_outline_pipeline(brief)

        assert result.grounding is not None
        assert result.grounding.facts == []


class TestResearchFailsOpen:
    async def test_a_broken_researcher_still_yields_a_book(
        self,
        outline_fakes: Callable[..., dict[type, FakeModel]],
        fake_research: Callable[..., None],
        brief: StoryBrief,
        outline: StoryOutline,
    ) -> None:
        """The decision recorded in the spec: fail closed on harm, open on
        enrichment. Grounding is enrichment, so a provider failure here costs the
        facts, not the story."""
        outline_fakes()
        fake_research(RuntimeError("provider exploded"))
        assert await run_outline_pipeline(brief) == outline

    async def test_a_missing_index_still_yields_a_book(
        self,
        outline_fakes: Callable[..., dict[type, FakeModel]],
        fake_research: Callable[..., None],
        brief: StoryBrief,
        outline: StoryOutline,
    ) -> None:
        """Nobody has run the ingestion script. Everything is unprovenanced, so
        everything is dropped, and the run continues."""
        outline_fakes()
        fake_research(StoryGrounding(facts=[_grounded_fact()]), known_chunks=())

        result = await run_outline_pipeline(brief)

        assert result.model_dump(exclude={"grounding"}) == outline.model_dump(
            exclude={"grounding"}
        )
        # The dropped fact leaves *empty* grounding attached, not absent grounding.
        # "We looked and the corpus could not vouch for anything" is a different
        # state from "we never looked", and the Writer must be told the first
        # rather than left to infer it.
        assert result.grounding is not None
        assert result.grounding.facts == []


class TestWebLedgerConstruction:
    """Whether the web half is built at all, asserted on construction.

    Deliberately not "does the tool refuse when disabled". A tool that exists and
    declines has already constructed a client and read a key, which is exactly
    what MAX_WEB_SEARCHES=0 is supposed to prevent -- and what keeps this suite
    offline. The question is whether it was *built*.

    Uses `REAL_BUILD_CONTEXT`, captured at import before the autouse stub can
    replace it, rather than trying to unpatch -- which would depend on fixture
    resolution order between two autouse fixtures and fail confusingly when it
    lost. It never invokes anything: it inspects what came back, and both the
    model factory and the agent builder are replaced, so no provider is reached
    and no weights are loaded.
    """

    def test_no_ledger_and_no_web_tool_by_default(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from sparkstory.config import settings as live
        from sparkstory.workflows import plan_outline as module

        monkeypatch.setattr(live, "max_web_searches", 0)
        # These tests are about the ledger, not about retrieval, so the store
        # is stubbed rather than pointed at a real database. Before the move to
        # Postgres the equivalent was `knowledge_root = tmp_path`, i.e. an empty
        # index; `build_store` now needs DATABASE_URL, which a runner has not got.
        monkeypatch.setattr(module, "build_store", lambda *a, **k: object())
        # The embedder joined this list when EMBEDDING_MODEL moved to the hosted
        # Gemini entry: retrieval used to need no credential, so it was the one
        # seam these tests could leave real. `build_research_context` constructs
        # it eagerly, so on a keyless machine it now raises before the assertion
        # below is reached. Stubbed for the same reason as the three seams
        # beside it -- nothing here embeds anything.
        monkeypatch.setattr(module, "get_embedder", lambda *a, **k: object())
        monkeypatch.setattr(module, "get_chat_model", lambda *a, **k: object())
        monkeypatch.setattr(
            module, "build_researcher_agent", lambda model, tools: tools
        )

        _tools, _store, ledger = REAL_BUILD_CONTEXT()
        assert ledger is None
        assert "search_web" not in {t.name for t in _tools}

    def test_a_ledger_and_the_tool_appear_when_enabled(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from sparkstory.config import settings as live
        from sparkstory.workflows import plan_outline as module

        monkeypatch.setattr(live, "max_web_searches", 3)
        # These tests are about the ledger, not about retrieval, so the store
        # is stubbed rather than pointed at a real database. Before the move to
        # Postgres the equivalent was `knowledge_root = tmp_path`, i.e. an empty
        # index; `build_store` now needs DATABASE_URL, which a runner has not got.
        monkeypatch.setattr(module, "build_store", lambda *a, **k: object())
        monkeypatch.setattr(module, "get_embedder", lambda *a, **k: object())
        monkeypatch.setattr(module, "get_chat_model", lambda *a, **k: object())
        monkeypatch.setattr(
            module, "build_researcher_agent", lambda model, tools: tools
        )

        _tools, _store, ledger = REAL_BUILD_CONTEXT()
        assert ledger is not None
        assert "search_web" in {t.name for t in _tools}

    def test_each_run_gets_its_own_ledger(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Run-scoped ids. A shared ledger would let one run's web:1 resolve
        against another run's page -- wrong in the most plausible way."""
        from sparkstory.config import settings as live
        from sparkstory.workflows import plan_outline as module

        monkeypatch.setattr(live, "max_web_searches", 3)
        # These tests are about the ledger, not about retrieval, so the store
        # is stubbed rather than pointed at a real database. Before the move to
        # Postgres the equivalent was `knowledge_root = tmp_path`, i.e. an empty
        # index; `build_store` now needs DATABASE_URL, which a runner has not got.
        monkeypatch.setattr(module, "build_store", lambda *a, **k: object())
        monkeypatch.setattr(module, "get_embedder", lambda *a, **k: object())
        monkeypatch.setattr(module, "get_chat_model", lambda *a, **k: object())
        monkeypatch.setattr(
            module, "build_researcher_agent", lambda model, tools: tools
        )

        _a, _b, first = REAL_BUILD_CONTEXT()
        _c, _d, second = REAL_BUILD_CONTEXT()
        assert first is not second


class TestMemoryReachesThePlanner:
    """Memory's read half: fetched before planning, rendered into the prompt.

    Patched at ``build_memory_store`` in this module -- the same seam shape as
    ``build_research_context``, and for the same reason: a test that fakes only the
    model would reach a real database here.
    """

    @staticmethod
    def _remembering(*records: MemoryRecord) -> Any:
        class _Store:
            def fetch(self, child_id: str, kind: Any = None) -> list[MemoryRecord]:
                return list(records)

        return _Store()

    @staticmethod
    def _fact(text: str, subject: str = "Kit") -> MemoryRecord:
        return MemoryRecord(
            child_id="maryam-5",
            kind=MemoryKind.SEMANTIC,
            text=text,
            subject=subject,
            source_request_id="req-0",
        )

    def _with_child_id(self, brief: StoryBrief) -> StoryBrief:
        payload = brief.model_dump()
        payload["child"]["child_id"] = "maryam-5"
        return StoryBrief.model_validate(payload)

    async def test_no_child_id_never_touches_the_store(
        self,
        monkeypatch: pytest.MonkeyPatch,
        outline_fakes: Callable[..., dict[type, FakeModel]],
        brief: StoryBrief,
    ) -> None:
        """Memory is opt-in, and 612 existing tests supply no child_id."""

        def _fail() -> Any:
            raise AssertionError("memory must not be read without a child_id")

        monkeypatch.setattr(_plan_outline, "build_memory_store", _fail)
        outline_fakes()
        await run_outline_pipeline(brief)

    async def test_a_remembered_fact_reaches_the_planner_prompt(
        self,
        monkeypatch: pytest.MonkeyPatch,
        outline_fakes: Callable[..., dict[type, FakeModel]],
        brief: StoryBrief,
    ) -> None:
        monkeypatch.setattr(
            _plan_outline,
            "build_memory_store",
            lambda: self._remembering(self._fact("A fox with a white-tipped tail.")),
        )
        fakes = outline_fakes()
        await run_outline_pipeline(self._with_child_id(brief))

        sent = str(fakes[StoryOutline].calls[0])
        assert "white-tipped tail" in sent

    async def test_an_unreachable_store_still_produces_a_plan(
        self,
        monkeypatch: pytest.MonkeyPatch,
        outline_fakes: Callable[..., dict[type, FakeModel]],
        brief: StoryBrief,
    ) -> None:
        """Fails open: memory is enrichment, and no book at all is the failure."""

        def _boom() -> Any:
            raise RuntimeError("postgres is down")

        monkeypatch.setattr(_plan_outline, "build_memory_store", _boom)
        outline_fakes()
        result = await run_outline_pipeline(self._with_child_id(brief))
        assert result.title

    async def test_a_disagreeing_plan_reports_a_conflict(
        self,
        monkeypatch: pytest.MonkeyPatch,
        outline_fakes: Callable[..., dict[type, FakeModel]],
        brief: StoryBrief,
        outline: StoryOutline,
    ) -> None:
        """The parent meets the disagreement at the approval point."""
        planned = outline.characters[0]
        monkeypatch.setattr(
            _plan_outline,
            "build_memory_store",
            lambda: self._remembering(
                self._fact("Something entirely different.", subject=planned.name)
            ),
        )
        outline_fakes()
        result = await run_outline_pipeline(self._with_child_id(brief))

        assert len(result.memory_conflicts) == 1
        assert result.memory_conflicts[0].subject == planned.name

    async def test_an_agreeing_plan_reports_nothing(
        self,
        monkeypatch: pytest.MonkeyPatch,
        outline_fakes: Callable[..., dict[type, FakeModel]],
        brief: StoryBrief,
        outline: StoryOutline,
    ) -> None:
        """The conflict path needs its negative direction asserted too,
        or 'a conflict was reported' proves only that something was reported."""
        planned = outline.characters[0]
        monkeypatch.setattr(
            _plan_outline,
            "build_memory_store",
            lambda: self._remembering(
                self._fact(planned.description, subject=planned.name)
            ),
        )
        outline_fakes()
        result = await run_outline_pipeline(self._with_child_id(brief))

        assert result.memory_conflicts == []
