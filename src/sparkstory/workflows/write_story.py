"""The story engine: brief -> outline -> page plan -> prose.

Built on LangGraph's **functional API**,  each stage is an ``async def``
decorated ``@task``, awaited inside one ``@entrypoint``.

Why the functional API rather than ``StateGraph``. so the review loop and
media generation of later sessions can follow its code.
And with no shared state object there is no state schema and no
merge semantics: values flow as ordinary arguments and returns, which for a linear
pipeline is simply readable Python. Verified against the LangGraph docs:
``entrypoint()`` needs no checkpointer and no ``config``; ``thread_id`` becomes
required only once a checkpointer exists, and ``interrupt()`` -- the
human-in-the-loop step of a later session -- requires one. The cost accepted: no
auto-generated diagram, which ``StateGraph.get_graph().draw_mermaid()`` would give.

Tasks are thin adapters. They choose a model, run a node, and check the result.
Nodes stay free of LangGraph and remain testable without it, which is what keeps
the framework replaceable rather than load-bearing.
"""

from collections.abc import Callable
from typing import Any
from uuid import uuid4

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.func import entrypoint, task

from sparkstory.config import settings
from sparkstory.entities.exceptions import (
    StoryStructureError,
    UnsafeContentError,
)
from sparkstory.entities.reviews import ProseReviews, ProseRubric
from sparkstory.entities.stories import (
    PagePlan,
    Story,
    StoryBrief,
    StoryOutline,
    StoryProse,
)
from sparkstory.models.get_model import get_chat_model
from sparkstory.nodes.plot_planner import PlotPlannerNode
from sparkstory.nodes.prose_critic import ProseCriticNode
from sparkstory.nodes.writer import WriterNode
from sparkstory.observability.tracing import build_handler
from sparkstory.retrieval.embed import get_embedder
from sparkstory.retrieval.pg_store import build_store
from sparkstory.retrieval.provenance import drop_unprovenanced
from sparkstory.utils.logging_utils import get_logger
from sparkstory.workflows.retries import RETRY_POLICY
from sparkstory.workflows.reviews import (
    deterministic_prose_reviews,
    draft_score,
    drop_unroutable_prose_reviews,
    format_pacing_report,
)
from sparkstory.workflows.types import StoryWorkflowInput
from sparkstory.workflows.validation import (
    validate_outline,
    validate_page_plan,
    validate_prose,
)

logger = get_logger(__name__)


@task(retry_policy=RETRY_POLICY)
async def plan_pages(
    request_id: str, brief: StoryBrief, outline: StoryOutline
) -> PagePlan:
    """Stage 2: lay the beats out across pages."""
    logger.info("[%s] stage=plot model=%s", request_id, settings.plot_model)
    node = PlotPlannerNode(
        model=get_chat_model(settings.plot_model),
        brief=brief,
        outline=outline,
    )
    plan = await node.ainvoke()
    validate_page_plan(brief, outline, plan)
    # After validation, not before: reporting the pacing of a plan that is about
    # to be rejected as structurally broken is noise.
    logger.info("[%s] %s", request_id, format_pacing_report(outline, plan))
    return plan


@task(retry_policy=RETRY_POLICY)
async def write_prose(
    request_id: str,
    brief: StoryBrief,
    outline: StoryOutline,
    page_plan: PagePlan,
    reviews: ProseReviews | None = None,
) -> StoryProse:
    """Stage 3: write the words, or rewrite them from reviews."""
    logger.info(
        "[%s] stage=write model=%s revision=%s",
        request_id,
        settings.writer_model,
        reviews is not None,
    )
    node = WriterNode(
        model=get_chat_model(settings.writer_model),
        brief=brief,
        outline=outline,
        page_plan=page_plan,
        reviews=reviews,
    )
    prose = await node.ainvoke()
    validate_prose(page_plan, prose)
    return prose


@task(retry_policy=RETRY_POLICY)
async def critique_prose(
    request_id: str, brief: StoryBrief, page_plan: PagePlan, prose: StoryProse
) -> ProseReviews:
    """Judge the finished words, from a critic and from counting.

    Two sources, one list. Counting how many pages open with the same word does
    not need a model -- but a check that only raises cannot fix anything, so the
    counted findings become reviews and merge with the judged ones before the
    Writer ever sees them.
    """
    logger.info(
        "[%s] stage=critique_prose model=%s", request_id, settings.prose_critic_model
    )
    node = ProseCriticNode(
        model=get_chat_model(settings.prose_critic_model),
        brief=brief,
        page_plan=page_plan,
        prose=prose,
        max_reviews=settings.max_reviews_per_pass,
    )
    judged = drop_unroutable_prose_reviews(await node.ainvoke(), page_plan)
    counted = deterministic_prose_reviews(prose, page_plan)
    if counted:
        logger.info("[%s] %d counted finding(s)", request_id, len(counted))
    return ProseReviews(prose=prose, reviews=judged.reviews + counted)


def build_story_workflow(
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> Any:
    """Compile the workflow, optionally with a checkpointer.

    The checkpointer is a parameter, so the session that adds a
    human confirmation step can make runs resumable without restructuring
    anything here. Nothing passes one today: with nothing to interrupt, an
    in-memory checkpointer would only add a store nobody reads.
    """

    @entrypoint(checkpointer=checkpointer)
    async def story_workflow(payload: StoryWorkflowInput) -> Story:
        request_id = payload["request_id"]
        brief = payload["brief"]
        outline = payload["outline"]

        # The outline arrives from the caller, so this validates *someone else's*
        # data -- in the MCP flow, an outline an LLM client threaded through from
        # plan_story. It can be stale, hand-edited or invented. Checked before a
        # single model call is paid for.
        validate_outline(brief, outline)

        # And so does its grounding, which is why it is re-verified rather than
        # trusted. `validate_outline` checks structural fit; this checks *origin*,
        # which is strictly stronger: every chunk_id is resolved against the store
        # and `source` is overwritten from it, so a fabricated `source: "NASA"` is
        # unreachable rather than merely detectable. That is the same move
        # drop_unprovenanced already makes on the planning side, applied at the
        # point the data stops being ours.
        #
        # It DROPS rather than raising, matching drop_unroutable_prose_reviews: a
        # book is better than no book, and a brief whose whole grounding is
        # fabricated simply becomes an ungrounded run.
        #
        # Guarded on `is not None` for a reason beyond tidiness: `build_store`
        # raises ConfigurationError when DATABASE_URL is unset, so verifying
        # unconditionally would make an ungrounded write_story newly require
        # Postgres -- a behaviour change for every caller that never researched.
        #
        # ledger=None deliberately, and this is the feature's accepted scope limit.
        # A WebLedger is per-run and in-memory, so a `web:<n>` id minted during
        # planning cannot be resolved here and its fact is dropped as though
        # invented. Corpus facts only. It is logged rather than silent, because a
        # book quietly losing its grounding with nothing to read afterwards is
        # finding M's failure mode.
        if outline.grounding is not None:
            verified = drop_unprovenanced(
                outline.grounding,
                build_store(
                    settings.database_url,
                    get_embedder(settings.embedding_model),
                    settings.embedding_model,
                ),
            )
            dropped = len(outline.grounding.facts) - len(verified.facts)
            if dropped:
                logger.warning(
                    "[%s] dropped %d unprovenanced fact(s) from the caller's "
                    "grounding; note a web:<n> id cannot be verified here",
                    request_id,
                    dropped,
                )
            outline = outline.model_copy(update={"grounding": verified})

        page_plan = await plan_pages(request_id, brief, outline)
        prose = await write_prose(request_id, brief, outline, page_plan)

        # `+ 1` -- and this is the one place the two loops differ in shape.
        # brown's loop, which the outline loop above copies, ends on an edit that
        # nobody reviewed. Harmless there and harmless for the outline, but the
        # safety gate below has to judge the draft we are actually returning: a
        # gate reading findings from before the final rewrite would destroy books
        # whose safety problem that rewrite had already fixed. So the prose loop
        # runs max_prose_revisions rewrites but one more critique, always ending
        # on a critique. One extra call buys a gate that means what it says.
        reviews = ProseReviews(prose=prose, reviews=[])
        best_prose, best_reviews, best_score = prose, reviews, None
        for attempt in range(settings.max_prose_revisions + 1):
            reviews = await critique_prose(request_id, brief, page_plan, prose)
            score = draft_score(reviews.reviews)
            if best_score is None or score < best_score:
                best_prose, best_reviews, best_score = prose, reviews, score
            if not reviews.reviews:
                logger.info(
                    "[%s] prose approved after %d rewrite(s)", request_id, attempt
                )
                break
            if attempt == settings.max_prose_revisions:
                logger.warning(
                    "[%s] prose still has %d finding(s) after %d rewrite(s)",
                    request_id,
                    len(reviews.reviews),
                    settings.max_prose_revisions,
                )
                break
            logger.info(
                "[%s] rewriting prose: %d finding(s), attempt %d/%d",
                request_id,
                len(reviews.reviews),
                attempt + 1,
                settings.max_prose_revisions,
            )
            prose = await write_prose(request_id, brief, outline, page_plan, reviews)

        # The best draft seen, not the last -- and its own reviews travel with
        # it, so the gate below judges the book actually being returned.
        prose, reviews = best_prose, best_reviews

        # Fail closed, and only on safety. A craft finding that survives gives a
        # flatter book; a safety finding that survives means something the parent
        # asked to keep out of their child's bedtime is still in it, and no book
        # is better than that book.
        unsafe = [r for r in reviews.reviews if r.rubric is ProseRubric.SAFETY]
        if unsafe:
            raise UnsafeContentError(
                "Could not write a story meeting this brief's safety "
                f"constraints. Unresolved after {settings.max_prose_revisions} "
                f"rewrite(s): {'; '.join(r.comment for r in unsafe)}"
            )

        return Story(outline=outline, page_plan=page_plan, pages=prose.pages)

    return story_workflow


#: Compiled once per process. Compiling is pure -- no network, no API key -- so
#: doing it at import is safe, and it keeps a per-request cost out of every call.
STORY_WORKFLOW = build_story_workflow()


async def run_story_pipeline(
    brief: StoryBrief,
    outline: StoryOutline,
    on_task_result: Callable[[str, Any], None] | None = None,
    *,
    request_id: str | None = None,
) -> Story:
    """Write a complete story from a brief and an approved plan.

    Mints the ``request_id`` here rather than inside the workflow, so it survives
    a resume -- see :class:`~sparkstory.workflows.types.StoryWorkflowInput`.

    Args:
        brief: What to write.
        outline: The approved plan to build from. Produced by
            ``run_outline_pipeline``. Never planned here: the whole point is
            that the book matches the outline a parent was shown.
        on_task_result: Called with ``(task_name, result)`` as each ``@task``
            completes, including every loop iteration. The revision loops run
            inside the entrypoint, so the returned ``Story`` shows only the
            drafts that survived -- a run that converged badly is
            indistinguishable from one that passed first time. Optional, and
            nothing on the MCP path passes it: an operator debugging a run wants
            every iteration, a client wants the book.
        request_id: Identifies this run in the logs and, when tracing is on, as
            the Opik thread id. Pass the same value ``run_outline_pipeline``
            received to make one book one thread. The MCP path passes nothing:
            a client's ``write_story`` call is its own run, and may carry an
            outline this server never planned.

    Raises:
        MissingAPIKeyError: a configured model's API key is not set.
        UnknownModelError: a ``*_MODEL`` setting is not a known model id.
        StoryStructureError: the outline does not fit the brief, or an agent's
            output was well-formed but structurally wrong.
        UnsafeContentError: a safety finding survived every rewrite, so no book
            is returned.
    """
    request_id = request_id or str(uuid4())
    logger.info(
        "[%s] writing story: age=%d level=%s tone=%s pages=%d",
        request_id,
        brief.child.age,
        brief.child.reading_level.value,
        brief.tone.value,
        brief.page_count,
    )

    # astream unconditionally rather than branching on whether a callback was
    # given: one code path means the debug script exercises exactly what the MCP
    # tool exercises. `stream_mode="updates"` yields one mapping per completed
    # task, keyed by task name; the entrypoint's own return arrives the same way.
    story: Story | None = None
    # See run_outline_pipeline: one attachment covers every task beneath it, and
    # a None handler is filtered out rather than passed into the callback list.
    tracer = build_handler(request_id, tags=["write_story"])
    async for update in STORY_WORKFLOW.astream(
        StoryWorkflowInput(request_id=request_id, brief=brief, outline=outline),
        stream_mode="updates",
        config={"callbacks": [t for t in [tracer] if t is not None]},
    ):
        for name, value in update.items():
            if isinstance(value, Story):
                story = value
                continue
            if on_task_result is not None:
                on_task_result(name, value)

    if story is None:  # pragma: no cover - the entrypoint always returns a Story
        raise StoryStructureError("The workflow completed without producing a story.")

    logger.info(
        "[%s] finished %r: %d pages",
        request_id,
        story.outline.title,
        len(story.pages),
    )
    return story
