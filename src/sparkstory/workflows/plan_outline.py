"""The outline pipeline: plan the story's structure and revise it until a critic
approves.

Split out of ``write_story.py`` when planning became ``plan_story``'s job. The
reason is a product one, not a tidiness one: a parent confirming an outline must
be shown the outline the book is actually built from, and while planning lived
inside ``write_story`` the preview was a *different* planning call that produced
a *different* story. Two runs on the same premise named the fox Finn and Kit.

So the critic lives here now. ``plan_story`` costs 2-4 calls instead of 1, and
``write_story`` costs that much less, because nothing plans twice.
"""

from collections.abc import Callable
from typing import Any
from uuid import uuid4

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.func import entrypoint, task

from sparkstory.config import settings
from sparkstory.entities.exceptions import StoryStructureError
from sparkstory.entities.grounding import StoryGrounding
from sparkstory.entities.reviews import OutlineReviews
from sparkstory.entities.stories import StoryBrief, StoryOutline
from sparkstory.memory.conflicts import find_conflicts
from sparkstory.memory.render import render_memory
from sparkstory.memory.store import build_memory_store
from sparkstory.memory.types import MemoryKind, MemoryRecord
from sparkstory.models.get_model import get_chat_model
from sparkstory.nodes.outline_critic import OutlineCriticNode
from sparkstory.nodes.researcher import ResearcherNode, build_researcher_agent
from sparkstory.nodes.story_planner import StoryPlannerNode
from sparkstory.observability.tracing import build_handler
from sparkstory.retrieval.embed import get_embedder
from sparkstory.retrieval.pg_store import PgVectorStore, build_store
from sparkstory.retrieval.provenance import drop_unprovenanced
from sparkstory.retrieval.tools import build_retrieval_tools
from sparkstory.retrieval.web.ledger import WebLedger
from sparkstory.utils.logging_utils import get_logger
from sparkstory.workflows.retries import RETRY_POLICY
from sparkstory.workflows.reviews import draft_score, drop_unroutable_outline_reviews
from sparkstory.workflows.types import OutlineWorkflowInput
from sparkstory.workflows.validation import validate_outline

logger = get_logger(__name__)


def build_research_context() -> tuple[Any, PgVectorStore, WebLedger | None]:
    """Build the research agent, the store its tools search, and the web ledger.

    Returned together because each is needed twice: the tools search the store and
    write to the ledger, and provenance filtering asks both whether a cited id
    exists.

    A named function rather than inline construction so tests can replace the whole
    research half in one patch -- the same seam shape as ``get_chat_model`` in this
    module, and patched here rather than at its definition for the same reason.

    Built per call rather than cached. The index is 58 chunks, so re-reading it
    costs a file read, while caching it would serve a stale index after a re-ingest
    without anything saying so. Model weights *are* cached, inside ``get_embedder``.

    **The ledger is ``None`` unless ``MAX_WEB_SEARCHES`` is above zero**, and that
    is what switches the whole web feature off: no ledger means
    ``build_retrieval_tools`` does not build the tool, so no client is constructed
    and no key is read. Per call rather than module-level for the same reason its
    ids are run-scoped -- a shared ledger would let one run's ``web:1`` resolve
    against another run's page.
    """
    store = build_store(
        settings.database_url,
        get_embedder(settings.embedding_model),
        settings.embedding_model,
    )
    ledger = WebLedger() if settings.max_web_searches > 0 else None
    agent = build_researcher_agent(
        model=get_chat_model(settings.researcher_model),
        # The store goes straight to the tools -- there is no HybridIndex wrapper
        # any more, because fusing a vector ranking with a keyword ranking now
        # happens inside one SQL statement rather than in Python over a corpus
        # loaded into memory.
        tools=build_retrieval_tools(store, ledger=ledger),
    )
    return agent, store, ledger


@task(retry_policy=RETRY_POLICY)
async def research(request_id: str, brief: StoryBrief) -> StoryGrounding:
    """Stage 0: find what this story must not get wrong, before it is planned.

    The only stage that chooses its own actions -- it decides whether to search at
    all, which collection to search, and what to search for.

    Provenance filtering runs here rather than in the entrypoint so that the
    artifact this task reports is the grounding that was actually *used*. Facts the
    corpus cannot support never reach the planner, and never appear in a run
    artifact as though they had.

    Exceptions are deliberately allowed to propagate, so ``RETRY_POLICY`` still
    applies to a transient provider failure. The decision to continue without
    grounding is made by the caller -- see ``outline_workflow``.
    """
    logger.info(
        "[%s] stage=research model=%s max_steps=%d",
        request_id,
        settings.researcher_model,
        settings.max_research_steps,
    )
    agent, store, ledger = build_research_context()
    grounding = await ResearcherNode(
        agent=agent, brief=brief, max_steps=settings.max_research_steps
    ).ainvoke()
    return drop_unprovenanced(grounding, store, ledger=ledger)


def fetch_memory(request_id: str, brief: StoryBrief) -> tuple[str, list[MemoryRecord]]:
    """Read what earlier books established, as prompt text and as records.

    Returns both because the two halves have different jobs: the text goes to the
    planner, and the records are what a freshly-planned outline is compared
    against to find conflicts.

    **Fails open**, like research above it. Memory is enrichment: a book planned
    without it is less consistent than it could be, while no book at all is a
    failure. A child with no ``child_id`` never reaches the store at all, which
    is the common case and must cost nothing.

    Not a ``@task``: it makes no model call, so wrapping it in one would buy
    retries on a database read while adding a checkpoint boundary to replay.
    """
    if not brief.child.child_id:
        return "", []
    try:
        records = build_memory_store().fetch(brief.child.child_id)
    except Exception:
        # ERROR with the exception, matching the research failure below: the
        # likeliest cause is an unreachable database, and the symptom otherwise
        # reads as "this child has no memory" -- which is indistinguishable from a
        # first book, and would be read as working.
        logger.exception("[%s] could not read memory; planning without it", request_id)
        return "", []
    logger.info("[%s] recalled %d memories", request_id, len(records))
    return render_memory(records), records


@task(retry_policy=RETRY_POLICY)
async def plan_outline(
    request_id: str,
    brief: StoryBrief,
    reviews: OutlineReviews | None = None,
    grounding: StoryGrounding | None = None,
    memory: str = "",
) -> StoryOutline:
    """Stage 1: plan the story's structure, or revise it from reviews.

    One task for both, because there is no editor node -- the generator does the
    editing. Validation runs on a revision exactly as it does on a first draft:
    a fix that breaks the beat cap must be caught, not trusted because a critic
    asked for it.
    """
    logger.info(
        "[%s] stage=plan model=%s revision=%s",
        request_id,
        settings.planner_model,
        reviews is not None,
    )
    node = StoryPlannerNode(
        model=get_chat_model(settings.planner_model),
        brief=brief,
        reviews=reviews,
        grounding=grounding,
        memory=memory,
    )
    outline = await node.ainvoke()
    validate_outline(brief, outline)
    return outline


@task(retry_policy=RETRY_POLICY)
async def critique_outline(
    request_id: str, brief: StoryBrief, outline: StoryOutline
) -> OutlineReviews:
    """Judge the plan before any prose is paid for.

    An empty review list is the signal that the plan is good, not a failure to
    review -- see the loop in ``outline_workflow``.
    """
    logger.info(
        "[%s] stage=critique_outline model=%s",
        request_id,
        settings.outline_critic_model,
    )
    node = OutlineCriticNode(
        model=get_chat_model(settings.outline_critic_model),
        brief=brief,
        outline=outline,
        max_reviews=settings.max_reviews_per_pass,
    )
    return drop_unroutable_outline_reviews(await node.ainvoke(), outline)


def build_outline_workflow(
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> Any:
    """Compile the outline workflow, optionally with a checkpointer.

    Same seam as ``build_story_workflow``, and unused for the same reason:
    with nothing to interrupt, a checkpointer is only a store nobody reads.
    """

    @entrypoint(checkpointer=checkpointer)
    async def outline_workflow(payload: OutlineWorkflowInput) -> StoryOutline:
        request_id = payload["request_id"]
        brief = payload["brief"]

        # Fail open, and this is the one place that decides it. Grounding is
        # enrichment: a book with no retrieved facts is a book, while no book at
        # all is a failure -- the opposite of the safety gate, which fails closed.
        # The rule is not "fail closed", it is "fail closed on harm, open on
        # enrichment".
        #
        # Logged at ERROR rather than WARNING, with the exception, because the
        # likeliest cause is a misconfigured provider and the symptom otherwise
        # reads as "research found nothing" -- the difference between a one-line
        # fix and an hour in the wrong layer.
        grounding: StoryGrounding | None = None
        if settings.max_research_steps > 0:
            try:
                grounding = await research(request_id, brief)
            except Exception:
                logger.exception(
                    "[%s] research failed; planning without grounding", request_id
                )
        else:
            logger.info("[%s] research skipped (MAX_RESEARCH_STEPS=0)", request_id)

        # Read before planning, so the planner is told what is already fixed
        # rather than being corrected afterwards. Costs no model call.
        memory_text, remembered = fetch_memory(request_id, brief)

        outline = await plan_outline(
            request_id, brief, grounding=grounding, memory=memory_text
        )

        # A plain `for` in the entrypoint body. Safe against the resume trap:
        # this body re-executes when a run resumes, but every @task inside is
        # replayed from the checkpoint, so the control flow repeats without
        # re-paying for a single call.
        #
        # `+ 1` so that *every* draft gets critiqued, including the last. Scoring
        # drafts to keep the best one is impossible if the final draft is never
        # scored -- it would be discarded unseen, making a cap of 2 behave like 1.
        # And a loop that ends on an unreviewed revision cannot say whether it
        # converged.
        best_outline, best_outline_score = outline, None
        for attempt in range(settings.max_outline_revisions + 1):
            reviews = await critique_outline(request_id, brief, outline)
            score = draft_score(reviews.reviews)
            if best_outline_score is None or score < best_outline_score:
                best_outline, best_outline_score = outline, score
            if not reviews.reviews:
                logger.info(
                    "[%s] outline approved after %d revision(s)", request_id, attempt
                )
                break
            if attempt == settings.max_outline_revisions:
                # Keeping the best draft rather than raising: a plan a critic
                # still dislikes is worse than one it approved, and far better
                # than no book at all.
                logger.warning(
                    "[%s] outline still has %d finding(s) after %d revision(s)",
                    request_id,
                    len(reviews.reviews),
                    settings.max_outline_revisions,
                )
                break
            logger.info(
                "[%s] revising outline: %d finding(s), attempt %d/%d",
                request_id,
                len(reviews.reviews),
                attempt + 1,
                settings.max_outline_revisions,
            )
            # Grounding travels into the revision too. Without it a revised plan
            # would silently lose every constraint the first draft was built to
            # respect, and the critic would have no way to notice. Memory travels
            # for the same reason: a revision that forgot what Kit looks like
            # would be free to re-describe him, and the critic has no rubric for
            # consistency with a book it has never seen.
            outline = await plan_outline(
                request_id, brief, reviews, grounding, memory=memory_text
            )

        # The best draft seen, not the last. A later revision can be worse: the
        # critic cannot reliably tell a feeling shown subtly from one that is
        # absent, so it re-flags a good page and the generator degrades it.
        #
        # Grounding is attached here, once, to the draft actually being returned.
        # Not inside `plan_outline`, because the loop discards drafts and grounding
        # on a discarded draft is noise in the run artifacts -- and this is the only
        # place that knows *which* outline won.
        #
        # This is what carries research past `plan_story`. Until now the grounding
        # was computed, planned from, and dropped when this returned a bare outline,
        # so the Writer had never seen a fact and a craft device could only ever be
        # described in a beat summary rather than used -- a planner told to repeat a
        # phrase wrote "they repeat the phrase" instead of repeating one.
        #
        # `is not None` rather than a truthiness check: empty grounding means
        # "research ran and found nothing", which is a correct and common answer,
        # and it must stay distinguishable from "research never ran": a run that
        # retrieved nothing renders identically in both world-rule modes, so
        # comparing against it is vacuous.
        if grounding is not None:
            best_outline = best_outline.model_copy(update={"grounding": grounding})

        # Conflicts are found against the draft that won, for the same reason
        # grounding is attached to it: the loop discards drafts, and a conflict
        # reported from a discarded one would point at wording no parent will see.
        #
        # Compared at the *plan* stage rather than after the book, so the parent
        # meets the disagreement at the approval point that already exists. The
        # cost is that a character the Writer describes differently in prose is
        # not caught here -- that surfaces on the next book, when extraction runs
        # over what was actually written.
        if remembered:
            planned = [
                MemoryRecord(
                    child_id=brief.child.child_id or "",
                    kind=MemoryKind.SEMANTIC,
                    text=character.description,
                    subject=character.name,
                    source_request_id=request_id,
                )
                for character in best_outline.characters
            ]
            conflicts = find_conflicts(new=planned, stored=remembered)
            if conflicts:
                logger.info(
                    "[%s] %d memory conflict(s) for the parent to resolve",
                    request_id,
                    len(conflicts),
                )
                best_outline = best_outline.model_copy(
                    update={"memory_conflicts": conflicts}
                )
        return best_outline

    return outline_workflow


#: Compiled once per process. Compiling is pure -- no network, no API key.
OUTLINE_WORKFLOW = build_outline_workflow()


async def run_outline_pipeline(
    brief: StoryBrief,
    on_task_result: Callable[[str, Any], None] | None = None,
    *,
    request_id: str | None = None,
) -> StoryOutline:
    """Plan a story's structure and revise it until a critic approves.

    This is what ``plan_story`` runs. It is no longer "one cheap call": the
    outline returned here is the one a parent is shown *and* the one
    ``write_story`` builds from, so it has to be worth approving.

    Args:
        brief: What to plan.
        on_task_result: Called with ``(task_name, result)`` as each ``@task``
            completes, including every loop iteration. Optional, and nothing on
            the MCP path passes it.
        request_id: Identifies this run in the logs and, when tracing is on, as
            the Opik thread id. Supply it to join this call to a later
            ``run_story_pipeline`` as one book -- which only a caller that knows
            both stages belong together may do. The MCP path passes nothing and
            gets a fresh id, because ``plan_story`` and ``write_story`` are
            separate tool calls that may be minutes apart and may not even
            concern the same outline.

    Raises:
        MissingAPIKeyError: a configured model's API key is not set.
        UnknownModelError: a ``*_MODEL`` setting is not a known model id.
        StoryStructureError: the planner produced more beats than the brief has
            pages, and could not be talked out of it.
    """
    request_id = request_id or str(uuid4())
    logger.info(
        "[%s] planning outline: age=%d level=%s pages=%d",
        request_id,
        brief.child.age,
        brief.child.reading_level.value,
        brief.page_count,
    )

    # The entrypoint's own return is identified by *name*, not by type. The story
    # pipeline can use `isinstance(value, Story)` because nothing else returns
    # one; here `plan_outline` also returns a StoryOutline, so a type check would
    # mistake a task result for the final answer -- and hand every intermediate
    # draft to on_task_result as if it were the finished plan.
    outline: StoryOutline | None = None
    # One attachment per pipeline: LangGraph propagates callbacks down the
    # runnable tree, so every @task and every node call beneath this is covered
    # without touching any of them. `build_handler` returns None when tracing is
    # off, and a None inside the callback list is an AttributeError in the middle
    # of a paid run, so it is filtered rather than passed through.
    tracer = build_handler(request_id, tags=["plan_outline"])
    async for update in OUTLINE_WORKFLOW.astream(
        OutlineWorkflowInput(request_id=request_id, brief=brief),
        stream_mode="updates",
        config={"callbacks": [t for t in [tracer] if t is not None]},
    ):
        for name, value in update.items():
            if name == "outline_workflow":
                outline = value
                continue
            if on_task_result is not None:
                on_task_result(name, value)

    if outline is None:  # pragma: no cover - the entrypoint always returns one
        raise StoryStructureError("The workflow completed without an outline.")

    logger.info("[%s] planned %r", request_id, outline.title)
    return outline
