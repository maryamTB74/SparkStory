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
from sparkstory.models.get_model import get_chat_model
from sparkstory.nodes.outline_critic import OutlineCriticNode
from sparkstory.nodes.researcher import ResearcherNode, build_researcher_agent
from sparkstory.nodes.story_planner import StoryPlannerNode
from sparkstory.retrieval.embed import get_embedder
from sparkstory.retrieval.hybrid import HybridIndex
from sparkstory.retrieval.provenance import drop_unprovenanced
from sparkstory.retrieval.store import LocalVectorStore
from sparkstory.retrieval.tools import build_retrieval_tools
from sparkstory.utils.logging_utils import get_logger
from sparkstory.workflows.retries import RETRY_POLICY
from sparkstory.workflows.reviews import draft_score, drop_unroutable_outline_reviews
from sparkstory.workflows.types import OutlineWorkflowInput
from sparkstory.workflows.validation import validate_outline

logger = get_logger(__name__)


def build_research_context() -> tuple[Any, LocalVectorStore]:
    """Build the research agent and the store its tools search.

    Returned together because the store is needed twice: the tools search it, and
    provenance filtering asks it whether a cited chunk exists.

    A named function rather than inline construction so tests can replace the whole
    research half in one patch -- the same seam shape as ``get_chat_model`` in this
    module, and patched here rather than at its definition for the same reason.

    Built per call rather than cached. The index is 58 chunks, so re-reading it
    costs a file read, while caching it would serve a stale index after a re-ingest
    without anything saying so. Model weights *are* cached, inside ``get_embedder``.
    """
    store = LocalVectorStore(
        root=settings.knowledge_root,
        embedder=get_embedder(settings.embedding_model),
    )
    agent = build_researcher_agent(
        model=get_chat_model(settings.researcher_model),
        tools=build_retrieval_tools(HybridIndex(store=store)),
    )
    return agent, store


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
    agent, store = build_research_context()
    grounding = await ResearcherNode(
        agent=agent, brief=brief, max_steps=settings.max_research_steps
    ).ainvoke()
    return drop_unprovenanced(grounding, store)


@task(retry_policy=RETRY_POLICY)
async def plan_outline(
    request_id: str,
    brief: StoryBrief,
    reviews: OutlineReviews | None = None,
    grounding: StoryGrounding | None = None,
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

        outline = await plan_outline(request_id, brief, grounding=grounding)

        # A plain `for` in the entrypoint body, as brown's generate_article.py
        # does it. Safe against the resume trap: this body re-executes when a run
        # resumes, but every @task inside is replayed from the checkpoint, so the
        # control flow repeats without re-paying for a single call.
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
            # respect, and the critic would have no way to notice.
            outline = await plan_outline(request_id, brief, reviews, grounding)

        # The best draft seen, not the last. A later revision can be worse: the
        # critic cannot reliably tell a feeling shown subtly from one that is
        # absent, so it re-flags a good page and the generator degrades it.
        return best_outline

    return outline_workflow


#: Compiled once per process. Compiling is pure -- no network, no API key.
OUTLINE_WORKFLOW = build_outline_workflow()


async def run_outline_pipeline(
    brief: StoryBrief,
    on_task_result: Callable[[str, Any], None] | None = None,
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

    Raises:
        MissingAPIKeyError: a configured model's API key is not set.
        UnknownModelError: a ``*_MODEL`` setting is not a known model id.
        StoryStructureError: the planner produced more beats than the brief has
            pages, and could not be talked out of it.
    """
    request_id = str(uuid4())
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
    async for update in OUTLINE_WORKFLOW.astream(
        OutlineWorkflowInput(request_id=request_id, brief=brief),
        stream_mode="updates",
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
