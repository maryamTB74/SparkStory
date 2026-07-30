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

from typing import Any
from uuid import uuid4

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.func import entrypoint, task
from langgraph.types import RetryPolicy, default_retry_on

from sparkstory.config import settings
from sparkstory.entities.exceptions import ConfigurationError, StoryStructureError
from sparkstory.entities.stories import (
    PagePlan,
    Story,
    StoryBrief,
    StoryOutline,
    StoryProse,
)
from sparkstory.models.get_model import get_chat_model
from sparkstory.nodes.plot_planner import PlotPlannerNode
from sparkstory.nodes.story_planner import StoryPlannerNode
from sparkstory.nodes.writer import WriterNode
from sparkstory.utils.logging_utils import get_logger
from sparkstory.workflows.types import StoryWorkflowInput
from sparkstory.workflows.validation import (
    validate_outline,
    validate_page_plan,
    validate_prose,
)

logger = get_logger(__name__)


def _retry_on(exc: Exception) -> bool:
    """Retry transient failures only.

    LangGraph's ``default_retry_on`` declines to retry ``ValueError``, and
    pydantic's ``ValidationError`` is one -- so schema failures are excluded for
    free. It returns ``True`` for exception types it does not recognise, though,
    which includes every error of ours. Both current kinds must be excluded, and
    both exclusions were earned:

    ``ConfigurationError``
        Found by running it. A missing ``GOOGLE_API_KEY`` was retried three times,
        printing three tracebacks for a problem whose fix is one line in ``.env``.
        Trying again cannot make a key appear.

    ``StoryStructureError``
        Retrying re-sends an identical prompt and re-rolls the dice, while hiding
        how often an agent gets a page count wrong -- exactly the frequency data
        the evaluator-optimizer loop of a later session is designed from.

    Listed explicitly rather than excluding ``SparkStoryError`` wholesale: an
    upstream ``ProviderError``, when one exists, *should* be retried.
    """
    if isinstance(exc, ConfigurationError | StoryStructureError):
        return False
    return default_retry_on(exc)


#:``max_attempts=3``, narrowed by ``_retry_on`` above.
RETRY_POLICY = RetryPolicy(max_attempts=3, retry_on=_retry_on)


@task(retry_policy=RETRY_POLICY)
async def plan_outline(request_id: str, brief: StoryBrief) -> StoryOutline:
    """Stage 1: plan the story's structure."""
    logger.info("[%s] stage=plan model=%s", request_id, settings.planner_model)
    node = StoryPlannerNode(
        model=get_chat_model(settings.planner_model),
        brief=brief,
    )
    outline = await node.ainvoke()
    validate_outline(brief, outline)
    return outline


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
    return plan


@task(retry_policy=RETRY_POLICY)
async def write_prose(
    request_id: str,
    brief: StoryBrief,
    outline: StoryOutline,
    page_plan: PagePlan,
) -> StoryProse:
    """Stage 3: write the words."""
    logger.info("[%s] stage=write model=%s", request_id, settings.writer_model)
    node = WriterNode(
        model=get_chat_model(settings.writer_model),
        brief=brief,
        outline=outline,
        page_plan=page_plan,
    )
    prose = await node.ainvoke()
    validate_prose(page_plan, prose)
    return prose


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

        outline = await plan_outline(request_id, brief)
        page_plan = await plan_pages(request_id, brief, outline)
        prose = await write_prose(request_id, brief, outline, page_plan)

        return Story(outline=outline, page_plan=page_plan, pages=prose.pages)

    return story_workflow


#: Compiled once per process. Compiling is pure -- no network, no API key -- so
#: doing it at import is safe, and it keeps a per-request cost out of every call.
STORY_WORKFLOW = build_story_workflow()


async def run_story_pipeline(brief: StoryBrief) -> Story:
    """Write a complete story from a brief.

    Mints the ``request_id`` here rather than inside the workflow, so it survives
    a resume -- see :class:`~sparkstory.workflows.types.StoryWorkflowInput`.

    Raises:
        MissingAPIKeyError: a configured model's API key is not set.
        UnknownModelError: a ``*_MODEL`` setting is not a known model id.
        StoryStructureError: an agent's output was well-formed but structurally
            wrong. Not retried and not translated: a later session turns this into
            a retry carrying the message as feedback.
    """
    request_id = str(uuid4())
    logger.info(
        "[%s] writing story: age=%d level=%s tone=%s pages=%d",
        request_id,
        brief.child.age,
        brief.child.reading_level.value,
        brief.tone.value,
        brief.page_count,
    )

    story: Story = await STORY_WORKFLOW.ainvoke(
        StoryWorkflowInput(request_id=request_id, brief=brief)
    )

    logger.info(
        "[%s] finished %r: %d pages",
        request_id,
        story.outline.title,
        len(story.pages),
    )
    return story
