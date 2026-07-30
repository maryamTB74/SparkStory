"""Tool implementation for story planning.

This layer is thin on purpose, and it is not redundant with
``nodes/story_planner.py``. Its job is translation: the handler raises exceptions
that are meaningful to us, and an MCP client needs errors that are meaningful to
*it*. A bare traceback tells a client agent nothing actionable; a ``ToolError``
naming the missing environment variable does.

The rule is that only ``ConfigurationError`` is translated, because it is the one
category an operator can act on. Everything else propagates.

That distinction is deliberately drawn on our *own* exception type rather than a
built-in. An earlier version caught ``RuntimeError`` to mean "missing API key",
which would have relabelled any unrelated ``RuntimeError`` from LangChain or the
transport as a configuration problem -- turning a real bug into a confidently
wrong message.
"""

from fastmcp.exceptions import ToolError

from sparkstory.entities.exceptions import ConfigurationError
from sparkstory.entities.stories import StoryBrief, StoryOutline
from sparkstory.nodes.story_planner import plan_story
from sparkstory.utils.logging_utils import get_logger

logger = get_logger(__name__)


async def plan_story_tool(brief: StoryBrief) -> StoryOutline:
    """Plan a story, mapping configuration failures to client-facing errors."""
    try:
        return await plan_story(brief)
    except ConfigurationError as exc:
        # Operator error with a known fix. The original messages already name the
        # variable to set or list the valid model ids, so pass them through.
        logger.error("Story planning failed -- configuration: %s", exc)
        raise ToolError(str(exc)) from exc
