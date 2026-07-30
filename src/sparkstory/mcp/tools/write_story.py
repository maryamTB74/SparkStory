"""Tool implementation for writing a complete story.

Same translation job as ``plan_story``: only ``ConfigurationError`` becomes a
``ToolError``, because it is the one category an operator can act on. Everything
else propagates.

That deliberately includes ``StoryStructureError``. It is not operator-fixable, so
dressing it up as a configuration message would send debugging to the wrong layer
-- the same defect this rule was written for after an earlier version caught bare
``RuntimeError`` to mean "missing API key".
"""

from fastmcp.exceptions import ToolError

from sparkstory.entities.exceptions import ConfigurationError
from sparkstory.entities.stories import Story, StoryBrief
from sparkstory.utils.logging_utils import get_logger
from sparkstory.workflows.write_story import run_story_pipeline

logger = get_logger(__name__)


async def write_story_tool(brief: StoryBrief) -> Story:
    """Write a story, mapping configuration failures to client-facing errors."""
    try:
        return await run_story_pipeline(brief)
    except ConfigurationError as exc:
        logger.error("Story writing failed -- configuration: %s", exc)
        raise ToolError(str(exc)) from exc
