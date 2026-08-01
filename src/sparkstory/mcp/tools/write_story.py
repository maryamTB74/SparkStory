"""Tool implementation for writing a complete story.

Two categories become a ``ToolError`` here, for two different reasons.

``ConfigurationError``
    An operator can act on it -- the message names the variable to set.

``UnsafeContentError``
    Nobody can act on it by changing configuration, and it is not a bug either:
    it means the system worked and the answer is no. A caller that asked for a
    story deserves to hear that in a sentence, and the caller is often an LLM
    agent, which can act on a sentence and cannot act on a stack trace.

Everything else propagates.

That deliberately includes ``StoryStructureError``. It is not operator-fixable, so
dressing it up as a configuration message would send debugging to the wrong layer
-- the same defect this rule was written for after an earlier version caught bare
``RuntimeError`` to mean "missing API key".
"""

from fastmcp.exceptions import ToolError

from sparkstory.entities.exceptions import ConfigurationError, UnsafeContentError
from sparkstory.entities.stories import Story, StoryBrief
from sparkstory.utils.logging_utils import get_logger
from sparkstory.workflows.write_story import run_story_pipeline

logger = get_logger(__name__)


async def write_story_tool(brief: StoryBrief) -> Story:
    """Write a story, mapping client-actionable outcomes to ToolError."""
    try:
        return await run_story_pipeline(brief)
    except ConfigurationError as exc:
        logger.error("Story writing failed -- configuration: %s", exc)
        raise ToolError(str(exc)) from exc
    except UnsafeContentError as exc:
        # WARNING rather than ERROR: nothing is broken. The guardrail did its
        # job, and the parent can adjust the brief and ask again -- which is why
        # the finding itself travels in the message.
        logger.warning("Story writing refused -- safety: %s", exc)
        raise ToolError(str(exc)) from exc
