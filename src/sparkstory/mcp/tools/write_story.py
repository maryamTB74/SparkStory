"""Tool implementation for writing a complete story.

Two categories become a ``ToolError`` here, for two different reasons.

``ConfigurationError``
    An operator can act on it -- the message names the variable to set.

``UnsafeContentError``
    Nobody can act on it by changing configuration, and it is not a bug either:
    it means the system worked and the answer is no. A caller that asked for a
    story deserves to hear that in a sentence, and the caller is often an LLM
    agent, which can act on a sentence and cannot act on a stack trace.

``StoryStructureError``, *but only from validating the caller's own outline*
    The outline is an argument now, threaded through by an LLM client from
    ``plan_story``. It can be stale, hand-edited or invented, and "these beats
    do not fit that page count" is something the caller can fix.

Everything else propagates -- including a ``StoryStructureError`` raised anywhere
inside the pipeline, which means *our* agent produced nonsense. That one is not
operator-fixable, so dressing it up as a client-facing message would send
debugging to the wrong layer -- the same defect this rule was written for after
an earlier version caught bare ``RuntimeError`` to mean "missing API key". The
distinction is drawn by *where* validation runs, not by catching more broadly:
hence the separate ``try`` below.
"""

from fastmcp.exceptions import ToolError

from sparkstory.entities.exceptions import (
    ConfigurationError,
    StoryStructureError,
    UnsafeContentError,
)
from sparkstory.entities.stories import Story, StoryBrief, StoryOutline
from sparkstory.mcp.tools.destinations import resolve_destination
from sparkstory.mcp.tools.pdf import render_pdf_beside
from sparkstory.utils.logging_utils import get_logger
from sparkstory.workflows.validation import validate_outline
from sparkstory.workflows.write_story import run_story_pipeline

logger = get_logger(__name__)

#: Named to match what `scripts/write_one_story.py` already writes, so a book
#: saved through MCP and one saved through the script are the same file.
_STORY_FILENAME = "story.json"


async def write_story_tool(
    brief: StoryBrief, outline: StoryOutline, output_directory: str
) -> Story:
    """Write a story from an approved plan, mapping client-actionable outcomes
    to ToolError.

    ``output_directory`` is required, matching its two media siblings. Before
    this, the prose was the one artifact that existed only as a tool result: a
    client reported the book as ready and had no path to give, and closing the
    session lost it. The pictures and the audio were files; the book was not.
    """
    try:
        # Here rather than inside the `try` below, so that only the caller's own
        # mistake is translated. The pipeline validates again -- cheaply, and it
        # is the workflow's own precondition, not something to trust a caller to
        # have checked.
        validate_outline(brief, outline)
    except StoryStructureError as exc:
        logger.error("Story writing failed -- outline does not fit brief: %s", exc)
        raise ToolError(str(exc)) from exc

    try:
        story = await run_story_pipeline(brief, outline)
    except ConfigurationError as exc:
        logger.error("Story writing failed -- configuration: %s", exc)
        raise ToolError(str(exc)) from exc
    except UnsafeContentError as exc:
        # WARNING rather than ERROR: nothing is broken. The guardrail did its
        # job, and the parent can adjust the brief and ask again -- which is why
        # the finding itself travels in the message.
        logger.warning("Story writing refused -- safety: %s", exc)
        raise ToolError(str(exc)) from exc

    # Saving is separate from generating, and its failure is a different kind.
    # The caller chose this path and can choose a better one, so an OSError here
    # is client-actionable in exactly the sense this module's docstring defines
    # -- and it must arrive as a sentence, because the book has already been
    # paid for by the time it can happen.
    directory = resolve_destination(output_directory)
    destination = directory / _STORY_FILENAME
    try:
        directory.mkdir(parents=True, exist_ok=True)
        destination.write_text(story.model_dump_json(indent=2), encoding="utf-8")
    except OSError as exc:
        logger.error("Story written but could not be saved to %s: %s", destination, exc)
        raise ToolError(
            f"The story was written but could not be saved to {destination}: {exc}"
        ) from exc

    logger.info("Story saved to %s", destination)
    return story.model_copy(
        update={
            "saved_to": str(directory),
            "pdf_saved_to": render_pdf_beside(story, directory),
        }
    )
