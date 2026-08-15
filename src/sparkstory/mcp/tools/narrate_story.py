"""Tool implementation for narrating a finished story.

Same translation rule as its siblings.

``ConfigurationError``
    An operator can act on it -- the message names the variable to set. This
    includes ``AudioConfigurationError``, which is a subclass precisely so that an
    unset ``XAI_API_KEY`` reaches a client as a sentence rather than a traceback.
    It is raised before any page is attempted, so a missing key costs nothing
    rather than failing once per page.

**What does not appear here is the interesting part**, exactly as in
``illustrate_story``. A per-page failure never reaches this layer, because
narration fails soft: a page whose audio fails is recorded as ``FAILED`` and the
run continues. So there is no translation for it -- the ``StoryNarration``
returned *is* the report, and a client reads ``is_complete`` and
``pages_narrated`` to learn what became of the book. This is verified live rather
than assumed: pointing the voice table at an unknown id failed all ten pages of a
real run, wrote no ``story.mp3``, and completed.

The alternative -- raising on partial success -- would throw away a narrated book
over one missing page.
"""

from pathlib import Path

from fastmcp.exceptions import ToolError

from sparkstory.entities.exceptions import ConfigurationError
from sparkstory.entities.narration import StoryNarration
from sparkstory.entities.stories import Story, StoryBrief
from sparkstory.utils.logging_utils import get_logger
from sparkstory.workflows.narrate import run_narration_pipeline

logger = get_logger(__name__)


async def narrate_story_tool(
    brief: StoryBrief, story: Story, output_directory: str
) -> StoryNarration:
    """Read a finished book aloud, one audio file per page plus a stitched whole.

    ``output_directory`` is a required argument rather than a setting, for the
    reason its sibling gives: the audio belongs beside whatever the caller is
    assembling, and only the caller knows where that is.

    There is no ``voice`` argument. ``StoryBrief`` already carries one and the
    pipeline reads it there, so a second source would need the artifact to record
    which of the two won.

    Note the argument order: this signature mirrors ``illustrate_story_tool``
    (``brief, story``) while ``run_narration_pipeline`` takes ``story, brief``.
    The tools are what a client sees, so they stay consistent with each other.
    """
    try:
        return await run_narration_pipeline(story, brief, Path(output_directory))
    except ConfigurationError as exc:
        logger.error("Narration failed -- configuration: %s", exc)
        raise ToolError(str(exc)) from exc
    # No `except AudioGenerationError`, and its absence is deliberate -- the same
    # note `illustrate_story.py` carries. A per-page failure never propagates: it
    # is caught where it happens and recorded in `StoryNarration`, which is what
    # "narration fails soft" means. Catching it here would be dead code today
    # and, worse, would be the obvious place for someone to later convert a
    # partial success into a raised error, undoing the design.
