"""Tool implementation for illustrating a finished story.

Same translation rule as its siblings, with one addition that is specific to this
stage.

``ConfigurationError``
    An operator can act on it -- the message names the variable to set. This
    includes ``ImageConfigurationError``, which is a subclass precisely so that an
    unset ``XAI_API_KEY`` reaches a client as a sentence rather than a traceback.

``StoryStructureError``
    Only from ``validate_illustration_plan``, which means *our own* Director returned
    a plan that does not cover the book. That is a bug rather than a client error, so
    it is deliberately **not** translated -- it propagates, exactly as a
    ``StoryStructureError`` from inside ``write_story``'s pipeline does. The rule
    this follows is the sibling module's: translate what the caller can act on, and
    let our own malformed output surface as the defect it is.

**What does not appear here is the interesting part.** A per-image failure never
reaches this layer, because illustration fails soft: a page whose image fails
leaves its frame blank and the run continues. So there is no translation for it --
the ``StoryArt`` returned *is* the report, and a client reads ``fully_conditioned``
to learn whether the consistency mechanism actually ran. That is deliberate: the
alternative is a tool that raises on partial success, which would throw away a book
over one missing picture.
"""

from fastmcp.exceptions import ToolError

from sparkstory.entities.exceptions import ConfigurationError
from sparkstory.entities.illustration import StoryArt
from sparkstory.entities.stories import Story, StoryBrief
from sparkstory.mcp.tools.destinations import resolve_destination
from sparkstory.mcp.tools.pdf import render_pdf_beside
from sparkstory.utils.logging_utils import get_logger
from sparkstory.workflows.illustrate import run_illustration_pipeline

logger = get_logger(__name__)


async def illustrate_story_tool(
    brief: StoryBrief, story: Story, output_directory: str
) -> StoryArt:
    """Draw a reference portrait per character, then one picture per page.

    ``output_directory`` is a required argument rather than a setting, because the
    images belong beside whatever the caller is assembling and only the caller knows
    where that is. A setting here would be configuration for a decision the caller
    already has to make.
    """
    try:
        # Resolved rather than used raw, and shared with `write_story` so one
        # name cannot mean two directories. It did: this tool passed the string
        # straight to `Path(...)`, so a caller's "kim-metocondry" put the book
        # under `outputs/` and the pictures in the repository root.
        destination = resolve_destination(output_directory)
        art = await run_illustration_pipeline(brief, story, destination)
    except ConfigurationError as exc:
        logger.error("Illustration failed -- configuration: %s", exc)
        raise ToolError(str(exc)) from exc
    # No `except ImageGenerationError`, and its absence is deliberate. A per-image
    # failure never propagates -- it is caught where it happens and recorded in
    # `StoryArt`, which is what "illustration fails soft" means. Catching it here
    # would be dead code today and, worse, would be the obvious place for someone to
    # later convert a partial success into a raised error, undoing the design.

    # Re-render the book now that there are pictures to put in it. `write_story`
    # made a PDF before any image existed, so the file sitting beside the JSON is
    # the text-only book; this replaces it with the illustrated one, and it is the
    # only way an illustrated PDF can exist over MCP -- the two halves arrive in
    # separate tool calls and the earlier one cannot be reached again.
    #
    # Outside the `try` above on purpose. That block translates
    # `ConfigurationError` into a client-facing sentence, and a
    # `StoryStructureError` from the renderer is not a configuration problem; it
    # would arrive at the client as one. `render_pdf_beside` swallows both of its
    # own failure modes and reports them as `None`, so there is nothing to catch.
    #
    # Passing `art` unconditionally rather than checking whether anything drew:
    # `render_pdf` treats `None` and an all-failed `StoryArt` identically by
    # design, so a run where every image failed re-renders the same text-only
    # book it would have produced anyway. Branching here would add a path with no
    # different outcome.
    return art.model_copy(
        update={"pdf_saved_to": render_pdf_beside(story, destination, art)}
    )
