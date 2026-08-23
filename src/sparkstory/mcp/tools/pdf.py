"""Rendering a book's PDF beside its JSON, for the two tools that can produce one.

Shared by `write_story` and `illustrate_story` for the reason `destinations.py`
gives about itself: they must agree on the filename, and copying the function
lets them drift. It is a separate module rather than part of `destinations.py`
because that module answers *where a tool may write*, which is a policy about
client input; this answers *how a book becomes a PDF*, which is not.

**Why two tools render the same file, which looks like duplicated work and is
not.** A book's PDF can only include pictures that exist when it is rendered,
and over MCP the prose and the pictures arrive in separate tool calls:
`write_story` runs first and there are no illustrations yet, `illustrate_story`
runs later and cannot reach back into a call that has already returned. So the
first render is the text-only book -- which is a complete, correct product, and
was the only product before illustration existed -- and the second replaces it
with the illustrated one.

`scripts/write_one_story.py` has never needed this: it does the whole run in one
process and still holds the `StoryArt` when it renders, so it passes `art`
directly and renders once. The split is a property of the tool boundary, not of
the renderer.

The alternative was for `write_story` to skip the PDF and leave it to a later
call. Rejected: illustration is optional and expensive, so a parent who declines
pictures -- or whose `ILLUSTRATION_ENABLED` is false -- would get no PDF at all,
and the cheapest path in the system would lose an artifact it currently
produces.
"""

from pathlib import Path

from sparkstory.entities.exceptions import StoryStructureError
from sparkstory.entities.illustration import StoryArt
from sparkstory.entities.stories import Story
from sparkstory.renderers import render_pdf
from sparkstory.utils.logging_utils import get_logger

logger = get_logger(__name__)

#: Matched to what `scripts/write_one_story.py` writes and to what
#: `scripts/build_pdf.py` regenerates, so a book saved through MCP and one saved
#: through the script are the same file. That interchangeability is what makes
#: this artifact safe to lose: it is a rendering of the book, not the book.
PDF_FILENAME = "story.pdf"


def render_pdf_beside(
    story: Story, directory: Path, art: StoryArt | None = None
) -> str | None:
    """Write `story` as a PDF next to its JSON, or return `None` saying it did not.

    Args:
        story: The finished book.
        directory: The already-resolved directory the book was saved into.
        art: Illustrations to place, if any. `None` renders the text-only book,
            which is `render_pdf`'s own documented default rather than a special
            case here -- and a `StoryArt` whose every image failed renders
            identically, so a caller is not asked to check first.

    Returns the path as a string on success, `None` on failure.

    Deliberately soft, and the reason is that a PDF and the book are not the same
    kind of thing. `story.json` *is* the book: without it there is nothing. The
    PDF is a rendering of it that `scripts/build_pdf.py` reconstructs from that
    JSON alone, so losing it costs a command rather than a run. Raising would
    discard a book that is sitting on disk, correct and complete, and by the time
    this runs the model calls have already been paid for.

    Two failures are caught rather than one, because both are reachable and
    neither is a bug in this layer. `render_pdf` raises `StoryStructureError`
    when a page's text overflows its frame -- a property of the prose, not of the
    filesystem, and one it refuses to paper over by truncating. `OSError` covers
    the disk, and is reached separately from the JSON write because a PDF is far
    larger and can fail where a few kilobytes of JSON did not.
    """
    path = directory / PDF_FILENAME
    try:
        render_pdf(story, path, art)
    except (StoryStructureError, OSError) as exc:
        # ERROR, not WARNING: unlike a safety refusal, nothing about this is the
        # system working as intended, and it needs to be visible in the log
        # rather than inferred from an absent field. The field is what a client
        # can act on; the log is what says why.
        logger.error("Story saved to %s but the PDF could not be made: %s", path, exc)
        return None

    logger.info(
        "Book PDF saved to %s (%s)",
        path,
        "with illustrations" if art is not None else "text only",
    )
    return str(path)
