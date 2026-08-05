"""Turning a finished `Story` into a book-shaped PDF.

The name is `brown`'s, reserved in CLAUDE.md for book assembly. It lays out the
pages and fills the upper 55% of each one with that page's illustration -- or
leaves it blank when there is none, which is the text-only book Session 10
shipped. Reserving the space before there was anything to put in it is why
Session 6 filled a frame instead of forcing a relayout.

Deliberately pure. It takes a `Story`, a path and optionally a `StoryArt`,
imports nothing from the rest of the package except the entities, reads no
settings, and generates nothing -- so its entire behaviour is determined by its
arguments and its tests need no fakes at all. In particular it never asks whether
an image *should* exist; it draws what it is handed.

It takes no `StoryBrief`, so the title page carries the title and logline but
not a dedication. That is a consequence of the signature, chosen deliberately;
see docs/superpowers/specs/2026-08-03-pdf-assembly-design.md.
"""

from pathlib import Path

from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Frame, Paragraph

from sparkstory.entities.exceptions import StoryStructureError
from sparkstory.entities.illustration import StoryArt
from sparkstory.entities.stories import Story
from sparkstory.utils.logging_utils import get_logger

logger = get_logger(__name__)

#: Square, because picture books are square or landscape far more often than
#: A4, and square survives being read on a screen as well as printed. Fixed
#: rather than configurable: Rule 3, one caller.
_PAGE = 200 * mm

_MARGIN = 18 * mm

#: The illustration slot. Blank in a text-only book, an image frame in Session
#: 6. Expressed as a fraction of the page so the two halves cannot drift apart.
_ART_FRACTION = 0.55

#: Large because the audience is five and the *reader* is an adult holding the
#: book at arm's length.
_BODY = ParagraphStyle(
    "body",
    fontName="Helvetica",
    fontSize=16,
    leading=24,
)

_TITLE = ParagraphStyle(
    "title",
    fontName="Helvetica-Bold",
    fontSize=32,
    leading=38,
    alignment=TA_CENTER,
)

_LOGLINE = ParagraphStyle(
    "logline",
    fontName="Helvetica-Oblique",
    fontSize=14,
    leading=20,
    alignment=TA_CENTER,
    spaceBefore=12,
)


def render_pdf(story: Story, path: Path, art: StoryArt | None = None) -> None:
    """Write `story` to `path` as a PDF, one page per story page.

    Args:
        story: The finished book.
        path: Where to write the PDF.
        art: Illustrations to place, if any. Optional and defaulted so the
            text-only behaviour stays reachable and tested unchanged -- a book
            with no pictures is not an error state, it is the product Session 10
            shipped. `None` and a `StoryArt` with no usable images behave
            identically, deliberately: a fully failed illustration run must
            degrade to exactly the text-only book rather than to a third code
            path nobody tests.

    Raises `StoryStructureError` if a page's text does not fit its frame --
    never truncates, because a book that looks finished and is quietly missing
    words is the worst outcome available here.
    """
    canvas = Canvas(str(path), pagesize=(_PAGE, _PAGE))

    _draw_title_page(canvas, story)
    for page in story.pages:
        canvas.showPage()
        _draw_story_page(canvas, page.page_number, page.text)
        # After the text, so a corrupt image cannot leave a page with a picture
        # and no words. Text is the part that must survive.
        if art is not None:
            _draw_illustration(canvas, art.page_image(page.page_number))

    canvas.save()


def _draw_title_page(canvas: Canvas, story: Story) -> None:
    # Theme and characters are planning artifacts; a book does not print them.
    frame = Frame(
        _MARGIN,
        _PAGE * 0.4,
        _PAGE - 2 * _MARGIN,
        _PAGE * 0.45,
        showBoundary=0,
    )
    frame.addFromList(
        [
            Paragraph(story.outline.title, _TITLE),
            Paragraph(story.outline.logline, _LOGLINE),
        ],
        canvas,
    )


def _draw_illustration(canvas: Canvas, image_path: Path | None) -> None:
    """Place one page's illustration in the reserved upper area.

    Does nothing when there is no image. A missing picture and a failed one are the
    same page, so the caller is not asked to tell them apart.

    The image is scaled to *fit inside* the frame and centred, never stretched to
    fill it. Grok returns 3:4 portrait images while the frame is wider than it is
    tall, so filling would distort every character in the book -- and a subtly
    stretched face is the kind of wrong that is obvious to a reader and invisible
    in a test asserting the image was placed.
    """
    if image_path is None:
        return

    frame_bottom = _PAGE * (1 - _ART_FRACTION)
    frame_width = _PAGE - 2 * _MARGIN
    frame_height = _PAGE * _ART_FRACTION - _MARGIN

    # One `try` around reading *and* drawing, which is load-bearing and was found by
    # a failing test. PIL decodes lazily: `getSize()` reads only the header, so a
    # truncated file passes it and then raises inside `drawImage`. Guarding only the
    # read looked correct, passed a corrupt-file test that happened to fail at the
    # header, and would have crashed a real book on a partial download.
    #
    # Deliberately broad: reportlab surfaces whatever PIL raises, and there is no
    # shared base class to name. A page with no picture is the right outcome for a
    # bad file -- the alternative is destroying a finished book over one download.
    try:
        reader = ImageReader(str(image_path))
        source_width, source_height = reader.getSize()
        if source_width <= 0 or source_height <= 0:
            return

        # Scaled to fit *inside* the frame and centred, never stretched to fill it.
        # Grok returns 3:4 portrait images while this frame is wider than it is
        # tall, so filling would distort every character in the book -- and a
        # subtly stretched face is obvious to a reader while being invisible to a
        # test that only asserts an image was placed.
        scale = min(frame_width / source_width, frame_height / source_height)
        width = source_width * scale
        height = source_height * scale

        canvas.drawImage(
            reader,
            (_PAGE - width) / 2,
            frame_bottom + (frame_height - height) / 2,
            width=width,
            height=height,
            # A generated illustration has no alpha channel to honour, and asking
            # reportlab to compute a mask on every page is slow for nothing.
            mask=None,
        )
    except Exception:  # noqa: BLE001
        logger.warning(
            "could not place illustration %s, leaving the frame blank", image_path
        )
        return


def _draw_story_page(canvas: Canvas, number: int, text: str) -> None:
    # The upper _ART_FRACTION of the page is left untouched. Genuinely blank
    # rather than a boxed placeholder: blank space reads as deliberate in a
    # text-only book, and a caption saying "illustration here" would not.
    width = _PAGE - 2 * _MARGIN
    height = _PAGE * (1 - _ART_FRACTION) - 2 * _MARGIN

    paragraph = Paragraph(text.replace("\n", "<br/>"), _BODY)
    # Measured before drawing, because a Frame silently drops what does not fit
    # and there is no return value that reports it.
    _, used = paragraph.wrap(width, height)
    if used > height:
        raise StoryStructureError(
            f"page {number} does not fit its frame: needs {used:.0f}pt of "
            f"{height:.0f}pt available. Refusing to truncate."
        )

    Frame(_MARGIN, _MARGIN, width, height, showBoundary=0).addFromList(
        [paragraph], canvas
    )

    canvas.setFont("Helvetica", 9)
    canvas.drawCentredString(_PAGE / 2, _MARGIN * 0.5, str(number))
