"""Turning a finished `Story` into a book-shaped PDF.

The name is `brown`'s, reserved in CLAUDE.md for book assembly. This is the
text-only half: it lays out the pages and leaves the upper 55% of every one of
them blank, which is where Session 6's illustration goes. Reserving that space
now means an image later fills a frame rather than forcing a relayout.

Deliberately pure. It takes a `Story` and a path, imports nothing from the rest
of the package except the entities, and reads no settings -- so its entire
behaviour is determined by its arguments and its tests need no fakes at all.

It takes no `StoryBrief`, so the title page carries the title and logline but
not a dedication. That is a consequence of the signature, chosen deliberately;
see docs/superpowers/specs/2026-08-03-pdf-assembly-design.md.
"""

from pathlib import Path

from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Frame, Paragraph

from sparkstory.entities.exceptions import StoryStructureError
from sparkstory.entities.stories import Story

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


def render_pdf(story: Story, path: Path) -> None:
    """Write `story` to `path` as a PDF, one page per story page.

    Raises `StoryStructureError` if a page's text does not fit its frame --
    never truncates, because a book that looks finished and is quietly missing
    words is the worst outcome available here.
    """
    canvas = Canvas(str(path), pagesize=(_PAGE, _PAGE))

    _draw_title_page(canvas, story)
    for page in story.pages:
        canvas.showPage()
        _draw_story_page(canvas, page.page_number, page.text)

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
