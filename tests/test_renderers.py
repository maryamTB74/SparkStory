"""The PDF renderer.

Structural assertions only. Extracting text back out of the PDF would need a
second dependency and would mostly test reportlab; what these can prove is that
a file was produced, that it is a PDF, and that it has one page per story page
plus a title page. Whether it *looks* like a book cannot be reasoned about and
is answered only by opening the file.
"""

from pathlib import Path

import pytest

from sparkstory.entities.exceptions import StoryStructureError
from sparkstory.entities.stories import Story, StoryPage
from sparkstory.renderers import render_pdf


def test_writes_a_pdf_file(story: Story, tmp_path: Path) -> None:
    out = tmp_path / "story.pdf"

    render_pdf(story, out)

    assert out.exists()
    assert out.stat().st_size > 0


def test_output_is_a_pdf(story: Story, tmp_path: Path) -> None:
    out = tmp_path / "story.pdf"

    render_pdf(story, out)

    assert out.read_bytes().startswith(b"%PDF")


def test_one_page_per_story_page_plus_a_title_page(
    story: Story, tmp_path: Path
) -> None:
    out = tmp_path / "story.pdf"

    render_pdf(story, out)

    # Counting `/Type /Page` objects is the cheapest structural page count that
    # needs no reader library. The trailing delimiter excludes `/Type /Pages`,
    # the tree node, which would otherwise inflate the count by one.
    body = out.read_bytes()
    found = body.count(b"/Type /Page\n") + body.count(b"/Type /Page ")

    assert found == len(story.pages) + 1


def test_overflowing_text_raises_rather_than_truncating(
    story: Story, tmp_path: Path
) -> None:
    # A frame that silently drops text is the failure mode worth refusing: the
    # book would look finished and be missing words. StoryPage caps text at
    # 1200 characters, so this is not reachable from a real run -- which is
    # exactly why it needs a test rather than a comment.
    flooded = story.model_copy(
        update={
            "pages": [
                StoryPage(page_number=1, text="word " * 240),
                *story.pages[1:],
            ]
        }
    )

    with pytest.raises(StoryStructureError, match="page 1"):
        render_pdf(flooded, tmp_path / "story.pdf")
