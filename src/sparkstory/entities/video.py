"""What a video run produced, and what it failed to produce.

The third of these records, after ``StoryArt`` and ``StoryNarration``, and like
both it is **ours**: no model produces a ``StoryVideo``, nothing here is ever
bound as an output schema, and no docstring in this module is prompt text.

There is no plan entity here, and that absence is the shape of the feature. A
picture has to be invented, so illustration needs a Director for appearances, a
style bible and per-page prompts. A video of a finished book invents nothing: the
picture is the page's illustration, the length is the page's narration, and the
camera move is arithmetic on the page number. Nothing is decided, so nothing is
planned -- the same argument ``entities/narration.py`` makes.

**Paths, not bytes**, following both siblings for the same reason: a video in a
Pydantic model would reach every log line and every run artifact, and a minute of
H.264 is far worse than the scraped page text that made the point originally.
"""

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel


class VideoStatus(StrEnum):
    """What became of one page in the video.

    **Four states rather than a boolean**, for the reason the consistency judge
    had to be given a field of its own: ``ArtStatus.CONDITIONED`` reported that the
    mechanism *ran* rather than that it *worked*, and collapsing those two
    questions cost three live runs in which a fox's paws changed colour.

    ``HELD`` and ``ANIMATED`` are both successes and are not the same success. A
    book that is entirely ``HELD`` is a video of blank cards with narration over
    it, which no single boolean distinguishes from a finished product.
    """

    #: Picture and audio: a real clip with a camera move over the illustration.
    ANIMATED = "animated"
    #: Audio but no picture -- a plain card held for exactly its narration.
    HELD = "held"
    #: No audio, so no duration, so no place on the timeline. See the module
    #: docstring of ``workflows/animate.py`` for why audio is the spine.
    EXCLUDED = "excluded"
    #: Had both and clip-making failed anyway. Distinct from EXCLUDED because one
    #: is a decision and the other is a fault.
    FAILED = "failed"


class VideoItem(BaseModel):
    """One page's place in the video, or the record of its absence."""

    page_number: int
    status: VideoStatus
    #: The page's measured narration length, which is what the clip was built to
    #: fill. ``None`` exactly when ``EXCLUDED`` -- a page with no audio has no
    #: duration, and inventing one is the thing the audio-as-spine rule refuses.
    #: ``HELD`` and ``FAILED`` both have audio and so both carry a duration.
    duration: float | None
    #: Why, in words, for EXCLUDED and FAILED. ``None`` otherwise. A video that is
    #: shorter than its book must say which page went and why, or it repeats the
    #: degraded-web-path failure in a new medium: output that looks entirely fine
    #: while the mechanism did not fully run.
    reason: str | None


class StoryVideo(BaseModel):
    """Every page of a book's video, plus the file that was assembled."""

    #: ``None`` when nothing was assembled -- which includes the all-excluded run.
    #: No file at all rather than an empty one, because an empty video plays as
    #: nothing and nothing is indistinguishable from success on a casual glance.
    path: Path | None
    fps: int
    items: list[VideoItem]

    @property
    def pages_animated(self) -> int:
        """How many pages got a real clip. Read against ``len(items)`` for "5 of 6"."""
        return sum(1 for i in self.items if i.status is VideoStatus.ANIMATED)

    @property
    def is_complete(self) -> bool:
        """True only when every page reached the video *and* there was a page.

        A ``HELD`` page counts as complete: it is on the timeline for its full
        narration, which is what "the book is watchable end to end" means. Whether
        it had a picture is ``pages_animated``'s question.

        The ``bool(self.items)`` half is load-bearing rather than defensive:
        ``all([])`` is ``True``, so without it a run that animated nothing would
        report as fully animated. A check with no room to fail proves nothing, and
        this one would have failed in the direction that looks like success.
        ``StoryNarration.is_complete`` carries the same guard.
        """
        return bool(self.items) and all(
            i.status in (VideoStatus.ANIMATED, VideoStatus.HELD) for i in self.items
        )
