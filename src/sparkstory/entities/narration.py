"""What a narration run produced, and what it failed to produce.

The audio counterpart of ``StoryArt``, and like it this is **our record**: no
model produces a ``StoryNarration``, nothing here is ever bound as an output
schema, and no docstring in this module is prompt text. That is the same split
``entities/illustration.py`` argues for -- mixing the two would mean a model
writing into the fields we use to decide whether the feature worked.

There is no ``IllustrationPlan`` equivalent here, and that absence is the whole
shape of this feature. A picture has to be invented, so illustration needs an
agent to decide appearances, a style bible and per-page prompts. A page's
narration script is ``StoryPage.text``. Nothing is decided, so nothing is
planned, and no model is involved between the finished book and the audio.

**Paths, not bytes.** Following ``StoryArt`` for a reason finding O already
recorded against the web ledger: base64 audio in a Pydantic model would reach
``story.json``, every log line and every run artifact. An 80 KB page of audio is
considerably worse than a scraped page snippet.

**``sha256`` is what makes "the audio matches the printed page" checkable**
rather than merely intended. Narration speaks ``page.text`` verbatim -- no model
rewrites it -- and this field is how a finished run proves that after the fact.
It is the same move ``drop_unprovenanced`` makes for a citation: convert a claim
we would otherwise have to trust into one we can check.
"""

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel


class NarrationStatus(StrEnum):
    """Whether one page ended up with audio."""

    NARRATED = "narrated"
    FAILED = "failed"


class NarrationItem(BaseModel):
    """One page's narration, or the record of its failure."""

    page_number: int
    status: NarrationStatus
    # `None` exactly when `status` is FAILED. Absent rather than empty, and that
    # is deliberate: a zero-byte MP3 on disk plays as silence, and silence is
    # indistinguishable from success on a casual listen.
    path: Path | None
    # SHA-256 of the page text this audio was generated from. Recorded even for a
    # failed page, so a later run can tell "this page was never narrated" from
    # "this page was narrated from different words".
    sha256: str


class StoryNarration(BaseModel):
    """Every page of a book's narration, plus the stitched whole."""

    # The provider id actually sent, not the `Voice` a parent chose. Recorded so a
    # run answers "which voice was this?" by reading a file rather than
    # re-deriving the mapping -- the same reason `meta.json` records
    # `world_rules`, without which finding L could not have been written.
    voice_id: str
    speed: float
    items: list[NarrationItem]
    # `None` when nothing was stitched, which includes the all-failed run.
    stitched: Path | None

    @property
    def pages_narrated(self) -> int:
        """How many pages have audio. Read against ``len(items)`` for "6 of 8"."""
        return sum(1 for i in self.items if i.status is NarrationStatus.NARRATED)

    @property
    def is_complete(self) -> bool:
        """True only when every page narrated *and* there was a page at all.

        The ``bool(self.items)`` half is load-bearing rather than defensive:
        ``all([])`` is ``True``, so without it a run that narrated nothing would
        report as fully narrated. Rule 24 -- a check with no room to fail proves
        nothing, and this one would have failed in the direction that looks like
        success.
        """
        return bool(self.items) and all(
            i.status is NarrationStatus.NARRATED for i in self.items
        )

    def page_audio(self, page_number: int) -> Path | None:
        """The audio for one page, or ``None`` if it has none.

        Mirrors ``StoryArt.page_image``: the caller is not asked to tell a page
        that failed from a page that was never attempted, because for the thing
        it is deciding -- can I play this page? -- they are the same page.
        """
        for item in self.items:
            if item.page_number == page_number:
                return item.path
        return None
