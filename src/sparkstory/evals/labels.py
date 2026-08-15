"""Human labels for one book, in the shape the judge answers in.

Deliberately mirrors ``BookScores`` field for field, so comparing a human against
a judge is a field lookup rather than a translation -- and so the *same* function
compares two humans, which is what the ceiling measurement needs.

These files are committed rather than written to ``outputs/``. They are the
expensive artifact here: hours of a person's attention, unreproducible by any
rerun, where a book can always be regenerated. ``outputs/`` is documented as
disposable and has already lost eight books once.
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

from annotated_types import Ge, Le
from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    from sparkstory.entities.stories import Story

#: The dimensions a label carries, matching ``BookScores.JUDGED_DIMENSIONS``.
#: There is a test asserting the two agree, so a dimension added to one and not
#: the other cannot be silently unscored.
LABEL_DIMENSIONS: tuple[str, ...] = ("delight", "showing", "momentum")

#: A binary verdict, or `None` for a page nobody has labelled yet.
#:
#: `None` rather than a default of 0, because a half-finished skeleton must fail
#: loudly rather than score as a book that earned nothing. There is deliberately
#: no third "unsure" state: it would need a rule for comparing against a binary
#: judge, and every candidate rule changes the denominator arguably.
Verdict = Annotated[int, Ge(0), Le(1)] | None


class PageLabel(BaseModel):
    """One person's verdict on one page."""

    # `_text` is inlined into the skeleton so labelling needs no second window,
    # and dropped here. Pydantic ignores unknown keys by default; this states it,
    # so a later `extra="forbid"` cannot break every skeleton silently.
    model_config = ConfigDict(extra="ignore")

    page_number: int = Field(ge=1)
    delight: Verdict = None
    showing: Verdict = None
    momentum: Verdict = None
    #: Free text carrying no weight in any score. It exists so a disagreement is
    #: readable later -- the argument `JudgedScores.reasons` already makes, and
    #: what finding X showed is needed: a number localises a suspicion without
    #: settling it, so the words have to be there to read.
    note: str = ""


class BookLabels(BaseModel):
    """Every page of one book, labelled by one person in one sitting."""

    #: Matches ``BookScorecard.name``, which is how a label file finds its book.
    book: str
    labeller: str
    #: 1 for the first pass; 2 for the blind relabel that measures the ceiling.
    pass_number: int = Field(ge=1)
    pages: list[PageLabel] = Field(min_length=1)

    @model_validator(mode="after")
    def _reject_duplicate_pages(self) -> BookLabels:
        seen = [page.page_number for page in self.pages]
        if len(seen) != len(set(seen)):
            raise ValueError(f"duplicate page numbers in labels for {self.book}")
        return self

    def by_page(self) -> dict[int, PageLabel]:
        """Labels keyed by page number."""
        return {page.page_number: page for page in self.pages}


def is_complete(labels: BookLabels) -> bool:
    """Whether every page carries a verdict on every dimension.

    Lets a caller distinguish *not started* from *filled in wrong*. ``agreement``
    raises on an unlabelled verdict, which is right when scoring and wrong when
    merely deciding whether a file is ready to score -- an untouched pass-2
    skeleton would otherwise crash a report that has nothing to do with it.

    Args:
        labels: The file to check.

    Returns:
        True when nothing is left null.
    """
    return all(
        getattr(page, dimension) is not None
        for page in labels.pages
        for dimension in LABEL_DIMENSIONS
    )


def load_labels(path: Path) -> BookLabels:
    """Read one committed label file.

    Args:
        path: The JSON file to read.

    Returns:
        The parsed labels.
    """
    return BookLabels.model_validate(json.loads(path.read_text(encoding="utf-8")))


def skeleton(
    story: Story, *, book: str, labeller: str, pass_number: int
) -> dict[str, object]:
    """A pre-filled label file with every score left null.

    The page text is inlined as ``_text`` so labelling is typing digits rather
    than transcribing prose. A transcription error would surface later as a
    disagreement and be indistinguishable from a real one -- which is the failure
    mode this whole measurement exists to avoid.

    Args:
        story: The finished book to label.
        book: The scorecard name this file will be matched against.
        labeller: Who is filling it in.
        pass_number: 1 normally; 2 for the blind relabel.

    Returns:
        A dict ready to be written as JSON.
    """
    return {
        "book": book,
        "labeller": labeller,
        "pass_number": pass_number,
        "pages": [
            {
                "page_number": page.page_number,
                "_text": page.text,
                "delight": None,
                "showing": None,
                "momentum": None,
                "note": "",
            }
            for page in story.pages
        ],
    }
