"""Tests for the human-label schema."""

import json
from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import ValidationError

from sparkstory.entities.stories import Story
from sparkstory.evals.labels import (
    LABEL_DIMENSIONS,
    BookLabels,
    PageLabel,
    load_labels,
    skeleton,
)
from sparkstory.evals.metrics.types import BookScores


def test_underscore_text_is_ignored() -> None:
    """`_text` is labelling scaffolding, not data.

    It is inlined into a skeleton so labelling needs no second window, and it must
    never become a field the comparison could read.
    """
    page = PageLabel.model_validate(
        {
            "page_number": 1,
            "_text": "Sam holds Ted close.",
            "delight": 0,
            "showing": 1,
            "momentum": 0,
        }
    )

    assert not hasattr(page, "_text")
    assert page.delight == 0


def test_unlabelled_page_is_none_not_zero() -> None:
    """An unfilled skeleton must be distinguishable from a book of zeros."""
    page = PageLabel.model_validate({"page_number": 1})

    assert page.delight is None
    assert page.showing is None
    assert page.momentum is None


@pytest.mark.parametrize("bad", [2, -1, 5])
def test_score_outside_zero_one_is_rejected(bad: int) -> None:
    """Binary only, matching `CriterionScore`'s bounds.

    There is deliberately no "unsure" value: a third state needs a rule for how it
    compares against a binary judge, and every candidate rule -- count as a
    disagreement, drop the page, count as a match -- changes the denominator in a
    way that is arguable.
    """
    with pytest.raises(ValidationError):
        PageLabel.model_validate({"page_number": 1, "delight": bad})


def test_load_labels_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "teddy-bear.json"
    path.write_text(
        json.dumps(
            {
                "book": "eval-teddy-bear",
                "labeller": "maryam",
                "pass_number": 1,
                "pages": [
                    {
                        "page_number": 1,
                        "_text": "ignored",
                        "delight": 0,
                        "showing": 1,
                        "momentum": 0,
                        "note": "flat report",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    labels = load_labels(path)

    assert labels.book == "eval-teddy-bear"
    assert labels.pass_number == 1
    assert labels.pages[0].note == "flat report"


def test_duplicate_page_numbers_are_rejected() -> None:
    """Two labels for one page would silently overwrite each other in `by_page`."""
    with pytest.raises(ValidationError):
        BookLabels.model_validate(
            {
                "book": "x",
                "labeller": "maryam",
                "pass_number": 1,
                "pages": [{"page_number": 1}, {"page_number": 1}],
            }
        )


def test_label_dimensions_match_the_judge() -> None:
    """A dimension added to one side and not the other would go unscored."""
    assert LABEL_DIMENSIONS == BookScores.JUDGED_DIMENSIONS


def test_skeleton_inlines_page_text_and_leaves_scores_null(
    book_factory: Callable[..., Story],
) -> None:
    """Labelling should be typing digits, not transcribing prose."""
    story = book_factory(["Sam holds Ted close.", "The wind moved."])

    result = skeleton(story, book="eval-teddy-bear", labeller="maryam", pass_number=1)

    assert result["book"] == "eval-teddy-bear"
    assert result["pass_number"] == 1
    pages = result["pages"]
    assert pages[0]["_text"] == "Sam holds Ted close."
    assert pages[0]["delight"] is None
    assert pages[0]["momentum"] is None
    assert pages[1]["page_number"] == 2


def test_is_complete_distinguishes_an_unstarted_skeleton(
    book_factory: Callable[..., Story],
) -> None:
    """An untouched skeleton is "not started", not "filled in wrong".

    The distinction has a real cost: the alignment report generates a pass-2
    skeleton for the ceiling, and without this an unstarted relabel crashed the
    whole run rather than being skipped -- so the report could not be re-read
    until the labelling was finished.
    """
    from sparkstory.evals.labels import is_complete

    story = book_factory(["a", "b"])
    blank = BookLabels.model_validate(
        skeleton(story, book="b", labeller="maryam", pass_number=2)
    )
    assert not is_complete(blank)

    filled = BookLabels.model_validate(
        {
            "book": "b",
            "labeller": "maryam",
            "pass_number": 2,
            "pages": [
                {"page_number": 1, "delight": 1, "showing": 0, "momentum": 1},
                {"page_number": 2, "delight": 0, "showing": 1, "momentum": 0},
            ],
        }
    )
    assert is_complete(filled)


def test_is_complete_is_false_for_a_partly_filled_file(
    book_factory: Callable[..., Story],
) -> None:
    """Half-labelled is not complete, and must not be silently scored."""
    from sparkstory.evals.labels import is_complete

    partial = BookLabels.model_validate(
        {
            "book": "b",
            "labeller": "maryam",
            "pass_number": 1,
            "pages": [
                {"page_number": 1, "delight": 1, "showing": 0, "momentum": 1},
                {"page_number": 2, "delight": 0, "showing": None, "momentum": 0},
            ],
        }
    )

    assert not is_complete(partial)


def test_a_skeleton_is_loadable_but_not_scoreable(
    book_factory: Callable[..., Story],
) -> None:
    """It must parse, and it must refuse to be compared until it is filled in.

    The pair matters: a skeleton that parsed *and* scored would report perfect
    agreement for a book nobody had labelled.
    """
    from sparkstory.evals.alignment import agreement

    story = book_factory(["a page"])
    labels = BookLabels.model_validate(
        skeleton(story, book="b", labeller="maryam", pass_number=1)
    )

    assert labels.pages[0].delight is None
    with pytest.raises(ValueError, match="unlabelled"):
        agreement(labels, labels)
