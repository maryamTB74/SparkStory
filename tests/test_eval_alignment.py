"""Tests for judge-versus-human agreement."""

import pytest

from sparkstory.evals.alignment import agreement, from_judge, pooled
from sparkstory.evals.labels import BookLabels
from sparkstory.evals.metrics.types import BookScores, CriterionScore, PageScore


def _labels(*rows: tuple[int, int, int, int]) -> BookLabels:
    """Build labels from `(page, delight, showing, momentum)` rows."""
    return BookLabels(
        book="b",
        labeller="test",
        pass_number=1,
        pages=[
            {"page_number": p, "delight": d, "showing": s, "momentum": m}
            for p, d, s, m in rows
        ],
    )


def test_identical_labels_agree_completely() -> None:
    rows = [(1, 1, 0, 1), (2, 0, 1, 0)]
    result = agreement(_labels(*rows), _labels(*rows))

    assert result.delight == 1.0
    assert result.showing == 1.0
    assert result.momentum == 1.0
    assert result.n == 2
    assert result.disagreements == []


def test_opposite_labels_agree_never() -> None:
    result = agreement(
        _labels((1, 1, 1, 1), (2, 1, 1, 1)),
        _labels((1, 0, 0, 0), (2, 0, 0, 0)),
    )

    assert result.delight == 0.0
    assert result.showing == 0.0
    assert result.momentum == 0.0


def test_one_page_differing_of_eight() -> None:
    """The arithmetic the whole feature rests on."""
    same = [(p, 1, 1, 1) for p in range(1, 9)]
    other = [(p, 1, 1, 1) for p in range(1, 8)] + [(8, 0, 1, 1)]
    result = agreement(_labels(*same), _labels(*other))

    assert result.delight == 0.875
    assert result.showing == 1.0
    assert result.n == 8


def test_disagreements_are_reported_by_page_and_dimension() -> None:
    """A number localises a problem; the list is what makes it readable.

    A score does not settle a question, so the words behind it have to be
    reachable.
    """
    result = agreement(_labels((1, 1, 0, 1)), _labels((1, 0, 0, 1)))

    assert len(result.disagreements) == 1
    assert "p1" in result.disagreements[0]
    assert "delight" in result.disagreements[0]


def test_missing_page_raises() -> None:
    """Never divide by a smaller denominator; never default to agreement."""
    with pytest.raises(ValueError, match="different pages"):
        agreement(_labels((1, 1, 1, 1), (2, 1, 1, 1)), _labels((1, 1, 1, 1)))


def test_unlabelled_dimension_raises() -> None:
    """A null must not be scored as a match.

    A check with no room to fail proves nothing, and that applies to the
    instrument itself: a comparison that cannot fail in the direction it is most
    likely to be wrong is worthless.
    """
    partial = BookLabels(
        book="b",
        labeller="test",
        pass_number=1,
        pages=[{"page_number": 1, "delight": None, "showing": 1, "momentum": 1}],
    )
    with pytest.raises(ValueError, match="delight"):
        agreement(partial, _labels((1, 1, 1, 1)))


def test_from_judge_adapts_scores_into_labels() -> None:
    scores = BookScores(
        pages=[
            PageScore(
                page_number=1,
                delight=CriterionScore(score=1, reason="r"),
                showing=CriterionScore(score=0, reason="r"),
                momentum=CriterionScore(score=1, reason="r"),
            )
        ]
    )

    labels = from_judge(scores, book="eval-teddy-bear")

    assert labels.labeller == "judge"
    assert labels.book == "eval-teddy-bear"
    assert labels.pages[0].delight == 1
    assert labels.pages[0].showing == 0


def test_agreement_is_symmetric() -> None:
    """Human-vs-judge and human-vs-human are one function.

    The ceiling measurement is `agreement(pass_1, pass_2)` -- the same call with
    different arguments -- which is what makes "the judge scored 0.75" and
    "Maryam scored 0.75 against herself" comparable numbers.
    """
    a = _labels((1, 1, 0, 1), (2, 0, 1, 0))
    b = _labels((1, 1, 1, 1), (2, 0, 1, 0))

    assert agreement(a, b).delight == agreement(b, a).delight
    assert agreement(a, b).showing == agreement(b, a).showing
    assert agreement(a, b).n == agreement(b, a).n


def test_pooled_weights_by_page_count_not_by_book() -> None:
    """Two books of different lengths contribute their pages, not their means."""
    short = agreement(_labels((1, 1, 1, 1)), _labels((1, 0, 1, 1)))  # 0 of 1
    long_rows = [(p, 1, 1, 1) for p in range(1, 4)]
    long_book = agreement(_labels(*long_rows), _labels(*long_rows))  # 3 of 3

    result = pooled([short, long_book])

    # 3 of 4 pages agreed, not the mean of 0.0 and 1.0.
    assert result.delight == 0.75
    assert result.n == 4


def test_pooled_concatenates_disagreements() -> None:
    a = agreement(_labels((1, 1, 1, 1)), _labels((1, 0, 1, 1)))
    b = agreement(_labels((1, 1, 1, 1)), _labels((1, 1, 0, 1)))

    result = pooled([a, b])

    assert len(result.disagreements) == 2


def test_kappa_is_zero_when_agreement_is_only_chance() -> None:
    """Two labellers who both say 1 most of the time agree often and mean nothing.

    The first live run is why this exists: raw `delight` agreement was 0.475,
    which reads as partial agreement and was in fact *below* the 0.507 expected
    from the two base rates alone.
    """
    # Both say 1 on 3 of 4 pages, but never on the same page pattern beyond luck.
    a = _labels((1, 1, 1, 1), (2, 1, 1, 1), (3, 1, 1, 1), (4, 0, 0, 0))
    b = _labels((1, 1, 1, 1), (2, 1, 1, 1), (3, 0, 0, 0), (4, 1, 1, 1))

    result = agreement(a, b)

    assert result.delight == 0.5
    # chance = 0.75*0.75 + 0.25*0.25 = 0.625, so observed 0.5 is worse than chance
    assert result.kappa["delight"] < 0


def test_kappa_is_one_for_perfect_agreement() -> None:
    rows = [(1, 1, 0, 1), (2, 0, 1, 0)]
    result = agreement(_labels(*rows), _labels(*rows))

    assert result.kappa["delight"] == 1.0


def test_kappa_is_defined_when_one_side_is_constant() -> None:
    """A labeller who says 1 on every page makes chance agreement 1.0.

    Kappa is undefined there (0/0). It must report 0.0 -- no information -- rather
    than raising or reporting a perfect score.
    """
    always = _labels((1, 1, 1, 1), (2, 1, 1, 1))
    result = agreement(always, always)

    assert result.kappa["delight"] == 0.0


def test_pooled_carries_kappa() -> None:
    """Pooling recomputes kappa over all pages rather than averaging kappas."""
    a = agreement(
        _labels((1, 1, 1, 1), (2, 0, 0, 0)), _labels((1, 1, 1, 1), (2, 0, 0, 0))
    )
    result = pooled([a])

    assert result.kappa["delight"] == 1.0


def test_pooled_rejects_an_empty_list() -> None:
    """Nothing to pool is a caller error, not a score of zero."""
    with pytest.raises(ValueError, match="no books"):
        pooled([])
