"""How often two labellers say the same thing about the same book.

Lesson 30's alignment score, which the course states as a LaTeX formula in a
markdown cell and never implements::

    agreement = sum(human == judge) / n

Per dimension, compared **per page**. Book-level agreement would compare two
averages, and two means can match while every page disagrees -- an error on one
page cancelling an opposite error on another.

Raw agreement, following the course, **plus chance-corrected agreement, which the
design argued against and the first real measurement immediately needed.** The
argument was that kappa adds a second number needing its own interpretation. That
is true and it is outweighed: the first run scored raw agreement 0.475 on
``delight``, which reads as *partial agreement* and is in fact **worse than
chance** — two labellers with those base rates agree 0.507 of the time by luck
alone. A raw number cannot distinguish "they half agree" from "they do not agree
at all", and that is the distinction the whole instrument exists to draw.

So ``AlignmentScores`` carries both. Read ``kappa`` first: ~0 means no agreement
beyond chance regardless of how respectable the raw figure looks.

**This measures application of a rubric, not the rubric.** A labeller's verdicts
are independent of the judge and not of the rubric they approved, so a judge
scoring 100% here is a well-calibrated judge of whatever the rubric asks.
Closing that gap needs a labeller who did not approve it.

Nothing here gates anything. Lesson 30's own judge scored 62.5% and the lesson
proceeded; a low number is a result, not a failure.
"""

from pydantic import BaseModel, Field

from sparkstory.evals.labels import LABEL_DIMENSIONS, BookLabels
from sparkstory.evals.metrics.types import BookScores


class AlignmentScores(BaseModel):
    """Agreement per dimension between two labellers, over one or more books."""

    delight: float
    showing: float
    momentum: float
    #: Pages compared. Reported because 0.875 over 8 pages and over 40 are very
    #: different claims, and the denominator is the only thing that says which.
    n: int
    #: Cohen's kappa per dimension: agreement above what the two base rates would
    #: produce by luck. **Read this before the raw figures.** 0 means no agreement
    #: beyond chance however respectable the raw number looks; negative means worse
    #: than chance. 0.0 when one labeller was constant, where kappa is undefined.
    kappa: dict[str, float] = Field(default_factory=dict)
    #: How often each side said 1, per dimension: `{"delight": (human, judge)}`.
    #: Kept so `pooled` can recompute kappa over every page rather than averaging
    #: per-book kappas, which is not a valid operation on a chance-corrected
    #: statistic.
    ones: dict[str, tuple[int, int]] = Field(default_factory=dict)
    #: "p3 delight: 1 vs 0", one per disagreement, in page order.
    disagreements: list[str] = Field(default_factory=list)


def from_judge(scores: BookScores, *, book: str) -> BookLabels:
    """Adapt a judge's answer into the label shape.

    The adaptation lives here rather than inside ``agreement`` so that comparing
    two humans and comparing a human to a judge stay one code path -- which is
    what makes the two resulting numbers comparable rather than merely alike.

    Args:
        scores: The judge's per-page verdicts.
        book: The book name to record on the result.

    Returns:
        The same verdicts as labels, attributed to ``"judge"``.
    """
    return BookLabels(
        book=book,
        labeller="judge",
        pass_number=1,
        pages=[
            {
                "page_number": page.page_number,
                **{
                    dimension: getattr(page, dimension).score
                    for dimension in LABEL_DIMENSIONS
                },
            }
            for page in scores.pages
        ],
    )


def agreement(a: BookLabels, b: BookLabels) -> AlignmentScores:
    """Percent agreement per dimension, compared page by page.

    Symmetric in ``a`` and ``b``, which is what lets the ceiling measurement
    (one person's two passes) reuse this unchanged.

    Raises rather than skipping whenever the two do not describe the same pages,
    or whenever a compared verdict is unlabelled. Both failures would otherwise
    inflate the score: a skipped page shrinks the denominator, and a null scored
    as a match is a comparison that cannot fail in the direction it is most
    likely to be wrong.

    Args:
        a: One labeller's verdicts.
        b: The other's, over the same pages.

    Returns:
        One agreement value per dimension, the page count compared, and every
        disagreement as readable text.

    Raises:
        ValueError: If the page sets differ, or any compared verdict is None.
    """
    left, right = a.by_page(), b.by_page()
    if left.keys() != right.keys():
        raise ValueError(
            f"labels cover different pages: {sorted(left)} vs {sorted(right)}"
        )

    matches = dict.fromkeys(LABEL_DIMENSIONS, 0)
    ones_a = dict.fromkeys(LABEL_DIMENSIONS, 0)
    ones_b = dict.fromkeys(LABEL_DIMENSIONS, 0)
    disagreements: list[str] = []

    for number in sorted(left):
        for dimension in LABEL_DIMENSIONS:
            one = getattr(left[number], dimension)
            two = getattr(right[number], dimension)
            if one is None or two is None:
                raise ValueError(
                    f"p{number} {dimension} is unlabelled; "
                    "a null must not be scored as agreement"
                )
            ones_a[dimension] += one
            ones_b[dimension] += two
            if one == two:
                matches[dimension] += 1
            else:
                disagreements.append(f"p{number} {dimension}: {one} vs {two}")

    total = len(left)
    ones = {d: (ones_a[d], ones_b[d]) for d in LABEL_DIMENSIONS}
    return AlignmentScores(
        **{dimension: matches[dimension] / total for dimension in LABEL_DIMENSIONS},
        n=total,
        kappa={d: _kappa(matches[d] / total, ones[d], total) for d in LABEL_DIMENSIONS},
        ones=ones,
        disagreements=disagreements,
    )


def _kappa(observed: float, ones: tuple[int, int], total: int) -> float:
    """Cohen's kappa from observed agreement and each side's rate of saying 1.

    Args:
        observed: The share of pages the two agreed on.
        ones: How many 1s each labeller gave.
        total: Pages compared.

    Returns:
        Agreement above chance, or 0.0 when chance agreement is 1.0 -- which
        happens when both labellers were constant, and where kappa is 0/0. A
        constant labeller carries no information, so 0.0 says the right thing;
        raising would lose an otherwise valid report, and 1.0 would claim perfect
        agreement for a labeller who never looked at the page.
    """
    rate_a, rate_b = ones[0] / total, ones[1] / total
    expected = rate_a * rate_b + (1 - rate_a) * (1 - rate_b)
    if expected >= 1.0:
        return 0.0
    return (observed - expected) / (1 - expected)


def pooled(results: list[AlignmentScores]) -> AlignmentScores:
    """Combine per-book agreement into one figure per dimension.

    Weighted by pages, not by book: an 8-page and a 12-page book contribute 8 and
    12 comparisons. A mean of means would weight a short book equally, which
    silently changes what the number is a proportion of.

    Args:
        results: One entry per book.

    Returns:
        Agreement over every page of every book, with all disagreements kept.

    Raises:
        ValueError: If ``results`` is empty.
    """
    if not results:
        raise ValueError("no books to pool")

    total = sum(result.n for result in results)
    values = {
        dimension: sum(getattr(result, dimension) * result.n for result in results)
        / total
        for dimension in LABEL_DIMENSIONS
    }
    # Kappa is recomputed from the pooled counts, never averaged: a chance-corrected
    # statistic has a different denominator per book, so a mean of kappas is not a
    # kappa of anything.
    ones = {
        dimension: (
            sum(result.ones.get(dimension, (0, 0))[0] for result in results),
            sum(result.ones.get(dimension, (0, 0))[1] for result in results),
        )
        for dimension in LABEL_DIMENSIONS
    }
    return AlignmentScores(
        **values,
        n=total,
        kappa={d: _kappa(values[d], ones[d], total) for d in LABEL_DIMENSIONS},
        ones=ones,
        disagreements=[line for result in results for line in result.disagreements],
    )
