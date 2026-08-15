"""Assemble one book's scorecard from both halves.

The computed half cannot fail: it is arithmetic over data Pydantic already
validated. The judged half is a model call, so it can fail, and it can fail after
the expensive part of a measurement is already done. Those two facts are why the
halves are separate fields and why a judge failure degrades a scorecard rather than
losing it -- the course does the same thing with a per-sample error string, so one
bad sample does not cost the other nine.
"""

import logging

from sparkstory.entities.stories import Story
from sparkstory.evals.metrics.deterministic import deterministic_scores
from sparkstory.evals.metrics.judge import BookJudge, aggregate_page_scores
from sparkstory.evals.metrics.types import BookScorecard

logger = logging.getLogger(__name__)


async def score_book(
    story: Story,
    *,
    name: str,
    notes: list[str],
    judge: BookJudge | None = None,
) -> BookScorecard:
    """Measure one book.

    Args:
        story: The finished book.
        name: Label for this book in the report, normally its run directory name.
        notes: Grounding ``story_note`` values, for the recital metrics. Empty when
            the run was ungrounded, which records ``None`` rather than 0 -- nothing
            was recited because there was nothing to recite.
        judge: An optional judge. Without one only the computed metrics run, which
            is free and needs no network.

    Returns:
        A scorecard whose computed numbers are always present, and whose judged
        numbers are present only if the judge answered usably.
    """
    card = BookScorecard(
        name=name,
        page_count=len(story.pages),
        deterministic=deterministic_scores(story, notes),
    )
    if judge is None:
        return card

    try:
        scores = await judge.ainvoke()
        card.judged = aggregate_page_scores(scores, page_count=len(story.pages))
        # After aggregation, never before: `aggregate_page_scores` raises when the
        # judge did not cover the book exactly once, and a scorecard must not carry
        # verdicts that failed that check -- those are precisely the ones an
        # alignment run would compare against the wrong denominator.
        card.judged_pages = scores
    # Broad by intention: any judge failure -- provider, schema, or a response that
    # does not cover the book -- must leave the computed half intact, because those
    # numbers are already paid for and cost nothing to keep. The error is recorded
    # on the card so an absent judged half is never silent.
    except Exception as error:  # noqa: BLE001
        card.judge_error = f"{type(error).__name__}: {error}"
        logger.warning("Judge failed for %s: %s", name, card.judge_error)

    return card
