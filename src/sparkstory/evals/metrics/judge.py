"""A judge that scores a finished book page by page.

Three dimensions, none of which the prose critic uses. That is deliberate: the
critic already scores plan fidelity, read-aloud quality, interiority, reading level
and safety on every run, and it is the thing under test. A judge sharing its
rubrics would measure the critic and inherit its blind spots, so a number moving
would tell us nothing about the book.

One call per book rather than one per page. Cheaper, and ``momentum`` is a question
about sequence -- whether this page pulls toward the next -- which a judge shown a
single page in isolation cannot answer.

**Measured against human labels on 2026-08-13, and it failed.** Cohen's kappa was
**-0.066 delight, +0.121 showing, -0.060 momentum** over 40 pages of five books --
agreement no better than chance. ``momentum`` scored 1 on 35 of 40 pages, so it is
close to a constant. The rubric also admits two defensible readings of
``momentum`` -- a page ending on a question literally leaves something open, while
a formulaic question appended to seven pages of eight does not pull a reader
forward -- and it does not choose between them.

So a judged number moving is **not** evidence that a book changed, and this is now
a measurement rather than a caveat. That is why ``BookScorecard`` keeps these
values apart from the computed ones and why nothing here may gate a build.

Two things worth knowing before revising the rubric. The disagreement on
``momentum`` is a genuine ambiguity in the wording rather than a judge error -- a
page ending "What comes after that?" does literally leave something open, so a 1
follows the rubric as written. And the human ceiling has not been measured, so
``delight``'s failure may be a question with no stable answer rather than a judge
that cannot answer it.

**``momentum`` was rewritten on 2026-08-16 to resolve that ambiguity** (finding
MM). It now judges the *situation* a page leaves behind rather than the
punctuation it ends on, and names the lazy reading it must reject -- a tacked-on
question over a settled scene -- the way ``showing`` already names "she felt
proud". Two reasons it is worded that way. A criterion with only a positive
description gets satisfied by its cheapest reading (rule 13), and that is exactly
how the old wording failed. And ``question_ending_ratio`` already counts question
marks deterministically and for free, so a judge scoring punctuation would spend
a model call duplicating a metric that cannot be gamed -- while measuring the
tic's *presence* rather than whether it does any work.

**The revision is unmeasured.** Nothing here is evidence until the harness is
re-run against labels, and the ceiling problem is untouched by it: finding NN put
one labeller's self-agreement on ``momentum`` at kappa **0.000**, so a rewritten
rubric could improve the judge and still score ~0 against an unstable target.
Re-label before concluding anything, and expect the 35-of-40 constant to move
first -- if it does not, the wording is not what was wrong.
"""

from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.runnables import Runnable

from sparkstory.entities.stories import Story
from sparkstory.evals.metrics.types import BookScores, JudgedScores
from sparkstory.nodes.base import Node

# Prompt text. Each dimension states what earns a 1 *and* what earns a 0, because
# a criterion with only a positive description is satisfied by the laziest reading
# of it -- the failure that turned "a named feeling beats an absent one" into
# pasted emotion labels. `showing` names the specific lazy answer it must reject.
_INSTRUCTIONS = """You are reading a finished picture book written for a young child.

Score every page on three questions. Give 1 or 0 and one sentence of reason.

delight: would a child ask to hear this page again? Score 1 for a page with
something to enjoy in the sound or in the picture it makes -- a rhythm, a
surprise, a line worth repeating. Score 0 for a page that only reports what
happened.

showing: is feeling conveyed through action, image or speech rather than named?
Score 1 when a child could tell how someone feels from what they do or say. Score
0 when a feeling is named outright, as in "she felt proud" or "joy filled him".

momentum: does this page make a listener want to turn it? Judge the *situation*
the page leaves behind, not the punctuation it ends on. Score 1 when something
is unresolved after the page: a character wants something they have not got, has
just tried something whose outcome is unknown, or faces a change they have not
answered yet. Score 0 when the page settles -- the want is met, the attempt has
landed, nothing is pending -- even if the last sentence is phrased as a question.
A question mark is not momentum. "What comes after that?" tacked onto a page
where nothing is outstanding scores 0, because the words invite a turn while the
story gives no reason for one. Judge the final page instead on whether it closes
the book satisfyingly, and score 1 if it does.

Score each page on its own words. Do not reward a page for what a neighbouring
page does, and do not lower a score because the book as a whole disappoints you.
Return one entry per page, in order, covering every page exactly once."""


class BookJudge(Node):
    """Scores one finished book on three dimensions, one entry per page."""

    output_schema = BookScores

    def __init__(self, model: Runnable[Any, Any], *, story: Story) -> None:
        """Bind the schema and keep the book to be judged.

        Args:
            model: An unbound chat model, normally from ``get_chat_model``.
            story: The finished book. Only its prose is sent -- the plan is
                withheld deliberately, so a page is judged on what a child would
                actually hear rather than on how faithfully it followed notes.
        """
        super().__init__(model)
        self._story = story

    def _prompt(self) -> str:
        """The instructions followed by the book's pages."""
        pages = "\n\n".join(
            f'<page number="{page.page_number}">\n{page.text}\n</page>'
            for page in self._story.pages
        )
        return f"{_INSTRUCTIONS}\n\n<book>\n{pages}\n</book>"

    async def ainvoke(self) -> BookScores:
        """Score every page of the book."""
        return await self.model.ainvoke([HumanMessage(content=self._prompt())])


def aggregate_page_scores(scores: BookScores, page_count: int) -> JudgedScores:
    """Average each dimension across pages, keeping every reason.

    Raises rather than averaging when the returned pages do not cover the book
    exactly once. A judge that skipped a page would otherwise divide by a smaller
    denominator and score a shorter answer higher -- a measurement that cannot
    fail in the direction it is most likely to be wrong.

    Args:
        scores: What the judge returned.
        page_count: How many pages the book actually has.

    Returns:
        One value per dimension, each the share of pages scoring 1.

    Raises:
        ValueError: If the scored page numbers are not exactly 1..page_count.
    """
    seen = [page.page_number for page in scores.pages]
    if len(seen) != len(set(seen)) or set(seen) != set(range(1, page_count + 1)):
        raise ValueError(
            f"judge scored page numbers {sorted(seen)}, "
            f"expected each of 1..{page_count} exactly once"
        )

    values: dict[str, float] = {}
    reasons: dict[str, list[str]] = {}
    for dimension in BookScores.JUDGED_DIMENSIONS:
        scored = [getattr(page, dimension) for page in scores.pages]
        values[dimension] = sum(item.score for item in scored) / len(scored)
        reasons[dimension] = [
            f"p{page.page_number}: {getattr(page, dimension).reason}"
            for page in scores.pages
        ]

    return JudgedScores(**values, reasons=reasons)
