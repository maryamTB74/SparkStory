"""Prose Critic: judges the finished words against the plan and against the child.

**What it is deliberately not given: the outline.** It needs the page plan, for
``plan_fidelity``, and the brief, for ``safety`` and ``reading_level``. The
outline sits a level above and this node is not judging structure -- that is the
Outline Critic's job, one stage earlier. Fewer tokens, and a critic that cannot
see an artifact cannot invent findings about it.

**Why each page is rendered beside its own notes.** Session 2's book told the
plan one page late: page 4's words were page 3's moment, all the way through.
Nothing caught it, because ``validate_prose`` compares page *numbers* and those
matched perfectly. Read as a continuous story the drift is invisible; read
page-by-page against the notes it is obvious. The rendering is the check.

**One rubric here is not like the others.** ``safety`` fails closed -- a finding
that survives the last revision means no book is returned at all. The prompt says
so, because a critic that does not know the cost of a false positive will raise
one cheaply.
"""

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable

from sparkstory.entities.guidelines import READING_LEVEL_GUIDANCE
from sparkstory.entities.reviews import ProseReviews, ProseReviewsOutput
from sparkstory.entities.stories import PagePlan, StoryBrief, StoryProse
from sparkstory.nodes.base import Node
from sparkstory.utils.logging_utils import get_logger

logger = get_logger(__name__)


PROSE_CRITIC_SYSTEM_PROMPT = """\
You are a children's picture-book editor reading a finished manuscript aloud \
before it goes to the illustrator. You are listening as much as reading. Find \
what will not work for the child this book was written for.

Judge the words against five requirements.

**Plan fidelity.** Each page was written from three notes: what the picture \
shows, what changes inside the main character, and the question the page leaves \
open. The words on a page must be that page's moment -- not the previous page's, \
not a blend of two. Check each page against its own notes, in order. A page that \
quietly tells the page before's moment throws the rest of the book one behind, \
and it is easy to miss because every page still reads well on its own.

**Read aloud.** This will be spoken, probably at bedtime, probably many times. \
Listen for rhythm. Listen for a refrain worth waiting for -- for a young child \
the refrain is the pleasure. Listen for sound-play, for a line of speech, for a \
sentence that turns. Flag pages that are flat, pages that all begin the same way, \
and pages that read like a summary of events rather than a story being told.

**Interiority.** The notes said what changes inside the main character on each \
page. The feeling has to be *in* the words, carried by what the character does \
and notices. "She shuts her eyes and the wish grows deep inside" is interiority. \
"Joy surges through" is a label pasted where the feeling should be, and it is \
**just as much a failure as leaving the feeling out** -- often worse, because it \
reads as filler and stops anyone looking again. Flag both: a page with no inner \
change at all, and a page that announces the change instead of showing it.

**Reading level.** Vocabulary and sentence length must suit the child's level, \
which you are given below. Too hard shuts the child out; too easy bores them.

**Safety.** Anything the parent asked to avoid must not appear: not directly, not \
alluded to, not by a near-synonym. Peril must stay gentle and resolved -- a \
character may be worried, lost or disappointed, never endangered or humiliated, \
and nothing may humiliate the child the book is written for. Raise this only when \
something is genuinely there. A safety finding that cannot be fixed means the \
family gets no book at all, so a wrong one costs them the whole book.

How to review:
- Only flag what is genuinely wrong. If the words meet every requirement, \
**return nothing** rather than invent a concern. An empty review is a normal \
outcome and is what tells us the book is ready.
- Say *why* it fails and what it costs a child hearing it, not only what fails.
- **Never quote the notes back in your comment.** The author reads your words \
with the notes already in front of them, and a note repeated in a review is a \
phrase they will paste onto the page instead of writing one. Describe what is \
missing; do not supply the wording.
- Name the page when the problem is on one page. Leave the page empty when it is \
about the book as a whole, such as every page opening the same way.
- **Do not rewrite the words.** Somebody else does that, with your notes in hand.
- Do not comment on the pictures, the page count, or the plan itself. The plan \
was reviewed before the words were written."""


def render_prose_review_request(
    brief: StoryBrief, page_plan: PagePlan, prose: StoryProse, max_reviews: int
) -> str:
    """Render the child, the plan and the words as the human half of the prompt."""
    child = brief.child
    lines = [
        f"Review this manuscript. Return at most {max_reviews} reviews, most "
        "important first.",
        "",
        f"The book is for: {child.name}, age {child.age}, {child.pronouns.value}",
        f"Reading level: {child.reading_level.value}",
        f"  Guidance: {READING_LEVEL_GUIDANCE[child.reading_level]}",
    ]
    if brief.avoid:
        lines.append(f"Must not appear anywhere: {', '.join(brief.avoid)}")
    if brief.must_include:
        lines.append(f"The parent asked for: {', '.join(brief.must_include)}")

    # Each page beside its own notes. Read as a continuous story a one-page drift
    # is invisible; paired like this it is obvious. This layout is the check.
    by_number = {page.page_number: page for page in prose.pages}
    lines += ["", "Pages, each with the notes it was written from:"]
    for planned in page_plan.pages:
        written = by_number.get(planned.page_number)
        lines += [
            "",
            f"Page {planned.page_number}",
            f"  notes -- shows: {planned.visual_action}",
            f"  notes -- inside: {planned.emotional_shift}",
        ]
        if planned.page_turn_hook:
            lines.append(f"  notes -- leaves open: {planned.page_turn_hook}")
        # Marked rather than skipped: a page silently absent from the list reads
        # to the critic as a page with nothing wrong with it.
        lines.append(f"  words: {written.text if written else '(missing)'}")

    return "\n".join(lines)


class ProseCriticNode(Node):
    """Finds what will not work in the finished words."""

    output_schema = ProseReviewsOutput

    def __init__(
        self,
        model: Runnable[Any, Any],
        brief: StoryBrief,
        page_plan: PagePlan,
        prose: StoryProse,
        max_reviews: int = 5,
    ) -> None:
        super().__init__(model)
        self.brief = brief
        self.page_plan = page_plan
        self.prose = prose
        # In the prompt, not the schema: a truncating `max_length` would keep
        # whichever findings came first and discard the model's own ranking.
        self.max_reviews = max_reviews

    async def ainvoke(self) -> ProseReviews:
        """Review the manuscript.

        Returns:
            A :class:`ProseReviews` carrying the prose and its reviews. An empty
            review list is the workflow's signal to stop rewriting.
        """
        logger.info("Reviewing prose: pages=%d", len(self.prose.pages))

        found: ProseReviewsOutput = await self.model.ainvoke(
            [
                SystemMessage(content=PROSE_CRITIC_SYSTEM_PROMPT),
                HumanMessage(
                    content=render_prose_review_request(
                        self.brief, self.page_plan, self.prose, self.max_reviews
                    )
                ),
            ]
        )

        logger.info("Prose reviews: %d", len(found.reviews))
        return ProseReviews(prose=self.prose, reviews=found.reviews)
