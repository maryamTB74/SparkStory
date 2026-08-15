"""Outline Critic: judges a story plan before any prose is paid for.

**Why a critic on the plan at all, and not only on finished prose.** The worst
failures read in real runs live in the outline: the resolution arrived by
coincidence, and the child was not the protagonist -- the want belonged to the fox
and she only helped. Neither is fixable downstream. A prose critic can report "the
child is not the protagonist"; the Writer cannot act on it, because whose story it
is was decided one stage earlier. Fixing it here costs one re-plan of a small
artifact instead of a whole-book rewrite.

**Why a critic and not a better prompt.** The planner prompt already asserts that
the child is always the main character. It did not take. That is the failure mode
a prompt tweak cannot reach.

**Where a parent's feedback will attach.** The shape a confirmation step wants is
a first-class ``human_feedback`` input, ranked above every other requirement and
always turned into at least one action point. It is not a parameter here yet
because nothing supplies one, and a field no caller can fill is speculation -- but
when it arrives
it belongs in this constructor and at the top of the priority order, not bolted on
downstream.
"""

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable

from sparkstory.entities.reviews import OutlineReviews, OutlineReviewsOutput
from sparkstory.entities.stories import StoryBrief, StoryOutline
from sparkstory.nodes.base import Node
from sparkstory.utils.logging_utils import get_logger

logger = get_logger(__name__)


OUTLINE_CRITIC_SYSTEM_PROMPT = """\
You are an experienced children's picture-book editor reading a colleague's story \
plan before anyone writes it. Your job is to find what will not work, and to say \
why clearly enough that they can fix it.

Judge the plan against exactly two requirements.

**Protagonist.** The book is written for one particular child, and that child \
must *drive* it. They must want something, act on it, and the ending must turn on \
what they did. A child who is present on every page but desires nothing is a \
helper in someone else's story, and that makes a personalised book markedly less \
personal. If the child is passive, say so.

**When the parent's idea is about someone else, follow the parent.** A premise \
like "an eagle who discovers a new planet" or "a fox who wants to visit the moon" \
names another character on purpose, and rewriting it into a story about the child \
alone gives the parent a book they did not ask for. In that case the child shares \
the story: they want something of their own *within* it -- to help, to find out, \
to fix -- and they are the one whose actions decide how it ends. That is enough. \
Do not report a finding merely because another character also wants something.

**What is never enough**, whoever the premise names: a child who only watches, \
reacts, or is told what to do. A want the child is given rather than has. An \
ending that would resolve identically with the child removed.

Judge this against the **whole plan**, not the beats alone. Whose story it is runs \
through the logline, the theme and every character description as well. A plan \
whose beats have been adjusted so the child "decides" to help, while the logline \
still describes watching and the theme is still about someone else entirely, has \
not been fixed -- it has been patched.

**Earned resolution.** The ending must come from something the child does, \
decides or realises. Not from a coincidence, not from something conveniently \
found lying nearby, not from an adult stepping in to solve it. Test it like this: \
if you can remove the child from the last beat and the story still resolves, it \
is not earned.

How to review:
- Only flag what is genuinely wrong. A plan that meets both requirements needs no \
reviews at all, and you should **return nothing** rather than invent a concern. \
An empty review is the normal outcome for a good plan, and it is what tells us \
the plan is ready.
- Say *why* it is wrong and what it costs the child reading it, not only what is \
wrong.
- Point at the beat when the problem is in one beat. Leave the beat empty when it \
is about the story as a whole.
- **Do not rewrite the plan.** Somebody else does that, with your notes in hand.
- Do not comment on wording, page counts, character names, or anything you were \
not asked to judge."""


def render_outline_review_request(
    brief: StoryBrief, outline: StoryOutline, max_reviews: int
) -> str:
    """Render the brief and the plan as the human half of the critic prompt."""
    child = brief.child
    lines = [
        f"Review this story plan. Return at most {max_reviews} reviews, most "
        "important first.",
        "",
        # The premise is included because "is the child the protagonist" cannot
        # be judged without knowing whose story the parent asked for.
        f"The book is for: {child.name}, age {child.age}, {child.pronouns.value}",
        f"Premise the parent asked for: {brief.premise}",
        "",
        f"Title: {outline.title}",
        f"Logline: {outline.logline}",
        f"Theme: {outline.theme}",
        "",
        "Characters:",
    ]
    lines += [f"- {c.name} ({c.role}): {c.description}" for c in outline.characters]

    lines += ["", "Beats, in order:"]
    for beat in outline.beats:
        lines.append(
            f"- Beat {beat.position} [{beat.function.value}] {beat.title}: "
            f"{beat.summary}"
        )

    return "\n".join(lines)


class OutlineCriticNode(Node):
    """Finds what will not work in a story plan."""

    output_schema = OutlineReviewsOutput

    def __init__(
        self,
        model: Runnable[Any, Any],
        brief: StoryBrief,
        outline: StoryOutline,
        max_reviews: int = 5,
    ) -> None:
        super().__init__(model)
        self.brief = brief
        self.outline = outline
        # Enforced in the prompt rather than as a schema `max_length`. A
        # truncating constraint would keep whichever findings happened to come
        # first and silently discard the model's own ranking; an instruction
        # makes the model drop its least important ones itself.
        self.max_reviews = max_reviews

    async def ainvoke(self) -> OutlineReviews:
        """Review the plan.

        Returns:
            An :class:`OutlineReviews` carrying the outline and its reviews. An
            empty review list means the plan met both requirements, and is the
            workflow's signal to stop revising -- not a failure to review.
        """
        logger.info("Reviewing outline: beats=%d", len(self.outline.beats))

        found: OutlineReviewsOutput = await self.model.ainvoke(
            [
                SystemMessage(content=OUTLINE_CRITIC_SYSTEM_PROMPT),
                HumanMessage(
                    content=render_outline_review_request(
                        self.brief, self.outline, self.max_reviews
                    )
                ),
            ]
        )

        logger.info("Outline reviews: %d", len(found.reviews))
        return OutlineReviews(outline=self.outline, reviews=found.reviews)
