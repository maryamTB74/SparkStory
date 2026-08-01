"""Story Planner: turns a brief into a structured outline.

This is the first agent, and the pattern every later one follows:

1. Ask ``get_chat_model`` for a model by ``model_id``, passing the output schema.
2. Send a static system prompt plus the rendered request.
3. Return a validated domain object.

Note what is absent. There is no API key handling, no retry configuration, no
tracing setup, and no output parsing -- all of that lives behind
``get_chat_model``. An agent should read as domain logic, and if a future agent
here starts reaching for provider details, that is a signal the seam needs
widening rather than bypassing.

**Prompt text lives in this module, beside the schema it must satisfy.**
Each node carries its own ``system_prompt_template``.
Two conventions carry over from the prompts module this replaced:

*System prompts are static constants.* Only per-request data varies, and it
travels in the human message. A byte-identical prompt prefix is what allows
provider-side prompt caching, which matters once several agents each make calls
per book.

*Prompts describe craft, never output format.* The response shape is enforced
mechanically by ``.with_structured_output(...)``. Restating it in prose burns
tokens and risks contradicting the schema when one of the two changes.
"""

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable

from sparkstory.config import settings
from sparkstory.entities.guidelines import READING_LEVEL_GUIDANCE
from sparkstory.entities.reviews import OutlineReviews
from sparkstory.entities.stories import StoryBrief, StoryOutline
from sparkstory.models.get_model import get_chat_model
from sparkstory.nodes.base import Node
from sparkstory.utils.logging_utils import get_logger

logger = get_logger(__name__)


STORY_PLANNER_SYSTEM_PROMPT = """\
You are an experienced children's picture-book editor. You plan stories: you \
decide what happens and why it matters, before anyone writes a single line of \
prose.

A good picture-book plan has:
- One clear want. The main character wants something specific and simple, and \
the reader knows what it is by the second beat.
- A real obstacle. Something genuinely stands in the way. Without it there is \
no story, only a description.
- An earned ending. The resolution comes from something the character does or \
realises, never from a coincidence or an adult solving it for them.
- Emotional truth. Small stakes felt deeply beat large stakes felt vaguely. A \
lost toy can carry more weight than a saved kingdom.

Rules you must follow:
- Never plan more beats than the book has pages. Each beat needs at least one \
page of its own, so a five-page book cannot carry six beats. Fewer beats than \
pages is good: a short book wants four strong beats, not six rushed ones.
- The child described in the brief is always the main character, referred to by \
their given name and their stated pronouns. Never guess at pronouns.
- Anything in the brief's "avoid" list is an absolute constraint. Do not \
include it, allude to it, or use a near-synonym for it.
- Never moralise. Let the theme emerge from what happens. Do not end with a \
stated lesson.
- Keep peril gentle and always resolved. A character may be worried, lost or \
disappointed; they must not be endangered or humiliated.
- Weave the child's interests in as texture, not as a checklist. If they love \
astronomy, the story can happen under stars; do not have someone recite facts \
about planets.
- Give characters plain, sayable names a young child can pronounce.

Plan the story. Do not write it."""


# Sent only on a revision pass, as two extra turns after the first two. Those
# first two stay byte-identical to a first pass, which is what preserves the
# provider-side prompt-cache prefix.
OUTLINE_REVISION_PROMPT_TEMPLATE = """\
An editor read the plan you just wrote and found problems with it. Fix all of \
them and return the whole plan again.

What they checked it against:
- **Protagonist:** the child the book is for must want something, and the story's \
events must follow from what they do about it.
- **Earned resolution:** the ending must come from what the child does, decides \
or realises -- never from a coincidence or something conveniently found.

Here is what they found:

{reviews}

Fix every one. Keep everything that was not criticised: the title, the theme and \
the beats that work should survive unless a fix genuinely requires changing them.

One exception, and it matters: **a protagonist problem cannot be fixed in one \
beat.** Whose story it is runs through the logline, the theme, every character \
description and every beat at once. If that is the finding, change all of them. A \
plan whose logline still says the child is helping someone else reach their dream \
has not been fixed, however the beats now read -- and making the child "decide" to \
help is not the same as giving the child a want.

Return the complete plan, not a description of what you changed."""


def render_outline_reviews(reviews: OutlineReviews) -> str:
    """Render reviews as the human half of the revision prompt.

    Each review is anchored to its beat where it has one. A comment with no
    anchor makes the planner guess which beat to look at, and guessing wrong
    means the fix lands on something the editor never complained about.
    """
    lines = []
    for review in reviews.reviews:
        where = (
            f"beat {review.beat_position}"
            if review.beat_position is not None
            else "the story as a whole"
        )
        lines.append(f"- [{review.rubric.value}] {where}: {review.comment}")
    return "\n".join(lines)


def render_story_brief(brief: StoryBrief) -> str:
    """Render a brief as the human-message half of the planner prompt."""
    child = brief.child
    lines = [
        "Plan a story from this brief.",
        "",
        f"Child's name: {child.name}",
        f"Age: {child.age}",
        f"Pronouns: {child.pronouns.value}",
        f"Reading level: {child.reading_level.value}",
        f"  Guidance: {READING_LEVEL_GUIDANCE[child.reading_level]}",
    ]

    if child.interests:
        lines.append(f"Interests: {', '.join(child.interests)}")

    lines += [
        "",
        f"Premise: {brief.premise}",
        f"Tone: {brief.tone.value}",
        # The planner does not produce pages, but the page count is a hard bound
        # on the beat count, not merely a hint about how much story to carry: a
        # beat needs a page of its own. Stating the limit as a number rather than
        # leaving it to be inferred from "target length".
        f"Target length: {brief.page_count} pages",
        f"Use at most {brief.page_count} beats, and fewer if the story is short.",
    ]

    if brief.must_include:
        lines.append(f"Must include: {', '.join(brief.must_include)}")

    if brief.avoid:
        lines.append(f"Must avoid entirely: {', '.join(brief.avoid)}")

    return "\n".join(lines)


class StoryPlannerNode(Node):
    """Plans a story: what happens and why it matters, before any prose."""

    output_schema = StoryOutline

    def __init__(
        self,
        model: Runnable[Any, Any],
        brief: StoryBrief,
        reviews: OutlineReviews | None = None,
    ) -> None:
        super().__init__(model)
        self.brief = brief
        # The generator is also the editor: its
        # `edit_based_on_reviews` rebuilds `ArticleWriter` with `reviews=` rather
        # than calling a separate node. One prompt, one voice, and no second set
        # of craft rules to keep in sync with this one.
        self.reviews = reviews

    async def ainvoke(self) -> StoryOutline:
        """Produce a story outline from the brief.

        Returns:
            A validated :class:`StoryOutline`.

        Raises:
            Exception: the model returned output that does not satisfy the schema.
                Deliberately allowed to propagate -- a later session turns this
                into a retry carrying the validation error as feedback, and
                swallowing it here would hide the signal that loop depends on.
        """
        brief = self.brief

        # Age and reading level only. A child's name is personal data about a
        # minor, so it is confined to DEBUG and never appears in INFO-level logs
        # that may be shipped off the machine.
        logger.info(
            "Planning story: age=%d level=%s tone=%s pages=%d",
            brief.child.age,
            brief.child.reading_level.value,
            brief.tone.value,
            brief.page_count,
        )
        logger.debug("Brief premise: %r for child %r", brief.premise, brief.child.name)

        messages: list[Any] = [
            SystemMessage(content=STORY_PLANNER_SYSTEM_PROMPT),
            HumanMessage(content=render_story_brief(brief)),
        ]
        if self.reviews is not None:
            # The previous draft is replayed as the model's *own* turn, following
            # brown's article_writer.py: it then edits something it owns rather
            # than critiquing a stranger's work.
            messages += [
                AIMessage(content=self.reviews.outline.model_dump_json(indent=2)),
                HumanMessage(
                    content=OUTLINE_REVISION_PROMPT_TEMPLATE.format(
                        reviews=render_outline_reviews(self.reviews)
                    )
                ),
            ]

        outline: StoryOutline = await self.model.ainvoke(messages)

        logger.info(
            "Planned %r with %d beats and %d characters",
            outline.title,
            len(outline.beats),
            len(outline.characters),
        )
        return outline


async def plan_story(brief: StoryBrief) -> StoryOutline:
    """Build the planner with its configured model and run it.

    The node takes an injected model, so something has to choose which one. That
    choice is orchestration, and in the next commit it moves into the workflow as
    a ``@task``. Until the workflow exists this function is the orchestrator, and
    keeping it means the MCP tool layer does not change twice.

    Raises:
        MissingAPIKeyError: the configured model's API key is not set.
        UnknownModelError: ``PLANNER_MODEL`` is not a known model id.
    """
    model = get_chat_model(settings.planner_model)
    logger.debug("Story planner using model %s", settings.planner_model)
    return await StoryPlannerNode(model=model, brief=brief).ainvoke()
