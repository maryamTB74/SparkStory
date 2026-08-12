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
from sparkstory.entities.grounding import StoryGrounding
from sparkstory.entities.guidelines import READING_LEVEL_GUIDANCE
from sparkstory.entities.reviews import OutlineReviews
from sparkstory.entities.stories import StoryBrief, StoryOutline, WorldRules
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
plan where the child only watches has not been fixed however the beats now read, \
and making the child "decide" to help is not the same as giving the child \
something to do.

Fix it by making the child *act*, not by removing the other character. If the \
parent's idea is about an eagle or a fox, that character stays and keeps wanting \
what it wants -- the child needs a want of their own alongside it, and the ending \
has to turn on what the child does.

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


def render_grounding(grounding: StoryGrounding, world_rules: WorldRules) -> str:
    """Render research findings as extra instructions for the planner.

    **Only ``story_note`` is rendered. ``claim`` never is**, and that is the
    whole point of splitting the two fields. This planner's prompt already forbids
    a character reciting facts, and handing it a fact is handing it something to
    recite -- non-obvious rule 13, where the laziest way to satisfy an instruction
    is the one that gets taken. A rule about the world cannot be pasted into a
    story; a sentence about the Moon can.

    Same reasoning applies to ``source``: attribution matters for checking a claim
    afterwards, and means nothing to a story planner, so it stays out of the prompt.

    ``world_rules`` decides how the facts are framed, and it is deliberately a
    *required* argument: a default here could drift from ``StoryBrief``'s default
    and the divergence would be invisible, since both produce a perfectly valid
    Only the facts half exists now: the craft corpus, and with it the device half
    of this renderer, was removed.

    Returns an empty string when there is nothing, so a brief with no grounding
    renders byte-identically to how it rendered before this feature existed --
    which keeps the provider-side prompt-cache prefix intact.
    """
    if not grounding.facts:
        return ""

    lines = [""]
    if grounding.facts:
        if world_rules is WorldRules.REALISTIC:
            lines += [
                "This story is set in the real world, and these things are true of it.",
                "Do not state them, explain them, or have anyone mention them. "
                "Let the story simply obey them:",
            ]
        else:
            # Facts as texture, not law. The one-big-lie principle: accurate
            # furniture is what makes the impossible thing believable, so the
            # retrieved facts earn their place by furnishing the world rather
            # than by policing it. "break as few as you can" is the spec's
            # recommendation B as wording rather than machinery -- asking the
            # planner to nominate its violation would be one more decision it
            # could get wrong, and asking it to be sparing is free.
            lines += [
                "These things are true of the real world. Use them as detail, to "
                "make the impossible parts feel real: a story is more believable, "
                "not less, when everything around the magic is accurate. The "
                "premise may break them -- a fox may fly to the Moon -- but break "
                "as few as you can, and never break one by accident.",
                "Do not state them, explain them, or have anyone mention them:",
            ]
        lines += [f"- {fact.story_note}" for fact in grounding.facts]

    # The craft-device block used to render here, and it carried the fix for
    # finding Q: a repeated line must be built from the story's own words, because
    # handed a "repeat a short line" device beside a story_note the cheapest way to
    # satisfy both is to repeat the note -- which the eagle run did, verbatim, in
    # three beats. That instruction went with the devices it constrained.

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
        grounding: StoryGrounding | None = None,
        memory: str = "",
    ) -> None:
        super().__init__(model)
        self.brief = brief
        # What research found, or None when research was skipped or failed. Travels
        # in the human turn beside the brief rather than in the system prompt,
        # which stays a static constant so the cached prefix survives.
        self.grounding = grounding
        # What earlier books for this child established, already rendered by
        # `memory.render.render_memory`. A string rather than the records
        # themselves, for the same reason `render_grounding` returns text: exactly
        # one place decides how memory is worded to a model, which keeps the rule 1
        # audit a single file to read.
        #
        # Empty string when this child has no memory, or when the brief carries no
        # `child_id` at all -- the overwhelmingly common case, and the one that
        # must behave exactly as it did before this field existed.
        self.memory = memory
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

        request = render_story_brief(brief)
        if self.grounding is not None:
            # No new constructor argument: the brief is already on the node, and
            # world rules belong to the brief rather than to the renderer.
            request += render_grounding(self.grounding, self.brief.world_rules)
        if self.memory:
            # After grounding, so the ordering matches the two constraints'
            # relative authority: what the world is like, then what this child's
            # own earlier books already fixed. A character described in book 1
            # cannot be re-described here, while a retrieved fact is a constraint
            # on a world nobody has committed to yet.
            request += "\n\n" + self.memory

        messages: list[Any] = [
            SystemMessage(content=STORY_PLANNER_SYSTEM_PROMPT),
            HumanMessage(content=request),
        ]
        if self.reviews is not None:
            # The previous draft is replayed as the model's *own* turn, following
            # brown's article_writer.py: it then edits something it owns rather
            # than critiquing a stranger's work.
            #
            # `exclude={"grounding"}` because this is the only place a whole
            # outline is serialised to a model, and grounding carries `chunk_id`
            # and `source` -- a storage key and an attribution, neither of which is
            # story material (rule 1). The specific hazard is worse than noise:
            # the planner is the one node that both sees this schema and could
            # fill the field itself, and an invented `chunk_id` is dropped
            # silently by `drop_unprovenanced`, losing a real fact with no error.
            #
            # Excluded at the *serialisation* point rather than on the field,
            # because the field must survive serialisation everywhere else -- run
            # artifacts, the checkpointer, and the client round trip all need it.
            # `memory_conflicts` is excluded for the same reasons as `grounding`,
            # and it is the more dangerous of the two to replay. It is bookkeeping
            # about *earlier books* rather than story material (rule 1), it is
            # filled by code after planning, and showing the planner a field it is
            # told to leave empty is an invitation to fill it -- with fabricated
            # disagreements a parent would then be asked to resolve.
            messages += [
                AIMessage(
                    content=self.reviews.outline.model_dump_json(
                        indent=2, exclude={"grounding", "memory_conflicts"}
                    )
                ),
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
