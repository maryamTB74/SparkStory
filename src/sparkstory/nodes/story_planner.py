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

from langchain_core.messages import HumanMessage, SystemMessage

from sparkstory.config import settings
from sparkstory.entities.guidelines import READING_LEVEL_GUIDANCE
from sparkstory.entities.stories import StoryBrief, StoryOutline
from sparkstory.models.get_model import get_chat_model
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
        # The planner does not produce pages, but knowing the target length tells
        # it how much story the beats have to carry.
        f"Target length: {brief.page_count} pages",
    ]

    if brief.must_include:
        lines.append(f"Must include: {', '.join(brief.must_include)}")

    if brief.avoid:
        lines.append(f"Must avoid entirely: {', '.join(brief.avoid)}")

    return "\n".join(lines)


async def plan_story(brief: StoryBrief) -> StoryOutline:
    """Produce a story outline from a brief.

    Args:
        brief: What the user asked for, including the child's profile.

    Returns:
        A validated :class:`StoryOutline`.

    Raises:
        MissingAPIKeyError: the configured model's API key is not set.
        UnknownModelError: ``PLANNER_MODEL`` is not a known model id.
        Exception: the model returned output that does not satisfy the schema.
            Deliberately allowed to propagate -- a later session turns this into
            a retry carrying the validation error as feedback, and swallowing it
            here would hide the signal that loop depends on.
    """
    # Age and reading level only. A child's name is personal data about a minor,
    # so it is confined to DEBUG and never appears in INFO-level logs that may be
    # shipped off the machine.
    logger.info(
        "Planning story: age=%d level=%s tone=%s pages=%d model=%s",
        brief.child.age,
        brief.child.reading_level.value,
        brief.tone.value,
        brief.page_count,
        settings.planner_model,
    )
    logger.debug("Brief premise: %r for child %r", brief.premise, brief.child.name)

    model = get_chat_model(settings.planner_model, schema=StoryOutline)

    outline = await model.ainvoke(
        [
            SystemMessage(content=STORY_PLANNER_SYSTEM_PROMPT),
            HumanMessage(content=render_story_brief(brief)),
        ]
    )

    logger.info(
        "Planned %r with %d beats and %d characters",
        outline.title,
        len(outline.beats),
        len(outline.characters),
    )
    return outline
