"""Writer: turns a page plan into the words of the book.

**One call for the whole book, not one per page.** The text is small -- roughly
500 words at ``early_reader``, 1,500-2,000 at ``confident`` -- so the entire book
fits comfortably in one response. Voice consistency is the dominant quality risk
in children's prose, because a register that shifts between pages is immediately
audible when a book is read aloud, and a single call makes consistency structural
rather than hoped for.

Per-page generation was rejected: pages blind to their neighbours produce three
openings of "Suddenly", the same favourite adjective throughout, and no built
rhythm. Per-page *revision* is a different thing and arrives in a later session,
where a page flagged by a critic is rewritten with the whole text as context.
"""

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable

from sparkstory.entities.guidelines import READING_LEVEL_GUIDANCE
from sparkstory.entities.stories import (
    PagePlan,
    StoryBrief,
    StoryOutline,
    StoryProse,
)
from sparkstory.nodes.base import Node
from sparkstory.utils.logging_utils import get_logger

logger = get_logger(__name__)


WRITER_SYSTEM_PROMPT = """\
You are a much-loved children's picture-book author. You are given a book laid \
out page by page, and you write the words that go on each page.

Write for reading aloud. A picture book is heard before it is read:
- Rhythm matters more than information. Read every line in your head; if it \
stumbles, rewrite it.
- Vary how sentences begin. Repetition is powerful when you choose it and \
careless when you do not.
- Let the pictures do the looking. Do not describe how characters or places \
appear -- no colours, clothing or features. Words that contradict an illustration \
cannot be fixed without a rewrite.
- Trust the child. Say the true thing simply rather than explaining it.

Each page gives you three notes: what the picture shows, what changes inside the \
main character, and the question the page leaves open. They are notes, not a \
draft. **Never copy their wording** -- if a line of yours shares a phrase with a \
note, rewrite it. Write the page so all three are true of it: the action \
happens, the feeling is in the words rather than named, and the reader wants to \
turn over.

Do not write the same shape of page eight times. Some pages are one line. Some \
are dialogue. Some are a sound. A page that is always "action, then feeling, \
then question" is a form, not a story.

Hard rules:
- Write exactly one entry for every page you are given, in order, and write that \
page's moment and no other. Do not merge pages, skip pages, or add pages.
- The child named in the brief is the main character, referred to by their name \
and their stated pronouns. Never guess at pronouns.
- Anything in the "avoid" list must not appear: not directly, not alluded to, not \
by a near-synonym.
- Never state the theme and never end with a lesson. If the story works, the \
child feels it without being told.
- Keep any peril gentle and resolved. A character may be worried, lost or \
disappointed; they must never be endangered or humiliated.
- Weave the child's interests in as texture, not as a list of facts.
- No page numbers, titles, headings or stage directions in the text. Only the \
words that are printed on the page."""


def render_prose_request(
    brief: StoryBrief, outline: StoryOutline, page_plan: PagePlan
) -> str:
    """Render the plan and the child's constraints as the human half of the prompt."""
    child = brief.child
    lines = [
        f"Write the words for all {len(page_plan.pages)} pages of this book.",
        "",
        f"Title: {outline.title}",
        f"Theme (never state it aloud): {outline.theme}",
        f"Tone: {brief.tone.value}",
        "",
        f"Child's name: {child.name}",
        f"Age: {child.age}",
        f"Pronouns: {child.pronouns.value}",
        f"Reading level: {child.reading_level.value}",
        f"  Guidance: {READING_LEVEL_GUIDANCE[child.reading_level]}",
    ]

    if child.interests:
        lines.append(f"Interests: {', '.join(child.interests)}")
    if brief.must_include:
        lines.append(f"Must include: {', '.join(brief.must_include)}")
    if brief.avoid:
        lines.append(f"Must avoid entirely: {', '.join(brief.avoid)}")

    lines += ["", "Characters:"]
    lines += [f"- {c.name} ({c.role}): {c.description}" for c in outline.characters]

    lines += ["", "Pages:"]
    for page in page_plan.pages:
        who = ", ".join(page.characters_present) or "no one named"
        lines.append(
            f"- Page {page.page_number} | setting: {page.setting} | present: {who}"
        )
        # Separately labelled and indented so the three notes read as three
        # things. Run together on one line they become a sentence to lift.
        lines.append(f"    shows: {page.visual_action}")
        lines.append(f"    inside: {page.emotional_shift}")
        # Omitted rather than rendered empty on the final page: a bare
        # "leaves open:" invites an invented cliffhanger where the book ends.
        if page.page_turn_hook:
            lines.append(f"    leaves open: {page.page_turn_hook}")

    return "\n".join(lines)


class WriterNode(Node):
    """Writes the whole book's prose in one pass."""

    output_schema = StoryProse

    def __init__(
        self,
        model: Runnable[Any, Any],
        brief: StoryBrief,
        outline: StoryOutline,
        page_plan: PagePlan,
    ) -> None:
        super().__init__(model)
        self.brief = brief
        self.outline = outline
        self.page_plan = page_plan

    async def ainvoke(self) -> StoryProse:
        """Produce the words for every page.

        Returns:
            A validated :class:`StoryProse`. That the pages line up with the plan
            is checked in ``workflows/validation.py``, not here.
        """
        logger.info(
            "Writing prose: pages=%d level=%s",
            len(self.page_plan.pages),
            self.brief.child.reading_level.value,
        )

        prose: StoryProse = await self.model.ainvoke(
            [
                SystemMessage(content=WRITER_SYSTEM_PROMPT),
                HumanMessage(
                    content=render_prose_request(
                        self.brief, self.outline, self.page_plan
                    )
                ),
            ]
        )

        logger.info(
            "Wrote %d pages, %d words total",
            len(prose.pages),
            sum(len(page.text.split()) for page in prose.pages),
        )
        return prose
