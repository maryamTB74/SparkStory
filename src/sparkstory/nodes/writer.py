"""Writer: turns a page plan into the words of the book.

**One call for the whole book, not one per page.** The text is small -- roughly
500 words at ``early_reader``, 1,500-2,000 at ``confident`` -- so the entire book
fits comfortably in one response. Voice consistency is the dominant quality risk
in children's prose, because a register that shifts between pages is immediately
audible when a book is read aloud, and a single call makes consistency structural
rather than hoped for.

Per-page generation was rejected: pages blind to their neighbours produce three
openings of "Suddenly", the same favourite adjective throughout, and no built
rhythm.

**Revision keeps that property rather than trading it away.** When a critic
returns findings this node runs again with ``reviews`` set, and it regenerates
the *whole* book, not the flagged pages -- following ``brown``, whose
``edit_based_on_reviews`` rebuilds ``ArticleWriter`` rather than calling an editor
node. Patching individual pages would reintroduce exactly the seam that one-call
generation exists to avoid. The cost is that untouched pages could drift, so the
revision prompt insists they come back word for word.
"""

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable

from sparkstory.entities.guidelines import READING_LEVEL_GUIDANCE
from sparkstory.entities.reviews import ProseReviews
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


# Sent only on a revision pass, as two extra turns after the first two. Those
# first two stay byte-identical to a first pass, preserving the provider-side
# prompt-cache prefix.
PROSE_REVISION_PROMPT_TEMPLATE = """\
An editor read your manuscript aloud and found problems with it. Fix all of them \
and return the whole book again.

Here is what they found:

{reviews}

Satisfying a note never means writing the note down. If an editor says a page's \
feeling never arrives, the fix is to show it happening -- in what the character \
does, notices or says. Copying the note onto the page is the one thing that \
cannot work: it is the failure they were describing, spelled out.

Fix every one. **Leave every page they did not mention unchanged** -- word for \
word. You are returning the whole book because that is how the pages are handed \
over, not because this is a fresh draft, and a voice that shifts between passes \
is audible when a book is read aloud twice in a week.

Keep every rule you were given the first time: the reading level, the pronouns, \
the things to avoid, one entry per page in order, and no stated lesson at the \
end."""


def render_prose_reviews(reviews: ProseReviews) -> str:
    """Render reviews as the human half of the revision prompt.

    Anchored to a page where the review has one, and explicitly *not* anchored
    where it does not. A comment with no anchor makes the Writer guess which page
    to change, and a guess lands the fix on a page the editor never mentioned; a
    book-wide finding given an invented page number gets fixed on that one page
    instead of all of them.
    """
    lines = []
    for review in reviews.reviews:
        where = (
            f"page {review.page_number}"
            if review.page_number is not None
            else "the book as a whole"
        )
        lines.append(f"- [{review.rubric.value}] {where}: {review.comment}")
    return "\n".join(lines)


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
        reviews: ProseReviews | None = None,
    ) -> None:
        super().__init__(model)
        self.brief = brief
        self.outline = outline
        self.page_plan = page_plan
        # Following brown, the generator is the editor. The whole book is
        # rewritten rather than patched -- which is also what keeps the voice
        # consistent, since it was always one call for all pages anyway.
        self.reviews = reviews

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

        messages: list[Any] = [
            SystemMessage(content=WRITER_SYSTEM_PROMPT),
            HumanMessage(
                content=render_prose_request(self.brief, self.outline, self.page_plan)
            ),
        ]
        if self.reviews is not None:
            # The previous draft replayed as the model's own turn, following
            # brown's article_writer.py: it edits something it owns.
            messages += [
                AIMessage(content=self.reviews.prose.model_dump_json(indent=2)),
                HumanMessage(
                    content=PROSE_REVISION_PROMPT_TEMPLATE.format(
                        reviews=render_prose_reviews(self.reviews)
                    )
                ),
            ]

        prose: StoryProse = await self.model.ainvoke(messages)

        logger.info(
            "Wrote %d pages, %d words total",
            len(prose.pages),
            sum(len(page.text.split()) for page in prose.pages),
        )
        return prose
