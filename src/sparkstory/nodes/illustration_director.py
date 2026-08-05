"""Illustration Director: decides what the book looks like, once, for every page.

**Why this is an agent and not a deterministic transform.** Nothing in a finished
`Story` says what anyone looks like. `CharacterSketch.description` is *narrative* --
"a fox who longs to visit the moon" -- and the Plot Planner is explicitly forbidden
from describing appearance ("the pictures decide that later"). So somebody has to
invent a visual identity for each character and a shared look for the book, and
inventing it coherently across eight pages is a judgement, not a mapping.

**Why it is one call and not one per page.** Consistency is the whole problem. A
per-page call would decide the palette eight times and get eight answers; deciding
once and reusing it is the mechanism. This is also why the node returns a
`style_bible` at all rather than letting each page prompt stand alone.

**Where its output goes.** The character portraits are generated first and become
*reference images* for every page -- so this node's `appearance` text matters twice:
once as the portrait prompt, and then indirectly through the portrait it produced.
A vague appearance yields a generic portrait, and every page inherits it.

The prompt deliberately does **not** name a medium or an art style. Grok's image
model has a strong default look, and prescribing "watercolour" in the system prompt
would fight it invisibly -- the model would produce its default and nothing would
report the mismatch. Choosing the style is the `style_bible`'s job, per book.
"""

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable

from sparkstory.entities.illustration import MAX_REFERENCE_IMAGES, IllustrationPlan
from sparkstory.entities.stories import Story, StoryBrief
from sparkstory.nodes.base import Node
from sparkstory.utils.logging_utils import get_logger

logger = get_logger(__name__)


ILLUSTRATION_DIRECTOR_SYSTEM_PROMPT = """\
You are the art director of a children's picture book. The story is written. Your \
job is to decide what the book looks like, so that a team of illustrators working \
separately would produce pictures that belong in the same book.

You make three kinds of decision.

**The shared look.** Choose it once and it governs every page: a small palette \
named colour by colour, how lines and edges are drawn, where the light comes \
from, and how full or empty the backgrounds are. Be concrete enough that two \
illustrators could not disagree. Naming a medium is not a decision -- thousands \
of books are "watercolour picture books" and no two look alike.

**What each character looks like.** Give each of them two or three specific, \
visible features another artist could match without seeing your picture: build, \
colouring, markings, and one thing worn or carried that never changes from page \
to page. That unchanging item is what a child recognises first, so choose it \
deliberately.

Describe only what is visible. Never write what a character feels, wants or is \
like, and never use a word that judges instead of describing: "majestic", \
"soulful", "adorable" and "beautiful" give an illustrator nothing to draw.

**What happens in each picture.** One picture per page, in order, none skipped \
and none repeated. Say where it happens, who is in it, what they are doing, and \
where the light falls.

Rules that matter more than they look:

- **A picture carries no writing.** Never ask for a title, a caption, a letter, a \
sign or a word of any kind inside a picture. Words drawn into an illustration \
cannot be corrected later and ruin the page.
- **Never copy the story's sentences into a picture description.** You are \
describing an image, not retelling the page. Write what a viewer would see.
- **At most {max_refs} characters in any one picture.** This is a hard limit. \
When a page involves more people than that, choose the {max_refs} the picture is \
really about and leave the others out of it -- a crowded picture-book page is a \
badly composed one anyway.
- **Do not contradict the words on the page.** If the text says night, the \
picture is at night.
- **Draw the child as the story's own age.** These books are made for one \
particular child and they will look for themselves in the pictures."""


def render_illustration_request(brief: StoryBrief, story: Story) -> str:
    """Render the finished story as the human half of the prompt.

    The *prose* is included, not just the page plan. The plan's `visual_action`
    notes are what the picture shows, but only the prose says what actually
    happened -- and a picture contradicting the words on its own page is the one
    error a reader cannot miss.
    """
    outline = story.outline
    lines = [
        f"Design the pictures for this {len(story.pages)}-page book.",
        "",
        f"Title: {outline.title}",
        f"Theme: {outline.theme}",
        f"Tone: {brief.tone.value}",
        "",
        f"It was written for {brief.child.name}, aged {brief.child.age}, "
        f"who uses {brief.child.pronouns.value} pronouns.",
    ]
    if brief.child.interests:
        lines.append(f"Their interests: {', '.join(brief.child.interests)}.")

    # `avoid` reaches this node deliberately. A parent's exclusions are about the
    # book, not only about its words, and an illustration is the one part of a
    # picture book a pre-reader consumes unaided.
    if brief.avoid:
        lines += [
            "",
            "Must not appear, in the pictures either: " + ", ".join(brief.avoid) + ".",
        ]

    lines += ["", "The cast:"]
    for character in outline.characters:
        lines.append(f"- {character.name} ({character.role}): {character.description}")

    lines += ["", "The pages, each with its words and the picture it needs:"]
    plans = {plan.page_number: plan for plan in story.page_plan.pages}
    for page in story.pages:
        plan = plans.get(page.page_number)
        lines.append("")
        lines.append(f"Page {page.page_number}")
        if plan is not None:
            lines.append(f"  Setting: {plan.setting}")
            lines.append(f"  The picture shows: {plan.visual_action}")
            if plan.characters_present:
                lines.append(f"  Present: {', '.join(plan.characters_present)}")
        # Last, and labelled as the words, so the model reads it as a constraint on
        # the picture rather than as text to reproduce inside it.
        lines.append(f"  The words on the page: {page.text}")

    return "\n".join(lines)


class IllustrationDirectorNode(Node):
    """Decides the style bible, each character's appearance, and each picture."""

    output_schema = IllustrationPlan

    def __init__(
        self, model: Runnable[Any, Any], brief: StoryBrief, story: Story
    ) -> None:
        super().__init__(model)
        self.brief = brief
        self.story = story

    async def ainvoke(self) -> IllustrationPlan:
        """Design the book's pictures."""
        logger.debug(
            "directing illustrations for %r (%d pages)",
            self.story.outline.title,
            len(self.story.pages),
        )
        plan: IllustrationPlan = await self.model.ainvoke(
            [
                SystemMessage(
                    ILLUSTRATION_DIRECTOR_SYSTEM_PROMPT.format(
                        max_refs=MAX_REFERENCE_IMAGES
                    )
                ),
                HumanMessage(render_illustration_request(self.brief, self.story)),
            ]
        )
        return plan
