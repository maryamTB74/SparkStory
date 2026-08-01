"""Plot Planner: maps story beats onto the pages of a book.

**Why this is an agent and not a deterministic transform.** Turning 6 beats into
12 pages *is* the pacing decision. A climax may earn three pages -- turning a page
is itself a dramatic beat, which is physically how suspense works in a picture
book -- while a setup beat gets one. Proportional allocation in code would give
every beat two pages and produce a book with no rhythm. That judgement is the
autonomy that makes this an agent.

Its output is checkable, unlike prose: see ``workflows/validation.py``.
"""

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable

from sparkstory.entities.guidelines import READING_LEVEL_GUIDANCE
from sparkstory.entities.stories import PagePlan, StoryBrief, StoryOutline
from sparkstory.nodes.base import Node
from sparkstory.utils.logging_utils import get_logger

logger = get_logger(__name__)


PLOT_PLANNER_SYSTEM_PROMPT = """\
You are an experienced children's picture-book editor laying out a book. The \
story has already been planned as a sequence of beats. Your job is to decide how \
those beats fall across the pages.

The rule that matters most: **one page is one moment.** A page carries a single \
picture, so everything on it must be drawable as one image. "He packed his bag, \
walked all night, and arrived at dawn" is three pages, not one.

How to pace a book:
- Give a beat more pages when it needs room and fewer when it does not. A climax \
usually wants two or three; a setup often wants one.
- Use the page turn. End a page on the question and answer it overleaf. That \
pause is the only suspense device a picture book has.
- Keep the pages in story order. A picture book does not jump backwards or \
forwards in time.
- Every beat must appear. If a beat has no page, that part of the story simply \
does not happen.
- Vary the setting where the story allows it. Twelve pages in one room is dull \
to look at.

For each page you record three separate notes, and they must stay separate:
- **What the picture shows** -- the one action or image, drawable as one image.
- **What changes inside** the main character: what they feel, notice or decide. \
Every page shifts something, even slightly. A page where nothing changes inside \
is a page the book does not need.
- **The question the page turn leaves open** -- except on the last page, which \
answers rather than asks.

Write all three as **notes, not narration**. "rocket tips over, Pip's ears \
flatten" is a note. "The rocket tipped over and Pip's ears flattened" is a \
sentence from the finished book, and writing one here means the author has \
nothing left to do but copy it. Never write a sentence someone could print.

What not to do:
- Do not write the story. Someone else writes the words.
- Do not invent characters. Use only the characters you were given.
- Do not describe how anyone or anything looks -- no colours, clothing or \
features. The pictures decide that later, and text that contradicts them cannot \
be fixed without a rewrite."""


def render_page_plan_request(brief: StoryBrief, outline: StoryOutline) -> str:
    """Render the outline and target length as the human half of the prompt."""
    lines = [
        f"Lay this story out across exactly {brief.page_count} pages.",
        "",
        f"Title: {outline.title}",
        f"Theme: {outline.theme}",
        f"Tone: {brief.tone.value}",
        f"Reading level: {brief.child.reading_level.value}",
        # The level governs how much text a page can hold, and therefore how much
        # action can fit on one page before it needs two.
        f"  Guidance: {READING_LEVEL_GUIDANCE[brief.child.reading_level]}",
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

    if brief.avoid:
        lines += ["", f"Must avoid entirely: {', '.join(brief.avoid)}"]

    return "\n".join(lines)


class PlotPlannerNode(Node):
    """Decides how many pages each beat gets, and what happens on each."""

    output_schema = PagePlan

    def __init__(
        self,
        model: Runnable[Any, Any],
        brief: StoryBrief,
        outline: StoryOutline,
    ) -> None:
        super().__init__(model)
        self.brief = brief
        self.outline = outline

    async def ainvoke(self) -> PagePlan:
        """Produce a page-by-page plan.

        Returns:
            A validated :class:`PagePlan`. Structural correctness against the
            brief and outline is *not* checked here -- see
            ``workflows/validation.py`` for why that lives in the workflow.
        """
        logger.info(
            "Planning pages: beats=%d target_pages=%d",
            len(self.outline.beats),
            self.brief.page_count,
        )

        plan: PagePlan = await self.model.ainvoke(
            [
                SystemMessage(content=PLOT_PLANNER_SYSTEM_PROMPT),
                HumanMessage(
                    content=render_page_plan_request(self.brief, self.outline)
                ),
            ]
        )

        logger.info("Planned %d pages", len(plan.pages))
        return plan
