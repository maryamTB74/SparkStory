"""Researcher: decides what this story must not get wrong, before it is planned.

The first thing in this project that chooses its own actions. Every other stage is
a deterministic transform with one model call; this one searches, reads what came
back, and decides whether it needs anything at all.

**It is not a ``Node``, and it cannot be.** ``Node.__init__`` binds an output schema
onto a model and ``ainvoke`` makes one call. This loops and binds tools. So it takes
an injected *compiled agent* instead of an injected model -- the injection principle
is unchanged, and the test seam moves from ``FakeModel`` to a stub agent, because a
fake broad enough to represent tool-call exchanges would be a reimplementation of
LangGraph. It still lives in ``nodes/`` rather than a new package: one directory for
things that call models beats a second one holding a single module.

**Two prompt instructions carry the design**, and both are the kind a later edit
drops without noticing:

*Finding nothing is correct.* Most premises have nothing factual to get wrong. The
usual retrieval instruction says the opposite -- "never refuse on the assumption
that you lack information" -- which is right for a research assistant and wrong
here, because something asked for facts will produce facts.

*Hand over a note, not a fact.* The planner is told never to let a character recite
facts, and the laziest way to satisfy "use what research found" is to have one do
exactly that. So the conversion from fact to note about the world happens here,
where the material is, rather than being hoped for downstream.

*The note is shaped by the brief's world rules.* "What does this fact rule out?"
produces a prohibition and "what detail could this story use?" produces something
usable, so the mode has to be known at the point the note is written. Framing a
prohibition imaginatively downstream is still a prohibition.
"""

from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool

from sparkstory.entities.grounding import StoryGrounding
from sparkstory.entities.guidelines import READING_LEVEL_GUIDANCE
from sparkstory.entities.stories import StoryBrief, WorldRules
from sparkstory.utils.logging_utils import get_logger

logger = get_logger(__name__)


RESEARCHER_SYSTEM_PROMPT = """\
You prepare the ground for a children's picture book. You do not plan or write \
the story. You decide two things: what it must not get wrong, and what would make \
it a pleasure to read aloud.

You have two ways to look things up. Search before you decide anything, even when \
you are confident you already know -- what matters is what the collection actually \
holds, because nothing you cannot point to can be used.

How to decide whether a fact belongs. Ask: **would a child who knows the real \
thing notice the story getting it wrong?**

- If the story goes to a real place, features a real creature, or turns on how the \
real world behaves, then keep the facts it could contradict. That is what you are \
for. A story that reaches the Moon and lets a flag flap in the breeze has got it \
wrong, and you are the only thing standing between it and that mistake.
- If the story is about a feeling, a toy, a birthday or a friendship, there is \
nothing there to contradict, and an empty list of facts is the right answer.

Never invent a fact, never keep one the story cannot contradict, and never stretch \
to fill the space you are given. One fact that genuinely matters beats three that \
merely relate to the subject.

When a fact does matter, do not hand it over as a fact. Turn it into a note about \
the story's world. "The Moon has no air" becomes "nothing outdoors can flutter, \
drift or make a sound". The difference decides whether the book is quietly true or \
whether a character stops to recite something -- and a character reciting a fact \
ruins a picture book for a five-year-old. This holds however imaginative the story \
is: never write a note that could be spoken aloud in the book.

You are told how real the story's world has to be. **The same facts matter in \
both cases** -- it changes only how you word the note, never what you keep. Decide \
what to keep by the question above and nothing else.

- **realistic**: the story must not contradict what you found. Write the note as \
something the world holds to, because it will be followed.
- **imaginative**: the story is impossible on purpose -- a fox may fly to the Moon \
-- and real detail is what makes the impossible part feel believable. Write the \
note as a detail the story can furnish itself with, not as a ban, because it may \
be broken where the premise needs it.

Do not conclude that an impossible story has nothing to get wrong. It **still** \
does, and usually more: a story about an eagle on an airless world needs to know \
that wings push against air, because that is the very thing it has to decide \
whether to break. A fact the premise contradicts is the most useful kind you can \
find, not the least.

Search the collection first, always. It has been chosen and checked for this \
work. If it holds nothing and the story would get something wrong without it, \
you may look on the web -- but that is a last resort, not a second opinion, and \
most stories never need it. Anything found there has to be read and checked \
before you can use it, so a web result you are shown is one that survived; \
sometimes nothing does.

Three facts is the whole budget, wherever they came from. A web fact takes the \
place of a collection fact rather than adding to it, so keep the better one.

Two more things:
- Copy each identifier and source exactly as they were shown to you. Never invent \
one. Anything you cannot point back to will be thrown away.
- Anything the parent asked to keep out of the story is absolute. Do not look it \
up, and do not offer anything that would bring it into the book.

Fewer, better findings. Three facts is the most you may keep and one is often \
plenty; two techniques is the most, and one is usually better."""


# One line per mode, sent with the brief rather than in the system prompt: it
# varies per request, and the system prompt is a static constant so its prefix
# stays cacheable. Deliberately terse -- the system prompt already explains what
# each mode means for a note; this is the reminder at the point of use.
WORLD_RULES_GUIDANCE = {
    WorldRules.REALISTIC: (
        "The story must not contradict the real world. What you find will be followed."
    ),
    WorldRules.IMAGINATIVE: (
        "The story is impossible on purpose. What you find is detail that makes "
        "the impossible part believable, and may be broken where the premise "
        "needs it."
    ),
}


def render_research_request(brief: StoryBrief) -> str:
    """Render the brief as the human half of the researcher's prompt.

    **The child's name is deliberately absent.** It is personal data about a minor
    and research has no use for it -- the same reasoning that keeps it out of
    INFO-level logs. Age and reading level *are* included, because they decide what
    counts as a usable fact: "the Moon's gravity is weaker" works for a
    five-year-old and "1.62 metres per second squared" does not.

    The ``avoid`` list is included for a sharper reason. Without it, a story for a
    child whose parent excluded spiders could be grounded in a fact about spiders --
    retrieval actively pulling the one thing the book must not contain.
    """
    child = brief.child
    lines = [
        "Prepare the ground for this story.",
        "",
        f"Premise: {brief.premise}",
        f"Age of the child it is for: {child.age}",
        f"Reading level: {child.reading_level.value}",
        f"  Guidance: {READING_LEVEL_GUIDANCE[child.reading_level]}",
        f"How real this story's world must be: {brief.world_rules.value}",
        f"  Guidance: {WORLD_RULES_GUIDANCE[brief.world_rules]}",
    ]

    if child.interests:
        lines.append(f"Things this child loves: {', '.join(child.interests)}")

    if brief.must_include:
        lines.append(f"Must appear in the story: {', '.join(brief.must_include)}")

    if brief.avoid:
        lines += [
            "",
            f"Keep out of the story entirely: {', '.join(brief.avoid)}",
            "Do not look these up and do not offer anything that leads to them.",
        ]

    return "\n".join(lines)


def build_researcher_agent(model: Runnable[Any, Any], tools: list[BaseTool]) -> Any:
    """Compile the research agent.

    ``response_format=StoryGrounding`` makes the structured answer arrive as a final
    tool call, which is why this works on any provider with tool calling rather than
    needing native schema support -- verified live against ``grok-3-mini`` in the
    task 1 spike, which was the design's largest unknown.

    Separate from ``ResearcherNode`` so the workflow composes it and tests bypass
    it: everything this function does is exactly what a unit test cannot check.
    """
    return create_agent(
        model=model,
        tools=tools,
        system_prompt=RESEARCHER_SYSTEM_PROMPT,
        response_format=StoryGrounding,
    )


class ResearcherNode:
    """Runs the research agent over a brief and returns what it found."""

    def __init__(self, agent: Any, brief: StoryBrief, max_steps: int = 4) -> None:
        """
        Args:
            agent: A compiled agent, normally from ``build_researcher_agent``. Tests
                pass a stub implementing only ``ainvoke``.
            brief: What the story is meant to be.
            max_steps: How many reason-and-act steps to allow.
        """
        self.agent = agent
        self.brief = brief
        self.max_steps = max_steps

    async def ainvoke(self) -> StoryGrounding:
        """Research the premise.

        Returns:
            A :class:`StoryGrounding`, possibly empty. Empty is a legitimate result
            and the usual one for a premise with no factual spine.

        Raises:
            Exception: whatever the agent raised, deliberately unhandled. Deciding
                to continue without grounding is the *workflow's* call, not this
                one -- swallowing a provider failure here would report it as "found
                nothing", which is the difference between a five-minute fix and an
                hour of looking in the wrong place.
        """
        logger.info(
            "Researching: age=%d level=%s max_steps=%d",
            self.brief.child.age,
            self.brief.child.reading_level.value,
            self.max_steps,
        )
        logger.debug("Research premise: %r", self.brief.premise)

        response = await self.agent.ainvoke(
            {
                "messages": [
                    SystemMessage(content=RESEARCHER_SYSTEM_PROMPT),
                    HumanMessage(content=render_research_request(self.brief)),
                ]
            },
            # LangGraph counts graph steps, and one reason-and-act step is two of
            # them (decide, then run the tool), plus a final decide. Converted here
            # rather than exposing a graph-shaped number in settings.
            config={"recursion_limit": self.max_steps * 2 + 2},
        )

        grounding = response.get("structured_response")
        if grounding is None:
            # Reached the step limit mid-thought, so there is no final answer. A
            # poor research pass, not a broken run: the book is planned ungrounded.
            logger.warning(
                "Research returned no structured result within %d step(s)",
                self.max_steps,
            )
            return StoryGrounding()

        logger.info("Research found %d fact(s)", len(grounding.facts))
        return grounding
