"""The Researcher.

Tested with a **stub agent**, not ``FakeModel``. That is not a preference: a
``Node`` fake only needs ``with_structured_output`` and ``ainvoke``, while an agent
binds tools and exchanges tool-call messages, and a fake broad enough to represent
that would be a reimplementation of LangGraph. So the seam moves one level up --
``ResearcherNode`` takes an injected agent, and ``build_researcher_agent`` is the
part only a live run can verify (which the task 1 spike did).

The prompt assertions are the valuable half of this file. The instruction that
returning no facts is correct is the one the whole design rests on, and it is the
kind of line a later prompt edit quietly drops.
"""

from typing import Any

import pytest

from sparkstory.entities.grounding import GroundedFact, StoryGrounding
from sparkstory.entities.stories import (
    ChildProfile,
    Pronouns,
    ReadingLevel,
    StoryBrief,
    WorldRules,
)
from sparkstory.nodes.researcher import (
    RESEARCHER_SYSTEM_PROMPT,
    ResearcherNode,
    render_research_request,
)


class StubAgent:
    """Stands in for a compiled ReAct agent: records input, returns a canned result."""

    def __init__(self, result: Any) -> None:
        self._result = result
        self.calls: list[dict] = []

    async def ainvoke(self, payload: dict, **_: Any) -> dict:
        self.calls.append(payload)
        if isinstance(self._result, Exception):
            raise self._result
        return {"structured_response": self._result, "messages": []}


def a_brief(**overrides: Any) -> StoryBrief:
    payload: dict = {
        "child": ChildProfile(
            name="Maryam",
            age=5,
            pronouns=Pronouns.SHE_HER,
            reading_level=ReadingLevel.EARLY_READER,
            interests=["foxes", "astronomy"],
        ),
        "premise": "a fox who wants to visit the moon",
    }
    payload.update(overrides)
    return StoryBrief(**payload)


def some_grounding() -> StoryGrounding:
    return StoryGrounding(
        facts=[
            GroundedFact(
                claim="The Moon has no air.",
                story_note="Nothing outdoors can flutter.",
                source="NASA",
                chunk_id="moon#1",
            )
        ],
    )


class TestSystemPrompt:
    def test_authorises_finding_nothing(self) -> None:
        """The load-bearing instruction. Without it an agent asked for facts will
        produce facts, and the task 1 spike's empty-list result was the single most
        important thing it demonstrated."""
        lowered = RESEARCHER_SYSTEM_PROMPT.lower()
        assert "empty" in lowered
        assert "invent" in lowered

    def test_demands_a_note_rather_than_a_fact(self) -> None:
        """The recital trap: the planner prompt already forbids a character
        reciting facts, and the laziest way to satisfy "use what was found" is to
        have one recite it.

        This asserted the word "rule" until the field became `story_note`. The
        intent is unchanged -- the prompt must demand a *converted* form rather
        than a raw fact -- but "rule" is the realistic-only vocabulary the rename
        moved away from, and a note is not a rule in an imaginative story.
        """
        lowered = RESEARCHER_SYSTEM_PROMPT.lower()
        assert "note" in lowered
        assert "recite" in lowered or "recited" in lowered

    def test_tells_it_to_search_before_deciding(self) -> None:
        """Cheap insurance, following lesson 6's forced-tool-call and lesson 11's
        "always search first": an agent that already knows about the Moon will
        skip retrieval and cite nothing."""
        assert "search" in RESEARCHER_SYSTEM_PROMPT.lower()

    def test_forbids_inventing_an_identifier(self) -> None:
        """A fabricated id is dropped by provenance filtering, so an agent that
        invents them silently produces no grounding at all."""
        assert "id" in RESEARCHER_SYSTEM_PROMPT.lower()

    def test_does_not_leak_our_machinery(self) -> None:
        """Same audit the other node prompts get."""
        for term in ("langchain", "pydantic", "json", "schema", "workflow", "rubric"):
            assert term not in RESEARCHER_SYSTEM_PROMPT.lower(), term

    def test_describes_craft_not_output_format(self) -> None:
        """Output shape is enforced mechanically; restating it burns tokens and
        risks contradicting the real constraint."""
        assert "max_length" not in RESEARCHER_SYSTEM_PROMPT

    def test_explains_what_each_world_rule_means_for_a_note(self) -> None:
        """The note is written here, so the mode has to be understood here.

        "What does this fact rule out?" produces a prohibition; "what detail
        could this story use?" produces something usable. Task 3 changed how the
        note is framed to the planner, but framing a prohibition imaginatively
        is still a prohibition -- the wording has to be right at the point it is
        written.
        """
        lowered = RESEARCHER_SYSTEM_PROMPT.lower()
        assert "realistic" in lowered
        assert "imaginative" in lowered

    def test_still_forbids_a_note_that_could_be_spoken(self) -> None:
        """The guard that must survive the imaginative wording.

        The laziest way to write a "usable detail" is to write a story-shaped
        sentence -- which is exactly the recital trap, arriving one stage
        earlier. Loosening "not a line anyone says" to accommodate texture would
        undo what splitting `claim` from `story_note` bought.
        """
        lowered = RESEARCHER_SYSTEM_PROMPT.lower()
        assert "recite" in lowered or "recited" in lowered


class TestRenderResearchRequest:
    def test_includes_the_premise(self) -> None:
        assert "a fox who wants to visit the moon" in render_research_request(a_brief())

    def test_includes_age_and_reading_level(self) -> None:
        """Both change what counts as a usable fact: "gravity is weaker" is fine
        for a five-year-old, "1.62 metres per second squared" is not."""
        rendered = render_research_request(a_brief())
        assert "5" in rendered
        assert "early_reader" in rendered

    def test_omits_the_child_s_name(self) -> None:
        """Personal data about a minor, and research has no use for it. The same
        reasoning keeps the name out of INFO-level logs."""
        assert "Maryam" not in render_research_request(a_brief())

    def test_includes_the_avoid_list(self) -> None:
        """A fact about spiders retrieved for a child whose parent excluded spiders
        would be grounding the story in the one thing it must not contain."""
        rendered = render_research_request(a_brief(avoid=["spiders", "the dark"]))
        assert "spiders" in rendered

    def test_omits_the_avoid_section_when_empty(self) -> None:
        assert "avoid" not in render_research_request(a_brief()).lower()

    def test_includes_interests_because_they_shape_what_is_worth_looking_up(
        self,
    ) -> None:
        assert "astronomy" in render_research_request(a_brief())

    @pytest.mark.parametrize("world_rules", list(WorldRules))
    def test_includes_the_world_rules(self, world_rules: WorldRules) -> None:
        """Per-brief, so it belongs in the human half rather than the static
        system prompt -- which is what keeps the cached prefix byte-identical."""
        rendered = render_research_request(a_brief(world_rules=world_rules))
        assert world_rules.value in rendered


class TestResearcherNode:
    async def test_returns_the_agent_s_grounding(self) -> None:
        agent = StubAgent(some_grounding())
        result = await ResearcherNode(agent=agent, brief=a_brief()).ainvoke()
        assert result.facts[0].chunk_id == "moon#1"

    async def test_sends_the_system_prompt_and_the_request(self) -> None:
        agent = StubAgent(some_grounding())
        await ResearcherNode(agent=agent, brief=a_brief()).ainvoke()
        messages = agent.calls[0]["messages"]
        assert any("fox" in str(getattr(m, "content", m)) for m in messages)

    async def test_an_empty_result_is_returned_as_is(self) -> None:
        """Not converted to None, not treated as a failure. Empty is an answer."""
        result = await ResearcherNode(
            agent=StubAgent(StoryGrounding()), brief=a_brief()
        ).ainvoke()
        assert result.facts == []

    async def test_a_missing_structured_response_becomes_empty_grounding(self) -> None:
        """An agent that hit its step limit mid-thought returns no structured
        result. That is a bad research pass, not a broken book -- the node reports
        empty and the run continues ungrounded."""

        class NoResult:
            async def ainvoke(self, payload: dict, **_: Any) -> dict:
                return {"messages": []}

        result = await ResearcherNode(agent=NoResult(), brief=a_brief()).ainvoke()
        assert result.facts == []

    async def test_an_agent_error_propagates(self) -> None:
        """The *workflow* decides to fail open, not the node. Swallowing it here
        would hide a broken provider behind "no facts found", which is exactly the
        failure that took 17 seconds to produce and an hour to understand in
        Session 8."""
        agent = StubAgent(RuntimeError("provider exploded"))
        with pytest.raises(RuntimeError, match="provider exploded"):
            await ResearcherNode(agent=agent, brief=a_brief()).ainvoke()


class TestWebSearchInstructions:
    """Added with the web tool. The corpus is curated and free; the web is
    neither, so preference is not a style choice."""

    def test_the_collection_comes_first(self) -> None:
        lowered = RESEARCHER_SYSTEM_PROMPT.lower()
        assert "collection first" in lowered
        assert "last resort" in lowered

    def test_the_fact_budget_is_shared_across_sources(self) -> None:
        """Three facts total, not three plus three.

        `StoryGrounding.facts` caps at 3 and the cap is enforced by the schema,
        so an agent that treats it as per-source has its extra facts rejected
        with no explanation. Saying it plainly is cheaper than a validation error
        the model cannot see.
        """
        assert "takes the place of" in RESEARCHER_SYSTEM_PROMPT.lower()

    def test_the_finding_i_guards_all_survive(self) -> None:
        """The prompt where under-grounding lived, checked rather than assumed.

        Session 5's finding I was a prompt edit that stopped grounding entirely,
        and Session 9's task 4 re-checked these for the same reason: this file is
        one paragraph away from the wording that caused it.
        """
        lowered = RESEARCHER_SYSTEM_PROMPT.lower()
        assert "empty" in lowered
        assert "invent" in lowered
        assert "search" in lowered
        assert "note" in lowered


class TestModeDoesNotChangeWhatIsWorthKeeping:
    """Finding S. The mode decides how a note is *written*, never whether a fact
    is worth keeping.

    Task 4's wording said an imaginative story "is impossible on purpose" and
    that a prohibition "is of no use here", which reads as *facts matter less*.
    Live result: the same eagle brief retrieved one fact under `realistic` and
    **zero** under `imaginative`, twice. The imaginative rendering then became
    unreachable on that premise -- the branch cannot treat a fact as texture if
    no fact arrives.

    So the keep-or-drop criterion has to be stated once, mode-independently, and
    the mode paragraph must be visibly about phrasing.
    """

    def test_the_keep_criterion_is_stated_once_and_is_mode_free(self) -> None:
        """ "Would a child who knows the real thing notice?" is the whole test,
        and it must not be qualified by world rules."""
        prompt = RESEARCHER_SYSTEM_PROMPT
        assert "would a child who knows the real" in prompt.lower()
        # The criterion sentence must not mention either mode.
        criterion = prompt.lower().split("would a child who knows the real")[1][:400]
        assert "imaginative" not in criterion
        assert "realistic" not in criterion

    def test_the_mode_paragraph_says_it_changes_wording_not_selection(self) -> None:
        lowered = RESEARCHER_SYSTEM_PROMPT.lower()
        assert "the same facts matter in both" in lowered

    def test_an_impossible_premise_does_not_excuse_finding_nothing(self) -> None:
        """The specific failure: 'the story is impossible anyway, so nothing can
        be got wrong'. An eagle on an airless world is exactly when the fact
        matters most -- it is what the story has to decide to break."""
        lowered = RESEARCHER_SYSTEM_PROMPT.lower()
        assert "impossible" in lowered
        assert "still" in lowered
