"""What research is allowed to return.

The most important test in this module is
``TestStoryGrounding::test_finding_nothing_is_representable``. Most premises have
nothing to get factually wrong, so an empty ``facts`` list is the correct answer
rather than a degenerate one -- and a reflexive ``min_length=1`` would make it
unrepresentable. The symptom would not be a schema error; it would be an agent
inventing facts about a teddy bear to satisfy its own output contract.
"""

import pytest
from pydantic import ValidationError

from sparkstory.entities.grounding import GroundedFact, StoryGrounding


def a_fact(chunk_id: str = "moon#1") -> GroundedFact:
    return GroundedFact(
        claim="The Moon has no air.",
        story_note="Nothing outdoors can flutter, drift or make a sound.",
        source="NASA -- Moon Facts",
        chunk_id=chunk_id,
    )


class TestGroundedFact:
    def test_carries_its_provenance(self) -> None:
        """``chunk_id`` is what lets a later session prove a claim came from us
        rather than from a client's imagination."""
        fact = a_fact()
        assert fact.chunk_id == "moon#1"
        assert fact.source == "NASA -- Moon Facts"

    @pytest.mark.parametrize("field", ["claim", "story_note", "source", "chunk_id"])
    def test_every_field_is_required(self, field: str) -> None:
        """No field here is optional. A fact with no source is an assertion, and
        a fact with no constraint is material the planner will have recited."""
        payload = a_fact().model_dump()
        del payload[field]
        with pytest.raises(ValidationError):
            GroundedFact(**payload)

    def test_rejects_blank_text(self) -> None:
        with pytest.raises(ValidationError):
            GroundedFact(
                claim="",
                story_note="Nothing can flutter.",
                source="NASA",
                chunk_id="moon#1",
            )


class TestStoryGrounding:
    def test_finding_nothing_is_representable(self) -> None:
        """Non-obvious rule 14, one level up from the review loops: the empty
        answer must validate, or the agent is forced to invent one."""
        assert StoryGrounding(facts=[]).facts == []

    def test_facts_defaults_to_empty(self) -> None:
        """So a model that omits the field entirely is not a hard failure."""
        assert StoryGrounding().facts == []

    def test_caps_facts_at_three(self) -> None:
        """The planner treats a budget as a target -- finding D, unchanged across
        four live runs -- so the budget is small."""
        facts = [a_fact(f"moon#{i}") for i in range(3)]
        assert len(StoryGrounding(facts=facts).facts) == 3
        with pytest.raises(ValidationError):
            StoryGrounding(facts=[a_fact(f"moon#{i}") for i in range(4)])
