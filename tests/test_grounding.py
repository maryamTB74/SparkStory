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
from sparkstory.entities.stories import StoryOutline


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


class TestOutlineCarriesGrounding:
    """The field that carries research past ``plan_story``.

    Before this, ``run_outline_pipeline`` returned a bare outline, so the grounding
    was computed, planned from, and dropped -- which is why the Writer had never
    seen a fact and why a craft device could only ever be *described* in a beat
    summary (findings J and Q).

    Optional with a ``None`` default deliberately: ``MAX_RESEARCH_STEPS=0`` must
    stay a valid configuration, and an ungrounded run must be representable,
    because it is the control arm of the A/B this feature is judged by.
    """

    def test_an_outline_defaults_to_no_grounding(self, outline: StoryOutline) -> None:
        assert outline.grounding is None

    def test_an_outline_accepts_grounding(self, outline: StoryOutline) -> None:
        """Constructed through ``model_validate``, not ``model_copy``.

        ``model_copy(update=...)`` skips validation and will happily set an
        attribute the model does not declare, so it passes whether or not the
        field exists -- rule 24's trap (a check with no room to fail) inside a
        test. Validation is what actually proves the field is declared.
        """
        payload = outline.model_dump()
        payload["grounding"] = StoryGrounding(facts=[a_fact()]).model_dump()
        grounded = StoryOutline.model_validate(payload)
        assert grounded.grounding is not None
        assert grounded.grounding.facts[0].chunk_id == "moon#1"

    def test_grounding_survives_a_json_round_trip(self, outline: StoryOutline) -> None:
        """The field must serialise. It crosses the MCP tool boundary and, once
        runs are resumable, a checkpointer -- so a field that died in JSON would
        fail only when a client threaded it, which is the most expensive place to
        find out.

        Built through ``model_validate`` for the reason given above: a
        ``model_copy`` variant would serialise nothing and still pass, because
        ``model_dump_json`` only emits declared fields.
        """
        payload = outline.model_dump()
        payload["grounding"] = StoryGrounding(facts=[a_fact()]).model_dump()
        grounded = StoryOutline.model_validate(payload)
        restored = StoryOutline.model_validate_json(grounded.model_dump_json())
        assert restored.grounding is not None
        assert restored.grounding.facts[0].story_note == (
            "Nothing outdoors can flutter, drift or make a sound."
        )
