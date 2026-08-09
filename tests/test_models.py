"""Domain schema constraints.

These assert the *boundaries*, not the happy path, because the boundaries are
what protect later stages: a 200-page book or a two-beat story would otherwise
propagate silently into illustration and narration, where it costs real money.
"""

import pytest
from pydantic import ValidationError

from sparkstory.entities.grounding import GroundedFact, StoryGrounding
from sparkstory.entities.stories import (
    CharacterSketch,
    ChildProfile,
    NarrativeFunction,
    Pronouns,
    ReadingLevel,
    StoryBeat,
    StoryBrief,
    StoryOutline,
    WorldRules,
)


class TestChildProfile:
    @pytest.mark.parametrize("age", [2, 7, 12])
    def test_accepts_ages_in_range(self, age: int) -> None:
        assert ChildProfile(name="Sam", age=age).age == age

    @pytest.mark.parametrize("age", [1, 0, -3, 13, 25])
    def test_rejects_ages_out_of_range(self, age: int) -> None:
        with pytest.raises(ValidationError):
            ChildProfile(name="Sam", age=age)

    def test_rejects_empty_name(self) -> None:
        with pytest.raises(ValidationError):
            ChildProfile(name="", age=5)

    def test_pronouns_default_to_they_them(self) -> None:
        """A name does not indicate pronouns, so the neutral default must hold."""
        assert ChildProfile(name="Sam", age=5).pronouns is Pronouns.THEY_THEM

    def test_rejects_invented_pronouns(self) -> None:
        with pytest.raises(ValidationError):
            ChildProfile(name="Sam", age=5, pronouns="whatever")

    def test_reading_level_defaults_to_early_reader(self) -> None:
        assert (
            ChildProfile(name="Sam", age=5).reading_level is ReadingLevel.EARLY_READER
        )


class TestStoryBrief:
    @pytest.mark.parametrize("pages", [4, 12, 24])
    def test_accepts_page_counts_in_range(
        self, child: ChildProfile, pages: int
    ) -> None:
        assert (
            StoryBrief(child=child, premise="a lost hat", page_count=pages).page_count
            == pages
        )

    @pytest.mark.parametrize("pages", [0, 3, 25, 500])
    def test_rejects_page_counts_out_of_range(
        self, child: ChildProfile, pages: int
    ) -> None:
        with pytest.raises(ValidationError):
            StoryBrief(child=child, premise="a lost hat", page_count=pages)

    def test_rejects_too_short_premise(self, child: ChildProfile) -> None:
        with pytest.raises(ValidationError):
            StoryBrief(child=child, premise="x")

    def test_optional_lists_default_empty(self, child: ChildProfile) -> None:
        b = StoryBrief(child=child, premise="a lost hat")
        assert b.must_include == []
        assert b.avoid == []

    def test_world_rules_has_exactly_two_values(self) -> None:
        """Two modes, not a genre taxonomy.

        A third value would be config for a distinction nothing branches on
        (Rule 3), and "mostly real with one licensed miracle" is wording inside
        `imaginative` rather than a mode of its own. Asserting the exact set
        makes adding one a deliberate conversation.
        """
        assert {rule.value for rule in WorldRules} == {"realistic", "imaginative"}

    def test_world_rules_defaults_to_imaginative(self, child: ChildProfile) -> None:
        """The default follows the product, and it is a behaviour change.

        A caller who supplies no `world_rules` now gets different planning
        behaviour than before this field existed. Most personalised picture
        books are imaginative -- a talking fox is not realistic -- and the
        grounded/ungrounded A/B run produced the weaker book under the
        realistic rendering on this project's own standing premise.
        """
        assert (
            StoryBrief(child=child, premise="a lost hat").world_rules
            is WorldRules.IMAGINATIVE
        )

    def test_explicit_realistic_round_trips(self, child: ChildProfile) -> None:
        """This field crosses the MCP boundary as JSON, so serialisation counts."""
        brief = StoryBrief(
            child=child, premise="a lost hat", world_rules=WorldRules.REALISTIC
        )
        assert (
            StoryBrief.model_validate_json(brief.model_dump_json()).world_rules
            is WorldRules.REALISTIC
        )

    def test_rejects_invented_world_rules(self, child: ChildProfile) -> None:
        with pytest.raises(ValidationError):
            StoryBrief(child=child, premise="a lost hat", world_rules="whimsical")

    def test_world_rules_is_described_for_the_model(self) -> None:
        """The description is prompt text: it reaches the planner and the schema."""
        field = StoryBrief.model_json_schema()["properties"]["world_rules"]
        assert field.get("description")


class TestStoryOutline:
    @pytest.mark.parametrize("n", [4, 6, 8])
    def test_accepts_beat_counts_in_range(self, outline: StoryOutline, n: int) -> None:
        beats = [
            outline.beats[0].model_copy(update={"position": i + 1}) for i in range(n)
        ]
        assert len(outline.model_copy(update={"beats": beats}).beats) == n

    @pytest.mark.parametrize("n", [0, 1, 3, 9, 20])
    def test_rejects_beat_counts_out_of_range(
        self, outline: StoryOutline, n: int
    ) -> None:
        beats = [outline.beats[0]] * n
        with pytest.raises(ValidationError):
            StoryOutline(
                title=outline.title,
                logline=outline.logline,
                theme=outline.theme,
                characters=outline.characters,
                beats=beats,
            )

    def test_requires_at_least_one_character(self, outline: StoryOutline) -> None:
        with pytest.raises(ValidationError):
            StoryOutline(
                title=outline.title,
                logline=outline.logline,
                theme=outline.theme,
                characters=[],
                beats=outline.beats,
            )

    def test_beat_position_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            StoryBeat(
                position=0,
                function=NarrativeFunction.SETUP,
                title="T",
                summary="Something happens at the very beginning of the story.",
            )


def effective_description(prop: dict, defs: dict) -> str | None:
    """The description the model actually receives for a property.

    Pydantic omits a field-level ``description`` when it is byte-identical to the
    referenced type's docstring, since the ``$ref`` target already carries it --
    so a bare ``{"$ref": ...}`` is not an undescribed field. Resolving the ref is
    what MCP clients see too, because FastMCP inlines ``$defs``.
    """
    if "description" in prop:
        return prop["description"]

    ref = prop.get("$ref") or next(
        (item["$ref"] for item in prop.get("anyOf", []) if "$ref" in item), None
    )
    if ref:
        return defs.get(ref.rsplit("/", 1)[-1], {}).get("description")

    # Arrays describe their contents rather than themselves in some shapes.
    if "items" in prop:
        return effective_description(prop["items"], defs)
    return None


class TestSchemaIsPromptText:
    """Docstrings and descriptions reach the model, so their content matters."""

    def test_every_field_carries_a_description(self) -> None:
        """An undescribed field gives the model nothing to work from."""
        for model in (
            StoryOutline,
            StoryBeat,
            CharacterSketch,
            ChildProfile,
            StoryBrief,
            StoryGrounding,
            GroundedFact,
        ):
            schema = model.model_json_schema()
            defs = schema.get("$defs", {})
            for name, prop in schema["properties"].items():
                assert effective_description(prop, defs), (
                    f"{model.__name__}.{name} has no description the model can see"
                )

    def test_no_internal_rationale_leaks_into_prompts(self) -> None:
        """Engineering notes belong in `#` comments, which never reach the model.

        Regression test: class docstrings become the schema `description`, and an
        earlier version of these models sent phrases like "the Canon Agent" and
        "spend tokens" to Gemini as part of its instructions.
        """
        leaked_terms = (
            "Canon Agent",
            "later session",
            "spend tokens",
            "mutable object",
            # Added with the grounding models, whose `#` comments discuss MCP
            # tools, clients and provenance -- none of which is the researcher's
            # business and all of which would read to it as part of the task.
            "MCP",
            "client",
            "provenance",
            "finding D",
        )
        for model in (
            StoryOutline,
            StoryBrief,
            CharacterSketch,
            StoryBeat,
            ChildProfile,
            StoryGrounding,
            GroundedFact,
        ):
            schema_text = str(model.model_json_schema())
            for term in leaked_terms:
                assert term not in schema_text, f"{model.__name__} leaks {term!r}"
