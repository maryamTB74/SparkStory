"""Prompt assembly.

Prompt text is behaviour, not presentation: an omitted constraint changes what
the model produces. These tests treat the rendered prompt as an output worth
asserting on.
"""

import importlib
import pkgutil
import re

import pytest

import sparkstory.nodes
from sparkstory.entities.grounding import GroundedFact, StoryGrounding
from sparkstory.entities.guidelines import READING_LEVEL_GUIDANCE
from sparkstory.entities.stories import (
    ChildProfile,
    PagePlan,
    ReadingLevel,
    StoryBrief,
    StoryOutline,
    WorldRules,
)
from sparkstory.nodes.outline_critic import OUTLINE_CRITIC_SYSTEM_PROMPT
from sparkstory.nodes.plot_planner import PLOT_PLANNER_SYSTEM_PROMPT
from sparkstory.nodes.story_planner import (
    OUTLINE_REVISION_PROMPT_TEMPLATE,
    STORY_PLANNER_SYSTEM_PROMPT,
    render_grounding,
    render_story_brief,
)
from sparkstory.nodes.writer import (
    PROSE_REVISION_PROMPT_TEMPLATE,
    WRITER_SYSTEM_PROMPT,
    render_prose_request,
)


class TestReadingLevelGuidance:
    def test_every_level_has_guidance(self) -> None:
        """Adding a ReadingLevel must not silently miss the prompt."""
        missing = [
            level for level in ReadingLevel if level not in READING_LEVEL_GUIDANCE
        ]
        assert not missing, f"no guidance for {missing}"


class TestRenderStoryBrief:
    def test_includes_child_details(self, brief: StoryBrief) -> None:
        rendered = render_story_brief(brief)
        assert brief.child.name in rendered
        assert str(brief.child.age) in rendered
        assert brief.child.pronouns.value in rendered

    def test_includes_reading_level_guidance(self, brief: StoryBrief) -> None:
        assert READING_LEVEL_GUIDANCE[brief.child.reading_level] in render_story_brief(
            brief
        )

    def test_includes_hard_constraints(self, brief: StoryBrief) -> None:
        rendered = render_story_brief(brief)
        for avoided in brief.avoid:
            assert avoided in rendered

    def test_omits_empty_optional_sections(self, child: ChildProfile) -> None:
        """Empty headings are wasted tokens and invite the model to fill them."""
        minimal = StoryBrief(
            child=child.model_copy(update={"interests": []}), premise="a lost hat"
        )
        rendered = render_story_brief(minimal)
        assert "Interests:" not in rendered
        assert "Must include:" not in rendered
        assert "Must avoid entirely:" not in rendered


class TestSystemPrompt:
    def test_describes_craft_not_output_format(self) -> None:
        """Structured output enforces shape; restating it wastes tokens and drifts."""
        lowered = STORY_PLANNER_SYSTEM_PROMPT.lower()
        for forbidden in ("json", "schema", "```", "field"):
            assert forbidden not in lowered, (
                f"prompt describes output format: {forbidden!r}"
            )

    def test_states_the_non_negotiable_rules(self) -> None:
        lowered = STORY_PLANNER_SYSTEM_PROMPT.lower()
        assert "pronouns" in lowered
        assert "avoid" in lowered
        assert "moralise" in lowered

    def test_caps_beats_by_the_page_count(self) -> None:
        """A 6-beat outline for a 5-page book is unbuildable, so say so up front.

        Only the prompt can prevent this; validation can merely reject it after a
        model call has been paid for.
        """
        assert "never plan more beats than the book has pages" in (
            STORY_PLANNER_SYSTEM_PROMPT.lower()
        )

    def test_the_beat_limit_is_rendered_as_a_number(self, brief: StoryBrief) -> None:
        """ "Target length" alone left the limit to be inferred, and it was not."""
        assert f"at most {brief.page_count} beats" in render_story_brief(brief)


class TestPlotPlannerSystemPrompt:
    def test_demands_notes_not_narration(self) -> None:
        """The schema description alone is what failed before: the planner
        emitted finished sentences and the Writer paraphrased them."""
        assert "notes, not narration" in PLOT_PLANNER_SYSTEM_PROMPT.lower()

    def test_names_all_three_note_fields(self) -> None:
        """A prompt describing only two leaves the third to the schema, which
        is the situation this change exists to leave."""
        lowered = PLOT_PLANNER_SYSTEM_PROMPT.lower()
        assert "what the picture shows" in lowered
        assert "what changes inside" in lowered
        assert "page turn" in lowered

    def test_describes_craft_not_output_format(self) -> None:
        lowered = PLOT_PLANNER_SYSTEM_PROMPT.lower()
        for forbidden in ("json", "schema", "```"):
            assert forbidden not in lowered, (
                f"prompt describes output format: {forbidden!r}"
            )


class TestRenderProseRequest:
    def test_renders_all_three_notes_per_page(
        self, brief: StoryBrief, outline: StoryOutline, page_plan: PagePlan
    ) -> None:
        rendered = render_prose_request(brief, outline, page_plan)
        first = page_plan.pages[0]
        assert first.visual_action in rendered
        assert first.emotional_shift in rendered
        assert first.page_turn_hook in rendered

    def test_omits_the_hook_on_the_final_page(
        self, brief: StoryBrief, outline: StoryOutline, page_plan: PagePlan
    ) -> None:
        """A 'leaves open:' line with nothing after it invites an invented
        cliffhanger on the page where the book ends."""
        rendered = render_prose_request(brief, outline, page_plan)
        last_block = rendered.rsplit("Page ", 1)[1]
        assert "leaves open:" not in last_block


class TestWriterSystemPrompt:
    def test_forbids_copying_the_notes(self) -> None:
        """Finding #1: four of eight pages were the plan with the tense changed."""
        assert "never copy their wording" in WRITER_SYSTEM_PROMPT.lower()

    def test_requires_all_three_notes_to_land(self) -> None:
        """Without this, 'do not copy' is satisfiable by ignoring the notes --
        which is the page-drift half of the same defect."""
        assert "all three" in WRITER_SYSTEM_PROMPT.lower()


#: Words that belong to us, not to a children's book editor. Each would read to
#: the model as part of its task. An earlier version transmitted "the Canon
#: Agent" and "spend tokens" to Gemini exactly this way.
_INTERNAL_TERMS = (
    "langgraph",
    "langchain",
    "pydantic",
    "json",
    "schema",
    "workflow",
    "pipeline",
    "spend tokens",
    "token budget",
    "agent",
    "node",
    "rubric",
    # Shorthand for this project's own history -- a numbered rule, a lettered
    # finding, a session. It reads as an instruction to a model that cannot
    # resolve it, and it belongs in a `#` comment, which never reaches one.
    # Note "finding" and "session" alone are ordinary words a prompt may use:
    # a critic reports findings. Only the citation forms are banned, so the
    # patterns below are matched as regexes rather than as substrings.
    "non-obvious rule",
    "claude.md",
)

#: Citation forms of the same shorthand, which a plain substring cannot catch
#: without also banning the ordinary English words they are built from.
_INTERNAL_PATTERNS = (
    r"finding [a-z]{1,2}\b",
    r"\bsession \d",
    r"\brule \d",
)


def _prompt_constants() -> dict[str, str]:
    """Every ``*_PROMPT`` / ``*_PROMPT_TEMPLATE`` across the nodes package.

    Discovered rather than listed. A hardcoded list of modules silently stops
    covering new nodes, and it stops covering them at exactly the moment one is
    added -- which is when the audit is most needed.
    """
    found: dict[str, str] = {}
    for info in pkgutil.iter_modules(sparkstory.nodes.__path__):
        module = importlib.import_module(f"sparkstory.nodes.{info.name}")
        for name in dir(module):
            if name.endswith(("_PROMPT", "_PROMPT_TEMPLATE")):
                found[f"{info.name}.{name}"] = getattr(module, name)
    return found


class TestNoInternalTermsLeak:
    def test_the_audit_actually_finds_prompts(self) -> None:
        """A broken walk would make every assertion below vacuously pass, which
        is worse than no audit because it reads as coverage."""
        found = _prompt_constants()
        assert len(found) >= 5, f"only found {sorted(found)}"

    def test_the_audit_covers_every_node_module(self) -> None:
        """A node whose prompt is built inline rather than as a named constant
        would slip past the discovery entirely."""
        covered = {name.split(".")[0] for name in _prompt_constants()}
        modules = {
            info.name
            for info in pkgutil.iter_modules(sparkstory.nodes.__path__)
            if info.name not in {"base", "__init__"}
        }
        assert modules <= covered, f"no prompt constant found in {modules - covered}"

    def test_no_prompt_mentions_our_machinery(self) -> None:
        for where, text in _prompt_constants().items():
            lowered = text.lower()
            for term in _INTERNAL_TERMS:
                assert term not in lowered, f"{where} leaks {term!r} to the model"

    def test_no_prompt_cites_our_own_history(self) -> None:
        """A model cannot resolve "rule 13" or "finding Q", so such a citation
        reads as an instruction referring to something absent."""
        for where, text in _prompt_constants().items():
            lowered = text.lower()
            for pattern in _INTERNAL_PATTERNS:
                match = re.search(pattern, lowered)
                assert match is None, f"{where} cites {match.group()!r} to the model"


class TestResourceTextIsPromptText:
    """A resource is read by a *client's model*, so it is prompt text too.

    The tools and prompts have been audited from the start; resources are a newer
    surface with the same property, and it would be easy to treat their output as
    a debug dump because that is what an introspection endpoint usually is.
    """

    def _resource_text(self) -> dict[str, str]:
        from sparkstory.mcp.resources.library import read_corpus, read_library

        return {"library": read_library(), "corpus": read_corpus()}

    def test_the_audit_actually_reads_something(self) -> None:
        # A sweep over empty strings passes without checking anything.
        assert all(text for text in self._resource_text().values())

    def test_no_resource_mentions_our_machinery(self) -> None:
        # `json` and `schema` are excluded from the term list here: a resource
        # legitimately reports file counts and formats, and the terms that matter
        # for a resource are the orchestration ones.
        terms = [t for t in _INTERNAL_TERMS if t not in {"json", "schema"}]
        for where, text in self._resource_text().items():
            lowered = text.lower()
            for term in terms:
                assert term not in lowered, f"{where} leaks {term!r} to the model"


class TestProseRevisionPrompt:
    def test_forbids_writing_the_note_down(self) -> None:
        """Live-run regression: an interiority finding was "fixed" by copying
        the note onto the page, which is the failure it described."""
        assert "satisfying a note never means writing the note down" in (
            PROSE_REVISION_PROMPT_TEMPLATE.lower()
        )

    def test_still_protects_uncriticised_pages(self) -> None:
        """The live run left 5 of 8 pages byte-identical. Keep that."""
        assert "unchanged" in PROSE_REVISION_PROMPT_TEMPLATE.lower()


class TestOutlineRevisionPrompt:
    def test_exempts_protagonist_from_keeping_what_was_not_criticised(self) -> None:
        """Live-run regression (outputs/20260730-232426-*). "Keep everything that
        was not criticised" licensed a one-clause patch to beat 1 while the
        logline still read "to help her fox friend reach the moon"."""
        lowered = OUTLINE_REVISION_PROMPT_TEMPLATE.lower()
        assert "cannot be fixed in one beat" in lowered
        assert "logline" in lowered

    def test_still_protects_uncriticised_beats(self) -> None:
        """The exemption must not become a licence to churn everything."""
        assert "keep everything that was not criticised" in (
            OUTLINE_REVISION_PROMPT_TEMPLATE.lower()
        )


class TestRenderGrounding:
    """What research found, as the planner sees it.

    The assertion that matters is ``test_the_claim_itself_is_never_rendered``. The
    planner prompt already says "do not have someone recite facts about planets",
    and the laziest way to satisfy "use what research found" is to have a character
    recite it, because an instruction gets satisfied the laziest legal way.
    Splitting `claim` from `story_note` only
    helps if the claim genuinely never reaches the prompt.
    """

    def test_renders_each_note(self) -> None:
        rendered = render_grounding(
            StoryGrounding(
                facts=[
                    GroundedFact(
                        claim="The Moon has no air.",
                        story_note="Nothing outdoors can flutter or drift.",
                        source="NASA -- Earth's Moon",
                        chunk_id="moon#1",
                    )
                ]
            ),
            WorldRules.REALISTIC,
        )
        assert "Nothing outdoors can flutter or drift." in rendered

    @pytest.mark.parametrize("world_rules", list(WorldRules))
    def test_the_claim_itself_is_never_rendered(self, world_rules: WorldRules) -> None:
        """In neither mode. Imaginative treats facts as texture rather than law,
        which is exactly the reading under which handing over the raw claim would
        seem harmless -- so this is checked in both."""
        rendered = render_grounding(
            StoryGrounding(
                facts=[
                    GroundedFact(
                        claim="The Moon has no air.",
                        story_note="Nothing outdoors can flutter or drift.",
                        source="NASA -- Earth's Moon",
                        chunk_id="moon#1",
                    )
                ]
            ),
            world_rules,
        )
        assert "The Moon has no air." not in rendered

    def test_attribution_is_never_rendered(self) -> None:
        """A source matters for checking a claim later and means nothing to a story
        planner, so it is not worth a single token of its attention."""
        rendered = render_grounding(
            StoryGrounding(
                facts=[
                    GroundedFact(
                        claim="The Moon has no air.",
                        story_note="Nothing outdoors can flutter.",
                        source="NASA -- Earth's Moon",
                        chunk_id="moon#1",
                    )
                ]
            ),
            WorldRules.REALISTIC,
        )
        assert "NASA" not in rendered
        assert "moon#1" not in rendered

    @pytest.mark.parametrize("world_rules", list(WorldRules))
    def test_both_modes_forbid_stating_the_note_aloud(
        self, world_rules: WorldRules
    ) -> None:
        """The load-bearing assertion of the whole feature.

        Facts-as-texture is what reintroduces the recital trap that splitting
        `claim` from `story_note` was built to prevent: once a fact is "detail
        that makes the magic believable", reciting it reads as using it. This one
        sentence is the only thing standing between texture and a character
        delivering a science lecture, and it is deliberately the *same* string in
        both branches so the two cannot drift apart.
        """
        rendered = render_grounding(
            StoryGrounding(
                facts=[
                    GroundedFact(
                        claim="The Moon has no air.",
                        story_note="Nothing outdoors can flutter.",
                        source="NASA",
                        chunk_id="moon#1",
                    )
                ]
            ),
            world_rules,
        )
        assert "Do not state them" in rendered

    def test_realistic_mode_is_unchanged(self) -> None:
        """Today's text, verbatim, so nothing regresses for a realistic story."""
        rendered = render_grounding(
            StoryGrounding(
                facts=[
                    GroundedFact(
                        claim="The Moon has no air.",
                        story_note="Nothing outdoors can flutter.",
                        source="NASA",
                        chunk_id="moon#1",
                    )
                ]
            ),
            WorldRules.REALISTIC,
        )
        assert "This story is set in the real world" in rendered
        assert "Let the story simply obey them" in rendered

    def test_imaginative_mode_permits_breaking_a_fact_but_sparingly(self) -> None:
        """Three things this wording has to do at once, each from a live failure.

        Facts are *detail* rather than law; the premise may break them, because
        the run this feature exists to fix refused to let a paper rocket fly; and
        breaks should be few, which is the spec's recommendation B expressed as
        wording rather than as machinery the planner could get wrong.
        """
        rendered = render_grounding(
            StoryGrounding(
                facts=[
                    GroundedFact(
                        claim="The Moon has no air.",
                        story_note="Nothing outdoors can flutter.",
                        source="NASA",
                        chunk_id="moon#1",
                    )
                ]
            ),
            WorldRules.IMAGINATIVE,
        )
        assert "Use them as detail" in rendered
        assert "The premise may break them" in rendered
        assert "break as few as you can" in rendered
        assert "Let the story simply obey them" not in rendered

    @pytest.mark.parametrize("world_rules", list(WorldRules))
    def test_empty_grounding_renders_nothing_at_all(
        self, world_rules: WorldRules
    ) -> None:
        """So a brief with no grounding produces a byte-identical prompt to the one
        this project sent before research existed -- which is what keeps the
        provider-side cached prefix, and every existing prompt test, intact. True
        in both modes, or the mode itself would change an ungrounded prompt."""
        assert render_grounding(StoryGrounding(), world_rules) == ""

    def test_a_brief_with_no_grounding_is_unchanged(self, brief: StoryBrief) -> None:
        assert render_story_brief(brief) + render_grounding(
            StoryGrounding(), brief.world_rules
        ) == (render_story_brief(brief))


class TestProtagonistYieldsToThePremise:
    """The rubric must not overrule the parent's own idea.

    The eagle brief ("an eagle who discover a new planet") produced the same
    single `protagonist` finding three times and hit the cap, because the rubric
    demanded the want belong to the child while the premise named an eagle. The
    compromise made the eagle an "experiment visitor", which reads oddly and is
    not what was asked for.

    **The floor stays.** Every instruction that relaxes a constraint is a licence
    to under-fix, and the obvious under-fix here is a passive child -- the exact
    defect the rubric was created for, where the want belonged to the animal and
    the child only helped. So the relaxation is about *exclusivity of the want*,
    never about the child acting.
    """

    def test_a_premise_naming_another_character_is_respected(self) -> None:
        lowered = OUTLINE_CRITIC_SYSTEM_PROMPT.lower()
        assert "follow the parent" in lowered
        assert "do not report a finding merely because another character" in lowered

    def test_the_child_must_still_drive_the_story(self) -> None:
        """The floor. Sharing the story is allowed; watching it is not."""
        lowered = OUTLINE_CRITIC_SYSTEM_PROMPT.lower()
        assert "what is never enough" in lowered
        assert "only watches" in lowered

    def test_the_ending_must_still_turn_on_the_child(self) -> None:
        """The one test that cannot be satisfied by a bystander."""
        lowered = OUTLINE_CRITIC_SYSTEM_PROMPT.lower()
        assert "would resolve identically with the child removed" in lowered

    def test_the_revision_prompt_does_not_tell_the_planner_to_delete_the_premise(
        self,
    ) -> None:
        """Otherwise the two prompts fight: the critic now permits a shared
        story while the reviser is still told to move the want wholesale, and
        the planner would keep deleting the eagle to satisfy it."""
        lowered = OUTLINE_REVISION_PROMPT_TEMPLATE.lower()
        assert "not by removing the other character" in lowered
