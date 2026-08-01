"""Prompt assembly.

Prompt text is behaviour, not presentation: an omitted constraint changes what
the model produces. These tests treat the rendered prompt as an output worth
asserting on.
"""

import importlib
import pkgutil

import sparkstory.nodes
from sparkstory.entities.guidelines import READING_LEVEL_GUIDANCE
from sparkstory.entities.stories import (
    ChildProfile,
    PagePlan,
    ReadingLevel,
    StoryBrief,
    StoryOutline,
)
from sparkstory.nodes.plot_planner import PLOT_PLANNER_SYSTEM_PROMPT
from sparkstory.nodes.story_planner import (
    OUTLINE_REVISION_PROMPT_TEMPLATE,
    STORY_PLANNER_SYSTEM_PROMPT,
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
