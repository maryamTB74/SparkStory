"""The reshaped scene plan, and the review models.

These are Pydantic models bound with `with_structured_output`, so their
docstrings and field descriptions are prompt text. The tests here guard both
the shape and what that text says.
"""

import logging

import pytest

from sparkstory.entities.reviews import (
    OutlineReview,
    OutlineReviews,
    OutlineReviewsOutput,
    OutlineRubric,
    ProseReview,
    ProseReviews,
    ProseReviewsOutput,
    ProseRubric,
)
from sparkstory.entities.stories import (
    NarrativeFunction,
    PagePlan,
    ScenePlan,
    StoryOutline,
    StoryPage,
    StoryProse,
)
from sparkstory.workflows.reviews import (
    deterministic_prose_reviews,
    drop_unroutable_outline_reviews,
    drop_unroutable_prose_reviews,
    format_pacing_report,
    pages_per_beat,
)


class TestScenePlan:
    def test_splits_the_summary_into_three_fields(self) -> None:
        """One prose-shaped sentence is what the Writer paraphrased. Three
        orthogonal notes cannot be concatenated into a page."""
        page = ScenePlan(
            page_number=1,
            beat_position=1,
            setting="the garden at night",
            visual_action="Maryam at the window, moon low over the fence",
            emotional_shift="wonder tips into wanting",
            page_turn_hook="how could anyone get up there?",
            characters_present=["Maryam"],
        )
        assert page.visual_action
        assert page.emotional_shift
        assert page.page_turn_hook

    def test_scene_summary_is_gone(self) -> None:
        """A leftover field would let the Plot Planner keep emitting prose."""
        assert "scene_summary" not in ScenePlan.model_fields

    def test_page_turn_hook_is_optional(self) -> None:
        """The last page answers rather than asks."""
        page = ScenePlan(
            page_number=8,
            beat_position=4,
            setting="the garden at night",
            visual_action="Maryam and Pip asleep under the window",
            emotional_shift="wanting settles into having",
            characters_present=["Maryam", "Pip"],
        )
        assert page.page_turn_hook is None

    def test_descriptions_forbid_finished_prose(self) -> None:
        """The old description said 'one or two sentences', which is an
        instruction to write prose. These must not."""
        schema = ScenePlan.model_json_schema()
        for field in ("visual_action", "emotional_shift"):
            description = schema["properties"][field]["description"].lower()
            assert "sentence" not in description or "not" in description

    def test_page_plan_still_bounds_page_count(self) -> None:
        assert PagePlan.model_fields["pages"].metadata


class TestPacingMeasurement:
    def test_counts_pages_for_every_beat(
        self, outline: StoryOutline, page_plan: PagePlan
    ) -> None:
        """The conftest plan is (1,1,2,2,3,3,3,3,4,4)."""
        assert pages_per_beat(outline, page_plan) == {1: 2, 2: 2, 3: 4, 4: 2}

    def test_a_beat_with_no_pages_counts_zero(
        self, outline: StoryOutline, page_plan: PagePlan
    ) -> None:
        """This runs before validate_page_plan has necessarily approved the
        plan, so it must not raise on the way there -- and a bare Counter would
        omit the beat entirely, hiding the case you most want to see."""
        trimmed = page_plan.model_copy(
            update={"pages": [p for p in page_plan.pages if p.beat_position != 4]}
        )
        assert pages_per_beat(outline, trimmed)[4] == 0

    def test_report_names_the_narrative_function(
        self, outline: StoryOutline, page_plan: PagePlan
    ) -> None:
        """'beat 3 got 4 pages' is data; 'climax=4' is a finding."""
        report = format_pacing_report(outline, page_plan)
        assert f"{NarrativeFunction.CLIMAX.value}=4" in report


class TestOutlineReviewEntities:
    def test_empty_review_list_is_valid(self) -> None:
        """Empty is the stop signal. A reflexive min_length=1 would make the
        loop unable to terminate, and it would look like a critic that never
        approves rather than like a schema mistake."""
        assert OutlineReviewsOutput(reviews=[]).reviews == []

    def test_beat_position_is_optional_for_story_level_findings(self) -> None:
        """'The child is not the protagonist' is about the whole story."""
        review = OutlineReview(
            rubric=OutlineRubric.PROTAGONIST,
            comment="The want belongs to Pip; Maryam only helps.",
        )
        assert review.beat_position is None

    def test_rubrics_cover_the_two_observed_outline_failures(self) -> None:
        """Findings #6 and #7. Deliberately brittle: a third rubric should cost
        a conscious decision and a note on which failure motivated it."""
        assert set(OutlineRubric) == {
            OutlineRubric.PROTAGONIST,
            OutlineRubric.EARNED_RESOLUTION,
        }

    def test_wrapper_carries_the_outline_it_describes(
        self, outline: StoryOutline
    ) -> None:
        """One object, so a review list cannot be paired with the wrong draft."""
        assert OutlineReviews(outline=outline, reviews=[]).outline is outline

    def test_wrapper_is_not_the_output_schema(self) -> None:
        """Binding the fat model would pay the critic to echo the outline back
        on every pass."""
        assert "outline" not in OutlineReviewsOutput.model_fields

    def test_descriptions_leak_no_internal_terms(self) -> None:
        """The output schema is sent to the critic, so its text is prompt text."""
        rendered = str(OutlineReviewsOutput.model_json_schema()).lower()
        for internal in ("langgraph", "workflow", "pydantic", "node"):
            assert internal not in rendered


def _outline_review(beat: int | None) -> OutlineReview:
    return OutlineReview(
        rubric=OutlineRubric.PROTAGONIST,
        beat_position=beat,
        comment="The want belongs to Pip; Maryam only helps him get it.",
    )


class TestDropUnroutableOutlineReviews:
    def test_keeps_reviews_pointing_at_real_beats(self, outline: StoryOutline) -> None:
        reviews = OutlineReviews(outline=outline, reviews=[_outline_review(2)])
        assert len(drop_unroutable_outline_reviews(reviews, outline).reviews) == 1

    def test_keeps_story_level_reviews(self, outline: StoryOutline) -> None:
        """None means 'the story as a whole', not 'a missing beat'."""
        reviews = OutlineReviews(outline=outline, reviews=[_outline_review(None)])
        assert len(drop_unroutable_outline_reviews(reviews, outline).reviews) == 1

    def test_drops_a_review_citing_a_beat_that_does_not_exist(
        self, outline: StoryOutline
    ) -> None:
        """The outline fixture has four beats. Passing beat 9 through would
        invite the planner to invent one to satisfy it."""
        reviews = OutlineReviews(outline=outline, reviews=[_outline_review(9)])
        assert drop_unroutable_outline_reviews(reviews, outline).reviews == []

    def test_the_drop_is_logged_at_warning(
        self, outline: StoryOutline, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Silent filtering would hide a critic that has stopped making sense."""
        reviews = OutlineReviews(outline=outline, reviews=[_outline_review(9)])
        with caplog.at_level(logging.WARNING):
            drop_unroutable_outline_reviews(reviews, outline)
        assert "9" in caplog.text

    def test_dropping_does_not_change_the_carried_outline(
        self, outline: StoryOutline
    ) -> None:
        reviews = OutlineReviews(outline=outline, reviews=[_outline_review(9)])
        assert drop_unroutable_outline_reviews(reviews, outline).outline is outline


class TestProseReviewEntities:
    def test_empty_review_list_is_valid(self) -> None:
        """Empty is the stop signal, same as on the outline side."""
        assert ProseReviewsOutput(reviews=[]).reviews == []

    def test_page_number_is_optional_for_book_level_findings(self) -> None:
        """'Six of eight pages open with a name' is about no single page."""
        review = ProseReview(
            rubric=ProseRubric.READ_ALOUD,
            comment="Six of eight pages begin with a character's name.",
        )
        assert review.page_number is None

    def test_rubrics_cover_the_observed_prose_failures_plus_safety(self) -> None:
        """Four from failures read in a real run, plus safety -- which is a
        guardrail rather than a quality rubric."""
        assert set(ProseRubric) == {
            ProseRubric.PLAN_FIDELITY,
            ProseRubric.READ_ALOUD,
            ProseRubric.INTERIORITY,
            ProseRubric.READING_LEVEL,
            ProseRubric.SAFETY,
        }

    def test_wrapper_carries_the_prose_it_describes(self, prose: StoryProse) -> None:
        assert ProseReviews(prose=prose, reviews=[]).prose is prose

    def test_wrapper_is_not_the_output_schema(self) -> None:
        """Binding the fat model would pay the critic to echo the book back."""
        assert "prose" not in ProseReviewsOutput.model_fields

    def test_safety_is_addressable_in_code(self) -> None:
        """Fail-closed needs to single safety out of a mixed list, which is the
        whole reason the rubric is an enum and not a free-text string."""
        mixed = [
            ProseReview(rubric=ProseRubric.READ_ALOUD, comment="This page drones."),
            ProseReview(rubric=ProseRubric.SAFETY, comment="A spider on page four."),
        ]
        assert [r for r in mixed if r.rubric is ProseRubric.SAFETY]


def _prose_opening_with(first_words: list[str]) -> StoryProse:
    return StoryProse(
        pages=[
            StoryPage(page_number=i, text=f"{word} looked up at the moon.")
            for i, word in enumerate(first_words, start=1)
        ]
    )


_DRONING = ["Maryam"] * 6 + ["Softly", "Under", "The", "Up"]
_VARIED = ["Maryam", "Softly", "Under", "The", "Up", "Pip", "Down", "Away", "In", "So"]


class TestDeterministicProseReviews:
    def test_flags_repetitive_openings(self, page_plan: PagePlan) -> None:
        """An early book opened six of its eight pages with a character's name."""
        reviews = deterministic_prose_reviews(_prose_opening_with(_DRONING), page_plan)
        assert [r.rubric for r in reviews] == [ProseRubric.READ_ALOUD]

    def test_says_how_many_and_which_word(self, page_plan: PagePlan) -> None:
        """'Vary how sentences begin' is the instruction that already failed. A
        count and the actual word is a finding rather than a restatement."""
        reviews = deterministic_prose_reviews(_prose_opening_with(_DRONING), page_plan)
        comment = reviews[0].comment
        assert "6" in comment
        assert "Maryam" in comment

    def test_varied_openings_produce_no_review(self, page_plan: PagePlan) -> None:
        reviews = deterministic_prose_reviews(_prose_opening_with(_VARIED), page_plan)
        assert reviews == []

    def test_the_threshold_is_proportional_not_a_flat_count(
        self, page_plan: PagePlan
    ) -> None:
        """Three of ten pages sharing an opening is unremarkable; a flat count of
        three would fire here and on 3-of-24 too."""
        mild = ["Maryam"] * 3 + ["Softly", "Under", "The", "Up", "Pip", "Down", "In"]
        assert deterministic_prose_reviews(_prose_opening_with(mild), page_plan) == []

    def test_the_finding_is_book_level(self, page_plan: PagePlan) -> None:
        """No single page is at fault; the pattern is."""
        reviews = deterministic_prose_reviews(_prose_opening_with(_DRONING), page_plan)
        assert reviews[0].page_number is None

    def test_punctuation_does_not_hide_a_repeat(self, page_plan: PagePlan) -> None:
        """'Maryam,' and 'Maryam' are the same drone to a listener."""
        noisy = ['"Maryam,'] + ["Maryam!"] * 5 + ["Softly", "Under", "The", "Up"]
        reviews = deterministic_prose_reviews(_prose_opening_with(noisy), page_plan)
        assert len(reviews) == 1

    def test_an_empty_page_does_not_crash_the_count(self, page_plan: PagePlan) -> None:
        """validate_prose rejects blank pages, but this runs on unvalidated
        prose during a revision pass."""
        prose = StoryProse(
            pages=[StoryPage(page_number=i, text=" ") for i in range(1, 11)]
        )
        assert deterministic_prose_reviews(prose, page_plan) == []


class TestDropUnroutableProseReviews:
    def test_drops_a_review_citing_a_page_that_does_not_exist(
        self, prose: StoryProse, page_plan: PagePlan
    ) -> None:
        """The page_plan fixture has ten pages."""
        reviews = ProseReviews(
            prose=prose,
            reviews=[
                ProseReview(
                    rubric=ProseRubric.READ_ALOUD,
                    page_number=99,
                    comment="This page stumbles when read aloud.",
                )
            ],
        )
        assert drop_unroutable_prose_reviews(reviews, page_plan).reviews == []

    def test_keeps_book_level_reviews(
        self, prose: StoryProse, page_plan: PagePlan
    ) -> None:
        reviews = ProseReviews(
            prose=prose,
            reviews=[
                ProseReview(
                    rubric=ProseRubric.READ_ALOUD,
                    comment="Every page opens the same way.",
                )
            ],
        )
        assert len(drop_unroutable_prose_reviews(reviews, page_plan).reviews) == 1

    def test_keeps_reviews_pointing_at_real_pages(
        self, prose: StoryProse, page_plan: PagePlan
    ) -> None:
        reviews = ProseReviews(
            prose=prose,
            reviews=[
                ProseReview(
                    rubric=ProseRubric.INTERIORITY,
                    page_number=4,
                    comment="Page four is all action; the feeling never arrives.",
                )
            ],
        )
        assert len(drop_unroutable_prose_reviews(reviews, page_plan).reviews) == 1
