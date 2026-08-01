"""Structural checks on model output.

These are the assertions a Pydantic schema cannot make, because they compare the
output to the *request*. Each test breaks a valid fixture in exactly one way, so a
failure names the rule that broke.
"""

import pytest

from sparkstory.entities.exceptions import (
    ConfigurationError,
    SparkStoryError,
    StoryStructureError,
)
from sparkstory.entities.stories import (
    PagePlan,
    ScenePlan,
    StoryBrief,
    StoryOutline,
    StoryPage,
    StoryProse,
)
from sparkstory.workflows.validation import (
    validate_outline,
    validate_page_plan,
    validate_prose,
)


class TestOutlineFitsThePageCount:
    """
    A 5-page brief produced a 6-beat outline. Every beat needs a page of its own,
    so the page plan could not satisfy "every beat gets a page" -- an invariant
    that was arithmetically impossible to meet. Both models are individually
    valid (4-24 pages, 4-8 beats), so nothing prevented the combination, and the
    fixtures here used 10 pages with 4 beats and never came close.
    """

    def test_the_fixture_fits(self, brief: StoryBrief, outline: StoryOutline) -> None:
        validate_outline(brief, outline)

    def test_more_beats_than_pages_is_rejected(
        self, brief: StoryBrief, outline: StoryOutline
    ) -> None:
        five_beats = outline.model_copy(
            update={
                "beats": [
                    *outline.beats,
                    outline.beats[-1].model_copy(update={"position": 5}),
                ]
            }
        )
        four_pages = brief.model_copy(update={"page_count": 4})

        with pytest.raises(StoryStructureError, match="5 beats but the book has only"):
            validate_outline(four_pages, five_beats)

    def test_fewer_beats_than_pages_is_fine(
        self, brief: StoryBrief, outline: StoryOutline
    ) -> None:
        """A short book wants four strong beats, not one beat per page."""
        validate_outline(brief.model_copy(update={"page_count": 24}), outline)

    def test_exactly_as_many_beats_as_pages_is_fine(
        self, brief: StoryBrief, outline: StoryOutline
    ) -> None:
        """The boundary: one page per beat is tight but buildable."""
        validate_outline(brief.model_copy(update={"page_count": 4}), outline)


class TestValidPlan:
    def test_the_fixture_passes(
        self, brief: StoryBrief, outline: StoryOutline, page_plan: PagePlan
    ) -> None:
        """Guards the other tests: they only mean something if the baseline is valid."""
        validate_page_plan(brief, outline, page_plan)


class TestPageCount:
    def test_too_few_pages_is_rejected(
        self, brief: StoryBrief, outline: StoryOutline, page_plan: PagePlan
    ) -> None:
        short = PagePlan(pages=page_plan.pages[:-1])
        with pytest.raises(StoryStructureError, match="9 pages but the brief asked"):
            validate_page_plan(brief, outline, short)

    def test_too_many_pages_is_rejected(
        self, brief: StoryBrief, outline: StoryOutline, page_plan: PagePlan
    ) -> None:
        extra = page_plan.pages[-1].model_copy(update={"page_number": 11})
        long = PagePlan(pages=[*page_plan.pages, extra])
        with pytest.raises(StoryStructureError, match="11 pages"):
            validate_page_plan(brief, outline, long)


class TestBeatCoverage:
    def test_a_dropped_beat_is_rejected(
        self, brief: StoryBrief, outline: StoryOutline, page_plan: PagePlan
    ) -> None:
        """A missing climax is invisible in the output: the plan still reads fine."""
        pages = [
            page.model_copy(update={"beat_position": 2})
            if page.beat_position == 3
            else page
            for page in page_plan.pages
        ]
        with pytest.raises(StoryStructureError, match=r"Beats \[3\] have no page"):
            validate_page_plan(brief, outline, PagePlan(pages=pages))

    def test_a_beat_that_does_not_exist_is_rejected(
        self, brief: StoryBrief, outline: StoryOutline, page_plan: PagePlan
    ) -> None:
        pages = list(page_plan.pages)
        pages[-1] = pages[-1].model_copy(update={"beat_position": 9})
        with pytest.raises(StoryStructureError, match=r"cites beats \[9\]"):
            validate_page_plan(brief, outline, PagePlan(pages=pages))


class TestStoryOrder:
    def test_pages_moving_backwards_are_rejected(
        self, brief: StoryBrief, outline: StoryOutline, page_plan: PagePlan
    ) -> None:
        """Catches a shuffled structure whose individual pages all look fine."""
        pages = list(page_plan.pages)
        pages[2], pages[8] = (
            pages[2].model_copy(update={"beat_position": 4}),
            pages[8].model_copy(update={"beat_position": 2}),
        )
        with pytest.raises(StoryStructureError, match="move backwards"):
            validate_page_plan(brief, outline, PagePlan(pages=pages))


class TestProse:
    def test_matching_prose_passes(
        self, page_plan: PagePlan, prose: StoryProse
    ) -> None:
        validate_prose(page_plan, prose)

    def test_a_missing_page_is_rejected(
        self, page_plan: PagePlan, prose: StoryProse
    ) -> None:
        with pytest.raises(StoryStructureError, match="Prose covers pages"):
            validate_prose(page_plan, StoryProse(pages=prose.pages[:-1]))

    def test_reordered_pages_are_rejected(
        self, page_plan: PagePlan, prose: StoryProse
    ) -> None:
        swapped = [prose.pages[1], prose.pages[0], *prose.pages[2:]]
        with pytest.raises(StoryStructureError):
            validate_prose(page_plan, StoryProse(pages=swapped))

    def test_whitespace_only_text_is_rejected(
        self, page_plan: PagePlan, prose: StoryProse
    ) -> None:
        """`min_length=1` rejects "" but not "   ", which prints as a blank page."""
        pages = list(prose.pages)
        pages[3] = StoryPage(page_number=pages[3].page_number, text="   ")
        with pytest.raises(StoryStructureError, match=r"Pages \[4\] have no text"):
            validate_prose(page_plan, StoryProse(pages=pages))


class TestExceptionTaxonomy:
    def test_is_not_a_configuration_error(self) -> None:
        """No operator can fix malformed model output by editing .env.

        If this inherited ConfigurationError, the tool layer would translate it
        into a client-facing "fix your configuration" message and send debugging
        to entirely the wrong layer.
        """
        assert not issubclass(StoryStructureError, ConfigurationError)
        assert issubclass(StoryStructureError, SparkStoryError)


class TestScenePlanSchema:
    def test_page_and_beat_positions_are_one_based(self) -> None:
        with pytest.raises(ValueError):
            ScenePlan(
                page_number=0,
                beat_position=1,
                setting="a garden",
                visual_action="something happens here",
                emotional_shift="a small change",
            )
