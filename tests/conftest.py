"""Shared fixtures.

Fixtures build objects explicitly rather than reading from ``.env`` so tests do
not depend on whatever happens to be configured on the machine running them.
"""

from collections.abc import Callable

import pytest

from sparkstory.entities.stories import (
    CharacterSketch,
    ChildProfile,
    NarrativeFunction,
    PagePlan,
    Pronouns,
    ReadingLevel,
    ScenePlan,
    Story,
    StoryBeat,
    StoryBrief,
    StoryOutline,
    StoryPage,
    StoryProse,
    Tone,
)


@pytest.fixture
def child() -> ChildProfile:
    return ChildProfile(
        name="Maryam",
        age=5,
        pronouns=Pronouns.SHE_HER,
        reading_level=ReadingLevel.EARLY_READER,
        interests=["foxes", "astronomy"],
    )


@pytest.fixture
def brief(child: ChildProfile) -> StoryBrief:
    return StoryBrief(
        child=child,
        premise="a fox who wants to visit the moon",
        tone=Tone.MAGICAL,
        page_count=10,
        must_include=["a paper rocket"],
        avoid=["spiders", "the dark"],
    )


@pytest.fixture
def outline() -> StoryOutline:
    """A minimal valid outline: exactly the four-beat floor."""
    return StoryOutline(
        title="Maryam and the Paper Rocket",
        logline="A girl and a fox build a rocket to say goodnight to the moon.",
        theme="trying something new even when it feels too big",
        characters=[
            CharacterSketch(
                name="Maryam",
                role="main character",
                description="A curious girl who loves looking at the stars.",
            ),
            CharacterSketch(
                name="Pip",
                role="loyal companion",
                description="A small fox who is braver than he feels.",
            ),
        ],
        beats=[
            StoryBeat(
                position=1,
                function=NarrativeFunction.SETUP,
                title="Goodnight, moon",
                summary=(
                    "Maryam waves to the moon each night and wishes it were closer."
                ),
                characters_present=["Maryam"],
            ),
            StoryBeat(
                position=2,
                function=NarrativeFunction.INCITING_INCIDENT,
                title="The paper rocket",
                summary=(
                    "Pip brings her a paper rocket and suggests they visit instead."
                ),
                characters_present=["Maryam", "Pip"],
            ),
            StoryBeat(
                position=3,
                function=NarrativeFunction.CLIMAX,
                title="Too high",
                summary="The rocket will not fly and Maryam nearly gives up trying.",
                characters_present=["Maryam", "Pip"],
            ),
            StoryBeat(
                position=4,
                function=NarrativeFunction.RESOLUTION,
                title="A closer moon",
                summary="They find the moon reflected in a puddle and say goodnight.",
                characters_present=["Maryam", "Pip"],
            ),
        ],
    )


#: Beat each page draws from, in order: ten pages covering all four beats, with the
#: climax given the most room. Matches `brief.page_count` and stays non-decreasing,
#: so this plan passes validation and a test that wants a failure must break it
#: deliberately.
_PAGE_BEATS = (1, 1, 2, 2, 3, 3, 3, 3, 4, 4)


@pytest.fixture
def page_plan() -> PagePlan:
    """A valid ten-page plan for the `outline` fixture and `brief.page_count`."""
    return PagePlan(
        pages=[
            ScenePlan(
                page_number=number,
                beat_position=beat,
                setting="the garden at night",
                visual_action=f"a single quiet moment, drawn on page {number}",
                emotional_shift="something small shifts",
                # None on the last page only, so the fixture exercises both
                # branches of the optional field rather than just one.
                page_turn_hook=None if number == len(_PAGE_BEATS) else "and then?",
                characters_present=["Maryam", "Pip"],
            )
            for number, beat in enumerate(_PAGE_BEATS, start=1)
        ]
    )


#: One distinct opening word per page. Deliberately varied: the deterministic
#: read-aloud check in `workflows/reviews.py` flags pages that share an opening,
#: so prose reading "Page 1.", "Page 2." would produce a finding in every test
#: that runs the prose loop and drown out whatever that test was measuring.
_PAGE_OPENINGS = (
    "Maryam waited by the window.",
    "Softly, the wind turned over.",
    "Up above, something glittered.",
    "Pip pressed his nose to the glass.",
    "Down came a scattering of light.",
    "Away it went, over the fence.",
    "In the garden nothing moved at all.",
    "Nobody spoke for a long moment.",
    "Then the whole sky leaned closer.",
    "So they said goodnight to the moon.",
)


@pytest.fixture
def prose(page_plan: PagePlan) -> StoryProse:
    """Prose matching the `page_plan` fixture page for page."""
    return StoryProse(
        pages=[
            StoryPage(
                page_number=page.page_number,
                text=_PAGE_OPENINGS[page.page_number - 1],
            )
            for page in page_plan.pages
        ]
    )


@pytest.fixture
def story(outline: StoryOutline, page_plan: PagePlan, prose: StoryProse) -> Story:
    """A finished ten-page story, assembled from the fixtures above.

    Composed rather than hand-written so it cannot drift from the plan and
    prose fixtures the rest of the suite uses.
    """
    return Story(outline=outline, page_plan=page_plan, pages=prose.pages)


@pytest.fixture
def book_factory() -> Callable[..., Story]:
    """Builds a ``Story`` from page texts, for tests that measure one.

    A factory rather than a fixture because the eval metrics are *about* the exact
    words on the page, so each test needs its own book. The ``story`` fixture
    above is the opposite: one fixed book shared by tests that only need a valid
    one.
    """

    def build(page_texts: list[str], beat_summaries: list[str] | None = None) -> Story:
        summaries = beat_summaries or [
            f"Something happens in beat {i + 1}, at some length." for i in range(4)
        ]
        built_outline = StoryOutline(
            title="A Title",
            logline="One sentence that captures the whole story here.",
            theme="a theme worth exploring",
            characters=[
                CharacterSketch(
                    name="Kit",
                    role="main character",
                    description="A child who wonders.",
                )
            ],
            beats=[
                StoryBeat(
                    position=i + 1,
                    function=NarrativeFunction.SETUP,
                    title=f"Beat {i + 1}",
                    summary=summary,
                )
                for i, summary in enumerate(summaries)
            ],
        )
        # Padded to `PagePlan`'s floor of 4 regardless of how many pages of prose
        # a test needs. Nothing being measured reads the plan -- `beats_per_page`
        # divides by the prose pages -- so the padding cannot affect a result.
        built_plan = PagePlan(
            pages=[
                ScenePlan(
                    page_number=i + 1,
                    beat_position=1,
                    setting="a garden",
                    visual_action="Kit looks up, eyes wide",
                    emotional_shift="curiosity",
                )
                for i in range(max(4, len(page_texts)))
            ]
        )
        return Story(
            outline=built_outline,
            page_plan=built_plan,
            pages=[
                StoryPage(page_number=i + 1, text=text)
                for i, text in enumerate(page_texts)
            ],
        )

    return build
