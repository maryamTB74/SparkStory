"""HTML rendering properties that matter, asserted as strings.

These are not snapshot tests. Each one pins a property with a reason: no guessed
pronouns, the whole outline shown, no player for a file that is not there.
"""

from pathlib import Path

import pytest

from sparkstory.entities.stories import (
    CharacterSketch,
    ChildProfile,
    NarrativeFunction,
    PagePlan,
    ScenePlan,
    Story,
    StoryBeat,
    StoryBrief,
    StoryOutline,
    StoryPage,
)
from sparkstory.mcp.ui.jobs import Job, JobState
from sparkstory.mcp.ui.pages import render_book, render_form, render_job


def test_the_form_offers_every_pronoun_option() -> None:
    html = render_form()

    assert 'value="she/her"' in html
    assert 'value="he/him"' in html
    assert 'value="they/them"' in html


def test_no_pronoun_option_is_preselected() -> None:
    # A default here is a guess wearing a UI. Q4 of try_prompt.py has failed a
    # live run on exactly this, and the server prompt tells clients never to
    # infer pronouns from a name.
    html = render_form()

    pronoun_block = html.split('aria-label="Pronouns"')[1].split("</div>")[0]

    assert "checked" not in pronoun_block


def test_the_pronoun_inputs_are_required() -> None:
    html = render_form()

    assert 'name="pronouns"' in html
    assert "required" in html


def test_the_form_posts_to_plan() -> None:
    html = render_form()

    assert "/plan" in html


def test_the_form_does_not_write_to_stdout(capsys) -> None:
    render_form()

    assert capsys.readouterr().out == ""


@pytest.fixture
def outline() -> StoryOutline:
    return StoryOutline(
        title="Maryam and the Paper Rocket",
        logline="Maryam builds a paper rocket to send her wish to the moon.",
        theme="turning a wish into something you can send into the sky",
        characters=[
            CharacterSketch(
                name="Maryam",
                role="protagonist",
                description="A five-year-old who builds things",
            ),
            CharacterSketch(
                name="Kit", role="friend", description="A fox with white paws"
            ),
        ],
        beats=[
            StoryBeat(
                position=1,
                function=NarrativeFunction.SETUP,
                title="The fox in the garden",
                summary="Maryam finds a fox sitting under the apple tree.",
            ),
            StoryBeat(
                position=2,
                function=NarrativeFunction.INCITING_INCIDENT,
                title="Kit points up",
                summary="Kit points at the moon and will not look away.",
            ),
            StoryBeat(
                position=3,
                function=NarrativeFunction.CLIMAX,
                title="Folding the rocket",
                summary="Maryam folds a paper rocket while Kit watches.",
            ),
            StoryBeat(
                position=4,
                function=NarrativeFunction.RESOLUTION,
                title="The rocket flies",
                summary="They send the paper rocket up into the sky together.",
            ),
        ],
    )


@pytest.fixture
def brief() -> StoryBrief:
    # page_count 4, not 2: `StoryOutline.beats` requires 4-8, and a beat needs a
    # page of its own, so `len(beats) <= page_count`. A 2-page brief with a valid
    # outline is not constructible.
    return StoryBrief(
        child=ChildProfile(name="Maryam", age=5),
        premise="a fox who wants to visit the moon",
        page_count=4,
    )


@pytest.fixture
def awaiting(brief: StoryBrief, outline: StoryOutline) -> Job:
    return Job(
        id="job-1",
        state=JobState.AWAITING_APPROVAL,
        brief=brief,
        original_premise=brief.premise,
        outline=outline,
    )


def test_the_outline_page_renders_every_beat(awaiting: Job) -> None:
    # Q2 as a test. An LLM client may summarise a plan; HTML must not.
    html = render_job(awaiting)

    assert "Maryam finds a fox sitting under the apple tree." in html
    assert "Kit points at the moon and will not look away." in html
    assert "Maryam folds a paper rocket while Kit watches." in html
    assert "They send the paper rocket up into the sky together." in html


def test_the_outline_page_renders_title_theme_and_characters(awaiting: Job) -> None:
    html = render_job(awaiting)

    assert "Maryam and the Paper Rocket" in html
    assert "turning a wish into something you can send into the sky" in html
    assert "A fox with white paws" in html


def test_the_outline_page_offers_approve_and_revise(awaiting: Job) -> None:
    html = render_job(awaiting)

    assert "/approve" in html
    assert "/revise" in html


def test_a_planning_job_shows_progress_not_an_outline(brief: StoryBrief) -> None:
    job = Job(
        id="job-2",
        state=JobState.PLANNING,
        brief=brief,
        original_premise=brief.premise,
        detail="critiquing the outline",
    )

    html = render_job(job)

    assert "critiquing the outline" in html
    assert "/approve" not in html


def test_a_failed_job_shows_its_error(brief: StoryBrief) -> None:
    job = Job(
        id="job-3",
        state=JobState.FAILED,
        brief=brief,
        original_premise=brief.premise,
        error="GOOGLE_API_KEY is not set",
    )

    html = render_job(job)

    assert "GOOGLE_API_KEY is not set" in html


def test_rendered_values_are_escaped() -> None:
    # A premise is free text typed by a person; prose is written by a model.
    brief = StoryBrief(
        child=ChildProfile(name="Maryam", age=5),
        premise="<script>alert('x')</script>",
    )
    job = Job(
        id="job-4",
        state=JobState.FAILED,
        brief=brief,
        original_premise=brief.premise,
        error="<script>alert('boom')</script>",
    )

    html = render_job(job)

    assert "<script>alert('boom')" not in html
    assert "&lt;script&gt;" in html


def _page_plan() -> PagePlan:
    # PagePlan.pages requires 4-24 entries, matching the 4-page brief.
    return PagePlan(
        pages=[
            ScenePlan(
                page_number=number,
                beat_position=number,
                setting="A garden under an apple tree",
                visual_action="Maryam crouches beside a fox",
                emotional_shift="curiosity",
            )
            for number in range(1, 5)
        ]
    )


def _story(outline: StoryOutline) -> Story:
    return Story(
        outline=outline,
        page_plan=_page_plan(),
        pages=[
            StoryPage(page_number=1, text="Maryam found a fox."),
            StoryPage(page_number=2, text="The fox looked up."),
            StoryPage(page_number=3, text="She folded the paper."),
            StoryPage(page_number=4, text="The rocket flew."),
        ],
    )


def test_the_book_page_renders_every_page(awaiting: Job, outline: StoryOutline) -> None:
    job = Job(
        id="job-5",
        state=JobState.COMPLETE,
        brief=awaiting.brief,
        original_premise=awaiting.original_premise,
        outline=outline,
        story=_story(outline),
    )

    html = render_book(
        job, media={"pages": [], "story_audio": None, "video": None, "pdf": None}
    )

    assert "Maryam found a fox." in html
    assert "The fox looked up." in html
    assert "She folded the paper." in html
    assert "The rocket flew." in html


def test_the_book_page_renders_media_that_exists(
    awaiting: Job, outline: StoryOutline
) -> None:
    job = Job(
        id="job-6",
        state=JobState.COMPLETE,
        brief=awaiting.brief,
        original_premise=awaiting.original_premise,
        outline=outline,
        story=_story(outline),
        run_directory=Path("/tmp/run"),
    )
    media = {
        "pages": [
            {"number": 1, "image": "page-01.jpg", "audio": "page-01.mp3"},
            {"number": 2, "image": None, "audio": None},
            {"number": 3, "image": None, "audio": None},
            {"number": 4, "image": None, "audio": None},
        ],
        "story_audio": "story.mp3",
        "video": None,
        "pdf": "story.pdf",
    }

    html = render_book(job, media)

    assert "/job/job-6/file/page-01.jpg" in html
    assert "/job/job-6/file/story.mp3" in html
    assert "/job/job-6/file/story.pdf" in html


def test_the_book_page_renders_no_player_for_absent_media(
    awaiting: Job, outline: StoryOutline
) -> None:
    # A player for a file that does not exist is a lie.
    job = Job(
        id="job-7",
        state=JobState.COMPLETE,
        brief=awaiting.brief,
        original_premise=awaiting.original_premise,
        outline=outline,
        story=_story(outline),
    )
    media = {
        "pages": [
            {"number": number, "image": None, "audio": None} for number in range(1, 5)
        ],
        "story_audio": None,
        "video": None,
        "pdf": None,
    }

    html = render_book(job, media)

    assert "<audio" not in html
    assert "<video" not in html
    assert "<img" not in html
