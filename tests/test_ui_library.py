"""The library: reaching books that already exist on disk.

Every already-paid-for illustration and recording on this machine belongs to a run
older than any live job, so without this the section 5.2 media rendering is
unreachable in normal use. See spec open question 3.
"""

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from sparkstory.entities.stories import (
    CharacterSketch,
    NarrativeFunction,
    PagePlan,
    ScenePlan,
    Story,
    StoryBeat,
    StoryOutline,
    StoryPage,
)
from sparkstory.mcp.ui import handlers
from sparkstory.mcp.ui.library import list_books, load_book


def _story(title: str = "Maryam and the Paper Rocket") -> Story:
    outline = StoryOutline(
        title=title,
        logline="Maryam builds a paper rocket to send her wish to the moon.",
        theme="a wish sent into the sky",
        characters=[
            CharacterSketch(
                name="Maryam", role="protagonist", description="Builds things"
            ),
            CharacterSketch(name="Kit", role="friend", description="A fox"),
        ],
        beats=[
            StoryBeat(
                position=n,
                function=fn,
                title=f"Beat {n}",
                summary=f"Something happens in beat number {n}.",
            )
            for n, fn in enumerate(
                [
                    NarrativeFunction.SETUP,
                    NarrativeFunction.INCITING_INCIDENT,
                    NarrativeFunction.CLIMAX,
                    NarrativeFunction.RESOLUTION,
                ],
                start=1,
            )
        ],
    )
    return Story(
        outline=outline,
        page_plan=PagePlan(
            pages=[
                ScenePlan(
                    page_number=n,
                    beat_position=n,
                    setting="A garden",
                    visual_action="A fox appears",
                    emotional_shift="surprise",
                )
                for n in range(1, 5)
            ]
        ),
        pages=[
            StoryPage(page_number=n, text=f"Page {n} of the story.")
            for n in range(1, 5)
        ],
    )


@pytest.fixture
def outputs(tmp_path: Path) -> Path:
    root = tmp_path / "outputs"
    root.mkdir()

    plain = root / "20260814-101500-a-fox-who-wants-to-visit-the-moon"
    plain.mkdir()
    (plain / "story.json").write_text(_story().model_dump_json())
    # A brief holds the child's name. It must never reach the listing.
    (plain / "brief.json").write_text('{"child": {"name": "Maryam"}}')

    rich = root / "20260815-093000-a-child-plants-a-seed"
    rich.mkdir()
    (rich / "story.json").write_text(_story("The Seed and the Sun").model_dump_json())
    (rich / "page-01.jpg").write_bytes(b"jpeg")
    (rich / "story.mp3").write_bytes(b"mp3")
    (rich / "story.mp4").write_bytes(b"mp4")
    (rich / "story.pdf").write_bytes(b"pdf")

    baseline = root / "eval-teddy-bear"
    baseline.mkdir()
    (baseline / "story.json").write_text(_story("Sam and Ted").model_dump_json())

    # A --stage plan run: no book, so it must not be listed.
    partial = root / "20260813-120000-an-unfinished-idea"
    partial.mkdir()
    (partial / "plan_outline-1.json").write_text("{}")

    return root


def test_list_books_returns_finished_books_newest_first(outputs: Path) -> None:
    books = list_books(outputs)

    assert [book["title"] for book in books] == [
        "The Seed and the Sun",
        "Maryam and the Paper Rocket",
        "Sam and Ted",
    ]


def test_list_books_skips_runs_with_no_story(outputs: Path) -> None:
    books = list_books(outputs)

    assert all("unfinished" not in book["run_id"] for book in books)


def test_list_books_reports_which_media_exists(outputs: Path) -> None:
    books = {book["title"]: book for book in list_books(outputs)}

    rich = books["The Seed and the Sun"]
    assert rich["has_images"] is True
    assert rich["has_audio"] is True
    assert rich["has_video"] is True
    assert rich["has_pdf"] is True

    plain = books["Maryam and the Paper Rocket"]
    assert plain["has_images"] is False
    assert plain["has_audio"] is False
    assert plain["has_video"] is False
    assert plain["has_pdf"] is False


def test_the_listing_never_exposes_the_premise_or_a_childs_name(
    outputs: Path,
) -> None:
    # A directory name contains the premise; brief.json contains the child. The
    # listing shows the story's own title and the timestamp instead.
    books = list_books(outputs)
    rendered = json.dumps(books)

    assert "a-fox-who-wants-to-visit-the-moon" not in rendered
    assert "a-child-plants-a-seed" not in rendered
    assert "wants to visit" not in rendered


def test_list_books_on_a_missing_directory_is_empty() -> None:
    assert list_books(Path("/nonexistent-outputs")) == []


def test_load_book_returns_the_story_and_its_directory(outputs: Path) -> None:
    loaded = load_book(outputs, "20260815-093000")

    assert loaded is not None
    story, directory = loaded
    assert story.outline.title == "The Seed and the Sun"
    assert directory.name == "20260815-093000-a-child-plants-a-seed"


def test_load_book_accepts_a_committed_baseline_directory(outputs: Path) -> None:
    loaded = load_book(outputs, "eval-teddy-bear")

    assert loaded is not None
    assert loaded[0].outline.title == "Sam and Ted"


@pytest.mark.parametrize(
    "run_id",
    [
        "../.env",
        "../../etc/passwd",
        "/etc/passwd",
        "..",
        ".",
        "not-a-run-id",
        "20260815-a-child-plants-a-seed",  # no time
        "20260815-093000-a-child-plants-a-seed",  # the directory name, not the id
        "99999999-999999",  # well-formed, no such run
        "",
    ],
)
def test_load_book_rejects_anything_that_is_not_a_run_id(
    outputs: Path, run_id: str
) -> None:
    assert load_book(outputs, run_id) is None


def test_load_book_returns_none_for_a_run_with_no_story(outputs: Path) -> None:
    assert load_book(outputs, "20260813-120000") is None


def _request(path_params: dict[str, str] | None = None) -> Any:
    class _Request:
        def __init__(self) -> None:
            self.path_params = path_params or {}

    return _Request()


def test_get_library_lists_books(
    monkeypatch: pytest.MonkeyPatch, outputs: Path
) -> None:
    monkeypatch.setattr(handlers, "_OUTPUTS", outputs)

    response = asyncio.run(handlers.get_library(_request()))
    html = response.body.decode()

    assert response.status_code == 200
    assert "The Seed and the Sun" in html
    assert "Maryam and the Paper Rocket" in html
    assert "a-fox-who-wants-to-visit-the-moon" not in html


def test_get_library_book_renders_media_that_exists(
    monkeypatch: pytest.MonkeyPatch, outputs: Path
) -> None:
    monkeypatch.setattr(handlers, "_OUTPUTS", outputs)

    response = asyncio.run(
        handlers.get_library_book(_request({"run_id": "20260815-093000"}))
    )
    html = response.body.decode()

    assert response.status_code == 200
    assert "<img" in html
    assert "<audio" in html
    assert "<video" in html
    assert "story.pdf" in html


def test_get_library_book_returns_404_for_a_bad_run_id(
    monkeypatch: pytest.MonkeyPatch, outputs: Path
) -> None:
    monkeypatch.setattr(handlers, "_OUTPUTS", outputs)

    response = asyncio.run(handlers.get_library_book(_request({"run_id": "../.env"})))

    assert response.status_code == 404


def test_get_library_file_serves_an_artifact(
    monkeypatch: pytest.MonkeyPatch, outputs: Path
) -> None:
    monkeypatch.setattr(handlers, "_OUTPUTS", outputs)

    response = asyncio.run(
        handlers.get_library_file(
            _request(
                {
                    "run_id": "20260815-093000",
                    "name": "page-01.jpg",
                }
            )
        )
    )

    assert response.status_code == 200
    assert response.media_type == "image/jpeg"
    assert response.body == b"jpeg"


def test_get_library_file_refuses_brief_json(
    monkeypatch: pytest.MonkeyPatch, outputs: Path
) -> None:
    monkeypatch.setattr(handlers, "_OUTPUTS", outputs)

    response = asyncio.run(
        handlers.get_library_file(
            _request(
                {
                    "run_id": "20260814-101500",
                    "name": "brief.json",
                }
            )
        )
    )

    assert response.status_code == 404


def test_get_library_file_refuses_traversal_in_both_segments(
    monkeypatch: pytest.MonkeyPatch, outputs: Path
) -> None:
    monkeypatch.setattr(handlers, "_OUTPUTS", outputs)

    bad_run = asyncio.run(
        handlers.get_library_file(_request({"run_id": "../..", "name": "page-01.jpg"}))
    )
    bad_name = asyncio.run(
        handlers.get_library_file(
            _request(
                {
                    "run_id": "20260815-093000",
                    "name": "../brief.json",
                }
            )
        )
    )

    assert bad_run.status_code == 404
    assert bad_name.status_code == 404
