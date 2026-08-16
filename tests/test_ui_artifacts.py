"""Turning a URL segment into a file on disk, safely.

The precedent for testing this properly: `_run_id` truncating an arbitrary
directory name could emit half a premise, and three offline tests missed it
because every fixture was well-formed.
"""

from pathlib import Path

import pytest

from sparkstory.mcp.ui.artifacts import (
    MEDIA_TYPES,
    available_media,
    resolve_artifact,
)


@pytest.fixture
def run_directory(tmp_path: Path) -> Path:
    run = tmp_path / "20260814-101500-a-fox-who-wants-to-visit-the-moon"
    run.mkdir()
    (run / "page-01.jpg").write_bytes(b"jpeg")
    (run / "page-02.jpg").write_bytes(b"jpeg")
    (run / "page-01.mp3").write_bytes(b"mp3")
    (run / "story.mp3").write_bytes(b"mp3")
    (run / "story.pdf").write_bytes(b"pdf")
    (run / "portrait-maryam.jpg").write_bytes(b"jpeg")
    (run / "brief.json").write_text("{}")
    return run


@pytest.mark.parametrize(
    "name",
    [
        "page-01.jpg",
        "page-02.jpg",
        "page-01.mp3",
        "story.mp3",
        "story.pdf",
        "portrait-maryam.jpg",
    ],
)
def test_allowlisted_names_resolve(run_directory: Path, name: str) -> None:
    assert resolve_artifact(run_directory, name) == run_directory / name


@pytest.mark.parametrize(
    "name",
    [
        "brief.json",  # exists, but is not media -- holds a child's name
        "meta.json",
        "story.json",
        "run.log",
        "page-1.jpg",  # one digit; the pattern wants two
        "page-001.jpg",
        "story.exe",
        "PAGE-01.JPG",  # case matters; the writers emit lowercase
        "",
    ],
)
def test_names_outside_the_allowlist_are_rejected(
    run_directory: Path, name: str
) -> None:
    assert resolve_artifact(run_directory, name) is None


@pytest.mark.parametrize(
    "name",
    [
        "../brief.json",
        "../../.env",
        "../../../etc/passwd",
        "..%2f..%2f.env",
        "/etc/passwd",
        "subdir/page-01.jpg",
        "page-01.jpg/../../.env",
        "./page-01.jpg",
    ],
)
def test_traversal_attempts_are_rejected(run_directory: Path, name: str) -> None:
    # Rejected by the allowlist pattern rather than by stripping `../`: a type
    # that cannot express the attack beats code that remembers to check.
    assert resolve_artifact(run_directory, name) is None


def test_a_missing_but_allowlisted_file_resolves_to_none(run_directory: Path) -> None:
    assert resolve_artifact(run_directory, "story.mp4") is None


def test_a_symlink_escaping_the_run_directory_is_rejected(
    run_directory: Path, tmp_path: Path
) -> None:
    # The second guard. The name is allowlisted and the file exists, but it
    # resolves outside the run directory.
    secret = tmp_path / "secret.jpg"
    secret.write_bytes(b"not yours")
    (run_directory / "page-09.jpg").symlink_to(secret)

    assert resolve_artifact(run_directory, "page-09.jpg") is None


def test_every_allowlisted_extension_has_a_media_type() -> None:
    # An .mp3 served as octet-stream downloads instead of playing.
    assert MEDIA_TYPES[".jpg"] == "image/jpeg"
    assert MEDIA_TYPES[".mp3"] == "audio/mpeg"
    assert MEDIA_TYPES[".mp4"] == "video/mp4"
    assert MEDIA_TYPES[".pdf"] == "application/pdf"


def test_available_media_reports_only_what_exists(run_directory: Path) -> None:
    media = available_media(run_directory, page_count=2)

    assert media["pages"] == [
        {"number": 1, "image": "page-01.jpg", "audio": "page-01.mp3"},
        {"number": 2, "image": "page-02.jpg", "audio": None},
    ]
    assert media["story_audio"] == "story.mp3"
    assert media["pdf"] == "story.pdf"
    assert media["video"] is None


def test_available_media_on_a_run_with_no_media(tmp_path: Path) -> None:
    bare = tmp_path / "20260814-101500-bare"
    bare.mkdir()

    media = available_media(bare, page_count=2)

    assert media["pages"] == [
        {"number": 1, "image": None, "audio": None},
        {"number": 2, "image": None, "audio": None},
    ]
    assert media["story_audio"] is None
    assert media["pdf"] is None
    assert media["video"] is None


def test_available_media_with_no_run_directory() -> None:
    media = available_media(None, page_count=3)

    assert media["pages"] == [
        {"number": 1, "image": None, "audio": None},
        {"number": 2, "image": None, "audio": None},
        {"number": 3, "image": None, "audio": None},
    ]
    assert media["story_audio"] is None
