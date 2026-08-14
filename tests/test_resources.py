"""The two read-only resources, and what they must never say.

A resource's text is read by a *client's model*, so non-obvious rule 1 applies to
everything these return: it is prompt material, not a debug dump.

Every test here builds its own run directory under ``tmp_path``. Pointing them at
the real ``outputs/`` would make them depend on whatever was generated last --
which is disposable by design and has already lost eight books.
"""

import json
from pathlib import Path

from fastmcp import Client

from sparkstory.mcp.resources.library import read_corpus, read_library
from sparkstory.mcp.server import create_server


def _make_run(
    root: Path,
    name: str = "20260811-231648-a-child-who-plants-a-seed",
    *,
    title: str = "Maryam and the Seed",
    pages: int = 8,
    child_name: str = "Maryam",
    pdf: bool = False,
    audio: bool = False,
    story: bool = True,
) -> Path:
    """Build a run directory shaped like a real one, child's name included."""
    run = root / name
    run.mkdir(parents=True)

    (run / "brief.json").write_text(
        json.dumps({"child": {"name": child_name, "age": 5}, "premise": "a seed"})
    )
    (run / "meta.json").write_text(
        json.dumps({"child_id": f"{child_name.lower()}-5", "pages": pages})
    )
    if story:
        (run / "story.json").write_text(
            json.dumps(
                {
                    "outline": {"title": title},
                    "pages": [{"page_number": i} for i in range(1, pages + 1)],
                }
            )
        )
    if pdf:
        (run / "story.pdf").write_bytes(b"%PDF-1.4")
    if audio:
        (run / "story.mp3").write_bytes(b"\xff\xf3")
    return run


class TestLibraryDoesNotLeak:
    """The hazard this resource is built around.

    Run directories on disk are named after the premise
    (``20260811-231338-a-fox-who-wants-to-visit-the-moon``), every ``brief.json``
    holds a child's name, and every ``meta.json`` holds a ``child_id``. None of
    that may reach a client's model, which is the same hazard Session 12 recorded
    one door over when Opik uploaded briefs containing a child's name.
    """

    def test_a_childs_name_does_not_reach_the_output(self, tmp_path: Path) -> None:
        _make_run(tmp_path, child_name="Sunniva")

        assert "Sunniva" not in read_library(tmp_path)

    def test_a_child_id_does_not_reach_the_output(self, tmp_path: Path) -> None:
        _make_run(tmp_path, child_name="Sunniva")

        assert "sunniva-5" not in read_library(tmp_path)

    def test_the_premise_in_a_directory_name_does_not_reach_the_output(
        self, tmp_path: Path
    ) -> None:
        # The run id is reported, but a *slug of the premise* is not the run id.
        # Reporting the raw directory name would smuggle the premise out.
        _make_run(tmp_path, name="20260811-231338-a-fox-who-wants-to-visit-the-moon")

        assert "a-fox-who-wants-to-visit-the-moon" not in read_library(tmp_path)


class TestLibraryResource:
    """What a finished book looks like from outside."""

    def test_it_reports_a_finished_book(self, tmp_path: Path) -> None:
        _make_run(tmp_path, title="Maryam and the Seed", pages=8)

        text = read_library(tmp_path)

        assert "Maryam and the Seed" in text
        assert "8" in text

    def test_it_reports_whether_a_pdf_and_audio_exist(self, tmp_path: Path) -> None:
        _make_run(tmp_path, name="20260101-000000-a", pdf=True, audio=False)

        text = read_library(tmp_path)

        assert "pdf" in text.lower()
        assert "20260101-000000" in text

    def test_an_empty_outputs_directory_is_not_an_error(self, tmp_path: Path) -> None:
        # A server that has made nothing is a valid server. Raising here would
        # make a fresh checkout look broken.
        assert read_library(tmp_path) is not None

    def test_a_missing_outputs_directory_is_not_an_error(self, tmp_path: Path) -> None:
        assert read_library(tmp_path / "nope") is not None

    def test_a_run_with_no_story_is_skipped(self, tmp_path: Path) -> None:
        # `--stage plan` runs exist on disk and have no book. Skipping beats
        # raising: one incomplete directory must not break the whole listing.
        _make_run(tmp_path, name="20260101-000000-planned", story=False)
        _make_run(tmp_path, name="20260102-000000-written", title="A Real Book")

        text = read_library(tmp_path)

        assert "A Real Book" in text
        assert "20260101-000000-planned" not in text

    def test_a_directory_that_is_not_timestamped_is_not_truncated(
        self, tmp_path: Path
    ) -> None:
        # Found by reading a live stdio run rather than by a test: the committed
        # baseline directories are named `eval-eagle-planet`, and blind
        # "keep the first two hyphen-separated parts" turned that into
        # `eval-eagle`. Truncating an arbitrary name is worse than useless --
        # it can emit *half a premise* while looking like an id.
        _make_run(tmp_path, name="eval-eagle-planet", title="A Real Book")

        assert "eval-eagle-planet" in read_library(tmp_path)

    def test_a_timestamped_directory_still_loses_its_premise(
        self, tmp_path: Path
    ) -> None:
        _make_run(tmp_path, name="20260811-231338-a-fox-who-wants-the-moon")

        text = read_library(tmp_path)

        assert "20260811-231338" in text
        assert "fox" not in text.lower()

    def test_unreadable_json_does_not_break_the_listing(self, tmp_path: Path) -> None:
        broken = tmp_path / "20260101-000000-broken"
        broken.mkdir()
        (broken / "story.json").write_text("{not json")
        _make_run(tmp_path, name="20260102-000000-fine", title="A Real Book")

        assert "A Real Book" in read_library(tmp_path)


class TestCorpusResource:
    """Retrieval corpus stats, which finding 27 says to read before a comparison."""

    def test_it_reports_a_chunk_count(self) -> None:
        assert "chunk" in read_corpus().lower()

    def test_it_names_the_embedding_model(self) -> None:
        from sparkstory.config import settings

        assert settings.embedding_model in read_corpus()


class TestResourcesAreRegistered:
    """The server actually advertises them."""

    async def test_both_resources_are_exposed(self) -> None:
        async with Client(create_server()) as client:
            uris = {str(r.uri) for r in await client.list_resources()}

        assert uris >= {"sparkstory://library", "sparkstory://corpus"}

    async def test_a_resource_can_be_read_over_mcp(self) -> None:
        async with Client(create_server()) as client:
            contents = await client.read_resource("sparkstory://corpus")

        assert contents[0].text
