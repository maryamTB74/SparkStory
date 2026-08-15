"""The assembly's argument construction, checked without running ffmpeg."""

from pathlib import Path

import pytest

from sparkstory.video.assemble import build_concat_file


def test_the_concat_list_quotes_every_path(tmp_path: Path) -> None:
    """ffmpeg's concat demuxer splits on whitespace, so an unquoted path with a
    space in it silently truncates. Run directories are named after the premise,
    which is exactly where a space comes from."""
    listing = build_concat_file([tmp_path / "a b.mp4", tmp_path / "c.mp4"])

    assert f"file '{tmp_path / 'a b.mp4'}'" in listing
    assert f"file '{tmp_path / 'c.mp4'}'" in listing


def test_a_single_quote_in_a_path_is_escaped(tmp_path: Path) -> None:
    """A premise like "a child's garden" is an ordinary brief and would end the
    quoted string early, so the demuxer would read a truncated path."""
    listing = build_concat_file([tmp_path / "a child's garden.mp4"])

    assert r"'\''" in listing, "the demuxer's own escape form is missing"


def test_the_order_is_the_order_given(tmp_path: Path) -> None:
    """The concat list *is* the page order; nothing downstream re-sorts it."""
    listing = build_concat_file(
        [tmp_path / "clip-01.mp4", tmp_path / "clip-02.mp4", tmp_path / "clip-03.mp4"]
    )

    assert (
        listing.index("clip-01") < listing.index("clip-02") < listing.index("clip-03")
    )


def test_relative_paths_are_made_absolute(tmp_path: Path, monkeypatch) -> None:
    """The demuxer resolves a relative path against *the list file's* directory,
    not the process cwd -- so a repo-relative path from ``narration.json`` is
    looked for inside the run directory and is not there.

    Found by running it: assembly failed with "Impossible to open
    outputs/<run>/page-01.mp3" while that file plainly existed.
    """
    monkeypatch.chdir(tmp_path)
    listing = build_concat_file([Path("outputs/run/page-01.mp3")])

    assert str(tmp_path) in listing, "a relative path survived into the list"
    assert "file 'outputs/run" not in listing


def test_an_empty_list_is_refused() -> None:
    """Concatenating nothing produces a zero-byte file, and a zero-byte video
    plays as nothing -- indistinguishable from success on a casual glance."""
    with pytest.raises(ValueError):
        build_concat_file([])
