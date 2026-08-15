"""Page selection, fail-soft and the record -- all with no ffmpeg.

Two module-level seams are patched: ``build_clip_maker``, so no encoder runs, and
``read_duration``, so no ffprobe runs. What is left is the workflow's own logic,
which is the part worth testing here -- the ffmpeg-dependent checks are
``video``-marked and live in ``test_video_live.py``.
"""

from collections.abc import Callable
from pathlib import Path

import pytest

from sparkstory.entities.video import VideoStatus
from sparkstory.models.fake_clip_maker import FakeClipMaker
from sparkstory.workflows import animate as animate_module

#: What the patched duration reader returns for every page. Any fixed number
#: works: nothing here asserts on the value, only that it reached the maker.
_DURATION = 2.0


@pytest.fixture(autouse=True)
def _no_ffprobe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Measure nothing; every page's audio is `_DURATION` long."""

    async def fixed(_path: Path) -> float:
        return _DURATION

    monkeypatch.setattr(animate_module, "read_duration", fixed)


@pytest.fixture(autouse=True)
def _no_assembly(monkeypatch: pytest.MonkeyPatch) -> None:
    """Write a stub file instead of running ffmpeg's concat + mux.

    Patched rather than faked through the seam because assembly is not behind
    one: it is ours, not a provider's, and its argument construction is tested
    directly in ``test_video_assemble.py``.
    """

    async def stub(clips: list[Path], audios: list[Path], destination: Path) -> Path:
        assert len(clips) == len(audios), "selection disagreed with itself"
        destination.write_bytes(b"assembled")
        return destination

    monkeypatch.setattr(animate_module, "assemble", stub)


@pytest.fixture
def fake_maker(monkeypatch: pytest.MonkeyPatch) -> FakeClipMaker:
    """Patch the module-level factory -- the seam ``narrate.py`` uses for the
    same job, and the reason a model cannot travel in a workflow payload."""
    maker = FakeClipMaker()
    monkeypatch.setattr(animate_module, "build_clip_maker", maker.as_maker)
    return maker


async def test_a_page_with_no_audio_is_excluded(
    tmp_path: Path,
    fake_maker: FakeClipMaker,
    video_fixtures: Callable[..., tuple],
) -> None:
    """Audio is the spine: a page with no narration has no duration, and
    inventing one is exactly what the rule refuses."""
    story, art, narration = video_fixtures(pages=3, missing_audio={2})

    video = await animate_module.run_video_pipeline(story, art, narration, tmp_path)

    excluded = [i for i in video.items if i.status is VideoStatus.EXCLUDED]
    assert [i.page_number for i in excluded] == [2]
    assert excluded[0].duration is None, "an excluded page must not carry a length"
    assert excluded[0].reason is not None, "a missing page must say why"
    assert video.is_complete is False
    # The maker was never asked about page 2 at all.
    assert [page for _had, _d, page in fake_maker.calls] == [1, 3]


async def test_a_page_with_no_picture_is_held_not_dropped(
    tmp_path: Path,
    fake_maker: FakeClipMaker,
    video_fixtures: Callable[..., tuple],
) -> None:
    """It has narration, so it belongs on the timeline for its full length."""
    story, art, narration = video_fixtures(pages=3, missing_image={2})

    video = await animate_module.run_video_pipeline(story, art, narration, tmp_path)

    held = [i for i in video.items if i.status is VideoStatus.HELD]
    assert [i.page_number for i in held] == [2]
    assert held[0].duration == _DURATION, "a held page still has its own length"
    # Asked for, with no image and a real duration.
    assert (False, _DURATION, 2) in fake_maker.calls
    # And it is still a complete video: every page reached the timeline.
    assert video.is_complete is True
    assert video.pages_animated == 2, "a held page is not an animated one"


async def test_an_all_excluded_run_writes_no_file(
    tmp_path: Path,
    fake_maker: FakeClipMaker,
    video_fixtures: Callable[..., tuple],
) -> None:
    """An empty video plays as nothing, and nothing is indistinguishable from
    success on a casual glance. Narration's precedent, same argument."""
    story, art, narration = video_fixtures(pages=3, missing_audio={1, 2, 3})

    video = await animate_module.run_video_pipeline(story, art, narration, tmp_path)

    assert video.path is None
    assert not (tmp_path / "story.mp4").exists()
    assert video.is_complete is False
    assert fake_maker.calls == [], "nothing should have been encoded"


async def test_a_failed_clip_does_not_destroy_the_video(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    video_fixtures: Callable[..., tuple],
) -> None:
    """Fail soft per page, like illustration and narration."""
    maker = FakeClipMaker(fail_on=(2,))
    monkeypatch.setattr(animate_module, "build_clip_maker", maker.as_maker)
    story, art, narration = video_fixtures(pages=3)

    video = await animate_module.run_video_pipeline(story, art, narration, tmp_path)

    failed = [i for i in video.items if i.status is VideoStatus.FAILED]
    assert [i.page_number for i in failed] == [2]
    assert failed[0].duration == _DURATION, "it had audio, so it has a length"
    assert failed[0].reason is not None
    assert video.pages_animated == 2
    assert video.path is not None, "the rest of the book still assembles"


async def test_items_come_back_in_page_order(
    tmp_path: Path,
    fake_maker: FakeClipMaker,
    video_fixtures: Callable[..., tuple],
) -> None:
    """Pages are encoded concurrently and the record is read by a human."""
    story, art, narration = video_fixtures(pages=4, missing_audio={2})

    video = await animate_module.run_video_pipeline(story, art, narration, tmp_path)

    assert [i.page_number for i in video.items] == [1, 2, 3, 4]


async def test_the_record_is_written_beside_the_video(
    tmp_path: Path,
    fake_maker: FakeClipMaker,
    video_fixtures: Callable[..., tuple],
) -> None:
    """So a run answers "which pages made it?" by reading a file rather than by
    re-deriving it -- the same reason narration writes ``narration.json``."""
    story, art, narration = video_fixtures(pages=2)

    await animate_module.run_video_pipeline(story, art, narration, tmp_path)

    record = (tmp_path / "video.json").read_text()
    assert '"animated"' in record
    assert '"fps": 30' in record


async def test_a_vanished_audio_file_excludes_its_page(
    tmp_path: Path,
    fake_maker: FakeClipMaker,
    video_fixtures: Callable[..., tuple],
) -> None:
    """The artifact says the page has audio and the file is gone.

    Not the same case as a FAILED narration item: here ``narration.json`` claims
    success, so only checking the file catches it. ``outputs/`` is disposable and
    has lost books before, which makes this the realistic version of the missing
    page rather than an invented one.
    """
    story, art, narration = video_fixtures(pages=3)
    vanished = narration.page_audio(2)
    assert vanished is not None
    vanished.unlink()

    video = await animate_module.run_video_pipeline(story, art, narration, tmp_path)

    excluded = [i for i in video.items if i.status is VideoStatus.EXCLUDED]
    assert [i.page_number for i in excluded] == [2]
