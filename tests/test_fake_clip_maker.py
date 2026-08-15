"""The fake records how it was asked, and fails where it is told to."""

import pytest

from sparkstory.entities.exceptions import VideoGenerationError
from sparkstory.models.fake_clip_maker import TINY_MP4, FakeClipMaker


async def test_it_records_whether_there_was_a_picture() -> None:
    """A held card and an animated page are different outcomes, so the fake must
    let a test tell them apart by what was *asked for*."""
    maker = FakeClipMaker()

    await maker.make_clip(b"jpeg-bytes", 2.0, 1)
    await maker.make_clip(None, 3.0, 2)

    assert maker.calls == [(True, 2.0, 1), (False, 3.0, 2)]


async def test_it_fails_on_the_pages_it_is_told_to() -> None:
    """Matching on page number rather than call index, because pages are
    processed concurrently and call order is not the workflow's contract."""
    maker = FakeClipMaker(fail_on=(3,))

    with pytest.raises(VideoGenerationError):
        await maker.make_clip(b"jpeg-bytes", 1.0, 3)

    clip = await maker.make_clip(b"jpeg-bytes", 1.0, 4)
    assert clip.video_format == "mp4"


async def test_a_failed_call_is_still_recorded() -> None:
    """So a test can assert every page was attempted, not merely that some
    succeeded -- the same property ``FakeSpeechProvider.calls`` provides."""
    maker = FakeClipMaker(fail_on=(1,))

    with pytest.raises(VideoGenerationError):
        await maker.make_clip(b"jpeg-bytes", 1.0, 1)

    assert maker.calls == [(True, 1.0, 1)]


def test_the_canned_bytes_are_a_real_mp4_container() -> None:
    """These bytes came from the real encoder, not from a keyboard.

    A structural check rather than a length one -- ``tests/test_video_live.py``
    is what pins them against freshly generated output. This is the cheap guard
    that the base64 in the module has not been truncated or mangled.
    """
    assert TINY_MP4[4:8] == b"ftyp", "not an MP4 container"
    assert b"avc1" in TINY_MP4[:200], "not H.264"
    assert len(TINY_MP4) > 1000, "too small to carry a real encoded frame"
