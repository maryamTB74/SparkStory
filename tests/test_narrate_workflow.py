"""Narrating a finished book: fan-out, fail-soft, and a stitched whole."""

import hashlib
from collections.abc import Callable
from pathlib import Path

import pytest
from fastmcp.exceptions import ToolError

from sparkstory.entities.exceptions import AudioConfigurationError
from sparkstory.entities.narration import NarrationStatus
from sparkstory.entities.stories import ReadingLevel, Story, StoryBrief, Voice
from sparkstory.mcp.tools.narrate_story import narrate_story_tool
from sparkstory.models.fake_speech_model import MP3_SILENCE, FakeSpeechProvider
from sparkstory.workflows import narrate as narrate_module
from sparkstory.workflows.narrate import (
    _VOICES,
    _speed_for,
    run_narration_pipeline,
    stitch,
)


@pytest.fixture
def speak() -> Callable[..., FakeSpeechProvider]:
    """Install a fake provider and hand it back for assertions.

    Patches `build_speech_model` rather than passing a model in, which is the seam
    shape `plan_outline.py` uses for research: a model holding a closure over an
    API key cannot travel in a workflow payload, because a payload has to survive
    a checkpointer.
    """

    def install(
        monkeypatch: pytest.MonkeyPatch, **kwargs: object
    ) -> FakeSpeechProvider:
        provider = FakeSpeechProvider(**kwargs)  # type: ignore[arg-type]
        monkeypatch.setattr(narrate_module, "build_speech_model", provider.as_model)
        return provider

    return install


def test_every_voice_maps_to_a_verified_provider_id() -> None:
    # All four of these returned 200 from the live endpoint on 2026-08-12. An
    # unknown id is a 404 at generation time -- i.e. after the book is written.
    verified = {"eve", "leo", "orion", "atlas"}
    assert set(_VOICES) == set(Voice)
    assert set(_VOICES.values()) <= verified


def test_speed_covers_every_reading_level_and_stays_in_range() -> None:
    # An unmapped level would KeyError at generation time, after the book is
    # written -- so completeness is asserted rather than assumed.
    speeds = {level: _speed_for(level) for level in ReadingLevel}
    assert len(speeds) == len(ReadingLevel)
    assert all(0.7 <= s <= 1.5 for s in speeds.values())


def test_speed_is_slower_for_younger_readers() -> None:
    # Ordering, not exact values: the point is that a pre-reader being read to is
    # paced more slowly than a confident reader following along.
    assert _speed_for(ReadingLevel.PRE_READER) < _speed_for(ReadingLevel.CONFIDENT)
    assert _speed_for(ReadingLevel.EARLY_READER) < 1.0


#: One **correctly sized** MPEG-2 Layer III frame, matching what the live provider
#: returns: `ff f3` sync, MPEG-2, Layer III, 128 kbps at 24 kHz.
#:
#: The length is not arbitrary. A frame's declared size is
#: `576 / 8 * bitrate / sample_rate` = `576 / 8 * 128000 / 24000` = **384 bytes**,
#: so the payload is 380 bytes after the 4-byte header. That matters because
#: `test_a_stitched_file_walks_end_to_end_as_contiguous_frames` walks *declared*
#: lengths -- a fixture whose real length disagreed with its header would fail the
#: test for the wrong reason, which is its own version of the bug being guarded.
_MPEG2_FRAME_BYTES = 384
_MPEG2_FRAME = b"\xff\xf3\xc4\xc4" + b"\x00" * (_MPEG2_FRAME_BYTES - 4)


def test_stitch_concatenates_with_no_separator() -> None:
    # No gap, and that is measured rather than lazy: two attempts at an
    # inter-page pause both corrupted the stream (9.2% and 8.6% of bytes
    # walkable), while plain concatenation walks 100%. See `_SILENCE_FRAMES`.
    assert stitch([_MPEG2_FRAME, _MPEG2_FRAME]) == _MPEG2_FRAME * 2


def test_stitch_of_one_part_is_that_part() -> None:
    assert stitch([_MPEG2_FRAME]) == _MPEG2_FRAME


def test_stitch_of_nothing_is_empty() -> None:
    assert stitch([]) == b""


def test_a_stitched_file_walks_end_to_end_as_contiguous_frames() -> None:
    """The assertion both broken stitchers would have failed.

    Offline tests asserted sync bits and layer, which a plausible-but-wrong
    constant satisfies -- that is exactly how an MPEG-1 gap survived into MPEG-2
    audio. Walking declared frame *lengths* is what catches it: a fabricated frame
    whose length does not match its payload leaves the parser mid-stream.
    """
    joined = stitch([_MPEG2_FRAME] * 4)

    offset = frames = 0
    while offset < len(joined) - 4:
        if joined[offset] != 0xFF or (joined[offset + 1] & 0xE0) != 0xE0:
            break
        if (joined[offset + 1] >> 1) & 0x03 != 0x01:  # Layer III
            break
        bitrate_index = (joined[offset + 2] >> 4) & 0x0F
        sample_index = (joined[offset + 2] >> 2) & 0x03
        if bitrate_index in (0, 15) or sample_index == 3:
            break
        # MPEG-2 Layer III: 576 samples per frame, its own bitrate table.
        bitrate = [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0][
            bitrate_index
        ]
        rate = [22050, 24000, 16000, 0][sample_index]
        size = int(576 / 8 * bitrate * 1000 / rate) + ((joined[offset + 2] >> 1) & 1)
        offset += max(size, 1)
        frames += 1

    assert frames == 4, f"expected 4 frames, walked {frames}"
    assert offset == len(joined), (
        f"walked {offset} of {len(joined)} bytes -- the stream is not contiguous"
    )


def test_the_fake_and_the_real_provider_agree_on_mpeg_version() -> None:
    """A fake must be valid in the format the real provider returns.

    ``MP3_SILENCE`` was MPEG-1 while the live provider returns MPEG-2, and nothing
    offline noticed. This pins the fake to the format real runs produce, so the
    two cannot drift apart again silently.
    """
    assert (MP3_SILENCE[1] >> 3) & 0x03 == (_MPEG2_FRAME[1] >> 3) & 0x03


async def test_it_narrates_every_page_and_writes_the_files(
    tmp_path: Path,
    story: Story,
    brief: StoryBrief,
    monkeypatch: pytest.MonkeyPatch,
    speak: Callable[..., FakeSpeechProvider],
) -> None:
    provider = speak(monkeypatch)
    narration = await run_narration_pipeline(story, brief, tmp_path)

    page_count = len(story.pages)
    assert narration.pages_narrated == page_count
    assert narration.is_complete is True
    assert len(provider.calls) == page_count

    for item in narration.items:
        assert item.path is not None
        assert item.path.exists()
        assert item.path.read_bytes() == MP3_SILENCE

    assert narration.stitched is not None
    assert narration.stitched.exists()
    assert narration.stitched.stat().st_size > 0


async def test_it_speaks_the_page_text_verbatim(
    tmp_path: Path,
    story: Story,
    brief: StoryBrief,
    monkeypatch: pytest.MonkeyPatch,
    speak: Callable[..., FakeSpeechProvider],
) -> None:
    # The whole no-agent decision rests on this: the audio and the printed page
    # say the same words. The hash is what makes it checkable after the fact
    # rather than merely intended.
    provider = speak(monkeypatch)
    narration = await run_narration_pipeline(story, brief, tmp_path)

    spoken = [text for text, _voice, _speed in provider.calls]
    assert spoken == [page.text for page in story.pages]

    for page, item in zip(story.pages, narration.items, strict=True):
        assert item.sha256 == hashlib.sha256(page.text.encode()).hexdigest()


async def test_it_uses_the_voice_and_pace_from_the_brief(
    tmp_path: Path,
    story: Story,
    brief: StoryBrief,
    monkeypatch: pytest.MonkeyPatch,
    speak: Callable[..., FakeSpeechProvider],
) -> None:
    male = brief.model_copy(update={"voice": Voice.MALE})
    provider = speak(monkeypatch)
    await run_narration_pipeline(story, male, tmp_path)

    assert {voice for _t, voice, _s in provider.calls} == {_VOICES[Voice.MALE]}
    expected = _speed_for(brief.child.reading_level)
    assert {speed for _t, _v, speed in provider.calls} == {expected}


async def test_the_record_names_the_provider_id_not_the_enum(
    tmp_path: Path,
    story: Story,
    brief: StoryBrief,
    monkeypatch: pytest.MonkeyPatch,
    speak: Callable[..., FakeSpeechProvider],
) -> None:
    # A run that does not record which mode it ran in cannot be compared against
    # another afterwards -- `meta.json` records `world_rules` for that reason. The
    # same argument applies here: a run must answer "which voice was this?"
    # without re-deriving the mapping.
    speak(monkeypatch)
    narration = await run_narration_pipeline(story, brief, tmp_path)
    assert narration.voice_id == _VOICES[Voice.FEMALE]
    assert narration.voice_id != Voice.FEMALE.value


async def test_one_failed_page_still_produces_a_playable_book(
    tmp_path: Path,
    story: Story,
    brief: StoryBrief,
    monkeypatch: pytest.MonkeyPatch,
    speak: Callable[..., FakeSpeechProvider],
) -> None:
    # Fails soft, like illustration and unlike write_story. A missing page of
    # narration is a missing page of narration; it must not destroy the book.
    target = story.pages[1]
    speak(monkeypatch, fail_on=(target.text[:20],))

    narration = await run_narration_pipeline(story, brief, tmp_path)

    assert narration.pages_narrated == len(story.pages) - 1
    assert narration.is_complete is False

    failed = next(i for i in narration.items if i.page_number == target.page_number)
    assert failed.status is NarrationStatus.FAILED
    assert failed.path is None

    # And the stitched file exists, built from what survived.
    assert narration.stitched is not None
    assert narration.stitched.exists()


async def test_an_all_failed_run_writes_no_silent_story_file(
    tmp_path: Path,
    story: Story,
    brief: StoryBrief,
    monkeypatch: pytest.MonkeyPatch,
    speak: Callable[..., FakeSpeechProvider],
) -> None:
    # The finding-N case: an empty story.mp3 plays as silence, and silence is
    # indistinguishable from success on a casual listen. So there must be no file
    # at all rather than an empty one.
    provider = speak(monkeypatch, fail_on=(" ",))

    narration = await run_narration_pipeline(story, brief, tmp_path)

    assert narration.pages_narrated == 0
    assert narration.stitched is None
    assert not (tmp_path / "story.mp3").exists()
    assert all(i.status is NarrationStatus.FAILED for i in narration.items)
    # Every page was still attempted -- a run that gave up after the first
    # failure would look identical in the record without this.
    assert len(provider.calls) == len(story.pages)


async def test_pages_are_recorded_in_page_order_not_completion_order(
    tmp_path: Path,
    story: Story,
    brief: StoryBrief,
    monkeypatch: pytest.MonkeyPatch,
    speak: Callable[..., FakeSpeechProvider],
) -> None:
    # Pages run concurrently, so the order they finish in is not the order they
    # belong in.
    speak(monkeypatch)
    narration = await run_narration_pipeline(story, brief, tmp_path)

    assert [i.page_number for i in narration.items] == [
        p.page_number for p in story.pages
    ]
    names = [i.path.name for i in narration.items if i.path]
    assert names == sorted(names)  # zero-padded, so lexical order is page order


async def test_the_stitched_file_is_the_pages_in_order(
    tmp_path: Path,
    story: Story,
    brief: StoryBrief,
    monkeypatch: pytest.MonkeyPatch,
    speak: Callable[..., FakeSpeechProvider],
) -> None:
    speak(monkeypatch)
    narration = await run_narration_pipeline(story, brief, tmp_path)
    assert narration.stitched is not None
    expected = stitch([MP3_SILENCE] * len(story.pages))
    assert narration.stitched.read_bytes() == expected


async def test_it_writes_a_narration_record_to_disk(
    tmp_path: Path,
    story: Story,
    brief: StoryBrief,
    monkeypatch: pytest.MonkeyPatch,
    speak: Callable[..., FakeSpeechProvider],
) -> None:
    # `save_json` once could not serialise the web ledger, and the live run died
    # after paying for the search. A Path is not JSON-serialisable by default,
    # and this record is full of them.
    import json

    speak(monkeypatch)
    await run_narration_pipeline(story, brief, tmp_path)

    record = tmp_path / "narration.json"
    assert record.exists()
    loaded = json.loads(record.read_text())
    assert loaded["voice_id"] == _VOICES[Voice.FEMALE]
    assert len(loaded["items"]) == len(story.pages)


class TestToolErrorTranslation:
    """The MCP tool layer, which translates only what a caller can act on.

    Same rule as `write_story_tool` and `illustrate_story_tool`: a
    `ConfigurationError` names a variable an operator can set, so it becomes a
    `ToolError`; anything else is our own defect and propagates as one.
    """

    async def test_a_missing_key_becomes_a_tool_error(
        self,
        tmp_path: Path,
        story: Story,
        brief: StoryBrief,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An unset XAI_API_KEY reaches a client as a sentence, not a traceback."""

        def raise_missing_key(*_: object, **__: object) -> None:
            raise AudioConfigurationError(
                "Model 'grok-speech' requires XAI_API_KEY, which is not set."
            )

        monkeypatch.setattr(narrate_module, "build_speech_model", raise_missing_key)

        with pytest.raises(ToolError, match="XAI_API_KEY"):
            await narrate_story_tool(brief, story, str(tmp_path))

    async def test_a_partly_failed_narration_returns_rather_than_raises(
        self,
        tmp_path: Path,
        story: Story,
        brief: StoryBrief,
        monkeypatch: pytest.MonkeyPatch,
        speak: Callable[..., FakeSpeechProvider],
    ) -> None:
        """Narration fails soft, so the tool must not convert partial success
        into an error. The `StoryNarration` returned IS the report -- a client
        reads `is_complete`. Verified live by the `live-rejected` run.
        """
        speak(monkeypatch, fail_on=(story.pages[1].text[:20],))

        narration = await narrate_story_tool(brief, story, str(tmp_path))

        assert narration.is_complete is False
        assert narration.pages_narrated == len(story.pages) - 1

    async def test_it_passes_its_arguments_through_in_the_right_order(
        self,
        tmp_path: Path,
        story: Story,
        brief: StoryBrief,
        monkeypatch: pytest.MonkeyPatch,
        speak: Callable[..., FakeSpeechProvider],
    ) -> None:
        """`run_narration_pipeline` takes (story, brief, directory) while
        `run_illustration_pipeline` takes (brief, story, directory). Two
        adjacent tools whose pipelines disagree on argument order is exactly the
        swap a type checker cannot catch, since both are Pydantic models.
        """
        speak(monkeypatch)

        narration = await narrate_story_tool(brief, story, str(tmp_path))

        assert narration.voice_id == _VOICES[brief.voice]
        assert len(narration.items) == len(story.pages)
        assert (tmp_path / "narration.json").exists()
