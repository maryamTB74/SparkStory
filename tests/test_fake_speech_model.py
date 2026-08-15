"""The fake speech provider: valid audio, and a recording of how it was asked."""

import pytest

from sparkstory.entities.exceptions import AudioGenerationError
from sparkstory.models.fake_speech_model import MP3_SILENCE, FakeSpeechProvider


async def test_speak_returns_decodable_mp3_bytes() -> None:
    # A fake that is wrong in a plausible direction passes exact-match tests and
    # fails only downstream. `_PNG_4X3`'s predecessor was a hand-copied
    # PNG that PIL rejected -- every test in its own module passed, and only the
    # renderer failed. So assert the frame header, not the length.
    provider = FakeSpeechProvider()
    audio = await provider.speak("Page one.", "eve", 0.9)

    assert audio.audio_format == "mp3"
    assert len(audio.data) > 0
    # 11 bits of frame sync, then MPEG-2 Layer III -- what any decoder reads first,
    # and what the live provider actually returns. This asserted MPEG-1 until a
    # live run showed the provider sends MPEG-2 at 24 kHz; the mismatch corrupted
    # the stitched file while every offline test passed.
    assert audio.data[0] == 0xFF
    assert audio.data[1] & 0xE0 == 0xE0
    assert (audio.data[1] >> 3) & 0x03 == 0b10  # MPEG-2
    assert (audio.data[1] >> 1) & 0x03 == 0b01  # Layer III


async def test_the_fakes_frame_length_matches_its_declared_size() -> None:
    """A frame whose real length disagrees with its header breaks a decoder.

    This is the assertion that would have caught the stitcher defect offline: the
    header says 128 kbps at 24 kHz, which is 384 bytes per frame, so the bytes have
    to actually be 384 long or a parser walking declared lengths lands mid-stream.
    """
    provider = FakeSpeechProvider()
    audio = await provider.speak("Page one.", "eve", 1.0)

    bitrate = [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0][
        (audio.data[2] >> 4) & 0x0F
    ]
    rate = [22050, 24000, 16000, 0][(audio.data[2] >> 2) & 0x03]
    declared = int(576 / 8 * bitrate * 1000 / rate) + ((audio.data[2] >> 1) & 1)

    assert bitrate == 128
    assert rate == 24000
    assert len(audio.data) == declared


async def test_it_records_how_it_was_asked() -> None:
    # The interesting assertions in this feature are about the request, not the
    # waveform: was the right voice used, was the pace derived from the brief.
    provider = FakeSpeechProvider()
    await provider.speak("Page one.", "eve", 0.9)
    await provider.speak("Page two.", "leo", 1.1)

    assert provider.calls == [
        ("Page one.", "eve", 0.9),
        ("Page two.", "leo", 1.1),
    ]


async def test_fail_on_matches_text_rather_than_call_order() -> None:
    # Pages generate concurrently under asyncio.gather, so a test that said "the
    # third call fails" would depend on scheduling. Matching the text lets a test
    # say "page 3 fails" and mean it -- FakeImageProvider's reasoning.
    provider = FakeSpeechProvider(fail_on=("page three",))

    ok = await provider.speak("This is page one.", "eve", 1.0)
    assert ok.data

    with pytest.raises(AudioGenerationError):
        await provider.speak("This is page three.", "eve", 1.0)

    # The failed call is still recorded: a test asserting "every page was
    # attempted" needs it.
    assert len(provider.calls) == 2


async def test_fail_on_is_case_insensitive() -> None:
    provider = FakeSpeechProvider(fail_on=("PAGE THREE",))
    with pytest.raises(AudioGenerationError):
        await provider.speak("this is page three.", "eve", 1.0)


async def test_as_model_produces_a_usable_speech_model() -> None:
    provider = FakeSpeechProvider()
    model = provider.as_model()
    audio = await model.speak("Page one.", "eve", 1.0)
    assert audio.audio_format == "mp3"
    assert provider.calls == [("Page one.", "eve", 1.0)]


def test_the_silence_constant_is_a_valid_frame() -> None:
    assert MP3_SILENCE[0] == 0xFF
    assert MP3_SILENCE[1] & 0xE0 == 0xE0
