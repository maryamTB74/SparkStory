"""A speech provider that returns canned audio, for tests and offline runs.

The sibling of ``FakeModel``, ``FakeEmbedder`` and ``FakeImageProvider``, and the
thing that keeps "no network required" true now that the package can speak. Same
reasoning as those three: a fake passed to a constructor cannot rot the way a
monkeypatch of a module attribute by string path can, and it lets a test assert on
*what was asked for* rather than only on what came back.

**It records the text, voice and speed of every call**, because the interesting
assertions in this feature are about the request rather than the waveform: was the
right voice used, was the pace derived from the reading level, and did a failed
page leave the book playable.

``FakeSpeechProvider`` is deliberately not a ``FakeModel`` subclass, for the same
reason ``FakeImageProvider`` is not: a speech model has no
``with_structured_output`` and binds no schema, so inheriting would offer methods
that mean nothing here.

**The bytes are a real MPEG frame header, because a fake that is wrong in a
plausible direction passes every test written about it.** ``FakeEmbedder``'s
first hash was wrong in a plausible direction and passed its own tests;
``_PNG_4X3``'s predecessor was a hand-copied PNG that PIL rejected, and every test
in its module passed while only the renderer failed. So this is a valid frame and
the tests assert the sync bits rather than the length.
"""

from sparkstory.entities.exceptions import AudioGenerationError
from sparkstory.models.get_speech_model import GeneratedAudio, SpeechModel

#: One MPEG-**2** Layer III frame plus a zeroed payload, matching what the live
#: provider actually returns: `ff f3` is the sync word (11 set bits) followed by
#: version bits `10` (MPEG-2) and layer bits `01` (Layer III), then 128 kbps at
#: 24 kHz.
#:
#: **It was MPEG-1 (`ff fb`, 44.1 kHz) until a live run proved otherwise**, and
#: that mismatch caused a real defect: the stitcher used these bytes as its
#: inter-page gap, so `story.mp3` changed sample rate mid-stream and a decoder lost
#: sync at the first join -- only 9.2% of the file parsed as contiguous frames.
#: Every offline test passed throughout, because they asserted the sync bits and
#: the layer, which a *plausible* header satisfies.
#:
#: The lesson: a fake must be valid in the format the real provider returns, not
#: merely valid, and an assertion has to check structure that *composes* -- walking
#: declared frame lengths end to end -- rather than one frame in isolation. There is
#: now a test pinning this constant's MPEG version to the one real runs produce, so
#: the fake and the provider cannot drift apart silently again.
#: 384 bytes exactly, because that is what the header declares: an MPEG-2 Layer III
#: frame at 128 kbps / 24 kHz is `576 / 8 * 128000 / 24000` bytes. A fake whose real
#: length disagreed with its own header would leave a parser walking declared
#: lengths mid-stream -- the second half of the same defect.
_FRAME_BYTES = 384
MP3_SILENCE = b"\xff\xf3\xc4\xc4" + b"\x00" * (_FRAME_BYTES - 4)


class FakeSpeechProvider:
    """Stands in for a speech provider, recording how it was asked.

    Args:
        fail_on: Substrings of the text that should fail instead of returning
            audio. A text containing any of them (case-insensitively) raises
            ``AudioGenerationError``. Matching on the *text* rather than a call
            index is what lets a test say "page 3 fails" without depending on the
            order the workflow happens to run in -- which matters here because
            pages are narrated concurrently.
    """

    def __init__(self, *, fail_on: tuple[str, ...] = ()) -> None:
        self.fail_on = fail_on
        #: ``(text, voice_id, speed)`` per call, in call order. A failed call is
        #: recorded too, so a test can assert every page was attempted.
        self.calls: list[tuple[str, str, float]] = []

    async def speak(self, text: str, voice_id: str, speed: float) -> GeneratedAudio:
        self.calls.append((text, voice_id, speed))

        lowered = text.lower()
        for needle in self.fail_on:
            if needle.lower() in lowered:
                raise AudioGenerationError(
                    f"FakeSpeechProvider was asked to fail on {needle!r}."
                )

        return GeneratedAudio(data=MP3_SILENCE, audio_format="mp3")

    def as_model(self) -> SpeechModel:
        """Wrap this provider in the dataclass the workflow is injected with."""
        return SpeechModel(speak=self.speak)
