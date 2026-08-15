"""A clip maker that returns canned video, for tests and offline runs.

The sibling of ``FakeModel``, ``FakeEmbedder``, ``FakeImageProvider`` and
``FakeSpeechProvider``, and the thing that keeps "no network required, no ffmpeg
required" true now that the package can make video. Same reasoning as those four: a
fake passed to a constructor cannot rot the way a monkeypatch of a module attribute
by string path can, and it lets a test assert on *what was asked for* rather than
only on what came back.

**It records whether there was a picture, the duration and the page number**,
because the interesting assertions in this feature are about the request: was the
clip built to the page's own narration length, and did a page with no illustration
become a held card rather than being dropped.

**A fake's bytes must be valid in the format the real provider returns, and this
project has already paid for that once.**
``FakeSpeechProvider``'s canned MP3 was MPEG-1 at 44.1 kHz while the provider
returns MPEG-2 at 24 kHz. Both are legal frames; every offline test passed; and
``story.mp3`` was corrupt, with 9.2% of it parsing as contiguous frames. A fake
wrong in a *plausible* direction passes the tests written about it and fails
downstream on real data.

So ``TINY_MP4`` is **not hand-written**. It was produced by running the real
encoder with the settings ``video/kenburns.py`` actually uses -- one frame of the
held-card colour at the production size, codec and pixel format -- and captured
verbatim. That makes the fake correct by construction rather than by inspection,
which is the difference the speech fake did not have.
``tests/test_video_live.py`` then pins it against freshly generated output, so the
two cannot drift apart silently if the encoder or its settings ever change.
"""

import base64

from sparkstory.entities.exceptions import VideoGenerationError
from sparkstory.models.get_clip_maker import Clip, ClipMaker

#: One real frame of H.264 in a fragmented MP4: 1280x720, ``yuv420p``, produced by
#: ``libx264 -preset veryfast -crf 20 -movflags frag_keyframe+empty_moov`` -- the
#: exact settings ``video/kenburns.py`` encodes with.
#:
#: Base64 rather than a hex literal because these are 1886 bytes of genuine encoder
#: output, and the point is that no human chose any of them. **Nor did a human
#: transcribe them:** the first version of this constant was hand-split into
#: string literals and lost three bytes at a line boundary, which ffprobe reported
#: as ``Invalid NAL unit size (218 > 215)`` -- a file that opens, reports its codec
#: and dimensions, and cannot be decoded. The literal below was emitted by a script
#: that asserted the round trip before writing it, and
#: ``tests/test_video_live.py`` is what caught the mangled one.
TINY_MP4 = base64.b64decode(
    "AAAAJGZ0eXBpc29tAAACAGlzb21pc282aXNvMmF2YzFtcDQxAAAC721vb3YAAABsbXZoZAAA"
    "AAAAAAAAAAAAAAAAA+gAAAAAAAEAAAEAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAQAA"
    "AAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIAAAHxdHJhawAA"
    "AFx0a2hkAAAAAwAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABAAAAAAAA"
    "AAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAQAAAAAUAAAAC0AAAAAABjW1kaWEAAAAgbWRoZAAA"
    "AAAAAAAAAAAAAAAAPAAAAAAAVcQAAAAAAC1oZGxyAAAAAAAAAAB2aWRlAAAAAAAAAAAAAAAA"
    "VmlkZW9IYW5kbGVyAAAAAThtaW5mAAAAFHZtaGQAAAABAAAAAAAAAAAAAAAkZGluZgAAABxk"
    "cmVmAAAAAAAAAAEAAAAMdXJsIAAAAAEAAAD4c3RibAAAAKxzdHNkAAAAAAAAAAEAAACcYXZj"
    "MQAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAAUAAtAASAAAAEgAAAAAAAAAAQAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAABj//wAAADZhdmNDAWQAH//hABpnZAAfrNlAUAW7ARAA"
    "AAMAEAAAAwPA8YMZYAEABWjvhvLA/fj4AAAAABBwYXNwAAAAAQAAAAEAAAAQc3R0cwAAAAAA"
    "AAAAAAAAEHN0c2MAAAAAAAAAAAAAABRzdHN6AAAAAAAAAAAAAAAAAAAAEHN0Y28AAAAAAAAA"
    "AAAAAChtdmV4AAAAIHRyZXgAAAAAAAAAAQAAAAEAAAAAAAAAAAAAAAAAAABidWR0YQAAAFpt"
    "ZXRhAAAAAAAAACFoZGxyAAAAAAAAAABtZGlyYXBwbAAAAAAAAAAAAAAAAC1pbHN0AAAAJal0"
    "b28AAAAdZGF0YQAAAAEAAAAATGF2ZjU4Ljc2LjEwMAAAAHBtb29mAAAAEG1maGQAAAAAAAAA"
    "AQAAAFh0cmFmAAAAJHRmaGQAAAA5AAAAAQAAAAAAAAMTAAACAAAAA5ABAQAAAAAAFHRmZHQB"
    "AAAAAAAAAAAAAAAAAAAYdHJ1bgAAAAUAAAABAAAAeAIAAAAAAAOYbWRhdAAAAq4GBf//qtxF"
    "6b3m2Ui3lizYINkj7u94MjY0IC0gY29yZSAxNjMgcjMwNjAgNWRiNmFhNiAtIEguMjY0L01Q"
    "RUctNCBBVkMgY29kZWMgLSBDb3B5bGVmdCAyMDAzLTIwMjEgLSBodHRwOi8vd3d3LnZpZGVv"
    "bGFuLm9yZy94MjY0Lmh0bWwgLSBvcHRpb25zOiBjYWJhYz0xIHJlZj0xIGRlYmxvY2s9MTow"
    "OjAgYW5hbHlzZT0weDM6MHgxMTMgbWU9aGV4IHN1Ym1lPTIgcHN5PTEgcHN5X3JkPTEuMDA6"
    "MC4wMCBtaXhlZF9yZWY9MCBtZV9yYW5nZT0xNiBjaHJvbWFfbWU9MSB0cmVsbGlzPTAgOHg4"
    "ZGN0PTEgY3FtPTAgZGVhZHpvbmU9MjEsMTEgZmFzdF9wc2tpcD0xIGNocm9tYV9xcF9vZmZz"
    "ZXQ9MCB0aHJlYWRzPTE4IGxvb2thaGVhZF90aHJlYWRzPTUgc2xpY2VkX3RocmVhZHM9MCBu"
    "cj0wIGRlY2ltYXRlPTEgaW50ZXJsYWNlZD0wIGJsdXJheV9jb21wYXQ9MCBjb25zdHJhaW5l"
    "ZF9pbnRyYT0wIGJmcmFtZXM9MyBiX3B5cmFtaWQ9MiBiX2FkYXB0PTEgYl9iaWFzPTAgZGly"
    "ZWN0PTEgd2VpZ2h0Yj0xIG9wZW5fZ29wPTAgd2VpZ2h0cD0xIGtleWludD0yNTAga2V5aW50"
    "X21pbj0yNSBzY2VuZWN1dD00MCBpbnRyYV9yZWZyZXNoPTAgcmNfbG9va2FoZWFkPTEwIHJj"
    "PWNyZiBtYnRyZWU9MSBjcmY9MjAuMCBxY29tcD0wLjYwIHFwbWluPTAgcXBtYXg9NjkgcXBz"
    "dGVwPTQgaXBfcmF0aW89MS40MCBhcT0xOjEuMDAAgAAAANpliIQAJ//SeB7vpPMo1hp7xf/f"
    "bTdQZiEmOaBYbkECxgiQIAAAAwAAAwAAAwAAAwADmGlSwj5L9MkrqAAAAwAAAwOoAAboABMw"
    "AD2AAMkAA4gAD+AATIACGgAPsABigAOsAAADAAADAAADAAADAAADAAADAAADAAADAAADAAAD"
    "AAADAAADAAADAAADAAADAAADAAADAAADAAADAAADAAADAAADAAADAAADAAADAAADAAADAAAD"
    "AAADAAADAAADAAADAAADAAADAAADAAADAAADAAADAAADAADZgQAAAENtZnJhAAAAK3RmcmEB"
    "AAAAAAAAAQAAAAAAAAABAAAAAAAAAAAAAAAAAAADEwEBAQAAABBtZnJvAAAAAAAAAEM="
)


class FakeClipMaker:
    """Stands in for a clip maker, recording how it was asked.

    Args:
        fail_on: Page numbers that should raise instead of returning a clip.
            Matching on the page number rather than a call index is what lets a
            test say "page 3 fails" without depending on the order the workflow
            happens to run in -- which matters here because pages are processed
            concurrently, exactly as ``FakeSpeechProvider`` argues.
    """

    def __init__(self, *, fail_on: tuple[int, ...] = ()) -> None:
        self.fail_on = fail_on
        #: ``(had_image, duration, page_number)`` per call, in call order. A failed
        #: call is recorded too, so a test can assert every page was attempted.
        self.calls: list[tuple[bool, float, int]] = []

    async def make_clip(
        self, image: bytes | None, duration: float, page_number: int
    ) -> Clip:
        self.calls.append((image is not None, duration, page_number))

        if page_number in self.fail_on:
            raise VideoGenerationError(
                f"FakeClipMaker was asked to fail on page {page_number}."
            )

        return Clip(data=TINY_MP4, video_format="mp4")

    def as_maker(self) -> ClipMaker:
        """Wrap this fake in the dataclass the workflow is injected with."""
        return ClipMaker(make_clip=self.make_clip)
