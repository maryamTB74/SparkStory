"""The single seam through which a still becomes moving picture.

The fifth seam, after ``get_chat_model``, ``get_embedder``, ``get_image_model``
and ``get_speech_model``, and separate from all four for the reason they are
separate from each other: each takes a different kind of input and returns a
different kind of output, so one factory over five would only branch on which kind
an entry is.

**Why this lives in ``models/`` when there is no model in it.** ``models/`` means
LLM wiring, and the only implementation today shells out to a local ffmpeg -- which
is not that. It goes here anyway because the seam's *job* is identical to
``get_image_model``'s (one factory, one protocol, swap the implementation without
touching the caller) and because build 2 puts a real image-to-video provider behind
it, at which point ``models/`` is exactly right. Putting it in ``video/`` today
means moving it in the session that adds a provider, and being right for build 2
beats being tidy for build 1.

**``async`` despite ffmpeg being local**, because build 2 is network-bound and the
workflow above must not change shape when the implementation does. The Ken Burns
maker implements it with ``asyncio.create_subprocess_exec`` and gets real
concurrency across pages for free.

**No setting selects between makers.** There is one entry, so there is nothing to
select between, and a ``video_provider`` setting would be config for a feature
that does not exist.

**``duration`` is a request, and how well a provider honours it was the open
question here.** Ken Burns honours it exactly, because a filter over a still can
run for any length. The design was written expecting a provider could not -- the
assumption being a fixed 4-5 second clip that would have to be looped, frozen or
joined to fill a page's narration.

**A spike against xAI on 2026-08-14 found otherwise, and the correction is
recorded rather than quietly dropped.** ``grok-imagine-video`` accepted
``duration: 10`` and returned a clip reporting exactly that, so on this provider a
page's own narration length may be requestable directly and the looping problem
may not arise at all. Two caveats keep this from being settled: only one value was
ever confirmed before the account's credit ran out, and **the endpoint silently
ignores unknown fields** -- ``zzz_not_a_field`` returned 200 -- so a status code
proves nothing about whether a parameter was read. Only measuring the returned
file does.

So the seam keeps ``duration`` as a request rather than a guarantee, and an
implementation that cannot honour it exactly still confines that problem to one
function rather than letting it leak into the assembly.

The inversion -- make the video the spine and stretch the audio to fit fixed clips
-- is rejected regardless: it means resampling narration, and this project has
already corrupted one audio stream by manipulating it arithmetically.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from sparkstory.entities.exceptions import VideoConfigurationError


@dataclass(frozen=True)
class Clip:
    """One page's moving picture, as bytes plus the format they are in."""

    data: bytes
    #: Lowercase, no dot -- "mp4". Names the file on disk, so it is read from what
    #: was actually produced rather than assumed. ``GeneratedAudio.audio_format``
    #: carries the same guard for the same reason: a provider changing codec must
    #: not leave us writing one format into a filename claiming another.
    video_format: str


#: ``(image, duration, page_number) -> clip``.
#:
#: ``image`` is ``bytes | None``: a page with narration and no illustration is a
#: held card, which is a legitimate output rather than a failure, and giving the
#: seam a second method for that case would mean build 2 implementing two things.
#:
#: ``page_number`` rather than a camera move, because a *move* is a Ken Burns
#: concept and this seam must not presume its only implementation. Each maker
#: decides what a page number means to it; a provider may well ignore it.
ClipFn = Callable[[bytes | None, float, int], Awaitable[Clip]]


@dataclass(frozen=True)
class ClipMaker:
    """What the video workflow needs from a clip source."""

    make_clip: ClipFn


#: Every maker id this package knows. One -- see the module docstring.
KNOWN_MAKERS = frozenset({"kenburns"})


def get_clip_maker(maker_id: str) -> ClipMaker:
    """Build the clip maker named by ``maker_id``.

    Raises:
        VideoConfigurationError: the id is unknown, or ffmpeg is not installed.
            Not retryable -- neither a typo nor a missing binary is fixed by trying
            again, and a config error that is retried prints one traceback per
            attempt for a problem whose fix is a single line.
    """
    if maker_id not in KNOWN_MAKERS:
        known = ", ".join(sorted(KNOWN_MAKERS))
        raise VideoConfigurationError(
            f"Unknown clip maker {maker_id!r}. Known entries: {known}."
        )

    # Imported here rather than at module scope so that importing the seam does
    # not drag in the subprocess layer, and so the ffmpeg check happens when a
    # maker is *built* -- once per run -- rather than at import time.
    from sparkstory.video.ffmpeg import require_ffmpeg
    from sparkstory.video.kenburns import KenBurnsClipMaker

    # Checked here rather than per page: every page would fail identically, so
    # failing after doing the work tells nobody anything they could not have known.
    require_ffmpeg()
    return KenBurnsClipMaker().as_maker()
