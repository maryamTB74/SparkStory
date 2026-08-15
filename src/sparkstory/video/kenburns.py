"""A still picture, held for a measured time, with a slow move over it.

**The move is arithmetic on the page number**, not a model's choice and not
random. The argument against a model is that an instruction gets satisfied the
laziest legal way: the laziest answer to "pick a camera move for this scene" is
*zoom in*, every time, so the call buys the fixed
option back with added noise and a per-page cost. It is the same thing
``narrate.py`` refused when it declined a per-page delivery note. Random is
rejected for a different reason -- it forfeits reproducibility, which is what lets
the assembly test in ``tests/test_video_live.py`` assert anything real.

**Two details here are defects if they are got wrong, and neither is obvious.**

``zoompan`` quantises zoom per frame, so panning across a source at its native size
visibly stutters. The still is upscaled by ``_UPSCALE`` first, which makes
sub-pixel movement land on real pixels. On a book for a five-year-old, a stuttering
picture is the whole product.

The frame count is computed once, here, and passed as ``d=``. Deriving it inside
the filter risks a clip one frame short of its audio; six of those and the video
drifts out of sync with its own narration by the end. ``frames_for`` rounds rather
than truncating for exactly this reason.

**``yuv420p`` is not a free choice.** Its absence is the classic "plays in VLC,
black in Safari" bug, and no unit test catches it -- only a live run does.

**The picture is fitted with padding, never cropped.** A crop can cut a character
out of frame, which is the one thing an illustrated book cannot afford: the
personalised child is usually the subject, and the pages are squarer than 16:9.
"""

from enum import StrEnum

from sparkstory.models.get_clip_maker import Clip, ClipMaker
from sparkstory.video.ffmpeg import run_ffmpeg

#: 30 is the floor at which a slow pan reads as motion rather than as steps. Not a
#: setting: nothing selects between values, and a second frame rate
#: would need a second set of measured frame counts.
FPS = 30

#: 720p. Large enough to look like a book on a phone or a laptop, small enough that
#: a minute encodes in seconds.
WIDTH = 1280
HEIGHT = 720

#: How far the still is enlarged before panning, to defeat ``zoompan``'s per-frame
#: quantisation. See the module docstring: without it the picture stutters.
_UPSCALE = 4

#: How far a zoom travels over a whole clip. 1.15 is a slow push at any page
#: length, because the per-frame rate is derived from the frame count rather than
#: fixed.
_ZOOM_RANGE = 0.15

#: The colour behind a padded picture, and the whole frame of a held card -- a page
#: that has narration and no illustration. Near-black rather than pure black so a
#: card reads as deliberate rather than as a dropout.
_CARD_COLOUR = "0x14141a"


class Move(StrEnum):
    """The four camera moves, cycled by page number."""

    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"
    PAN_RIGHT = "pan_right"
    PAN_LEFT = "pan_left"


_MOVES = (Move.ZOOM_IN, Move.ZOOM_OUT, Move.PAN_RIGHT, Move.PAN_LEFT)


def move_for(page_number: int) -> Move:
    """Which move this page gets. Pure, so a re-render is byte-comparable."""
    return _MOVES[(page_number - 1) % len(_MOVES)]


def frames_for(duration: float) -> int:
    """How many frames fill ``duration`` seconds.

    Rounded, never truncated: truncation loses up to a frame per page and the
    error accumulates across a book. At least one, because a zero-frame clip is an
    empty file that the concat would silently drop.
    """
    return max(1, round(duration * FPS))


def _zoom_expression(move: Move, frames: int) -> str:
    """The ``zoompan`` zoom term for this move.

    Both zooms are written against ``on`` -- the output frame index -- rather than
    against the accumulating ``zoom`` variable. An accumulator reads more
    naturally and drifts: its per-frame error compounds over hundreds of frames,
    so the end point depends on the clip length. ``on`` lands on the same zoom at
    the same fraction of every clip.
    """
    end = 1 + _ZOOM_RANGE
    if move is Move.ZOOM_IN:
        return f"1.0+{_ZOOM_RANGE}*on/{frames}"
    if move is Move.ZOOM_OUT:
        return f"{end}-{_ZOOM_RANGE}*on/{frames}"
    # Both pans hold a constant slight zoom, so there is room to move within frame.
    return f"{1 + _ZOOM_RANGE / 2}"


def _pan_expressions(move: Move, frames: int) -> tuple[str, str]:
    """The ``zoompan`` x and y terms for this move."""
    centre_x = "iw/2-(iw/zoom/2)"
    centre_y = "ih/2-(ih/zoom/2)"
    if move is Move.PAN_RIGHT:
        return (f"(iw-iw/zoom)*on/{frames}", centre_y)
    if move is Move.PAN_LEFT:
        return (f"(iw-iw/zoom)*(1-on/{frames})", centre_y)
    return (centre_x, centre_y)


def _filter_for(move: Move, frames: int) -> str:
    """The whole filter chain for one clip.

    Upscale, then pan, then fit to the output frame with padding. Verified by
    running: a 90-frame zoom-in over a real page image produced exactly 90 frames
    at 1280x720 ``yuv420p``, with the character fully in frame at both ends.
    """
    zoom = _zoom_expression(move, frames)
    pan_x, pan_y = _pan_expressions(move, frames)
    return (
        f"scale={WIDTH * _UPSCALE}:-2,"
        f"zoompan=z='{zoom}':x='{pan_x}':y='{pan_y}'"
        f":d={frames}:s={WIDTH}x{HEIGHT}:fps={FPS},"
        f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=decrease,"
        f"pad={WIDTH}:{HEIGHT}:(ow-iw)/2:(oh-ih)/2:color={_CARD_COLOUR},"
        "setsar=1"
    )


def _encode_args() -> list[str]:
    """Output settings shared by every clip.

    Identical across clips so the assembly can concatenate with ``-c copy`` rather
    than re-encoding -- which is both faster and lossless. ``yuv420p`` is
    load-bearing; see the module docstring.

    **``frag_keyframe+empty_moov`` rather than ``+faststart``, and this was found
    by running it.** A clip is written to ``pipe:1``, and ``+faststart`` has to
    seek back to the start to move the ``moov`` atom to the front -- so ffmpeg
    exits 1 with *"muxer does not support non seekable output"*. A fragmented MP4
    needs no seek and concatenates identically.

    The first attempt passed a hand-run spike because that spike wrote to a
    **file**, where seeking works: a check whose conditions differ from production
    in one detail proves nothing about that detail. ``+faststart`` still belongs
    on the assembled ``story.mp4``, which is
    written to a real path and is the file a player actually opens.
    """
    return [
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-movflags",
        "frag_keyframe+empty_moov",
        "-f",
        "mp4",
    ]


class KenBurnsClipMaker:
    """Turns one still into ``duration`` seconds of moving picture, locally."""

    async def make_clip(
        self, image: bytes | None, duration: float, page_number: int
    ) -> Clip:
        """One page's clip.

        ``image`` of ``None`` produces a held card of the same length: a page with
        narration and no illustration still belongs on the timeline, because the
        timeline is made of audio.

        Raises:
            VideoGenerationError: ffmpeg exited non-zero. The retryable class.
        """
        frames = frames_for(duration)
        seconds = frames / FPS

        if image is None:
            args = [
                "-f",
                "lavfi",
                "-i",
                f"color=c={_CARD_COLOUR}:s={WIDTH}x{HEIGHT}:r={FPS}",
                "-frames:v",
                str(frames),
                *_encode_args(),
                "pipe:1",
            ]
            return Clip(data=await run_ffmpeg(args), video_format="mp4")

        args = [
            "-loop",
            "1",
            "-t",
            f"{seconds:.6f}",
            "-i",
            "pipe:0",
            "-vf",
            _filter_for(move_for(page_number), frames),
            "-frames:v",
            str(frames),
            *_encode_args(),
            "pipe:1",
        ]
        return Clip(data=await run_ffmpeg(args, stdin=image), video_format="mp4")

    def as_maker(self) -> ClipMaker:
        """Wrap this in the dataclass the workflow is injected with."""
        return ClipMaker(make_clip=self.make_clip)
