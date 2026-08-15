"""Measuring what is actually in a media file.

**These are the measurement half of the feature, not a convenience.** The
acceptance criterion for the assembly is arithmetic -- total frames equals the sum
of per-page frame counts -- and it can only be checked by reading the finished
file rather than by reasoning about it.

**``probe_frame_count`` counts decoded frames rather than reading the container's
duration**, and that distinction is load-bearing. A concatenated stream's header
duration is an *estimate*: the demuxer reports what the first segment implies. So
an assertion on it is a check with no room to fail -- which is exactly how a
corrupt ``story.mp3`` once passed three layers of tests. Counting frames costs a
full decode and is worth it.
"""

import asyncio
from pathlib import Path

from sparkstory.entities.exceptions import VideoGenerationError
from sparkstory.video.ffmpeg import FFPROBE

#: Matches ``ffmpeg.py``'s excerpt, for the same reason.
_ERROR_EXCERPT = 500


async def _probe(args: list[str]) -> str:
    """Run ffprobe and return its stdout, stripped.

    Raises:
        VideoGenerationError: a non-zero exit. Retryable, like every other
            subprocess failure here.
    """
    process = await asyncio.create_subprocess_exec(
        FFPROBE,
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, err = await process.communicate()

    if process.returncode != 0:
        tail = err.decode("utf-8", "replace")[-_ERROR_EXCERPT:]
        raise VideoGenerationError(f"ffprobe exited {process.returncode}: {tail}")

    return out.decode("utf-8", "replace").strip()


async def probe_audio_duration(path: Path) -> float:
    """How long this audio file plays, in seconds.

    This is what sets a page's place on the timeline, so it is measured from the
    file rather than estimated from word count. The audio-as-spine rule in
    ``workflows/animate.py`` rests on this being a measurement.

    Raises:
        VideoGenerationError: ffprobe failed, or reported no parsable duration.
    """
    raw = await _probe(
        [
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            str(path),
        ]
    )
    try:
        return float(raw)
    except ValueError as exc:
        raise VideoGenerationError(
            f"ffprobe reported no duration for {path}: {raw!r}"
        ) from exc


async def probe_frame_count(path: Path) -> int:
    """How many video frames this file actually contains.

    Decoded, not read from the header -- see the module docstring. Slow by design.

    Raises:
        VideoGenerationError: ffprobe failed, or reported no parsable count.
    """
    raw = await _probe(
        [
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "csv=p=0",
            str(path),
        ]
    )
    try:
        return int(raw)
    except ValueError as exc:
        raise VideoGenerationError(
            f"ffprobe reported no frame count for {path}: {raw!r}"
        ) from exc
