"""Running ffmpeg, and failing usefully when it is not there.

**Direct ``create_subprocess_exec`` rather than ``pydub`` or a wrapper package.**
Those add a dependency that shells out to the same binary while hiding the
arguments -- and the arguments are the part that has to be exactly right here
(``zoompan`` framing, ``yuv420p``, frame counts). Keeping them readable and
greppable in one module is worth more than the convenience.

**stdout and stderr are always captured, never inherited.** MCP stdio
transport carries JSON-RPC on stdout, and a subprocess inheriting it would corrupt
the protocol in a way that surfaces as a JSON parse error looking nothing like its
cause. ffmpeg writes its banner and progress to stderr by default, but "by default"
is not a guarantee worth betting the transport on.
"""

import asyncio
import shutil

from sparkstory.entities.exceptions import (
    VideoConfigurationError,
    VideoGenerationError,
)
from sparkstory.utils.logging_utils import get_logger

logger = get_logger(__name__)

FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"

#: How much of ffmpeg's stderr to quote in an error. It is verbose even on
#: success -- codec banners, build flags, per-stream metadata -- and the useful
#: line is the last one.
_ERROR_EXCERPT = 500


def require_ffmpeg() -> None:
    """Check both binaries exist, before any work is done.

    Raises:
        VideoConfigurationError: either binary is missing. Raised once at the top
            of a run rather than per page, because every page would fail
            identically -- the same call ``run_narration_pipeline`` makes for a
            missing API key. Not retryable: trying again cannot install a binary.
    """
    missing = [name for name in (FFMPEG, FFPROBE) if shutil.which(name) is None]
    if missing:
        raise VideoConfigurationError(
            f"{' and '.join(missing)} not found on PATH, and video needs "
            f"{'them' if len(missing) > 1 else 'it'}. "
            "Install ffmpeg (it ships ffprobe): apt install ffmpeg."
        )


async def run_ffmpeg(args: list[str], *, stdin: bytes | None = None) -> bytes:
    """Run ffmpeg with ``args``, returning what it wrote to stdout.

    Args:
        args: Everything after the binary name.
        stdin: Bytes to feed it, for the pipe-in-pipe-out case.

    Returns:
        stdout as bytes -- the encoded clip, when the arguments ask for one.

    Raises:
        VideoGenerationError: a non-zero exit. The retryable class: a killed
            subprocess or a full disk is transient in the way a missing binary is
            not.
    """
    process = await asyncio.create_subprocess_exec(
        FFMPEG,
        *args,
        stdin=asyncio.subprocess.PIPE if stdin is not None else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, err = await process.communicate(input=stdin)

    if process.returncode != 0:
        tail = err.decode("utf-8", "replace")[-_ERROR_EXCERPT:]
        raise VideoGenerationError(f"ffmpeg exited {process.returncode}: {tail}")

    return out
