"""Joining per-page clips into one book, with the narration under it.

**Two concatenations, not one.** The video is joined with ffmpeg's concat demuxer,
the audio with a second, and the two are muxed together. The alternative -- one
filter graph over N inputs -- is a single invocation and is rejected: a filter
graph's failure is a wall of stderr naming a stream label, while a demuxer's is a
line naming a file. On a stage whose whole purpose is assembling artifacts that may
individually be missing, the legible failure is worth the extra process.

**``-c copy`` for the video, re-encoding nothing.** Every clip was produced by the
same encoder at the same settings, so they concatenate without a second encode.
That is what makes assembly fast and, more importantly, lossless -- a re-encode of
a slow pan over a flat illustration is exactly where banding appears.

**The audio is concatenated from the page files rather than reusing
``story.mp3``.** They are the same bytes -- narration's ``stitch`` is plain
concatenation -- but only the per-page files carry a *known page boundary*.
Reusing the stitched file would mean trusting that it contains exactly the pages
that survived selection, which is false the moment one page is excluded.

**``+faststart`` belongs here and not on the clips.** It relocates the ``moov``
atom to the front so a player can start before downloading the whole file, and
doing that needs a seek -- which a pipe cannot do. This output is a real path, so
it works; a clip is written to ``pipe:1``, so there it fails with *"muxer does not
support non seekable output"*. Found by running it.
"""

from pathlib import Path

from sparkstory.video.ffmpeg import run_ffmpeg


def build_concat_file(paths: list[Path]) -> str:
    """The body of an ffmpeg concat list.

    Every path is single-quoted, because the demuxer's parser splits on
    whitespace otherwise -- and run directories here are named after the story
    premise, which is exactly where a space comes from. Internal single quotes are
    escaped in the demuxer's own dialect, since "a child's garden" is an ordinary
    premise rather than an exotic one.

    **Every path is also made absolute, and that was found by running it.** The
    concat demuxer resolves a relative path against *the list file's own
    directory*, not the process working directory. ``narration.json`` stores
    repo-root-relative paths and the list is written into the run directory, so
    ffmpeg looked for ``outputs/<run>/page-01.mp3`` *inside* ``outputs/<run>/``
    and failed with "Impossible to open". Resolving here rather than at the call
    site keeps the rule with the format that imposes it.

    Raises:
        ValueError: the list is empty. Concatenating nothing yields a zero-byte
            file, and a zero-byte video plays as nothing -- which is
            indistinguishable from success on a casual glance. The caller decides
            what to do instead; it must not be handed an empty list.
    """
    if not paths:
        raise ValueError("There is nothing to concatenate.")

    lines = []
    for path in paths:
        # The demuxer's escape for a literal quote inside a quoted string: end the
        # string, emit an escaped quote, reopen it.
        escaped = str(path.resolve()).replace("'", r"'\''")
        lines.append(f"file '{escaped}'")
    return "\n".join(lines) + "\n"


async def assemble(
    clip_paths: list[Path], audio_paths: list[Path], destination: Path
) -> Path:
    """Join the clips, join the audio, mux them, write ``destination``.

    Args:
        clip_paths: One per included page, in page order.
        audio_paths: One per included page, in the same order.
        destination: Where the finished ``story.mp4`` goes.

    Returns:
        ``destination``.

    Raises:
        ValueError: the two lists disagree in length, or either is empty. A
            mismatch means selection produced two different answers about which
            pages are in the video, which is a bug rather than a degradation.
        VideoGenerationError: ffmpeg failed.
    """
    if len(clip_paths) != len(audio_paths):
        raise ValueError(
            f"{len(clip_paths)} clips against {len(audio_paths)} audio files: "
            "selection disagreed with itself."
        )

    work = destination.parent
    video_list = work / "_concat-video.txt"
    audio_list = work / "_concat-audio.txt"
    video_list.write_text(build_concat_file(clip_paths), encoding="utf-8")
    audio_list.write_text(build_concat_file(audio_paths), encoding="utf-8")

    try:
        await run_ffmpeg(
            [
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(video_list),
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(audio_list),
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                # Deliberately no `-shortest`: the two streams are built to the
                # same measured durations, so a difference is a bug and
                # `-shortest` would hide it by truncating to the smaller.
                "-movflags",
                "+faststart",
                str(destination),
            ]
        )
    finally:
        # Removed even when ffmpeg fails: they are scratch, and leaving them in a
        # run directory beside the book invites reading them as artifacts.
        video_list.unlink(missing_ok=True)
        audio_list.unlink(missing_ok=True)

    return destination
