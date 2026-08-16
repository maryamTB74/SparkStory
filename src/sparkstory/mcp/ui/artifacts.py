"""The guarded path from a URL segment to a file in a run directory.

This is the only place in the project that reads a file chosen by an HTTP
request, so it carries two independent guards and its own tests.

**Guard 1 -- an allowlist, not sanitisation.** A name must match one of a small
set of patterns for artifacts the pipelines actually write. Rejecting by pattern
rather than stripping ``../`` is the same call ``ChildId`` made: a value that
cannot express the attack beats code that remembers to check. It also keeps
``brief.json`` and ``meta.json`` unreachable, which matters because they hold a
child's name and a ``child_id``.

**Guard 2 -- resolved-path confinement.** Even an allowlisted name is confirmed
to resolve inside the run directory before anything is read, which catches a
symlink pointing elsewhere. Belt and braces, because the cost of being wrong here
is somebody else's child's illustrations.
"""

import re
from pathlib import Path

#: Artifacts the pipelines write and a parent may see. Deliberately excludes
#: `brief.json` (the child's name), `meta.json` (the `child_id`), `run.log` and
#: every numbered planning artifact.
ARTIFACT_PATTERN = re.compile(
    r"^(?:page-\d{2}\.(?:jpg|mp3)|portrait-[a-z0-9-]+\.jpg|story\.(?:mp3|mp4|pdf))$"
)

#: Set explicitly rather than guessed. An `.mp3` served as
#: `application/octet-stream` downloads instead of playing.
MEDIA_TYPES = {
    ".jpg": "image/jpeg",
    ".mp3": "audio/mpeg",
    ".mp4": "video/mp4",
    ".pdf": "application/pdf",
}


def resolve_artifact(run_directory: Path, name: str) -> Path | None:
    """The file this name refers to, or None if it may not be served.

    Returns None for a name outside the allowlist, a path that escapes the run
    directory, or a file that does not exist. One return value for all three
    because the caller's response is the same 404 either way: distinguishing
    "not allowed" from "not there" tells a prober which names are real.
    """
    if not ARTIFACT_PATTERN.match(name):
        return None

    candidate = run_directory / name
    try:
        resolved = candidate.resolve(strict=True)
        root = run_directory.resolve(strict=True)
    except OSError:
        # Missing file, broken symlink, or an unreadable directory.
        return None

    if not resolved.is_relative_to(root):
        return None
    if not resolved.is_file():
        return None
    return resolved


def available_media(run_directory: Path | None, page_count: int) -> dict[str, object]:
    """What media exists for this run, as names the book page can link.

    Absent files produce ``None`` rather than a placeholder: the page renders
    nothing for them, so a parent never sees a broken player. This never
    generates anything -- it reports what a CLI run already made.
    """
    pages: list[dict[str, object]] = []
    for number in range(1, page_count + 1):
        image = f"page-{number:02d}.jpg"
        audio = f"page-{number:02d}.mp3"
        pages.append(
            {
                "number": number,
                "image": image if _present(run_directory, image) else None,
                "audio": audio if _present(run_directory, audio) else None,
            }
        )

    return {
        "pages": pages,
        "story_audio": "story.mp3" if _present(run_directory, "story.mp3") else None,
        "video": "story.mp4" if _present(run_directory, "story.mp4") else None,
        "pdf": "story.pdf" if _present(run_directory, "story.pdf") else None,
    }


def _present(run_directory: Path | None, name: str) -> bool:
    """Whether this artifact is servable. Goes through the same guard as a GET."""
    if run_directory is None:
        return False
    return resolve_artifact(run_directory, name) is not None
