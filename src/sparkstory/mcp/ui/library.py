"""Reading finished books off disk, so the browser can reach them.

**Why this exists, written after it was nearly not built.** Spec section 5.2 renders
a book's pictures and audio, but reached the run directory through a *job* -- and a
job dies with the server process. The first live session showed what that means: every
already-paid-for illustration and recording on the machine belonged to a run older than
any live job, so the media feature was unreachable in normal use. A display feature
nothing can navigate to is not a feature.

**This reads and never writes.** A job is still the only thing that generates anything,
which keeps the tamper-resistance argument in section 5.3 untouched: nothing here can
approve, revise or spend money.

**What crosses to the browser, and what must not.** A run directory is named after the
premise, `brief.json` holds a child's name and `meta.json` holds a `child_id`. The
listing therefore carries the *story's own title* and the timestamp, and never the
directory name -- the same rule `resources/library.py` follows for the MCP resource.
The standing exception applies: a title is authored output and often contains the
child's name, which crosses deliberately. A brief never does.
"""

import json
import re
from pathlib import Path

from pydantic import ValidationError

from sparkstory.entities.stories import Story
from sparkstory.mcp.ui.artifacts import resolve_artifact
from sparkstory.utils.logging_utils import get_logger

logger = get_logger(__name__)

#: Two shapes, both matched strictly rather than sanitised. `YYYYMMDD-HHMMSS-<slug>`
#: is a generated run; `eval-<name>` is a committed baseline directory. This is a full
#: match rather than a prefix check because truncating an arbitrary name can emit half
#: a premise while looking like an id.
_DIRECTORY = re.compile(r"^(?:\d{8}-\d{6}-[a-z0-9-]+|eval-[a-z0-9-]+)$")

#: What a *URL* may carry. A directory name is `<timestamp>-<premise slug>`, and the
#: premise is a parent's words about their child -- so the public id is the timestamp
#: alone, which is unique across every run on disk. `eval-*` baselines carry no
#: premise and are addressed by name.
#:
#: This was caught by a test rather than by design: the first version put the whole
#: directory name in the href, which put the premise in the browser's address bar,
#: its history, and any shared link. The same leak, one door over.
_PUBLIC_ID = re.compile(r"^(?:\d{8}-\d{6}|eval-[a-z0-9-]+)$")

#: Shown in the listing so a parent can see at a glance what a book has.
_MEDIA_MARKERS = {
    "has_images": "page-01.jpg",
    "has_audio": "story.mp3",
    "has_video": "story.mp4",
    "has_pdf": "story.pdf",
}


def list_books(outputs_root: Path) -> list[dict[str, object]]:
    """Every finished book on disk, newest first.

    A directory with no ``story.json`` is skipped rather than raising: a
    ``--stage plan`` run has no book, and one incomplete directory must not break
    the whole listing.
    """
    if not outputs_root.is_dir():
        return []

    books: list[dict[str, object]] = []
    # Newest first, with the committed baselines last. Plain reverse-sorting the
    # names puts `eval-*` at the top, because "e" sorts above a digit -- and a
    # baseline has no date to be newest by.
    directories = sorted(
        (d for d in outputs_root.iterdir() if d.is_dir() and _DIRECTORY.match(d.name)),
        key=lambda d: (d.name.startswith("eval-"), _sort_key(d.name)),
    )
    for directory in directories:
        story_file = directory / "story.json"
        if not story_file.is_file():
            continue

        try:
            payload = json.loads(story_file.read_text())
        except OSError, json.JSONDecodeError:
            logger.warning("skipping unreadable story: %s", directory.name)
            continue

        outline = payload.get("outline", {})
        book: dict[str, object] = {
            # The timestamp alone, never the full directory name -- the slug is the
            # premise, and a URL is the one place it would be most visible.
            "run_id": _public_id(directory.name),
            # The story's own title, never the directory name -- which is the
            # premise, and the premise is the parent's words about their child.
            "title": outline.get("title", "Untitled"),
            "made": _readable_date(directory.name),
            "pages": len(payload.get("pages", [])),
        }
        for key, marker in _MEDIA_MARKERS.items():
            book[key] = (directory / marker).is_file()
        books.append(book)

    return books


def _directory_for(outputs_root: Path, public_id: str) -> Path | None:
    """The run directory a public id refers to, or None if it may not be read.

    A public id is a timestamp, so this searches for the directory beginning with
    it rather than joining it as a path. **The URL segment therefore never becomes
    a path component**, which is a stronger guarantee than sanitising one: there is
    no string a caller can send that reaches the filesystem unmatched.
    """
    if not _PUBLIC_ID.match(public_id):
        return None
    if not outputs_root.is_dir():
        return None

    for directory in outputs_root.iterdir():
        if not directory.is_dir() or not _DIRECTORY.match(directory.name):
            continue
        if _public_id(directory.name) != public_id:
            continue

        try:
            resolved = directory.resolve(strict=True)
            root = outputs_root.resolve(strict=True)
        except OSError:
            return None
        # Belt and braces, exactly as `resolve_artifact` does: a matched directory
        # could still be a symlink pointing somewhere else.
        if not resolved.is_relative_to(root) or resolved == root:
            return None
        return resolved
    return None


def load_book(outputs_root: Path, run_id: str) -> tuple[Story, Path] | None:
    """One book and its directory, or None if it may not be read.

    Returns None for a run id outside the pattern, a path escaping ``outputs/``, a
    directory with no book, or a book that will not parse. One return value for all
    four because the caller's response is the same 404: distinguishing "not allowed"
    from "not there" tells a prober which names are real.
    """
    resolved = _directory_for(outputs_root, run_id)
    if resolved is None:
        return None

    story_file = resolved / "story.json"
    if not story_file.is_file():
        return None

    try:
        story = Story.model_validate(json.loads(story_file.read_text()))
    except OSError, json.JSONDecodeError, ValidationError:
        logger.warning("could not read a book: %s", run_id)
        return None

    return story, resolved


def resolve_library_artifact(outputs_root: Path, run_id: str, name: str) -> Path | None:
    """An artifact inside a library run, behind both guards.

    The run id goes through ``load_book``'s pattern and confinement checks, and the
    file name goes through ``resolve_artifact``'s allowlist -- so ``brief.json`` and
    ``meta.json`` stay unreachable here exactly as they are on the job route.
    """
    resolved = _directory_for(outputs_root, run_id)
    if resolved is None:
        return None

    return resolve_artifact(resolved, name)


def _public_id(directory_name: str) -> str:
    """The part of a directory name a URL may carry -- the timestamp, not the slug."""
    matched = re.match(r"^(\d{8}-\d{6})-", directory_name)
    return matched.group(1) if matched else directory_name


def _sort_key(run_id: str) -> str:
    """Newest first among dated runs; alphabetical among baselines."""
    matched = re.match(r"^(\d{8}-\d{6})", run_id)
    if matched is None:
        return run_id
    # Inverted so a plain ascending sort puts the newest timestamp first, which
    # keeps the two-part key above readable as "baselines last, then newest".
    return "".join(
        chr(ord("9") - int(c)) if c.isdigit() else c for c in matched.group(1)
    )


def _readable_date(run_id: str) -> str:
    """The date part of a run id, or the id itself for a baseline directory."""
    matched = re.match(r"^(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})", run_id)
    if matched is None:
        return "committed baseline"
    year, month, day, hour, minute = matched.groups()
    return f"{year}-{month}-{day} {hour}:{minute}"
