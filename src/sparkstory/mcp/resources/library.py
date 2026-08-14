"""Read-only introspection: what this server has made, and what it knows.

**Everything returned here is prompt text** (non-obvious rule 1). A resource is
read by a *client's model*, not by a developer tailing a log, so these are
written as answers rather than as dumps.

That constraint does real work in ``read_library``. A run directory is named
after the premise -- ``20260811-231338-a-fox-who-wants-to-visit-the-moon`` --
``brief.json`` holds the child's name, and ``meta.json`` holds a ``child_id``.
None of it may leave: the run id is the timestamp alone, the title comes from the
story rather than the directory, and no brief is ever opened. Session 12 recorded
the same hazard one door over, when Opik uploaded briefs carrying a child's name.

**Nothing here writes, takes a parameter, or reads ``data/``.** ``data/`` is real
persistence and holds per-child memory; a read-only endpoint over it is a privacy
decision nobody has made.
"""

import json
import re
from pathlib import Path

from sparkstory.config import _PROJECT_ROOT, settings

#: Where finished runs land. Disposable by design, which is exactly why the
#: library is computed on read rather than stored anywhere.
_OUTPUTS = _PROJECT_ROOT / "outputs"

#: The committed fact corpus. Read from disk rather than queried from Postgres:
#: a resource that needed a live database would fail on a fresh checkout and
#: would drag a service into the offline test suite.
_CORPUS = _PROJECT_ROOT / "corpus"

#: A generated run directory is `YYYYMMDD-HHMMSS-<premise slug>`. Matched
#: strictly rather than by counting hyphens: the committed baseline directories
#: are named `eval-eagle-planet`, and "keep the first two parts" turned that into
#: `eval-eagle` -- truncating an arbitrary name, which can emit *half a premise*
#: while looking like an id. Found by reading a live stdio run, not by a test.
_RUN_ID = re.compile(r"^(\d{8}-\d{6})-.+$")


def _run_id(directory_name: str) -> str:
    """A run's id, with the premise slug removed if there is one."""
    matched = _RUN_ID.match(directory_name)
    return matched.group(1) if matched else directory_name


def read_library(outputs_root: Path | None = None) -> str:
    """List the finished books on disk.

    Args:
        outputs_root: Where to look. Injected so tests build their own runs
            rather than depending on whatever was generated last.
    """
    root = _OUTPUTS if outputs_root is None else outputs_root
    if not root.is_dir():
        return "No books have been made yet."

    lines: list[str] = []
    for directory in sorted(root.iterdir()):
        story_file = directory / "story.json"
        if not story_file.is_file():
            # A `--stage plan` run has no book. Skipping beats raising: one
            # incomplete directory must not break the whole listing.
            continue

        try:
            story = json.loads(story_file.read_text())
        except OSError, json.JSONDecodeError:
            continue

        title = story.get("outline", {}).get("title", "untitled")
        pages = len(story.get("pages", []))
        extras = [
            name
            for name, present in (
                ("PDF", (directory / "story.pdf").is_file()),
                ("audio", (directory / "story.mp3").is_file()),
            )
            if present
        ]
        lines.append(
            f"- {_run_id(directory.name)} | {title} | {pages} pages"
            + (f" | {', '.join(extras)}" if extras else "")
        )

    if not lines:
        return "No books have been made yet."
    return "Books made by this server:\n" + "\n".join(lines)


def read_corpus(corpus_root: Path | None = None) -> str:
    """Report what the retrieval corpus contains.

    Finding 27 is why this exists: *check the fact count before comparing two
    grounded runs*, because a run that retrieves nothing renders identically in
    both world-rule modes and still reads as a successful control.
    """
    root = _CORPUS if corpus_root is None else corpus_root
    if not root.is_dir():
        return "No corpus is present."

    files = sorted(p for p in root.rglob("*.md") if p.name != "README.md")
    # A chunk is a non-empty paragraph, which is what `ingest.py` splits on. An
    # approximation stated as one, rather than a number implying a live index.
    chunks = sum(
        len([b for b in p.read_text().split("\n\n") if b.strip()]) for p in files
    )

    return (
        f"Retrieval corpus: {len(files)} files, roughly {chunks} chunks.\n"
        f"Files: {', '.join(p.stem for p in files)}\n"
        f"Embedding model: {settings.embedding_model}"
    )
