"""Render `story.json` from an existing run directory as a PDF.

    uv run python scripts/build_pdf.py outputs/20260802-202502-an-eagle-who-...

Separate from `write_one_story.py` so a layout change can be checked against
the runs already on disk without paying for another eleven model calls.

Untested glue, like `write_one_story.py`, and kept thin for that reason:
finding P is what an untested script costs when it fails at a live run.
"""

import sys
from pathlib import Path

from sparkstory.entities.stories import Story
from sparkstory.renderers import render_pdf


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2

    run_dir = Path(sys.argv[1])
    source = run_dir / "story.json"
    if not source.exists():
        print(f"no story.json in {run_dir}", file=sys.stderr)
        return 1

    # Validating rather than rendering the raw dict: on an old run this reports
    # schema drift immediately, instead of the renderer failing on a missing
    # key halfway down a page.
    story = Story.model_validate_json(source.read_text(encoding="utf-8"))

    out = run_dir / "story.pdf"
    render_pdf(story, out)
    print(f"wrote {out}  ({len(story.pages)} pages + title)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
