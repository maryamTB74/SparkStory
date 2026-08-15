"""Write one pre-filled label file per book, ready for a human to fill in.

Run once before labelling::

    uv run python scripts/make_label_skeletons.py \\
        --books tests/fixtures/evals/baseline/2026-08-04 \\
        --out tests/fixtures/evals/labels/2026-08-13

Every score comes out ``null``; the page text is inlined as ``_text`` so
labelling is typing digits rather than transcribing prose.

Refuses to overwrite an existing file, because that file may hold hours of
labelling and there is no way to get them back.
"""

import argparse
import json
import sys
from pathlib import Path

from sparkstory.entities.stories import Story
from sparkstory.evals.labels import skeleton


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--books",
        type=Path,
        required=True,
        help="a directory of <book>/story.json",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--labeller", default="maryam")
    parser.add_argument(
        "--pass-number",
        type=int,
        default=1,
        help="2 for the blind relabel that measures the human ceiling",
    )
    parser.add_argument(
        "--only",
        default=None,
        help="one book name, for generating a single pass-2 skeleton",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.pass_number == 1 else ".pass2"
    written = 0

    for story_path in sorted(args.books.glob("*/story.json")):
        name = story_path.parent.name
        if args.only and name != args.only:
            continue

        target = args.out / f"{name}{suffix}.json"
        if target.exists():
            print(f"  skipping {name}: {target} exists")
            continue

        story = Story.model_validate_json(story_path.read_text(encoding="utf-8"))
        target.write_text(
            json.dumps(
                skeleton(
                    story,
                    book=name,
                    labeller=args.labeller,
                    pass_number=args.pass_number,
                ),
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"  wrote {target} ({len(story.pages)} pages)")
        written += 1

    if not written:
        print(f"nothing written; no <book>/story.json under {args.books}")
        return 1

    print(f"\n{written} skeleton(s). Fill in 0 or 1 for every score.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
