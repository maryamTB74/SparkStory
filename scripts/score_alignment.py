"""Measure how often the judge agrees with a human, per page and per dimension.

    uv run python scripts/score_alignment.py \\
        --labels tests/fixtures/evals/labels/2026-08-13 \\
        --scorecards tests/fixtures/evals/baseline/2026-08-13

Reports nothing that gates anything. A low number here is a result, not a failure.

Read the pooled row against the ceiling rather than against 1.0. A judge matching
what one person scores against themselves is performing at human level, and the
honest conclusion in that case is that the question is fuzzy -- not that the
judge is bad.
"""

import argparse
import json
import sys
from pathlib import Path

from sparkstory.evals.alignment import AlignmentScores, agreement, from_judge, pooled
from sparkstory.evals.labels import is_complete, load_labels
from sparkstory.evals.metrics.types import BookScorecard

_HEADER = f"{'book':<24} {'delight':>8} {'showing':>9} {'momentum':>9} {'n':>5}"


def _row(name: str, scores: AlignmentScores) -> str:
    return (
        f"{name:<24} {scores.delight:>8.3f} {scores.showing:>9.3f} "
        f"{scores.momentum:>9.3f} {scores.n:>5}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--scorecards", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--date", default="latest", help="stamped into the output filename"
    )
    args = parser.parse_args()

    # Pass-2 files are relabels, compared against pass 1 rather than the judge.
    label_files = sorted(
        path for path in args.labels.glob("*.json") if not path.stem.endswith(".pass2")
    )
    if not label_files:
        print(f"no label files in {args.labels}")
        return 1

    per_book: list[tuple[str, AlignmentScores]] = []
    ceiling: list[tuple[str, AlignmentScores]] = []
    skipped = 0

    for path in label_files:
        labels = load_labels(path)

        relabel = args.labels / f"{labels.book}.pass2.json"
        if relabel.is_file():
            # An unstarted skeleton is skipped rather than scored or raised on: the
            # ceiling is optional, and a blank relabel must not block a report about
            # the judge.
            second = load_labels(relabel)
            if is_complete(second):
                ceiling.append((labels.book, agreement(labels, second)))
            else:
                print(f"  {labels.book}: pass-2 relabel is not finished; no ceiling")

        card_path = args.scorecards / f"{labels.book}.json"
        if not card_path.is_file():
            print(f"  no scorecard for {labels.book} at {card_path}")
            skipped += 1
            continue

        card = BookScorecard.model_validate_json(card_path.read_text(encoding="utf-8"))
        if card.judged_pages is None:
            print(
                f"  {labels.book}: scorecard has no per-page verdicts. "
                "Re-judge it with run_evals.py before scoring alignment."
            )
            skipped += 1
            continue

        per_book.append(
            (
                labels.book,
                agreement(labels, from_judge(card.judged_pages, book=labels.book)),
            )
        )

    if not per_book:
        print("nothing to compare")
        return 1

    print("\nJudge alignment — agreement with the human label, per page\n")
    print(_HEADER)
    print("-" * len(_HEADER))
    for name, scores in per_book:
        print(_row(name, scores))

    overall = pooled([scores for _, scores in per_book])
    print("-" * len(_HEADER))
    print(_row("POOLED", overall))

    # Chance-corrected, and printed directly under the raw figures because the raw
    # ones are misleading on their own: 0.475 reads as partial agreement and can be
    # worse than two biased coins.
    print("\nAbove chance (Cohen's kappa) — read this before the rows above\n")
    print(
        f"{'':24} " + "  ".join(f"{d:>9}" for d in ("delight", "showing", "momentum"))
    )
    print(
        f"{'kappa':<24} "
        + "  ".join(
            f"{overall.kappa[d]:>+9.3f}" for d in ("delight", "showing", "momentum")
        )
    )
    print(
        f"{'said 1 (human/judge)':<24} "
        + "  ".join(
            f"{overall.ones[d][0]:>4}/{overall.ones[d][1]:<4}"
            for d in ("delight", "showing", "momentum")
        )
    )
    if all(abs(overall.kappa[d]) < 0.2 for d in overall.kappa):
        print(
            "\n  Every kappa is near zero: the two labellers agree no better than "
            "chance.\n  Do not read the raw percentages as partial agreement."
        )

    if ceiling:
        print("\nHuman ceiling — the same person, relabelling blind\n")
        print(_HEADER)
        print("-" * len(_HEADER))
        for name, scores in ceiling:
            print(_row(name, scores))
        print(
            "\nRead the pooled row against this, not against 1.0: a judge matching "
            "the ceiling is performing at human level."
        )
    else:
        print(
            "\nNo pass-2 relabel found, so there is no ceiling. A low number above "
            "cannot yet be attributed between a disagreeing judge and a question "
            "that has no stable answer."
        )

    if overall.disagreements:
        print(f"\nDisagreements ({len(overall.disagreements)}):")
        for line in overall.disagreements:
            print(f"  {line}")
        print(
            "\nRead the pages behind these before concluding who was right. "
            "There is precedent for the hand label being the thing that was wrong."
        )

    if skipped:
        print(f"\n{skipped} book(s) skipped; see above.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    target = args.out_dir / f"alignment-{args.date}.json"
    target.write_text(
        json.dumps(
            {
                "pooled": overall.model_dump(),
                "per_book": {name: scores.model_dump() for name, scores in per_book},
                "ceiling": {name: scores.model_dump() for name, scores in ceiling},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
