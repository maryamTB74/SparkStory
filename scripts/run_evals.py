"""Score finished books, either from past runs or from the fixture briefs.

Two modes, and the split is the point.

``--from-run`` scores books that already exist on disk. It costs nothing when
``--no-judge`` is passed, which makes it the cheap way to check that the metrics
detect defects known to be in a book before spending anything on a baseline.

``--fixtures`` generates a book per committed brief and then scores it. That is a
full pipeline run each, so it costs real calls, and it is what produces a baseline
two prompt versions can be compared across.

Usage:

    uv run python scripts/run_evals.py --from-run outputs --all --no-judge
    uv run python scripts/run_evals.py --from-run outputs/20260802-202502-an-eagle
    JUDGE_MODEL=grok-3-mini-critic uv run python scripts/run_evals.py --fixtures

Deliberately prints no total. Eight numbers per book and no ninth summarising
them: a single blended score would rank a book-wide defect equal to a local one,
and whatever is averaged becomes the thing a loop optimises.
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from pydantic import BaseModel

from sparkstory.config import settings
from sparkstory.entities.stories import Story, StoryBrief
from sparkstory.evals.briefs import load_fixture_briefs
from sparkstory.evals.metrics.judge import BookJudge
from sparkstory.evals.metrics.types import BookScorecard
from sparkstory.evals.scorecard import score_book
from sparkstory.models.get_model import get_chat_model
from sparkstory.observability.dataset import experiment_config, upload_fixture_briefs
from sparkstory.utils.logging_utils import configure_logging
from sparkstory.workflows.plan_outline import run_outline_pipeline
from sparkstory.workflows.write_story import run_story_pipeline

logger = logging.getLogger("run_evals")

#: Columns printed for the computed half, in report order.
_DET_COLUMNS = (
    ("openers", "distinct_opener_ratio", "{:.3f}"),
    ("q_end", "question_ending_ratio", "{:.3f}"),
    ("words/pg", "words_per_page", "{:.1f}"),
    ("beats/pg", "beats_per_page", "{:.3f}"),
    ("recite_b", "fact_recital_beats", "{}"),
    ("recite_p", "fact_recital_prose", "{}"),
)
_JUDGED_COLUMNS = (
    ("delight", "delight"),
    ("showing", "showing"),
    ("momentum", "momentum"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--from-run",
        type=Path,
        help="score an existing run directory, or a parent of them with --all",
    )
    source.add_argument(
        "--fixtures",
        action="store_true",
        help="generate a book per committed fixture brief, then score it",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="with --from-run, score every child directory holding a story.json",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="computed metrics only: free, offline, and ungameable",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=f"registry entry for the judge (default {settings.judge_model})",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs"),
        help="where --fixtures writes its runs",
    )
    parser.add_argument(
        "--opik",
        action="store_true",
        help="upload the fixture briefs to Opik and record the run there",
    )
    return parser.parse_args()


def read_notes(run_dir: Path, story: Story | None = None) -> list[str]:
    """Grounding ``story_note`` values for a run on disk.

    Prefers the notes carried by the finished book, because those are the ones the
    Writer was actually given: a fact that failed the ``write_story`` boundary check
    is in ``research-1.json`` and *not* in the book, so scoring prose against it
    would measure a note the Writer never saw.

    Falls back to the research artifacts, which is what every run produced before
    the outline carried its grounding -- including the five committed baseline books,
    whose scorecards must stay reproducible.

    Missing both means an ungrounded run, which returns ``[]`` and records ``None``
    for the recital metrics -- absence of a measurement rather than a clean score.
    """
    if story is not None and story.outline.grounding is not None:
        return [fact.story_note for fact in story.outline.grounding.facts]

    notes: list[str] = []
    for path in sorted(run_dir.glob("research*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("Could not parse %s, ignoring it", path)
            continue
        notes.extend(
            fact["story_note"]
            for fact in payload.get("facts", [])
            if "story_note" in fact
        )
    return notes


def make_judge(story: Story, model_id: str | None, disabled: bool) -> BookJudge | None:
    """A judge for this book, or None when judging is off."""
    if disabled:
        return None
    return BookJudge(get_chat_model(model_id or settings.judge_model), story=story)


async def score_run_dir(run_dir: Path, args: argparse.Namespace) -> BookScorecard:
    """Score a book already written to disk."""
    story = Story.model_validate_json(
        (run_dir / "story.json").read_text(encoding="utf-8")
    )
    card = await score_book(
        story,
        name=run_dir.name,
        notes=read_notes(run_dir, story),
        judge=make_judge(story, args.model, args.no_judge),
    )
    (run_dir / "scorecard.json").write_text(
        card.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    return card


async def generate_and_score(
    name: str, brief: StoryBrief, args: argparse.Namespace
) -> BookScorecard:
    """Run the real pipeline for one brief, then score what it produced."""
    run_dir = args.out_dir / f"eval-{name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # The callback now exists only to write the artifact. The *notes* come from
    # `story.outline.grounding` below, which is a stricter source: this callback
    # sees what research returned, while the outline carries what survived the
    # `write_story` boundary check -- and a fact dropped there never reached the
    # Writer, so scoring the book against it would measure a note the book was
    # never given.
    #
    # research-1.json is still written here, because the fact count is what decides
    # whether a grounded comparison meant anything (rule 27), and findings M and S
    # are both what happens when nobody checks it.
    def capture(task_name: str, value: object) -> None:
        if task_name != "research":
            return
        if isinstance(value, BaseModel):
            (run_dir / "research-1.json").write_text(
                value.model_dump_json(indent=2) + "\n", encoding="utf-8"
            )

    # Following the course, which tags its evaluation runs separately so a
    # measurement is distinguishable from a real book. The pipelines build their
    # own tracers; this records which fixture the spans underneath belong to.
    logger.info("eval run for fixture %r", name)

    outline = await run_outline_pipeline(brief, on_task_result=capture)
    story = await run_story_pipeline(brief, outline)

    (run_dir / "story.json").write_text(
        story.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "brief.json").write_text(
        brief.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )

    # Read from the finished book rather than accumulated during the run, so the
    # notes scored against are exactly the ones the Writer was given. `None`
    # grounding means research never ran, which yields no notes and records the
    # recital metrics as absent -- not as a clean score.
    grounding = story.outline.grounding
    notes = [fact.story_note for fact in grounding.facts] if grounding else []

    card = await score_book(
        story,
        name=name,
        notes=notes,
        judge=make_judge(story, args.model, args.no_judge),
    )
    (run_dir / "scorecard.json").write_text(
        card.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    return card


def print_table(cards: list[BookScorecard]) -> None:
    """One row per book, computed and judged columns kept visibly apart."""
    name_width = max([len(card.name[:38]) for card in cards] + [4])
    header = ["book".ljust(name_width), "pp"]
    header += [label.rjust(8) for label, _, _ in _DET_COLUMNS]
    header += ["|"] + [label.rjust(8) for label, _ in _JUDGED_COLUMNS]
    print("\n" + "  ".join(header))
    print("-" * (len(" ".join(header)) + 8))

    for card in cards:
        row = [card.name[:38].ljust(name_width), str(card.page_count).rjust(2)]
        for _, field, fmt in _DET_COLUMNS:
            value = getattr(card.deterministic, field)
            row.append(("-" if value is None else fmt.format(value)).rjust(8))
        row.append("|")
        for _, field in _JUDGED_COLUMNS:
            value = None if card.judged is None else getattr(card.judged, field)
            row.append(("-" if value is None else f"{value:.3f}").rjust(8))
        print("  ".join(row))

    for card in cards:
        if card.judge_error:
            print(f"\n  {card.name}: judge unavailable -- {card.judge_error}")
    # Stated rather than implied: a reader who sees nine columns and no total
    # should know the absence is deliberate.
    print("\nNo overall score, by design. Read the columns separately.")


async def main() -> int:
    args = parse_args()
    configure_logging()

    # Before anything is generated. An upload that fails after a fixture run has
    # paid for five books would be the expensive way to learn a key is missing,
    # and finding P is the worked example: the untested script died at the live
    # run, after the search had already been paid for.
    if args.opik:
        if not args.fixtures:
            print("--opik applies to --fixtures runs only")
            return 1
        try:
            count = upload_fixture_briefs()
        except RuntimeError as error:
            print(f"--opik: {error}")
            return 1
        print(f"uploaded {count} briefs to Opik")
        logger.info("experiment config: %s", experiment_config())

    targets: list[tuple[str, Path]] = []
    if args.fixtures:
        briefs = load_fixture_briefs()
    elif args.all:
        targets = [
            (path.parent.name, path.parent)
            for path in sorted(args.from_run.glob("*/story.json"))
        ]
        if not targets:
            print(f"no directory under {args.from_run} contains a story.json")
            return 1
    else:
        if not (args.from_run / "story.json").is_file():
            print(f"no story.json in {args.from_run}")
            return 1
        targets = [(args.from_run.name, args.from_run)]

    cards: list[BookScorecard] = []
    failures = 0

    if args.fixtures:
        for name, brief in briefs.items():
            # Per-book isolation, following the course: one brief that fails must
            # not cost the books that would have succeeded after it.
            try:
                cards.append(await generate_and_score(name, brief, args))
                print(f"  scored {name}", flush=True)
            except Exception:
                failures += 1
                logger.exception("Failed to generate or score %s", name)
    else:
        for name, run_dir in targets:
            try:
                cards.append(await score_run_dir(run_dir, args))
            except Exception:
                failures += 1
                logger.exception("Failed to score %s", name)

    if not cards:
        print("nothing scored")
        return 1

    print_table(cards)
    if failures:
        print(f"\n{failures} book(s) failed; see the log above.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
