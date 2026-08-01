"""Run the story engine against the real API, print the book, and save the run.

A debugging entry point, not part of the server.

Why a script rather than the MCP server: the server speaks JSON-RPC over stdout,
so nothing may print to stdout and every message is wrapped in protocol framing.
For judging whether the prose is any good you want to read the book, and for
debugging a stage you want its raw output. Both are easier here.

Each run is saved under ``outputs/<timestamp>-<premise-slug>/``:

    brief.json       exactly what was asked for -- the run is reproducible
    meta.json        which model ran each stage, and how long it took
    outline.json     stage 1 output
    page_plan.json   stage 2 output
    story.json       the whole provenance chain
    story.md         the book, readable
    run.log          every log line, including the request_id and any traceback

Artifacts are written **as each stage finishes**, so a crash in the writer still
leaves the outline and page plan that led to it. That is the point: the run worth
inspecting is usually the one that failed.

``outputs/`` is gitignored. Every file holds a real child's name, which is
personal data about a minor, and these would bloat the repository. To keep one as
a sample, copy it somewhere tracked deliberately.

Examples::

    uv run python scripts/write_one_story.py
    uv run python scripts/write_one_story.py --stage plan          # cheapest
    uv run python scripts/write_one_story.py --stage plot          # + page plan
    uv run python scripts/write_one_story.py --debug               # log prompts
    uv run python scripts/write_one_story.py --level pre_reader --pages 6
    uv run python scripts/write_one_story.py --name Ada --age 8 \
        --level confident --premise "a girl who befriends a thunderstorm"

Logs go to stderr and the book to stdout, so the book can be saved on its own::

    uv run python scripts/write_one_story.py > story.txt
"""

import argparse
import asyncio
import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

from sparkstory.config import settings
from sparkstory.entities.stories import (
    ChildProfile,
    Pronouns,
    ReadingLevel,
    Story,
    StoryBrief,
    Tone,
)
from sparkstory.models.get_model import get_chat_model
from sparkstory.nodes.plot_planner import PlotPlannerNode
from sparkstory.nodes.story_planner import StoryPlannerNode
from sparkstory.utils.logging_utils import configure_logging
from sparkstory.workflows.write_story import run_story_pipeline

logger = logging.getLogger(__name__)

# Words per page above which text is probably too dense for the level. A smell
# test for reading the output, not a rule the code enforces -- judging length
# against reading level belongs to the reading-level rubric in a later session.
_DENSE_PAGE_WORDS = {
    ReadingLevel.PRE_READER: 30,
    ReadingLevel.EARLY_READER: 60,
    ReadingLevel.DEVELOPING: 100,
    ReadingLevel.CONFIDENT: 180,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="Maryam")
    parser.add_argument("--age", type=int, default=5)
    parser.add_argument(
        "--pronouns",
        default=Pronouns.SHE_HER.value,
        choices=[p.value for p in Pronouns],
    )
    parser.add_argument(
        "--level",
        default=ReadingLevel.EARLY_READER.value,
        choices=[level.value for level in ReadingLevel],
    )
    parser.add_argument(
        "--tone", default=Tone.MAGICAL.value, choices=[t.value for t in Tone]
    )
    parser.add_argument("--premise", default="a fox who wants to visit the moon")
    parser.add_argument("--pages", type=int, default=10)
    parser.add_argument("--interests", nargs="*", default=["foxes", "astronomy"])
    parser.add_argument("--must-include", nargs="*", default=["a paper rocket"])
    parser.add_argument("--avoid", nargs="*", default=["spiders", "the dark"])
    parser.add_argument(
        "--stage",
        choices=["plan", "plot", "all"],
        default="all",
        help="Stop after this stage. 'plan' is one model call; 'all' is three.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="DEBUG logging, which includes the rendered prompts.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs"),
        help="Where run directories are created (default: outputs/).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Print only; write nothing to disk.",
    )
    return parser.parse_args()


def build_brief(args: argparse.Namespace) -> StoryBrief:
    return StoryBrief(
        child=ChildProfile(
            name=args.name,
            age=args.age,
            pronouns=Pronouns(args.pronouns),
            reading_level=ReadingLevel(args.level),
            interests=args.interests,
        ),
        premise=args.premise,
        tone=Tone(args.tone),
        page_count=args.pages,
        must_include=args.must_include,
        avoid=args.avoid,
    )


def _slug(text: str, limit: int = 40) -> str:
    """A filesystem-safe fragment of the premise, for recognising runs by name."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:limit] or "story"


def make_run_dir(base: Path, brief: StoryBrief) -> Path:
    """Create ``base/<timestamp>-<slug>/``.

    Timestamp first so directories sort chronologically, and the premise after it
    so a run is recognisable without opening anything. The child's name is
    deliberately *not* in the path: directory names show up in shell history and
    screen shares, and the name is inside brief.json anyway.
    """
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = base / f"{stamp}-{_slug(brief.premise)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def attach_file_log(path: Path) -> logging.Handler:
    """Also write log records to ``path``.

    Added here rather than in ``configure_logging`` on purpose: the MCP server
    must not silently start writing files, and its log lifecycle is not a
    debugging run's. The formatter matches the console one so a saved log reads
    the same as the one you watched.
    """
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logging.getLogger().addHandler(handler)
    return handler


def build_meta(args: argparse.Namespace, started: float, **extra: object) -> dict:
    """Assemble meta.json.

    One builder rather than a literal at each exit point, so every run -- whether
    it finished, stopped at a stage, or failed -- records the same fields. The
    first version wrote a different subset at each exit, which made a stopped run
    indistinguishable from one that died before finishing.
    """
    return {
        "at": datetime.now().isoformat(timespec="seconds"),
        "stage_requested": args.stage,
        "models": {
            "planner": settings.planner_model,
            "plot": settings.plot_model,
            "writer": settings.writer_model,
        },
        "seconds": round(time.monotonic() - started, 1),
        "note": "the run's request_id appears in run.log",
        **extra,
    }


def save_json(run_dir: Path | None, name: str, payload: BaseModel | dict) -> None:
    """Write one artifact, if saving is enabled."""
    if run_dir is None:
        return
    text = (
        payload.model_dump_json(indent=2)
        if isinstance(payload, BaseModel)
        else json.dumps(payload, indent=2)
    )
    (run_dir / name).write_text(text + "\n", encoding="utf-8")


def story_markdown(story: Story, brief: StoryBrief) -> str:
    """Render the book as readable markdown.

    Lives in this script rather than in the package because it is a debugging
    convenience, not the product's rendering. When a session needs real book
    output, the course's reserved name for it is ``renderers.py``.
    """
    lines = [
        f"# {story.outline.title}",
        "",
        f"*{story.outline.logline}*",
        "",
        f"For {brief.child.name}, age {brief.child.age} "
        f"({brief.child.pronouns.value}) — {brief.child.reading_level.value}",
        "",
        "---",
        "",
    ]
    for scene, page in zip(story.page_plan.pages, story.pages, strict=True):
        lines += [
            f"### Page {page.page_number}",
            "",
            page.text,
            "",
            f"<!-- beat {scene.beat_position} · {scene.setting} · "
            f"{scene.visual_action} · inside: {scene.emotional_shift} -->",
            "",
        ]
    return "\n".join(lines)


async def main() -> int:
    args = parse_args()
    configure_logging()
    if args.debug:
        logging.getLogger("sparkstory").setLevel(logging.DEBUG)

    brief = build_brief(args)
    run_dir = None if args.no_save else make_run_dir(args.out_dir, brief)
    handler = attach_file_log(run_dir / "run.log") if run_dir else None
    started = time.monotonic()

    # Saved before any model call, so a run that fails immediately still records
    # what was asked for and which models were meant to answer.
    save_json(run_dir, "brief.json", brief)
    save_json(run_dir, "meta.json", build_meta(args, started, outcome="running"))

    try:
        return await run_stages(args, brief, run_dir, started)
    except Exception:
        # Logged rather than allowed to propagate, so the traceback lands in
        # run.log beside the artifacts of the stages that did succeed.
        logger.exception("Run failed")
        save_json(
            run_dir,
            "meta.json",
            build_meta(args, started, outcome="failed, see run.log"),
        )
        return 1
    finally:
        if handler is not None:
            logging.getLogger().removeHandler(handler)
            handler.close()
        if run_dir is not None:
            print(f"\nsaved to {run_dir}/")


async def run_stages(
    args: argparse.Namespace,
    brief: StoryBrief,
    run_dir: Path | None,
    started: float,
) -> int:
    """Run up to the requested stage, printing and saving as each completes."""
    outline = await StoryPlannerNode(
        model=get_chat_model(settings.planner_model), brief=brief
    ).ainvoke()
    save_json(run_dir, "outline.json", outline)

    print(f"\n{'=' * 66}\n  {outline.title}\n  {outline.logline}\n{'=' * 66}")
    print(f"\nTheme: {outline.theme}\n")
    for character in outline.characters:
        print(f"  {character.name} ({character.role}): {character.description}")
    print(f"\nBeats ({len(outline.beats)} for {brief.page_count} pages):")
    for beat in outline.beats:
        print(f"  {beat.position}. [{beat.function.value}] {beat.title}")
        print(f"     {beat.summary}")

    if args.stage == "plan":
        save_json(
            run_dir,
            "meta.json",
            build_meta(
                args, started, outcome="ok", beats=len(outline.beats), stage_run="plan"
            ),
        )
        return 0

    plan = await PlotPlannerNode(
        model=get_chat_model(settings.plot_model), brief=brief, outline=outline
    ).ainvoke()
    save_json(run_dir, "page_plan.json", plan)

    print(f"\n{'-' * 66}\nPage plan ({len(plan.pages)} pages)\n{'-' * 66}")
    for page in plan.pages:
        print(f"  p{page.page_number} (beat {page.beat_position}) {page.setting}")
        print(f"     shows:  {page.visual_action}")
        print(f"     inside: {page.emotional_shift}")
        if page.page_turn_hook:
            print(f"     hook:   {page.page_turn_hook}")

    if args.stage == "plot":
        save_json(
            run_dir,
            "meta.json",
            build_meta(
                args,
                started,
                outcome="ok",
                stage_run="plot",
                beats=len(outline.beats),
                pages=len(plan.pages),
            ),
        )
        return 0

    # The full pipeline re-runs planning. Wasteful, and deliberately so: this path
    # must exercise the workflow exactly as the MCP tool does, and stitching a
    # half-finished run into it would be testing something else.
    story = await run_story_pipeline(brief)
    save_json(run_dir, "story.json", story)
    if run_dir is not None:
        (run_dir / "story.md").write_text(
            story_markdown(story, brief), encoding="utf-8"
        )

    print(f"\n{'=' * 66}\n  {story.outline.title}\n{'=' * 66}")
    for scene, page in zip(story.page_plan.pages, story.pages, strict=True):
        print(f"\n-- page {page.page_number}  ({scene.setting})")
        print(f"   {page.text}")

    words = [len(page.text.split()) for page in story.pages]
    limit = _DENSE_PAGE_WORDS[brief.child.reading_level]
    dense = [
        page.page_number
        for page, count in zip(story.pages, words, strict=True)
        if count > limit
    ]
    print(f"\n{'=' * 66}")
    print(f"total words: {sum(words)}   longest page: {max(words)}")
    level = brief.child.reading_level.value
    print(f"pages over {limit} words for {level}: {dense or 'none'}")

    # Nothing enforces `avoid` yet -- the writer is only asked. Until the safety
    # rubric exists, checking by eye is the whole of the check.
    hits = [
        page.page_number
        for page in story.pages
        if any(term.lower() in page.text.lower() for term in brief.avoid)
    ]
    print(f"pages with an 'avoid' term (unenforced, check by eye): {hits or 'none'}")

    save_json(
        run_dir,
        "meta.json",
        build_meta(
            args,
            started,
            outcome="ok",
            stage_run="all",
            beats=len(story.outline.beats),
            pages=len(story.pages),
            total_words=sum(words),
            longest_page_words=max(words),
            pages_over_level_limit=dense,
            pages_with_avoid_term=hits,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
