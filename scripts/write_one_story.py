"""Run the story engine against the real API, print the book, and save the run.

A debugging entry point, not part of the server.

Why a script rather than the MCP server: the server speaks JSON-RPC over stdout,
so nothing may print to stdout and every message is wrapped in protocol framing.
For judging whether the prose is any good you want to read the book, and for
debugging a stage you want its raw output. Both are easier here.

Each run is saved under ``outputs/<timestamp>-<premise-slug>/``:

    brief.json          exactly what was asked for -- the run is reproducible
    meta.json           which model ran each stage, and how long it took
    research-N.json     what research found, after unprovenanced items were dropped
    plan_outline-N.json every outline draft, in order
    critique_outline-N  what the critic said about each one
    plan_pages-N.json   the page plan the book was written from
    write_prose-N.json  every prose draft
    critique_prose-N    what the critic said about each one
    story.json          the whole provenance chain, with the drafts that survived
    story.md            the book, readable
    run.log             every log line, including the request_id and any traceback

Artifacts are written **as each task finishes**, so a crash in the writer still
leaves the outline and page plan that led to it. That is the point: the run worth
inspecting is usually the one that failed.

**One plan per run.** The numbered files are the only plans, and they are the
ones the book was built from. There used to be an `outline.json` and a
`page_plan.json` written by this script's *own* planning calls, while the book
came from the workflow's -- so the two disagreed, and comparing prose against the
wrong one invented character-name bugs that did not exist. Twice, in two
different sessions. `page_plan.json` now appears only under ``--stage plot``,
where the run stops before any competing plan exists.

``outputs/`` is gitignored. Every file holds a real child's name, which is
personal data about a minor, and these would bloat the repository. To keep one as
a sample, copy it somewhere tracked deliberately.

Examples::

    uv run python scripts/write_one_story.py
    uv run python scripts/write_one_story.py --stage research      # cheapest
    uv run python scripts/write_one_story.py --stage plan          # + outline loop
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
from uuid import uuid4

from pydantic import BaseModel

from sparkstory.config import settings
from sparkstory.entities.illustration import ArtStatus
from sparkstory.entities.narration import NarrationStatus
from sparkstory.entities.stories import (
    ChildProfile,
    Pronouns,
    ReadingLevel,
    Story,
    StoryBrief,
    StoryOutline,
    Tone,
    Voice,
    WorldRules,
)
from sparkstory.models.get_model import get_chat_model
from sparkstory.nodes.plot_planner import PlotPlannerNode
from sparkstory.nodes.researcher import ResearcherNode
from sparkstory.renderers import render_pdf
from sparkstory.retrieval.provenance import drop_unprovenanced
from sparkstory.utils.logging_utils import configure_logging
from sparkstory.workflows.illustrate import run_illustration_pipeline
from sparkstory.workflows.narrate import run_narration_pipeline
from sparkstory.workflows.plan_outline import (
    build_research_context,
    run_outline_pipeline,
)
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
    parser.add_argument(
        "--world-rules",
        default=WorldRules.IMAGINATIVE.value,
        choices=[rule.value for rule in WorldRules],
        help=(
            "How far the story's world must obey the real one. 'realistic' makes "
            "every retrieved fact binding; 'imaginative' makes them detail the "
            "premise may break."
        ),
    )
    parser.add_argument(
        "--child-id",
        default=None,
        help=(
            "Stable id for this child, so their stories remember each other. "
            "Omit it and the run reads and writes no memory at all, which is the "
            "behaviour every run before this flag existed had."
        ),
    )
    parser.add_argument("--premise", default="a fox who wants to visit the moon")
    parser.add_argument("--pages", type=int, default=10)
    parser.add_argument("--interests", nargs="*", default=["foxes", "astronomy"])
    parser.add_argument("--must-include", nargs="*", default=["a paper rocket"])
    parser.add_argument("--avoid", nargs="*", default=["spiders", "the dark"])
    parser.add_argument(
        "--stage",
        choices=["research", "plan", "plot", "all"],
        default="all",
        help=(
            "Stop after this stage. 'research' is the cheapest and needs no "
            "planning; 'plan' runs the outline loop; 'all' writes the book."
        ),
    )
    parser.add_argument(
        "--max-web-searches",
        type=int,
        default=None,
        help=(
            "Override MAX_WEB_SEARCHES for this run. 0 disables the web tool "
            "entirely, which is the default."
        ),
    )
    parser.add_argument(
        "--no-verify-web",
        action="store_true",
        help=(
            "Skip fetching each cited page. Leaves every web URL unchecked, so "
            "the facts citing them are dropped as unprovenanced -- useful only "
            "for seeing what search returned before verification."
        ),
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
    parser.add_argument(
        "--pdf",
        action="store_true",
        help="Also write story.pdf beside story.md. Ignored under --no-save.",
    )
    parser.add_argument(
        "--illustrate",
        action="store_true",
        help=(
            "Draw a reference portrait per character, then one picture per page, "
            "into the run directory. The most expensive flag here by a wide "
            "margin -- roughly one image per page plus one per character. "
            "Implies --pdf, since the pictures exist to go in the book. Ignored "
            "under --no-save, because the images need somewhere to live."
        ),
    )
    parser.add_argument(
        "--narrate",
        action="store_true",
        help=(
            "Read the finished book aloud: one MP3 per page plus a stitched "
            "story.mp3, into the run directory. Ignored under --no-save, because "
            "the audio needs somewhere to live."
        ),
    )
    parser.add_argument(
        "--voice",
        choices=[v.value for v in Voice],
        default=Voice.FEMALE.value,
        help=(
            "Which voice reads the story aloud. Recorded in meta.json, without "
            "which comparing two voices is an anecdote rather than evidence."
        ),
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
            child_id=args.child_id,
        ),
        premise=args.premise,
        tone=Tone(args.tone),
        world_rules=WorldRules(args.world_rules),
        voice=Voice(args.voice),
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


def grounding_meta(outline: StoryOutline) -> dict:
    """What research the book was actually built from, for `meta.json`.

    The count is here for a specific reason: non-obvious rule 27 says to check the
    fact count *before* comparing two runs, because a run that retrieved nothing
    renders identically in both world-rule modes -- so a comparison against it is
    vacuous while still looking like a successful control. Findings M and S are both
    that mistake, one session apart. Recording it beside `world_rules` means the two
    fields that decide whether an A/B means anything sit in the same file.

    `chunk_ids` rather than the notes themselves: this is an audit trail, and the
    notes are already in `research-1.json` and in the outline.

    Distinguishes three states, which a bare count could not: `null` means research
    never ran (`MAX_RESEARCH_STEPS=0`), `0` means it ran and found nothing, and a
    number means it found something.
    """
    if outline.grounding is None:
        return {"facts": None, "chunk_ids": None}
    return {
        "facts": len(outline.grounding.facts),
        "chunk_ids": [fact.chunk_id for fact in outline.grounding.facts],
    }


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
        # Not decoration. An A/B is two runs differing in one field, and a run
        # whose artifacts do not say which mode produced them is not evidence --
        # this project has already lost debugging time to reading the wrong
        # artifact (finding E).
        "world_rules": args.world_rules,
        # Same argument as world_rules: a second book for the same child is only
        # evidence about memory if the artifact says which child it was for. None
        # here means the run read and wrote nothing, which must stay
        # distinguishable from a child whose memory happened to be empty.
        "child_id": args.child_id,
        # And the same argument again, for the same reason `--world-rules` was
        # recorded in Session 9: comparing two voices is only evidence if each
        # run's artifacts say which voice produced it. Recorded even when
        # --narrate is off, so a run that could have been narrated and was not
        # stays distinguishable from one narrated in the default voice.
        "voice": args.voice,
        "narrated": args.narrate,
        "models": {
            "researcher": settings.researcher_model,
            "embedder": settings.embedding_model,
            "planner": settings.planner_model,
            "plot": settings.plot_model,
            "writer": settings.writer_model,
            "memory_extractor": settings.memory_extractor_model,
            "narrator": settings.narrator_model,
        },
        "max_research_steps": settings.max_research_steps,
        # Read from settings rather than from args, so an override on the command
        # line and one in .env are recorded identically -- a run's artifacts have
        # to say what actually happened, not what was typed.
        "max_web_searches": settings.max_web_searches,
        "verify_web_claims": settings.verify_web_claims,
        "seconds": round(time.monotonic() - started, 1),
        "note": "the run's request_id appears in run.log",
        **extra,
    }


def _as_jsonable(value: object) -> object:
    """Last-resort encoder for values json does not know.

    Pydantic models nested inside a plain dict, chiefly. Raises for anything
    else rather than stringifying it, because a silently str()-ed object in an
    artifact reads as data and is not.
    """
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    # `StoryArt` carries `Path` values, and a dict of models containing paths is
    # the shape the illustration stage streams. Finding P is why this is here
    # rather than discovered live: the same class of bug -- a value json cannot
    # encode -- killed a run *after* the search had been paid for, and images
    # cost considerably more than a search.
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialise {type(value).__name__}")


def save_json(run_dir: Path | None, name: str, payload: BaseModel | dict) -> None:
    """Write one artifact, if saving is enabled."""
    if run_dir is None:
        return
    text = (
        payload.model_dump_json(indent=2)
        if isinstance(payload, BaseModel)
        # `default=` rather than a bare dumps: a dict whose *values* are models
        # -- web_sources.json is `{"sources": [WebSource, ...]}` -- is not
        # serialisable otherwise, and the failure lands at the live run because
        # nothing here is unit tested.
        else json.dumps(payload, indent=2, default=_as_jsonable)
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

    # Applied to the live settings object before anything reads it, so the flags
    # behave exactly as the environment variables would. Mutating settings is
    # acceptable here and nowhere else: this script *is* the operator, and the
    # alternative is threading two more arguments through every stage.
    if args.max_web_searches is not None:
        settings.max_web_searches = args.max_web_searches
    if args.no_verify_web:
        settings.verify_web_claims = False

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
    # One id for the whole run, minted here rather than inside either pipeline.
    # `--stage all` runs two entrypoints, and left to themselves they mint one id
    # each -- correct for the MCP path, where they are separate tool calls that
    # may be minutes apart, but it means one book produces two traces that
    # nothing can join. This script is a single operator action, so it knows the
    # two stages belong together and the pipelines cannot.
    request_id = str(uuid4())

    # One numbered file per completed task, shared by both pipelines so the
    # numbering runs straight through a run. The revision loops live inside the
    # entrypoints, so a returned Story shows only the drafts that survived --
    # `critique_outline-1.json` holding an empty review list is the single most
    # useful fact in a run, because it says the critic approved on the first
    # pass, and that is what validates or kills the MAX_*_REVISIONS default of 2.
    iterations: dict[str, int] = {}

    def save_iteration(task_name: str, value: object) -> None:
        index = iterations[task_name] = iterations.get(task_name, 0) + 1
        if isinstance(value, BaseModel):
            save_json(run_dir, f"{task_name}-{index}.json", value)

    if args.stage == "research":
        # Called directly, exactly as `--stage plot` calls the Plot Planner: this
        # path stops before any competing artifact exists, so there is no second
        # version of anything to confuse it with. The full run gets its grounding
        # from the workflow as `research-1.json` instead.
        agent, store, ledger = build_research_context()
        grounding = await ResearcherNode(
            agent=agent, brief=brief, max_steps=settings.max_research_steps
        ).ainvoke()
        kept = drop_unprovenanced(grounding, store, ledger=ledger)
        save_json(run_dir, "research.json", kept)
        if ledger is not None:
            # What the web was asked and what came back, including sources that
            # failed verification -- a run whose artifacts do not record what
            # happened is not evidence (Session 9, finding M).
            save_json(run_dir, "web_sources.json", {"sources": ledger.sources})

        print(f"\n{'=' * 66}\n  Research\n{'=' * 66}")
        print(f"\nfacts ({len(kept.facts)}):")
        for item in kept.facts:
            print(f"  - {item.story_note}")
            print(f"      from {item.chunk_id} ({item.source})")
            print(f"      claim: {item.claim}")
        dropped = len(grounding.facts) - len(kept.facts)
        print(f"\ndropped as unprovenanced: {dropped}")

        save_json(
            run_dir,
            "meta.json",
            build_meta(
                args,
                started,
                outcome="ok",
                stage_run="research",
                facts=len(kept.facts),
                dropped_unprovenanced=dropped,
            ),
        )
        return 0

    # Planned once, and this is the outline the book is built from -- there is no
    # second planning call any more. That is what retires the artifact trap:
    # `outline.json` used to come from a throwaway plan while the book came from
    # another, so the two disagreed and the disagreement looked like a bug. Twice.
    outline = await run_outline_pipeline(
        brief, on_task_result=save_iteration, request_id=request_id
    )

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
                args,
                started,
                outcome="ok",
                beats=len(outline.beats),
                stage_run="plan",
                grounding=grounding_meta(outline),
            ),
        )
        return 0

    if args.stage == "plot":
        # Only this path calls the Plot Planner directly. A full run gets its
        # page plan from the workflow as `plan_pages-1.json`; producing a second
        # one here for display is the same trap one stage down -- two plans that
        # disagree, and no way to tell from the filename which the book used.
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
                grounding=grounding_meta(outline),
            ),
        )
        return 0

    # Built from the outline above -- the same object, not a re-plan. This path
    # now exercises the workflow exactly as the MCP tool does, because the tool
    # threads an outline in too.
    story = await run_story_pipeline(
        brief, outline, on_task_result=save_iteration, request_id=request_id
    )
    save_json(run_dir, "story.json", story)

    # Illustration comes before the PDF, since the PDF is what the pictures go in.
    # `art` stays None when the flag is off or nothing is being saved, and
    # `render_pdf(story, path, None)` is exactly the text-only book -- so there is
    # one render call, not one per branch.
    art = None
    if args.illustrate and run_dir is not None:
        art = await run_illustration_pipeline(
            brief, story, run_dir, on_task_result=save_iteration
        )
        save_json(run_dir, "art.json", art)
        drawn = sum(1 for item in art.pages if item.status is not ArtStatus.FAILED)
        print(f"\nillustrated {drawn}/{len(art.pages)} pages")
        # Printed rather than left in the artifact, because it is the one thing
        # about this stage that a person reading the terminal needs to know: a
        # book whose pictures were not reference-conditioned may show the same
        # character three different ways.
        print(f"fully conditioned: {art.fully_conditioned}")
        for item in art.portraits + art.pages:
            if item.status is not ArtStatus.CONDITIONED:
                print(f"  {item.key}: {item.status.value} -- {item.detail}")

    # After the PDF rather than before it, because narration and illustration are
    # independent: audio does not go in the book, and a failure here must not cost
    # the pictures. `run_narration_pipeline` writes narration.json itself.
    if args.narrate and run_dir is not None:
        narration = await run_narration_pipeline(
            story, brief, run_dir, on_task_result=save_iteration
        )
        print(
            f"\nnarrated {narration.pages_narrated}/{len(narration.items)} pages "
            f"as {narration.voice_id!r} at speed {narration.speed:.2f}"
        )
        # Printed rather than left in the artifact for the same reason
        # `fully_conditioned` is: a partially narrated book is the thing a person
        # reading the terminal needs told, because an unplayable page 6 is
        # invisible until someone reaches page 6.
        for item in narration.items:
            if item.status is not NarrationStatus.NARRATED:
                print(f"  page {item.page_number}: {item.status.value}")
        if narration.stitched is None:
            print("  no story.mp3 written -- nothing was narrated")

    if run_dir is not None:
        (run_dir / "story.md").write_text(
            story_markdown(story, brief), encoding="utf-8"
        )
        # Inside the guard deliberately: --no-save leaves run_dir None, and a
        # PDF write outside it would crash on None / "story.pdf".
        if args.pdf or args.illustrate:
            render_pdf(story, run_dir / "story.pdf", art)

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

    # The safety critic enforces `avoid` now, and a run carrying an unresolved
    # safety finding raises rather than reaching this line. This substring check
    # stays as a cross-check on the critic: a hit here, on a run that passed,
    # means the critic missed a literal match -- the cheapest possible false
    # negative, and the one most worth knowing about.
    hits = [
        page.page_number
        for page in story.pages
        if any(term.lower() in page.text.lower() for term in brief.avoid)
    ]
    print(f"pages with a literal 'avoid' term (critic cross-check): {hits or 'none'}")

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
            # From `story.outline`, not the pre-`write_story` outline: the boundary
            # check may have dropped a fact, and this must record what the book was
            # built from rather than what the caller supplied.
            grounding=grounding_meta(story.outline),
            total_words=sum(words),
            longest_page_words=max(words),
            pages_over_level_limit=dense,
            pages_with_avoid_term=hits,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
