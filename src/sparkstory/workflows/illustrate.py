"""Drawing a finished book: portraits first, then every page against them.

A third `@entrypoint`, deliberately separate from `write_story` rather than a stage
inside it. The reason is this project's own precedent: Session 8 split planning from
writing because one tool doing both meant a failure in either half destroyed the
other half's work. Illustration has that property more sharply -- it is the most
expensive and most failure-prone stage in the system, while prose is the part five
sessions have been spent improving. Coupling them means one 503 discards a book that
passed both critics.

It also buys re-illustrating without re-writing, which is what makes comparing two
illustration attempts possible at all: the `Story` is held fixed, so a difference is
attributable. Under the coupled design every experiment re-rolls the prose.

**Two shapes, and the difference is a barrier.** Portraits run first and must all be
attempted before any page starts, because a page's references are portraits -- that
is a genuine barrier, not a stylistic one. The pages are mutually independent, so
they run concurrently under `asyncio.gather`; serial generation would make an 8-page
book roughly 8x slower for nothing. This is lesson 05's parallel fan-out, which
CLAUDE.md's course inventory already names as the shape for per-page illustration.

**Illustration fails soft, and that is the point of the whole module.** A page whose
image fails leaves that page's frame blank and the book still renders -- `render_pdf`
draws a blank upper 55% when given no art, so a partially illustrated book is
natively supported. This is the opposite of `write_story`, which fails closed, and
the asymmetry is deliberate: a surviving safety finding means something a parent
asked to exclude is in their child's book, while a missing picture on page 6 means a
book with a missing picture.

The exception is a **portrait** failure, which silently removes reference
conditioning from every page that character appears on. That is finding N's failure
mode -- output that looks fine while the mechanism did not run -- so it is logged as
a warning and recorded in `StoryArt`, which is what lets a run answer "was this book
actually reference-conditioned?" by reading a file.
"""

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.func import entrypoint, task

from sparkstory.config import settings
from sparkstory.entities.exceptions import ImageGenerationError
from sparkstory.entities.illustration import (
    MAX_REFERENCE_IMAGES,
    ArtItem,
    ArtStatus,
    CharacterAppearance,
    ConsistencyVerdict,
    IllustrationPlan,
    PageArt,
    StoryArt,
)
from sparkstory.entities.stories import Story, StoryBrief
from sparkstory.models.get_image_model import ImageModel, get_image_model
from sparkstory.models.get_model import get_chat_model
from sparkstory.nodes.consistency_judge import ConsistencyJudgeNode
from sparkstory.nodes.illustration_director import IllustrationDirectorNode
from sparkstory.utils.logging_utils import get_logger
from sparkstory.workflows.retries import RETRY_POLICY
from sparkstory.workflows.types import IllustrationWorkflowInput
from sparkstory.workflows.validation import validate_illustration_plan

logger = get_logger(__name__)


def _who_prefix(page: PageArt, appearances: dict[str, str]) -> str:
    """Restate what each character on this page looks like.

    **Finding U is why this exists**, and it is the one thing the first live run
    changed about the design. A page prompt names its characters without describing
    them -- "Pip beside her on the sill" -- because the reference portrait is
    supposed to carry identity. That assumption is only half true. It held perfectly
    for a five-year-old girl across five pages, and failed for the fox beside her,
    who arrived as a tabby cat on page 1 and a white dog on page 6. Given a name and
    no species, the model substitutes a generic pet.

    Every upstream stage was correct -- the outline said fox, the Director's
    `appearance` said fox with named markings, and the portrait *was* a fox. Only the
    page prompt was silent. So identity now travels in the text as well as the image,
    which costs nothing: `appearance` was already being generated and simply was not
    reaching the prompt.

    Prefixed rather than appended, because an image model weights the start of a
    prompt most heavily and *who is in this picture* is the thing that was being lost.
    """
    described = [
        f"{name} is {appearances[name]}"
        for name in page.characters_present
        if name in appearances
    ]
    if not described:
        return ""
    return " ".join(described) + " "


def _style_suffix(style_bible: str) -> str:
    """The sentence appended to every prompt so one look governs the book.

    Appended rather than prepended: an image model weights the start of a prompt
    most heavily, and what this picture *shows* must outrank how it is drawn. The
    no-writing instruction is repeated here even though the Director was also told
    it, because that instruction constrains the *image model*, and the Director
    only decides what to ask for.
    """
    return (
        f" Style, the same for every picture in this book: {style_bible}"
        " No text, letters, words, captions or signs anywhere in the image."
    )


@task(retry_policy=RETRY_POLICY)
async def direct_illustrations(
    request_id: str, brief: StoryBrief, story: Story
) -> IllustrationPlan:
    """Stage 1: decide the shared look, each appearance, and each picture."""
    logger.info(
        "[%s] stage=direct model=%s",
        request_id,
        settings.illustration_director_model,
    )
    node = IllustrationDirectorNode(
        model=get_chat_model(settings.illustration_director_model),
        brief=brief,
        story=story,
    )
    return await node.ainvoke()


def _write_image(directory: Path, name: str, data: bytes, suffix: str) -> Path:
    """Write one image beside the book and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.{suffix}"
    path.write_bytes(data)
    return path


async def _draw_portrait(
    model: ImageModel,
    character: CharacterAppearance,
    style_bible: str,
    directory: Path,
) -> tuple[ArtItem, bytes | None]:
    """Draw one reference portrait, returning its record and its bytes.

    The bytes are returned as well as written, because every page conditions on
    them and re-reading from disk would be a second failure mode for no gain.
    """
    prompt = character.portrait_prompt + _style_suffix(style_bible)
    try:
        image = await model.generate(prompt)
    except ImageGenerationError as exc:
        # A warning, not an error, and never a raise. But it must be loud: every
        # page this character appears on now loses its reference, which is exactly
        # the silent degradation finding N is about.
        logger.warning(
            "portrait for %r failed, pages with them will be unconditioned: %s",
            character.name,
            exc,
        )
        return (
            ArtItem(key=character.name, status=ArtStatus.FAILED, detail=str(exc)),
            None,
        )

    # A portrait is generated from a prompt alone -- there is nothing to condition
    # it on -- so CONDITIONED would be a lie here. It is the *source* of
    # conditioning, and `fully_conditioned` requires every item to be CONDITIONED,
    # so the status has to say what actually happened.
    path = _write_image(
        directory, f"portrait-{character.name}", image.data, image.image_format
    )
    return (
        ArtItem(
            key=character.name,
            status=ArtStatus.CONDITIONED,
            path=path,
            detail="generated from its own prompt; the reference for every page",
        ),
        image.data,
    )


async def _draw_page(
    model: ImageModel,
    page: PageArt,
    style_bible: str,
    portraits: dict[str, bytes],
    appearances: dict[str, str],
    directory: Path,
) -> ArtItem:
    """Draw one page, conditioned on its characters' portraits where they exist."""
    # Who, then what happens, then how it is drawn. See `_who_prefix` for why the
    # first part is not left to the reference image alone.
    prompt = _who_prefix(page, appearances) + page.prompt + _style_suffix(style_bible)
    # Only characters whose portrait actually succeeded. A name with no portrait is
    # dropped from the references rather than failing the page: the character is
    # still described in the prompt, so the picture is drawn, just not conditioned.
    references = [
        portraits[name] for name in page.characters_present if name in portraits
    ]
    references = references[:MAX_REFERENCE_IMAGES]

    try:
        if references:
            image = await model.edit(prompt, references)
            status = ArtStatus.CONDITIONED
            used = ", ".join(n for n in page.characters_present if n in portraits)
            detail = f"conditioned on: {used}"
        else:
            # No portraits available -- either the page has nobody in it, or every
            # portrait it needed failed. Both produce a picture; neither produces a
            # *consistent* one, so the status distinguishes it and the artifact
            # carries why.
            image = await model.generate(prompt)
            status = ArtStatus.UNCONDITIONED
            detail = (
                "no reference portrait available"
                if page.characters_present
                else "no characters on this page"
            )
    except ImageGenerationError as exc:
        logger.warning(
            "page %d image failed, leaving it blank: %s", page.page_number, exc
        )
        return ArtItem(
            key=str(page.page_number), status=ArtStatus.FAILED, detail=str(exc)
        )

    path = _write_image(
        directory, f"page-{page.page_number:02d}", image.data, image.image_format
    )
    return ArtItem(key=str(page.page_number), status=status, path=path, detail=detail)


@task(retry_policy=RETRY_POLICY)
async def draw_portraits(
    request_id: str, plan: IllustrationPlan, directory: str
) -> tuple[list[ArtItem], dict[str, bytes]]:
    """Stage 2: one reference portrait per character, before any page is drawn.

    Sequential, and this is the barrier the module docstring names -- every page
    conditions on these, so none can start until they are attempted. They are also
    run one at a time rather than gathered: there are few of them, and a rate limit
    hit here would cost every page its reference, which is the one failure worth
    being slow to avoid.
    """
    model = get_image_model(settings.illustrator_model)
    records: list[ArtItem] = []
    portraits: dict[str, bytes] = {}

    for character in plan.characters:
        record, data = await _draw_portrait(
            model, character, plan.style_bible, Path(directory)
        )
        records.append(record)
        if data is not None:
            portraits[character.name] = data

    logger.info(
        "[%s] %d/%d portraits drawn",
        request_id,
        len(portraits),
        len(plan.characters),
    )
    return records, portraits


async def _judge(
    *,
    name: str,
    image: bytes,
    reference_image: bytes | None = None,
    appearance: str | None = None,
) -> ConsistencyVerdict | None:
    """One comparison, or None if the judge could not be asked.

    Fails soft, like every other per-image step in this module, and for a sharper
    reason than the drawing does: a judge is a *check*, so a broken check must never
    be able to destroy a book that is fine. `None` therefore means "nobody looked",
    which is exactly what `fully_consistent` treats it as.
    """
    try:
        node = ConsistencyJudgeNode(
            model=get_chat_model(settings.consistency_judge_model),
            name=name,
            image=image,
            reference_image=reference_image,
            appearance=appearance,
        )
        return await node.ainvoke()
    except Exception as exc:  # noqa: BLE001 -- a check may never break a book
        # Loud, for finding N's reason: a check that silently stopped running
        # produces a book that looks judged and is not.
        logger.warning("could not judge %r, leaving it unjudged: %s", name, exc)
        return None


@task(retry_policy=RETRY_POLICY)
async def check_portraits(
    request_id: str,
    plan: IllustrationPlan,
    records: list[ArtItem],
    portraits: dict[str, bytes],
) -> tuple[list[ArtItem], dict[str, bytes]]:
    """Stage 2b: does each portrait show the character it was told to?

    Runs *before* any page is paid for, and this ordering is the point. One live
    run's Director wrote "small black ant" and the portrait came back green, and
    every page then copied the green faithfully -- so a check comparing pages to
    portraits would have called that book consistent while the character was wrong
    throughout. A wrong reference poisons every page conditioned on it, which makes
    this the cheapest check in the module and the one with the most leverage: two
    calls, before the expensive stage.

    A portrait that fails is **dropped from the references**, so pages fall back to
    the existing unconditioned path rather than conditioning on a picture known to
    be wrong. It is deliberately not deleted and not redrawn: the file stays on disk
    with its verdict recorded, because a run that quietly discarded its own evidence
    would be unauditable.
    """
    appearances = {c.name: c.appearance for c in plan.characters}
    checked: list[ArtItem] = []
    kept = dict(portraits)

    for record in records:
        data = portraits.get(record.key)
        if data is None:
            # Never drawn, so there is nothing to look at. `consistency` stays None,
            # which reads as "nobody looked" rather than as a pass.
            checked.append(record)
            continue

        verdict = await _judge(
            name=record.key, image=data, appearance=appearances[record.key]
        )
        checked.append(record.model_copy(update={"consistency": verdict}))

        if verdict is not None and not verdict.matches:
            logger.warning(
                "[%s] portrait for %r does not match its description (%s: %s);"
                " pages will not be conditioned on it",
                request_id,
                record.key,
                verdict.attribute,
                verdict.difference,
            )
            kept.pop(record.key, None)

    logger.info(
        "[%s] %d/%d portraits usable as references",
        request_id,
        len(kept),
        len(portraits),
    )
    return checked, kept


#: How many image requests may be in flight at once.
#:
#: Measured, like everything else about this endpoint: the first live run fired six
#: pages concurrently and page 2 came back `429 resource-exhausted ... Requests per
#: Second (actual/limit): 5/5`. Set below that limit rather than at it, because the
#: portraits and any retry share the same budget.
#:
#: A bound rather than a retry alone. `RETRY_POLICY` already retries a 429, but N
#: simultaneous requests against a 5/s ceiling collide again on every attempt --
#: retrying an over-subscription reproduces it. Not a setting, per Rule 3: it
#: describes the provider's limit, not a preference, and there is one provider.
_MAX_CONCURRENT_IMAGES = 4


@task(retry_policy=RETRY_POLICY)
async def draw_pages(
    request_id: str,
    plan: IllustrationPlan,
    portraits: dict[str, bytes],
    directory: str,
) -> list[ArtItem]:
    """Stage 3: the pages, concurrently but bounded by the provider's rate limit."""
    model = get_image_model(settings.illustrator_model)
    appearances = {c.name: c.appearance for c in plan.characters}
    logger.info(
        "[%s] drawing %d pages, %d at a time",
        request_id,
        len(plan.pages),
        _MAX_CONCURRENT_IMAGES,
    )

    limit = asyncio.Semaphore(_MAX_CONCURRENT_IMAGES)

    async def draw(page: PageArt) -> ArtItem:
        async with limit:
            return await _draw_page(
                model, page, plan.style_bible, portraits, appearances, Path(directory)
            )

    records = await asyncio.gather(*(draw(page) for page in plan.pages))
    # Sorted by page number rather than left in completion order: these are written
    # to an artifact a human reads, and gather preserves argument order anyway --
    # this is insurance against that changing, not a fix for a known bug.
    return sorted(records, key=lambda item: int(item.key))


@task(retry_policy=RETRY_POLICY)
async def check_pages(
    request_id: str,
    plan: IllustrationPlan,
    records: list[ArtItem],
    portraits: dict[str, bytes],
) -> list[ArtItem]:
    """Stage 3b: does each page still show the character its portrait established?

    **Report-only. Nothing is redrawn.** Deliberately, and the reason is rule 17: a
    revision can be worse than what it replaced and the loop cannot tell. Prose
    solves that with `draft_score`, and there is no equivalent for a picture --
    "fewer findings from a judge whose noise floor is unmeasured" is exactly what
    rule 29 warns against. So this measures first. Whether a redraw loop is worth
    building is a decision to make once the false-positive rate is known, and the
    committed runs in `outputs/` are enough to measure it without generating
    anything.

    Only `CONDITIONED` pages are judged: an unconditioned page had no reference, so
    there is nothing for it to be consistent *with*, and a failed page has no image.
    Judging either would spend a call to produce a finding no redraw could fix.

    Unlike `check_portraits`, this reads the image back from disk -- `draw_pages`
    returns paths rather than bytes, because a page's bytes have no second consumer
    the way a portrait's do. That is one more failure mode (a missing or unreadable
    file), which is why the read is inside the guarded helper's try and a page whose
    file has gone simply stays unjudged.
    """
    if not settings.judge_pages:
        logger.info("[%s] page judging is off", request_id)
        return records

    by_page = {page.page_number: page for page in plan.pages}
    checked: list[ArtItem] = []
    judged = 0

    for record in records:
        page = by_page.get(int(record.key))
        # The reference to compare against is the first portrait this page was
        # actually conditioned on. One rather than all of them: `ConsistencyVerdict`
        # answers for one character, and asking about two at once would return a
        # single verdict covering both, which is unattributable.
        reference = next(
            (
                portraits[name]
                for name in (page.characters_present if page else [])
                if name in portraits
            ),
            None,
        )
        if (
            record.status is not ArtStatus.CONDITIONED
            or record.path is None
            or reference is None
        ):
            checked.append(record)
            continue

        name = next(n for n in page.characters_present if n in portraits)  # type: ignore[union-attr]
        try:
            image = record.path.read_bytes()
        except OSError as exc:
            logger.warning("could not read %s to judge it: %s", record.path, exc)
            checked.append(record)
            continue

        verdict = await _judge(name=name, image=image, reference_image=reference)
        checked.append(record.model_copy(update={"consistency": verdict}))
        if verdict is not None:
            judged += 1
            if not verdict.matches:
                logger.warning(
                    "[%s] page %s does not match %r's portrait (%s: %s)",
                    request_id,
                    record.key,
                    name,
                    verdict.attribute,
                    verdict.difference,
                )

    mismatched = sum(
        1 for i in checked if i.consistency is not None and not i.consistency.matches
    )
    logger.info(
        "[%s] judged %d pages, %d did not match", request_id, judged, mismatched
    )
    return checked


def build_illustration_workflow(
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> Any:
    """Compile the illustration workflow, optionally with a checkpointer."""

    @entrypoint(checkpointer=checkpointer)
    async def illustration_workflow(payload: IllustrationWorkflowInput) -> StoryArt:
        request_id = payload["request_id"]
        brief = payload["brief"]
        story = payload["story"]
        directory = payload["directory"]

        plan = await direct_illustrations(request_id, brief, story)

        # Checked after directing and before any image is paid for: the plan is one
        # cheap chat call, and a plan that does not cover the book is the one
        # illustration failure that cannot be recorded in `StoryArt` -- nothing ever
        # tried to draw the page, so there is no item to mark FAILED. It therefore
        # raises, unlike every per-image failure below, which degrade.
        validate_illustration_plan(story, plan)

        portrait_records, portraits = await draw_portraits(request_id, plan, directory)
        # Between drawing the references and using them: a portrait that does not
        # match its own description is dropped here, so the pages below fall back to
        # the unconditioned path instead of inheriting a wrong character.
        portrait_records, portraits = await check_portraits(
            request_id, plan, portrait_records, portraits
        )
        page_records = await draw_pages(request_id, plan, portraits, directory)
        page_records = await check_pages(request_id, plan, page_records, portraits)

        art = StoryArt(
            style_bible=plan.style_bible,
            portraits=portrait_records,
            pages=page_records,
        )
        drawn = sum(1 for item in page_records if item.status is not ArtStatus.FAILED)
        # Both properties, because they answer different questions and a book can
        # honestly be one and not the other: `fully_conditioned` says the mechanism
        # ran, `fully_consistent` says it worked. Reporting only the first is what
        # three live runs did while a fox's paws changed colour.
        logger.info(
            "[%s] illustrated %d/%d pages, fully_conditioned=%s fully_consistent=%s",
            request_id,
            drawn,
            len(page_records),
            art.fully_conditioned,
            art.fully_consistent,
        )
        return art

    return illustration_workflow


#: Compiled once per process. Compiling is pure -- no network, no API key.
ILLUSTRATION_WORKFLOW = build_illustration_workflow()


async def run_illustration_pipeline(
    brief: StoryBrief,
    story: Story,
    directory: Path,
    on_task_result: Callable[[str, Any], None] | None = None,
) -> StoryArt:
    """Illustrate a finished story, writing images into ``directory``.

    Args:
        brief: The brief the story was written from. Needed for the child's age
            and the parent's ``avoid`` list, which constrains pictures as well as
            words.
        story: The finished book. Never re-written here.
        directory: Where images are written. Created if absent.
        on_task_result: Called with ``(task_name, result)`` as each ``@task``
            completes, for the debug script's numbered artifacts. Nothing on the
            MCP path passes it.

    Returns:
        A ``StoryArt`` recording every image and, honestly, what became of it. A
        run where every image failed returns a ``StoryArt`` whose items are all
        ``FAILED`` rather than raising -- the caller renders a text-only book.

    Raises:
        ImageConfigurationError: the illustrator model is unknown or its key is
            unset. Raised rather than degraded, because it means no image can
            possibly be drawn and every page would fail identically.
        StoryStructureError: the Director returned a plan that does not cover the
            book -- a page missing, duplicated, or naming an undescribed character.
            The one illustration failure that raises, because a page nobody tried to
            draw cannot be recorded in the returned ``StoryArt``.
    """
    request_id = str(uuid4())
    logger.info(
        "[%s] illustrating %r: %d pages, model=%s",
        request_id,
        story.outline.title,
        len(story.pages),
        settings.illustrator_model,
    )

    art: StoryArt | None = None
    async for update in ILLUSTRATION_WORKFLOW.astream(
        IllustrationWorkflowInput(
            request_id=request_id,
            brief=brief,
            story=story,
            directory=str(directory),
        ),
        stream_mode="updates",
    ):
        for name, value in update.items():
            if isinstance(value, StoryArt):
                art = value
                continue
            if on_task_result is not None:
                on_task_result(name, value)

    if art is None:  # pragma: no cover - the entrypoint always returns StoryArt
        raise ImageGenerationError("The workflow completed without producing art.")
    return art
