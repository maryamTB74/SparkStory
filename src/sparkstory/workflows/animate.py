"""Turning a finished book into a watchable minute.

A fifth ``@entrypoint``, deliberately separate from ``write_story``,
``illustrate`` and ``narrate`` rather than a stage inside any of them. The reason
is this project's precedent three times over: planning was split from writing,
then illustration, then narration, each time because one tool doing both meant a
failure in either half destroyed the other's work. Video inherits that split and
adds one thing: it is the only stage that consumes all three.

Holding the three fixed is also what makes a re-render *attributable*: change the
camera move and nothing else about the book changed. A coupled design would
re-roll the prose on every render and confound every comparison.

**No barrier, and no planning stage.** Every page is independent, so all pages run
concurrently under ``asyncio.gather``. Illustration needs a portrait phase before
its pages because a page conditions on portraits; there is no video equivalent,
because nothing here decides anything.

**Nothing is chosen by a model, and that is the design.** No shot list, no camera
direction, no per-page pacing note. The picture is the page's illustration, the
length is the page's narration, and the move is arithmetic on the page number.
An instruction gets satisfied the laziest legal way, and the laziest answer to
"pick a camera move for this scene" is *zoom in* every time -- so the call buys
back the fixed option with added noise and added cost.
It is the same refusal ``narrate.py`` made about a per-page delivery note.

**Audio is the spine.** A page is in the video only if it has narration that can be
read: a page with no audio has no *duration*, and the only way to include it would
be to invent one from word count or a constant. That would be a fabricated number
in the one place this stage is otherwise entirely derived from measured artifacts.
A picture, by contrast, is optional -- a narrated page with no illustration becomes
a held card for its full length.

**The accepted cost, stated rather than discovered: a video can be shorter than its
book.** That is the failure mode this project has been bitten by before -- output
that looks fine while the mechanism did not fully run -- so ``StoryVideo`` records
every page and the reason any of them is absent, and the pipeline logs "5 of 6"
rather than reporting success.

**Video fails soft per page and loud on the stage.** A page whose clip fails is
recorded ``FAILED`` and the rest are assembled. But a run where *nothing* survived
writes no ``story.mp4`` at all rather than a zero-byte one: an empty video plays as
nothing, and nothing is indistinguishable from success on a casual glance. That is
narration's rule for ``story.mp3``, and the argument transfers unchanged.
"""

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.func import entrypoint, task

from sparkstory.entities.exceptions import VideoGenerationError
from sparkstory.entities.illustration import StoryArt
from sparkstory.entities.narration import StoryNarration
from sparkstory.entities.stories import Story
from sparkstory.entities.video import StoryVideo, VideoItem, VideoStatus
from sparkstory.models.get_clip_maker import ClipMaker, get_clip_maker
from sparkstory.utils.logging_utils import get_logger
from sparkstory.video.assemble import assemble
from sparkstory.video.kenburns import FPS
from sparkstory.video.probe import probe_audio_duration
from sparkstory.workflows.retries import RETRY_POLICY
from sparkstory.workflows.types import VideoWorkflowInput

logger = get_logger(__name__)

#: The only maker there is. One implementation means nothing to select between, so
#: it gets no setting -- see ``get_clip_maker``.
_MAKER_ID = "kenburns"

#: How many clips may be encoded at once. Unlike ``_MAX_CONCURRENT_IMAGES`` and
#: ``_MAX_CONCURRENT_SPEECH`` this is not a provider's rate limit -- it is local
#: CPU, and ffmpeg already threads within a single encode. Four keeps a book's
#: worth of encodes overlapping without making each one slower by contention. If a
#: live run turns out to be CPU-bound, that measurement replaces this guess.
_MAX_CONCURRENT_CLIPS = 4

#: Indirected so the offline tests stay offline: reading a duration needs ffprobe,
#: and the workflow's own logic -- selection, fail-soft, the record -- needs no
#: subprocess at all. Patched by name, exactly as ``build_clip_maker`` is.
read_duration = probe_audio_duration


def build_clip_maker() -> ClipMaker:
    """Build the clip maker this stage uses.

    A module-level factory rather than an argument threaded through the workflow
    payload, following ``build_speech_model`` in ``narrate.py`` and
    ``build_research_context`` in ``plan_outline.py``. A payload has to survive a
    checkpointer and a maker is not serialisable; it also gives tests one seam to
    patch that covers the whole stage.
    """
    return get_clip_maker(_MAKER_ID)


async def select_pages(
    story: Story, art: StoryArt, narration: StoryNarration
) -> list[tuple[int, Path | None, Path, float]]:
    """Which pages are in the video, with their picture, audio and length.

    Returns ``(page_number, image_or_None, audio, duration)`` for every page whose
    audio exists and can be measured, in page order. A page with no usable audio is
    absent from this list entirely, and the caller records it as ``EXCLUDED``.

    A page's *image* being missing is not a reason to drop it -- that is the held
    card, and it is why the second element is optional.
    """
    selected: list[tuple[int, Path | None, Path, float]] = []

    for page in story.pages:
        audio = narration.page_audio(page.page_number)
        if audio is None or not audio.exists():
            continue
        try:
            duration = await read_duration(audio)
        except VideoGenerationError:
            # A present but unreadable file is a different problem from an absent
            # one, and it is the one worth a log line: the artifact claims this
            # page has audio.
            logger.warning("could not measure %s, excluding its page", audio)
            continue

        image = art.page_image(page.page_number)
        if image is not None and not image.exists():
            image = None
        selected.append((page.page_number, image, audio, duration))

    return selected


@task(retry_policy=RETRY_POLICY)
async def animate_pages(
    request_id: str,
    story: Story,
    art: StoryArt,
    narration: StoryNarration,
    directory: str,
) -> StoryVideo:
    """Select, encode every page concurrently, then assemble what survived."""
    maker = build_clip_maker()
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)

    selected = await select_pages(story, art, narration)
    chosen = {page for page, _image, _audio, _duration in selected}

    logger.info(
        "[%s] animating %d of %d pages at %d fps, %d at a time",
        request_id,
        len(selected),
        len(story.pages),
        FPS,
        _MAX_CONCURRENT_CLIPS,
    )

    limit = asyncio.Semaphore(_MAX_CONCURRENT_CLIPS)

    async def encode(
        page_number: int, image: Path | None, audio: Path, duration: float
    ) -> tuple[VideoItem, Path | None, Path | None]:
        async with limit:
            data = image.read_bytes() if image is not None else None
            try:
                clip = await maker.make_clip(data, duration, page_number)
            except VideoGenerationError as exc:
                # Warning rather than error: an expected event in a fail-soft
                # stage. The rest of the book still assembles.
                logger.warning("page %d could not be animated: %s", page_number, exc)
                return (
                    VideoItem(
                        page_number=page_number,
                        status=VideoStatus.FAILED,
                        duration=duration,
                        reason=str(exc),
                    ),
                    None,
                    None,
                )

        path = out / f"clip-{page_number:02d}.{clip.video_format}"
        path.write_bytes(clip.data)
        return (
            VideoItem(
                page_number=page_number,
                status=VideoStatus.ANIMATED if image is not None else VideoStatus.HELD,
                duration=duration,
                reason=None if image is not None else "no illustration for this page",
            ),
            path,
            audio,
        )

    results = await asyncio.gather(
        *(encode(page, image, audio, d) for page, image, audio, d in selected)
    )

    excluded: list[tuple[VideoItem, Path | None, Path | None]] = [
        (
            VideoItem(
                page_number=page.page_number,
                status=VideoStatus.EXCLUDED,
                duration=None,
                reason="no narration for this page, so it has no length",
            ),
            None,
            None,
        )
        for page in story.pages
        if page.page_number not in chosen
    ]

    # Sorted by page number rather than left in completion order: this is written
    # to an artifact a human reads, and the clip order is the book's order.
    ordered = sorted([*results, *excluded], key=lambda triple: triple[0].page_number)
    items = [item for item, _clip, _audio in ordered]
    pairs = [
        (clip, audio)
        for _item, clip, audio in ordered
        if clip is not None and audio is not None
    ]

    path: Path | None = None
    if pairs:
        path = await assemble(
            [clip for clip, _audio in pairs],
            [audio for _clip, audio in pairs],
            out / "story.mp4",
        )
    else:
        # No file at all rather than an empty one -- see the module docstring. An
        # error rather than a warning, because unlike a single failed page this
        # means the stage did nothing.
        logger.error("[%s] no page could be animated; writing no story.mp4", request_id)

    video = StoryVideo(path=path, fps=FPS, items=items)
    logger.info(
        "[%s] animated %d/%d pages, complete=%s",
        request_id,
        video.pages_animated,
        len(items),
        video.is_complete,
    )
    return video


def build_video_workflow(
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> Any:
    """Compile the video workflow, optionally with a checkpointer."""

    @entrypoint(checkpointer=checkpointer)
    async def video_workflow(payload: VideoWorkflowInput) -> StoryVideo:
        return await animate_pages(
            payload["request_id"],
            payload["story"],
            payload["art"],
            payload["narration"],
            payload["directory"],
        )

    return video_workflow


#: Compiled once per process. Compiling is pure -- no subprocess, no binary check.
VIDEO_WORKFLOW = build_video_workflow()


async def run_video_pipeline(
    story: Story,
    art: StoryArt,
    narration: StoryNarration,
    directory: Path,
    *,
    on_task_result: Callable[[str, Any], None] | None = None,
) -> StoryVideo:
    """Assemble a video from a finished, illustrated, narrated book.

    Args:
        story: The finished book. Never modified. Supplies page order and count.
        art: Its illustrations. A page whose image is missing becomes a held card.
        narration: Its audio. A page whose narration is missing is excluded.
        directory: Where clips and ``story.mp4`` are written. Created if absent.
        on_task_result: Called with ``(task_name, result)`` as each ``@task``
            completes, for the debug script's numbered artifacts. Nothing on the
            MCP path passes it.

    Returns:
        A ``StoryVideo`` recording every page and what became of it. A run where
        every page was excluded returns one whose ``path`` is ``None`` rather than
        raising.

    Raises:
        VideoConfigurationError: ffmpeg is not installed. Raised before any page is
            attempted, because every page would fail identically -- the same call
            ``run_narration_pipeline`` makes for a missing API key.
    """
    request_id = str(uuid4())
    logger.info(
        "[%s] animating %r: %d pages",
        request_id,
        story.outline.title,
        len(story.pages),
    )

    directory.mkdir(parents=True, exist_ok=True)

    video: StoryVideo | None = None
    async for update in VIDEO_WORKFLOW.astream(
        VideoWorkflowInput(
            request_id=request_id,
            story=story,
            art=art,
            narration=narration,
            directory=str(directory),
        ),
        stream_mode="updates",
    ):
        for name, value in update.items():
            if isinstance(value, StoryVideo):
                video = value
                continue
            if on_task_result is not None:
                on_task_result(name, value)

    if video is None:  # pragma: no cover - the entrypoint always returns one
        raise VideoGenerationError("The workflow completed without producing a record.")

    # Written here rather than by the caller, so the record exists even when the
    # caller forgets -- and because `model_dump_json` handles the Paths that a bare
    # `json.dumps` would choke on -- a lesson paid for once by a live run that
    # died on serialisation after the expensive work was already done.
    (directory / "video.json").write_text(
        video.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )

    return video
