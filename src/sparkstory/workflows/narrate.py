"""Reading a finished book aloud: one file per page, then the whole thing.

A fourth ``@entrypoint``, deliberately separate from ``write_story`` and from
``illustrate`` rather than a stage inside either. The reason is this project's own
precedent, twice over: Session 8 split planning from writing because one tool doing
both meant a failure in either half destroyed the other's work, and Session 6 split
illustration off for the sharper version of the same argument -- it is expensive and
failure-prone while prose is the part five sessions were spent improving.

Narration inherits that argument and adds one of its own. Holding the ``Story``
fixed is what makes comparing two voices *attributable*: re-narrate without
re-writing and the only thing that changed is the voice. Under a coupled design
every voice comparison would re-roll the prose, which is exactly the confound
finding L had to be written around.

**No barrier, and no planning stage.** Every page is independent, so all pages run
concurrently under ``asyncio.gather``; serial narration would make a ten-page book
roughly ten times slower for nothing. Illustration needs a portrait phase before
its pages because a page conditions on portraits -- there is no narration
equivalent, because nothing decides how a book sounds. The script is
``StoryPage.text`` and the voice is on the brief.

**Nothing here is chosen by a model, and that is the design.** No Narration
Director, no per-page delivery note, no script rewriter. Three reasons, in order of
weight: a model rewriting the prose would make the audio and the printed page
disagree, in a book whose words fail closed on safety findings a post-hoc rewrite
would bypass; rule 13 says a per-page "delivery note" gets satisfied with *"read
warmly"*, which a ``speed`` derived from ``ReadingLevel`` gives for free with no
call; and verbatim text is what makes "the audio matches page 6" checkable at all.
``NarrationItem.sha256`` is that check made durable.

**Narration fails soft, like illustration and unlike ``write_story``.** A page whose
audio fails is recorded ``FAILED``, its file is absent, and the book still plays. The
asymmetry is deliberate: a surviving safety finding means something a parent asked to
exclude is in their child's book, while a missing page of narration means a missing
page of narration.

The exception is an **all-failed run**, which writes no ``story.mp3`` at all rather
than an empty one. That is finding N's failure mode in audio form -- an empty file
plays as silence, and silence is indistinguishable from success on a casual listen.
"""

import asyncio
import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.func import entrypoint, task

from sparkstory.config import settings
from sparkstory.entities.exceptions import AudioGenerationError
from sparkstory.entities.narration import (
    NarrationItem,
    NarrationStatus,
    StoryNarration,
)
from sparkstory.entities.stories import (
    ReadingLevel,
    Story,
    StoryBrief,
    StoryPage,
    Voice,
)
from sparkstory.models.get_speech_model import SpeechModel, get_speech_model
from sparkstory.utils.logging_utils import get_logger
from sparkstory.workflows.retries import RETRY_POLICY
from sparkstory.workflows.types import NarrationWorkflowInput

logger = get_logger(__name__)

#: Which provider voice reads for each `Voice` on the brief.
#:
#: The provider's vocabulary lives here and nowhere else, which is the whole point
#: of the enum: swapping to ElevenLabs is this dict plus one seam, and no MCP
#: client has to change. Every id was accepted by the live endpoint on 2026-08-12,
#: and an unknown one is a 404 at generation time -- i.e. after the book is
#: written -- so a test asserts both ids are ones that answered 200.
#:
#: `eve` is the provider's own documented default, so it has external
#: justification. **`leo` is a guess**: the roster returns 26 voices whose only
#: distinguishing field is `gender`, with no expressive metadata whatsoever, so
#: nothing in the data recommends one of the 19 male voices over another. `orion`
#: and `atlas` also answered 200 and are the obvious alternatives if a listen says
#: this one is wrong for a bedtime story. That listen has not happened.
_VOICES: dict[Voice, str] = {
    Voice.FEMALE: "eve",
    Voice.MALE: "leo",
}

#: Reading pace per level, within the provider's 0.7-1.5 range.
#:
#: Derived from the brief rather than chosen by a model, deliberately (see the
#: module docstring). Exhaustive over `ReadingLevel` rather than defaulted: a
#: missing level would `KeyError` at generation time, after the book was written,
#: and there is a test asserting every level is present.
#:
#: A pre-reader is being read *to* and needs the slowest pace; a confident reader
#: may be following the words, where an unnaturally slow voice is its own problem.
#: The values are a starting point, not a measurement -- nobody has listened yet.
_SPEEDS: dict[ReadingLevel, float] = {
    ReadingLevel.PRE_READER: 0.85,
    ReadingLevel.EARLY_READER: 0.9,
    ReadingLevel.DEVELOPING: 0.95,
    ReadingLevel.CONFIDENT: 1.0,
}

#: **There is no inter-page gap, and that is a measured decision rather than an
#: omission.** Two attempts at one both corrupted the stream, and the second is the
#: instructive failure:
#:
#: 1. A hardcoded ``MP3_SILENCE * 4`` declared MPEG-1 at 44.1 kHz while the provider
#:    returns **MPEG-2 at 24 kHz**. Splicing one into the other changes sample rate
#:    mid-stream, and a decoder walking ``story.mp3`` lost sync at the first join:
#:    **9.2%** of the file parsed as contiguous frames.
#: 2. Copying the page's *own* 4-byte header and zeroing a 96-byte payload fixed the
#:    format mismatch -- one version, one sample rate -- and still broke the walk at
#:    **8.6%**, because a real frame at 128 kbps / 24 kHz is 384 bytes. The decoder
#:    read the declared length and landed in the middle of the next fabricated
#:    header.
#:
#: Plain concatenation with no gap walks **100% of bytes -- 1070 frames, 25.7
#: seconds, all ten pages**. So the pause is dropped: a correct stream beats a
#: cosmetic beat between pages, and the pages already end on natural silence.
#:
#: The general lesson is rule 22's, one level up. Both broken versions were *valid
#: in isolation* and the offline tests asserted exactly that -- sync bits and layer
#: -- which is precisely what let a plausible-but-wrong constant pass and fail on
#: real audio. Do not reintroduce a gap without walking a real stitched file.

#: How many TTS requests may be in flight at once.
#:
#: Unlike `_MAX_CONCURRENT_IMAGES`, this is **not** measured against an observed
#: 429 -- no narration run has hit a rate limit yet, because no narration run has
#: happened. It is set to the image endpoint's proven-safe number on the
#: assumption that one account's limits are related across endpoints, which is a
#: guess. If a live run returns 429, that is the finding that replaces this value
#: with a measurement. Not a setting, per Rule 3: it describes a provider limit
#: rather than a preference.
_MAX_CONCURRENT_SPEECH = 4


def build_speech_model() -> SpeechModel:
    """Build the speech model narration will use.

    A module-level factory rather than an argument threaded through the workflow
    payload, following ``build_research_context`` in ``plan_outline.py``. Two
    reasons, and the second is the load-bearing one:

    * A workflow payload has to survive a checkpointer, and a ``SpeechModel``
      holds a closure over an API key -- it is not serialisable, and the failure
      would appear only once resumable runs arrive. ``types.py`` already records
      this trap for ``Path``.
    * It gives tests one seam to patch that covers the whole stage.
    """
    return get_speech_model(settings.narrator_model)


def _speed_for(level: ReadingLevel) -> float:
    """How fast to read, for a child at this reading level."""
    return _SPEEDS[level]


def _page_filename(page_number: int) -> str:
    """``page-03.mp3``. Zero-padded so lexical order is page order on disk."""
    return f"page-{page_number:02d}.mp3"


def stitch(parts: list[bytes]) -> bytes:
    """Join per-page audio into one continuous file.

    Plain concatenation, no separator -- see the ``_SILENCE_FRAMES`` note above for
    the two gap attempts that were measured and dropped. MP3 frames are
    self-contained, so concatenating whole files yields one legal stream: a real
    stitched book walks **100% of its bytes as contiguous frames**, verified with a
    frame parser on live audio.

    **The remaining cost is accepted rather than overlooked.** The file carries no
    Xing/Info header and no ID3 tag -- verified absent on a real run -- and those
    are what hold duration for a variable-bitrate stream. A player therefore
    estimates length from the first frame, so reported duration and seeking may be
    inaccurate even though playback is correct.

    The alternative is ``pydub``, which needs **ffmpeg** -- a *system* binary, not a
    Python dependency. That would break "clone the repo and it works" and add a CI
    install step, to fix metadata on a file whose purpose is to be played from the
    beginning at bedtime. If seeking turns out to matter, that is the finding that
    buys the dependency.
    """
    return b"".join(parts)


async def _narrate_page(
    model: SpeechModel,
    page: StoryPage,
    voice_id: str,
    speed: float,
    directory: Path,
) -> tuple[NarrationItem, bytes | None]:
    """Narrate one page, returning its record and its audio.

    Never raises. A page that cannot be narrated is recorded ``FAILED`` and the
    book goes on without it -- the fail-soft rule from the module docstring, at the
    point it actually applies.
    """
    digest = hashlib.sha256(page.text.encode()).hexdigest()
    try:
        audio = await model.speak(page.text, voice_id, speed)
    except AudioGenerationError:
        # Warning rather than error: this is an expected event in a fail-soft
        # stage. `exc_info` is on because the provider's own message is the only
        # clue to whether it was a 429, a refusal or a bad voice id.
        logger.warning("page %d could not be narrated", page.page_number, exc_info=True)
        return (
            NarrationItem(
                page_number=page.page_number,
                status=NarrationStatus.FAILED,
                path=None,
                sha256=digest,
            ),
            None,
        )

    # The extension comes from what the provider actually returned, so a codec
    # switch cannot leave us writing one format into a filename claiming another.
    name = _page_filename(page.page_number)
    if audio.audio_format != "mp3":
        name = f"page-{page.page_number:02d}.{audio.audio_format}"
    path = directory / name
    path.write_bytes(audio.data)

    return (
        NarrationItem(
            page_number=page.page_number,
            status=NarrationStatus.NARRATED,
            path=path,
            sha256=digest,
        ),
        audio.data,
    )


@task(retry_policy=RETRY_POLICY)
async def narrate_pages(
    request_id: str,
    story: Story,
    brief: StoryBrief,
    directory: str,
) -> StoryNarration:
    """Narrate every page concurrently, then stitch what survived."""
    model = build_speech_model()
    voice_id = _VOICES[brief.voice]
    speed = _speed_for(brief.child.reading_level)
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)

    logger.info(
        "[%s] narrating %d pages as %r at speed %.2f, %d at a time",
        request_id,
        len(story.pages),
        voice_id,
        speed,
        _MAX_CONCURRENT_SPEECH,
    )

    limit = asyncio.Semaphore(_MAX_CONCURRENT_SPEECH)

    async def narrate(page: StoryPage) -> tuple[NarrationItem, bytes | None]:
        async with limit:
            return await _narrate_page(model, page, voice_id, speed, out)

    # Not `return_exceptions=True`: `_narrate_page` already converts a provider
    # failure into a FAILED record, so an exception escaping it is a real bug and
    # should not be silently folded into the results.
    results = await asyncio.gather(*(narrate(page) for page in story.pages))

    # Sorted by page number rather than left in completion order. `gather`
    # preserves argument order, so this is insurance against that changing rather
    # than a fix for a known bug -- but the file is read by a human and the
    # stitched audio is built from this list, so the ordering is load-bearing.
    ordered = sorted(results, key=lambda pair: pair[0].page_number)
    items = [item for item, _audio in ordered]
    audio_parts = [audio for _item, audio in ordered if audio is not None]

    stitched: Path | None = None
    if audio_parts:
        stitched = out / "story.mp3"
        stitched.write_bytes(stitch(audio_parts))
    else:
        # No file at all rather than an empty one: an empty story.mp3 plays as
        # silence, and silence is indistinguishable from success on a casual
        # listen. An error rather than a warning, because unlike a single failed
        # page this means the stage did nothing at all.
        logger.error(
            "[%s] no page could be narrated; writing no %s",
            request_id,
            "story.mp3",
        )

    narration = StoryNarration(
        voice_id=voice_id,
        speed=speed,
        items=items,
        stitched=stitched,
    )
    logger.info(
        "[%s] narrated %d/%d pages, complete=%s",
        request_id,
        narration.pages_narrated,
        len(items),
        narration.is_complete,
    )
    return narration


def build_narration_workflow(
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> Any:
    """Compile the narration workflow, optionally with a checkpointer."""

    @entrypoint(checkpointer=checkpointer)
    async def narration_workflow(payload: NarrationWorkflowInput) -> StoryNarration:
        return await narrate_pages(
            payload["request_id"],
            payload["story"],
            payload["brief"],
            payload["directory"],
        )

    return narration_workflow


#: Compiled once per process. Compiling is pure -- no network, no API key.
NARRATION_WORKFLOW = build_narration_workflow()


async def run_narration_pipeline(
    story: Story,
    brief: StoryBrief,
    directory: Path,
    *,
    on_task_result: Callable[[str, Any], None] | None = None,
) -> StoryNarration:
    """Narrate a finished story, writing audio into ``directory``.

    There is deliberately no ``speech_model`` argument. The provider comes from
    ``build_speech_model`` above, which tests patch -- the same seam shape
    ``plan_outline.py`` uses for ``build_research_context``. An injected model
    cannot travel in the workflow payload, because a payload must survive a
    checkpointer and a model holding a closure over an API key is not
    serialisable; ``types.py`` records the same trap for ``Path``.

    Args:
        story: The finished book. Never re-written here.
        brief: Read for exactly two things -- ``voice`` and the child's
            ``reading_level``. Nothing else about it reaches the provider.
        directory: Where audio is written. Created if absent.
        on_task_result: Called with ``(task_name, result)`` as each ``@task``
            completes, for the debug script's numbered artifacts. Nothing on the
            MCP path passes it.

    Returns:
        A ``StoryNarration`` recording every page and, honestly, what became of
        it. A run where every page failed returns one whose items are all
        ``FAILED`` and whose ``stitched`` is ``None``, rather than raising.

    Raises:
        AudioConfigurationError: the narrator model is unknown or its key is
            unset. Raised before any page is attempted, rather than degraded,
            because it means no page can possibly be narrated and every one would
            fail identically -- the same call illustration makes for a missing
            image key.
    """
    request_id = str(uuid4())
    logger.info(
        "[%s] narrating %r: %d pages, model=%s, voice=%s",
        request_id,
        story.outline.title,
        len(story.pages),
        settings.narrator_model,
        brief.voice.value,
    )

    directory.mkdir(parents=True, exist_ok=True)

    narration: StoryNarration | None = None
    async for update in NARRATION_WORKFLOW.astream(
        NarrationWorkflowInput(
            request_id=request_id,
            brief=brief,
            story=story,
            directory=str(directory),
        ),
        stream_mode="updates",
    ):
        for name, value in update.items():
            if isinstance(value, StoryNarration):
                narration = value
                continue
            if on_task_result is not None:
                on_task_result(name, value)

    if narration is None:  # pragma: no cover - the entrypoint always returns one
        raise AudioGenerationError("The workflow completed without producing audio.")

    # Written here rather than by the caller, so the record exists even when the
    # caller forgets -- and because a `StoryNarration` is a BaseModel,
    # `model_dump_json` handles the Paths that a bare `json.dumps` would choke on.
    # Finding P is the bill for learning that at a live run instead.
    (directory / "narration.json").write_text(
        narration.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )

    return narration
