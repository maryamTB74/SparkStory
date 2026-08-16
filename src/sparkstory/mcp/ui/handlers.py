"""Request to response for the web UI.

**These call the pipelines directly, not the ``*_tool`` wrappers.** The wrappers
translate ``ConfigurationError`` into ``ToolError``, which is a JSON-RPC concept
and means nothing to a browser. Two surfaces over one core, each translating into
its own protocol's vocabulary -- which is the same reason ``mcp/tools/`` exists as
a layer at all.

**Error policy.** ``ConfigurationError``'s message is shown to the parent,
because it names the variable to set and an operator can act on it -- a run once
died in 17 seconds to an unset critic key. Every other exception
fails the job with a generic sentence and logs the traceback: an unexpected
exception's text is written for a developer, and putting it in HTML is both
confusing and a disclosure habit worth not forming.
"""

import asyncio
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response

from sparkstory.config import _PROJECT_ROOT
from sparkstory.entities.exceptions import ConfigurationError, UnsafeContentError
from sparkstory.entities.stories import ChildProfile, StoryBrief
from sparkstory.mcp.ui.artifacts import MEDIA_TYPES, available_media, resolve_artifact
from sparkstory.mcp.ui.jobs import JobState, registry
from sparkstory.mcp.ui.library import (
    list_books,
    load_book,
    resolve_library_artifact,
)
from sparkstory.mcp.ui.pages import (
    render_book,
    render_form,
    render_job,
    render_library,
    render_library_book,
)
from sparkstory.utils.logging_utils import get_logger
from sparkstory.workflows.plan_outline import run_outline_pipeline
from sparkstory.workflows.write_story import run_story_pipeline

logger = get_logger(__name__)

#: Where runs land, matching `scripts/write_one_story.py`.
_OUTPUTS = _PROJECT_ROOT / "outputs"

#: Background tasks are kept referenced. asyncio holds only a weak reference to
#: a bare `create_task` result, so without this a job can be garbage collected
#: mid-run -- a failure that looks like the pipeline silently hanging.
_running: set[asyncio.Task[None]] = set()

_GENERIC_FAILURE = "The story could not be planned. Check the server log for details."


async def get_form(request: Request) -> HTMLResponse:
    """The brief form."""
    return HTMLResponse(render_form())


async def post_plan(request: Request) -> JSONResponse:
    """Validate a brief, register a job, and start planning in the background."""
    try:
        payload = await request.json()
    except ValueError:
        return JSONResponse({"error": "Expected a JSON body."}, status_code=400)

    try:
        brief = _build_brief(payload)
    except (ValidationError, ValueError, KeyError) as exc:
        # Pydantic already knows every rule; re-implementing them here would be a
        # second validator that drifts from the first.
        return JSONResponse({"error": _readable(exc)}, status_code=400)

    job = registry.create(brief)
    _spawn(_plan_in_background(job.id))
    return JSONResponse({"job_id": job.id}, status_code=202)


async def get_job(request: Request) -> HTMLResponse:
    """The job page: progress, the outline to approve, or a failure."""
    job = registry.get(request.path_params["job_id"])
    if job is None:
        return HTMLResponse("<h1>No such job</h1>", status_code=404)
    return HTMLResponse(render_job(job))


async def get_status(request: Request) -> JSONResponse:
    """Polled every two seconds by the job page."""
    job = registry.get(request.path_params["job_id"])
    if job is None:
        return JSONResponse({"error": "No such job."}, status_code=404)
    return JSONResponse(
        {"state": job.state.value, "detail": job.detail, "error": job.error}
    )


def _spawn(coroutine: Any) -> None:
    """Run a coroutine in the background, keeping a strong reference to it."""
    task = asyncio.create_task(coroutine)
    _running.add(task)
    task.add_done_callback(_running.discard)


async def _plan_in_background(job_id: str) -> None:
    """Run the outline pipeline and record the result on the job."""
    job = registry.get(job_id)
    if job is None:  # pragma: no cover - the registry is append-only
        return

    def _progress(task_name: str, _result: object) -> None:
        registry.update(job_id, detail=_describe(task_name))

    try:
        outline = await run_outline_pipeline(job.brief, _progress)
    except ConfigurationError as exc:
        logger.error("[ui:%s] planning failed -- configuration: %s", job_id, exc)
        registry.update(job_id, state=JobState.FAILED, error=str(exc))
        return
    except Exception:
        logger.exception("[ui:%s] planning failed", job_id)
        registry.update(job_id, state=JobState.FAILED, error=_GENERIC_FAILURE)
        return

    registry.update(
        job_id,
        state=JobState.AWAITING_APPROVAL,
        outline=outline,
        detail="",
    )


def _describe(task_name: str) -> str:
    """A task name a parent can read."""
    readable = {
        "research_premise": "reading up on the subject",
        "plan_outline": "planning the story",
        "critique_outline": "critiquing the outline",
        "plan_pages": "planning the pages",
        "write_prose": "writing the pages",
        "critique_prose": "reading it back",
    }
    return readable.get(task_name, task_name.replace("_", " "))


def _build_brief(payload: dict[str, Any]) -> StoryBrief:
    """Assemble a StoryBrief from the flat form body.

    The form is flat and the schema is nested: `name`, `age`, `pronouns`,
    `reading_level` and `interests` belong to `ChildProfile`, everything else to
    `StoryBrief`. Every constraint stays in the entities -- this only reshapes.
    """
    child = ChildProfile(
        name=str(payload.get("name", "")).strip(),
        age=int(payload["age"]),
        pronouns=payload["pronouns"],
        reading_level=payload.get("reading_level", "early_reader"),
        interests=_split(payload.get("interests")),
    )
    return StoryBrief(
        child=child,
        premise=str(payload.get("premise", "")).strip(),
        tone=payload.get("tone", "gentle"),
        world_rules=payload.get("world_rules", "imaginative"),
        page_count=int(payload.get("page_count", 6)),
        must_include=_split(payload.get("must_include")),
        avoid=_split(payload.get("avoid")),
    )


def _split(value: object) -> list[str]:
    """A comma-separated field as a list, dropping blanks."""
    if not value:
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _readable(exc: Exception) -> str:
    """A validation failure phrased for a parent rather than for a developer."""
    if isinstance(exc, ValidationError):
        problems = []
        for error in exc.errors():
            field = ".".join(str(part) for part in error["loc"]) or "the form"
            problems.append(f"{field}: {error['msg']}")
        return "; ".join(problems)
    if isinstance(exc, KeyError):
        return f"{exc.args[0]}: this field is required"
    return str(exc)


def _run_directory_for(job_id: str, premise: str) -> Path:
    """Where this job's artifacts go.

    Named like every other run -- `YYYYMMDD-HHMMSS-<premise slug>` -- so
    `sparkstory://library`, `make score-books` and the eval harness all see a UI
    run exactly as they see a CLI one.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", premise.lower()).strip("-")[:60]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return _OUTPUTS / f"{stamp}-{slug}"


async def post_approve(request: Request) -> JSONResponse:
    """Accept the outline and start writing the book.

    **The request body is ignored entirely**, and that is the point. The book is
    built from the outline in the server's own job record, so a modified DOM
    changes what the parent saw and not what gets written. Sending the outline
    back would rebuild the MCP path's exact weakness in HTML.
    """
    job_id = request.path_params["job_id"]
    if registry.get(job_id) is None:
        return JSONResponse({"error": "No such job."}, status_code=404)

    # Check and transition together, or two clicks start two writers against one
    # run directory.
    job = registry.transition(job_id, JobState.AWAITING_APPROVAL, JobState.WRITING)
    if job is None:
        current = registry.get(job_id)
        return JSONResponse(
            {"error": f"This job is {current.state.value}, not awaiting approval."},
            status_code=409,
        )

    _spawn(_write_in_background(job_id))
    return JSONResponse({"state": JobState.WRITING.value}, status_code=202)


async def post_revise(request: Request) -> JSONResponse:
    """Ask for a different plan, optionally saying what to change.

    Re-plans from scratch rather than feeding the note to the outline critic as a
    review. Passing the note through as a reviewer argument is arguably better and
    needs a prompt change; every instruction constraining a revision is also a
    licence to under-fix, and the critic loop is not the thing to disturb in a UI
    session.
    """
    job_id = request.path_params["job_id"]
    if registry.get(job_id) is None:
        return JSONResponse({"error": "No such job."}, status_code=404)

    try:
        payload = await request.json()
    except ValueError:
        payload = {}
    note = str(payload.get("note", "")).strip()

    job = registry.transition(job_id, JobState.AWAITING_APPROVAL, JobState.PLANNING)
    if job is None:
        current = registry.get(job_id)
        return JSONResponse(
            {"error": f"This job is {current.state.value}, not awaiting approval."},
            status_code=409,
        )

    if note:
        # A new brief rather than a mutation: the record is frozen, and a second
        # revise must compose on the first rather than replace it.
        amended = job.brief.model_copy(
            update={"premise": f"{job.brief.premise}. {note}"}
        )
        registry.update(job_id, brief=amended, outline=None, detail="")
    else:
        registry.update(job_id, outline=None, detail="")

    _spawn(_plan_in_background(job_id))
    return JSONResponse({"state": JobState.PLANNING.value}, status_code=202)


async def get_book(request: Request) -> HTMLResponse:
    """The finished book, with whatever media the run directory holds."""
    job = registry.get(request.path_params["job_id"])
    if job is None:
        return HTMLResponse("<h1>No such job</h1>", status_code=404)
    if job.story is None:
        return HTMLResponse("<h1>This book is not written yet</h1>", status_code=404)

    media = available_media(job.run_directory, len(job.story.pages))
    return HTMLResponse(render_book(job, media))


async def get_file(request: Request) -> Response:
    """Serve one artifact from this job's run directory.

    The only route here that reads a file chosen by a URL. Both guards live in
    `artifacts.resolve_artifact`; this returns 404 for every rejection, so a
    prober cannot tell "not allowed" from "not there".
    """
    job = registry.get(request.path_params["job_id"])
    if job is None or job.run_directory is None:
        return Response(status_code=404)

    path = resolve_artifact(job.run_directory, request.path_params["name"])
    if path is None:
        return Response(status_code=404)

    return Response(
        content=path.read_bytes(),
        media_type=MEDIA_TYPES.get(path.suffix, "application/octet-stream"),
    )


async def get_library(request: Request) -> HTMLResponse:
    """Every finished book on disk.

    Reads only. A job is still the only thing that writes, so nothing here can
    approve, revise or spend money.
    """
    return HTMLResponse(render_library(list_books(_OUTPUTS)))


async def get_library_book(request: Request) -> HTMLResponse:
    """One book from disk, with whatever media its directory holds."""
    run_id = request.path_params["run_id"]
    loaded = load_book(_OUTPUTS, run_id)
    if loaded is None:
        return HTMLResponse("<h1>No such book</h1>", status_code=404)

    story, directory = loaded
    media = available_media(directory, len(story.pages))
    return HTMLResponse(render_library_book(story, media, run_id, directory))


async def get_library_file(request: Request) -> Response:
    """Serve one artifact from a library run.

    Two guards, as on the job route: the run id must match the run-directory
    pattern and stay inside ``outputs/``, and the file name must be on the
    artifact allowlist -- so ``brief.json`` and ``meta.json`` stay unreachable.
    """
    path = resolve_library_artifact(
        _OUTPUTS, request.path_params["run_id"], request.path_params["name"]
    )
    if path is None:
        return Response(status_code=404)

    return Response(
        content=path.read_bytes(),
        media_type=MEDIA_TYPES.get(path.suffix, "application/octet-stream"),
    )


async def _write_in_background(job_id: str) -> None:
    """Run the story pipeline from the outline the server is holding."""
    job = registry.get(job_id)
    if job is None or job.outline is None:  # pragma: no cover
        return

    run_directory = _run_directory_for(job_id, job.original_premise)
    run_directory.mkdir(parents=True, exist_ok=True)
    registry.update(job_id, run_directory=run_directory)

    def _progress(task_name: str, _result: object) -> None:
        registry.update(job_id, detail=_describe(task_name))

    try:
        story = await run_story_pipeline(job.brief, job.outline, _progress)
    except ConfigurationError as exc:
        logger.error("[ui:%s] writing failed -- configuration: %s", job_id, exc)
        registry.update(job_id, state=JobState.FAILED, error=str(exc))
        return
    except UnsafeContentError as exc:
        # Not a bug: the guardrail worked and the answer is no. The finding
        # travels in the message so a parent can adjust the brief.
        logger.warning("[ui:%s] writing refused -- safety: %s", job_id, exc)
        registry.update(job_id, state=JobState.FAILED, error=str(exc))
        return
    except Exception:
        logger.exception("[ui:%s] writing failed", job_id)
        registry.update(
            job_id,
            state=JobState.FAILED,
            error="The book could not be written. Check the server log for details.",
        )
        return

    (run_directory / "story.json").write_text(story.model_dump_json(indent=2))
    registry.update(job_id, state=JobState.COMPLETE, story=story, detail="")
