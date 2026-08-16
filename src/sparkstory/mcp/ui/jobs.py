"""Job state for the web UI, held between planning and approval.

**Why the server holds the outline.** On the MCP path a client threads
``StoryOutline`` from ``plan_story`` into ``write_story`` as an ordinary tool
argument, so nothing distinguishes an approved outline from a fabricated one
(open item 8). Here the outline never leaves the server's own record: the browser
is shown it, and approval sends back only a job id. A tampered DOM changes what
the parent saw, not what gets built.

That buys tamper-resistance, **not** human verification -- nothing proves a human
rather than a script POSTed to /approve.

**Why a dataclass and not a Pydantic model.** This never crosses a process
boundary, is never sent to a model, and never validates external input. It is
also frozen, so every mutation goes through the registry rather than through two
writers holding the same object.
"""

from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from uuid import uuid4

from sparkstory.entities.stories import Story, StoryBrief, StoryOutline


class JobState(StrEnum):
    """Where a job has got to.

    Five states rather than a boolean plus a payload. A boolean cannot
    distinguish *running* from *failed*, and that collapse is what ``ArtStatus``
    was widened to avoid after ``CONDITIONED`` reported that a mechanism had run
    rather than that it had worked.
    """

    PLANNING = "planning"
    AWAITING_APPROVAL = "awaiting_approval"
    WRITING = "writing"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class Job:
    """One parent's book, from brief to finished story."""

    id: str
    state: JobState
    brief: StoryBrief
    #: What the parent originally typed. `revise` amends the brief's premise, so
    #: without this the form and the book would show the amended text as if it
    #: were the request.
    original_premise: str
    outline: StoryOutline | None = None
    story: Story | None = None
    run_directory: Path | None = None
    #: Human-readable progress, fed by the pipelines' `on_task_result` hook.
    detail: str = ""
    error: str | None = None


class JobRegistry:
    """An in-process store of jobs, keyed by id.

    Cleared on restart by design: artifacts on disk survive, in-flight jobs do
    not. Acceptable for one person on one machine, and the first thing that
    breaks if this is ever run for strangers -- see spec section 5.5.2.

    No cleanup, no TTL, no eviction. A policy for a problem nobody has is config
    for a feature that does not exist.
    """

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}

    def create(self, brief: StoryBrief) -> Job:
        """Register a new job in PLANNING and return it."""
        # uuid4 rather than anything derived from the premise or the clock: a
        # guessable id is a way to read someone else's book.
        job = Job(
            id=str(uuid4()),
            state=JobState.PLANNING,
            brief=brief,
            original_premise=brief.premise,
        )
        self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        """The job, or None if there is no such id."""
        return self._jobs.get(job_id)

    def update(self, job_id: str, **changes: object) -> Job:
        """Replace fields on a job. Raises KeyError if it does not exist."""
        job = self._jobs[job_id]
        updated = replace(job, **changes)  # type: ignore[arg-type]
        self._jobs[job_id] = updated
        return updated

    def transition(
        self, job_id: str, expected: JobState, new_state: JobState, **changes: object
    ) -> Job | None:
        """Move a job between states, but only from ``expected``.

        Check and write happen together deliberately. Two POSTs to /approve would
        otherwise both pass a bare state check and start ``write_story`` twice
        against the same run directory. Returns None when the job is missing or
        is in some other state, which the handler reports as 409.
        """
        job = self._jobs.get(job_id)
        if job is None or job.state is not expected:
            return None
        return self.update(job_id, state=new_state, **changes)


#: The process-wide registry. Handlers use this; tests build their own.
registry = JobRegistry()
