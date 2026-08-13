"""Inputs a workflow is invoked with.

A workflow takes one input value,
so anything it needs travels in a single typed mapping.
"""

from typing import TypedDict

from sparkstory.entities.stories import Story, StoryBrief, StoryOutline


class StoryWorkflowInput(TypedDict):
    """What ``story_workflow`` is invoked with.

    ``request_id`` is supplied by the *caller* rather than generated inside the
    workflow, and that is not incidental. A LangGraph ``@entrypoint`` body
    re-executes when a run resumes, while ``@task`` results are replayed from the
    checkpoint -- so an id minted inside the body would change on every resume.
    Once a later session pauses a run for a parent's confirmation, that resume is
    the normal path, and a correlation id that changes each time is worse than
    none. Minting it outside makes it stable by construction.

    ``outline`` arrives from the caller -- from ``plan_story``, and in the MCP
    flow from a plan a parent approved. The workflow never plans.
    """

    request_id: str
    brief: StoryBrief
    outline: StoryOutline


class OutlineWorkflowInput(TypedDict):
    """What ``outline_workflow`` is invoked with.

    ``request_id`` is supplied by the caller for the same reason as in
    :class:`StoryWorkflowInput`: an ``@entrypoint`` body re-executes on resume,
    so an id minted inside it would change every time.
    """

    request_id: str
    brief: StoryBrief


class NarrationWorkflowInput(TypedDict):
    """What ``narration_workflow`` is invoked with.

    ``story`` is a *finished* book and is never modified, for the same reason
    illustration never modifies one: narration is a separate entrypoint precisely
    so a provider failure cannot damage prose that already passed both critics.

    ``brief`` is here for exactly two fields -- ``voice`` and the child's
    ``reading_level``, which set who reads and how fast. Nothing else about the
    brief reaches the provider, because the script is the page text.

    ``directory`` is a ``str`` rather than a ``Path`` for the reason
    :class:`IllustrationWorkflowInput` records: a workflow input has to survive a
    checkpointer, and a ``Path`` works against an in-memory saver while failing
    against SQLite.
    """

    request_id: str
    brief: StoryBrief
    story: Story
    directory: str


class IllustrationWorkflowInput(TypedDict):
    """What ``illustration_workflow`` is invoked with.

    ``story`` is a *finished* book and is never modified. Illustration is a
    separate entrypoint precisely so a failure here cannot damage prose that
    already passed both critics.

    ``directory`` is a ``str`` rather than a ``Path``, and that is not cosmetic:
    this mapping is a workflow input, so a checkpointer has to serialise it. A
    ``Path`` survives an in-memory saver and fails against a SQLite one, which is
    the kind of defect that appears only once resumable runs arrive. The tasks
    rebuild a ``Path`` at the point of use.
    """

    request_id: str
    brief: StoryBrief
    story: Story
    directory: str
