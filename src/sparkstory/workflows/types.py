"""Inputs a workflow is invoked with.

A workflow takes one input value,
so anything it needs travels in a single typed mapping.
"""

from typing import TypedDict

from sparkstory.entities.stories import StoryBrief


class StoryWorkflowInput(TypedDict):
    """What ``story_workflow`` is invoked with.

    ``request_id`` is supplied by the *caller* rather than generated inside the
    workflow, and that is not incidental. A LangGraph ``@entrypoint`` body
    re-executes when a run resumes, while ``@task`` results are replayed from the
    checkpoint -- so an id minted inside the body would change on every resume.
    Once a later session pauses a run for a parent's confirmation, that resume is
    the normal path, and a correlation id that changes each time is worse than
    none. Minting it outside makes it stable by construction.
    """

    request_id: str
    brief: StoryBrief
