"""Build the LangChain callback that sends spans to Opik."""

import logging
from typing import Any

from sparkstory.config import settings
from sparkstory.observability.opik_utils import configure

logger = logging.getLogger(__name__)


def trace_metadata() -> dict[str, Any]:
    """The settings that distinguish one run from another.

    Deliberately not every setting. These are the values a reader comparing two
    traces would need in order to say what was different about them.
    """
    return {
        "planner_model": settings.planner_model,
        "plot_model": settings.plot_model,
        "writer_model": settings.writer_model,
        "researcher_model": settings.researcher_model,
        "outline_critic_model": settings.outline_critic_model,
        "prose_critic_model": settings.prose_critic_model,
        "max_outline_revisions": settings.max_outline_revisions,
        "max_prose_revisions": settings.max_prose_revisions,
    }


def build_handler(request_id: str, tags: list[str] | None = None) -> Any | None:
    """A callback that sends this run's spans to Opik, or None.

    Args:
        request_id: The run's id, reused as Opik's thread id so that a log line
            and a trace share a value.
        tags: Optional labels, e.g. ``["plan_outline"]`` or ``["eval"]``.

    Returns:
        An ``OpikTracer``, or None when tracing is off or unavailable. Callers
        filter None out of the callback list, so there is nothing to branch on.
    """
    if not configure():
        return None

    # Imported here, not at module scope: see the package docstring. There is a
    # test asserting the disabled path leaves sys.modules untouched.
    from opik.integrations.langchain import OpikTracer

    try:
        return OpikTracer(
            tags=tags,
            thread_id=request_id,
            metadata=trace_metadata(),
        )
    # A tracer we cannot build is a trace we do not get, not a run we lose.
    except Exception as error:  # noqa: BLE001
        logger.warning("Could not build the Opik tracer, continuing: %s", error)
        return None
