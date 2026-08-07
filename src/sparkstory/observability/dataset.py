"""Publish the committed fixture briefs as an Opik dataset.

The local JSON stays the source of truth. This is a second display of numbers
that already exist, never a replacement -- if the two disagree, the local
scorecards are right, because they are what the offline tests cover.

Unlike the rest of this package, the upload **raises** rather than warning. The
difference is who asked: tracing is a side effect nobody requested, so it
degrades silently, while ``--opik`` is an explicit instruction and a flag that
quietly does nothing is the failure this project has recorded twice.
"""

import logging
from typing import Any

from sparkstory.config import settings
from sparkstory.evals.briefs import load_fixture_briefs
from sparkstory.observability.opik_utils import configure

logger = logging.getLogger(__name__)

#: Measured, not assumed. The same five books judged twice at temperature 0.0
#: moved by up to 0.25 on `delight` -- two pages of an eight-page book. See
#: tests/fixtures/evals/baseline/2026-08-04/README.md.
JUDGED_RESOLUTION_LIMIT = 0.25

_VARIANCE_NOTE = (
    "Judged scores were measured twice on identical books at temperature 0.0 "
    "and moved by up to 0.25. A judged difference below 0.25 is not evidence. "
    "The deterministic metrics are arithmetic and reproduce exactly."
)

#: The Opik dataset the fixture briefs are published to.
DATASET_NAME = "sparkstory-fixture-briefs"

_DATASET_DESCRIPTION = (
    "SparkStory fixture briefs. Each pins world_rules explicitly, so a run is "
    "comparable against its own past even if the schema default moves."
)


def experiment_config() -> dict[str, Any]:
    """What an experiment run must record alongside its scores.

    The resolution limit is in here rather than only in a README because Opik's
    experiment view will happily plot a 0.1 movement as a trend line, and a
    caveat that does not travel with the data is a caveat nobody reads.
    """
    return {
        "planner_model": settings.planner_model,
        "writer_model": settings.writer_model,
        "outline_critic_model": settings.outline_critic_model,
        "prose_critic_model": settings.prose_critic_model,
        "judge_model": settings.judge_model,
        "judged_resolution_limit": JUDGED_RESOLUTION_LIMIT,
        "judge_variance_note": _VARIANCE_NOTE,
    }


def upload_fixture_briefs(dataset_name: str = DATASET_NAME) -> int:
    """Upload every committed fixture brief as an Opik dataset item.

    Args:
        dataset_name: The Opik dataset to create or replace.

    Returns:
        How many items were uploaded.

    Raises:
        RuntimeError: Tracing is disabled, or Opik could not be configured.
    """
    if not settings.opik_enabled:
        raise RuntimeError("Uploading needs OPIK_ENABLED=true and an OPIK_API_KEY.")
    if not configure():
        raise RuntimeError(
            "Opik is enabled but could not be configured; see the log above."
        )

    # Imported here, not at module scope: see the package docstring.
    import opik

    briefs = load_fixture_briefs()
    items = [
        {"name": name, "brief": brief.model_dump(mode="json")}
        for name, brief in briefs.items()
    ]

    dataset = opik.Opik().get_or_create_dataset(
        name=dataset_name,
        description=_DATASET_DESCRIPTION,
    )
    # Cleared first so that re-uploading replaces rather than accumulates. Five
    # briefs uploaded twice must stay five items, not become ten.
    dataset.clear()
    dataset.insert(items)

    logger.info(
        "Uploaded %d fixture briefs to Opik dataset %r", len(items), dataset_name
    )
    return len(items)
