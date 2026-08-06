"""The committed fixture briefs.

Committed rather than generated, and committed under ``tests/`` rather than
``data/``: ``data/`` is gitignored and reserved for real persistence, and a baseline
that lives only in ``outputs/`` does not survive -- most run directories on this
machine have already lost their ``story.json``, including every book the earlier
output reviews were written from.

Each brief pins ``world_rules`` explicitly. A fixture inheriting the schema default
could not be compared against its own past if that default moved, and the grounding
branch it selects is invisible in the finished book.
"""

import json
from pathlib import Path

from sparkstory.entities.stories import StoryBrief

# Depth-sensitive, and wrong counts fail by resolving somewhere plausible rather
# than by raising -- the same trap as `_PROJECT_ROOT` in config.py, whose count
# changed silently when its module moved. This file is
# src/sparkstory/evals/briefs.py, so the repo root is three parents up.
_REPO_ROOT = Path(__file__).resolve().parents[3]

#: Fixture briefs live beside the tests that assert on them.
FIXTURE_BRIEF_DIR = _REPO_ROOT / "tests" / "fixtures" / "evals" / "briefs"


def load_fixture_briefs() -> dict[str, StoryBrief]:
    """Every committed fixture brief, keyed by file stem.

    Returns:
        Briefs by name, validated. A malformed fixture raises here rather than at
        the live run that would already have paid for it.

    Raises:
        FileNotFoundError: If the fixture directory is missing, which means the
            path above resolved wrongly rather than that there are no briefs.
    """
    if not FIXTURE_BRIEF_DIR.is_dir():
        raise FileNotFoundError(f"no fixture brief directory at {FIXTURE_BRIEF_DIR}")

    return {
        path.stem: StoryBrief.model_validate(
            json.loads(path.read_text(encoding="utf-8"))
        )
        for path in sorted(FIXTURE_BRIEF_DIR.glob("*.json"))
    }
