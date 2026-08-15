"""The committed fixture briefs, and the two known-defect stories.

The story tests here are the *instrument* test: they check that the metrics detect
defects already known to be in a book, rather than checking a book. A measurement
that cannot fail proves nothing, and this project has twice recorded a live run as
a pass when the comparison it made was vacuous.
"""

import json
from pathlib import Path

from sparkstory.entities.stories import Story, WorldRules
from sparkstory.evals.briefs import FIXTURE_BRIEF_DIR, load_fixture_briefs
from sparkstory.evals.metrics import deterministic as det

_STORIES = Path(__file__).parent / "fixtures" / "evals" / "stories"
#: The note that was pasted into three beats of a real run.
_NOTE = "wings need air to push against"


def _fixture_story(name: str) -> Story:
    return Story.model_validate_json((_STORIES / f"{name}.json").read_text())


def test_five_briefs_load_and_validate() -> None:
    assert set(load_fixture_briefs()) == {
        "moon-fox",
        "eagle-planet",
        "seed-garden",
        "teddy-bear",
        "submarine",
    }


def test_every_brief_pins_world_rules_explicitly() -> None:
    # A fixture leaving world_rules to the schema default could not be compared
    # against its own past if that default moved, and the branch it selects is
    # invisible in the finished book.
    for name in load_fixture_briefs():
        raw = json.loads((FIXTURE_BRIEF_DIR / f"{name}.json").read_text())
        assert "world_rules" in raw, f"{name} does not pin world_rules"


def test_both_world_rules_modes_are_represented() -> None:
    modes = {brief.world_rules for brief in load_fixture_briefs().values()}
    assert modes == {WorldRules.REALISTIC, WorldRules.IMAGINATIVE}


def test_briefs_cover_a_premise_with_no_factual_spine() -> None:
    # The teddy-bear premise consulted craft and never called search_facts, which
    # is the case a set of only factual premises would miss.
    assert "teddy" in load_fixture_briefs()["teddy-bear"].premise.lower()


def test_uniform_opener_fixture_scores_low_on_distinct_openers() -> None:
    # A uniform-opener book, reconstructed: the original is no longer on disk.
    story = _fixture_story("uniform-openers")
    assert det.distinct_opener_ratio(story) <= 0.625


def test_recited_fact_fixture_scores_high_on_beat_recital() -> None:
    # Fact recital: a story_note pasted verbatim into three beats.
    assert det.fact_recital_beats(_fixture_story("recited-fact"), [_NOTE]) >= 6


def test_the_two_fixtures_isolate_different_defects() -> None:
    # If both scored alike on both metrics the pair could not fail, and a passing
    # comparison at a saturated measurement is exactly the trap that let a
    # zero-fact control read as a successful run.
    uniform = _fixture_story("uniform-openers")
    recited = _fixture_story("recited-fact")
    assert det.fact_recital_beats(recited, [_NOTE]) > det.fact_recital_beats(
        uniform, [_NOTE]
    )
    assert det.distinct_opener_ratio(uniform) > det.distinct_opener_ratio(recited)
