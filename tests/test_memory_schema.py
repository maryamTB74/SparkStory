"""Memory's two schema fields.

These prove a field EXISTS, so every fixture is built through
`model_validate`. `model_copy(update=...)` skips validation and will happily set
a field the model does not declare -- which once made a test pass before the field
it was written to prove existed.
"""

import pytest
from pydantic import ValidationError

from sparkstory.entities.stories import ChildProfile, StoryBrief, StoryOutline
from sparkstory.nodes.story_planner import render_story_brief


def test_child_id_is_optional_so_memory_is_opt_in() -> None:
    """Every existing caller must keep working, and 612 tests supply none."""
    profile = ChildProfile.model_validate({"name": "Maryam", "age": 5})
    assert profile.child_id is None


def test_child_id_rejects_path_traversal_on_the_brief() -> None:
    """The guard is the type, and it must hold where a client actually sends it."""
    with pytest.raises(ValidationError):
        ChildProfile.model_validate(
            {"name": "Maryam", "age": 5, "child_id": "../../etc"}
        )


def test_child_id_is_accepted_when_well_formed() -> None:
    profile = ChildProfile.model_validate(
        {"name": "Maryam", "age": 5, "child_id": "maryam-5"}
    )
    assert profile.child_id == "maryam-5"


def test_child_id_is_not_sent_to_the_model() -> None:
    """A storage key is not story material.

    Verified against story_planner.py:195 -- render_story_brief takes a
    StoryBrief, not a ChildProfile.
    """
    brief_text = render_story_brief(
        StoryBrief.model_validate(
            {
                "child": {"name": "Maryam", "age": 5, "child_id": "maryam-5"},
                "premise": "a fox who wants to visit the moon",
            }
        )
    )
    assert "maryam-5" not in brief_text


def test_outline_defaults_to_no_conflicts(outline: StoryOutline) -> None:
    assert outline.memory_conflicts == []


def test_outline_carries_conflicts_when_given(outline: StoryOutline) -> None:
    payload = outline.model_dump()
    payload["memory_conflicts"] = [
        {
            "subject": "Kit",
            "stored_text": "Kit has a white-tipped tail.",
            "new_text": "Kit has a bushy red tail.",
        }
    ]
    rebuilt = StoryOutline.model_validate(payload)
    assert rebuilt.memory_conflicts[0].subject == "Kit"
    assert rebuilt.memory_conflicts[0].stored_text == "Kit has a white-tipped tail."
