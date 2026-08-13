"""Narration's domain types: the voice a parent picks, and the errors we raise."""

import pytest
from pydantic import ValidationError

from sparkstory.entities.exceptions import (
    AudioConfigurationError,
    AudioGenerationError,
    ConfigurationError,
    SparkStoryError,
)
from sparkstory.entities.stories import ChildProfile, StoryBrief, Voice


def test_voice_has_exactly_two_values() -> None:
    # Two, not four. The provider's roster carries no expressive metadata --
    # only `gender` -- so a `warm`/`gentle` split would be invented rather
    # than mapped. Spec section 5.
    assert {v.value for v in Voice} == {"female", "male"}


def test_brief_defaults_to_a_female_voice() -> None:
    # Optional with a default, so all pre-existing briefs keep working -- the
    # lesson `world_rules` taught when it changed behaviour for every caller.
    brief = StoryBrief(
        child=ChildProfile(name="Maryam", age=5),
        premise="a fox who wants to visit the moon",
    )
    assert brief.voice is Voice.FEMALE


def test_brief_accepts_an_explicit_voice() -> None:
    brief = StoryBrief(
        child=ChildProfile(name="Maryam", age=5),
        premise="a fox who wants to visit the moon",
        voice=Voice.MALE,
    )
    assert brief.voice is Voice.MALE


def test_brief_rejects_an_unknown_voice() -> None:
    with pytest.raises(ValidationError):
        StoryBrief(
            child=ChildProfile(name="Maryam", age=5),
            premise="a fox",
            voice="robot",  # type: ignore[arg-type]
        )


def test_audio_errors_sit_in_the_right_places() -> None:
    # Generation failures are retryable; configuration failures are not, and
    # `workflows/retries.py` branches on exactly this distinction (rule 10).
    assert issubclass(AudioGenerationError, SparkStoryError)
    assert issubclass(AudioConfigurationError, ConfigurationError)
    assert not issubclass(AudioGenerationError, ConfigurationError)
