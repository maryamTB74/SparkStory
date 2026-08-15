"""The fifth seam: does it name what it knows, and refuse what it does not?"""

import pytest

from sparkstory.entities.exceptions import VideoConfigurationError
from sparkstory.models.get_clip_maker import KNOWN_MAKERS, get_clip_maker


def test_an_unknown_maker_names_the_ones_it_knows() -> None:
    """The message has to be actionable -- it is a ConfigurationError."""
    with pytest.raises(VideoConfigurationError) as caught:
        get_clip_maker("runway")

    message = str(caught.value)
    assert "runway" in message
    assert "kenburns" in message


def test_there_is_exactly_one_maker_this_session() -> None:
    """One implementation means nothing to select between.

    A ``video_provider`` setting would be config for a feature that does not
    exist. It arrives with build 2, alongside the second entry.
    """
    assert frozenset({"kenburns"}) == KNOWN_MAKERS
