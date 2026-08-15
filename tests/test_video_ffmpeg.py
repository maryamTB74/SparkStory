"""The subprocess boundary: does a missing binary fail early and clearly?

Offline: ``shutil.which`` is patched, so these run identically on a machine with
ffmpeg and one without. The tests that need the real binary are ``video``-marked
and live in ``test_video_live.py``.
"""

import pytest

from sparkstory.entities.exceptions import VideoConfigurationError
from sparkstory.video import ffmpeg as ffmpeg_module


def test_a_missing_binary_names_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """Checked once before any page, so the message must say what to install.

    ``run_narration_pipeline`` makes the same call for a missing API key: every
    page would fail identically, so failing after doing the work tells nobody
    anything they could not have known up front.
    """
    monkeypatch.setattr(ffmpeg_module.shutil, "which", lambda _name: None)

    with pytest.raises(VideoConfigurationError) as caught:
        ffmpeg_module.require_ffmpeg()

    message = str(caught.value)
    assert "ffmpeg" in message
    assert "apt install" in message, "the message must be actionable"


def test_it_is_satisfied_when_both_binaries_are_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ffprobe is checked too: the frame-count assertions need it, and finding it
    missing at measurement time rather than at start-up is the same trap."""
    monkeypatch.setattr(ffmpeg_module.shutil, "which", lambda name: f"/usr/bin/{name}")

    ffmpeg_module.require_ffmpeg()


def test_a_missing_ffprobe_alone_is_still_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The interesting half: ffmpeg present, ffprobe absent.

    Without this, a machine with only ffmpeg passes the check and then fails at
    the first measurement -- after every clip has been encoded.
    """
    monkeypatch.setattr(
        ffmpeg_module.shutil,
        "which",
        lambda name: "/usr/bin/ffmpeg" if name == "ffmpeg" else None,
    )

    with pytest.raises(VideoConfigurationError) as caught:
        ffmpeg_module.require_ffmpeg()

    assert "ffprobe" in str(caught.value)
