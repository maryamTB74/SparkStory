"""The camera move is arithmetic, and the arithmetic has to be exact."""

from sparkstory.video.kenburns import FPS, Move, frames_for, move_for


def test_the_move_cycles_through_four() -> None:
    """A pure function of the page number, so a re-render is reproducible.

    Not a model call: an instruction gets satisfied the laziest legal way, and the
    laziest answer to "pick a camera move" is "zoom in" every time -- a paid call to
    recover the fixed option with added noise. Not random either -- that forfeits
    reproducibility, which is what lets the assembly test assert anything real.
    """
    assert move_for(1) is Move.ZOOM_IN
    assert move_for(2) is Move.ZOOM_OUT
    assert move_for(3) is Move.PAN_RIGHT
    assert move_for(4) is Move.PAN_LEFT
    assert move_for(5) is Move.ZOOM_IN


def test_frames_are_rounded_not_truncated() -> None:
    """Truncation loses up to a frame per page, and the error accumulates.

    Six pages a frame short is a fifth of a second of drift by the end, which is
    audible against narration. ``int()`` here would be the defect.
    """
    assert frames_for(1.0) == FPS
    assert frames_for(10.32) == round(10.32 * FPS)
    # Half a frame rounds up rather than away.
    assert frames_for(2.0 + (0.6 / FPS)) == FPS * 2 + 1


def test_a_zero_length_page_still_gets_one_frame() -> None:
    """A clip of zero frames is not a clip: ffmpeg would emit an empty file and
    the concat would silently lose the page."""
    assert frames_for(0.0) == 1


def test_every_move_is_reachable() -> None:
    """A check with no room to fail proves nothing: a fifth move nobody can
    reach would look like variety and produce none."""
    assert {move_for(n) for n in range(1, 9)} == set(Move)
