"""What a video run recorded, and what it refused to claim."""

from pathlib import Path

from sparkstory.entities.video import StoryVideo, VideoItem, VideoStatus


def _item(page: int, status: VideoStatus, duration: float | None = 1.0) -> VideoItem:
    return VideoItem(page_number=page, status=status, duration=duration, reason=None)


def test_pages_animated_counts_only_animated() -> None:
    video = StoryVideo(
        path=Path("story.mp4"),
        fps=30,
        items=[
            _item(1, VideoStatus.ANIMATED),
            _item(2, VideoStatus.HELD),
            _item(3, VideoStatus.EXCLUDED, None),
        ],
    )
    assert video.pages_animated == 1


def test_is_complete_is_false_when_a_page_was_excluded() -> None:
    video = StoryVideo(
        path=Path("story.mp4"),
        fps=30,
        items=[
            _item(1, VideoStatus.ANIMATED),
            _item(2, VideoStatus.EXCLUDED, None),
        ],
    )
    assert video.is_complete is False


def test_a_held_page_is_not_a_failure_but_is_not_animated() -> None:
    """HELD and ANIMATED are both successes and are not the same success.

    A book that is entirely HELD is blank cards with narration over it. The
    consistency judge's spec makes this argument at its section 6a: collapsing
    "the mechanism ran" into "the mechanism worked" cost three live runs.
    """
    video = StoryVideo(
        path=Path("story.mp4"),
        fps=30,
        items=[_item(1, VideoStatus.HELD), _item(2, VideoStatus.HELD)],
    )
    assert video.is_complete is True
    assert video.pages_animated == 0


def test_an_empty_run_is_not_complete() -> None:
    """``all([])`` is True, so without a guard a run that did nothing reports done.

    A check with no room to fail, in the direction that looks like success.
    ``StoryNarration.is_complete`` carries the same guard for the same reason.
    """
    video = StoryVideo(path=None, fps=30, items=[])
    assert video.is_complete is False
