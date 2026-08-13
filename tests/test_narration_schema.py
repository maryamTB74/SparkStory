"""What a narration run actually produced -- our record, never a model's output."""

from pathlib import Path

from sparkstory.entities.narration import (
    NarrationItem,
    NarrationStatus,
    StoryNarration,
)

_HASH = "a" * 64


def _item(page: int, status: NarrationStatus, path: Path | None) -> NarrationItem:
    return NarrationItem(page_number=page, status=status, path=path, sha256=_HASH)


def test_pages_narrated_counts_only_successes() -> None:
    # "6 of 8" has to be readable from a file, or a partially narrated book
    # looks identical to a complete one -- finding N's failure mode.
    narration = StoryNarration(
        voice_id="eve",
        speed=0.9,
        items=[
            _item(1, NarrationStatus.NARRATED, Path("page-01.mp3")),
            _item(2, NarrationStatus.FAILED, None),
            _item(3, NarrationStatus.NARRATED, Path("page-03.mp3")),
        ],
        stitched=Path("story.mp3"),
    )
    assert narration.pages_narrated == 2
    assert len(narration.items) == 3
    assert narration.is_complete is False


def test_is_complete_requires_every_page_and_a_non_empty_book() -> None:
    both = StoryNarration(
        voice_id="eve",
        speed=1.0,
        items=[
            _item(1, NarrationStatus.NARRATED, Path("page-01.mp3")),
            _item(2, NarrationStatus.NARRATED, Path("page-02.mp3")),
        ],
        stitched=Path("story.mp3"),
    )
    assert both.is_complete is True

    # An empty run is NOT complete. `all([])` is True, which would report a book
    # that narrated nothing as fully narrated -- rule 24, a check with no room
    # to fail.
    empty = StoryNarration(voice_id="eve", speed=1.0, items=[], stitched=None)
    assert empty.is_complete is False
    assert empty.pages_narrated == 0


def test_page_audio_returns_the_path_for_a_narrated_page() -> None:
    narration = StoryNarration(
        voice_id="leo",
        speed=1.0,
        items=[
            _item(1, NarrationStatus.NARRATED, Path("page-01.mp3")),
            _item(2, NarrationStatus.FAILED, None),
        ],
        stitched=None,
    )
    assert narration.page_audio(1) == Path("page-01.mp3")
    assert narration.page_audio(2) is None
    assert narration.page_audio(99) is None


def test_narration_carries_no_prompt_text() -> None:
    # This module is our record. If it ever gained a field a model fills in, a
    # model would be writing into the data we use to decide whether the feature
    # worked -- the split `entities/illustration.py` documents.
    schema = StoryNarration.model_json_schema()
    dumped = str(schema).lower()
    for term in ("you are", "your task", "prompt", "instruction"):
        assert term not in dumped
