"""Pulling durable facts out of a finished book."""

import pytest

from sparkstory.entities.stories import Story
from sparkstory.memory.extract import ExtractedFact, ExtractedMemories, MemoryExtractor
from sparkstory.memory.types import MemoryKind
from sparkstory.models.fake_model import FakeModel


@pytest.fixture
def extracted() -> ExtractedMemories:
    return ExtractedMemories(
        facts=[ExtractedFact(subject="Kit", text="A fox with a white-tipped tail.")],
        episode="Maryam built a paper rocket to send Kit to the moon.",
    )


def _extractor(model: FakeModel, story: Story) -> MemoryExtractor:
    return MemoryExtractor(
        model,
        story=story,
        child_id="maryam",
        request_id="req-1",
    )


async def test_extractor_returns_its_schema(
    story: Story, extracted: ExtractedMemories
) -> None:
    extractor = _extractor(FakeModel(extracted), story)
    result = await extractor.ainvoke()
    assert result.facts[0].subject == "Kit"


async def test_the_node_binds_its_output_schema(
    story: Story, extracted: ExtractedMemories
) -> None:
    """The Node ABC binds output_schema; this asserts the contract is declared."""
    model = FakeModel(extracted)
    _extractor(model, story)
    assert model.bound_schema is ExtractedMemories


async def test_the_prose_is_what_the_model_is_shown(
    story: Story, extracted: ExtractedMemories
) -> None:
    """Facts come from the words on the page, not from the outline.

    Extracting from the outline would only re-read what the planner already
    wrote, and would miss details the Writer invented in prose -- which are often
    the memorable ones.
    """
    model = FakeModel(extracted)
    await _extractor(model, story).ainvoke()

    sent = str(model.calls[0])
    assert story.pages[0].text in sent


def test_records_carry_the_child_and_run(
    story: Story, extracted: ExtractedMemories
) -> None:
    records = _extractor(FakeModel(extracted), story).to_records(extracted)

    assert all(r.child_id == "maryam" for r in records)
    assert all(r.source_request_id == "req-1" for r in records)


def test_one_episode_and_one_record_per_fact(
    story: Story, extracted: ExtractedMemories
) -> None:
    records = _extractor(FakeModel(extracted), story).to_records(extracted)

    kinds = [r.kind for r in records]
    assert kinds.count(MemoryKind.EPISODIC) == 1, "exactly one episode per book"
    assert kinds.count(MemoryKind.SEMANTIC) == 1


def test_a_fact_keeps_its_subject_and_an_episode_has_none(
    story: Story, extracted: ExtractedMemories
) -> None:
    records = _extractor(FakeModel(extracted), story).to_records(extracted)

    fact = next(r for r in records if r.kind is MemoryKind.SEMANTIC)
    episode = next(r for r in records if r.kind is MemoryKind.EPISODIC)
    assert fact.subject == "Kit"
    assert episode.subject is None, "an episode is about a book, not a subject"


def test_an_empty_extraction_is_legitimate(story: Story) -> None:
    """A book may establish nothing worth keeping, and that must not be an error.

    The same reasoning as finding I, where "empty is fine" had to be a real
    outcome rather than a failure -- except there the instruction caused
    under-grounding, so the wording matters as much as the code path.
    """
    empty = ExtractedMemories(facts=[], episode="")
    assert _extractor(FakeModel(empty), story).to_records(empty) == []


def test_a_whitespace_only_episode_is_not_stored(story: Story) -> None:
    """A model answering with a blank line must not create an empty memory."""
    blank = ExtractedMemories(facts=[], episode="   ")
    assert _extractor(FakeModel(blank), story).to_records(blank) == []
