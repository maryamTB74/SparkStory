"""Tests for metrics computed without a model.

Ordering matters more than equality here. ``FakeEmbedder``'s hash bug passed its
own "shared words score higher" test and was caught by a *ranking* assertion, so
every metric below is also checked on whether two books come out in the right
order.
"""

from collections.abc import Callable

from sparkstory.entities.stories import Story
from sparkstory.evals.metrics import deterministic as det

FOUR_PAGES = ["The fox ran.", "The bird sang.", "The moon rose.", "Kit slept."]


def test_distinct_opener_ratio_counts_distinct_first_words(
    book_factory: Callable[..., Story],
) -> None:
    build = book_factory
    # Three pages open with "The", one with "Kit" -> 2 distinct of 4.
    assert det.distinct_opener_ratio(build(FOUR_PAGES)) == 0.5


def test_distinct_opener_ratio_ignores_case_and_punctuation(
    book_factory: Callable[..., Story],
) -> None:
    build = book_factory
    # "The" and "the," are the same drone read aloud.
    story = build(["The fox ran.", "the, bird sang.", "A cat sat.", "A dog ran."])
    assert det.distinct_opener_ratio(story) == 0.5


def test_distinct_opener_ratio_orders_varied_above_uniform(
    book_factory: Callable[..., Story],
) -> None:
    build = book_factory
    uniform = build(["The a b.", "The c d.", "The e f.", "The g h."])
    varied = build(["Ada ran.", "Birds sang.", "Clouds moved.", "Dusk fell."])
    assert det.distinct_opener_ratio(varied) > det.distinct_opener_ratio(uniform)


def test_distinct_opener_ratio_is_not_fooled_by_swapping_one_uniform_opener(
    book_factory: Callable[..., Story],
) -> None:
    build = book_factory
    # The failure this metric exists for: a book that swapped every "Maryam" for
    # "The" is exactly as uniform, and a metric reporting the commonest token
    # would have called that a fix.
    before = build(["Maryam a.", "Maryam b.", "Maryam c.", "Maryam d."])
    after = build(["The a.", "The b.", "The c.", "The d."])
    assert det.distinct_opener_ratio(before) == det.distinct_opener_ratio(after)


def test_question_ending_ratio(book_factory: Callable[..., Story]) -> None:
    build = book_factory
    story = build(["Where now?", "He ran.", "Why not?", "She slept."])
    assert det.question_ending_ratio(story) == 0.5


def test_question_ending_ratio_ignores_trailing_whitespace(
    book_factory: Callable[..., Story],
) -> None:
    build = book_factory
    story = build(["Where now?  \n", "Who else? ", "Why not?\t", "How far?"])
    assert det.question_ending_ratio(story) == 1.0


def test_words_per_page(book_factory: Callable[..., Story]) -> None:
    build = book_factory
    story = build(["one two three", "four five", "six seven", "eight nine ten"])
    assert det.words_per_page(story) == 2.5


def test_beats_per_page(book_factory: Callable[..., Story]) -> None:
    build = book_factory
    story = build(
        ["a", "b", "c", "d", "e", "f", "g", "h"],
        beat_summaries=["S one here.", "S two here.", "S three ok.", "S four ok."],
    )
    assert det.beats_per_page(story) == 0.5


def test_longest_shared_run_finds_consecutive_words() -> None:
    assert det.longest_shared_run("wings need air to push", ["his wings need air"]) == 3


def test_longest_shared_run_requires_consecutive_words() -> None:
    # Every word present, none adjacent in the same order.
    assert det.longest_shared_run("air wings push", ["push the wings through air"]) == 1


def test_longest_shared_run_ignores_case_and_punctuation() -> None:
    assert det.longest_shared_run("no air", ['"No, air!" she said']) == 2


def test_longest_shared_run_is_zero_for_no_overlap() -> None:
    assert det.longest_shared_run("wings need air", ["a fox slept"]) == 0


def test_fact_recital_ranks_verbatim_above_incidental_overlap(
    book_factory: Callable[..., Story],
) -> None:
    build = book_factory
    # The ordering a length-normalised ratio got wrong on two real runs: a genuine
    # recital drawn from a long note must outrank a short innocent overlap.
    recited_note = "Wings need air to push against, so an eagle cannot fly here"
    innocent_note = "Nothing outdoors on the moon can flutter, drift or make a sound"
    four = ["a", "b", "c", "d"]
    recital = build(
        four, beat_summaries=["Wings need air to push against, he learns."] * 4
    )
    innocent = build(four, beat_summaries=["They walk on the moon together now."] * 4)
    assert det.fact_recital_beats(recital, [recited_note]) > det.fact_recital_beats(
        innocent, [innocent_note]
    )


def test_fact_recital_splits_beats_from_prose(
    book_factory: Callable[..., Story],
) -> None:
    build = book_factory
    # The planner recited it and the Writer did not: two defects, two numbers.
    note = "wings need air to push against"
    story = build(
        ["His wings found only empty space.", "b", "c", "d"],
        beat_summaries=["Wings need air to push against, he learns."] * 4,
    )
    assert det.fact_recital_beats(story, [note]) == 6
    assert det.fact_recital_prose(story, [note]) < 3


def test_fact_recital_takes_the_maximum_across_notes(
    book_factory: Callable[..., Story],
) -> None:
    build = book_factory
    story = build(
        ["a", "b", "c", "d"],
        beat_summaries=["Wings need air to push against, ok."] * 4,
    )
    notes = ["a fox likes plums", "wings need air to push against"]
    assert det.fact_recital_beats(story, notes) == 6


def test_fact_recital_is_none_without_notes(book_factory: Callable[..., Story]) -> None:
    build = book_factory
    # No notes means nothing could be recited -- absence of a measurement, which
    # is not the same as a clean score of zero.
    story = build(FOUR_PAGES)
    assert det.fact_recital_beats(story, []) is None
    assert det.fact_recital_prose(story, []) is None


def test_deterministic_scores_reports_every_metric(
    book_factory: Callable[..., Story],
) -> None:
    build = book_factory
    scores = det.deterministic_scores(build(FOUR_PAGES), [])
    assert scores.distinct_opener_ratio == 0.5
    # 3 + 3 + 3 + 2 words over 4 pages.
    assert scores.words_per_page == 2.75
    assert scores.beats_per_page == 1.0
    assert scores.fact_recital_beats is None
