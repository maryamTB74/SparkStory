"""Metrics computed from a finished book, with no model involved.

Every function here is pure: a ``Story`` in, a number out, no filesystem and no
provider. That is what makes them free to run, offline in the test suite, and
impossible for a model to game -- the one property the judged half cannot have.

Grounding notes arrive as plain strings rather than this module reading a run
directory, because ``Story`` does not carry its grounding and a metric that opened
files could not be unit tested against a hand-built book. The caller does the I/O.

Every value is raw. Choosing a threshold before seeing a distribution would be
inventing the measurement it is meant to summarise.
"""

import re

from sparkstory.entities.stories import Story
from sparkstory.evals.metrics.types import DeterministicScores

# Imported rather than reimplemented: this metric and the counted `read_aloud`
# review must mean the same thing by "the same opener", or the harness and the
# critic can disagree about a book with neither being wrong.
from sparkstory.workflows.reviews import _opening_word

#: Tokens for run-matching: lowercase, punctuation dropped, apostrophes kept so
#: "cannot" and "can't" stay distinct. Applied to both sides of every comparison,
#: so quoting style can neither hide a recital nor invent one.
_WORD = re.compile(r"[a-z0-9']+")


def _words(text: str) -> list[str]:
    """Lowercased word tokens, punctuation discarded."""
    return _WORD.findall(text.lower())


def distinct_opener_ratio(story: Story) -> float:
    """Share of the book's pages that open with a distinct word.

    Reports how many *distinct* openers there are rather than how often the
    commonest one recurs. A finding that named the commonest token got satisfied
    by swapping every "Maryam" for "The" -- uniform either way -- so the property
    wanted has to be the number the metric reports.

    1.0 means every page opens differently; 0.125 on an 8-page book means they all
    open with the same word.
    """
    openers = [_opening_word(page.text) for page in story.pages]
    return len(set(openers)) / len(openers)


def question_ending_ratio(story: Story) -> float:
    """Share of pages whose last character is a question mark."""
    ends = sum(1 for page in story.pages if page.text.rstrip().endswith("?"))
    return ends / len(story.pages)


def words_per_page(story: Story) -> float:
    """Mean words per page across the book."""
    return sum(len(_words(page.text)) for page in story.pages) / len(story.pages)


def beats_per_page(story: Story) -> float:
    """Beats per page. Below 1 leaves the plot planner room to pace a beat."""
    return len(story.outline.beats) / len(story.pages)


def longest_shared_run(note: str, targets: list[str]) -> int:
    """Longest run of consecutive words the note shares with any target.

    Consecutive rather than set overlap: a note about wings and air shares
    vocabulary with any story about flying, so counting shared *words* would
    report recital everywhere. Six words in the same order is a paste.

    Args:
        note: A grounding ``story_note``.
        targets: Texts to search -- beat summaries, or page prose.

    Returns:
        Length in words of the longest shared consecutive run, 0 if none.
    """
    note_words = _words(note)
    if not note_words:
        return 0

    best = 0
    for target in targets:
        target_words = _words(target)
        # Longest-first, stopping at `best`: a target that cannot beat the
        # incumbent is abandoned without enumerating its short runs, which makes
        # the common no-overlap case cheap.
        for length in range(min(len(note_words), len(target_words)), best, -1):
            windows = {
                tuple(target_words[i : i + length])
                for i in range(len(target_words) - length + 1)
            }
            if any(
                tuple(note_words[j : j + length]) in windows
                for j in range(len(note_words) - length + 1)
            ):
                best = length
                break
    return best


def _recital(texts: list[str], notes: list[str]) -> int | None:
    """Longest verbatim run any note shares with these texts.

    The maximum across notes, not the mean: one recited fact is the defect
    whatever the other notes did.
    """
    if not notes:
        return None
    return max(longest_shared_run(note, texts) for note in notes)


def fact_recital_beats(story: Story, notes: list[str]) -> int | None:
    """Longest verbatim run from a grounding note in the outline's beats.

    Reads ``summary`` only. A beat's ``title`` and ``characters_present`` are
    labels, so counting them would let a character named after a fact register as
    a recital.
    """
    return _recital([beat.summary for beat in story.outline.beats], notes)


def fact_recital_prose(story: Story, notes: list[str]) -> int | None:
    """Longest verbatim run from a grounding note in the printed prose.

    Separate from the beats number because the fixes differ: a refrain belongs to
    the Writer, so "the planner pasted it" and "the Writer pasted it" must not
    share a value. One live run pasted a note into three beats and had it rescued
    in prose, which is two numbers, not one.
    """
    return _recital([page.text for page in story.pages], notes)


def deterministic_scores(story: Story, notes: list[str]) -> DeterministicScores:
    """Every computed metric for one book."""
    return DeterministicScores(
        distinct_opener_ratio=distinct_opener_ratio(story),
        question_ending_ratio=question_ending_ratio(story),
        words_per_page=words_per_page(story),
        beats_per_page=beats_per_page(story),
        fact_recital_beats=fact_recital_beats(story, notes),
        fact_recital_prose=fact_recital_prose(story, notes),
    )
