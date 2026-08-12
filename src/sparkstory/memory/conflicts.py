"""Which new facts disagree with what is already stored.

**Deliberately not a model call.** "Is this the same fact?" looks like a job for
an LLM, and rule 13 is the argument against: the laziest answer to "do these
agree?" is *yes*, and a false negative here silently drops the contradiction this
package exists to surface. So the rule is mechanical, and it over-reports rather
than under-reports -- a parent dismissing a non-conflict costs a glance, while a
missed one costs the Finn/Kit bug.

**The accepted cost of that choice.** Any two differently-worded facts about one
subject are reported, so "A fox." and "A small fox." is a conflict. That will be
noisy, and the noise is visible to a parent rather than hidden -- which is the
right way round for a check whose failure mode is silence. If it proves too noisy
in Task 9's live run, the fix is a narrower *comparison*, not a model.

**What this deliberately does not do:** decide who is right. It reports the
disagreement and the parent decides. Nothing here calls ``supersede``.
"""

from sparkstory.memory.types import MemoryConflict, MemoryKind, MemoryRecord


def _normalise(text: str) -> str:
    """Fold the differences that are not disagreements: case and edge whitespace."""
    return " ".join(text.lower().split())


def find_conflicts(
    new: list[MemoryRecord], stored: list[MemoryRecord]
) -> list[MemoryConflict]:
    """Report every new semantic fact that restates a subject differently.

    Semantic records only. Two books that both visit the moon are a *repetition*,
    which the episodic tier exists to discourage -- not a contradiction, because
    neither book claims the other is wrong.

    Every disagreeing stored fact yields its own conflict rather than only the
    first: append-only means a subject accumulates rows, and a parent choosing
    between descriptions needs to see all of them.

    Args:
        new: Records extracted from the book just finished.
        stored: What was already known, from ``PgMemoryStore.fetch``.

    Returns:
        One entry per (new fact, disagreeing stored fact) pair. Empty when the
        subject is new, when the wording matches, or for episodic records.
    """
    by_subject: dict[str, list[MemoryRecord]] = {}
    for record in stored:
        if record.kind is MemoryKind.SEMANTIC and record.subject:
            by_subject.setdefault(record.subject, []).append(record)

    conflicts: list[MemoryConflict] = []
    for candidate in new:
        if candidate.kind is not MemoryKind.SEMANTIC or not candidate.subject:
            continue
        for existing in by_subject.get(candidate.subject, []):
            if _normalise(existing.text) != _normalise(candidate.text):
                conflicts.append(
                    MemoryConflict(
                        subject=candidate.subject,
                        stored_text=existing.text,
                        new_text=candidate.text,
                    )
                )
    return conflicts
