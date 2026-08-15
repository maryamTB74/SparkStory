"""Stored memory as prompt text.

Mirrors ``render_grounding``: the store returns records, and exactly one function
decides how they are worded to a model. Keeping that in one place is what makes
the "no internal vocabulary reaches the model" audit a single file to read rather
than a sweep of every node.

**The two tiers are framed oppositely, and that is the point.** A character fact
is something to *obey* -- Kit looks the way Kit looked last time. An episode is
something to *avoid* -- this child already had that story. Rendering them the same
way would invite the planner to reuse the plot it was shown, which is the exact
opposite of why episodes are kept.

**The episode wording had to be written against its own laziest reading.** The
cheapest way to satisfy "do not tell this story again" is a cosmetic swap -- the
same plot with a badger instead of a fox. The instruction therefore asks for a
different *shape*, not different
furniture. Whether that lands is a live-run question, not a test one.
"""

from sparkstory.memory.types import MemoryKind, MemoryRecord


def render_memory(records: list[MemoryRecord]) -> str:
    """Render what is known about this child, or ``""`` if nothing is.

    Returns the empty string rather than an empty heading when there is no
    memory: a first book should carry no section at all, because a heading with
    nothing under it reads as "this child has no characters" rather than "this is
    their first story".
    """
    if not records:
        return ""

    facts = [r for r in records if r.kind is MemoryKind.SEMANTIC]
    episodes = [r for r in records if r.kind is MemoryKind.EPISODIC]

    blocks: list[str] = []

    if facts:
        lines = [
            f"- {r.subject}: {r.text}" if r.subject else f"- {r.text}" for r in facts
        ]
        blocks.append(
            "This child has had stories before. These characters and details are "
            "already established, and this story must keep them exactly as they "
            "are:\n" + "\n".join(lines)
        )

    if episodes:
        lines = [f"- {r.text}" for r in episodes]
        blocks.append(
            "Stories this child has already been given. Do not tell one of these "
            "again -- give this book a different shape, not the same story with "
            "different characters:\n" + "\n".join(lines)
        )

    return "\n\n".join(blocks)
