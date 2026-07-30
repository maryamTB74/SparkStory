"""Writing guidance shared by the agents that generate text.

This lives in ``entities`` rather than on a node because more than one node needs
it: the planner sizes its beats by reading level, and the writer writes to it.
Guidance that only one node uses belongs on that node, beside its prompt.

The text here **reaches the model**, so it is written as instruction. Engineering
rationale goes in ``#`` comments.
"""

from sparkstory.entities.stories import ReadingLevel

#: Per-level writing guidance, keyed by enum rather than embedded in prose so
#: that adding a ReadingLevel cannot silently miss the prompt.
READING_LEVEL_GUIDANCE: dict[ReadingLevel, str] = {
    ReadingLevel.PRE_READER: (
        "Read aloud by an adult. Very short sentences, strong rhythm, lots of "
        "repetition. Concrete objects and actions only, no abstractions."
    ),
    ReadingLevel.EARLY_READER: (
        "Short simple sentences, mostly familiar words. One idea per sentence. "
        "Repetition and sound-play are welcome."
    ),
    ReadingLevel.DEVELOPING: (
        "Longer sentences are fine, and a few new words are good if their "
        "meaning is clear from context. Some simple dialogue."
    ),
    ReadingLevel.CONFIDENT: (
        "Short paragraphs, richer vocabulary, and more interior feeling. "
        "Subplots are acceptable if they resolve."
    ),
}
