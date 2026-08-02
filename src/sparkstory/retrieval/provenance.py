"""Keeping only the grounding the corpus actually supports.

**Returns, never raises** -- following the split this codebase already draws
between ``workflows/validation.py`` (raises on the impossible) and
``workflows/reviews.py`` (returns for the improvable), with
``drop_unroutable_outline_reviews`` as the direct precedent. An agent that
invented a fact should cost us the fact, not the book.

Two things happen here, and the second is the more valuable.

**Unsupported grounding is dropped.** A fact citing a chunk id we never stored
cannot be checked, so it goes. A fact citing a *craft* chunk goes too: that is a
category error rather than a typo, and a nursery rhyme cited as a fact about the
world would ground the story in something untrue of it.

**Attribution is overwritten from the store.** ``source`` is not the model's to
state -- the chunk knows where it came from. In the task 1 spike the model filled
``source`` with the chunk id, and a plausible fabrication ("Encyclopaedia
Britannica, 2019") would have survived any check that only looked for *a* source.
Overwriting makes the whole class of error unreachable instead of merely visible.

What is *not* checked, and is worth stating plainly: nothing here verifies that the
``claim`` follows from the chunk's text. The id is right and the attribution is
right, but a fact could still misread its source. Judging that needs a model, and a
model judging a model is the next session's problem.
"""

from sparkstory.entities.grounding import CraftDevice, GroundedFact, StoryGrounding
from sparkstory.retrieval.chunks import SourceKind
from sparkstory.retrieval.store import LocalVectorStore
from sparkstory.utils.logging_utils import get_logger

logger = get_logger(__name__)


def drop_unprovenanced(
    grounding: StoryGrounding, store: LocalVectorStore
) -> StoryGrounding:
    """Return grounding containing only what ``store`` can vouch for.

    The input is left untouched, so a run artifact can record what the agent
    actually returned and a later reader can see what was dropped.

    Args:
        grounding: What the Researcher returned.
        store: The index the retrieval tools searched.

    Returns:
        A new :class:`StoryGrounding`. Either list may come back empty, which is a
        legitimate result rather than a failure.
    """
    kept_facts: list[GroundedFact] = []
    for item in grounding.facts:
        chunk = store.get(item.chunk_id)
        if chunk is None:
            logger.warning(
                "dropping fact citing unknown chunk %r: %r", item.chunk_id, item.claim
            )
            continue
        if chunk.source_kind is not SourceKind.FACT:
            logger.warning(
                "dropping fact citing a %s chunk %r",
                chunk.source_kind.value,
                item.chunk_id,
            )
            continue
        # `model_copy` rather than mutation, so the input stays as the agent left
        # it. The claim and the note are the agent's own work and the whole
        # point of the step; only attribution is ours to correct.
        kept_facts.append(item.model_copy(update={"source": chunk.source}))

    kept_devices: list[CraftDevice] = []
    for craft in grounding.craft_devices:
        chunk = store.get(craft.chunk_id)
        if chunk is None:
            logger.warning(
                "dropping craft device citing unknown chunk %r: %r",
                craft.chunk_id,
                craft.device,
            )
            continue
        if chunk.source_kind is not SourceKind.CRAFT:
            logger.warning(
                "dropping craft device citing a %s chunk %r",
                chunk.source_kind.value,
                craft.chunk_id,
            )
            continue
        kept_devices.append(craft)

    dropped = (len(grounding.facts) - len(kept_facts)) + (
        len(grounding.craft_devices) - len(kept_devices)
    )
    if dropped:
        logger.info(
            "provenance: kept %d fact(s) and %d device(s), dropped %d",
            len(kept_facts),
            len(kept_devices),
            dropped,
        )

    return StoryGrounding(facts=kept_facts, craft_devices=kept_devices)
