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

from typing import TYPE_CHECKING

from sparkstory.entities.grounding import GroundedFact, StoryGrounding
from sparkstory.retrieval.chunks import SourceKind
from sparkstory.retrieval.protocol import ChunkStore
from sparkstory.retrieval.web.ledger import WEB_ID_PREFIX
from sparkstory.utils.logging_utils import get_logger

if TYPE_CHECKING:
    from sparkstory.retrieval.web.ledger import WebLedger

logger = get_logger(__name__)


def _keep_web_fact(item: GroundedFact, ledger: WebLedger | None) -> GroundedFact | None:
    """Decide a web-cited fact, mirroring the corpus rules one for one."""
    if ledger is None:
        logger.warning(
            "dropping fact citing the web %r with no sources consulted: %r",
            item.chunk_id,
            item.claim,
        )
        return None

    source = ledger.get(item.chunk_id)
    if source is None:
        logger.warning(
            "dropping fact citing unknown source %r: %r", item.chunk_id, item.claim
        )
        return None

    if not source.verified:
        # Present but unchecked. This is what makes VERIFY_WEB_CLAIMS=false safe
        # to offer: the source is recorded honestly and refused here, rather than
        # quietly grounding a book on something nothing confirmed.
        logger.warning(
            "dropping fact citing an unchecked source %r: %s", item.chunk_id, source.url
        )
        return None

    # `source` is ours to state, not the model's -- same rule as the store.
    return item.model_copy(update={"source": source.url})


def drop_unprovenanced(
    grounding: StoryGrounding,
    store: ChunkStore,
    ledger: WebLedger | None = None,
) -> StoryGrounding:
    """Return grounding containing only what ``store`` or ``ledger`` can vouch for.

    The input is left untouched, so a run artifact can record what the agent
    actually returned and a later reader can see what was dropped.

    Args:
        grounding: What the Researcher returned.
        store: The index the retrieval tools searched.
        ledger: Web sources this run fetched and checked, when the web tool was
            enabled. ``None`` means it was off, in which case **a web id resolves
            to nothing and the fact is dropped** -- an agent that invented one
            gets the same treatment as an invented chunk id.

    Returns:
        A new :class:`StoryGrounding`. Either list may come back empty, which is a
        legitimate result rather than a failure.

    A web fact is held to the same standard as a corpus one, deliberately. The id
    is resolved rather than trusted, and ``source`` is overwritten from the
    record, so "where did this come from?" has one answer of one strength rather
    than a strong answer for the corpus and a weaker one for the web. A web
    source additionally has to be **verified** -- present in the ledger is not
    enough, because ``VERIFY_WEB_CLAIMS=false`` records sources it never checked.
    """
    kept_facts: list[GroundedFact] = []
    for item in grounding.facts:
        if item.chunk_id.startswith(WEB_ID_PREFIX):
            kept = _keep_web_fact(item, ledger)
            if kept is not None:
                kept_facts.append(kept)
            continue

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

    dropped = len(grounding.facts) - len(kept_facts)
    if dropped:
        logger.info("provenance: kept %d fact(s), dropped %d", len(kept_facts), dropped)

    return StoryGrounding(facts=kept_facts)
