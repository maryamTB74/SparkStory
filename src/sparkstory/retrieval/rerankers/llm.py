"""Reranking by asking a model which candidates actually answer the query.

The model is shown ids and text and returns **ids only**. That asymmetry is the
point: it can reorder and it cannot rewrite, so a chunk cannot be quietly reworded
on its way to the agent. Same reasoning that makes the store overwrite a fact's
``source`` from the index rather than trusting what a model wrote there -- convert
a claim you would have to trust into one you can check.

Two ways a reranker can go wrong that fusion cannot, and both are handled here
rather than assumed away:

- **An id that was never a candidate.** The model is writing ids, so it can write
  one that does not exist, and a fact built from an invented id would cite a chunk
  nobody can look up. Dropped, with a warning.
- **A candidate the model omits.** Demoted behind what it chose, never deleted.
  Deleting would let one model call shrink the result set below ``top_k``, turning
  a ranking into a filter the caller never asked for.
"""

from typing import Any

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from sparkstory.retrieval.rerank import Reranker
from sparkstory.retrieval.types import SearchHit
from sparkstory.utils.logging_utils import get_logger

logger = get_logger(__name__)


class RankedIds(BaseModel):
    """Which of the candidates help answer the question, best first."""

    chunk_ids: list[str] = Field(
        description=(
            "The ids you were shown, ordered with the most useful first. Use only "
            "ids from the candidate list. Leave out any candidate that does not "
            "help answer the question."
        )
    )


_INSTRUCTIONS = """\
Below are candidate facts retrieved for a question. Decide which of them actually \
help answer it, and put the most useful first.

Question: {query}

Candidates:
{candidates}

Return the ids that help, best first. Leave out any that do not help. Use only ids \
from the list above."""


def build_llm_reranker(model: Any) -> Reranker:
    """Build a reranker over an already-constructed model.

    The model is injected rather than built here, matching how the nodes take a
    ``Runnable``: it keeps the provider decision at the composition site and lets a
    test pass a fake without touching a registry.
    """

    async def rerank(query: str, hits: list[SearchHit], top_k: int) -> list[SearchHit]:
        if not hits:
            # An empty candidate list is a legitimate result everywhere else in
            # retrieval, so it must not become an error -- or a model call.
            return []

        remaining = {hit.chunk.chunk_id: hit for hit in hits}
        candidates = "\n".join(
            f"- {hit.chunk.chunk_id}: {hit.chunk.text}" for hit in hits
        )
        answer = await model.ainvoke(
            [
                HumanMessage(
                    content=_INSTRUCTIONS.format(query=query, candidates=candidates)
                )
            ]
        )

        chosen: list[SearchHit] = []
        for chunk_id in answer.chunk_ids:
            hit = remaining.pop(chunk_id, None)
            if hit is None:
                logger.warning(
                    "reranker named %r, which was not among the candidates", chunk_id
                )
                continue
            chosen.append(hit)

        # Whatever it left out keeps fusion's order, behind everything it chose.
        chosen.extend(hit for hit in hits if hit.chunk.chunk_id in remaining)
        return chosen[:top_k]

    return rerank
