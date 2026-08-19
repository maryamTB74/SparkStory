"""Reordering the candidates fusion produced.

RRF fuses a vector ranking and a keyword ranking into one list, and that list has
until now gone straight to the agent. The decisions table describes the *agent* as
reranking, an idea borrowed from a course notebook whose prompt tells its agent not
to take rank 1 blindly -- but every live run here has the agent answering in a
single turn and using what it was handed. So nothing reranks, and the row describes
an intention rather than a behaviour.

**A seam rather than a function**, because more than one implementation is
plausible and they differ in ways that matter: a model call is flexible and costs
money and may not answer the same way twice, while a local cross-encoder is free and
deterministic and costs a large dependency. Which is better on this corpus is a
question for the labelled set, not for an argument -- so the shape here lets the
eval harness hand each one identical candidates and compare.

Only the LLM implementation exists today. The cross-encoder was specified and
deliberately deferred: it would add `torch` to a project with nine runtime
dependencies and a suite that has run offline since the first session, and that is
not worth paying before reranking has been shown to help at all.
"""

from collections.abc import Awaitable, Callable

from sparkstory.retrieval.types import SearchHit

#: Query, candidates, how many to keep. Returns at most ``top_k``.
#:
#: Async because the interesting implementation is a model call. A synchronous
#: signature would have forced the LLM reranker to block an event loop that the
#: retrieval tools already run inside.
#:
#: **Truncation belongs to the contract rather than to the caller.** Every
#: implementation is handed the same candidate list and must return the same
#: number, or a comparison between two rerankers stops measuring "which ranks
#: better" and starts measuring "which returns more".
Reranker = Callable[[str, list[SearchHit], int], Awaitable[list[SearchHit]]]


async def identity_reranker(
    query: str, hits: list[SearchHit], top_k: int
) -> list[SearchHit]:
    """Keep fusion's order. The control every other reranker is measured against.

    ``query`` is deliberately unused: this is what "no reranking" looks like when
    it runs through the same code path as the real thing, which is a better
    baseline than skipping the stage -- skipping would also skip the truncation
    and any bug living in the plumbing.
    """
    return hits[:top_k]
