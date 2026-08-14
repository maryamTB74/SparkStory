"""What one turn of a client conversation produced.

The one idea here is that **asking for a tool call and running it are different
events**, and both have to be reportable. ``ClientSession`` serves two front
ends that disagree about which one they care about: the REPL runs tools and
prints results, while ``scripts/try_prompt.py`` inspects what a model *requested*
and deliberately never calls ``write_story`` -- its arguments answer the
question, and running it would buy a whole book to learn nothing more.

A single list of "tool calls" would force one of those to reconstruct the other's
view, and would leave inspect mode with no way to say *it wanted this and we did
not do it*.

**Frozen dataclasses rather than Pydantic models.** The three reasons this project
reaches for Pydantic are crossing a process boundary, being bound as a model's
output schema, and validating untrusted input. None applies: these are records
built and read inside one client process. Nothing here is prompt text, so no
docstring in this module reaches a model (non-obvious rule 1).
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolCall:
    """A tool call a model asked for. Nothing here says it ran."""

    id: str
    name: str
    # Untyped on purpose. The shape is whatever the server's tool schema declares,
    # and the client fetches those at runtime rather than importing them -- which
    # is the whole point of talking over MCP instead of calling the tools
    # directly. Typing this would mean duplicating schemas we deliberately do not
    # import.
    args: dict[str, Any]


@dataclass(frozen=True)
class ExecutedCall(ToolCall):
    """A tool call that was actually sent to the server, and what came back.

    Extends ``ToolCall`` because an executed call *is* a requested call plus an
    outcome. The rejected alternative was one flat record with an ``executed``
    flag, which makes "requested but not run" and "ran and returned nothing" the
    same shape -- exactly the confusion inspect mode introduces, so the types
    must not blur it as well.
    """

    # Both optional because the record is built before the call is made. Exactly
    # one is populated afterwards: a result on success, an error on failure.
    result: Any | None = None
    # A failed call stays in ``TurnResult.executed``. It ran, it cost time, and
    # the model has to be shown the error to recover from it. Dropping failures
    # would make a failed turn indistinguishable from a skipped one.
    error: str | None = None


@dataclass(frozen=True)
class TurnResult:
    """One model turn: what it said, what it asked for, what ran.

    ``tool_calls`` empty is the loop's stop condition, so it must stay a valid
    value -- a reflexive "at least one" constraint would make termination
    impossible, which is non-obvious rule 14 in a new place.
    """

    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    executed: list[ExecutedCall] = field(default_factory=list)
