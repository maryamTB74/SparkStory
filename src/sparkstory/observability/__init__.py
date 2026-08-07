"""Opik wiring, and the one rule that governs it.

**No module in this package imports ``opik`` at module scope.** Every import sits
inside a function body behind the enabled check. Two concrete reasons: an
unguarded top-level import makes a heavy dependency tree mandatory for
``uv run sparkstory``, and it loads litellm and a second openai client into the
process of every user who never turns tracing on. There is a test asserting the
disabled path imports nothing.

Everything here fails open. A trace is worth less than the book it describes, so
a missing key, a bad workspace or an unreachable backend costs a warning and
nothing else.
"""

from sparkstory.observability.opik_utils import configure
from sparkstory.observability.tracing import build_handler

__all__ = ["build_handler", "configure"]
