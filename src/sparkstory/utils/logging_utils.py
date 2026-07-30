"""Logging configuration.

Two decisions here are worth more than they look.

**1. Every log record goes to stderr.**

The MCP stdio transport speaks JSON-RPC over *stdout*. A single stray
``print()`` or a log handler pointed at stdout corrupts that byte stream, and
the client disconnects with a JSON parse error that looks nothing like a
logging problem. ``logging.basicConfig`` already defaults to stderr, but the
stream is passed explicitly so that the requirement is visible to whoever
edits this next.

**2. Third-party loggers get their own level.**

In an LLM application most log volume is ``httpx`` request lines and Google
client chatter. Controlling those independently of our own modules.

These logs are human-readable diagnostics. Structured, machine-queryable
telemetry -- tokens, cost, latency, per-agent spans -- is a separate concern and
belongs in a tracing layer, not smuggled into log lines.
"""

import logging
import sys

from sparkstory.config import settings

#: Third-party loggers muted to ``log_level_dependencies``. These are the
#: libraries that emit per-request lines and would otherwise dominate output.
_DEPENDENCY_LOGGERS: tuple[str, ...] = (
    "httpx",
    "httpcore",
    "google",
    "google_genai",
    "langchain",
    "langchain_google_genai",
    "fastmcp",
    "mcp",
    # FastMCP's background task subsystem announces every registered task at
    # INFO on server start, which is noise for us and confusing in tool output.
    "docket",
    "urllib3",
    "asyncio",
)


def configure_logging() -> None:
    """Configure root and dependency loggers. Safe to call more than once."""
    logging.basicConfig(
        level=settings.log_level,
        stream=sys.stderr,  # never stdout -- see module docstring
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        # Plain basicConfig silently no-ops when the root logger already has a
        # handler, which makes it useless in tests and on re-entry. force=True
        # replaces existing handlers so the call is genuinely idempotent.
        force=True,
    )

    logging.getLogger().setLevel(settings.log_level)

    for name in _DEPENDENCY_LOGGERS:
        logging.getLogger(name).setLevel(settings.log_level_dependencies)


def get_logger(name: str) -> logging.Logger:
    """Return a module logger.

    Call sites use ``get_logger(__name__)`` so the logger name matches the
    module path. That keeps the dotted-name hierarchy intact, which is what
    lets a single ``logging.getLogger("sparkstory").setLevel(...)`` retune the
    whole application at once.
    """
    return logging.getLogger(name)
