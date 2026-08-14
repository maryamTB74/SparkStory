"""MCP resource registration.

Registration only -- no logic, mirroring ``routers/tools.py`` and
``routers/prompts.py``. Every function here delegates to
``sparkstory.mcp.resources``.

A resource differs from a tool in what it costs and what it changes: these read
files and return text, so a client may call them freely. Both are read-only by
design, and neither takes a parameter -- a parameterised resource over run
directories would be a path-traversal surface with an LLM on the other end.
"""

from fastmcp import FastMCP

from sparkstory.mcp.resources.library import read_corpus, read_library


def register_mcp_resources(mcp: FastMCP) -> None:
    """Attach every SparkStory resource to the given FastMCP instance."""

    @mcp.resource("sparkstory://library")
    def library() -> str:
        """The storybooks this server has finished: id, title, page count, and
        whether a PDF or narration was produced.
        """
        return read_library()

    @mcp.resource("sparkstory://corpus")
    def corpus() -> str:
        """What the story-facts corpus contains: how many files and chunks are
        available to ground a story, and which embedding model indexes them.
        """
        return read_corpus()
