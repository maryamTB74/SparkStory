"""MCP tool registration.

Registration only -- no business logic. Every function here is a thin wrapper
delegating to an implementation in ``sparkstory.mcp.tools``, which keeps the MCP
surface readable as an API listing and lets the implementations be tested
without an MCP server running.
"""

from fastmcp import FastMCP

from sparkstory.entities.stories import StoryBrief, StoryOutline
from sparkstory.mcp.tools.plan_story import plan_story_tool


def register_mcp_tools(mcp: FastMCP) -> None:
    """Attach every SparkStory tool to the given FastMCP instance."""

    @mcp.tool()
    async def plan_story(brief: StoryBrief) -> StoryOutline:
        """Plan a personalised children's story from a brief.

        Produces the structure of a story -- title, theme, characters and
        ordered beats -- without writing any prose. Review the outline before
        moving on to drafting, since every later stage builds on it.
        """
        return await plan_story_tool(brief)
