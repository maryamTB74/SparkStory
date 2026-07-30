"""MCP tool registration.

Registration only -- no business logic. Every function here is a thin wrapper
delegating to an implementation in ``sparkstory.mcp.tools``, which keeps the MCP
surface readable as an API listing and lets the implementations be tested
without an MCP server running.
"""

from fastmcp import FastMCP

from sparkstory.entities.stories import Story, StoryBrief, StoryOutline
from sparkstory.mcp.tools.plan_story import plan_story_tool
from sparkstory.mcp.tools.write_story import write_story_tool


def register_mcp_tools(mcp: FastMCP) -> None:
    """Attach every SparkStory tool to the given FastMCP instance."""

    @mcp.tool()
    async def plan_story(brief: StoryBrief) -> StoryOutline:
        """Plan a personalised children's story from a brief.

        Produces the structure of a story -- title, theme, characters and
        ordered beats -- without writing any prose. This is the cheap preview:
        show the outline to the user and get their approval before calling
        `write_story`, which is slower and builds on whatever this returns.
        """
        return await plan_story_tool(brief)

    @mcp.tool()
    async def write_story(brief: StoryBrief) -> Story:
        """Write a complete personalised children's story from a brief.

        Runs the whole pipeline: plans the structure, lays it out across pages,
        and writes the words for every page at the child's reading level.
        Returns the finished text together with the plan it came from.

        This takes considerably longer than `plan_story` and makes several model
        calls. If the user has not yet agreed to the story's premise, use
        `plan_story` first and confirm the outline with them.
        """
        return await write_story_tool(brief)
