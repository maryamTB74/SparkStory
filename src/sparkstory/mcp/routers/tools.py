"""MCP tool registration.

Registration only -- no business logic. Every function here is a thin wrapper
delegating to an implementation in ``sparkstory.mcp.tools``, which keeps the MCP
surface readable as an API listing and lets the implementations be tested
without an MCP server running.
"""

from fastmcp import FastMCP

from sparkstory.entities.illustration import StoryArt
from sparkstory.entities.stories import Story, StoryBrief, StoryOutline
from sparkstory.mcp.tools.illustrate_story import illustrate_story_tool
from sparkstory.mcp.tools.plan_story import plan_story_tool
from sparkstory.mcp.tools.write_story import write_story_tool


def register_mcp_tools(mcp: FastMCP) -> None:
    """Attach every SparkStory tool to the given FastMCP instance."""

    @mcp.tool()
    async def plan_story(brief: StoryBrief) -> StoryOutline:
        """Plan a personalised children's story from a brief.

        Produces the structure of a story -- title, theme, characters and
        ordered beats -- without writing any prose. It revises its own plan
        until it passes review, so the outline it returns is the one the book
        will be built from. Show it to the user, get their approval, then pass
        it to `write_story` unchanged.
        """
        return await plan_story_tool(brief)

    @mcp.tool()
    async def write_story(brief: StoryBrief, outline: StoryOutline) -> Story:
        """Write a complete personalised children's story from an approved plan.

        Takes the outline `plan_story` returned and the user approved, and
        builds the book from exactly that plan -- same title, same characters,
        same beats. Pass the outline through unchanged; do not edit it or write
        one yourself.

        Runs several model calls and takes considerably longer than
        `plan_story`. If the user has not yet agreed to the outline, confirm it
        with them first.
        """
        return await write_story_tool(brief, outline)

    @mcp.tool()
    async def illustrate_story(
        brief: StoryBrief, story: Story, output_directory: str
    ) -> StoryArt:
        """Draw the pictures for a story `write_story` has already written.

        Decides one shared visual style for the whole book, draws a reference
        portrait of each character, then draws every page from those portraits so
        the same character looks the same throughout. Images are written into
        `output_directory` as files; the result records where each one went.

        Call this only after the story is written, and pass that story through
        unchanged. It is the slowest and most expensive tool here -- roughly one
        image per page plus one per character -- so confirm with the user before
        illustrating a long book.

        A picture that cannot be drawn is reported rather than fatal: that page
        simply has no illustration. Check `fully_conditioned` on the result to see
        whether every picture was drawn from the character portraits; when it is
        false, `detail` on each item says what happened.
        """
        return await illustrate_story_tool(brief, story, output_directory)
