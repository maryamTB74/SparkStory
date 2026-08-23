"""MCP tool registration.

Registration only -- no business logic. Every function here is a thin wrapper
delegating to an implementation in ``sparkstory.mcp.tools``, which keeps the MCP
surface readable as an API listing and lets the implementations be tested
without an MCP server running.
"""

from fastmcp import FastMCP

from sparkstory.config import settings
from sparkstory.entities.illustration import StoryArt
from sparkstory.entities.narration import StoryNarration
from sparkstory.entities.stories import Story, StoryBrief, StoryOutline
from sparkstory.mcp.tools.illustrate_story import illustrate_story_tool
from sparkstory.mcp.tools.narrate_story import narrate_story_tool
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
    async def write_story(
        brief: StoryBrief, outline: StoryOutline, output_directory: str
    ) -> Story:
        """Write a complete personalised children's story from an approved plan.

        Takes the outline `plan_story` returned and the user approved, and
        builds the book from exactly that plan -- same title, same characters,
        same beats. Pass the outline through unchanged; do not edit it or write
        one yourself.

        The finished book is written into `output_directory` as `story.json`,
        and the result records where it went. Tell the user that path: it is
        how they find their book again once this conversation is over. Use the
        same directory for `illustrate_story` and `narrate_story` so a book and
        its media stay together.

        Runs several model calls and takes considerably longer than
        `plan_story`. If the user has not yet agreed to the outline, confirm it
        with them first.
        """
        return await write_story_tool(brief, outline, output_directory)

    # The two media tools are registered conditionally, so a server can be
    # deployed unable to spend money on images rather than merely asked not to.
    # A client cannot call what `list_tools` never showed it.
    if settings.illustration_enabled:

        @mcp.tool()
        async def illustrate_story(
            brief: StoryBrief, story: Story, output_directory: str
        ) -> StoryArt:
            """Draw the pictures for a story `write_story` has already written.

            Decides one shared visual style for the whole book, draws a reference
            portrait of each character, then draws every page from those portraits
            so the same character looks the same throughout. Images are written
            into `output_directory` as files; the result records where each went.

            Call this only after the story is written, and pass that story through
            unchanged. It is the slowest and most expensive tool here -- roughly
            one image per page plus one per character -- so confirm with the user
            before illustrating a long book.

            A picture that cannot be drawn is reported rather than fatal: that
            page simply has no illustration. Check `fully_conditioned` on the
            result to see whether every picture was drawn from the character
            portraits; when it is false, `detail` on each item says what happened.

            This also re-renders the book's PDF with the pictures in it, since
            the one `write_story` made was written before they existed. The
            result's `pdf_saved_to` is the illustrated PDF; it replaces the file
            at the path `write_story` reported.
            """
            return await illustrate_story_tool(brief, story, output_directory)

    # A separate switch rather than one shared "media" flag: narration is two
    # orders of magnitude cheaper than illustration, so an installation that
    # cannot afford pictures may still want a book read aloud.
    if settings.narration_enabled:

        @mcp.tool()
        async def narrate_story(
            brief: StoryBrief, story: Story, output_directory: str
        ) -> StoryNarration:
            """Read a story `write_story` has already written aloud.

            Speaks every page in the voice the brief asks for, at a pace chosen
            for the child's reading level, and writes one audio file per page plus
            a single `story.mp3` of the whole book into `output_directory`. The
            text is spoken exactly as written -- nothing is rephrased.

            Call this only after the story is written, and pass that story through
            unchanged. It does not need the pictures, so it can run before, after
            or instead of `illustrate_story`.

            A page that cannot be narrated is reported rather than fatal: that
            page simply has no audio. Check `is_complete` on the result to see
            whether the whole book was narrated; when it is false,
            `pages_narrated` says how many pages have audio and each item's
            `detail` says what happened. When no page could be narrated at all,
            there is no `story.mp3` rather than a silent one.
            """
            return await narrate_story_tool(brief, story, output_directory)
