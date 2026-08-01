"""MCP prompt registration.

Registration only -- no text, mirroring ``routers/tools.py``. Every function here
returns a constant from ``sparkstory.mcp.prompts``, which keeps this module
readable as a listing of what a client can invoke.

Prompts differ from tools in who reads them and when. A tool description helps a
client's model decide *whether* to call something; a prompt is a workflow a user
invokes deliberately, and its text drives several calls in sequence.
"""

from fastmcp import FastMCP

from sparkstory.mcp.prompts.create_storybook import CREATE_STORYBOOK_INSTRUCTIONS


def register_mcp_prompts(mcp: FastMCP) -> None:
    """Attach every SparkStory prompt to the given FastMCP instance."""

    @mcp.prompt()
    def create_storybook() -> str:
        """Create a personalised children's storybook, confirming the story's
        plan with the parent before writing it.
        """
        return CREATE_STORYBOOK_INSTRUCTIONS
