"""SparkStory MCP server entry point.

``create_server`` is kept separate from ``main`` deliberately. Tests build a
server in-process and call its tools directly without launching a transport, and
the in-memory transport used by the companion MCP client needs the same. If
these were fused, neither would be possible without spawning a subprocess.
"""

from fastmcp import FastMCP

from sparkstory.config import settings
from sparkstory.mcp.routers.tools import register_mcp_tools
from sparkstory.utils.logging_utils import configure_logging, get_logger

logger = get_logger(__name__)

#: Shown to MCP clients during the handshake. Clients surface this to their own
#: model, so it is written for an agent deciding whether to call these tools.
SERVER_INSTRUCTIONS = """\
SparkStory plans and writes personalised, illustrated children's storybooks.

Start by planning a story with `plan_story`, which returns a title, theme,
characters and ordered beats. Show the outline to the user and get their
approval before proceeding -- the outline determines everything downstream, and
changing it later means discarding work.

Every story is written for a specific child. Always establish their name, age
and pronouns before calling a tool. Never guess pronouns from a name."""


def create_server() -> FastMCP:
    """Build a configured FastMCP instance with all routers registered."""
    configure_logging()

    mcp = FastMCP(
        name=settings.server_name,
        version=settings.server_version,
        instructions=SERVER_INSTRUCTIONS,
    )

    register_mcp_tools(mcp)

    logger.info(
        "%s v%s ready (planner model: %s)",
        settings.server_name,
        settings.server_version,
        settings.planner_model,
    )
    return mcp


def main() -> None:
    """Console-script entry point. Serves over stdio."""
    # show_banner=False for the same reason logging goes to stderr: under stdio
    # transport, stdout carries JSON-RPC and nothing else may write to it.
    # FastMCP sends its banner to stderr, but stating the constraint is cheaper
    # than rediscovering it.
    create_server().run(show_banner=False)


if __name__ == "__main__":
    main()
