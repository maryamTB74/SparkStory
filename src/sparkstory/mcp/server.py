"""SparkStory MCP server entry point.

``create_server`` is kept separate from ``main`` deliberately. Tests build a
server in-process and call its tools directly without launching a transport, and
the in-memory transport used by the companion MCP client needs the same. If
these were fused, neither would be possible without spawning a subprocess.
"""

from argparse import ArgumentParser

from fastmcp import FastMCP

from sparkstory.config import settings
from sparkstory.mcp.routers.prompts import register_mcp_prompts
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
changing it later means discarding work. The `create_storybook` prompt walks
through this whole flow, including the confirmation step.

Once the user is happy, `write_story` produces the finished book from the outline
`plan_story` returned -- pass it through unchanged. It is slower and costs several
model calls, so do not call it on a premise the user has not agreed to.

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
    register_mcp_prompts(mcp)

    logger.info(
        "%s v%s ready (planner model: %s)",
        settings.server_name,
        settings.server_version,
        settings.planner_model,
    )
    return mcp


def _build_parser() -> ArgumentParser:
    """Build the CLI parser.

    Separate from ``main`` for the same reason ``create_server`` is: so a test can
    assert on the defaults without starting a server. The default transport is the
    one thing here with a real cost if it changes silently.
    """
    parser = ArgumentParser(
        prog="sparkstory", description="Run the SparkStory MCP server."
    )
    # FastMCP accepts four transports -- Transport is
    # Literal["stdio", "http", "sse", "streamable-http"] -- and only two are
    # offered here. `sse` is deprecated, and `streamable-http` is a synonym for
    # `http` (http_app() itself defaults to "http"). Four names for three
    # behaviours is an invitation to pick the deprecated one.
    parser.add_argument(
        "--transport",
        "-t",
        choices=["stdio", "http"],
        default="stdio",
        help="Transport to serve on (default: stdio).",
    )
    return parser


def main() -> None:
    """Console-script entry point. Serves over stdio unless told otherwise."""
    args = _build_parser().parse_args()

    mcp = create_server()

    # show_banner=False on both branches for the same reason logging goes to
    # stderr: under stdio transport, stdout carries JSON-RPC and nothing else may
    # write to it. FastMCP sends its banner to stderr, but stating the constraint
    # is cheaper than rediscovering it.
    if args.transport == "http":
        # logger, never print(). The course's server.py prints its startup banner
        # to stdout and *then* falls through to stdio in its else branch --
        # non-obvious rule 2, which it survives only because stdio is its
        # secondary path and ours is the default.
        #
        # `run(transport="http", ...)` forwards host and port through
        # **transport_kwargs, so uvicorn is neither imported nor declared as a
        # dependency. The course builds `mcp.http_app()` and calls uvicorn itself
        # because it mounts Descope's OAuth discovery endpoints plus its own UI
        # and download routes on the same app. SparkStory has none of those. If
        # auth is ever added, this is the line that grows back.
        logger.info(
            "Serving over HTTP on %s:%s", settings.server_host, settings.server_port
        )
        mcp.run(
            transport="http",
            host=settings.server_host,
            port=settings.server_port,
            show_banner=False,
        )
    else:
        mcp.run(show_banner=False)


if __name__ == "__main__":
    main()
