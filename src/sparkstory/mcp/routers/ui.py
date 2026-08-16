"""Web UI route registration.

Registration only -- no logic, matching ``routers/tools.py``. Every route is a
delegation to ``sparkstory.mcp.ui.handlers``, which keeps the surface readable as
a route table and lets the handlers be tested without a server running.

**These routes are served only under ``--transport http``.** Under stdio they are
registered and never reachable, because stdio has no HTTP. A 404 there is the
transport, not a bug.

**The server binds to ``127.0.0.1`` by default and there is no authentication.**
Changing ``SERVER_HOST`` exposes a child's name, their illustrations and the audio
of their book to anyone who can route to the port. Job ids are unguessable, which
is obscurity rather than authorization. See spec section 5.5 before deploying.
"""

from fastmcp import FastMCP

from sparkstory.mcp.ui import handlers


def register_ui_routes(mcp: FastMCP) -> None:
    """Attach the parent-facing web UI to the given FastMCP instance."""
    mcp.custom_route("/", methods=["GET"])(handlers.get_form)
    mcp.custom_route("/plan", methods=["POST"])(handlers.post_plan)
    mcp.custom_route("/job/{job_id}", methods=["GET"])(handlers.get_job)
    mcp.custom_route("/job/{job_id}/status", methods=["GET"])(handlers.get_status)
    mcp.custom_route("/job/{job_id}/approve", methods=["POST"])(handlers.post_approve)
    mcp.custom_route("/job/{job_id}/revise", methods=["POST"])(handlers.post_revise)
    mcp.custom_route("/job/{job_id}/book", methods=["GET"])(handlers.get_book)
    mcp.custom_route("/job/{job_id}/file/{name}", methods=["GET"])(handlers.get_file)
    # Read-only. A job writes; the library only reads, so nothing here can approve,
    # revise or spend money.
    mcp.custom_route("/library", methods=["GET"])(handlers.get_library)
    mcp.custom_route("/library/{run_id}", methods=["GET"])(handlers.get_library_book)
    mcp.custom_route("/library/{run_id}/file/{name}", methods=["GET"])(
        handlers.get_library_file
    )
