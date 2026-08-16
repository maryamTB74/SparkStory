"""Route registration -- that the routes exist and are wired."""

from fastmcp import FastMCP

from sparkstory.mcp.routers.ui import register_ui_routes


def _paths(mcp: FastMCP) -> set[str]:
    return {
        route.path for route in mcp.http_app().routes if getattr(route, "path", None)
    }


def test_every_ui_route_is_registered() -> None:
    mcp = FastMCP(name="test")
    register_ui_routes(mcp)

    paths = _paths(mcp)

    assert "/" in paths
    assert "/plan" in paths
    assert "/job/{job_id}" in paths
    assert "/job/{job_id}/status" in paths
    assert "/job/{job_id}/approve" in paths
    assert "/job/{job_id}/revise" in paths
    assert "/job/{job_id}/book" in paths
    assert "/job/{job_id}/file/{name}" in paths


def test_registration_writes_nothing_to_stdout(capsys) -> None:
    mcp = FastMCP(name="test")
    register_ui_routes(mcp)

    assert capsys.readouterr().out == ""


def test_create_server_registers_the_ui() -> None:
    from sparkstory.mcp.server import create_server

    paths = _paths(create_server())

    assert "/plan" in paths
