"""The parent-facing web UI.

Served by the MCP server itself over ``--transport http``, using
``@mcp.custom_route``. Under stdio transport these routes are registered and
never reachable, which is correct rather than a bug -- stdio has no HTTP.

Four modules, split by responsibility rather than by layer: ``jobs`` owns state
between planning and approval, ``artifacts`` owns the guarded path from a URL to
a file on disk, and ``pages`` owns HTML. ``handlers`` composes them.
"""
