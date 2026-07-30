"""SparkStory: an agentic MCP server for personalised children's storybooks.

Intentionally free of logic. The console-script entry point lives in
``sparkstory.mcp.server:main``; importing this package must stay cheap and free of
side effects, because MCP clients import before they configure anything.
"""
