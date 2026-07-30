"""MCP surface: what a client actually sees.

Uses FastMCP's in-memory transport, so the server is exercised through the real
MCP protocol without spawning a subprocess. This is why ``create_server`` is
separate from ``main``.
"""

import pytest
from fastmcp import Client

from sparkstory.mcp.server import create_server


class TestToolRegistration:
    async def test_plan_story_is_exposed(self) -> None:
        async with Client(create_server()) as client:
            names = [t.name for t in await client.list_tools()]
        assert "plan_story" in names

    async def test_tool_has_description_and_output_schema(self) -> None:
        async with Client(create_server()) as client:
            tool = next(t for t in await client.list_tools() if t.name == "plan_story")

        assert tool.description, (
            "clients rely on this to decide whether to call the tool"
        )
        assert tool.outputSchema, (
            "structured output means clients get a validated shape"
        )


class TestClientVisibleSchema:
    """A client can only call what the schema fully describes."""

    @pytest.fixture
    async def schema(self) -> dict:
        async with Client(create_server()) as client:
            tool = next(t for t in await client.list_tools() if t.name == "plan_story")
        return tool.inputSchema

    async def test_nested_definitions_are_inlined(self, schema: dict) -> None:
        """A client cannot resolve a $ref it was never given.

        FastMCP inlines Pydantic's $defs; this asserts that stays true, because a
        dangling reference would leave clients unable to construct a brief.
        """
        assert "$defs" not in schema

    async def test_child_fields_are_visible(self, schema: dict) -> None:
        child = schema["properties"]["brief"]["properties"]["child"]["properties"]
        assert {"name", "age", "pronouns", "reading_level", "interests"} <= set(child)

    async def test_constraints_survive_into_the_schema(self, schema: dict) -> None:
        """Bounds are guidance to the calling model, not only server-side checks."""
        age = schema["properties"]["brief"]["properties"]["child"]["properties"]["age"]
        assert age["minimum"] == 2
        assert age["maximum"] == 12

    async def test_pronoun_options_are_enumerated(self, schema: dict) -> None:
        pronouns = schema["properties"]["brief"]["properties"]["child"]["properties"][
            "pronouns"
        ]
        assert set(pronouns["enum"]) == {"she/her", "he/him", "they/them"}
        assert pronouns["default"] == "they/them"


class TestServerMetadata:
    async def test_instructions_are_advertised(self) -> None:
        """Clients surface these to their own model, so they must be present."""
        server = create_server()
        assert server.instructions
        assert "plan_story" in server.instructions
