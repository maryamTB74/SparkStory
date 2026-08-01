"""MCP surface: what a client actually sees.

Uses FastMCP's in-memory transport, so the server is exercised through the real
MCP protocol without spawning a subprocess. This is why ``create_server`` is
separate from ``main``.
"""

import pytest
from fastmcp import Client

from sparkstory.mcp.server import create_server


class TestToolRegistration:
    async def test_both_tools_are_exposed(self) -> None:
        async with Client(create_server()) as client:
            names = [t.name for t in await client.list_tools()]
        assert {"plan_story", "write_story"} <= set(names)

    @pytest.mark.parametrize("tool_name", ["plan_story", "write_story"])
    async def test_tool_has_description_and_output_schema(self, tool_name: str) -> None:
        async with Client(create_server()) as client:
            tool = next(t for t in await client.list_tools() if t.name == tool_name)

        assert tool.description, (
            "clients rely on this to decide whether to call the tool"
        )
        assert tool.outputSchema, (
            "structured output means clients get a validated shape"
        )

    async def test_write_story_tells_a_client_it_is_the_expensive_one(self) -> None:
        """An agent chooses between the two tools by reading their descriptions.

        Without this, a client agent will reasonably call write_story on a premise
        the parent has not seen, which spends several model calls on a story that
        may be discarded -- and removes the review step the HITL prompt depends on.
        """
        async with Client(create_server()) as client:
            tool = next(t for t in await client.list_tools() if t.name == "write_story")

        assert tool.description is not None
        assert "plan_story" in tool.description


class TestClientVisibleSchema:
    """A client can only call what the schema fully describes."""

    @pytest.fixture
    async def schema(self) -> dict:
        async with Client(create_server()) as client:
            tool = next(t for t in await client.list_tools() if t.name == "plan_story")
        return tool.inputSchema

    @pytest.mark.parametrize("tool_name", ["plan_story", "write_story"])
    async def test_nested_definitions_are_inlined(self, tool_name: str) -> None:
        """A client cannot resolve a $ref it was never given.

        FastMCP inlines Pydantic's $defs; this asserts that stays true, because a
        dangling reference would leave clients unable to construct a brief.
        """
        async with Client(create_server()) as client:
            tool = next(t for t in await client.list_tools() if t.name == tool_name)
        assert "$defs" not in tool.inputSchema

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


class TestPromptRegistration:
    """MCP prompts: the guided workflow a user invokes by name.

    Distinct from ``tests/test_prompts.py``, which covers the text we send to our
    own models. This is text we send to a *client's* model.
    """

    async def test_create_storybook_is_exposed(self) -> None:
        async with Client(create_server()) as client:
            names = [p.name for p in await client.list_prompts()]
        assert "create_storybook" in names

    async def test_prompt_takes_no_arguments(self) -> None:
        """Zero-argument by design, and the design depends on it.

        An argument renders as a form field in the client. A `child_name` field
        in particular would hand the model a name to infer pronouns from, which
        the instruction text explicitly forbids.
        """
        async with Client(create_server()) as client:
            prompt = next(
                p for p in await client.list_prompts() if p.name == "create_storybook"
            )
        assert not prompt.arguments

    async def test_prompt_has_a_description(self) -> None:
        """The menu text a client shows. Empty makes the prompt unusable."""
        async with Client(create_server()) as client:
            prompt = next(
                p for p in await client.list_prompts() if p.name == "create_storybook"
            )
        assert prompt.description


class TestPromptContent:
    """The instruction text is behaviour, not presentation.

    An omitted step changes what a client actually does, so the load-bearing
    lines are asserted. Full sentences are deliberately not asserted -- that
    makes the text unmaintainable and catches nothing a diff would not show.
    """

    @pytest.fixture
    async def text(self) -> str:
        async with Client(create_server()) as client:
            result = await client.get_prompt("create_storybook")
        return result.messages[0].content.text

    async def test_it_names_every_registered_tool(self, text: str) -> None:
        """Rename a tool and this prompt silently starts lying to client LLMs.

        Nothing else would fail: the string still renders, the prompt still
        lists, and a client is told to call a tool that no longer exists. The
        assertion runs registered-to-text rather than text-to-registered so that
        a rename is what breaks it.

        If a later tool is deliberately outside this workflow, add it to an
        explicit exclusion set here rather than deleting the test.
        """
        async with Client(create_server()) as client:
            registered = {t.name for t in await client.list_tools()}

        missing = {name for name in registered if name not in text}
        assert not missing, (
            f"the create_storybook prompt never mentions {sorted(missing)}, so a "
            "client following it cannot reach those tools"
        )

    async def test_planning_comes_before_writing(self, text: str) -> None:
        """The order is the workflow. Reversed, the confirmation is pointless."""
        assert text.index("plan_story") < text.index("write_story")

    async def test_it_forbids_calling_both_tools_in_one_turn(self, text: str) -> None:
        """Step 3's load-bearing instruction.

        Nothing enforces it server-side, so the wording is the only thing
        standing between a parent and a book they never approved.
        """
        same_turn_lines = [ln for ln in text.splitlines() if "same turn" in ln]
        assert same_turn_lines, "the prohibition on one-turn execution is missing"
        assert any("write_story" in ln for ln in same_turn_lines)

    async def test_it_requires_asking_for_pronouns(self, text: str) -> None:
        """The one instruction here with a real-world cost if ignored."""
        assert "pronouns" in text
        assert "Never infer" in text
