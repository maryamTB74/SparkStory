"""MCP surface: what a client actually sees.

Uses FastMCP's in-memory transport, so the server is exercised through the real
MCP protocol without spawning a subprocess. This is why ``create_server`` is
separate from ``main``.
"""

import pytest
from fastmcp import Client

from sparkstory.mcp.server import _build_parser, create_server, main


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

    async def test_write_story_takes_brief_and_outline(self) -> None:
        """A client must pass the approved plan, not just the brief.

        This is the schema half of the whole design: if `outline` were optional
        the server would silently fall back to building a book the parent never
        saw, which is precisely the defect this replaced.
        """
        async with Client(create_server()) as client:
            tool = next(t for t in await client.list_tools() if t.name == "write_story")

        assert {"brief", "outline"} <= set(tool.inputSchema["properties"])
        assert set(tool.inputSchema.get("required", [])) == {"brief", "outline"}


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

    async def test_it_tells_the_client_to_pass_the_outline_on(self, text: str) -> None:
        """The hardest thing this prompt asks for, and the flow breaks without it.

        `write_story` requires the outline now, so a client that does not thread
        the structured result of one tool into another tool's arguments gets an
        error instead of a book.

        Asserted against the line that *calls* the tool, not merely against a
        line mentioning both words -- the old "an outline you edited by hand
        never reaches the book" sentence satisfied that weaker check while
        telling the client the opposite of what is now true.
        """
        call_lines = [ln for ln in text.splitlines() if "call `write_story`" in ln]
        assert call_lines, "the prompt never tells the client to call write_story"
        assert any("outline" in ln for ln in call_lines), (
            "the write_story call instruction never mentions the outline"
        )

    async def test_it_does_not_call_planning_one_model_call(self, text: str) -> None:
        """plan_story runs the outline critic now, so the old claim is false.

        A client told planning is cheap will re-plan freely; it is 2-4 calls.
        """
        assert "one model call" not in text

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


class TestTransportSelection:
    """Which transport ``main`` serves on, and what reaches ``FastMCP.run``.

    None of this proves HTTP transport actually serves MCP -- that needs a real
    client against a real port and is deliberately left to a live run. What these
    do protect is the wiring: the default, and whether host and port are forwarded.
    """

    def test_default_transport_is_stdio(self) -> None:
        """The regression with the widest blast radius.

        ``uv run sparkstory`` with no arguments is how ``.mcp.json.sample``, the
        assignment's client config and ``make run`` all launch this. The course
        defaults to http, so "align with the course" is a plausible future edit --
        and its symptom would be every existing client hanging rather than an
        error. This test makes that edit fail loudly instead.
        """
        assert _build_parser().parse_args([]).transport == "stdio"

    @pytest.mark.parametrize("argv", [["--transport", "http"], ["-t", "http"]])
    def test_http_can_be_requested(self, argv: list[str]) -> None:
        assert _build_parser().parse_args(argv).transport == "http"

    @pytest.mark.parametrize("rejected", ["sse", "streamable-http", "grpc"])
    def test_only_two_transports_are_offered(self, rejected: str) -> None:
        """``sse`` is deprecated and ``streamable-http`` is a synonym for ``http``.

        FastMCP accepts all four. Offering four names for three behaviours invites
        someone to pick the deprecated one, so the restriction is deliberate --
        and this test is what makes lifting it a conscious act.
        """
        with pytest.raises(SystemExit):
            _build_parser().parse_args(["--transport", rejected])

    def test_stdio_path_passes_no_host_or_port(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """stdio has no address, and passing one would be a silent no-op."""
        captured: dict[str, object] = {}
        monkeypatch.setattr(
            "sparkstory.mcp.server.FastMCP.run",
            lambda self, **kwargs: captured.update(kwargs),
        )
        monkeypatch.setattr("sys.argv", ["sparkstory"])

        main()

        assert "host" not in captured
        assert "port" not in captured
        assert captured["show_banner"] is False

    def test_http_path_forwards_host_and_port_from_settings(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The test that actually exercises the mechanism.

        ``run(transport="http", host=..., port=...)`` forwards through
        ``**transport_kwargs``, which is why uvicorn is neither imported nor
        declared as a dependency. If that forwarding breaks, the server binds
        FastMCP's own defaults instead of the configured ones -- and the
        parser-level tests above would still pass, because the flag parsed fine.
        """
        captured: dict[str, object] = {}
        monkeypatch.setattr(
            "sparkstory.mcp.server.FastMCP.run",
            lambda self, **kwargs: captured.update(kwargs),
        )
        # String form, matching tests/test_observability_tracing.py: it patches the
        # attribute on the settings module itself, so it holds regardless of how a
        # module reached `settings`.
        monkeypatch.setattr("sparkstory.config.settings.server_host", "0.0.0.0")
        monkeypatch.setattr("sparkstory.config.settings.server_port", 9123)
        monkeypatch.setattr("sys.argv", ["sparkstory", "--transport", "http"])

        main()

        assert captured["transport"] == "http"
        assert captured["host"] == "0.0.0.0"
        assert captured["port"] == 9123
        assert captured["show_banner"] is False


class TestStdoutStaysSilent:
    """Non-obvious rule 2, as an executable check rather than a comment.

    Under stdio transport stdout carries JSON-RPC, so a single stray ``print``
    corrupts the protocol into a JSON parse error that looks nothing like its
    cause. This is not hypothetical: the course's own ``research_agent_part_3``
    prints its startup banner to stdout and *then* falls through to
    ``mcp.run(transport="stdio")`` in its else branch. It gets away with it
    because http is its default; ours is stdio, so the same code would break
    ``make run``.
    """

    def test_starting_on_stdio_writes_nothing_to_stdout(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(
            "sparkstory.mcp.server.FastMCP.run", lambda self, **kwargs: None
        )
        monkeypatch.setattr("sys.argv", ["sparkstory"])

        main()

        assert capsys.readouterr().out == "", (
            "something wrote to stdout before the stdio transport started; "
            "that corrupts JSON-RPC"
        )
