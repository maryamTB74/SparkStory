"""The client half of the MCP surface, exercised against the real server.

These use the in-memory transport rather than mocks, deliberately. The whole
value of an in-memory client is that it crosses the actual MCP boundary --
schemas are fetched, not imported -- and a mocked client would test nothing but
the mock.

What is asserted here is the *boundary*, not the contents. That the three real
tools come back proves the handshake; asserting their descriptions would
duplicate ``tests/test_server.py`` and break on every prompt edit.
"""

import json
from typing import Any

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from sparkstory.mcp.client.session import ClientSession


class FakeClientModel:
    """A model that plays the client, returning queued turns.

    ``models/fake_model.py`` is the wrong fake here and the reason is worth
    stating: it implements ``with_structured_output``, because every *node* binds
    an output schema. A client binds **tools** and answers with ``AIMessage``
    objects carrying ``tool_calls``. A fake that absorbed both would stop failing
    loudly when the client starts using something new, which is the property that
    fake's own docstring argues for.
    """

    def __init__(self, *turns: AIMessage) -> None:
        self._turns = list(turns)
        #: Message lists received, one entry per ``ainvoke`` call.
        self.calls: list[list[Any]] = []

    def bind_tools(self, _tools: list[Any]) -> FakeClientModel:
        return self

    async def ainvoke(self, messages: list[Any], **_: Any) -> AIMessage:
        self.calls.append(list(messages))
        if len(self._turns) > 1:
            return self._turns.pop(0)
        return self._turns[0]


def _brief_args() -> dict[str, Any]:
    """The smallest valid `plan_story` argument set."""
    return {
        "brief": {
            "child": {"name": "Maryam", "age": 5},
            "premise": "a fox who wants to visit the moon",
        }
    }


def _turn_calling(name: str, args: dict[str, Any], call_id: str = "c1") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"id": call_id, "name": name, "args": args}],
    )


def _final_turn(text: str = "Here is the plan.") -> AIMessage:
    return AIMessage(content=text)


class TestCapabilities:
    """What a client is handed at connect time."""

    async def test_it_fetches_the_servers_tools(self) -> None:
        async with ClientSession() as session:
            assert {t.name for t in session.tools} >= {
                "plan_story",
                "write_story",
                "illustrate_story",
            }

    async def test_it_fetches_the_create_storybook_prompt(self) -> None:
        async with ClientSession() as session:
            assert "create_storybook" in {p.name for p in session.prompts}

    async def test_it_reads_prompt_text_from_the_server(self) -> None:
        # Fetched rather than imported, which is what makes a prompt-obedience
        # harness measure what a client is really handed. `try_prompt.py` fetches
        # it the same way, which is the reason its results mean anything.
        async with ClientSession() as session:
            text = await session.get_prompt("create_storybook")

        assert "plan_story" in text
        assert "write_story" in text

    async def test_an_unknown_prompt_raises(self) -> None:
        async with ClientSession() as session:
            with pytest.raises(KeyError):
                await session.get_prompt("no-such-prompt")


class TestConnectionGuard:
    """Capabilities are unavailable until the session is open."""

    def test_tools_before_connecting_raises(self) -> None:
        # Raising rather than returning an empty list. An empty list reads as
        # "the server exposes no tools", which is a wrong answer wearing the
        # costume of a valid one -- and it would surface as a client that
        # silently never calls anything.
        session = ClientSession()
        with pytest.raises(RuntimeError):
            _ = session.tools

    async def test_tools_are_available_inside_the_context(self) -> None:
        session = ClientSession()
        async with session:
            assert session.tools


class _StubResult:
    """What `fastmcp.Client.call_tool` returns, narrowed to what we read."""

    def __init__(self, structured: dict[str, Any]) -> None:
        self.structured_content = structured


def _outline_payload() -> dict[str, Any]:
    """A structured result standing in for what `plan_story` returns.

    Deliberately nested and multi-typed. A flat string payload would round-trip
    through an f-string unharmed, so it could not detect the defect these tests
    exist to catch.
    """
    return {
        "title": "Kit and the Moon",
        "theme": "wanting something far away",
        "beats": [
            {"position": 1, "summary": "Maryam finds a fox"},
            {"position": 2, "summary": "They build a paper rocket"},
        ],
        "grounding": {
            "facts": [{"claim": "The Moon has no air.", "chunk_id": "moon#1"}]
        },
    }


class TestOutlineRoundTrip:
    """The approved outline must reach `write_story` byte-identical.

    A tempting shortcut appends tool results as
    ``f"Tool '{name}' executed successfully. Result: {result}"`` inside a
    ``role="user"`` text part. Do that and a ``StoryOutline`` becomes prose the
    model has to retype -- which is exactly the "client helpfully rebuilds the
    outline from its own summary" failure this rules out, arriving through our
    own client rather than a third party's.

    Live runs have confirmed the round trip byte-identical, nested grounding
    included. These tests are what stop it coming back.
    """

    async def test_a_tool_result_re_enters_history_as_a_tool_message(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        model = FakeClientModel(
            _turn_calling("plan_story", _brief_args()), _final_turn()
        )
        async with ClientSession(model=model) as session:
            monkeypatch.setattr(session, "_call_tool", _stub_call(_outline_payload()))
            await session.send("plan a story")

        # The second model call is the one that sees the tool result.
        history = model.calls[-1]
        tool_messages = [m for m in history if isinstance(m, ToolMessage)]
        assert tool_messages, (
            "the tool result never re-entered history as a ToolMessage"
        )

    async def test_an_outline_survives_the_round_trip_byte_identical(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        payload = _outline_payload()
        model = FakeClientModel(
            _turn_calling("plan_story", _brief_args()), _final_turn()
        )
        async with ClientSession(model=model) as session:
            monkeypatch.setattr(session, "_call_tool", _stub_call(payload))
            await session.send("plan a story")

        tool_message = next(m for m in model.calls[-1] if isinstance(m, ToolMessage))
        assert json.loads(tool_message.content) == payload

    async def test_the_tool_message_carries_the_calls_id(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Without a matching id the provider rejects the turn, and the symptom
        # is an API error rather than anything resembling the cause.
        model = FakeClientModel(
            _turn_calling("plan_story", _brief_args(), call_id="abc"), _final_turn()
        )
        async with ClientSession(model=model) as session:
            monkeypatch.setattr(session, "_call_tool", _stub_call(_outline_payload()))
            await session.send("plan a story")

        tool_message = next(m for m in model.calls[-1] if isinstance(m, ToolMessage))
        assert tool_message.tool_call_id == "abc"


def _stub_call(payload: dict[str, Any]):
    """Replace real tool execution with a canned structured result."""

    async def _call(name: str, args: dict[str, Any]) -> _StubResult:
        return _StubResult(payload)

    return _call


class TestToolLoop:
    """How many calls run, and when the loop stops."""

    async def test_a_turn_with_no_tool_calls_returns_the_text(self) -> None:
        model = FakeClientModel(_final_turn("Just talking."))
        async with ClientSession(model=model) as session:
            result = await session.send("hello")

        assert result.text == "Just talking."
        assert not result.tool_calls

    async def test_it_executes_every_call_in_a_turn_not_just_the_first(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The Researcher has been measured calling two to four tools *in parallel
        # in a single turn*. A loop that takes only the first call drops the rest
        # silently -- and the symptom would read as a model problem rather than a
        # client bug.
        parallel = AIMessage(
            content="",
            tool_calls=[
                {"id": "c1", "name": "plan_story", "args": _brief_args()},
                {"id": "c2", "name": "plan_story", "args": _brief_args()},
            ],
        )
        model = FakeClientModel(parallel, _final_turn())
        async with ClientSession(model=model) as session:
            monkeypatch.setattr(session, "_call_tool", _stub_call(_outline_payload()))
            result = await session.send("plan two stories")

        assert len(result.executed) == 2

    async def test_a_failed_tool_call_is_reported_to_the_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def _boom(name: str, args: dict[str, Any]) -> Any:
            raise RuntimeError("provider exploded")

        model = FakeClientModel(
            _turn_calling("plan_story", _brief_args()), _final_turn()
        )
        async with ClientSession(model=model) as session:
            monkeypatch.setattr(session, "_call_tool", _boom)
            result = await session.send("plan a story")

        assert result.executed[0].error is not None
        # The model must see the failure, or it cannot recover from it.
        assert any(isinstance(m, ToolMessage) for m in model.calls[-1])

    async def test_the_loop_stops_at_max_tool_turns(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A client that loops is a client that spends money: `write_story` is
        # three to seven model calls per invocation. A model that never stops
        # calling tools must hit a ceiling rather than run until the key does.
        forever = _turn_calling("plan_story", _brief_args())
        model = FakeClientModel(forever)
        async with ClientSession(model=model, max_tool_turns=3) as session:
            monkeypatch.setattr(session, "_call_tool", _stub_call(_outline_payload()))
            result = await session.send("plan forever")

        assert len(model.calls) == 3
        assert len(result.executed) == 3


class TestInspectMode:
    """Which tools run by default, and which are only looked at.

    `try_prompt.py` inspects `write_story` rather than calling it, and its
    reasoning generalises: *its arguments answer the question, and running it
    would buy a whole book to learn nothing more.* So inspect is the default and
    `--execute` is the deliberate choice.
    """

    async def test_plan_story_runs_by_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        model = FakeClientModel(
            _turn_calling("plan_story", _brief_args()), _final_turn()
        )
        async with ClientSession(model=model) as session:
            monkeypatch.setattr(session, "_call_tool", _stub_call(_outline_payload()))
            result = await session.send("plan a story")

        assert result.executed[0].result is not None

    @pytest.mark.parametrize("tool_name", ["write_story", "illustrate_story"])
    async def test_expensive_tools_are_not_executed_by_default(
        self, tool_name: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        called: list[str] = []

        async def _record(name: str, args: dict[str, Any]) -> Any:
            called.append(name)
            return _StubResult({})

        model = FakeClientModel(_turn_calling(tool_name, {}), _final_turn())
        async with ClientSession(model=model) as session:
            monkeypatch.setattr(session, "_call_tool", _record)
            await session.send("write it")

        assert called == []

    async def test_execute_mode_runs_them(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        called: list[str] = []

        async def _record(name: str, args: dict[str, Any]) -> Any:
            called.append(name)
            return _StubResult({"pages": []})

        model = FakeClientModel(_turn_calling("write_story", {}), _final_turn())
        async with ClientSession(model=model, execute=True) as session:
            monkeypatch.setattr(session, "_call_tool", _record)
            await session.send("write it")

        assert called == ["write_story"]

    async def test_inspect_mode_halts_rather_than_faking_a_result(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The trap this mode has to avoid, stated as a test.

        If the "not executed" message reads like a *result*, the model reasons
        onward from it and invents a book -- and the transcript reads like a
        successful run -- a mode that cannot visibly fail. So the message is
        phrased as a halt, and the turn ends
        rather than looping.
        """
        model = FakeClientModel(
            _turn_calling("write_story", {}),
            _final_turn("Here is your finished book!"),
        )
        async with ClientSession(model=model) as session:
            result = await session.send("write it")

        # The loop stopped: the model was never invited to continue from a
        # fabricated result, so its second queued turn was never reached.
        assert len(model.calls) == 1
        assert result.text == ""
        # And the call is still reported as requested, which is the whole point
        # of inspecting.
        assert [c.name for c in result.tool_calls] == ["write_story"]
        assert result.executed == []


class TestTransportSelection:
    """Two transports, and an unknown one fails at construction."""

    def test_an_unknown_transport_is_rejected_before_connecting(self) -> None:
        # At construction, not on first use. A bad transport name should cost
        # nothing and fail where the typo is, rather than after capabilities
        # have been fetched.
        with pytest.raises(ValueError, match="transport"):
            ClientSession(transport="carrier-pigeon")

    def test_stdio_builds_without_spawning_anything(self) -> None:
        # Constructing must not start a subprocess -- FastMCP defers that to
        # __aenter__. If this ever starts hanging, that assumption changed.
        session = ClientSession(transport="stdio")
        assert session is not None
