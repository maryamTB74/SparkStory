"""A client session against the SparkStory MCP server.

This is the half of the MCP surface that *consumes* rather than serves, and it
exists because nothing else here can answer the questions that matter about a
prompt. Every test in ``tests/test_server.py`` drives an in-memory client with no
LLM in it, so none of them exercises whether a client actually does what the
instructions say.

**Capabilities are fetched, never imported.** ``session.tools`` comes back over
the MCP boundary even under in-memory transport, so a harness built on this
measures what a client is really handed rather than what we believe we exposed.

**One session, two front ends.** ``repl.py`` drives it from a terminal;
``scripts/try_prompt.py`` drives it from a script and inspects requested calls
without running the expensive ones. That split is why ``send`` returns a
``TurnResult`` separating requested calls from executed ones.
"""

import json
from types import TracebackType
from typing import Any, Self

from fastmcp import Client
from langchain_core.messages import HumanMessage, ToolMessage

from sparkstory.config import _PROJECT_ROOT
from sparkstory.mcp.client.types import ExecutedCall, ToolCall, TurnResult
from sparkstory.mcp.server import create_server

#: How many times one ``send`` may go round the model-then-tools loop.
#:
#: A module constant rather than a setting, per Rule 3: nothing needs to vary it
#: yet, and a flag cannot be meaningfully written before the thing it gates. The
#: bound exists because a client that loops is a client that spends money --
#: ``write_story`` is three to seven model calls per invocation, so an unbounded
#: loop writes books until the key runs out.
MAX_TOOL_TURNS = 6

#: Tools that are inspected rather than run unless ``execute=True``.
#:
#: A name list rather than a cost heuristic, deliberately. A heuristic drifts as
#: tools are added and silently starts running something expensive; a list fails
#: at review, because a new tool forces someone to decide which set it joins.
#:
#: ``plan_story`` is absent on purpose -- it is the cheap path and inspecting it
#: would leave the client with no outline to thread anywhere, which is the thing
#: most worth watching.
INSPECT_ONLY = frozenset({"write_story", "illustrate_story"})

#: The transports offered here, and the reason there are two. ``in-memory``
#: shares a process with the server, which makes it fast and makes a stray
#: ``print()`` in server startup invisible. ``stdio`` runs the server as a
#: subprocess, and is therefore the only mode that can test non-obvious rule 2 --
#: stdout carries JSON-RPC and nothing else may write to it.
#:
#: HTTP is deliberately absent, for the reason ``server.py`` already gives about
#: its own choices: four names for three behaviours is an invitation to pick the
#: deprecated one.
TRANSPORTS = ("in-memory", "stdio")


def _build_client(transport: str) -> Client:
    """Build a FastMCP client for the named transport."""
    if transport == "in-memory":
        return Client(create_server())

    if transport == "stdio":
        # `_PROJECT_ROOT` is imported from config.py rather than recomputed.
        # Non-obvious rule 6: that constant is depth-sensitive and a wrong
        # `parents[N]` raises nothing at all -- it silently resolves against the
        # wrong directory. A second copy here would be a second thing to get
        # wrong, and the failure would present as "the stdio server will not
        # start" with no hint of the cause.
        return Client(
            {
                "mcpServers": {
                    "sparkstory": {
                        "transport": "stdio",
                        "command": "uv",
                        "args": [
                            "--directory",
                            str(_PROJECT_ROOT),
                            "run",
                            "sparkstory",
                            "--transport",
                            "stdio",
                        ],
                    }
                }
            }
        )

    raise ValueError(f"Unknown transport {transport!r}; expected one of {TRANSPORTS}.")


class ClientSession:
    """An open conversation with the SparkStory server."""

    def __init__(
        self,
        transport: str = "in-memory",
        model: Any | None = None,
        model_id: str | None = None,
        max_tool_turns: int = MAX_TOOL_TURNS,
        execute: bool = False,
    ) -> None:
        # Validated here rather than on first use, so a typo costs nothing and
        # fails at the line that contains it.
        self._client = _build_client(transport)
        self._transport = transport
        self._tools: list[Any] | None = None
        self._prompts: list[Any] | None = None
        self._resources: list[Any] | None = None
        # Injected for tests, built from the registry otherwise. Constructor
        # injection rather than monkeypatching a module global, for the reason
        # `models/fake_model.py` gives: a patch targets a string path, so a
        # rename leaves it pointing at nothing and the test passes for the wrong
        # reason.
        self._model = model
        self._model_id = model_id
        self._max_tool_turns = max_tool_turns
        self._execute_all = execute
        self._history: list[Any] = []

    async def __aenter__(self) -> Self:
        await self._client.__aenter__()
        self._tools = await self._client.list_tools()
        self._prompts = await self._client.list_prompts()
        self._resources = await self._client.list_resources()
        if self._model is None:
            self._model = self._build_model()
        return self

    def _build_model(self) -> Any:
        """Build the client's model through the one model seam.

        Imported here rather than at module scope so a session that is handed a
        model never touches the registry -- which is what keeps the offline tests
        offline.

        Going through ``get_chat_model`` is not optional. A client constructing a
        provider directly would bypass the registry, and rule 21 would fire on
        the first ``.env`` that pins every stage to one provider -- which is
        exactly what Maryam's does.
        """
        from sparkstory.config import settings
        from sparkstory.models.get_model import get_chat_model

        model = get_chat_model(self._model_id or settings.planner_model)
        return model.bind_tools(as_openai_tools(self.tools))

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self._client.__aexit__(exc_type, exc, tb)

    def _require_connected(self, fetched: list[Any] | None, what: str) -> list[Any]:
        """Return fetched capabilities, or explain that nothing is connected yet.

        Raises rather than returning an empty list. An empty list reads as "the
        server exposes none of these", which is a wrong answer in the shape of a
        valid one -- and it would surface as a client that mysteriously never
        calls anything.
        """
        if fetched is None:
            raise RuntimeError(
                f"Cannot read {what} before connecting; "
                "use `async with ClientSession() as session:`."
            )
        return fetched

    @property
    def tools(self) -> list[Any]:
        """Tool definitions as the server advertises them."""
        return self._require_connected(self._tools, "tools")

    @property
    def prompts(self) -> list[Any]:
        """Prompt definitions as the server advertises them."""
        return self._require_connected(self._prompts, "prompts")

    @property
    def resources(self) -> list[Any]:
        """Resource definitions as the server advertises them."""
        return self._require_connected(self._resources, "resources")

    async def get_prompt(self, name: str) -> str:
        """Fetch a prompt's text by name.

        Raises ``KeyError`` for an unknown name rather than returning empty text,
        for the same reason the connection guard raises: a client that silently
        proceeds with no instructions looks like a disobedient model.
        """
        if name not in {p.name for p in self.prompts}:
            available = sorted(p.name for p in self.prompts)
            raise KeyError(f"No prompt named {name!r}; server offers {available}.")

        result = await self._client.get_prompt(name)
        return "\n\n".join(
            message.content.text
            for message in result.messages
            if hasattr(message.content, "text")
        )

    async def read_resource(self, uri: str) -> str:
        """Read a resource's text by URI."""
        if uri not in {str(r.uri) for r in self.resources}:
            available = sorted(str(r.uri) for r in self.resources)
            raise KeyError(f"No resource at {uri!r}; server offers {available}.")

        contents = await self._client.read_resource(uri)
        return "\n".join(item.text for item in contents if hasattr(item, "text"))

    def _is_withheld(self, call: ToolCall) -> bool:
        """Is this a tool we look at rather than run?"""
        return not self._execute_all and call.name in INSPECT_ONLY

    async def _call_tool(self, name: str, args: dict[str, Any]) -> Any:
        """Send one tool call to the server.

        A seam of one line, so a test can stub tool execution without stubbing
        the loop that is actually under test.
        """
        return await self._client.call_tool(name, args)

    async def _execute(self, call: ToolCall) -> ExecutedCall:
        """Run one tool call and record what happened, success or failure."""
        try:
            result = await self._call_tool(call.name, call.args)
        except Exception as exc:  # noqa: BLE001 -- reported to the model, not raised
            # Client-side failures are reported, never raised. The model has to
            # see the error to recover from it, and a raise would end a
            # conversation over one bad call.
            return ExecutedCall(
                id=call.id, name=call.name, args=call.args, error=str(exc)
            )

        return ExecutedCall(
            id=call.id,
            name=call.name,
            args=call.args,
            result=getattr(result, "structured_content", None),
        )

    def _record(self, call: ExecutedCall) -> None:
        """Put a tool's outcome back into history as a ``ToolMessage``.

        **This is the line Q6 depends on.** The course client appends
        ``f"Tool '{name}' executed successfully. Result: {result}"`` into a
        ``role="user"`` text part, which turns a ``StoryOutline`` into prose the
        model must retype -- reintroducing the "client rebuilds the outline from
        its own summary" defect the whole design was changed to prevent, this
        time through our own client. JSON in a ``ToolMessage`` keeps it an object.
        """
        content = (
            f"Tool call failed: {call.error}"
            if call.error is not None
            else json.dumps(call.result, indent=2)
        )
        self._history.append(ToolMessage(content=content, tool_call_id=call.id))

    async def send(self, message: str) -> TurnResult:
        """Send one user message and run tools until the model stops asking."""
        self._history.append(HumanMessage(content=message))
        executed: list[ExecutedCall] = []
        requested: list[ToolCall] = []

        for _ in range(self._max_tool_turns):
            response = await self._model.ainvoke(self._history)
            self._history.append(response)

            calls = [
                ToolCall(id=c["id"], name=c["name"], args=c["args"])
                for c in (response.tool_calls or [])
            ]
            requested.extend(calls)

            if not calls:
                text = response.content if isinstance(response.content, str) else ""
                return TurnResult(text=text, tool_calls=requested, executed=executed)

            # Inspect mode halts the turn rather than answering with a stand-in
            # result. Anything phrased as a *result* -- even "not executed" --
            # is something the model reasons onward from, and it will happily
            # announce a finished book it never received. The transcript would
            # then read like a successful run, which is rule 24's shape: a mode
            # that cannot visibly fail. Ending the turn is what makes the
            # omission observable.
            withheld = [c for c in calls if self._is_withheld(c)]
            if withheld:
                return TurnResult(text="", tool_calls=requested, executed=executed)

            # Every call in the turn, not `calls[0]`. Session 5 measured the
            # Researcher calling two to four tools in parallel in a single turn;
            # taking only the first drops the rest silently, and the symptom
            # reads as a model problem rather than a client bug.
            for call in calls:
                ran = await self._execute(call)
                executed.append(ran)
                self._record(ran)

        return TurnResult(text="", tool_calls=requested, executed=executed)


def as_openai_tools(tools: list[Any]) -> list[dict]:
    """Convert MCP tool definitions into the shape ``bind_tools`` accepts.

    Eleven lines, and the whole of what ``langchain-mcp-adapters`` would replace.
    The course declares that dependency in ``mcp_client/pyproject.toml`` and
    imports it nowhere, driving ``fastmcp.Client`` directly instead -- so
    adopting it here would add a package to do what this function already does.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.inputSchema,
            },
        }
        for tool in tools
    ]
