"""An interactive client for the SparkStory MCP server.

Usage::

    uv run sparkstory-client                       # in-memory, inspect-only
    uv run sparkstory-client --transport stdio     # server as a subprocess
    uv run sparkstory-client --execute             # actually write the book

**Deliberately thin.** Every branch worth testing lives in ``commands.py`` and
``session.py``; this module is an ``input()`` loop and some printing. That split
is the lesson finding P records: ``scripts/write_one_story.py`` is untested by
choice, and it died at a live run *after* the search had been paid for. An
untestable surface should be as small as it can be, not as convenient.

**This writes to stdout, and that is correct.** Non-obvious rule 2 binds the
*server* -- under stdio transport its stdout carries JSON-RPC. A client is the
other end of that pipe, and its terminal output is the whole point.
"""

import argparse
import asyncio

from sparkstory.mcp.client.commands import (
    Command,
    CommandKind,
    describe_commands,
    parse,
)
from sparkstory.mcp.client.session import TRANSPORTS, ClientSession
from sparkstory.utils.logging_utils import configure_logging


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sparkstory-client",
        description="Talk to the SparkStory MCP server.",
    )
    parser.add_argument(
        "--transport",
        "-t",
        choices=list(TRANSPORTS),
        default="in-memory",
        help="in-memory runs the server in this process; stdio spawns it.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Registry id of the model that plays the client (default: planner_model).",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Actually run write_story and illustrate_story. Off by default: "
            "their arguments answer most questions, and running them buys a "
            "whole book."
        ),
    )
    return parser


def _print_capabilities(session: ClientSession) -> None:
    print(f"\nTools      : {', '.join(t.name for t in session.tools) or 'none'}")
    print(f"Prompts    : {', '.join(p.name for p in session.prompts) or 'none'}")
    print(f"Resources  : {', '.join(str(r.uri) for r in session.resources) or 'none'}")
    print(f"Commands   : {describe_commands()}\n")


async def _handle(command: Command, session: ClientSession) -> bool:
    """Act on one parsed command. Returns False to end the session."""
    match command.kind:
        case CommandKind.QUIT:
            return False

        case CommandKind.EMPTY:
            return True

        case CommandKind.UNKNOWN:
            print(f"Unknown command {command.text!r}. Try: {describe_commands()}")
            return True

        case CommandKind.LIST_TOOLS:
            for tool in session.tools:
                print(
                    f"  {tool.name}: {(tool.description or '').strip().splitlines()[0]}"
                )
            return True

        case CommandKind.LIST_PROMPTS:
            for prompt in session.prompts:
                print(f"  {prompt.name}: {(prompt.description or '').strip()}")
            return True

        case CommandKind.LIST_RESOURCES:
            for resource in session.resources:
                print(f"  {resource.uri}: {(resource.description or '').strip()}")
            return True

        case CommandKind.GET_RESOURCE:
            print(await session.read_resource(command.target))
            return True

        case CommandKind.GET_PROMPT:
            # A prompt is a workflow, so it goes to the model rather than being
            # printed -- that is the difference between a prompt and a resource.
            await _say(await session.get_prompt(command.target), session)
            return True

        case CommandKind.MESSAGE:
            await _say(command.text, session)
            return True

    return True


async def _say(message: str, session: ClientSession) -> None:
    """Send one message and report what the model did with it."""
    result = await session.send(message)

    for call in result.executed:
        outcome = f"failed: {call.error}" if call.error else "ok"
        print(f"  [tool] {call.name} -> {outcome}")

    withheld = {c.name for c in result.tool_calls} - {c.name for c in result.executed}
    for name in sorted(withheld):
        # Said plainly, and only to the human. The model was never told a
        # stand-in result -- see the halt in `session.send`.
        print(f"  [tool] {name} requested; not executed (pass --execute to run it)")

    if result.text:
        print(f"\n{result.text}\n")


async def _run(args: argparse.Namespace) -> int:
    async with ClientSession(
        transport=args.transport, model_id=args.model, execute=args.execute
    ) as session:
        _print_capabilities(session)

        while True:
            try:
                line = input("you> ")
            except EOFError, KeyboardInterrupt:
                print()
                return 0

            try:
                if not await _handle(parse(line), session):
                    return 0
            except Exception as exc:  # noqa: BLE001 -- a bad turn must not end the session
                print(f"  [error] {exc}")


def main() -> None:
    """Console-script entry point."""
    configure_logging()
    raise SystemExit(asyncio.run(_run(_build_parser().parse_args())))


if __name__ == "__main__":
    main()
