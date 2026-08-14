"""Play a client against the ``create_storybook`` prompt and measure obedience.

The test suite cannot answer the only questions that matter about an MCP prompt.
Every test in ``tests/test_server.py`` uses an in-memory client with **no LLM in
it**, so none of them exercises whether a client actually does what the
instructions say. This script puts a real model on the client side.

The relationship is inverted from ``write_one_story.py``: there, the script drives
the engine directly and no MCP is involved. Here SparkStory is the *server* and
the script is the *client*, which is the only arrangement in which the prompt is
under test at all.

Prompt text and tool schemas are fetched from the running server rather than
copied, so this measures what a client is really handed.

**What is measured, not judged.** After each turn the model's requested tool calls
are inspected. That turns "did it confirm with the user?" into an observation:

    turn 2 requests [plan_story]                -> stopped, question 1 passes
    turn 2 requests [plan_story, write_story]   -> did not stop, question 1 fails

The answer depends on which model plays the client, so ``--model`` exists and a
single run proves little on its own. A small model disobeying says little about a
frontier one, and vice versa.

Costs roughly six client calls plus up to two real ``plan_story`` calls, each of
which now runs the outline critic. ``write_story`` is **inspected, never called**:
its arguments answer the question, and running it would buy a whole book to learn
nothing more.

Examples::

    uv run python scripts/try_prompt.py
    uv run python scripts/try_prompt.py --model grok-4
"""

import argparse
import asyncio
import json
from typing import Any

from fastmcp import Client
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from sparkstory.config import settings
from sparkstory.mcp.client.session import as_openai_tools
from sparkstory.mcp.server import create_server
from sparkstory.utils.logging_utils import configure_logging

# Deliberately omits pronouns while making the child's apparent gender guessable
# from name and context. That is the whole point of question 4: the instructions
# say to ask, and "she's 5" plus a feminine-reading name is exactly the bait.
OPENING_MESSAGE = (
    "A story for Maryam, she's 5 and an early reader. She loves foxes and "
    "astronomy. Something about a fox who wants to visit the moon."
)

FOLLOW_UP_MESSAGE = "Make Maryam the one who wants to go, not the fox."

APPROVAL_MESSAGE = "That's perfect, yes. Please write it."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=settings.planner_model,
        help="Registry id of the model that plays the CLIENT (default: planner_model).",
    )
    return parser.parse_args()


# Imported from the client package rather than redefined here. It moved there
# when `ClientSession` was built, and two copies of a schema converter is exactly
# the kind of duplication that drifts silently -- the REPL would start binding
# tools in a shape this harness had stopped testing.
#
# What is deliberately NOT shared is the turn driving. `ClientSession.send` runs
# the tool loop to completion, which is the right shape for a REPL and the wrong
# one here: this script's whole method is stopping between turns to inspect what
# was requested, and `write_story` is never executed at all. Rebuilding that on
# top of a loop that runs to completion would risk the six verdicts to remove
# duplication that is not actually there.


def report_turn(label: str, response: AIMessage) -> list[dict]:
    """Print what the model did this turn and return its tool calls."""
    calls = response.tool_calls or []
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
    print(f"tool calls : {[c['name'] for c in calls] or 'none'}")
    text = (response.content or "").strip() if isinstance(response.content, str) else ""
    if text:
        print(f"said       : {text[:600]}")
    return calls


def verdict(question: str, passed: bool, detail: str) -> bool:
    print(f"  [{'PASS' if passed else 'FAIL'}] {question}\n         {detail}")
    return passed


async def main() -> int:
    args = parse_args()
    configure_logging()

    async with Client(create_server()) as client:
        prompt_text = (await client.get_prompt("create_storybook")).messages[0].content
        instructions = prompt_text.text
        tools = await client.list_tools()

        model = get_client_model(args.model, tools)

        # --- Turn 1: incomplete brief, pronouns withheld -------------------
        messages: list[Any] = [
            SystemMessage(content=instructions),
            HumanMessage(content=OPENING_MESSAGE),
        ]
        first = await model.ainvoke(messages)
        calls_1 = report_turn("TURN 1 -- brief with no pronouns given", first)
        messages.append(first)

        asked_first = not calls_1

        # --- Turn 2: supply what it should have asked for ------------------
        if asked_first:
            messages.append(HumanMessage(content="She uses she/her. Please go ahead."))
            second = await model.ainvoke(messages)
            calls_2 = report_turn("TURN 2 -- pronouns supplied", second)
            messages.append(second)
        else:
            # It planned without asking. Question 1 is still measurable from
            # turn 1, so reuse it rather than spending another call.
            second, calls_2 = first, calls_1

        planned = [c for c in calls_2 if c["name"] == "plan_story"]
        wrote = [c for c in calls_2 if c["name"] == "write_story"]

        # --- Turn 3: hand back a real outline, watch for the stop ----------
        shown_outline = ""
        if planned:
            call = planned[0]
            result = await client.call_tool("plan_story", call["args"])
            outline_json = json.dumps(result.structured_content or {}, indent=2)
            messages.append(ToolMessage(content=outline_json, tool_call_id=call["id"]))
            third = await model.ainvoke(messages)
            calls_3 = report_turn("TURN 3 -- outline returned", third)
            messages.append(third)
            shown_outline = third.content if isinstance(third.content, str) else ""
        else:
            calls_3 = []

        # --- Turn 4: ask for a change --------------------------------------
        calls_4: list[dict] = []
        if planned and not calls_3:
            messages.append(HumanMessage(content=FOLLOW_UP_MESSAGE))
            fourth = await model.ainvoke(messages)
            calls_4 = report_turn("TURN 4 -- change requested", fourth)
            messages.append(fourth)

        # --- Turns 5 and 6: approve, and watch what it sends to write_story -
        #
        # The hardest thing the prompt asks for, and the reason this harness had
        # to grow past turn 4: `write_story` takes the outline as an argument
        # now, so the client must carry a structured object out of one tool
        # result and into another tool's arguments. Nothing before this turn
        # tests that, and a client that gets it wrong gets an error rather than
        # a book.
        #
        # The write_story call is inspected, never executed. Its arguments are
        # the whole answer, and running it would spend a full book's worth of
        # calls to learn nothing more.
        approved_outline: dict = {}
        sent_outline: Any = None
        calls_6: list[dict] = []
        replanned = [c for c in calls_4 if c["name"] == "plan_story"]
        if replanned:
            call = replanned[0]
            result = await client.call_tool("plan_story", call["args"])
            approved_outline = result.structured_content or {}
            messages.append(
                ToolMessage(
                    content=json.dumps(approved_outline, indent=2),
                    tool_call_id=call["id"],
                )
            )
            fifth = await model.ainvoke(messages)
            report_turn("TURN 5 -- revised outline returned", fifth)
            messages.append(fifth)

            messages.append(HumanMessage(content=APPROVAL_MESSAGE))
            sixth = await model.ainvoke(messages)
            calls_6 = report_turn("TURN 6 -- approved, write it", sixth)
            for c in calls_6:
                if c["name"] == "write_story":
                    sent_outline = c["args"].get("outline")
                    print(f"outline arg: {'present' if sent_outline else 'MISSING'}")
                    break

    print(f"\n{'=' * 70}\nVERDICTS   (client model: {args.model})\n{'=' * 70}")
    results = [
        verdict(
            "Q1  stops between plan_story and write_story",
            bool(planned) and not wrote and not calls_3,
            f"planning turn requested {[c['name'] for c in calls_2]}; "
            f"turn after the outline requested "
            f"{[c['name'] for c in calls_3] or 'nothing (it stopped)'}",
        ),
        verdict(
            "Q2  shows the outline rather than summarising it",
            _looks_like_a_full_outline(shown_outline),
            f"{len(shown_outline.split())} words shown; "
            "read TURN 3 above to judge -- this heuristic is only a hint",
        ),
        verdict(
            "Q3  re-calls plan_story when changes are requested",
            any(c["name"] == "plan_story" for c in calls_4),
            f"requested {[c['name'] for c in calls_4] or 'nothing'} after "
            "being asked for a change",
        ),
        verdict(
            "Q4  asks for pronouns rather than inferring them",
            asked_first and "pronoun" in (first.content or "").lower(),
            "asked before planning" if asked_first else "planned without asking",
        ),
        verdict(
            "Q5  calls write_story once approved, with an outline argument",
            sent_outline is not None,
            f"approval turn requested {[c['name'] for c in calls_6] or 'nothing'}; "
            f"outline argument {'present' if sent_outline else 'absent'}",
        ),
        verdict(
            "Q6  passes back the outline plan_story returned, unchanged",
            _same_outline(sent_outline, approved_outline),
            _outline_diff(sent_outline, approved_outline),
        ),
    ]
    print(f"\n{sum(results)}/6 passed")
    return 0 if all(results) else 1


def _same_outline(sent: Any, approved: dict) -> bool:
    """Did the client thread the object through, or retype it?

    The failure this catches is not a refusal -- it is a client that helpfully
    rebuilds the outline from its own summary, producing a book built from a
    plan the parent never saw. That is the exact defect the whole design was
    changed to prevent, arriving through the client instead of the server.
    """
    return isinstance(sent, dict) and bool(approved) and sent == approved


def _outline_diff(sent: Any, approved: dict) -> str:
    if sent is None:
        return "no outline was sent, so there is nothing to compare"
    if not isinstance(sent, dict):
        return f"outline arrived as {type(sent).__name__}, not an object"
    if not approved:
        return "no approved outline was captured"
    if sent == approved:
        return "byte-identical to what plan_story returned"
    differing = sorted(
        k for k in set(sent) | set(approved) if sent.get(k) != approved.get(k)
    )
    missing = sorted(set(approved) - set(sent))
    return (
        f"differs in {differing}"
        + (f"; missing {missing}" if missing else "")
        + " -- the client rebuilt the outline instead of passing it through"
    )


def _looks_like_a_full_outline(text: str) -> bool:
    """Crude: a real outline listing beats runs long and enumerates.

    A hint, not a verdict -- question 2 is a judgement call and the printed turn
    above is the actual evidence.
    """
    return len(text.split()) > 80


def get_client_model(model_id: str, tools: list[Any]):
    """Build the model that plays the client, with the server's tools bound."""
    from sparkstory.models.get_model import get_chat_model

    return get_chat_model(model_id).bind_tools(as_openai_tools(tools))


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
