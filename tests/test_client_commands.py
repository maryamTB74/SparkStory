"""Parsing what a user typed at the client prompt.

Pure functions over strings, so all of this is testable without a server, a
model or a transport. The REPL itself stays thin precisely so that this is where
the behaviour lives.
"""

import pytest

from sparkstory.mcp.client.commands import Command, CommandKind, parse


def test_a_bare_message_is_a_message() -> None:
    parsed = parse("A story about a fox")

    assert parsed.kind is CommandKind.MESSAGE
    assert parsed.text == "A story about a fox"


@pytest.mark.parametrize(
    ("typed", "expected"),
    [
        ("/tools", CommandKind.LIST_TOOLS),
        ("/prompts", CommandKind.LIST_PROMPTS),
        ("/resources", CommandKind.LIST_RESOURCES),
        ("/quit", CommandKind.QUIT),
    ],
)
def test_listing_commands_are_recognised(typed: str, expected: CommandKind) -> None:
    assert parse(typed).kind is expected


def test_commands_ignore_surrounding_whitespace() -> None:
    assert parse("  /tools  ").kind is CommandKind.LIST_TOOLS


def test_a_prompt_command_carries_its_name() -> None:
    parsed = parse("/prompt/create_storybook")

    assert parsed.kind is CommandKind.GET_PROMPT
    assert parsed.target == "create_storybook"


def test_a_resource_command_keeps_a_uri_containing_slashes() -> None:
    # The trap. `sparkstory://library` contains the scheme separator, so a naive
    # split("/") mangles it into "sparkstory:" -- and the symptom is "resource
    # not found" on a resource that plainly exists in the listing.
    parsed = parse("/resource/sparkstory://library")

    assert parsed.kind is CommandKind.GET_RESOURCE
    assert parsed.target == "sparkstory://library"


def test_an_unknown_slash_command_is_not_sent_to_the_model() -> None:
    # A typo'd command must not silently become a chat message. That would spend
    # a model call and answer a question nobody asked -- and the user would read
    # the reply as the command having worked.
    parsed = parse("/tolos")

    assert parsed.kind is CommandKind.UNKNOWN
    assert parsed.text == "/tolos"


def test_a_prompt_command_with_no_name_is_unknown() -> None:
    assert parse("/prompt/").kind is CommandKind.UNKNOWN


def test_an_empty_line_is_empty() -> None:
    # Distinct from a message: the REPL re-prompts rather than invoking a model
    # on nothing.
    assert parse("   ").kind is CommandKind.EMPTY


def test_a_message_containing_a_slash_is_still_a_message() -> None:
    # Only a *leading* slash marks a command. "he/she" or a date must not be
    # swallowed as one.
    parsed = parse("a story about he/she pronouns")

    assert parsed.kind is CommandKind.MESSAGE


def test_parse_returns_a_command() -> None:
    assert isinstance(parse("/tools"), Command)
