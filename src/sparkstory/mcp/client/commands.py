"""Parsing what a user typed at the client prompt.

Parsing is separated from acting for one reason: it is the half that can be
tested exhaustively without a server, a model or a transport. ``repl.py`` stays
thin by delegating here, which matters because an ``input()`` loop is close to
untestable and finding P is the record of what a deliberately-untested script
costs -- it died at a live run after the search had already been paid for.

**No thinking toggle.** The course client offers ``/model-thinking-switch``,
which reads ``google.genai`` thought summaries. This project's model seam spans
two providers and Grok exposes no equivalent, so the command would work on one
provider and silently do nothing on the other. A command that lies about what it
did is worse than an absent one.
"""

from dataclasses import dataclass
from enum import StrEnum, auto

#: The prefix that makes a line a command rather than something to say to the
#: model. Only a *leading* slash counts -- "he/she" and "24/7" are ordinary text.
_PREFIX = "/"

#: Commands taking an argument, as `/<verb>/<target>`.
_GET_PROMPT = "/prompt/"
_GET_RESOURCE = "/resource/"


class CommandKind(StrEnum):
    """What a typed line turned out to be."""

    EMPTY = auto()
    MESSAGE = auto()
    LIST_TOOLS = auto()
    LIST_PROMPTS = auto()
    LIST_RESOURCES = auto()
    GET_PROMPT = auto()
    GET_RESOURCE = auto()
    QUIT = auto()
    # Distinct from MESSAGE deliberately. A typo'd command that fell through to
    # the model would spend a call answering a question nobody asked, and the
    # reply would read as the command having worked.
    UNKNOWN = auto()


@dataclass(frozen=True)
class Command:
    """One parsed line."""

    kind: CommandKind
    #: The original text, for MESSAGE and for reporting an UNKNOWN command back.
    text: str = ""
    #: The argument of `/prompt/<name>` or `/resource/<uri>`.
    target: str = ""


_LISTINGS = {
    "/tools": CommandKind.LIST_TOOLS,
    "/prompts": CommandKind.LIST_PROMPTS,
    "/resources": CommandKind.LIST_RESOURCES,
    "/quit": CommandKind.QUIT,
    "/exit": CommandKind.QUIT,
}


def parse(line: str) -> Command:
    """Turn a typed line into a :class:`Command`."""
    text = line.strip()

    if not text:
        return Command(CommandKind.EMPTY)

    if not text.startswith(_PREFIX):
        return Command(CommandKind.MESSAGE, text=text)

    if text in _LISTINGS:
        return Command(_LISTINGS[text], text=text)

    # `removeprefix` rather than `split("/")`. A resource URI is
    # `sparkstory://library`, which contains the scheme separator -- splitting on
    # every slash yields "sparkstory:" and the symptom is "resource not found"
    # on a resource sitting in the listing the user just read.
    for prefix, kind in (
        (_GET_PROMPT, CommandKind.GET_PROMPT),
        (_GET_RESOURCE, CommandKind.GET_RESOURCE),
    ):
        if text.startswith(prefix):
            target = text.removeprefix(prefix).strip()
            if target:
                return Command(kind, text=text, target=target)
            return Command(CommandKind.UNKNOWN, text=text)

    return Command(CommandKind.UNKNOWN, text=text)


def describe_commands() -> str:
    """One-line help, shown at startup and after an unknown command."""
    return "/tools  /prompts  /resources  /prompt/<name>  /resource/<uri>  /quit"
