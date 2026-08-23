"""Where a tool is allowed to write, and how a caller's directory name gets there.

Shared by the three tools that produce files -- `write_story`, `illustrate_story`
and `narrate_story` -- because they must agree. They did not, and the divergence
is visible in this repository: `write_story` confined its output under `outputs/`
while its two siblings passed the caller's string straight to `Path(...)`, so one
`output_directory` of "kim-metocondry" put the book at
`outputs/kim-metocondry/story.json` and the pictures at `./kim-metocondry/`. A
book and its illustrations ended up in two different places from one name, and
the stray directory sat untracked in the repository root.

That is why this lives in its own module rather than being imported from
`write_story.py`. Copying the function three times lets the three drift again,
and importing the prose tool from the illustration tool asserts a dependency
between them that does not exist -- what they actually share is this policy.
"""

from pathlib import Path

from fastmcp.exceptions import ToolError

#: Every artifact lands under here, whatever the caller asked for.
#:
#: `output_directory` is chosen by an LLM client, and a live run showed why that
#: matters: told to use `outputs/<name>`, the model passed `tara_star_river` and
#: the book landed in the repo root. A prompt can ask and cannot enforce -- the
#: same reason `ChildId` is a type rather than a sanitising call inside the
#: store. Confining here also means a client cannot write to an arbitrary path.
_OUTPUT_ROOT = Path("outputs")


def resolve_destination(output_directory: str) -> Path:
    """Place a caller's directory name under the output root.

    A leading `outputs/` is accepted and not doubled: a client that follows the
    prompt and one that does not must reach the same place, or obedience is what
    produces the surprising path.

    An absolute path is accepted only if it already points inside the root,
    which is what a client sends back when it passes `write_story`'s `saved_to`
    to a media tool. Anything else escaping is refused rather than quietly
    rewritten: a client that asked for somewhere it cannot have should be told
    so -- silently redirecting it would leave the artifacts somewhere the client
    will not look for them.
    """
    asked = Path(output_directory)
    root = _OUTPUT_ROOT.resolve()

    if asked.is_absolute():
        # Accepted only when it is already inside the root -- which means, in
        # practice, a path this function produced. `write_story` returns its
        # resolved directory as `saved_to`, and the prompt tells a client to
        # hand that straight back here so a book and its pictures stay
        # together. Refusing every absolute path would make obeying the prompt
        # the thing that fails, while letting any absolute path through would
        # give up the confinement entirely; the containment check below is what
        # separates the two, and `/etc/sparkstory` still does not survive it.
        destination = asked.resolve()
    else:
        # The literal string a client was told to send, not `_OUTPUT_ROOT.name`
        # -- the root is patchable in tests, and a client's "outputs/" prefix
        # should be recognised wherever books are actually being written.
        relative = Path(*asked.parts[1:]) if asked.parts[:1] == ("outputs",) else asked
        if not relative.parts:
            raise ToolError("output_directory must name a directory for the book.")
        destination = (_OUTPUT_ROOT / relative).resolve()

    if destination == root or not destination.is_relative_to(root):
        raise ToolError(
            f"output_directory must stay inside {_OUTPUT_ROOT}/: {output_directory}"
        )
    return destination
