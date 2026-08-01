"""Typed exception hierarchy.

Why this exists, given that built-in exceptions with precise messages had been
doing the job: the tool layer needs to decide *which* failures are safe to
report to a client, and doing that by built-in type is unsound. An earlier
version of ``mcp/tools/plan_story.py`` caught ``RuntimeError`` to mean "the API key
is missing" -- so any ``RuntimeError`` raised anywhere beneath it, from LangChain
internals to the transport, would have been reported to the user as
"GOOGLE_API_KEY is not set". Actively misleading, and precisely the kind of thing
that costs an hour of debugging the wrong layer.

The distinction that matters is **who can fix it**:

``ConfigurationError``
    The operator can fix it by editing ``.env`` or ``settings.py``. Safe to
    surface verbatim to a client, because the message is actionable.

Anything else
    A bug, or an upstream failure. Must propagate rather than be dressed up as
    a polite message.

Subclasses raised by a specific layer live with that layer:
``models/exceptions.py`` holds the ones ``models/get_model.py`` raises. Only the
shared bases live here, so ``entities`` depends on nothing.

Siblings intended for later sessions -- a ``ProviderError`` for upstream failures
once retry and fallback exist, and a ``GenerationError`` for output that fails
validation once the evaluator-optimizer loop can act on it -- are deliberately
not declared yet. Nothing raises them today, and empty classes are speculation.
"""


class SparkStoryError(Exception):
    """Base class for errors this package raises deliberately.

    Lets a caller distinguish our failures from arbitrary Python exceptions with
    a single ``except SparkStoryError``.
    """


class ConfigurationError(SparkStoryError):
    """Something in the configuration is wrong and an operator can fix it."""


class StoryStructureError(SparkStoryError):
    """A model's output is well-formed but structurally wrong.

    Raised when output satisfies its schema yet violates a rule the schema cannot
    express -- a page count that disagrees with the brief, a dropped beat, pages
    that wander backwards through the structure.

    Deliberately **not** a ``ConfigurationError``: no operator can fix it by
    editing ``.env``, so it must not be dressed up as a configuration problem.
    It is also excluded from workflow retries, because retrying with an identical
    prompt only re-rolls the dice. Session 4 catches it and retries *with the
    message as feedback*, which is the point of raising something specific.
    """


class UnsafeContentError(SparkStoryError):
    """A safety finding survived every revision, so no book is returned.

    Raised only after the Writer was shown the problem and did not fix it. This
    product writes for a named child whose parent listed things to keep out of
    it, so the guardrail fails closed: returning a book with a known safety
    finding still in it is worse than returning none.

    Deliberately **not** a ``ConfigurationError`` -- no operator fixes it by
    editing ``.env``. But unlike ``StoryStructureError`` it is not a bug either:
    it means the system worked and the answer is no. That distinction is why the
    tool layer translates it for the client instead of letting it propagate as an
    unhandled failure.
    """
