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

A ``ProviderError`` for upstream chat failures, once retry and fallback exist, is
still deliberately not declared: nothing raises it today, and empty classes are
speculation. ``ImageGenerationError`` below was in that category until image
generation was built and gave it a caller -- which is the bar to clear.
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
    prompt only re-rolls the dice. The revision loops catch it and retry *with the
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


class ImageGenerationError(SparkStoryError):
    """An image provider was reached and did not return a usable image.

    A 503, a rate limit, a timeout, a provider-refused prompt, or a response
    whose bytes are not an image. Transient by assumption, so unlike every other
    error here it **is** retried -- see ``workflows/retries.py``.

    Deliberately a sibling of ``ConfigurationError`` rather than a child. No
    operator fixes a 503 by editing ``.env``, and the tool layer must not offer a
    client a configuration message for it. Configuration failures in the image
    seam are ``ImageConfigurationError`` instead.

    Illustration fails **soft**: the workflow catches this per image and leaves
    that page's frame blank rather than destroying a finished book. So this being
    raised is a normal, expected event, not a run-ending one -- which is the
    opposite of ``UnsafeContentError`` above. What must never happen quietly is a
    *portrait* failing, because that silently removes reference conditioning from
    every page a character appears on; that is recorded in the artifact.
    """


class ImageConfigurationError(ConfigurationError):
    """An image model cannot be built: unknown model id, or a missing API key.

    A ``ConfigurationError`` subclass, and the placement is load-bearing rather
    than tidy. ``mcp/tools/`` translates only ``ConfigurationError`` into a
    ``ToolError``, so an unset ``XAI_API_KEY`` raised as anything else would reach
    a client as an opaque internal error instead of a sentence naming the variable
    to set. Inheriting also picks up the existing ``_retry_on`` exclusion, so a
    missing key cannot be retried three times the way ``GOOGLE_API_KEY`` was.
    """


class AudioGenerationError(SparkStoryError):
    """A speech provider was reached and returned no usable audio.

    The audio twin of ``ImageGenerationError``, with the same placement argument:
    a sibling of ``ConfigurationError`` rather than a child, because no operator
    fixes a 503 by editing ``.env``. Transient by assumption, so it **is** retried.

    Narration fails **soft**, like illustration: the workflow catches this per
    page and records that page as unnarrated rather than destroying a finished
    book. So being raised is a normal, expected event here.

    Two non-obvious cases are deliberately this class rather than a quiet
    fallback. An **empty body** on a 200 -- because a zero-byte MP3 plays as
    silence, and silence is indistinguishable from success on a casual listen. And
    an **unknown voice id**, which the live endpoint answers with 404 plus a JSON
    error body; writing that JSON into ``page-03.mp3`` would leave a file that
    looks like audio and is not.
    """


class AudioConfigurationError(ConfigurationError):
    """A speech model cannot be built: unknown model id, or a missing API key.

    A ``ConfigurationError`` subclass for exactly the reasons
    ``ImageConfigurationError`` is one: the tool layer translates only that class
    into a message naming the variable to set, and inheriting picks up the
    ``_retry_on`` exclusion so a missing key is not retried three times.
    """


class VideoGenerationError(SparkStoryError):
    """A video encoder ran and produced no usable clip.

    The third of these, after ``ImageGenerationError`` and
    ``AudioGenerationError``, with the same placement argument: a sibling of
    ``ConfigurationError`` rather than a child, because no operator fixes a
    non-zero ffmpeg exit by editing ``.env``.

    Retried, because the failures it covers are transient by assumption -- a
    killed subprocess, a full disk, a truncated write. A *malformed invocation* is
    not transient, but it is also a code defect caught offline rather than a
    runtime condition, so it is deliberately not special-cased. That is the same
    call ``AudioGenerationError`` makes about an unknown voice id answering 404.

    Video fails **soft** per page, like illustration and narration: a page whose
    clip fails is recorded and the video is assembled from the rest. So this being
    raised is a normal, expected event rather than a run-ending one.
    """


class VideoConfigurationError(ConfigurationError):
    """A clip maker cannot be built: ffmpeg is absent, or the maker id is unknown.

    A ``ConfigurationError`` subclass for exactly the reasons
    ``ImageConfigurationError`` and ``AudioConfigurationError`` are: the tool layer
    translates only that class into a message naming what to fix, and inheriting
    picks up the ``_retry_on`` exclusion so a missing binary is not retried three
    times.

    **ffmpeg's absence is checked once, before any page is processed**, rather
    than surfacing per page. Every page would fail identically, so failing after
    doing the work tells nobody anything they could not have known up front -- the
    same call ``run_narration_pipeline`` makes for a missing API key.
    """
