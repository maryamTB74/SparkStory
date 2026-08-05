"""What a ``@task`` retries, and what it must not.

Lives here rather than in either workflow module because both need it, and a
workflow importing another workflow to borrow a constant is a cycle waiting to
happen.
"""

from langgraph.types import RetryPolicy, default_retry_on

from sparkstory.entities.exceptions import (
    ConfigurationError,
    ImageGenerationError,
    StoryStructureError,
    UnsafeContentError,
)


def _retry_on(exc: Exception) -> bool:
    """Retry transient failures only.

    LangGraph's ``default_retry_on`` declines to retry ``ValueError``, and
    pydantic's ``ValidationError`` is one -- so schema failures are excluded for
    free. It returns ``True`` for exception types it does not recognise, though,
    which includes every error of ours. Both current kinds must be excluded, and
    both exclusions were earned:

    ``ConfigurationError``
        Found by running it. A missing ``GOOGLE_API_KEY`` was retried three times,
        printing three tracebacks for a problem whose fix is one line in ``.env``.
        Trying again cannot make a key appear.

    ``StoryStructureError``
        Retrying re-sends an identical prompt and re-rolls the dice, while hiding
        how often an agent gets a page count wrong -- exactly the frequency data
        the evaluator-optimizer loop of a later session is designed from.

    ``UnsafeContentError``
        Retrying cannot make content safe. It is raised only after the Writer was
        already shown the finding and failed to act on it, so an identical second
        attempt buys nothing but latency.

    Listed explicitly rather than excluding ``SparkStoryError`` wholesale: an
    upstream ``ProviderError``, when one exists, *should* be retried.

    ``ImageGenerationError``
        The first error here that **is** retried, and it is named explicitly rather
        than left to fall through. A 503 or a rate limit from an image endpoint is
        exactly what a retry is for. It would already be retried by the fall-through
        below, but rule 10 exists because that fall-through is a trap -- it returns
        ``True`` for everything it does not recognise, so silence here would be
        indistinguishable from an oversight. ``ImageConfigurationError`` is a
        ``ConfigurationError`` and so is excluded above; a missing key is not
        transient.
    """
    if isinstance(exc, ConfigurationError | StoryStructureError | UnsafeContentError):
        return False
    if isinstance(exc, ImageGenerationError):
        return True
    return default_retry_on(exc)


#:``max_attempts=3``, narrowed by ``_retry_on`` above.
RETRY_POLICY = RetryPolicy(max_attempts=3, retry_on=_retry_on)
