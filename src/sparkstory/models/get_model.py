"""The single seam through which every LLM is created.

Agents never construct a model themselves; they ask for one by ``model_id``
and get back a runnable with all the cross-cutting concerns already attached:

* provider parameters resolved from the model registry in ``settings``
* the correct API key looked up, with a precise error when it is absent
* structured output bound, when a schema is supplied
* retries delegated to LangChain via ``max_retries`` in the registry params

Keeping this a single function is the whole point. Tracing, a cost-and-token
audit, a rate limiter, or a fallback provider each become one edit here instead
of an edit in every agent -- which is exactly the retrofit this design avoids.
"""

from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from sparkstory.config import settings
from sparkstory.entities.exceptions import ConfigurationError
from sparkstory.models.exceptions import MissingAPIKeyError, UnknownModelError
from sparkstory.utils.logging_utils import get_logger

logger = get_logger(__name__)

#: Gemini 2.5 accepts ``thinking_budget``; Gemini 3.x accepts ``thinking_level``.
#: Passing both produces an opaque provider-side error, so a registry entry
#: carrying both is rejected here where we can name the offending model.
_EXCLUSIVE_THINKING_PARAMS = ("thinking_budget", "thinking_level")


def get_chat_model(
    model_id: str,
    schema: type[BaseModel] | None = None,
) -> Runnable[Any, Any]:
    """Build a chat model from the registry.

    Args:
        model_id: A key of ``settings.llm_configs``, e.g. ``"gemini-3.5-flash"``.
            Agents pass a value from settings (``settings.planner_model``)
            rather than a literal, so swapping a model is a config change.
        schema: Optional Pydantic model. When given, the returned runnable
            emits a validated instance of it instead of free text.

    Returns:
        A runnable. Note this is not necessarily a ``BaseChatModel``:
        ``with_structured_output`` returns a wrapping runnable, so the declared
        type is the broader ``Runnable``.

    Raises:
        UnknownModelError: ``model_id`` is not in the registry. The message lists
            the ids that are, because this is almost always a typo.
        ConfigurationError: the registry entry carries mutually exclusive
            thinking parameters.
        MissingAPIKeyError: the API key the model needs is not configured.

    All three are ``ConfigurationError`` subclasses, which is what lets the tool
    layer surface them to clients while letting genuine bugs propagate.
    """
    try:
        config = settings.llm_configs[model_id]
    except KeyError:
        known = ", ".join(sorted(settings.llm_configs))
        raise UnknownModelError(
            f"Unknown model_id {model_id!r}. Known models: {known}. "
            "Add an entry to Settings.llm_configs, or fix the *_MODEL value in .env."
        ) from None

    # Copy rather than mutate: `llm_configs` builds a fresh dict on every access
    # today, but that is an implementation detail somebody could reasonably
    # memoise later, at which point in-place edits would corrupt the registry.
    params: dict[str, Any] = dict(config.get("params", {}))

    conflicting = [p for p in _EXCLUSIVE_THINKING_PARAMS if p in params]
    if len(conflicting) > 1:
        raise ConfigurationError(
            f"Model {model_id!r} sets mutually exclusive params {conflicting}. "
            "Use 'thinking_budget' for Gemini 2.5 or 'thinking_level' for "
            "Gemini 3.x, not both."
        )

    api_key_env_var: str = config["api_key_env_var"]
    api_key = settings.api_key_for(api_key_env_var)
    if not api_key:
        raise MissingAPIKeyError(
            f"Model {model_id!r} requires {api_key_env_var}, which is not set. "
            f"Add {api_key_env_var} to your .env (see .env.sample)."
        )

    logger.debug(
        "Building model %r (identifier=%s, structured=%s)",
        model_id,
        config["identifier"],
        schema.__name__ if schema is not None else None,
    )

    model: Runnable[Any, Any] = init_chat_model(
        config["identifier"],
        api_key=api_key,
        **params,
    )

    if schema is not None:
        model = model.with_structured_output(schema)

    return model
