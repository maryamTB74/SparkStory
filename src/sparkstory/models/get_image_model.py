"""The single seam through which every image is generated.

The third seam, after ``get_chat_model`` and ``get_embedder``, and separate from
both for the same reason they are separate from each other: a chat model takes
messages and binds an output schema, an embedder takes strings and returns
vectors, and an image model takes a prompt and returns bytes. One factory over
all three would only branch on which kind an entry is.

What lives behind here, and why it is worth a seam at all:

* **The provider call.** Nothing else in the package knows an image endpoint
  exists. A second provider is one function and one registry entry.
* **API key resolution**, with the same precise error ``get_chat_model`` gives.
* **Bytes, not URLs.** A provider may return either; callers get bytes. A URL
  expires, and a run artifact pointing at an expired URL is worse than useless
  because it looks like a record.
* **The reference-image cap.** ``MAX_REFERENCE_IMAGES`` is enforced here, at the
  boundary that owns the limit, rather than trusted to every caller.

**The provider is a plain callable, not a Protocol.** ``retrieval/web/providers.py``
set that precedent for the same situation: one implementation today, no state to
hold, and a Protocol for a single implementation is the abstraction this codebase
defers until a second one exists.

**Why raw ``httpx`` rather than LangChain or the OpenAI SDK.** LangChain has no
image-generation runnable that would carry the xAI ``base_url`` the way a chat
entry does. The OpenAI SDK looked like the answer -- ``client.images.edit`` exists
and takes an ``image`` argument -- but it uploads multipart form data and xAI's
edits endpoint returns **415 Expected `Content-Type: application/json`**. So the
generations endpoint is OpenAI-compatible and the edits endpoint is not, which is
only discoverable by calling it. ``httpx`` is already a dependency (via
``firecrawl`` and the retrieval layer), so this adds no package, and
``retrieval/web/`` already sets the precedent of a provider called over raw HTTP.

**Everything about the edits endpoint here was measured, not read.** The
documentation says "up to 3 source images"; the live endpoint requires *exactly
two*. See ``MAX_REFERENCE_IMAGES``.
"""

import base64
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import httpx

from sparkstory.config import settings
from sparkstory.entities.exceptions import (
    ImageConfigurationError,
    ImageGenerationError,
)
from sparkstory.entities.illustration import MAX_REFERENCE_IMAGES
from sparkstory.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class GeneratedImage:
    """One image, as bytes plus the format they are in."""

    data: bytes
    #: Lowercase, no dot -- "png" or "jpg". Used to name the file on disk, so a
    #: provider that switches format does not leave us writing JPEG into a .png.
    image_format: str


#: A provider generates from a prompt alone, or edits against reference images.
#: Two callables rather than one with an optional argument: the endpoints differ,
#: and a provider that supports generation but not editing should be expressible.
ImageGenerator = Callable[[str], Awaitable[GeneratedImage]]
ImageEditor = Callable[[str, list[bytes]], Awaitable[GeneratedImage]]


@dataclass(frozen=True)
class ImageModel:
    """What the illustration workflow needs from an image provider."""

    generate: ImageGenerator
    edit: ImageEditor


#: Read once, at import: a 200 response carrying a 120 KB base64 payload is normal
#: here, and the default 5s timeout expires long before an image model answers.
_TIMEOUT = httpx.Timeout(240.0)


def _decode(payload: dict[str, Any], model_id: str) -> GeneratedImage:
    """Turn one provider response item into bytes.

    Raises:
        ImageGenerationError: the response carried no usable image. This is the
            "reached the provider, got nothing" case -- a refusal, a moderation
            block, or a shape we do not recognise -- and it is deliberately the
            retryable class rather than a configuration one.
    """
    b64 = payload.get("b64_json")
    if b64:
        # `mime_type` is xAI's own field, absent from the OpenAI shape. Trusted
        # over assuming PNG so a provider switching to JPEG does not leave us
        # writing JPEG bytes into a file named .png.
        mime = payload.get("mime_type") or "image/png"
        image_format = mime.rsplit("/", 1)[-1].lower()
        if image_format == "jpeg":
            image_format = "jpg"
        try:
            return GeneratedImage(data=base64.b64decode(b64), image_format=image_format)
        except (ValueError, TypeError) as exc:
            raise ImageGenerationError(
                f"Model {model_id!r} returned image data that is not valid "
                f"base64: {exc}"
            ) from exc

    # A URL is rejected rather than followed. It expires, so a run artifact
    # pointing at one would look like a record while being useless -- and
    # `response_format='b64_json'` is always sent, so receiving a URL means the
    # provider ignored it and the caller should know.
    if payload.get("url"):
        raise ImageGenerationError(
            f"Model {model_id!r} returned a URL rather than image data despite "
            "response_format='b64_json'. Add URL fetching to "
            "models/get_image_model.py if this provider cannot return bytes."
        )

    raise ImageGenerationError(
        f"Model {model_id!r} returned a response carrying no image."
    )


def _data_uri(data: bytes) -> str:
    """Wrap raw bytes as a PNG data URI, which is how the edits endpoint takes them."""
    return "data:image/png;base64," + base64.b64encode(data).decode()


def get_image_model(model_id: str) -> ImageModel:
    """Build an image model from the registry.

    Args:
        model_id: A key of ``settings.image_configs``. Callers pass a value from
            settings (``settings.illustrator_model``) rather than a literal, so
            swapping a model is a config change.

    Returns:
        An ``ImageModel`` carrying two coroutines. Nothing about the provider,
        its key or its base URL escapes this function.

    Raises:
        ImageConfigurationError: unknown ``model_id``, or its API key is unset.
            A ``ConfigurationError`` subclass, so the tool layer turns it into a
            message naming the variable to set and the workflow never retries it.
    """
    try:
        config = settings.image_configs[model_id]
    except KeyError:
        known = ", ".join(sorted(settings.image_configs)) or "(none configured)"
        raise ImageConfigurationError(
            f"Unknown image model_id {model_id!r}. Known image models: {known}. "
            "Add an entry to Settings.image_configs, or fix ILLUSTRATOR_MODEL "
            "in your .env."
        ) from None

    api_key_env_var: str = config["api_key_env_var"]
    api_key = settings.api_key_for(api_key_env_var)
    if not api_key:
        raise ImageConfigurationError(
            f"Image model {model_id!r} requires {api_key_env_var}, which is not "
            f"set. Add {api_key_env_var} to your .env (see .env.sample)."
        )

    params: dict[str, Any] = dict(config.get("params", {}))
    identifier: str = config["identifier"]
    base_url: str = params.pop("base_url", "https://api.x.ai/v1").rstrip("/")

    logger.debug("Building image model %r (identifier=%s)", model_id, identifier)

    async def _post(endpoint: str, body: dict[str, Any], what: str) -> GeneratedImage:
        """POST one image request and decode the single image it returns.

        Shared by both operations because everything except the body is identical,
        and the error translation is the part worth having in one place.
        """
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                response = await client.post(
                    f"{base_url}/{endpoint}",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={**body, **params},
                )
        # Every transport failure becomes the retryable class. Rule 10 wants that
        # choice made deliberately: a timeout or a dropped connection to an image
        # endpoint is exactly what a retry is for. A missing key never reaches
        # here, having been rejected above as a ConfigurationError.
        except httpx.HTTPError as exc:
            raise ImageGenerationError(
                f"Image model {model_id!r} could not be reached while {what}: {exc}"
            ) from exc

        if response.status_code != 200:
            # The body carries xAI's own message, which is unusually specific
            # (it names the offending field), so it is passed through rather than
            # replaced with a generic status line.
            raise ImageGenerationError(
                f"Image model {model_id!r} failed {what} with HTTP "
                f"{response.status_code}: {response.text[:300]}"
            )

        try:
            data = response.json().get("data") or []
        except ValueError as exc:
            raise ImageGenerationError(
                f"Image model {model_id!r} returned a non-JSON response while "
                f"{what}: {exc}"
            ) from exc

        if not data:
            raise ImageGenerationError(
                f"Image model {model_id!r} returned an empty response while {what}."
            )
        return _decode(data[0], model_id)

    async def generate(prompt: str) -> GeneratedImage:
        """Generate one image from a prompt alone."""
        return await _post(
            "images/generations",
            {"model": identifier, "prompt": prompt, "response_format": "b64_json"},
            "generating",
        )

    async def edit(prompt: str, references: list[bytes]) -> GeneratedImage:
        """Generate one image conditioned on reference images.

        The endpoint requires **exactly** ``MAX_REFERENCE_IMAGES`` references. One
        is padded by repeating it, which is measured to work: repeating a single
        portrait twice produced a correctly conditioned single-character picture.
        More than two is refused rather than truncated -- silently dropping a
        character would produce a picture missing someone the page needs, with
        nothing in the artifact explaining why.

        Raises:
            ImageGenerationError: no references, or more than the endpoint takes.
                Raised as the soft class deliberately, so a single bad page
                degrades to a blank frame rather than destroying a finished book.
        """
        if not references:
            raise ImageGenerationError(
                "edit() requires at least one reference image; use generate() "
                "for a prompt-only image."
            )
        if len(references) > MAX_REFERENCE_IMAGES:
            raise ImageGenerationError(
                f"edit() accepts at most {MAX_REFERENCE_IMAGES} reference "
                f"images, got {len(references)}."
            )

        # Pad to exactly the required count. `[a]` -> `[a, a]`.
        padded = list(references)
        while len(padded) < MAX_REFERENCE_IMAGES:
            padded.append(padded[-1])

        return await _post(
            "images/edits",
            {
                "model": identifier,
                "prompt": prompt,
                "image": [_data_uri(d) for d in padded],
                "response_format": "b64_json",
            },
            "editing",
        )

    return ImageModel(generate=generate, edit=edit)
