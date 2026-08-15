"""The single seam through which every word of narration is spoken.

The fourth seam, after ``get_chat_model``, ``get_embedder`` and
``get_image_model``, and separate from all three for the reason they are separate
from each other: a chat model takes messages and binds an output schema, an
embedder takes strings and returns vectors, an image model takes a prompt and
returns image bytes, and a speech model takes text and a voice and returns audio
bytes. One factory over four would only branch on which kind an entry is.

What lives behind here:

* **The provider call.** Nothing else in the package knows a TTS endpoint exists.
  A second provider is one function and one registry entry -- which matters more
  than usual here, because ElevenLabs is the answer if a live listen says these
  voices are wrong for a bedtime story.
* **API key resolution**, with the same precise error the other seams give.
* **Bytes, not URLs**, for the reason ``get_image_model`` states: a URL expires,
  and a run artifact pointing at an expired one is worse than useless because it
  looks like a record.
* **The format read from the response header**, never assumed, so a provider
  switching codec cannot leave us writing WAV bytes into a ``.mp3``.

**Everything here was measured against the live endpoint, not read.** Both of
xAI's documentation pages were partly wrong, in opposite directions:

* The REST reference says the response is JSON carrying a base64 ``audio`` field.
  It is **raw bytes** with ``Content-Type: audio/mpeg`` -- verified, magic
  ``ff f3``. The base64 shape appears only under ``with_timestamps``, which is
  never sent, so **that branch is deliberately not implemented**: a code path that
  cannot be tested against reality is worse than its absence.
* Neither page names a model id, and the endpoint needs none -- ``POST /v1/tts``
  returns 200 with no ``model`` field, and supplying one changes nothing. So the
  registry entry carries no ``identifier``.
* ``optimize_streaming_latency`` is documented as ``0-2`` on one page and
  ``0 | 1`` on the other. It is never sent; we write files, not streams.

``get_image_model`` records the same thing from the other endpoint: the docs said
up to 3 source images, the live endpoint requires exactly two. On this provider,
the endpoint is the specification.

**``language`` is the constant ``"en"``.** The API requires the field,
``StoryBrief`` has no language, and multi-language storybooks do not exist -- a
setting for a feature that does not exist cannot be written meaningfully.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import httpx

from sparkstory.config import settings
from sparkstory.entities.exceptions import (
    AudioConfigurationError,
    AudioGenerationError,
)
from sparkstory.utils.logging_utils import get_logger

logger = get_logger(__name__)

#: The API requires a language and the brief has none. See the module docstring.
_LANGUAGE = "en"

#: Generous for the same reason the image seam's is: a page of narration is ~80 KB
#: of audio, and the default 5s timeout expires long before the provider answers.
_TIMEOUT = httpx.Timeout(120.0)

#: Sent explicitly even though the endpoint defaults to mp3 without it. A provider
#: changing its default would otherwise silently change what we write to disk --
#: and the file extension is chosen from the response, so the two would disagree.
_OUTPUT_FORMAT = {"codec": "mp3"}

#: How much of an error body to quote. The measured failure is small JSON
#: (`{"error": "TTS synthesis failed: ..."}`), but a proxy returning an HTML
#: error page would fill the log with markup nobody reads.
_ERROR_EXCERPT = 300


@dataclass(frozen=True)
class GeneratedAudio:
    """One page of narration, as bytes plus the format they are in."""

    data: bytes
    #: Lowercase, no dot -- "mp3" or "wav". Names the file on disk, so it is read
    #: from the response's own Content-Type rather than assumed.
    audio_format: str


#: ``(text, voice_id, speed) -> audio``. A bounded float rather than an
#: instructions string, which is what an earlier design had: nothing a model
#: writes reaches this seam, so there is no free-text field to be filled lazily.
Speaker = Callable[[str, str, float], Awaitable[GeneratedAudio]]


@dataclass(frozen=True)
class SpeechModel:
    """What the narration workflow needs from a speech provider."""

    speak: Speaker


def _audio_from_response(response: httpx.Response, model_id: str) -> GeneratedAudio:
    """Turn one provider response into audio bytes.

    Raises:
        AudioGenerationError: the provider was reached and returned no usable
            audio -- a non-2xx status, a non-audio body, or an empty one.
            Deliberately the retryable class: a refusal or a rate limit is
            transient in the way a missing key is not.
    """
    content_type = (response.headers.get("content-type") or "").lower()

    if response.status_code >= 400:
        raise AudioGenerationError(
            f"Model {model_id!r} returned {response.status_code}: "
            f"{response.text[:_ERROR_EXCERPT]}"
        )

    # Checked before the body, because this is the case that would otherwise
    # write a JSON error or an HTML page into a file named .mp3 -- a file that
    # looks like audio and is not.
    if not content_type.startswith("audio/"):
        raise AudioGenerationError(
            f"Model {model_id!r} returned {content_type!r} rather than audio. "
            f"Body begins: {response.text[:200]!r}"
        )

    if not response.content:
        # A zero-byte file plays as silence, and silence is indistinguishable
        # from success on a casual listen, so this must not pass as a result.
        raise AudioGenerationError(
            f"Model {model_id!r} returned {response.status_code} carrying no audio."
        )

    # `audio/mpeg; charset=binary` must not become the extension `mpeg; charset`.
    audio_format = content_type.split(";", 1)[0].strip().rsplit("/", 1)[-1]
    if audio_format == "mpeg":
        audio_format = "mp3"

    return GeneratedAudio(data=response.content, audio_format=audio_format)


def get_speech_model(model_id: str) -> SpeechModel:
    """Build the speech model named by ``model_id``.

    Raises:
        AudioConfigurationError: the id is not in the registry, or its API key is
            unset. Not retryable -- the fix is one line in ``.env``, and a
            missing key retried three times prints three tracebacks for it.
    """
    config: dict[str, Any] | None = settings.speech_configs.get(model_id)
    if config is None:
        known = ", ".join(sorted(settings.speech_configs)) or "none"
        raise AudioConfigurationError(
            f"Unknown speech model {model_id!r}. Known entries: {known}."
        )

    env_var = config["api_key_env_var"]
    api_key = settings.api_key_for(env_var)
    if not api_key:
        raise AudioConfigurationError(
            f"{env_var} is not set, and speech model {model_id!r} needs it."
        )

    base_url = config["params"]["base_url"]

    async def speak(text: str, voice_id: str, speed: float) -> GeneratedAudio:
        payload = {
            "text": text,
            "voice_id": voice_id,
            "language": _LANGUAGE,
            "output_format": _OUTPUT_FORMAT,
            "speed": speed,
        }
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            try:
                response = await client.post(
                    f"{base_url}/tts",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
            except httpx.HTTPError as exc:
                # A transport failure is the retryable class, like a 503 -- and it
                # must not surface as a bare httpx error, or `_retry_on` would
                # classify it by LangGraph's default, which returns True for
                # everything it does not recognise.
                raise AudioGenerationError(
                    f"Model {model_id!r} could not be reached: {exc}"
                ) from exc

        return _audio_from_response(response, model_id)

    return SpeechModel(speak=speak)
