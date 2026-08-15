"""The speech seam: bytes out, and a loud error for anything else."""

import httpx
import pytest

from sparkstory.entities.exceptions import (
    AudioConfigurationError,
    AudioGenerationError,
)
from sparkstory.models.get_speech_model import (
    GeneratedAudio,
    _audio_from_response,
    get_speech_model,
)


def _response(status: int, content: bytes, content_type: str) -> httpx.Response:
    return httpx.Response(
        status_code=status,
        content=content,
        headers={"content-type": content_type},
        request=httpx.Request("POST", "https://api.x.ai/v1/tts"),
    )


def test_raw_mpeg_bytes_become_generated_audio() -> None:
    # Measured shape: the live endpoint returns raw bytes with
    # `Content-Type: audio/mpeg`, magic ff f3 -- NOT base64 JSON. The REST
    # reference page says otherwise and is wrong.
    audio = _audio_from_response(
        _response(200, b"\xff\xf3\xc4\xc4payload", "audio/mpeg"), "grok-speech"
    )
    assert isinstance(audio, GeneratedAudio)
    assert audio.data.startswith(b"\xff\xf3")
    assert audio.audio_format == "mp3"


def test_a_wav_content_type_is_honoured_rather_than_assumed() -> None:
    # The format names the file on disk, so trusting the header rather than
    # assuming mp3 is what stops us writing WAV bytes into a .mp3 -- the same
    # reasoning `get_image_model` applies to `mime_type`.
    audio = _audio_from_response(
        _response(200, b"RIFFdata", "audio/wav"), "grok-speech"
    )
    assert audio.audio_format == "wav"


def test_a_charset_suffix_does_not_leak_into_the_format() -> None:
    audio = _audio_from_response(
        _response(200, b"\xff\xf3data", "audio/mpeg; charset=binary"), "grok-speech"
    )
    assert audio.audio_format == "mp3"


def test_a_json_error_body_raises_rather_than_being_written_to_disk() -> None:
    # Measured: an unknown voice_id returns 404 with
    # {"error": "TTS synthesis failed: Voice 'x' not found"}. Writing that JSON
    # into page-03.mp3 would leave a file that looks like audio and is not.
    with pytest.raises(AudioGenerationError, match="not found"):
        _audio_from_response(
            _response(
                404,
                b'{"error":"TTS synthesis failed: Voice \'nope\' not found"}',
                "application/json",
            ),
            "grok-speech",
        )


def test_an_empty_body_raises() -> None:
    # A zero-byte MP3 plays as silence, and silence is indistinguishable from
    # success on a casual listen. Fail loudly instead.
    with pytest.raises(AudioGenerationError, match="no audio"):
        _audio_from_response(_response(200, b"", "audio/mpeg"), "grok-speech")


def test_a_non_audio_success_body_raises() -> None:
    with pytest.raises(AudioGenerationError, match="audio"):
        _audio_from_response(
            _response(200, b"<html>nope</html>", "text/html"), "grok-speech"
        )


def test_unknown_model_id_is_a_configuration_error() -> None:
    with pytest.raises(AudioConfigurationError, match="not-a-model"):
        get_speech_model("not-a-model")


def test_missing_key_is_a_configuration_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # Not retryable: the fix is one line in .env. Retrying it just prints a
    # traceback per attempt for a problem no retry can solve.
    from sparkstory.config import settings

    monkeypatch.setattr(settings, "xai_api_key", None)
    with pytest.raises(AudioConfigurationError, match="XAI_API_KEY"):
        get_speech_model("grok-speech")


async def test_speak_sends_the_measured_request_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The payload is asserted because every field in it was measured.

    No `model` key: the endpoint returns 200 without one. `language` is the
    hardcoded "en": the brief has no language field, and there is no point
    adding configuration for a feature that does not exist. `output_format`
    is sent explicitly even though it is optional, so a provider changing its
    default cannot silently change what we write to disk.
    """
    sent: dict[str, object] = {}

    class _FakeClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        async def __aenter__(self) -> _FakeClient:
            return self

        async def __aexit__(self, *exc: object) -> None:
            return None

        async def post(
            self, url: str, *, headers: dict[str, str], json: dict[str, object]
        ) -> httpx.Response:
            sent["url"] = url
            sent["headers"] = headers
            sent["json"] = json
            return _response(200, b"\xff\xf3audio", "audio/mpeg")

    monkeypatch.setattr(httpx, "AsyncClient", _FakeClient)

    model = get_speech_model("grok-speech")
    audio = await model.speak("Page one.", "eve", 0.9)

    assert audio.audio_format == "mp3"
    assert sent["url"] == "https://api.x.ai/v1/tts"
    payload = sent["json"]
    assert isinstance(payload, dict)
    assert payload["text"] == "Page one."
    assert payload["voice_id"] == "eve"
    assert payload["speed"] == 0.9
    assert payload["language"] == "en"
    assert payload["output_format"] == {"codec": "mp3"}
    assert "model" not in payload


async def test_a_transport_failure_is_the_retryable_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _DyingClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        async def __aenter__(self) -> _DyingClient:
            return self

        async def __aexit__(self, *exc: object) -> None:
            return None

        async def post(self, *args: object, **kwargs: object) -> httpx.Response:
            raise httpx.ConnectError("no route to host")

    monkeypatch.setattr(httpx, "AsyncClient", _DyingClient)

    model = get_speech_model("grok-speech")
    with pytest.raises(AudioGenerationError, match="could not be reached"):
        await model.speak("Page one.", "eve", 1.0)
