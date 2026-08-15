"""The image factory, and the request shape it builds.

Every fact asserted here about the endpoint was measured against the live xAI API
rather than read from documentation -- which was wrong about the model id, the
reference count *and* the transport. These tests are what stop a future
edit quietly reverting to the documented behaviour.

No network: the transport is replaced, so nothing here leaves the process.
"""

import base64
import json
from typing import Any

import httpx
import pytest

from sparkstory.config import Settings
from sparkstory.entities.exceptions import (
    ConfigurationError,
    ImageConfigurationError,
    ImageGenerationError,
)
from sparkstory.entities.illustration import MAX_REFERENCE_IMAGES
from sparkstory.models.get_image_model import get_image_model

_PIXEL = base64.b64encode(b"not-really-an-image").decode()


@pytest.fixture(autouse=True)
def _explicit_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the factory at a Settings built here, not at the machine's .env."""
    monkeypatch.setattr(
        "sparkstory.models.get_image_model.settings",
        Settings(  # type: ignore[call-arg]
            xai_api_key="test-key",
            _env_file=None,
        ),
    )


def _capture(
    monkeypatch: pytest.MonkeyPatch,
    *,
    status: int = 200,
    body: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Replace httpx's POST with a recorder, returning the list of sent bodies."""
    sent: list[dict[str, Any]] = []
    payload = body if body is not None else {"data": [{"b64_json": _PIXEL}]}

    async def fake_post(self: Any, url: str, **kwargs: Any) -> httpx.Response:
        sent.append(
            {"url": url, "json": kwargs.get("json"), "headers": kwargs["headers"]}
        )
        return httpx.Response(
            status,
            content=json.dumps(payload),
            headers={"content-type": "application/json"},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    return sent


class TestConfiguration:
    def test_unknown_id_raises_naming_the_known_ids(self) -> None:
        """The message must list valid ids, because this is almost always a typo."""
        with pytest.raises(ImageConfigurationError) as excinfo:
            get_image_model("grok-imgae")

        message = str(excinfo.value)
        assert "grok-image" in message
        assert "ILLUSTRATOR_MODEL" in message, "must name the setting to fix"

    def test_missing_key_raises_naming_the_variable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "sparkstory.models.get_image_model.settings",
            Settings(xai_api_key=None, _env_file=None),  # type: ignore[call-arg]
        )
        with pytest.raises(ImageConfigurationError) as excinfo:
            get_image_model("grok-image")

        assert "XAI_API_KEY" in str(excinfo.value)

    def test_config_errors_are_configuration_errors(self) -> None:
        """Load-bearing: mcp/tools/ translates only ConfigurationError.

        Placed anywhere else in the hierarchy, an unset key would reach an MCP
        client as an opaque internal error instead of a sentence naming the
        variable to set -- and would be retried three times on the way.
        """
        with pytest.raises(ConfigurationError):
            get_image_model("nope")


class TestGenerate:
    async def test_posts_to_the_generations_endpoint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sent = _capture(monkeypatch)
        await get_image_model("grok-image").generate("a small fox")

        assert sent[0]["url"] == "https://api.x.ai/v1/images/generations"
        assert sent[0]["json"]["prompt"] == "a small fox"
        assert sent[0]["json"]["model"] == "grok-imagine-image"

    async def test_sends_json_not_multipart(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Measured: the edits endpoint 415s on multipart, which the SDK sends."""
        sent = _capture(monkeypatch)
        await get_image_model("grok-image").generate("a small fox")

        assert sent[0]["headers"]["Content-Type"] == "application/json"

    async def test_asks_for_bytes_not_a_url(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A URL expires, so an artifact pointing at one only looks like a record."""
        sent = _capture(monkeypatch)
        await get_image_model("grok-image").generate("a small fox")

        assert sent[0]["json"]["response_format"] == "b64_json"

    async def test_returns_decoded_bytes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _capture(monkeypatch)
        image = await get_image_model("grok-image").generate("a small fox")

        assert image.data == b"not-really-an-image"


class TestImageFormat:
    async def test_trusts_the_providers_mime_type(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """xAI returns JPEG while asking for b64_json, and says so in `mime_type`.

        Assuming PNG would write JPEG bytes into a `.png` file and fail at PDF
        assembly -- one stage away from the cause.
        """
        _capture(
            monkeypatch,
            body={"data": [{"b64_json": _PIXEL, "mime_type": "image/jpeg"}]},
        )
        image = await get_image_model("grok-image").generate("a small fox")

        assert image.image_format == "jpg", "jpeg must normalise to a jpg suffix"

    async def test_defaults_to_png_when_unstated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _capture(monkeypatch, body={"data": [{"b64_json": _PIXEL}]})
        image = await get_image_model("grok-image").generate("a small fox")

        assert image.image_format == "png"


class TestEdit:
    async def test_pads_a_single_reference_to_the_required_two(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Measured: one reference returns 422, two works, and repeating one works.

        So a one-character page is padded rather than refused.
        """
        sent = _capture(monkeypatch)
        await get_image_model("grok-image").edit("a fox asleep", [b"portrait"])

        images = sent[0]["json"]["image"]
        assert len(images) == MAX_REFERENCE_IMAGES
        assert images[0] == images[1], "the single reference is repeated"

    async def test_sends_references_as_data_uris(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sent = _capture(monkeypatch)
        await get_image_model("grok-image").edit("a fox", [b"portrait"])

        assert sent[0]["json"]["image"][0].startswith("data:image/png;base64,")

    async def test_posts_to_the_edits_endpoint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sent = _capture(monkeypatch)
        await get_image_model("grok-image").edit("a fox", [b"a", b"b"])

        assert sent[0]["url"] == "https://api.x.ai/v1/images/edits"

    async def test_rejects_more_references_than_the_endpoint_takes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Refused rather than truncated: silently dropping a character would
        produce a picture missing someone the page needs, with nothing in the
        artifact explaining why."""
        sent = _capture(monkeypatch)
        with pytest.raises(ImageGenerationError, match="at most 2"):
            await get_image_model("grok-image").edit("x", [b"a", b"b", b"c"])

        assert not sent, "must fail before paying for a call"

    async def test_rejects_no_references(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sent = _capture(monkeypatch)
        with pytest.raises(ImageGenerationError, match="at least one"):
            await get_image_model("grok-image").edit("x", [])

        assert not sent


class TestFailures:
    async def test_http_error_becomes_a_generation_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Retryable, not a configuration problem: a 503 is what retries are for."""
        _capture(monkeypatch, status=503, body={"error": "overloaded"})
        with pytest.raises(ImageGenerationError) as excinfo:
            await get_image_model("grok-image").generate("a fox")

        assert "503" in str(excinfo.value)
        assert not isinstance(excinfo.value, ConfigurationError)

    async def test_passes_through_the_providers_own_message(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """xAI names the offending field, which is more useful than a status line."""
        _capture(
            monkeypatch,
            status=422,
            body={"error": "invalid type: string, expected struct ImageUrl"},
        )
        with pytest.raises(ImageGenerationError, match="ImageUrl"):
            await get_image_model("grok-image").generate("a fox")

    async def test_empty_data_becomes_a_generation_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _capture(monkeypatch, body={"data": []})
        with pytest.raises(ImageGenerationError, match="empty response"):
            await get_image_model("grok-image").generate("a fox")

    async def test_a_url_response_is_refused_not_followed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _capture(monkeypatch, body={"data": [{"url": "https://example.com/a.png"}]})
        with pytest.raises(ImageGenerationError, match="URL"):
            await get_image_model("grok-image").generate("a fox")

    async def test_a_transport_failure_becomes_a_generation_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def boom(self: Any, url: str, **kwargs: Any) -> httpx.Response:
            raise httpx.ConnectTimeout("timed out")

        monkeypatch.setattr(httpx.AsyncClient, "post", boom)
        with pytest.raises(ImageGenerationError, match="could not be reached"):
            await get_image_model("grok-image").generate("a fox")


class TestRetryClassification:
    def test_generation_errors_are_retried(self) -> None:
        """LangGraph's default retry predicate returns True for anything it does
        not recognise, so this must be asserted rather than assumed."""
        from sparkstory.workflows.retries import _retry_on

        assert _retry_on(ImageGenerationError("503")) is True

    def test_configuration_errors_are_not_retried(self) -> None:
        """Trying again cannot make an API key appear."""
        from sparkstory.workflows.retries import _retry_on

        assert _retry_on(ImageConfigurationError("no key")) is False
