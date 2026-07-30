"""The LLM factory itself.

Written because this seam had no direct coverage: its three failure modes were
only ever exercised through stubs that *raised* the errors, which asserts that
the tool layer translates them, not that the factory ever produces them. A typo
in a registry key or a renamed env var would have passed the suite.

No network: every test here fails before a request would be made.
"""

import pytest

from sparkstory.config import Settings
from sparkstory.entities.exceptions import ConfigurationError
from sparkstory.models.exceptions import MissingAPIKeyError, UnknownModelError
from sparkstory.models.get_model import get_chat_model


@pytest.fixture(autouse=True)
def _explicit_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the factory at a Settings built here, not at the machine's .env.

    Without this, whether a key is configured decides whether these tests pass.
    """
    monkeypatch.setattr(
        "sparkstory.models.get_model.settings",
        Settings(google_api_key="test-key", _env_file=None),  # type: ignore[call-arg]
    )


class TestUnknownModel:
    def test_raises_naming_the_known_ids(self) -> None:
        """The message must list valid ids, because this is almost always a typo."""
        with pytest.raises(UnknownModelError) as excinfo:
            get_chat_model("gemini-3.5-flsah")

        message = str(excinfo.value)
        assert "gemini-3.5-flash" in message
        assert "*_MODEL" in message, "must point the operator at the setting to fix"


class TestMissingAPIKey:
    def test_raises_naming_the_variable_to_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "sparkstory.models.get_model.settings",
            Settings(google_api_key=None, _env_file=None),  # type: ignore[call-arg]
        )
        with pytest.raises(MissingAPIKeyError, match="GOOGLE_API_KEY"):
            get_chat_model("gemini-3.5-flash")

    def test_a_blank_key_counts_as_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression: `GOOGLE_API_KEY=` arrives as "", not as absent."""
        monkeypatch.setattr(
            "sparkstory.models.get_model.settings",
            Settings(google_api_key="   ", _env_file=None),  # type: ignore[call-arg]
        )
        with pytest.raises(MissingAPIKeyError):
            get_chat_model("gemini-3.5-flash")


class TestExclusiveThinkingParams:
    def test_rejects_a_registry_entry_carrying_both(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Passing both is an opaque provider-side error; name the model instead."""

        class BadSettings(Settings):
            @property
            def llm_configs(self) -> dict[str, dict[str, object]]:
                return {
                    "confused": {
                        "identifier": "google_genai:gemini-3.5-flash",
                        "api_key_env_var": "GOOGLE_API_KEY",
                        "params": {"thinking_budget": 1000, "thinking_level": "low"},
                    }
                }

        monkeypatch.setattr(
            "sparkstory.models.get_model.settings",
            BadSettings(google_api_key="test-key", _env_file=None),  # type: ignore[call-arg]
        )
        with pytest.raises(ConfigurationError, match="mutually exclusive"):
            get_chat_model("confused")


class TestSuccessfulBuild:
    def test_returns_an_unbound_runnable(self) -> None:
        """Structured output is the node's job, so nothing is bound here."""
        model = get_chat_model("gemini-3.5-flash")
        assert hasattr(model, "ainvoke")
        assert hasattr(model, "with_structured_output")
