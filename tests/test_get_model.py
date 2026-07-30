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
        Settings(  # type: ignore[call-arg]
            google_api_key="test-key",
            xai_api_key="test-key",
            _env_file=None,
        ),
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


class TestRegistryIsWellFormed:
    """Guards every entry, including ones added later.

    The registry is the place a model gets swapped under load, so a malformed
    entry is discovered at the worst possible moment -- mid-outage, while
    switching away from a model that is down.
    """

    def test_every_entry_can_be_built(self) -> None:
        for model_id in Settings(_env_file=None).llm_configs:  # type: ignore[call-arg]
            get_chat_model(model_id)

    def test_no_entry_mixes_thinking_parameters(self) -> None:
        """Gemini 2.5 takes thinking_budget, 3.x takes thinking_level.

        Passing both is an opaque provider-side error, so the pairing is
        load-bearing and easy to get wrong when copying an existing entry.
        """
        for model_id, config in Settings(_env_file=None).llm_configs.items():  # type: ignore[call-arg]
            params = config.get("params", {})
            both = {"thinking_budget", "thinking_level"} <= set(params)
            assert not both, f"{model_id} sets both thinking parameters"

    def test_every_entry_declares_a_resolvable_api_key(self) -> None:
        """An api_key_env_var that api_key_for() does not know always reads as unset."""
        settings = Settings(  # type: ignore[call-arg]
            google_api_key="test-key",
            xai_api_key="test-key",
            _env_file=None,
        )
        for model_id, config in settings.llm_configs.items():
            resolved = settings.api_key_for(config["api_key_env_var"])
            assert resolved == "test-key", (
                f"{model_id} declares {config['api_key_env_var']!r}, which "
                "api_key_for() does not map"
            )
