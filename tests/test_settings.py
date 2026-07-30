"""Configuration behaviour.

The blank-environment-variable tests are regression tests for a real bug: with
``GOOGLE_API_KEY=`` in a .env file, pydantic received ``""`` rather than nothing
and built ``SecretStr('')``. That is not ``None``, so any ``is not None`` check
downstream reported the credential as configured, and we would have attempted to
authenticate with an empty string instead of raising a clear error.

Worth noting how the bug hid: the defect was in code that had been reviewed, but
was unreachable until a .env file declared the variable blank. .env files are
gitignored, so the trigger was never in the repository.
"""

import pytest
from pydantic import SecretStr

from sparkstory.config import Settings, settings


class TestBlankCredentialsBecomeNone:
    @pytest.mark.parametrize("blank", ["", "   ", "\t", "\n"])
    def test_blank_and_whitespace_normalise_to_none(self, blank: str) -> None:
        assert Settings(google_api_key=blank).google_api_key is None

    def test_real_value_is_preserved(self) -> None:
        s = Settings(google_api_key="a-real-key")
        assert isinstance(s.google_api_key, SecretStr)
        assert s.google_api_key.get_secret_value() == "a-real-key"

    def test_blank_key_is_reported_as_absent_not_empty(self) -> None:
        """The resolver must return None, never ``""``.

        An empty string is falsy, so the factory's ``if not api_key`` guard would
        still fire -- but returning ``""`` would silently break any future caller
        that checks ``is not None``.
        """
        assert Settings(google_api_key="").api_key_for("GOOGLE_API_KEY") is None


class TestSecretsAreMasked:
    def test_repr_does_not_leak(self) -> None:
        s = Settings(google_api_key="super-secret-value")
        assert "super-secret-value" not in repr(s.google_api_key)
        assert "super-secret-value" not in repr(s)
        assert "super-secret-value" not in str(s.google_api_key)


class TestApiKeyResolution:
    def test_resolves_known_env_var(self) -> None:
        s = Settings(google_api_key="g-key")
        assert s.api_key_for("GOOGLE_API_KEY") == "g-key"

    def test_unset_key_returns_none(self) -> None:
        assert Settings(google_api_key=None).api_key_for("GOOGLE_API_KEY") is None

    def test_unknown_env_var_returns_none(self) -> None:
        assert Settings().api_key_for("NOT_A_REAL_KEY") is None


class TestModelRegistry:
    def test_configured_planner_model_exists(self) -> None:
        """Guards against a typo in PLANNER_MODEL reaching runtime.

        Deliberately asserts against the live settings object, since the failure
        this catches is a misconfigured .env, not a code defect.
        """
        assert settings.planner_model in settings.llm_configs

    def test_every_entry_is_well_formed(self) -> None:
        for name, cfg in settings.llm_configs.items():
            assert "identifier" in cfg, f"{name} missing identifier"
            assert ":" in cfg["identifier"], f"{name} identifier needs provider prefix"
            assert cfg["api_key_env_var"], f"{name} missing api_key_env_var"

    def test_no_entry_mixes_exclusive_thinking_params(self) -> None:
        """thinking_budget is Gemini 2.5; thinking_level is Gemini 3.x."""
        for name, cfg in settings.llm_configs.items():
            params = cfg.get("params", {})
            both = "thinking_budget" in params and "thinking_level" in params
            assert not both, f"{name} sets both thinking params"
