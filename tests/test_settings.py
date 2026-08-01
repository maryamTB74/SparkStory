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
    def test_every_configured_model_exists(self) -> None:
        """Guards against a typo in any ``*_MODEL`` reaching runtime.

        Deliberately asserts against the live settings object, since the failure
        this catches is a misconfigured .env, not a code defect. Discovered from
        the fields rather than listed, so a new agent's model setting is covered
        the moment it is declared.
        """
        configured = {
            name: getattr(settings, name)
            for name in type(settings).model_fields
            if name.endswith("_model")
        }
        assert configured, "no *_model settings found -- the discovery broke"
        for name, value in configured.items():
            assert value in settings.llm_configs, (
                f"{name}={value!r} is not a key of llm_configs"
            )

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


class TestCriticSettings:
    def test_every_critic_entry_is_deterministic(self) -> None:
        """A critic returning different findings on identical input turns the
        empty-review-list stop signal into noise.

        Discovered by suffix rather than listed, so a third critic entry cannot
        drift in at a non-zero temperature.
        """
        critics = {
            name: cfg
            for name, cfg in Settings().llm_configs.items()
            if name.endswith("-critic")
        }
        assert critics, "no *-critic entries found -- the discovery broke"
        for name, cfg in critics.items():
            assert cfg["params"]["temperature"] == 0.0, f"{name} is not deterministic"

    def test_every_critic_entry_reuses_a_base_entry(self) -> None:
        """A registry entry, not a new model: `<base>-critic` must name the same
        identifier as `<base>`. The two-level registry exists so that per-node
        parameters cost one entry and no code."""
        configs = Settings().llm_configs
        for name, cfg in configs.items():
            if not name.endswith("-critic"):
                continue
            base = name.removesuffix("-critic")
            assert base in configs, f"{name} has no base entry {base!r}"
            assert cfg["identifier"] == configs[base]["identifier"]

    def test_a_critic_exists_for_each_provider_in_use(self) -> None:
        """Defaulting the critic to a provider the rest of the pipeline does not
        use makes the loop's one call the most likely thing in a run to fail."""
        configs = Settings().llm_configs
        critic_keys = {
            cfg["api_key_env_var"]
            for name, cfg in configs.items()
            if name.endswith("-critic")
        }
        assert {"GOOGLE_API_KEY", "XAI_API_KEY"} <= critic_keys

    def test_loop_budget_defaults(self) -> None:
        """Both are guesses until we have run data; pinned so a change is
        deliberate rather than drift."""
        fresh = Settings()
        assert fresh.max_outline_revisions == 2
        assert fresh.max_reviews_per_pass == 5

    def test_either_loop_may_be_set_to_zero_revisions(self) -> None:
        """Both loops run N revisions and N+1 critiques, so 0 means "critique
        once, never revise" for either. Not "skip the critic": every draft has
        to be scored for the loop to keep the best one, and for prose a
        guardrail must not be switchable off through a knob labelled
        "revisions"."""
        fresh = Settings(max_outline_revisions=0, max_prose_revisions=0)
        assert fresh.max_outline_revisions == 0
        assert fresh.max_prose_revisions == 0
