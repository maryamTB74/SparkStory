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

from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

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
            # `embedding_model` also ends in `_model` but names an entry in a
            # different registry: an embedder takes no messages and binds no
            # output schema, so it cannot live in `llm_configs`. Routed by name
            # rather than skipped, so it stays covered.
            registry = (
                settings.embedding_configs
                if name == "embedding_model"
                else settings.llm_configs
            )
            assert value in registry, f"{name}={value!r} is not in its registry"

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


class TestEmbeddingRegistry:
    """The second registry. Mirrors `llm_configs`, minus the credential."""

    def test_the_default_embedder_exists(self) -> None:
        fresh = Settings()
        assert fresh.embedding_model == "potion-base-8M"
        assert fresh.embedding_model in fresh.embedding_configs

    def test_every_entry_is_well_formed(self) -> None:
        for name, cfg in Settings().embedding_configs.items():
            assert cfg.get("identifier"), f"{name} missing identifier"
            assert cfg.get("dimensions"), f"{name} missing dimensions"

    def test_no_entry_needs_an_api_key(self) -> None:
        """A local model needs no credential, and that is the whole reason it was
        chosen: every Google call in this project's history has failed, and xAI
        has no embeddings endpoint. An entry that grew an `api_key_env_var` would
        mean that property had quietly been given up."""
        for name, cfg in Settings().embedding_configs.items():
            assert "api_key_env_var" not in cfg, f"{name} requires a key"

    def test_dimensions_match_the_model_that_was_measured(self) -> None:
        """Pinned from the task 1 spike, which measured 256. The store writes
        one .npy of this width, so a silent change would make an existing index
        unreadable rather than merely different."""
        assert Settings().embedding_configs["potion-base-8M"]["dimensions"] == 256


class TestResearchSettings:
    def test_defaults(self) -> None:
        fresh = Settings()
        assert fresh.max_research_steps == 4
        assert fresh.retrieval_top_k == 5
        assert fresh.knowledge_root.parts[-2:] == ("data", "knowledge")

    def test_the_default_index_path_is_absolute(self) -> None:
        """Non-obvious rules 4 and 6, in a new place. A relative default resolves
        against the *process* working directory, and an MCP client launches this
        server from wherever it likes -- so the failure would be "no index found"
        while data/knowledge plainly has one. Anchored to the repo root instead."""
        assert Settings().knowledge_root.is_absolute()

    def test_an_explicit_root_is_respected(self) -> None:
        """Whatever an operator sets wins, absolute or not: the anchoring is a
        default, not a policy."""
        assert Settings(knowledge_root=Path("/tmp/elsewhere")).knowledge_root == Path(
            "/tmp/elsewhere"
        )

    def test_research_can_be_switched_off_entirely(self) -> None:
        """0 skips research rather than running it with no budget, mirroring
        MAX_*_REVISIONS. This is also what makes the A/B acceptance test possible
        with no code change: one run grounded, one not, same premise."""
        assert Settings(max_research_steps=0).max_research_steps == 0


class TestWebSearchSettings:
    """The web tool is off unless asked for, and verified unless asked otherwise.

    Both defaults are load-bearing rather than tidy, and each is here because
    getting it wrong would be invisible.
    """

    def test_web_search_is_off_by_default(self) -> None:
        """This default is what keeps the suite offline.

        At 0 no client is constructed and no key is read, so the whole test suite
        keeps the no-network property it has held since Session 1. A default of
        anything else would make every test that reaches research a live call --
        which non-obvious rule 25 already caught once, when a test that faked only
        `get_chat_model` reached a real provider and still looked like it passed.

        Asserted against the **field default**, not against `Settings()`. A bare
        `Settings()` reads `.env`, so this test used to assert "*this machine's*
        .env does not enable web search" -- which is ambient, not a property of
        the code, and it duly broke the moment a real `.env` set MAX_WEB_SEARCHES.
        The schema default is the thing worth pinning.
        """
        assert Settings.model_fields["max_web_searches"].default == 0

    def test_web_search_accepts_a_budget(self) -> None:
        assert Settings(max_web_searches=3).max_web_searches == 3

    def test_web_search_rejects_a_negative_budget(self) -> None:
        with pytest.raises(ValidationError):
            Settings(max_web_searches=-1)

    def test_claims_are_verified_by_default(self) -> None:
        """The only check on a model-asserted URL, so the default must be safe.

        The search provider returns a URL the *model* wrote into a structured
        field -- exactly the shape of fabrication that made this project overwrite
        `source` from the store rather than trust it. Fetching the page is what
        turns that assertion into provenance, so `False` exists for tests and must
        never be what a caller gets by accident.
        """
        assert Settings.model_fields["verify_web_claims"].default is True

    def test_verification_can_be_disabled(self) -> None:
        assert Settings(verify_web_claims=False).verify_web_claims is False

    @pytest.mark.parametrize("field", ["perplexity_api_key", "firecrawl_api_key"])
    def test_blank_web_keys_normalise_to_none(self, field: str) -> None:
        """Non-obvious rule 3, for each new key separately.

        `FIRECRAWL_API_KEY=` arrives as `""`, not as absent, and pydantic would
        build `SecretStr('')` -- which is not None, so every `is not None` check
        downstream reports the credential as configured. Adding a field to the
        `_blank_to_none` validator is the edit most easily forgotten, so there is
        one test per key rather than one covering both.
        """
        assert getattr(Settings(**{field: "   "}), field) is None

    @pytest.mark.parametrize(
        ("env_var", "field"),
        [
            ("PERPLEXITY_API_KEY", "perplexity_api_key"),
            ("FIRECRAWL_API_KEY", "firecrawl_api_key"),
        ],
    )
    def test_api_key_for_resolves_the_web_keys(self, env_var: str, field: str) -> None:
        assert Settings(**{field: "k"}).api_key_for(env_var) == "k"

    def test_a_researcher_entry_exists_for_the_provider_in_use(self) -> None:
        """Same reasoning as the critic entries: the defaults point at Google
        while a working .env pins everything to Grok, and a stage that needs a
        provider nothing else uses is the likeliest thing in a run to fail."""
        configs = Settings().llm_configs
        researcher_keys = {
            cfg["api_key_env_var"]
            for name, cfg in configs.items()
            if name.endswith("-researcher")
        }
        assert "XAI_API_KEY" in researcher_keys

    def test_every_researcher_entry_reuses_a_base_entry(self) -> None:
        configs = Settings().llm_configs
        for name, cfg in configs.items():
            if not name.endswith("-researcher"):
                continue
            base = name.removesuffix("-researcher")
            assert base in configs, f"{name} has no base entry {base!r}"
            assert cfg["identifier"] == configs[base]["identifier"]
