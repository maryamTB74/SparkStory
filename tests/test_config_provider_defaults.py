"""The *_MODEL defaults follow whichever API key is actually present.

Every ``*_model`` field names a Google entry by default, because Google is the
project's stated default provider. But a ``.env`` holding only ``XAI_API_KEY``
used to leave all eleven pointing at models it could not authenticate, and the
failure was not uniform: most stages raise ``MissingAPIKeyError`` naming the
variable, while the memory extractor *fails open* and stores nothing with the
run otherwise looking completely normal.

So the defaults resolve against the keys present. These tests pin the four
key combinations and the one case that must survive the rewrite: a value the
operator set explicitly.
"""

import pytest

from sparkstory.config import Settings

# Every field the resolver may rewrite, paired with the entry it must hold on
# each provider. Written out rather than derived from the mapping under test:
# a table generated from the code it checks agrees with that code by
# construction and cannot fail.
_EXPECTED: dict[str, tuple[str, str]] = {
    "planner_model": ("gemini-3.5-flash", "grok-3-mini"),
    "plot_model": ("gemini-3.5-flash", "grok-3-mini"),
    "writer_model": ("gemini-3.5-flash", "grok-3-mini"),
    "researcher_model": ("gemini-3.5-flash", "grok-3-mini-researcher"),
    "memory_extractor_model": ("gemini-3.5-flash", "grok-3-mini"),
    "illustration_director_model": ("gemini-3.5-flash", "grok-3-mini"),
    "outline_critic_model": ("gemini-3.5-flash-critic", "grok-3-mini-critic"),
    "prose_critic_model": ("gemini-3.5-flash-critic", "grok-3-mini-critic"),
    "judge_model": ("gemini-3.5-flash-critic", "grok-3-mini-critic"),
    "reranker_model": ("gemini-3.5-flash-critic", "grok-3-mini-critic"),
    "consistency_judge_model": ("gemini-3.5-flash-critic", "grok-3-mini-critic"),
}


def _settings(**overrides: object) -> Settings:
    """Build Settings from explicit values only.

    ``_env_file=None`` matters: without it pydantic-settings reads the
    developer's real ``.env``, so a machine with both keys set would pass the
    xAI-only test for the wrong reason.
    """
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


@pytest.mark.parametrize("field", sorted(_EXPECTED))
def test_xai_key_alone_selects_the_grok_entry(field: str) -> None:
    settings = _settings(xai_api_key="xai-test")
    assert getattr(settings, field) == _EXPECTED[field][1]


@pytest.mark.parametrize("field", sorted(_EXPECTED))
def test_google_key_alone_keeps_the_gemini_entry(field: str) -> None:
    settings = _settings(google_api_key="google-test")
    assert getattr(settings, field) == _EXPECTED[field][0]


@pytest.mark.parametrize("field", sorted(_EXPECTED))
def test_both_keys_prefer_google(field: str) -> None:
    """Google wins a tie, so adding an xAI key never moves an existing setup."""
    settings = _settings(google_api_key="google-test", xai_api_key="xai-test")
    assert getattr(settings, field) == _EXPECTED[field][0]


@pytest.mark.parametrize("field", sorted(_EXPECTED))
def test_no_keys_keep_the_declared_defaults(field: str) -> None:
    """With nothing to resolve against, the declared default stands.

    The server must still start and list its tools with no credentials at all,
    so this cannot raise -- the targeted error belongs at the call site, which
    names the variable that is missing.
    """
    settings = _settings()
    assert getattr(settings, field) == _EXPECTED[field][0]


def test_an_explicit_value_survives_the_rewrite() -> None:
    """Only fields still holding their default are rewritten."""
    settings = _settings(xai_api_key="xai-test", writer_model="grok-4")
    assert settings.writer_model == "grok-4"
    assert settings.plot_model == "grok-3-mini"


def test_an_explicit_gemini_value_is_left_alone_under_an_xai_key() -> None:
    """A deliberate cross-provider choice is honoured, not corrected.

    Mixing providers is legitimate -- a Gemini critic beside a Grok writer is a
    real configuration -- so an operator naming an entry gets that entry even
    when its key is absent. The resulting failure names the variable, which is
    better than silently substituting a model they did not ask for.
    """
    settings = _settings(xai_api_key="xai-test", judge_model="gemini-2.5-pro")
    assert settings.judge_model == "gemini-2.5-pro"


@pytest.mark.parametrize("field", sorted(_EXPECTED))
def test_every_resolved_value_is_a_real_registry_entry(field: str) -> None:
    """A rewritten name must be constructible.

    The resolver writes bare strings, so a typo in the mapping would surface
    only at the first live call -- after a book had been paid for up to that
    stage.
    """
    settings = _settings(xai_api_key="xai-test")
    assert getattr(settings, field) in settings.llm_configs


@pytest.mark.parametrize("field", sorted(_EXPECTED))
def test_every_resolved_value_needs_the_key_that_selected_it(field: str) -> None:
    """The point of the resolver: the chosen entry uses the key we have."""
    settings = _settings(xai_api_key="xai-test")
    entry = settings.llm_configs[getattr(settings, field)]
    assert entry["api_key_env_var"] == "XAI_API_KEY"


def test_media_models_are_not_rewritten() -> None:
    """Images and speech have no Google entry, so there is nothing to resolve.

    Pinned so a later session widening the resolver has to face the question
    rather than discover it: adding a Gemini image entry without adding it here
    would leave illustration silently on xAI.
    """
    settings = _settings(xai_api_key="xai-test")
    assert settings.illustrator_model == "grok-image"
    assert settings.narrator_model == "grok-speech"


def test_the_embedder_is_not_rewritten() -> None:
    """Deliberately excluded, and this test is the record of why.

    ``db/models.py`` gives each embedder its own table, so rewriting this would
    repoint retrieval at a *different index* -- one that may hold nothing. A run
    whose research returns no facts still plans a complete, plausible book, so
    the failure would be invisible. An error naming GOOGLE_API_KEY is better.
    """
    settings = _settings(xai_api_key="xai-test")
    assert settings.embedding_model == "gemini-embedding"
