"""The narration registry: one entry, xAI, and no model id."""

from sparkstory.config import Settings


def test_narrator_model_defaults_to_an_xai_entry() -> None:
    # Rule 21 and finding CC: `.env` pins every stage to Grok, and a Google
    # default fails open, stores nothing, and leaves the run looking normal.
    settings = Settings()
    assert settings.narrator_model == "grok-speech"
    assert settings.narrator_model in settings.speech_configs


def test_speech_config_carries_a_key_and_a_base_url_and_no_identifier() -> None:
    entry = Settings().speech_configs["grok-speech"]
    assert entry["api_key_env_var"] == "XAI_API_KEY"
    assert entry["params"]["base_url"] == "https://api.x.ai/v1"
    # Measured, not assumed: POST /v1/tts returns 200 with no `model` field, so
    # an identifier here would be a value nothing reads. A *wrong* one fails at
    # the first live call and nowhere earlier -- exactly what a web-search-
    # asserted `grok-2-image` cost the image seam.
    assert "identifier" not in entry


def test_narrator_model_is_settable_by_env_alias() -> None:
    settings = Settings(NARRATOR_MODEL="grok-speech")
    assert settings.narrator_model == "grok-speech"
