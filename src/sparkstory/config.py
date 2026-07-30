"""Central configuration for the SparkStory MCP server.

No other module reads ``os.environ`` directly. Everything funnels through this
file so that every tunable is discoverable in one place, and so tests can
override behaviour by constructing a ``Settings`` instance rather than mutating
the process environment.

Credentials are typed ``SecretStr``, whose ``repr`` renders as ``**********``.
An accidental ``logger.info(settings)`` therefore cannot leak a key into a log
file.

Model configuration is deliberately two-level:

``llm_configs``
    A registry of *models*: name -> identifier, which API key it needs, and
    provider parameters.

``*_model`` fields
    Which registry entry each *task* uses (``planner_model``, and one per agent
    as later sessions add them).

"""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# src/sparkstory/config.py -> parents[2] is the repository root. Depth-sensitive:
# this module moved up one level from config/settings.py, and getting the count
# wrong does not raise -- it silently resolves .env against the wrong directory.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _package_version() -> str:
    """Read the version from installed package metadata.

    Single-sourced from ``pyproject.toml`` rather than repeated here. Hard-coding
    it in both places means the version advertised to MCP clients silently goes
    stale on the first ``uv version --bump``, and nothing fails to reveal it.
    """
    try:
        return version("sparkstory")
    except PackageNotFoundError:
        return "0.0.0+unknown"


class Settings(BaseSettings):
    """Values resolved from environment variables, then ``.env``, then defaults."""

    model_config = SettingsConfigDict(
        # Two candidates, in increasing priority. A bare ".env" resolves against
        # the *process working directory*, which is wrong whenever an MCP client
        # launches this server from somewhere else -- the failure mode is
        # "GOOGLE_API_KEY not set" while staring at a .env that plainly has it.
        # The absolute path anchored to the repo root covers that case.
        env_file=(".env", _PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        # Ignore unrelated variables that happen to be in the shell rather than
        # refusing to start. Without this, any stray env var is a hard error.
        extra="ignore",
        # Every field below declares an explicit `alias` (the env var name).
        # Without populate_by_name, `Settings(google_api_key="x")` would raise
        # and tests would be forced to use the SHOUTING_ALIAS form. This allows
        # both, so tests stay readable.
        populate_by_name=True,
    )

    # --- Server identity ------------------------------------------------
    # Reported to MCP clients during the initialize handshake.
    server_name: str = Field(
        default="SparkStory MCP Server",
        alias="SERVER_NAME",
        description="Server name advertised to MCP clients",
    )
    server_version: str = Field(
        default_factory=_package_version,
        alias="SERVER_VERSION",
        description="Server version advertised to MCP clients",
    )

    # --- Logging --------------------------------------------------------
    # Split levels: ours vs third-party. See utils/logging_utils.py for why.
    log_level: str = Field(
        default="INFO",
        alias="LOG_LEVEL",
        description="Log level for SparkStory's own modules",
    )
    log_level_dependencies: str = Field(
        default="WARNING",
        alias="LOG_LEVEL_DEPENDENCIES",
        description="Log level for noisy third-party libraries (httpx, google, ...)",
    )

    # --- Credentials ----------------------------------------------------
    # Optional at import time on purpose: the server must start and list its
    # tools without keys present. The LLM factory raises a targeted error
    # naming the specific missing key, which is far more useful than a crash
    # during import.
    google_api_key: SecretStr | None = Field(
        default=None,
        alias="GOOGLE_API_KEY",
        description="Google AI Studio key, used for Gemini text generation",
    )

    # --- Which model each task uses -------------------------------------
    # One field per agent. Values must be keys of `llm_configs` below.
    planner_model: str = Field(
        default="gemini-3.5-flash",
        alias="PLANNER_MODEL",
        description="Model used by the Story Planner agent",
    )

    # --- Validators -----------------------------------------------------
    @field_validator("google_api_key", mode="before")
    @classmethod
    def _blank_to_none(cls, value: object) -> object:
        """Treat a blank environment variable as unset.

        ``GOOGLE_API_KEY=`` in a .env file arrives here as ``""``, not as absent.
        Without this, pydantic builds ``SecretStr('')`` -- which is not ``None``,
        so every ``is not None`` check downstream reports the credential as
        configured and we try to authenticate with an empty string.

        Normalising once here means consumers can keep using the natural
        ``is None`` test instead of each remembering to check truthiness. It also
        covers the whitespace case, where a trailing space after ``=`` is
        invisible in an editor.
        """
        if isinstance(value, str) and not value.strip():
            return None
        return value

    # --- Model registry -------------------------------------------------
    @property
    def llm_configs(self) -> dict[str, dict[str, Any]]:
        """Registry of available models.

        ``identifier`` is in ``init_chat_model``'s ``provider:model`` form.

        ``params`` are forwarded to the provider. Note that ``thinking_budget``
        (Gemini 2.5) and ``thinking_level`` (Gemini 3.x) are mutually
        exclusive; the LLM factory rejects a config carrying both.

        ``max_retries`` is handled by LangChain rather than a wrapper of our
        own -- one less moving part to maintain.
        """
        return {
            "gemini-2.5-pro": {
                "identifier": "google_genai:gemini-2.5-pro",
                "api_key_env_var": "GOOGLE_API_KEY",
                "params": {
                    "temperature": 0.7,
                    "thinking_budget": 1000,
                    "include_thoughts": False,
                    "max_retries": 3,
                },
            },
            "gemini-3.5-flash": {
                "identifier": "google_genai:gemini-3.5-flash",
                "api_key_env_var": "GOOGLE_API_KEY",
                "params": {
                    "temperature": 1,
                    "thinking_level": "low",
                    "include_thoughts": False,
                    "max_retries": 3,
                },
            },
        }

    def api_key_for(self, env_var: str) -> str | None:
        """Resolve the plaintext API key for a registry entry's ``api_key_env_var``.

        A mapping rather than an if/elif chain: adding a provider edits one
        dictionary entry instead of growing a branch for every key.

        Returns ``None`` when the key is unset, leaving it to the caller to
        raise an error naming the model that needed it.
        """
        secrets: dict[str, SecretStr | None] = {
            "GOOGLE_API_KEY": self.google_api_key,
        }
        secret = secrets.get(env_var)
        return secret.get_secret_value() if secret is not None else None


settings = Settings()
