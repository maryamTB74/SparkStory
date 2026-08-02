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
    # Added when a second provider became necessary rather than in advance
    xai_api_key: SecretStr | None = Field(
        default=None,
        alias="XAI_API_KEY",
        description="xAI key, used for Grok text generation",
    )

    # --- Which model each task uses -------------------------------------
    # One field per agent. Values must be keys of `llm_configs` below.
    planner_model: str = Field(
        default="gemini-3.5-flash",
        alias="PLANNER_MODEL",
        description="Model used by the Story Planner agent",
    )
    plot_model: str = Field(
        default="gemini-3.5-flash",
        alias="PLOT_MODEL",
        description="Model used by the Plot Planner agent",
    )
    # One field per agent, all defaulting to the same registry entry so that we are
    # able to runs nodes at different settings -- its reviewer at temperature 0 and
    # thinking_level high, its writer at medium -- which this registry expresses as
    # separate entries, so raising only the writer's quality stays a config change.
    writer_model: str = Field(
        default="gemini-3.5-flash",
        alias="WRITER_MODEL",
        description="Model used by the Writer agent",
    )
    # Research runs before planning, so this is now the *first* model call in a
    # book. A broken value here fails the cheapest path in the system.
    researcher_model: str = Field(
        default="gemini-3.5-flash",
        alias="RESEARCHER_MODEL",
        description="Model used by the Researcher agent",
    )
    outline_critic_model: str = Field(
        default="gemini-3.5-flash-critic",
        alias="OUTLINE_CRITIC_MODEL",
        description="Model used by the Outline Critic agent",
    )
    prose_critic_model: str = Field(
        default="gemini-3.5-flash-critic",
        alias="PROSE_CRITIC_MODEL",
        description="Model used by the Prose Critic agent",
    )

    # --- Evaluator-optimizer loop budgets --------------------------------
    # Two knobs which separates how many review->edit rounds
    # to run (`num_reviews`) from how many findings one round may return
    # (`max_reviews_per_iteration`). Both values are guesses until we have run
    # data: the number that matters is how often a critic returns empty on the
    # first pass, which is why every iteration is written to the run artifacts.
    # Both loops run N revisions and N+1 critiques, so `0` means "critique once,
    # never revise" rather than "skip the critic". Every draft has to be scored
    # for the loop to keep the best one rather than the last, and a draft that is
    # never critiqued cannot be scored.
    max_outline_revisions: int = Field(
        default=2,
        ge=0,
        alias="MAX_OUTLINE_REVISIONS",
        description=(
            "How many times the outline may be revised from reviews. 0 still "
            "runs one critique."
        ),
    )
    # 0 here means "check but never rewrite", and the safety gate still fails
    # closed: a guardrail on a kids' product must not be switchable off through a
    # knob labelled "revisions".
    max_prose_revisions: int = Field(
        default=2,
        ge=0,
        alias="MAX_PROSE_REVISIONS",
        description=(
            "How many times the prose may be rewritten from reviews. 0 still "
            "runs one critique and still fails closed on a safety finding."
        ),
    )
    max_reviews_per_pass: int = Field(
        default=5,
        ge=1,
        alias="MAX_REVIEWS_PER_PASS",
        description="Cap on findings a critic may return in one pass",
    )

    # --- Research and retrieval ------------------------------------------
    # Names an entry in `embedding_configs`, not `llm_configs`: an embedder takes
    # no messages and binds no output schema, so it cannot be built by
    # `get_chat_model`. Two registries, two factories.
    embedding_model: str = Field(
        default="potion-base-8M",
        alias="EMBEDDING_MODEL",
        description="Model used to embed corpus chunks and search queries",
    )
    # A ceiling, not a target. The task 1 spike answered in a single turn -- both
    # tools called in parallel, then done -- so this has never been approached.
    # `0` skips research altogether, mirroring MAX_*_REVISIONS, which is also
    # what makes a grounded/ungrounded A/B possible with no code change.
    max_research_steps: int = Field(
        default=4,
        ge=0,
        alias="MAX_RESEARCH_STEPS",
        description=(
            "How many reasoning steps the Researcher may take. 0 skips research."
        ),
    )
    retrieval_top_k: int = Field(
        default=5,
        ge=1,
        alias="RETRIEVAL_TOP_K",
        description="Candidates each retrieval tool returns for the agent to judge",
    )
    # Anchored to the repo root rather than left relative, for the same reason
    # `env_file` is: a relative default resolves against the *process* working
    # directory, and an MCP client starts this server from wherever it likes. The
    # failure that would cause is "no index found" while the index plainly
    # exists, which is the least debuggable kind.
    knowledge_root: Path = Field(
        default=_PROJECT_ROOT / "data" / "knowledge",
        alias="KNOWLEDGE_ROOT",
        description="Directory holding the built knowledge index",
    )

    # --- Validators -----------------------------------------------------
    @field_validator("google_api_key", "xai_api_key", mode="before")
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
            # Same model, different temperature -- this is what the two-level
            # registry is for. A critic that returns different findings
            # on identical input turns the empty-review-list stop signal into noise.
            "gemini-3.5-flash-critic": {
                "identifier": "google_genai:gemini-3.5-flash",
                "api_key_env_var": "GOOGLE_API_KEY",
                "params": {
                    "temperature": 0.0,
                    "thinking_level": "low",
                    "include_thoughts": False,
                    "max_retries": 3,
                },
            },
            # --- xAI / Grok -------------------------------------------------
            # A second provider, not merely a second model.
            # Reached through the OpenAI-compatible surface: xAI implements
            # OpenAI's API, so `openai:<model>` plus a base_url needs no new
            # provider integration -- and `langchain-openai` is already a course
            # dependency. `base_url` travels in params like any other provider
            # parameter, so `get_chat_model` needs no change.
            "grok-4": {
                "identifier": "openai:grok-4",
                "api_key_env_var": "XAI_API_KEY",
                "params": {
                    "base_url": "https://api.x.ai/v1",
                    "temperature": 1,
                    "max_retries": 3,
                },
            },
            "grok-3-mini": {
                "identifier": "openai:grok-3-mini",
                "api_key_env_var": "XAI_API_KEY",
                "params": {
                    "base_url": "https://api.x.ai/v1",
                    "temperature": 1,
                    "max_retries": 3,
                },
            },
            # A critic on the same provider as the rest of the pipeline. Not
            # redundant with the Gemini critic entry: when every *_MODEL points
            # at Grok, defaulting the critic to Google makes the loop's one
            # Google call the most likely thing in the run to fail -- and a 503
            # there tells us nothing about whether the critic works.
            # The researcher decides which tool to call, not what to write, so it
            # wants near-determinism -- but not 0.0: an agent choosing among tools
            # benefits from a little slack, and unlike a critic it has no
            # empty-list stop signal that noise could corrupt. Exists as a Grok
            # entry for the same reason the Grok critic does (rule 21): the
            # defaults name Google while a working .env pins everything to Grok,
            # and research is now the *first* call in a book.
            "grok-3-mini-researcher": {
                "identifier": "openai:grok-3-mini",
                "api_key_env_var": "XAI_API_KEY",
                "params": {
                    "base_url": "https://api.x.ai/v1",
                    "temperature": 0.2,
                    "max_retries": 3,
                },
            },
            "grok-3-mini-critic": {
                "identifier": "openai:grok-3-mini",
                "api_key_env_var": "XAI_API_KEY",
                "params": {
                    "base_url": "https://api.x.ai/v1",
                    "temperature": 0.0,
                    "max_retries": 3,
                },
            },
        }

    # --- Embedding registry ---------------------------------------------
    @property
    def embedding_configs(self) -> dict[str, dict[str, Any]]:
        """Registry of available embedding models.

        Deliberately separate from ``llm_configs`` rather than a section within
        it. A chat model is built by ``get_chat_model``, takes messages and binds
        an output schema; an embedder is built by ``get_embedder``, takes strings
        and returns vectors. Sharing one registry would mean one factory had to
        branch on which kind an entry was.

        No ``api_key_env_var``, and that absence is the point: these models run
        locally. Every Google call in this project's history has failed with a
        503, and xAI has no embeddings endpoint at all -- so an embedder that
        needed a credential would put the whole retrieval layer behind the least
        reliable dependency we have.

        ``dimensions`` is recorded because the store writes one ``.npy`` of that
        width. Changing it silently makes an existing index unreadable rather
        than merely different, so it is pinned and tested.
        """
        return {
            # 256-dim static embeddings, distilled so inference is essentially
            # numpy. Measured in the task 1 spike: 3/3 at rank 1 on fact,
            # paraphrase and structural-craft queries, 3.6s to load.
            "potion-base-8M": {
                "identifier": "minishlab/potion-base-8M",
                "dimensions": 256,
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
            "XAI_API_KEY": self.xai_api_key,
        }
        secret = secrets.get(env_var)
        return secret.get_secret_value() if secret is not None else None


settings = Settings()
