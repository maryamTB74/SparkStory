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
from typing import Any, ClassVar

from pydantic import Field, SecretStr, field_validator, model_validator
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

    # --- HTTP transport -------------------------------------------------
    server_host: str = Field(
        default="127.0.0.1",
        alias="SERVER_HOST",
        description="Interface the HTTP transport binds to",
    )
    # `int` rather than `str` so pydantic rejects SERVER_PORT=abc at startup with
    # a field error naming the variable, instead of it surfacing as a TypeError
    # from inside the ASGI server.
    server_port: int = Field(
        default=8000,
        alias="SERVER_PORT",
        description="Port the HTTP transport listens on",
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
    # Both are read only when max_web_searches > 0, so an installation that never
    # raises that setting needs neither.
    perplexity_api_key: SecretStr | None = Field(
        default=None,
        alias="PERPLEXITY_API_KEY",
        description="Perplexity key, used for web search",
    )
    firecrawl_api_key: SecretStr | None = Field(
        default=None,
        alias="FIRECRAWL_API_KEY",
        description="Firecrawl key, used to fetch a cited page for checking",
    )
    # A fallback, used only when one of the two above is unusable -- never
    # because a search legitimately found nothing. Retrying an empty result on a
    # second provider is pressure to invent. The same lever has misfired once
    # already in the other direction: an instruction meant to stop invention
    # stopped grounding instead, and returned zero facts on a premise that
    # plainly had some.
    tavily_api_key: SecretStr | None = Field(
        default=None,
        alias="TAVILY_API_KEY",
        description="Tavily key, used as a fallback when the others fail",
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
    # Read only by the offline eval harness, never during a story run: a
    # measurement that could change a book would not be a measurement of it. On a
    # critic entry at temperature 0.0 for the same reason the critics are -- a
    # judge that answers differently on identical input turns a regression signal
    # into noise.
    judge_model: str = Field(
        default="gemini-3.5-flash-critic",
        alias="JUDGE_MODEL",
        description="Model used by the offline book judge",
    )
    # Reorders retrieval candidates before the Researcher sees them. Defaults to a
    # zero-temperature entry deliberately: a reranker that answers differently on
    # identical input converts retrieval into a stage nobody can reproduce, which
    # is what makes a movement in hit-rate readable at all. The provider moved to
    # Google with the resolver below; the temperature-0 requirement is what
    # matters and both critic entries satisfy it.
    # Whether temperature 0 is *enough* is measured rather than assumed; see the
    # repeatability check in tests/test_retrieval_eval.py.
    reranker_model: str = Field(
        default="gemini-3.5-flash-critic",
        alias="RERANKER_MODEL",
        description="Model that reorders retrieval candidates",
    )
    # Runs after a book is finished and delivered, so it is the one model call in
    # the system whose failure cannot cost a parent their story -- the write path
    # fails open. That makes the provider default worth stating: it is Google while
    # a .env pinning everything else to Grok would leave this stage the odd one out
    # with no key, and because the write path fails open it would store nothing
    # while the run looked completely normal. Set it with the other *_MODEL vars.
    memory_extractor_model: str = Field(
        default="gemini-3.5-flash",
        alias="MEMORY_EXTRACTOR_MODEL",
        description="Model that reads a finished book and says what to remember",
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
    # Defaults to the hosted model from 2026-08-19, at Maryam's request. The
    # local `potion-base-8M` entry remains in the registry, so reverting is one
    # env var and no re-ingest -- each embedder owns its own table.
    #
    # Note what this default costs, because it is the opposite of every other
    # `*_model` default here: retrieval now needs a network call and a credential
    # on the cheapest path in the system, and open item 3 records that Google has
    # historically 503'd in this project. `EMBEDDING_MODEL=potion-base-8M` is the
    # one-line recovery.
    embedding_model: str = Field(
        default="gemini-embedding",
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

    # --- Illustration ----------------------------------------------------
    # Names an entry in `image_configs`, not `llm_configs`: an image model takes a
    # prompt and returns bytes, so it cannot be built by `get_chat_model`. Three
    # registries, three factories.
    illustrator_model: str = Field(
        default="grok-image",
        alias="ILLUSTRATOR_MODEL",
        description="Model used to draw reference portraits and page pictures",
    )
    # The *planning* half runs on a chat model, because deciding how a book looks
    # is a writing task. Named separately from `illustrator_model` so the cheap
    # decision and the expensive drawing can be moved independently.
    #
    # This read `grok-3-mini` until the provider resolver below existed, because
    # three stages had died on a Google default while .env pinned everything else
    # to xAI. The resolver removes that hazard at its source -- an xAI-only .env
    # now rewrites this field along with the other ten -- so the declared default
    # can state the project's actual preference instead of hedging against a
    # misconfiguration.
    illustration_director_model: str = Field(
        default="gemini-3.5-flash",
        alias="ILLUSTRATION_DIRECTOR_MODEL",
        description="Model that decides the style bible and each page's picture",
    )
    # A chat model that can accept an image, so it names an `llm_configs` entry.
    # Defaults to the *critic* variant at temperature 0.0 rather than a new
    # registry entry: a judge that answers differently on identical input turns a
    # verdict into noise, which is the same reason the two prose critics have
    # zero-temperature entries of their own.
    #
    # **The measured evidence here is about xAI, and it no longer picks the
    # default.** A spike put `grok-4` and `grok-3-mini` in front of a green ant
    # described as black; both reported green, so the small model was chosen --
    # judging a picture does not need an expensive model. That result still holds
    # and is why the resolver below sends this to `grok-3-mini-critic` rather than
    # `grok-4` on an xAI-only setup.
    #
    # What it does NOT cover is Gemini: no Google model has ever judged an image
    # in this project. So the Google path here is the one stage whose provider is
    # asserted rather than measured, and the same green-ant check should be run
    # against it before a Gemini verdict is trusted.
    consistency_judge_model: str = Field(
        default="gemini-3.5-flash-critic",
        alias="CONSISTENCY_JUDGE_MODEL",
        description="Model that checks a picture against the reference it should match",
    )
    # Gates the *page* half only. Checking each portrait against its own written
    # description is 2 calls per book and catches a reference that was wrong before
    # any page was drawn, so it always runs; judging every page is one call per page
    # and is the half whose false-positive rate has not been measured yet.
    judge_pages: bool = Field(
        default=True,
        alias="JUDGE_PAGES",
        description="Judge each finished page against its reference portraits",
    )
    # --- Tool surface ----------------------------------------------------
    # Whether the two media tools are registered at all. A client cannot call
    # what it cannot see, so this is a *surface* decision rather than a runtime
    # check inside the tool.
    #
    # **"No config for features that do not exist" was asked and answered rather
    # than skipped.** `IMAGE_GENERATION_ENABLED` and `AUDIO_GENERATION_ENABLED`
    # were removed by name once, for gating features that did not exist -- a flag
    # cannot be meaningfully
    # written before the thing it gates. Both features now exist and are verified
    # live, and the driver is concrete: a deployed server where a client must not
    # be *able* to spend money on images, plus a reduced tool surface for a
    # served instance. That is the condition the original removal was waiting on.
    #
    # **Default True, unlike `max_web_searches`, and the difference is the
    # trigger.** Web search defaults to 0 because it reaches the network during
    # research, on a path the caller never asked for. These two run only when a
    # client calls them by name, so defaulting them off would hide working
    # features from every existing installation to serve a deployment that does
    # not exist yet. Set them false on the instance that needs it.
    illustration_enabled: bool = Field(
        default=True,
        alias="ILLUSTRATION_ENABLED",
        description="Offer the illustrate_story tool to clients",
    )
    narration_enabled: bool = Field(
        default=True,
        alias="NARRATION_ENABLED",
        description="Offer the narrate_story tool to clients",
    )
    # --- Narration -------------------------------------------------------
    # Names an entry in `speech_configs`, not `llm_configs` or `image_configs`: a
    # speech model takes text and a voice and returns audio bytes. Four
    # registries, four factories.
    #
    # Defaults to xAI, and the bill for not doing so is already recorded. Four
    # stages have now defaulted to Google while `.env` pinned everything else to
    # Grok, and the memory extractor's version of this failed *open* -- storing
    # nothing while the run looked completely normal. There is no narration
    # equivalent of failing open, but there is no reason to find out.
    #
    # There is deliberately no `narration_voice` beside it: the voice belongs to
    # the brief, because a parent chooses it per story and it crosses the MCP tool
    # boundary. See `StoryBrief.voice`.
    narrator_model: str = Field(
        default="grok-speech",
        alias="NARRATOR_MODEL",
        description="Model used to read the finished story aloud",
    )
    # There is deliberately no `max_images_per_book`. One was written and removed:
    # the image count is *derived*, not chosen -- one picture per page plus one
    # portrait per character -- and `StoryBrief` already caps pages at 24 while
    # `IllustrationPlan` caps characters at 6, so no valid brief can exceed 30. A
    # setting defaulted above a bound the schema already enforces can never fire,
    # and lowering it would reject a valid brief with "illustrate a shorter book"
    # after the book is already written. There is no point configuring a limit that
    # cannot bind; the structural half became `validate_illustration_plan`.
    # Illustration is turned off by not calling it -- the separate tool *is* the
    # switch.
    #
    # Where the corpus and its vectors live.
    #
    # Optional at import time, exactly like the API keys and for the same reason:
    # the MCP server must start and list its tools without a database reachable.
    # The failure belongs at the point of use, naming the variable, rather than at
    # import where it would stop a `--help` from working.
    #
    # The driver is spelled explicitly. A bare `postgresql://` URL makes
    # SQLAlchemy reach for psycopg2, which is not installed here, and the error
    # names a package nobody asked for. `postgresql+psycopg://` pins v3.
    database_url: str | None = Field(
        default=None,
        alias="DATABASE_URL",
        description="postgresql+psycopg://user:pass@host:port/db",
    )
    # Off by default, and that default is what keeps the test suite offline: at 0
    # no web client is constructed and no key is read. Mirrors max_research_steps,
    # where 0 also means "skip the stage" rather than "run it with no budget".
    max_web_searches: int = Field(
        default=0,
        ge=0,
        alias="MAX_WEB_SEARCHES",
        description=(
            "How many web searches research may run. 0 disables the web tool."
        ),
    )
    # The only thing standing between a model-asserted URL and the book. The
    # search provider hands back a URL the *model* wrote into a structured field,
    # which is the same shape of fabrication that made this project overwrite
    # `source` from the store rather than trust it; fetching the page is what
    # turns that assertion into provenance. False exists so a test can switch the
    # network off, so the default has to be the safe one.
    verify_web_claims: bool = Field(
        default=True,
        alias="VERIFY_WEB_CLAIMS",
        description="Fetch each cited page and confirm it supports the claim.",
    )

    # --- Opik (observability) --------------------------------------------
    # Off by default, and that default is what earns the other three fields their
    # place: they gate code that exists rather than a hypothetical.
    opik_enabled: bool = Field(
        default=False,
        alias="OPIK_ENABLED",
        description="Send traces to Opik. Off by default",
    )
    opik_api_key: SecretStr | None = Field(
        default=None,
        alias="OPIK_API_KEY",
        description="Opik key, read only when OPIK_ENABLED is true",
    )
    opik_workspace: str | None = Field(
        default=None,
        alias="OPIK_WORKSPACE",
        description="Opik workspace holding the project",
    )
    opik_project_name: str = Field(
        default="sparkstory",
        alias="OPIK_PROJECT_NAME",
        description="Opik project name traces are grouped under",
    )

    # --- Validators -----------------------------------------------------
    @field_validator(
        "google_api_key",
        "xai_api_key",
        "perplexity_api_key",
        "firecrawl_api_key",
        "tavily_api_key",
        "opik_api_key",
        mode="before",
    )
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

    #: Which registry entry replaces each Google default when only an xAI key is
    #: present. Written as data rather than as branches so that adding a stage is
    #: a field plus, at most, nothing: a new `*_model` defaulting to one of these
    #: two entries is resolved without touching this mapping at all.
    #:
    #: The pairing is by *temperature*, not by name. A critic, a judge and a
    #: reranker all need an entry at 0.0, because one that answers differently on
    #: identical input turns an empty-review stop signal into noise and a
    #: regression signal into variance.
    _GROK_EQUIVALENT: ClassVar[dict[str, str]] = {
        "gemini-3.5-flash": "grok-3-mini",
        "gemini-3.5-flash-critic": "grok-3-mini-critic",
    }

    #: Fields whose xAI equivalent is not the one the mapping above would give.
    #: The Researcher is the only one: `grok-3-mini-researcher` is the same model
    #: at temperature 0.2, which is what every live research run has actually used
    #: on the command line since Session 5.
    _GROK_OVERRIDE: ClassVar[dict[str, str]] = {
        "researcher_model": "grok-3-mini-researcher",
    }

    @model_validator(mode="after")
    def _resolve_models_to_an_available_provider(self) -> Settings:
        """Point every unset ``*_model`` field at a provider we hold a key for.

        Google is the declared default for all eleven chat stages. A .env holding
        only ``XAI_API_KEY`` therefore used to leave every one of them naming a
        model it could not authenticate, and the failures were not uniform: most
        stages raise ``MissingAPIKeyError`` naming the variable, but the memory
        extractor fails *open* -- it stores nothing, logs an exception nobody
        reads, and the run produces a book and looks completely normal. That cost
        a live session, and the fix of "remember to set all of them together" had
        already failed more than once.

        **Only a field still holding its declared default is rewritten.** An
        operator who names an entry gets that entry, including a deliberately
        cross-provider one -- a Gemini critic beside a Grok writer is a real
        configuration, and substituting a model they did not ask for would be
        worse than the error they get for a key they did not set.

        The accepted limit of that rule: an explicit ``WRITER_MODEL=gemini-3.5-flash``
        is indistinguishable from the default and is rewritten. Detecting it needs
        ``model_fields_set``, which pydantic-settings populates from the env file
        as well -- so the check would pass here and fail for anyone whose value
        arrives from ``.env``, which is everyone. A wrong answer on the rare case
        beats an inconsistent one on the common case.

        **Google wins when both keys are present**, so adding an xAI key to a
        working setup never silently moves a running system onto another provider.

        Deliberately does NOT touch three settings. ``illustrator_model`` and
        ``narrator_model`` have no Google entry to move between -- ``image_configs``
        and ``speech_configs`` register xAI only, so the branch could never fire.
        And ``embedding_model`` is excluded for a stronger reason: ``db/models.py``
        gives each embedder its own table, so rewriting it would repoint retrieval
        at a *different index*, one that may hold nothing. Research would return
        no facts, and a book with no grounding still plans out complete and
        plausible -- a silent wrong answer, where the missing-key error is a loud
        right one.
        """
        # `mode="after"` matters for correctness, not style: `_blank_to_none` runs
        # `mode="before"`, so by here `XAI_API_KEY=` with no value is already None
        # rather than an empty SecretStr that would read as a usable credential.
        if self.google_api_key is not None or self.xai_api_key is None:
            return self

        for field_name, field in type(self).model_fields.items():
            current = getattr(self, field_name)
            if current != field.default:
                continue
            replacement = self._GROK_OVERRIDE.get(
                field_name
            ) or self._GROK_EQUIVALENT.get(str(current))
            if replacement is not None:
                # object.__setattr__ is not needed -- Settings is mutable -- but
                # assigning through the normal path would re-run validation on a
                # model already being validated.
                setattr(self, field_name, replacement)
        return self

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
            # provider integration -- and `langchain-openai` is already a
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
            # entry for the same reason the Grok critic does: the
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

        ``provider`` is what ``get_embedder`` dispatches on, and it exists
        because there are now two genuinely different construction paths: local
        weights loaded once into numpy, and a hosted endpoint called per batch.
        Adding it is what keeps that a registry fact rather than an ``isinstance``
        check inside the factory.

        ``dimensions`` is recorded because it is pinned into both the pgvector
        column width *and* the table name (``chunks_<model>_<dims>``). Changing
        it makes an existing index unreachable rather than merely different --
        which is deliberate: a missing table is a clear failure, where a reused
        one would be a dimension mismatch discovered at query time.

        **On credentials, and this is a decision that reversed.** The local entry
        was chosen partly because it needs no key: every Google call in this
        project's history had failed with a 503 (open item 3), and xAI still has
        no embeddings endpoint at all -- verified again 2026-08-19 against
        ``docs.x.ai``, which exposes only chat and responses. So there is no Grok
        embedder to move to, and this registry is Google or local.

        Maryam has asked for the hosted model as the default anyway, and the cost
        of being wrong is now much lower than it was when the local entry was
        written: ``db/models.py`` gives every embedder *its own table*, so both
        indexes coexist and ``EMBEDDING_MODEL`` switches between them with no
        migration and no re-ingest of the other. Retrieval quality is measurable
        (``make test-corpus``), so this is a reversible choice with a number
        attached rather than a preference.

        **What is genuinely at risk and cannot be argued away:** retrieval stops
        being free, offline and deterministic. The local embedder has produced the
        same vector for the same query since Session 1, which is what made a
        falling hit-rate mean something. See the ``gemini-embedding`` entry.
        """
        return {
            # 256-dim static embeddings, distilled so inference is essentially
            # numpy. Measured in the task 1 spike: 3/3 at rank 1 on fact,
            # paraphrase and structural-craft queries, 3.6s to load.
            #
            # Kept, not deleted, now that the default has moved to the hosted
            # model. It is the only embedder here that has never failed, it costs
            # nothing to retain, and its table still exists -- so it is both the
            # A/B control and the fallback an operator can reach for with one env
            # var if Google is down.
            "potion-base-8M": {
                "provider": "model2vec",
                "identifier": "minishlab/potion-base-8M",
                "dimensions": 256,
            },
            # The default from 2026-08-19, at Maryam's request.
            #
            # **`-001` rather than `-2`, and that was decided by running both.**
            # The newer `gemini-embedding-2` returns exactly ONE embedding per
            # request however many texts it is given -- verified live: a list of
            # three came back as one. It is not broken, and one text at a time
            # gives correct distinct vectors; it simply does not batch. `-001`
            # returns three for three.
            #
            # That matters because ingest embeds the whole corpus. On `-2` a
            # 58-chunk ingest is 58 sequential round trips, and every search
            # query at runtime is its own request either way. Choosing the older
            # model buys batching, which is the difference between ~2 requests
            # and 58 per ingest.
            #
            # Written down because it is invisible from the documentation, which
            # describes `-2` as current and says nothing about this: it is rule
            # 34's lesson repeating on a third provider. The first version of
            # this entry named `-2` and the batching code was written around a
            # list-in-list-out contract that endpoint does not honour.
            #
            # `-001` does not auto-normalise below 3072 dimensions, unlike `-2`.
            # That costs nothing here because `GeminiEmbedder` normalises anyway
            # rather than trusting a provider to -- which is exactly the reason
            # that decision was made.
            #
            # 768 rather than 3072: the model is Matryoshka-trained, so a
            # truncated vector keeps most of its quality, while 3072 costs four
            # times the storage and index size. At 58 chunks quality is decided
            # by curation rather than dimensionality -- hit-rate@3 is already
            # 1.00 on the local 256-dim model, so the headroom being bought is at
            # @1 only. Verified live at 768: unit-length vectors, and two
            # unrelated sentences at cosine 0.58 rather than collapsed together.
            "gemini-embedding": {
                "provider": "google",
                "identifier": "gemini-embedding-001",
                "dimensions": 768,
                "api_key_env_var": "GOOGLE_API_KEY",
            },
        }

    # --- Image registry --------------------------------------------------
    @property
    def image_configs(self) -> dict[str, dict[str, Any]]:
        """Registry of available image-generation models.

        The third registry, and separate from the other two for the reason they
        are separate from each other: an image model takes a prompt and returns
        bytes, so it is built by ``get_image_model`` and shares no construction
        step with either a chat model or an embedder.

        Unlike ``embedding_configs`` these do need a credential, and it is
        deliberately xAI's. Open item 3 records that a successful Google
        generation has never completed in this project -- only a 503 -- while
        every live-verified stage here has gone through xAI. Putting the most
        expensive and most failure-prone stage in the system behind the least
        reliable provider available would be a choice against the evidence.

        ``base_url`` travels in ``params`` exactly as it does for ``grok-*`` chat
        entries: xAI's image endpoints are OpenAI-compatible, which is the same
        fact that let chat reach it with no new dependency.

        **There is no ``seed``**, and that is the provider's constraint rather than
        an omission. xAI abstracts seeds away server-side, and ``quality``,
        ``size`` and ``style`` are unsupported too. This retires the original
        plan of storing each portrait with its prompt *and seed* -- the
        multi-image edit endpoint replaces it, and is the stronger tool anyway: a
        seed reproduces an identical image, not the same character in a new pose,
        which is what every page after the first needs.
        """
        return {
            # Two entries, and the ids were read from `models.list()` on a real
            # key rather than from documentation: a web search asserted
            # `grok-2-image`, which this account cannot see at all. A wrong
            # identifier fails at the first live call and nowhere earlier, after
            # the whole planning stage has been paid for.
            "grok-image": {
                "identifier": "grok-imagine-image",
                "api_key_env_var": "XAI_API_KEY",
                "params": {
                    "base_url": "https://api.x.ai/v1",
                },
            },
            # The slower, better-looking sibling. A separate entry rather than a
            # parameter, exactly as `grok-3-mini-critic` is separate from
            # `grok-3-mini`: the registry's job is to make swapping a model a
            # config change, and a book is worth spending more on than a portrait
            # test run.
            "grok-image-quality": {
                "identifier": "grok-imagine-image-quality",
                "api_key_env_var": "XAI_API_KEY",
                "params": {
                    "base_url": "https://api.x.ai/v1",
                },
            },
        }

    @property
    def speech_configs(self) -> dict[str, dict[str, Any]]:
        """Text-to-speech models, by logical name.

        The fourth registry, and separate from the other three for the reason they
        are separate from each other: a speech model takes text and a voice and
        returns audio bytes, so it is built by ``get_speech_model`` and shares no
        construction step with a chat model, an embedder or an image model.

        **One entry, not two.** ``grok-image`` has a ``grok-image-quality``
        sibling because xAI ships two image models at different tiers. The TTS
        endpoint exposes no such tier -- 26 voices, one endpoint, no quality
        parameter -- so a second entry here would differ in nothing. The
        registry's job is to make swapping a *model* a config change, not to
        enumerate one model's parameters.

        **There is no ``identifier``, and that is measured rather than omitted.**
        ``POST /v1/tts`` returns 200 with no ``model`` field at all, and supplying
        one changes nothing. What varies is the ``voice_id`` request parameter,
        which belongs to the brief. An identifier here would be a value nothing
        reads -- and a *wrong* one fails at the first live call and nowhere
        earlier, which is what a web-search-asserted ``grok-2-image`` cost the
        image seam.

        ``base_url`` travels in ``params`` exactly as it does for ``grok-*`` chat
        and image entries.
        """
        return {
            "grok-speech": {
                "api_key_env_var": "XAI_API_KEY",
                "params": {
                    "base_url": "https://api.x.ai/v1",
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
            "XAI_API_KEY": self.xai_api_key,
            "PERPLEXITY_API_KEY": self.perplexity_api_key,
            "FIRECRAWL_API_KEY": self.firecrawl_api_key,
            "TAVILY_API_KEY": self.tavily_api_key,
        }
        secret = secrets.get(env_var)
        return secret.get_secret_value() if secret is not None else None


settings = Settings()
