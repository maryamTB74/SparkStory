# SparkStory

[![Python 3.14](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org/downloads/release/python-3140/)
[![CI](https://github.com/maryamTB74/SparkStory/actions/workflows/ci.yml/badge.svg)](https://github.com/maryamTB74/SparkStory/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-1139%20passing-brightgreen.svg)](#11-testing)

An agentic MCP server that turns a short idea into a personalised children's storybook.

A parent describes their child — name, age, pronouns, what they love — and a sketch of
an idea. SparkStory plans the story, writes it at the right reading level, illustrates
it with visually consistent characters, and assembles it into a book.

Built as the final project for the Towards AI *Agentic AI Engineering* course.

---

## What works today

`uv run sparkstory` starts a [FastMCP](https://gofastmcp.com) server with four tools, one
prompt and two resources.

Two tools are the story engine. The other two make media, and they are registered only when
enabled. A client cannot call a tool it was never shown, so a server can be deployed unable
to spend money on images rather than merely asked not to.

The engine is a six-agent pipeline on LangGraph with two evaluator-optimizer loops. A critic
judges each draft and the *generator* revises it — there is no separate editor agent.

Before anything is planned, a Researcher looks the story up. It is a ReAct agent over a
small local corpus, with an optional web tool that is off by default, and it decides for
itself whether the premise has anything to get factually wrong.

What it hands the planner is constraints, not facts. "The Moon has no air" arrives as
"nothing outdoors can flutter, drift or make a sound", so the book is quietly true instead
of stopping to explain itself. Most premises need nothing, and coming back with nothing is a
correct answer.

The two loops sit either side of the parent's approval. That is the arrangement the whole
design turns on: the outline a parent is shown is the outline the book is built from.

```
plan_story(brief)
  StoryBrief ──▶ Researcher ──▶ StoryGrounding    ≤3 constraints, ≤2 devices, may be empty
                  │  ├─ search_facts              real-world facts the story could get wrong
                  │  └─ search_web                off by default; fetched and checked first
                  ▼
             Story Planner ──────▶ StoryOutline   title, theme, characters, 4–8 beats
                      ▲                 │
                      └── Outline Critic ┘        ≤2 revisions, stop when it finds nothing
                                        │
                          shown to the parent ──▶ approved
                                        │
write_story(brief, outline)  ◀──────────┘         the same outline, passed back in
  validate_outline                                a mismatch is rejected before any call
                 Plot Planner ──▶ PagePlan        beats laid out across pages; each page
                                        │         gets three notes, not a summary
                      ▼                 │
                   Writer ──▶ StoryProse          the words for every page
                      ▲                 │
                      └── Prose Critic ──┘        ≤2 rewrites, plus counted findings
                                        │
                                      Story       fail closed on an unresolved safety finding
```

That costs six model calls when both critics approve first time, and fourteen at the
ceiling of two revisions each.

Each loop runs N revisions and N+1 critiques, so every draft gets judged, including the one
being returned. That is what lets a loop keep the **best** draft it saw rather than the last
— a revision can be worse than what it replaced, and the loop cannot tell unless it has
scored both.

A finished book can then be illustrated and narrated. Each is a separate pipeline, so
a provider failure cannot damage prose that already passed both critics. Narration writes one
MP3 per page plus a stitched `story.mp3` and reads each page verbatim: no model rewrites the
words, and a SHA-256 per page proves it. See [Narrating a book](#narrating-a-book).

## 1. Directory Layout

Two ideas drive the layout: domain schemas are separated from LLM wiring, and every model is
constructed in exactly one place.

- **`entities/`** is the domain — briefs, outlines, prose, reviews. It depends on nothing.
- **`models/`** is LLM wiring, which is the *opposite* of the common convention where
  `models/` holds domain schemas. Read an imported snippet's `models` import before trusting
  it.
- **`nodes/`** are the agents, each carrying its own prompt. **`workflows/`** orchestrates
  them. **`mcp/`** is the adapter, and holds no logic of its own.

The annotated file tree and the dependency rules are in
[docs/architecture.md](docs/architecture.md).

## 2. Prerequisites

- **Python 3.14** — pinned in `.python-version`
- **[uv](https://docs.astral.sh/uv/)** — standalone install, *not* inside a pyenv
  version. Check with `which uv`; it should be `~/.local/bin/uv`.
- **An API key from either provider** — Google AI Studio
  (https://aistudio.google.com/apikey) or xAI (https://console.x.ai/). Either one alone is
  enough to write a book, because every stage switches to whichever key you have. See
  [Models](#5-models).
- **An xAI key specifically, for pictures or narration.** Those two are xAI-only.

## 3. Install

**1. Clone the repository.**

```bash
git clone https://github.com/maryamTB74/SparkStory.git
cd SparkStory
```

**2. Install dependencies.**

```bash
make install                                   # uv sync + install the pre-commit hooks
```

Or without `make`:

```bash
uv sync                                        # dependencies only
```

`make install` is the same `uv sync` plus `pre-commit install`, so the hooks that keep
lint and formatting clean are active from your first commit.

`uv` reads `.python-version` and fetches Python 3.14.0 itself, so no system Python of that
version is needed.

**3. Build the knowledge index** — optional.

```bash
uv run python scripts/ingest_knowledge.py      # build the knowledge index (once)
```

This command reads `corpus/` and loads it into Postgres, so it needs `DATABASE_URL`
and a `make migrate` first. It also needs `GOOGLE_API_KEY`, since the default embedder is
hosted; set `EMBEDDING_MODEL=potion-base-8M` to embed locally with no key instead, which
downloads about 30 MB of weights the first time. Skip ingestion and everything still works:
research finds nothing, and stories are planned ungrounded.

## 4. Configure environment variables

```bash
cp .env.sample .env   # then edit .env and add your key
```

`.env.sample` documents every variable and is committed; `.env` is gitignored and must
never be committed. A pre-commit hook refuses it even with `git add -f`.

| Variable | Required | Default |
|---|---|---|
| `GOOGLE_API_KEY` | one of this or `XAI_API_KEY`, at call time | none |
| `XAI_API_KEY` | as an alternative to `GOOGLE_API_KEY`, or for illustration and narration, which are xAI-only | none |
| `PLANNER_MODEL` | no | `gemini-3.5-flash` |
| `PLOT_MODEL` | no | `gemini-3.5-flash` |
| `WRITER_MODEL` | no | `gemini-3.5-flash` |
| `OUTLINE_CRITIC_MODEL` | no | `gemini-3.5-flash-critic` |
| `PROSE_CRITIC_MODEL` | no | `gemini-3.5-flash-critic` |
| `MAX_OUTLINE_REVISIONS` | no | `2` — `0` still critiques once, never revises |
| `MAX_PROSE_REVISIONS` | no | `2` — `0` still critiques once and still fails closed on safety |
| `MAX_REVIEWS_PER_PASS` | no | `5` |
| `DATABASE_URL` | for retrieval and memory | none — `docker-compose.yml` brings up a matching pgvector instance |
| `RESEARCHER_MODEL` | no | `gemini-3.5-flash` — `MAX_RESEARCH_STEPS=0` skips research |
| `EMBEDDING_MODEL` | for retrieval | `gemini-embedding` — hosted, and the first embedder here that needs `GOOGLE_API_KEY`. `potion-base-8M` still works and runs locally with no key; each embedder has its own table, so switching back needs no migration and no re-ingest |
| `MAX_RESEARCH_STEPS` | no | `4` — `0` skips research entirely, which is how you compare a grounded run against an ungrounded one |
| `RETRIEVAL_TOP_K` | no | `5` |
| `RERANKER_MODEL` | no | `gemini-3.5-flash-critic` — reorders retrieval candidates. **Built and deliberately not wired in**: nothing in the pipeline calls it, so setting this has no effect today |
| `MAX_WEB_SEARCHES` | no | `0` — the web tool is **off**; above zero needs `PERPLEXITY_API_KEY` (or `TAVILY_API_KEY`) and `FIRECRAWL_API_KEY` |
| `ILLUSTRATION_ENABLED` | no | `true` — false unregisters `illustrate_story` entirely |
| `NARRATION_ENABLED` | no | `true` — false unregisters `narrate_story` entirely |
| `ILLUSTRATOR_MODEL` | no | `grok-image` |
| `ILLUSTRATION_DIRECTOR_MODEL` | no | `gemini-3.5-flash` — writing the visual plan is a writing task |
| `CONSISTENCY_JUDGE_MODEL` | no | `gemini-3.5-flash-critic` — but the only measured comparison here was between two xAI models, so no Google model has ever judged an image in this project |
| `JUDGE_PAGES` | no | `true` — false keeps the portrait check, skips the per-page one |
| `NARRATOR_MODEL` | no | `grok-speech` — reads a finished book aloud; needs `XAI_API_KEY` |
| `MEMORY_EXTRACTOR_MODEL` | no | `gemini-3.5-flash` — **fails open**: no key means nothing is stored and the run looks normal |
| `JUDGE_MODEL` | no | `gemini-3.5-flash-critic` — offline eval only, never a story run |
| `OPIK_ENABLED` | no | `false` — nothing imports opik below this |
| `LOG_LEVEL` | no | `INFO` |
| `LOG_LEVEL_DEPENDENCIES` | no | `WARNING` |
| `SERVER_NAME` | no | `SparkStory MCP Server` |
| `SERVER_VERSION` | no | read from package metadata |
| `SERVER_HOST` | no | `127.0.0.1` — HTTP transport only; must be `0.0.0.0` in Docker |
| `SERVER_PORT` | no | `8000` — HTTP transport only |

### You do not have to set any `*_MODEL` variable

All eleven chat stages default to Gemini. If the only key you have is `XAI_API_KEY`, each one
you have not set switches to its Grok equivalent at startup. So one key is enough, whichever
it is. Set both and Google wins. See [Models](#5-models).

**Three settings are left out of that, and can still catch you:**

- **`EMBEDDING_MODEL`** stays on Gemini, because each embedder has its own table — switching
  it would point retrieval at a different, possibly empty index. Research would find nothing,
  and a book planned with no grounding still comes out complete, so you would never notice.
  Set `EMBEDDING_MODEL=potion-base-8M` to embed locally with no key.
- **`ILLUSTRATOR_MODEL`** and **`NARRATOR_MODEL`** have no Gemini equivalent. Pictures and
  narration need `XAI_API_KEY` whatever else is set.

The server starts and lists its tools with no key at all. A missing key fails at call time,
naming the variable to set.

## 5. Models

Model configuration is two-level, which is what makes per-agent choices a config change
rather than a code change:

- **`llm_configs`** in `config.py` — a registry mapping a name to its provider
  identifier, the API key it needs, and its provider parameters.
- **`*_MODEL` settings** — which registry entry each agent uses.

Every `*_MODEL` names a Gemini entry by default. A validator on `Settings` swaps the ones you
have not set for their Grok equivalent when `GOOGLE_API_KEY` is missing and `XAI_API_KEY` is
present.

Entries are paired by temperature, so critics, judges and the reranker all land on a
`temperature 0.0` entry.

A value you set is used as written and never swapped — with one exception:
`WRITER_MODEL=gemini-3.5-flash` is indistinguishable from the default, so it *is* swapped
under an xAI-only key.

| Entry | Provider | Notes |
|---|---|---|
| `gemini-3.5-flash` | Google | Default. Returns `503 UNAVAILABLE` when demand spikes. |
| `gemini-3.5-flash-critic` | Google | The same model at `temperature 0.0`. A revision loop stops when a critic returns no findings, so a critic that answers differently each time makes that signal noise. |
| `gemini-2.5-pro` | Google | Slower, more expensive. |
| `grok-4` | xAI | A different provider, so Google congestion does not affect it. |
| `grok-3-mini` | xAI | Smaller and cheaper. |
| `grok-3-mini-critic` | xAI | `grok-3-mini` at `temperature 0.0`. |

xAI is reached through its OpenAI-compatible API (`openai:grok-*` plus a `base_url` in
the registry params), so it needs no separate provider integration.

Any setting can be overridden per run without editing `.env`:

```bash
WRITER_MODEL=grok-4 uv run python scripts/write_one_story.py
```

## 6. Optional features

Three things are built and off, or on but easy to miss. Each is independent of the story
engine.

### The web tool — off by default

A premise the corpus knows nothing about — submarines, dinosaurs, how a violin works — is
planned ungrounded, because `drop_unprovenanced` needs a resolvable id and the model's own
knowledge has none. The web tool gives such claims a source.

| Setting | Default | What it does |
|---|---|---|
| `MAX_WEB_SEARCHES` | `0` | Above zero enables `search_web`. At `0` nothing is built and neither key below is read. |
| `VERIFY_WEB_CLAIMS` | `true` | Fetch each cited page and check it supports the claim. |
| `PERPLEXITY_API_KEY` | unset | Web search — returns claims, each with a URL. |
| `TAVILY_API_KEY` | unset | Fallback search, used if Perplexity fails or has no key. |
| `FIRECRAWL_API_KEY` | unset | Fetches a cited page so the claim can be checked against it. |

**Why two services rather than one.** Perplexity returns claims each with a URL — but that
URL is written by the *model* into a structured field, so it is an assertion, not a source.
A plausible fabrication would be indistinguishable from a real citation, which is the exact
defect that made this project overwrite a corpus fact's `source` from the store rather than
trust what the agent wrote. Firecrawl fetches the page, and a claim survives only if the
page contains a sentence supporting it. A fabricated URL dies at the fetch.

The judge must quote the supporting sentence, and code then checks that quote against the
fetched text. That matters because the laziest answer to "is this claim
supported?" is *yes*; requiring a quote turns an agreeable verdict into a falsifiable one.

```bash
# on, for one run
uv run python scripts/write_one_story.py --stage research --max-web-searches 3 \
    --premise "a child who wants to know how a submarine sinks and rises"

# see what search returned before verification (facts are then dropped as unprovenanced)
uv run python scripts/write_one_story.py --stage research --max-web-searches 3 \
    --no-verify-web --premise "..."
```

A `--stage research` run writes `web_sources.json` recording every source consulted and
whether it passed. Corpus results are always preferred: the Researcher is told to search
the collection first and reach for the web only when it finds nothing, and the three-fact
budget is shared, so a web fact displaces a corpus one rather than adding to it.

### Narrating a book

A fourth pipeline reads a finished `Story` aloud through xAI's text-to-speech endpoint,
writing one MP3 per page plus a stitched `story.mp3` and a `narration.json` recording what
was produced and what failed.

| Setting | Default | What it does |
|---|---|---|
| `NARRATOR_MODEL` | `grok-speech` | Names an entry in `speech_configs`. Needs `XAI_API_KEY`. |

The voice is **not** a setting — it belongs to the brief, because a parent chooses it per
story and it crosses the MCP tool boundary:

| `voice` | Reads as |
|---|---|
| `female` *(default)* | `eve` |
| `male` | `leo` |

Two values rather than four expressive ones, and that is deliberate. The provider exposes
26 voices whose only distinguishing field is `gender` — there is no metadata saying which
voice is *warm* or *gentle*, so a richer enum would have been invention presented as a
mapping. Adding a value is one row here and one in `_VOICES`, once a listen shows it
audibly differs.

Reading pace comes from the child's `reading_level` (0.85 for a pre-reader up to 1.0 for a
confident one), computed in code. No model chooses it, and no model rewrites the text: the
script is `StoryPage.text` exactly. That is why "the audio matches the printed page" is
something you can check rather than something we intend.

```bash
uv run python scripts/write_one_story.py --narrate --voice male
```

Narration **fails soft**: a page whose audio fails is recorded as failed, its file is
absent, and the book still plays. If *nothing* narrates, no `story.mp3` is written at all.
Writing an empty one would be worse than writing none, because it plays as silence and a
listener cannot tell that apart from a book that simply worked.

### Tracing a run (Opik)

Off by default, and while it is off nothing imports it — so a run that does not trace pays
nothing for the dependency.

| Setting | Default | What it does |
|---|---|---|
| `OPIK_ENABLED` | `false` | Above this, the three below are read. At `false` no opik module is imported. |
| `OPIK_API_KEY` | unset | Opik key. A blank or missing value logs a warning and disables tracing. |
| `OPIK_WORKSPACE` | unset | The workspace holding the project. |
| `OPIK_PROJECT_NAME` | `sparkstory` | The project traces are grouped under. |

A book is five to eleven model calls across two tools, and comparing two of them has meant
diffing run directories. With tracing on, each pipeline sends one trace whose spans are the
individual nodes, keyed by the same `request_id` that appears in the logs — so a log
line leads to its trace. The model each stage used is recorded on the trace, which is what
makes two of them comparable.

**Every failure here is a warning, never an error.** A missing key, an unreachable
workspace or a backend that rejects the run costs the trace and nothing else; the book is
still written. Nobody asked for tracing as part of writing a book, so it is not allowed to
be the reason one fails.

```bash
# the eval harness can also publish its fixture briefs as an Opik dataset
uv run python scripts/run_evals.py --fixtures --opik
```

## 7. Available MCP tools and prompts

### `plan_story(brief: StoryBrief) -> StoryOutline`

Plans the structure — title, logline, theme, characters and ordered beats — without
writing prose. Show the outline to the parent and get approval before committing to a
full write.

**Two to four model calls: it runs the outline critic and revises its own plan.** The
outline it returns is the one a parent approves *and* the one `write_story` builds from,
so it has to be worth approving. It needs a working `OUTLINE_CRITIC_MODEL` — a critic
whose provider has no key set will fail the whole call.

### `write_story(brief, outline, output_directory) -> Story`

Builds the book from an outline `plan_story` returned — same title, same characters, same
beats. It never plans; pass the outline through unchanged. Three to seven model calls, so
much slower than `plan_story` — expect a few minutes.

The finished book is written into `output_directory` as `story.json`, and a `story.pdf` is
rendered beside it; the returned `Story` records where they went. Tell the parent that
path — it is how they find the book again once the conversation is over. Pass the same
directory to `illustrate_story` and `narrate_story` so a book and its media stay together.

`output_directory` is confined under `outputs/`, whatever the caller asks for. It is chosen
by an LLM client, and a prompt can ask but cannot enforce: told to use `outputs/<name>`, a
live client passed a bare name and the book landed in the repository root. A path escaping
the root is refused rather than quietly rewritten, so a client that asked for somewhere it
cannot have is told so instead of losing track of its own artifacts.

An outline that does not fit the brief (more beats than pages) is rejected with a
`ToolError` before any model call, since the outline arrives from a client and a mismatch
is the caller's mistake rather than a bug.

It raises a `ToolError` rather than returning a book if a safety finding survives every
rewrite. That is deliberate: for a book written for a named child, no book beats a book
carrying something the parent asked to keep out.

A `StoryBrief` carries the child (name, age, pronouns, reading level, interests), the
premise, a tone, `world_rules`, a page count, and `must_include` / `avoid` lists. Enums
constrain tone, reading level and world rules so the calling agent sees a fixed
vocabulary rather than guessing.

`world_rules` is a separate axis from `tone`, and the two are easy to confuse. Tone is
register — how the story feels. World rules are physics — whether the story's world may
be broken. A *gentle* story can break physics and an *adventurous* one can be strictly
real, so neither implies the other.

| Value | What it means |
|---|---|
| `imaginative` *(default)* | The idea itself is impossible — a fox who flies to the Moon. Retrieved facts are detail that makes the impossible parts believable, and the premise may break them. |
| `realistic` | Getting the real world right is part of the point. The story should never contradict a retrieved fact. |

**The default is `imaginative`, and it is chosen from evidence rather than from
compatibility.** On the standing test premise — *"a fox who wants to visit the moon"*,
`tone: magical` — the realistic rendering planned three failed launches and resolved with
the child holding a paper tube up to the Moon, while the ungrounded control let the rocket
fly. Most personalised picture books are imaginative; a talking fox is not realistic.

### `illustrate_story(brief, story, output_directory) -> StoryArt`

Draws the pictures for a story `write_story` has already written. Decides one shared visual
style for the whole book, draws a reference portrait of each character, then draws every
page *from those portraits* so the same character looks the same throughout.

**Registered only when `ILLUSTRATION_ENABLED` is true — it defaults to true**, so the tool
is offered unless you turn it off. Setting it false means a deployment that cannot afford
images does not advertise the tool at all: a client cannot call what `list_tools` never
showed it. It is the slowest and most expensive tool here: roughly one image per page plus
one per character.

A picture that cannot be drawn is reported rather than fatal — that page simply has no
illustration, and the book still assembles. `fully_conditioned` on the result says whether
every page was drawn from the portraits.

### `narrate_story(brief, story, output_directory) -> StoryNarration`

Reads a finished story aloud, writing one MP3 per page plus a stitched `story.mp3`. The text
is spoken exactly as written — no model rephrases it, and a SHA-256 per page proves it.

**Registered only when `NARRATION_ENABLED` is true — it too defaults to true.** It is a
separate switch from illustration rather than one shared "media" flag, because narration is
roughly two orders of magnitude cheaper: an installation that cannot afford pictures may
still want a book read aloud. It does not need the pictures, so it runs before, after or
instead of `illustrate_story`.

Fail-soft in the same shape: a page that cannot be narrated has no audio rather than killing
the run, and when no page could be narrated there is no `story.mp3` rather than a silent one.

### `create_storybook` (prompt)

The guided path, and the one to use in a real client. Takes no arguments and returns
instructions telling the client to gather a brief, call `plan_story`, show the parent the
full outline, **stop and wait for approval**, and only then call `write_story` — passing
the approved outline through unchanged. If the parent asks for changes, the client amends
the brief and plans again.

The confirmation is advisory. Nothing server-side verifies that a human approved — the
prompt instructs the client's model, which is what an MCP prompt is. A client
that ignores the instruction can still call `write_story` directly.

### Resources

Two read-only endpoints. Neither takes a parameter, neither writes, and both are cheap —
they read files and return text, so a client may call them freely.

| URI | Returns |
|---|---|
| `sparkstory://library` | Finished books on disk: run id, title, page count, and whether a PDF or narration exists |
| `sparkstory://corpus` | Retrieval corpus stats: file and chunk counts, and the embedding model |

A resource's text is read by a *client's model*, so it is prompt material rather than a
debug dump. `sparkstory://library` therefore reports the run's timestamp rather than its
directory name, since a run directory is named after the premise, and it never opens a
`brief.json`, which holds the child's name. Neither resource reads `data/`, which holds
per-child memory.

## 8. Example queries

Once the server is connected to a client, these are the things to ask it. The first is the
whole point of the project; the rest exercise one tool each.

**The guided workflow — this is the one to try first.** It runs the `create_storybook`
prompt, which is where the parent's approval step lives.

```text
Use the create_storybook prompt to make a book for my daughter Ada, who is 6,
uses she/her, and loves thunderstorms and the sea.
```

The client will ask for anything missing, call `plan_story`, show you the whole outline —
title, theme, characters and every beat — and then **stop and wait**. Reply with an
approval or a change ("make Ada the one who wants to sail"), and only then does it call
`write_story`.

**Plan an outline without writing the book** — one to four model calls, the cheapest way
to see the engine work.

```text
Call plan_story for a 8-page early-reader story about a fox who wants to visit the moon.
The child is Maryam, age 5, she/her, and the story must include a paper rocket.
```

**Illustrate a finished book** — needs `XAI_API_KEY` and `ILLUSTRATION_ENABLED=true`.

```text
Illustrate the story you just wrote.
```

**Narrate a finished book** — needs `XAI_API_KEY` and `NARRATION_ENABLED=true`. Roughly two
orders of magnitude cheaper than illustration, which is why the two have separate switches.

```text
Narrate that story with the eve voice.
```

**Read a resource** — no model call, and no cost.

```text
List the books in the SparkStory library.
```

To drive the server programmatically instead of through a client, see
[the MCP client configuration](#mcp-client-configuration) and [writing a story from the terminal](#10-write-a-story-from-the-terminal).

## 9. Run the server

```bash
make run              # stdio transport (the default)
```

Nothing may write to stdout: under stdio transport that channel carries JSON-RPC, so
all logging goes to stderr and the FastMCP banner is disabled. A stray `print()` corrupts
the protocol and surfaces as a JSON parse error that looks nothing like its cause. The rule
applies in every transport even though it only *matters* under stdio, because a rule with
an exception is one people forget in the mode where it counts.

### HTTP transport

```bash
uv run sparkstory --transport http
```

The MCP endpoint is **`http://127.0.0.1:8000/mcp`** — note the `/mcp` path, which is
FastMCP's default and is not visible from the command line.

| Variable | Default | Notes |
|---|---|---|
| `SERVER_HOST` | `127.0.0.1` | loopback only; see the warning below |
| `SERVER_PORT` | `8000` | |

Both are read **only** on this path; stdio has no address.

> **`SERVER_HOST=0.0.0.0` publishes an unauthenticated server that spends your API keys.**
> There is no auth layer, and every tool call costs real model calls, so anyone who can
> reach the port can generate books against your quota. The default binds loopback for that
> reason. Change it only when the network exposure is what you want.

`stdio` remains the default deliberately: `uv run sparkstory` with no arguments is what the
client configuration below invokes, so a default of `http` would break every existing client
— and the symptom would be a client that hangs rather than an error.

### The web UI

The HTTP transport also serves a browser interface, so a parent can make a book without an
MCP client. Start the server as above and open <http://127.0.0.1:8000/>.

The flow is: describe the child and the premise → read the plan SparkStory produced →
**approve it or ask for a different one** → read the finished book. The approval step is the
point: the outline is what the whole book is built from, and it is cheap to redo before
writing and expensive after.

**Unlike the MCP path, the server holds the approved outline itself.** The browser is shown
the plan and sends back only a job id, so the book is always built from the outline the
server planned rather than from anything the page could have altered. That is
tamper-resistance, not proof a human approved — see *Known gaps*.

**The UI never generates pictures, audio or video.** It displays them when a run directory
already contains them, so its only cost is the model calls for planning and writing. Media
is made with `scripts/write_one_story.py`.

> **The `SERVER_HOST=0.0.0.0` warning above applies with more force here.** These pages serve
> a child's name, their illustrations and the audio of their book, with no authentication of
> any kind. Job ids are unguessable, which is obscurity rather than a permission check. To use
> this from another device, put it behind something that authenticates — an SSH tunnel, a VPN,
> or a Cloud Run deployment with `--no-allow-unauthenticated` — rather than binding to
> `0.0.0.0`.

Under `--transport stdio` these routes are registered and never served. That is the transport,
not a bug: stdio has no HTTP.

### Docker

```bash
docker compose up -d          # build and serve on http://localhost:8000/mcp
docker compose logs -f        # all output is on stderr
docker compose down
```

One service. There is no `postgres` container because nothing is stored yet — see the
infrastructure spec for when that changes.

**Your `.env` is not in the image, and this matters.** The build context excludes it, so a
bare `docker run` gets the schema defaults and no keys at all. The container starts happily,
lists its tools, then dies on the first tool call.

Compose reads `.env` from the repository root and passes the values through, so run it
through compose unless you know why you are not.

The provider switching does not rescue this. It picks a provider from the keys present, and
with no keys present there is nothing to pick from.

Two volumes:

| Mount | Why |
|---|---|
| `./data:/app/data` | The knowledge index and the HuggingFace weight cache (`HF_HOME` points inside it). Without this, `model2vec` re-downloads ~59 MB per container — and if it cannot, retrieval returns nothing *silently*, because an absent index is treated as "no results" by design. |
| `./outputs:/app/outputs` | So run artifacts survive the container. Provisional: only `scripts/write_one_story.py` writes here today, not the MCP tools. |

The image installs no system libraries — no `apt-get` layer at all. `reportlab` is pure
Python and Pillow ships manylinux wheels with zlib, jpeg and freetype bundled, which is
verified by rendering a PDF inside the container rather than assumed. This is also why
`weasyprint` was rejected: it needs Pango and cairo as system libraries.

### MCP client configuration

For Cursor (`.cursor/mcp.json`) or Claude Desktop:

```json
{
  "mcpServers": {
    "sparkstory": {
      "command": "uv",
      "args": [
        "--directory",
        "/absolute/path/to/SparkStory",
        "run",
        "sparkstory"
      ],
      "env": {
        "GOOGLE_API_KEY": "your-google-api-key-here",
        "XAI_API_KEY": "your-xai-api-key-here"
      }
    }
  }
}
```

Replace `/absolute/path/to/SparkStory` with where you cloned the repository.

**The `env` block is optional and the worse of the two options.** The server resolves `.env`
from its own location rather than from the working directory, so it finds your keys wherever
a client launches it from. Putting them in `.env` instead keeps credentials out of a file
that often lives in a repository — and only one of the two keys is needed, since every stage
switches to whichever provider you have. Delete the `env` block entirely if you use `.env`.

## 10. Write a story from the terminal

```bash
uv run python scripts/write_one_story.py                     # full book
uv run python scripts/write_one_story.py --stage plan        # one model call
uv run python scripts/write_one_story.py --stage plot        # + the page plan
uv run python scripts/write_one_story.py --debug             # log the prompts
uv run python scripts/write_one_story.py --name Ada --age 8 \
    --level confident --premise "a girl who befriends a thunderstorm"
```

Each run is saved to `outputs/<timestamp>-<premise>/`: the brief, each stage's output,
`story.md`, and `run.log`. Artifacts are written as each stage completes, so a failure
still leaves the outputs that led to it. `outputs/` is gitignored — every run contains a
real child's name.

The best test of a children's story is reading it aloud, which is what this script is for.

## 11. Testing

```bash
make test             # 1139 tests, no network, ~18 seconds
make test-fast        # stop at the first failure
make ci-local         # format-check, lint-check and test — everything CI runs
```

**1139 tests pass offline and need no API key**, because agents are tested by passing a
`FakeModel` to the node constructor rather than by patching module attributes. `ruff check`
and `ruff format` are clean across `src`, `tests` and `scripts`, and the same three commands
run in GitHub Actions on every push ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)).

A further **23 tests are marked and excluded** from `make test`, because each needs
something a clean clone does not have:

```bash
make test-corpus      # retrieval hit-rate@1 and @3 — needs a built index; free only
                      # under EMBEDDING_MODEL=potion-base-8M, since the default
                      # embedder is hosted and charges per query
make test-vision      # the consistency judge on committed images — needs XAI_API_KEY
make test-video       # assembles a video from committed artifacts — needs ffmpeg; free
```

Excluding them is what keeps *clone the repo and the tests pass* true. `make test-video` is
free and local, so it is the one to run first if you want to see a marked test work.

## 12. Development

```bash
make help             # list every target
make fix              # auto-fix lint, then reformat
make check            # format + lint, without changing files
make clean            # drop caches
```

## 13. Troubleshooting

**`MissingAPIKeyError` although `.env` has the key.** A blank value (`GOOGLE_API_KEY=`)
is normalised to unset on purpose. Check the value is actually present.

**`503 UNAVAILABLE` from Gemini.** Capacity, not configuration — the newest model is the
most contended. It is retried three times with backoff first. To switch provider, remove
`GOOGLE_API_KEY` and leave `XAI_API_KEY` set: every stage then resolves to xAI on its own.
To move one stage only, name it — `WRITER_MODEL=grok-4`.

**`StoryStructureError: Outline has 6 beats but the book has only 5 pages`.** Not a bug.
Each beat needs at least one page, so beats cannot outnumber pages. Ask for more pages.

**`StoryStructureError: Beats [n] have no page`.** An agent produced well-formed but
structurally wrong output. Deliberately loud and not retried — a retry with an identical
prompt only re-rolls the dice.

**A tool call returns a JSON parse error.** Something wrote to stdout. See §7.

## 14. Known gaps

Stated explicitly rather than implied:

- **`avoid` is enforced by judgement, not by matching.** A safety critic reads every page
  against the brief's `avoid` list. A finding that survives every rewrite fails the run
  rather than returning the book.

  This is an LLM's judgement, so it can be wrong both ways. A near-synonym it does not
  recognise gets through, and a false positive costs the whole book. Both rates are worth
  watching.
- **Pacing is measured but not enforced.** Pages-per-beat is logged by narrative
  function. There is no loop over the page plan to route the finding to, and an
  unbalanced climax is bad rather than impossible, so nothing acts on it yet.
- **Reading level is judged, not measured.** The prose critic assesses it against the
  child's level; no readability score is computed.
- **Research is advisory and fails open.** A missing index, a broken embedder or a failed
  provider costs you the facts, not the book. The story is planned ungrounded and a line
  lands in the log.

  This is deliberate, and it is the opposite of the safety gate: fail closed on harm, fail
  open on enrichment.
- **`story.mp3` has no duration metadata.** The per-page MP3s are ordinary valid files, and
  the stitched book is a legal MPEG stream — verified by walking every frame of a real run.

  What plain concatenation does not produce is a Xing/Info header or an ID3 tag, and that is
  where duration lives for a variable-bitrate stream. A player therefore guesses the length
  from the first frame. Playing from the start is correct; scrubbing may not be.

  Fixing it properly needs `ffmpeg`, a system binary, which is not worth adding until
  seeking matters to someone.
- **There are no character voices.** One voice reads the whole book, and no model chooses
  emphasis or delivery.

  That is deliberate. A per-page "delivery note" is the kind of instruction a model satisfies
  with "read warmly" — a real cost for no real effect.
- **`leo` is an unverified pick.** The male voice was chosen from the provider's 19 male
  voices, which carry no distinguishing metadata beyond `gender`. It generates correctly;
  whether it is *right* for a bedtime story is a listening judgement nobody has made.
  `orion` and `atlas` are the obvious alternatives and both work.
- **A retrieved note reaches the planner, never the Writer.** So a fact shapes the
  *plan* and the prose inherits it indirectly. Nothing checks that the finished page obeys
  it, and a read-aloud device handed to a planner has been observed being *described*
  instead of used.
- **In `imaginative` mode a retrieved fact is advisory, by design.** The planner is told the
  premise may break a fact where it needs to, and asked to break as few as it can.

  Nothing counts the breaks or checks that any given one was deliberate. So a story can
  contradict the corpus and still be exactly what was asked for. Under `realistic` the same
  facts are framed as binding — but that too is an instruction to a model, not a check.
- **Web verification checks that a page says something, not that it is right.** With the web
  tool on, every cited URL is fetched and a claim survives only if the page contains a
  sentence supporting it.

  That catches a fabricated URL and a misread source. It does not catch a page that is
  confidently wrong — a bad blog post stating the claim plainly will pass. The corpus has the
  same limitation, since nobody has fact-checked it either, but the web is a far larger
  surface for it.
- **Nobody has fact-checked the corpus.** Its 46 factual statements are written from NASA
  and Simple English Wikipedia material, and no test can verify that a sentence is true.
- **Retrieval quality is measured, uniquely in this project.** `make test-corpus` reports
  hit-rate@1 and @3 over a 19-query labelled set. It needs a built index and a real
  embedder, so it is excluded from `make test`.
- **The video is a slideshow, not an animation.** `story.mp4` gives each page a Ken Burns
  pan held for exactly that page's narration, and the frame arithmetic is exact. But
  nothing *inside* a frame moves, and on being watched it was judged "not a video, purely
  pictures" — which is the honest description. Model-generated motion is unbuilt.
- **Nobody has listened to the narration.** Five runs produced structurally valid audio and
  the mechanism is verified; whether the voices *sound* like a bedtime story is a judgement
  no test makes.
- **The LLM judge is not trustworthy on its own.** Its per-page verdicts were compared
  against 120 human labels and agreed no better than chance (Cohen's kappa −0.066 / +0.121 /
  −0.060). A follow-up measured the *human* ceiling and found it barely above chance too, so
  the honest reading is that these questions are not consistently answerable as written
  rather than that the judge is broken. The six deterministic metrics have no such problem.
  Do not use a judged number to argue a book improved.
- **Nothing enforces that a human approved the outline.** The confirmation step is advisory:
  `plan_story` returns a plan, the prompt instructs the client to get approval, and
  `write_story` requires that plan as an argument. That raises the floor from nothing to
  "you must possess a plan", and a fabricated outline is still indistinguishable from an
  approved one.

## 15. Certification

Which course requirements are met, where each one is implemented, and what has actually been
verified: [docs/certification.md](docs/certification.md).

All four required custom features are banked, and every mandatory requirement is closed.

## 16. Roadmap

| # | Session | Status |
|---|---|---|
| 1 | Foundation + end-to-end vertical slice | Done |
| 2 | Multi-agent story engine on LangGraph | Done |
| 3 | Domain model + file-based canon store | **Dropped** — superseded by memory, which answers the same question |
| 4 | Critic panel + evaluator-optimizer loop | Done |
| 5 | Agentic RAG (ReAct + hybrid search + RRF) | Done |
| 6 | Illustration pipeline + multimodal consistency judge | Done |
| 7 | Human-in-the-loop workflow + the flagship MCP prompt | Done |
| 8 | Book assembly (PDF) + narration | Done |
| 9 | MCP client | Done |
| 10 | Observability + evaluation with an LLM judge | Done |
| 11 | Postgres + pgvector, Alembic migrations | Done |
| 12 | CI (GitHub Actions), Docker, HTTP transport | Done |
| 13 | Long-term memory, per child | Done |
| 14 | Cloud deployment | Not started |
| 15 | Polished UI | **Done** — a parent-facing web UI served from the MCP server itself, with the plan-approve step as its centre. See [The web UI](#the-web-ui) |
| 16 | Animated video | **Half done** — assembly works and is frame-exact; the animation itself is unbuilt. See [Known gaps](#14-known-gaps) |

## License

MIT — see [LICENSE](LICENSE).
