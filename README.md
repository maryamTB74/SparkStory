# SparkStory

[![Python 3.14](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org/downloads/release/python-3140/)
[![CI](https://github.com/maryamTB74/SparkStory/actions/workflows/ci.yml/badge.svg)](https://github.com/maryamTB74/SparkStory/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-1139%20passing-brightgreen.svg)](.github/workflows/ci.yml)

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
words, and a SHA-256 per page proves it.

## 1. Directory Layout

Two ideas drive the layout: domain schemas are separated from LLM wiring, and every model is
constructed in exactly one place.

- **`entities/`** is the domain — briefs, outlines, prose, reviews. It depends on nothing.
- **`models/`** is LLM wiring, which is the *opposite* of the common convention where
  `models/` holds domain schemas. Read an imported snippet's `models` import before trusting
  it.
- **`nodes/`** are the agents, each carrying its own prompt. **`workflows/`** orchestrates
  them. **`mcp/`** is the adapter, and holds no logic of its own.

The dependency rule is one-directional: `entities/` depends on nothing, `nodes/` and
`workflows/` depend on `entities/` and `models/`, and `mcp/` depends on the layers beneath it
and is depended on by none of them. A module that needs to import upward is a sign the logic
belongs a layer down.

## 2. Prerequisites

- **Python 3.14** — pinned in `.python-version`
- **[uv](https://docs.astral.sh/uv/)** — standalone install, *not* inside a pyenv
  version. Check with `which uv`; it should be `~/.local/bin/uv`.
- **An API key from either provider** — Google AI Studio
  (https://aistudio.google.com/apikey) or xAI (https://console.x.ai/). Either one alone is
  enough to write a book, because every stage switches to whichever key you have.
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

| Variable | Required | How to get it |
|---|---|---|
| `GOOGLE_API_KEY` | one of this or `XAI_API_KEY` | [Google AI Studio](https://aistudio.google.com/apikey) |
| `XAI_API_KEY` | as an alternative to `GOOGLE_API_KEY` — and required for illustration and narration, which are xAI-only | [xAI Console](https://console.x.ai/) |
| `DATABASE_URL` | only for retrieval and memory | `docker-compose.yml` brings up a matching pgvector instance; skip it and stories are planned ungrounded |

**Those three are the only settings with no default.** Every other variable — the eleven
`*_MODEL` choices, the revision caps, the feature switches, the log levels and the server
address — has a working value already, and each is documented with its default and its
trade-offs in [`.env.sample`](.env.sample). That includes the three optional subsystems —
the web tool (`MAX_WEB_SEARCHES`, off by default), narration (`NARRATION_ENABLED`) and Opik
tracing (`OPIK_ENABLED`) — each of which is off or self-contained and needs no extra setup
to leave alone.

### You do not have to set any `*_MODEL` variable

All eleven chat stages default to Gemini. If the only key you have is `XAI_API_KEY`, each one
you have not set switches to its Grok equivalent at startup. So one key is enough, whichever
it is. Set both and Google wins.

Model choice is two-level: a registry in `config.py` maps a name to its provider identifier,
API key and parameters, and the `*_MODEL` settings say which entry each agent uses. That is
what makes swapping a model a config change rather than a code change. Any setting can be
overridden for a single run:

```bash
WRITER_MODEL=grok-4 uv run python scripts/write_one_story.py
```

**Three settings are left out of that, and can still catch you:**

- **`EMBEDDING_MODEL`** stays on Gemini, because each embedder has its own table — switching
  it would point retrieval at a different, possibly empty index. Research would find nothing,
  and a book planned with no grounding still comes out complete, so you would never notice.
  Set `EMBEDDING_MODEL=potion-base-8M` to embed locally with no key.
- **`ILLUSTRATOR_MODEL`** and **`NARRATOR_MODEL`** have no Gemini equivalent. Pictures and
  narration need `XAI_API_KEY` whatever else is set.

The server starts and lists its tools with no key at all. A missing key fails at call time,
naming the variable to set.

## 5. Run the server

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
tamper-resistance, not proof a human approved.

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

## 6. Required features

The course's mandatory list, and where each one lives.

| Requirement | Where | Notes |
|---|---|---|
| **MCP server in Python using FastMCP** | `src/sparkstory/mcp/server.py` | `create_server()` is split from `main()`, so the same server runs over stdio, over HTTP, and in-memory inside tests |
| **At least one meaningful MCP tool** | `src/sparkstory/mcp/routers/tools.py` | Four. `plan_story` and `write_story` are the story engine; `illustrate_story` and `narrate_story` are registered *conditionally*, so a deployment can be unable to spend money on media rather than merely asked not to |
| **An MCP prompt tying tools into a workflow, with a user-confirmation step** | `src/sparkstory/mcp/prompts/create_storybook.py` | `create_storybook` instructs a client to gather a brief, call `plan_story`, show the parent the whole outline, **stop**, and only then call `write_story` passing that outline through unchanged. Verified against two live client models (`grok-3-mini` and `grok-4`), 6/6 each, including that the outline the client sends back is byte-identical to the one it received |
| **Public GitHub repository** | [maryamTB74/SparkStory](https://github.com/maryamTB74/SparkStory) | |
| **Initialised with uv, Python 3.14** | `pyproject.toml`, `.python-version` | 3.14.0 |
| **Clear `src/` structure, documented** | [§1 Directory Layout](#1-directory-layout) | Domain schemas, LLM wiring, agents and orchestration each in their own package |
| **Comprehensive README** | this file | |
| **No committed credentials; `.env.sample` provided** | `.env.sample` | A pre-commit hook refuses to commit `.env` even with `git add -f`; `.gitignore` stops the accident, the hook stops the override |

## 7. Custom features

Twelve of the course's optional list. Each is one bullet from that list.

**Structured outputs.** Every model call binds a Pydantic schema, and the schema is bound by
the *node* rather than the model factory, because a node's output type is its contract.
Docstrings on these models are prompt text: they become the JSON-schema `description` sent
to the model, so a regression test asserts that no internal vocabulary leaks into one.

**Workflow patterns.** Five LangGraph `@entrypoint` pipelines over the functional API: plan,
write, illustrate, narrate, animate. Sequential chains, a parallel fan-out per page, and a
barrier (every character portrait must exist before any page is drawn).

**Multi-agent orchestration.** Six specialised agents with per-agent model configuration —
Researcher, Story Planner, Outline Critic, Plot Planner, Writer, Prose Critic — plus an
Illustration Director and two judges. Each carries its own prompt and can run on its own
model; the critics run at `temperature 0.0` because a critic that answers differently on
identical input turns the loop's stop signal into noise.

**Advanced agentic pattern.** Two evaluator-optimizer loops, one over the outline
and one over the prose. Each runs N revisions and N+1 critiques so the returned draft has
always been judged, and each returns the *best* draft it saw rather than the last — a
revision can be worse than what it replaced, and the loop cannot tell.

**Agentic RAG (ReAct).** A Researcher runs before planning, choosing for itself whether the
premise has anything to get factually wrong and which index to search. It returns
*constraints*, not facts. Retrieval is hybrid, fusing BM25 with vector search by Reciprocal
Rank Fusion at k=60, and its quality is measured: `make test-corpus` reports hit-rate@1 and
@3 over a 19-query labelled set.

**Third-party API integration.** An optional web tool, off by default, over Perplexity with
a Tavily fallback. Because both return URLs the *model* wrote, a claim survives only if
Firecrawl can fetch the page and a judge quotes a supporting sentence that code then finds
in that page.

**MCP resources.** `sparkstory://library` and `sparkstory://corpus`, both read-only and
leak-guarded — run directories are named after the premise and briefs hold a child's name,
so the library reports timestamps and identifiers rather than opening any brief.

**MCP client.** `uv run sparkstory-client` — a `ClientSession` with a REPL, running over
either the in-memory or the stdio transport, with its own tool loop and an inspect mode that
shows what the model *would* call without executing it.

**Human-in-the-loop validation.** A web UI served from the MCP server itself via
`@mcp.custom_route`, so there is no separate frontend application. It implements the
generate-then-approve pattern as a real interaction: the plan is shown, and the book is
only written once the parent approves or asks for a different one. **The server holds the
approved outline**, so the browser sends back a job id rather than an outline — a stronger
guarantee than the MCP path, where the outline is a tool argument a client threads through.
See [The web UI](#the-web-ui).

**Long-term memory.** Two patterns, semantic and episodic, scoped per child over Postgres.
A character established in one book reaches the next book's planner. (Two, not three —
procedural was dropped deliberately, since it would derive guidance from judged scores whose
noise floor exceeds the effects it would claim to detect.)

**Multimodality.** Images generated per page, conditioned on a reference portrait per
character plus a shared style bible so the same character looks the same throughout; a
vision judge that reads an image back and checks it against the portrait it was drawn from;
and text-to-speech narration of the finished book.

**Observability and evaluation.** Opik tracing behind `OPIK_ENABLED`, off by default and
importing nothing when off. The eval harness scores a finished book on ten metrics: seven
computed in code and three LLM-judged, over five committed fixture briefs. An alignment
harness measures judge-versus-human agreement per page and per dimension — and found the
judge agreeing no better than chance, which is why the deterministic metrics carry the
weight here.

**Code quality, CI, Docker, and a database backend.** Pre-commit hooks, `ruff` lint and
format, GitHub Actions, a multi-stage `Dockerfile` and `docker-compose.yml`, and Postgres
with pgvector under Alembic migrations. 1139 offline tests requiring no network, plus 23
marked tests excluded from CI because they need real weights, a vision model, `ffmpeg`, or
a web-search key.

## License

MIT — see [LICENSE](LICENSE).
