# SparkStory MCP server -- container image.
#
#
# The app defaults to stdio transport, which in a container has nothing attached
# to it, so the CMD passes `--transport http` explicitly --

# ============================================================================
# Stage 1: builder -- resolve and install dependencies
# ============================================================================
FROM python:3.14-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Dependency files BEFORE the source. This is the layer-caching decision: a
# change to a .py file then reuses the cached dependency layer instead of
# re-resolving ~70 packages.
#
# README.md and LICENSE are copied despite being unused at runtime, because
# pyproject.toml references both (`readme = "README.md"` and
# `license-files = ["LICENSE"]`) and uv fails to build the project without them.
# Two builds of this image failed on exactly that, once per file.
COPY pyproject.toml uv.lock README.md LICENSE ./

# --frozen: fail if uv.lock disagrees with pyproject.toml, so a stale lock is a
#   build error rather than an image that silently differs from local.
# --no-dev: omit pytest, ruff and pre-commit. The runtime never uses them, and
#   `make ci-local` runs on the host and in CI, not in this image.
RUN uv sync --frozen --no-dev --no-install-project

# The project itself, installed after its dependencies so that editing source
# does not invalidate the dependency layer.
COPY src/ ./src/
RUN uv sync --frozen --no-dev

# ============================================================================
# Stage 2: runtime
# ============================================================================
FROM python:3.14-slim AS runtime

# No apt-get install layer at all, and that is the interesting part:
#
#   * no `libpq5` -- nothing here talks to Postgres until item 6 of the
#     infrastructure spec. Rule 3 applied to a Dockerfile.
#   * no `git` and no git credential configuration. Nothing in SparkStory clones anything.
#   * no system libraries for PDF rendering. Verified rather than assumed:
#     reportlab is pure Python and Pillow ships manylinux wheels with zlib, jpeg
#     and freetype bundled. pyproject.toml records that weasyprint was rejected
#     specifically because it needs Pango/cairo as system libraries and "would
#     break both `uv sync` on a clean clone and Phase B's Docker image".

RUN useradd --create-home --shell /bin/bash sparkstory

WORKDIR /app

COPY --from=builder --chown=sparkstory:sparkstory /app/.venv /app/.venv
COPY --chown=sparkstory:sparkstory src/ /app/src/

# So `sparkstory` resolves to the console script from [project.scripts].
ENV PATH="/app/.venv/bin:$PATH"

# Without this, log lines sit in a buffer and `docker logs` shows nothing during
# a run that takes minutes -- which is indistinguishable from a hang.
ENV PYTHONUNBUFFERED=1

# model2vec downloads the potion-base-8M weights (~59 MB) from HuggingFace on first use
# rather than vendoring them. Left at its default the cache lands in the
# container user's home directory and is lost whenever the container is replaced.
# The failure mode is silent: store.py treats an absent index as "return
# nothing", so research would quietly produce zero facts and the run would still
# look successful. Pointing HF_HOME at the mounted data/ volume means the
# download happens once, on the host, and every container reuses it.
ENV HF_HOME=/app/data/.hf-cache

USER sparkstory

EXPOSE 8000

CMD ["sparkstory", "--transport", "http"]
