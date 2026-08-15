# SparkStory developer commands.
#
#
# Every target is declared .PHONY. Without it, make
# treats a target as a file recipe and silently does nothing if a file or
# directory of the same name exists -- `clean`, `install` and `test` are the
# classic collisions, and the failure is confusing because make reports success.

RUFF   := uv run ruff
PYTEST := uv run pytest

# Everything ruff should look at. Kept in one variable so a new top-level
# directory is added once rather than in four targets.
#
# `alembic` is here because pre-commit lints whatever is *staged*, regardless of
# this variable -- so a migration could pass `make ci-local`, pass CI, and then be
# reformatted by the commit hook. The hook reporting "failed" for files it just
# fixed is confusing enough without the gap that caused it.
QA_PATHS := src tests scripts alembic

.DEFAULT_GOAL := help

.PHONY: help install hooks format-fix lint-fix format-check lint-check \
        fix check test test-fast test-corpus test-vision score-books \
        label-skeletons score-alignment migrate \
        migrate-down ingest \
        run ci-local clean

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

# --- Setup ---------------------------------------------------------------

install: ## Install dependencies and git hooks
	uv sync
	$(MAKE) hooks

hooks: ## Install the pre-commit git hooks
	uv run pre-commit install

# --- Quality: fix ---------------------------------------------------------

format-fix: ## Reformat code in place
	$(RUFF) format $(QA_PATHS)

lint-fix: ## Fix auto-fixable lint violations
	$(RUFF) check --fix $(QA_PATHS)

fix: lint-fix format-fix ## Fix lint then reformat
	@echo "Order matters: --fix can leave code the formatter still wants to reflow."

# --- Quality: check (non-mutating, matches CI) ----------------------------

format-check: ## Verify formatting without changing files
	$(RUFF) format --check $(QA_PATHS)

lint-check: ## Report lint violations without fixing
	$(RUFF) check $(QA_PATHS)

check: format-check lint-check ## Run all non-mutating quality checks

# --- Tests ---------------------------------------------------------------

test: ## Run the test suite
	$(PYTEST)

test-fast: ## Run tests, stopping at the first failure
	$(PYTEST) -x -q


test-corpus: ## Run retrieval quality tests (needs a built index)
	$(PYTEST) -m corpus || [ $$? -eq 5 ]


test-vision: ## Judge the known drifts in committed runs (needs XAI_API_KEY)
	$(PYTEST) -m vision || [ $$? -eq 5 ]


score-books: ## Score every past run on the computed metrics (free, no network)
	uv run python scripts/run_evals.py --from-run outputs --all --no-judge

label-skeletons: ## Write blank label files for the baseline books (offline)
	uv run python scripts/make_label_skeletons.py \
		--books tests/fixtures/evals/baseline/2026-08-04 \
		--out tests/fixtures/evals/labels/2026-08-13

score-alignment: ## Measure judge-vs-human agreement (free, offline, needs labels)
	uv run python scripts/score_alignment.py \
		--labels tests/fixtures/evals/labels/2026-08-13 \
		--scorecards tests/fixtures/evals/baseline/2026-08-13

# --- Database ------------------------------------------------------------
# Both targets need DATABASE_URL and a reachable Postgres:
#   docker compose up -d postgres

migrate: ## Apply database migrations (needs DATABASE_URL)
	uv run alembic upgrade head

migrate-down: ## Roll the last migration back, for checking one reverses cleanly
	uv run alembic downgrade -1

ingest: ## Load corpus/ into Postgres (offline; run after `make migrate`)
	uv run python scripts/ingest_knowledge.py

# --- Run -----------------------------------------------------------------

run: ## Start the MCP server over stdio (Ctrl-C to stop)
	uv run sparkstory

# --- Composite -----------------------------------------------------------

ci-local: check test ## Everything CI will run, before you push

# --- Housekeeping --------------------------------------------------------

clean: ## Remove caches and build artefacts (leaves .venv and data/ alone)
	rm -rf dist .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} +
