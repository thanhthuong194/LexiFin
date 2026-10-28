# Shortcuts for the commands in README.md; the CI workflow runs `make check`'s steps.
# Silver targets need DATA_DIR (Silver root) and take BRONZE_DIR (default:
# $(DATA_DIR)/bronze/sec_filings) and ARGS (more script options), e.g.
# `make silver DATA_DIR=data ARGS=--allow-taxonomy-network`.

.PHONY: install format lint types dead-code duplication test check silver audit

DATA_DIR ?=
ARGS ?=
BRONZE_DIR ?= $(DATA_DIR)/bronze/sec_filings
SILVER_ARGS = --data-dir $(DATA_DIR) --bronze-dir $(BRONZE_DIR)

install:
	uv sync --locked

format:
	uv run ruff format .

lint:
	uv run ruff check .

types:
	uv run mypy src

dead-code:
	uv run vulture src vulture_whitelist.py

duplication:
	npx --yes jscpd@4 --format python --gitignore --min-tokens 60 --threshold 3 \
		--ignore "**/tests/**,**/.venv/**" src scripts

test:
	uv run pytest

check: lint types dead-code duplication test
	uv run ruff format --check .

silver: _data_dir
	uv run python scripts/sec_filings/build_silver.py $(SILVER_ARGS) $(ARGS)

audit: _data_dir
	uv run python scripts/sec_filings/audit_silver.py $(SILVER_ARGS) $(ARGS)

.PHONY: _data_dir
_data_dir:
	@test -n "$(DATA_DIR)" || { echo "set DATA_DIR, e.g. make $(MAKECMDGOALS) DATA_DIR=data" >&2; exit 2; }
