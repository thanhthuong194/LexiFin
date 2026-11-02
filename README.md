# LexiFin

Data platform for financial filings. It downloads SEC 10-K/10-Q/8-K filings of the top-N companies in the IVV ETF (S&P 500) into an immutable **Bronze** layer, then builds a **Silver** layer of 19 Apache Iceberg tables with row contracts, byte-level lineage back to Bronze, and an integrity gate over all data. Gold (RAG, GraphRAG, ML) builds on Silver.

## Architecture

```
SEC EDGAR + iShares ──ingest──▶ Bronze (source files + append-only manifest)
                                   │
                                   ▼  build_silver.py: 12 stages, batched, resumable
                                Silver (19 Iceberg tables, SQLite catalog)
                                   │   filings → documents → filing_entities → narrative → tables
                                   │   → links → exhibit index → XBRL core → facts → enrichment
                                   │   → linkbases → integrity (read-only audit of all 19 tables)
                                   ▼
                                Gold (in progress)
```

- **Idempotent, incremental:** each stage writes only missing groups and skips complete ones; rerunning a command resumes. No global rollback: a failed batch writes nothing, earlier batches stay committed.
- **Contracts:** every row is validated by a pydantic model (`src/data_pipeline/contracts/`) before writing; the audit revalidates every row plus keys, 32 FKs, lineage and counts.
- **Bounded memory:** reads and writes in batches, never rereads whole tables; a 500-company build runs in 12 GB.
- **XBRL** is loaded with Arelle including the DTS, in parallel processes (`--xbrl-workers`), from an offline taxonomy cache.

Data model (grain, keys, relations, lineage): [`docs/data/sec_filings_silver_model.md`](docs/data/sec_filings_silver_model.md). Operations, measurements and known traps: [`AGENTS.md`](AGENTS.md).

## Setup

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
make install            # uv sync --locked
cp .env.example .env    # SEC requires a name and email in the User-Agent
```

## Run

| Task | Command |
| --- | --- |
| Download Bronze (calls SEC) | `uv run python scripts/ingest_sec_edgar.py --data-dir <dir> --company-limit 1 --lookback-years 1` |
| Build Silver | `make silver DATA_DIR=<silver> BRONZE_DIR=<bronze>` (first run: add `ARGS=--allow-taxonomy-network` to fill the taxonomy cache) |
| Audit Silver (read-only) | `make audit DATA_DIR=<silver> BRONZE_DIR=<bronze>` |
| Large Bronze (warm cache → benchmark → build, RAM-capped) | `scripts/sec_filings/run_silver.sh all <bronze> <out>` |
| Query Silver with SQL | `uv run python scripts/sec_filings/query_silver.py scripts/sec_filings/queries/item_1a.sql --data-dir <silver>` |

## Development

```bash
make check   # ruff format + lint, mypy strict, vulture, jscpd, pytest (~9 min, offline)
```

CI (`.github/workflows/ci.yml`) runs the same steps on every push to `main` and every pull request.

## Layout

```
src/data_pipeline/
  connectors/   external sources (SEC EDGAR, iShares)
  contracts/    per-table schema and validation
  transforms/   pure parsing (HTML, SEC header, XBRL)
  pipelines/    Bronze and Silver stages (read → build → validate → write in batches)
  storage/      Bronze, Iceberg, table specs
  quality/      Bronze checks, Silver audit
scripts/        CLI entry points
tests/          unit + integration (offline, fake SEC fixtures)
```
