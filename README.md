# LexiFin

Data platform cho hồ sơ tài chính: tải hồ sơ SEC 10-K/10-Q/8-K của top-N công ty trong quỹ IVV (S&P 500) về lớp **Bronze** (file gốc, bất biến), rồi dựng lớp **Silver** gồm 19 bảng Apache Iceberg có contract, lineage về tận byte của file Bronze và một integrity gate kiểm toàn bộ dữ liệu. Gold (RAG, GraphRAG, ML) dựng trên Silver.

## Kiến trúc

```
SEC EDGAR + iShares ──ingest──▶ Bronze (file gốc + manifest chỉ append)
                                   │
                                   ▼  build_silver.py: 12 stage, theo lô, chạy lại được
                                Silver (19 bảng Iceberg, catalog SQLite)
                                   │   filings → documents → filing_entities → narrative → tables
                                   │   → links → exhibit index → XBRL core → facts → enrichment
                                   │   → linkbases → integrity (audit chỉ đọc cả 19 bảng)
                                   ▼
                                Gold (đang làm)
```

- **Idempotent và incremental:** mỗi stage chỉ ghi group chưa có, skip group đã đủ; chạy lại lệnh là tiếp tục từ chỗ dừng. Không có global rollback: lô lỗi không ghi gì, các lô trước đã commit.
- **Contract:** mỗi row được validate bằng model pydantic (`src/data_pipeline/contracts/`) trước khi ghi; audit validate lại từng row cùng khoá, 32 FK, lineage và count.
- **Bộ nhớ có giới hạn:** đọc/ghi theo lô, không đọc lại cả bảng; build 500 công ty chạy trong 12 GB.
- **XBRL** được nạp bằng Arelle cùng DTS, song song nhiều process (`--xbrl-workers`), taxonomy lấy từ cache offline.

Model dữ liệu (grain, khoá, quan hệ, lineage): [`docs/data/sec_filings_silver_model.md`](docs/data/sec_filings_silver_model.md). Chi tiết vận hành, số đo và các bẫy đã gặp: [`AGENTS.md`](AGENTS.md).

## Cài đặt

Cần Python 3.13 và [uv](https://docs.astral.sh/uv/).

```bash
make install            # uv sync --locked
cp .env.example .env    # SEC yêu cầu tên + email trong User-Agent
```

## Chạy

| Việc | Lệnh |
| --- | --- |
| Tải Bronze (gọi SEC) | `uv run python scripts/ingest_sec_edgar.py --data-dir <dir> --company-limit 1 --lookback-years 1` |
| Dựng Silver | `make silver DATA_DIR=<silver> BRONZE_DIR=<bronze>` (lần đầu thêm `ARGS=--allow-taxonomy-network` để tải taxonomy vào cache) |
| Audit Silver (chỉ đọc) | `make audit DATA_DIR=<silver> BRONZE_DIR=<bronze>` |
| Bronze lớn (warm cache → benchmark → build, giới hạn RAM) | `scripts/sec_filings/run_silver.sh all <bronze> <out>` |
| Truy vấn Silver bằng SQL | `uv run python scripts/sec_filings/query_silver.py scripts/sec_filings/queries/item_1a.sql --data-dir <silver>` |

## Phát triển

```bash
make check   # ruff format + lint, mypy strict, vulture, jscpd, pytest (~9 phút, không gọi mạng)
```

CI (`.github/workflows/ci.yml`) chạy cùng các bước đó cho mỗi push lên `main` và mỗi pull request.

## Cấu trúc

```
src/data_pipeline/
  connectors/   nguồn ngoài (SEC EDGAR, iShares)
  contracts/    schema + validate của từng bảng
  transforms/   parse thuần (HTML, SEC header, XBRL)
  pipelines/    stage Bronze và Silver (đọc → dựng → validate → ghi theo lô)
  storage/      Bronze, Iceberg, định nghĩa bảng
  quality/      kiểm tra Bronze, audit Silver
scripts/        entry point CLI
tests/          unit + integration (offline, fixture giả lập SEC)
```
