# LexiFin

Hiện chỉ có SEC 10-K/10-Q/8-K của top-N công ty trong quỹ IVV (S&P 500). Không được phá: tính bất biến của Bronze và `filing_manifest.jsonl`, vì manifest quyết định filing nào được tải lại.

## Commands

- Cài: `uv sync --locked` (Python 3.13; `vnstock`/`vnai` lấy từ index riêng `vnstocks.com`).
- Test toàn bộ: `uv run pytest` (17 test, ~2 s, không gọi mạng; coverage bật sẵn trong `addopts`).
- Một test: `uv run pytest tests/unit/utils/test_timer.py::<tên_test> --no-cov`.
- Chạy ingest SEC (entry point duy nhất):
  `uv run python scripts/ingest_sec_edgar.py --data-dir <thư_mục_tạm> --company-limit 1 --lookback-years 1`
  Cần `SEC_COMPANY_NAME`, `SEC_EMAIL` trong `.env` (mẫu: `.env.example`). Bỏ `--data-dir` thì ghi vào `./data`.

## Domain terms

Tên dễ hiểu nhầm:

- universe: top-N theo tổng `Weight (%)` của IVV, gộp theo CIK (GOOGL + GOOG tính là một công ty). Đây không phải danh sách S&P 500 chính thức.
- `company_limit` (N, 1–500): state incremental được tách theo N. `load_usable_runs` chỉ đọc các run có cùng N, nên khi đổi N mọi công ty bị coi là `new` và quét lại toàn bộ `lookback_years`. Filing đã có trong manifest vẫn được bỏ qua, nhưng vẫn phải gọi SEC để liệt kê.
- `snapshot_id`: hash SHA-256 của hai file nguồn, không phải ngày. Thư mục `snapshot_date_<ts>` mang thời điểm bắt đầu run (UTC), không phải `holdings_as_of_date`.
- `cik`: chuỗi 10 chữ số, có số 0 ở đầu (`0000320193`), không phải int.
- `status="partial"`: có filing bị lỗi, nhưng script vẫn trả exit 0. `exited` chỉ được ghi ở run ngay sau khi công ty rời universe.
- `connectors/sec_filings/` chứa cả `ishares.py`, dù iShares không phải nguồn SEC.
- Marker `integration` không có nghĩa là test gọi mạng; các test này dùng fetcher/downloader giả. Test trong `tests/unit/utils/` không gắn marker, nên chạy `-m unit` sẽ bỏ sót chúng.
- File rỗng, không phải entry point: `main.py`, `Makefile`, `Dockerfile`, `src/data_pipeline/runner.py`, `connectors/registry.py`, `pipelines/{bronze,silver,gold}.py`, `contracts/{silver,gold}.py`, `quality/{silver,gold}.py`.
- `bronze-silver.md` gọi các bảng là "Delta table", nhưng dependency lại là `pyiceberg` + `duckdb`. Silver và Gold chưa có code. [?]

## Do not touch

- `data/` (gitignored; khoảng 1,7 GB, 343 filing): không sửa hay xoá bằng tay.
  - `filing_manifest.jsonl` chỉ được append. Xoá file filing thì lần chạy sau sẽ tải lại.
  - `_metadata/runs/run_*.json` là state incremental. Một file hỏng làm mọi run sau báo `Invalid SEC run metadata`.
  - Test và chạy thử thì dùng `tmp_path` hoặc `--data-dir` tạm.
- `.env`: chứa tên và email thật, được gửi tới SEC trong User-Agent. Không in ra, không commit.
- `uv.lock`: chỉ thay đổi qua `uv add` / `uv lock`.
- `infrastructure/gcp_vm/.terraform.lock.hcl`: sinh ra từ `terraform init`.
- `src/data_pipeline/erd/conceptual_erd/*.{png,svg,html}`: xuất từ `conceptual_erd.drawio`, nên sửa file `.drawio` rồi xuất lại. [?] Không rõ `Untitled Diagram.drawio` dùng để làm gì. [?]
- `logs/lexifin.jsonl`: log JSON của các run trước. Code chỉ ghi ra stdout, nên file này chắc được tạo bằng redirect. [?]

## Known traps

Những thứ nguy hiểm hoặc tốn tiền khi chạy:

- Ingest gọi SEC EDGAR và iShares thật, và tốn dung lượng: 5 công ty × 5 năm ≈ 1,7 GB, mất khoảng 8 phút (theo `logs/`). Ước lượng 500 công ty ≈ 170 GB, vượt đĩa 50 GB của VM Terraform. [?] Luôn thử với N=1, 1 năm, và `--data-dir` tạm.
- `sec-edgar-downloader` gọi `requests.get` không có timeout, nên có thể treo vô hạn. Ngày 2026-10-04 đã gặp: kẹt ở SYN-SENT qua IPv6 tới sec.gov. Khi chạy hãy đặt timeout từ bên ngoài.
- Không chạy hai tiến trình ingest cùng `--data-dir`: không có lock, và cả hai cùng append vào manifest. Rate limiter 10 req/s của SEC chỉ áp dụng trong một process. Vượt rate có thể bị SEC chặn IP. [?]
- `terraform apply` trong `infrastructure/gcp_vm/` sẽ tạo VM `e2-standard-4` với đĩa 50 GB ở `asia-southeast1`, tính tiền liên tục cho tới khi destroy. Không chạy `apply`/`destroy` khi chưa được yêu cầu. tfstate bị gitignore nên state chỉ nằm ở máy người đã chạy. [?]
- `my_ip` mặc định là `0.0.0.0/0`, nghĩa là firewall mở các port 22, 6333, 7474, 7687, 8501 ra toàn Internet.
- `docker-compose.yml` hard-code mật khẩu Neo4j và dùng image `qdrant:latest`. Dữ liệu ghi vào `qdrant_data/`, `neo4j_data/` cạnh file compose; hai thư mục này không bị gitignore nên `git add -A` sẽ add cả dữ liệu DB.
