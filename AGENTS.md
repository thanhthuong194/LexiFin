# LexiFin

Hiện chỉ có SEC 10-K/10-Q/8-K của top-N công ty trong quỹ IVV (S&P 500). Không được phá: tính bất biến của Bronze và `filing_manifest.jsonl`, vì manifest quyết định filing nào được tải lại.

## Commands

- Cài: `uv sync --locked` (Python 3.13; `vnstock`/`vnai` lấy từ index riêng `vnstocks.com`).
- Test toàn bộ: `uv run pytest` (260 test, ~75 s, không gọi mạng; coverage bật sẵn trong `addopts`).
- Một test: `uv run pytest tests/unit/utils/test_timer.py::<tên_test> --no-cov`.
- Chạy ingest SEC (Bronze):
  `uv run python scripts/ingest_sec_edgar.py --data-dir <thư_mục_tạm> --company-limit 1 --lookback-years 1`
  Cần `SEC_COMPANY_NAME`, `SEC_EMAIL` trong `.env` (mẫu: `.env.example`). Bỏ `--data-dir` thì ghi vào `./data`.
- Dựng Silver (Iceberg) từ toàn bộ SEC Bronze, không gọi mạng, theo thứ tự stage `sec_filings.filings` → `sec_filings.documents` (mỗi `<DOCUMENT>` block của `full-submission.txt` là một row) → `sec_filings.filing_entities` (mỗi entity-role block như `FILER:` trong SEC header là một row) → narrative (`sec_filings.sections` + `sec_filings.content_blocks`, dựng cùng lúc từ một lần parse primary HTML của bảng documents) → tables (`sec_filings.document_tables` + `sec_filings.document_table_cells`, một bảng cho mỗi block `table`) → links (`sec_filings.document_links`, mỗi `<a href>` của mọi document HTML trong bảng documents là một row) → XBRL core (`sec_filings.xbrl_instances` + `xbrl_contexts` + `xbrl_units`, một lần parse mỗi nguồn Inline XBRL/XML instance; chỉ phụ thuộc filings và documents). Các stage sau đọc parent từ bảng Silver cha, không quét lại Bronze; `bronze_to_silver.py` chỉ điều phối, mỗi stage nằm trong `pipelines/sec_filings/<stage>.py`:
  `uv run python scripts/sec_filings/build_silver.py --data-dir <data_dir>`
  Dựng từ đầu cho 343 filing ~3 phút 15 s (ước tính: ~2 phút 40 s đo trước khi có links, cộng 36 s của links), RSS tối đa ~6 GB (narrative 33 s → 2.803 section, 83.301 block; tables 109 s → 10.184 bảng, 878.380 cell; links 36 s → 8.881 document HTML, 365.221 link; XBRL core 23 s → 343 instance, 30.319 context, 699 unit; vì mỗi stage validate cả batch trong bộ nhớ trước lần ghi đầu). Chạy lại ~46 s, RSS ~4,3 GB: mọi group đã có đều được skip, nhưng narrative, tables và links vẫn đọc lại và kiểm tra group đã lưu; riêng links parse lại 914 document không có link (không có row để đánh dấu đã xử lý); XBRL core skip filing đã đủ mà không đọc Bronze (filing không có XBRL thì parse lại).
  Exit 1 khi một stage bị reject (filing lỗi, checksum conflict, document/entity/narrative/table/link group thiếu hoặc khác); stage đó không ghi gì và các stage sau không chạy. Narrative ghi sections trước blocks, tables ghi document_tables trước cells, XBRL core ghi instances → contexts → units: nếu commit thứ hai lỗi, lần chạy sau resume khi phần đã lưu khớp chính xác.
- Audit Silver chỉ đọc (không ghi, không tạo bảng, không đổi snapshot), không nằm trong build:
  `uv run python scripts/sec_filings/validate_silver.py --data-dir <data_dir> [--verify-source-hashes]`
  Kiểm schema/partition/location theo contract, khoá chính, FK + lineage, contract từng row, độ đầy đủ của từng group (dùng lại các `*_problem` của stage) và semantics bảng; `--verify-source-hashes` hash lại Bronze (submission, primary document, span của document và entity). Exit 0 = pass (có thể có warning), 1 = có error, 2 = không chạy được (thiếu catalog...). Trên `data/`: ~27 s, ~34 s khi verify hash, RSS ~2,2 GB; warning còn lại là 149 `public_document_count`, link unresolved, và 3 bảng XBRL chưa được audit (`unexpected_table`).

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
- `bronze-silver.md` gọi các bảng là "Delta table", nhưng dependency lại là `pyiceberg` + `duckdb`. Silver mới có các bảng Iceberg `sec_filings.filings`, `sec_filings.documents`, `sec_filings.filing_entities` (không partition), `sec_filings.sections`, `sec_filings.content_blocks`, `sec_filings.document_tables`, `sec_filings.document_table_cells` và `sec_filings.document_links`; Gold chưa có code.
- Mỗi `<table>` có ít nhất một cell nhìn thấy được là đúng một content block `block_type="table"` (`source_tag="table"`, text = các cell khác rỗng nối bằng `\n`, có thể rỗng); cell không còn là content block. Bảng lồng có cell là block riêng; bảng lồng không có cell thì text của nó thuộc cell bao ngoài. Heading nằm trong bảng thì cả block `table` là heading; `title` của section là text đã normalize (cell nối bằng khoảng trắng, không có `\n`). Nếu `content_blocks` còn row `table_cell` của layout cũ, narrative log `narrative_table_semantics_incompatible`, reject (exit 1) và không ghi gì cho tới khi rebuild có kiểm soát `sections` + `content_blocks`. Rebuild như vậy đã được làm cho `data/` ngày 2026-10-06 (drop/purge hai bảng qua catalog PyIceberg, rồi chạy lại build).
- Bảng SEC trong Bronze không dùng `<th>`, `<caption>` hay `thead`/`tfoot` (mọi cell là `td`, row group `body`) và ~69% cell rỗng (cell đệm); `colspan > 1` rất phổ biến. Cell ẩn (44.472 `<td style="display:none">`, đều rỗng, ở 93 filing) không được render nên không được lưu: `cell_count`/`row_count` chỉ đếm cell/row nhìn thấy được, nên khác số `<td>` đếm trong HTML thô.
- `content_blocks.source_xpath` là `getpath()` của DOM đã parse ở HTML mode, nên đoạn Inline XBRL giữ nguyên tên tag có tiền tố (`/html/body/ix:nonnumeric[1]/p[1]`, ~44% block). XPath chuẩn coi `ix:` là namespace prefix: muốn evaluate thì viết đoạn đó thành `*[name()='ix:nonnumeric'][1]`.
- Narrative không tự phát hiện thay đổi logic parser: group đã có được skip theo source hash. Đổi parser thì phải rebuild có kiểm soát hai bảng narrative.
- `document_links`: `source_document_id` và `target_document_id` đều trỏ tới `documents.document_id` (exhibit là một document, không có `target_exhibit_id`). Target chỉ resolve trong cùng filing theo đúng filename; SEC URL nằm dưới thư mục accession của filing khác luôn `unresolved` dù trùng tên file. Trong `data/`: ~97% row là `javascript:void(0)` của trang R do SEC sinh (`role=generated`, `not_applicable`); 861 exhibit HTML không có link; 1.154 SEC URL đều trỏ tới hồ sơ cũ (exhibit dẫn chiếu) nên `unresolved`; không có external URL.
- XBRL core: EDGAR bọc iXBRL `.htm` trong `<XBRL>` và `.xml` trong `<XML>`; bóc lớp vỏ SGML đó rồi mới parse XML (namespace-aware, không DTD/entity/network). Mỗi filing trong `data/` có đúng 1 Inline target mặc định (không named target, không `ix:tuple`) + 1 XML do EDGAR sinh (`TYPE=XML`, `*_htm.xml`); XML được gộp vào target (`source_document_ids` gồm cả hai) khi khớp schema ref, multiset (concept, contextRef, unitRef) và mọi context/unit của nó — XML của EDGAR bỏ unit không fact nào dùng nên Inline có thể nhiều unit hơn. `namespace_map` chỉ lấy từ nguồn canonical (Inline), vì prefix `""` của XML là `xbrli`. `segment_xml`/`scenario_xml` là XML canonical tự định nghĩa (prefix `ns0…` theo URI, QName trong `dimension`/`explicitMember` viết lại, bỏ khoảng trắng giữa element, member xếp theo dimension, rồi C14N) để hai biểu diễn cho cùng `context_hash`. Ba bảng không partition và không lặp `filing_id`/`filing_date` ở bảng con.
- `public_document_count` (header `PUBLIC DOCUMENT COUNT`) không đáng tin: filing nộp từ 2024-01-30 đến 2026-03-13 có header lớn hơn số `<DOCUMENT>` block thật 1–2 (149/343 filing). Documents stage chỉ log `submission_document_count_mismatch`, không fail.

## Do not touch

- `data/` (gitignored; khoảng 1,7 GB, 343 filing): không sửa hay xoá bằng tay.
  - `filing_manifest.jsonl` chỉ được append. Xoá file filing thì lần chạy sau sẽ tải lại.
  - `_metadata/runs/run_*.json` là state incremental. Một file hỏng làm mọi run sau báo `Invalid SEC run metadata`.
  - `_catalog/silver.db` (SQLite catalog) và `silver/` do PyIceberg quản lý: không sửa metadata hay Parquet bằng tay. Catalog lưu đường dẫn tuyệt đối, nên di chuyển `data/` sẽ làm catalog trỏ sai.
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
