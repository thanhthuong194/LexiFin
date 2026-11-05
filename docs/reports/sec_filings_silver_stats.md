# SEC filings: Bronze and Silver statistics (2026-11-03)

Scope: 10-K, 10-Q, 8-K of the top-N IVV companies, lookback 5 years (filing dates 2021-09 → 2026-09). Data lives on `homepc` under `/media/bee/DATA/02_projects/LexiFin/data`. The 500-company Silver is the main dataset; smaller tiers were benchmark runs (`scripts/sec_filings/benchmark_silver.py`, reports in `data/scale/<N>/report.md`).

Sources: Bronze `filing_manifest.jsonl` per tier; Iceberg snapshot summaries and DuckDB queries over the 500-company Silver (read-only); benchmark reports. Machine: 12 CPU, 15.5 GiB RAM, exFAT USB drive.

## 1. Overview by tier

| Companies | Filings | 10-K / 10-Q / 8-K | Bronze (GB, manifest) | Silver rows | Build | CPU | Peak RAM (GiB) | Filings/min | Rerun | Integrity |
|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 343 | 25 / 75 / 243 | 1.8 | 2,216,683 | 10.0 min | 18.3 min | 2.6 | 34.2 | 57 s | 0 FAIL |
| 50 | 3,850 | 239 / 719 / 2,892 | 24.2 | 28,164,309 | 2.7 h | 3.6 h | 6.3 | 23.8 | 9.8 min | 0 FAIL |
| 100 | 7,977 | 486 / 1,460 / 6,031 | 50.3 | 57,277,190 | 5.8 h | 7.6 h | 7.4 | 23.0 | 29.7 min | 0 FAIL |
| 200 | 16,141 | 984 / 2,957 / 12,200 | 103.0 | 113,261,183 | 12.3 h | 15.4 h | 6.5 | 21.9 | 1.2 h | 0 FAIL |
| **500** | **40,819** | **2,445 / 7,344 / 31,034** | **259.6** | **268,991,711** | ~41 h (resumed runs) | — | 7.3 | ~17 | — | **0 FAIL** |

- Rows and time grow linearly with filings (~6,600–7,000 Silver rows per filing; ~22 filings/min from 50 companies up). Peak RAM stays at 6–7.5 GiB because stages commit in batches.
- 500 companies: 40,823 manifest filings (1 failed download skipped). 4 accessions are joint filings stored under two CIKs and kept once, so 40,819 filings in Silver.
- The 500 build ran in 16 resumed processes while parser fixes landed (e.g. malformed XBRL, redefined extension concepts); the final integrity run took 18.4 h with the old SQLite scratch (now DuckDB).
- Tier code versions: 5/50/100 at `e1b39e4`+changes, 200 at `ef316fa`; 500 before `ef316fa`. Later commits do not change rows (fixture digests identical).

Filings per year (500 companies; 2021 and 2026 are partial years):

| Form | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|
| 10-K | 42 | 481 | 483 | 488 | 496 | 455 |
| 10-Q | 442 | 1,447 | 1,454 | 1,474 | 1,494 | 1,033 |
| 8-K | 1,549 | 6,109 | 6,084 | 6,143 | 6,347 | 4,798 |

## 2. Silver tables (500 companies)

| Table | Rows | Data files | Size (GB) |
|---|---|---|---|
| filings | 40,819 | 6 | 0.00 |
| documents | 1,610,127 | 87 | 0.11 |
| filing_entities | 45,039 | 1 | 0.01 |
| sections | 298,641 | 210 | 0.02 |
| content_blocks | 10,331,785 | 210 | 1.10 |
| document_tables | 1,255,505 | 590 | 0.06 |
| document_table_cells | 115,469,043 | 590 | 1.39 |
| document_links | 46,899,422 | 822 | 0.34 |
| exhibits | 243,188 | 338 | 0.02 |
| xbrl_instances | 40,763 | 205 | 0.00 |
| xbrl_contexts | 5,124,254 | 205 | 0.40 |
| xbrl_units | 102,417 | 205 | 0.00 |
| xbrl_concepts | 1,602,549 | 408 | 0.13 |
| xbrl_facts | 18,682,391 | 12,521 | 3.07 |
| xbrl_dimensions | 9,412,779 | 230 | 0.37 |
| xbrl_fact_footnotes | 296,960 | 214 | 0.01 |
| xbrl_labels | 17,902,707 | 408 | 1.69 |
| xbrl_networks | 1,570,598 | 408 | 0.10 |
| xbrl_relationships | 38,062,724 | 408 | 3.07 |
| **Total** | **268,991,711** | **18,066** | **11.9** |

Parquet data is 11.9 GB; the `silver/` folder is 19 GB on disk because of Iceberg metadata/manifests and exFAT cluster overhead. Silver is ~5% of Bronze.

## 3. Content

**Documents** (1,610,127 `<DOCUMENT>` blocks, 226 GB of content in Bronze):

| Role | Count | Content (GB) | In Silver |
|---|---|---|---|
| graphic (jpg/gif) | 205,999 | 48.2 | metadata only |
| generated (SEC R pages `.htm`) | 831,295 | 45.0 | links only |
| primary_document (10-K/10-Q/8-K) | 40,819 | 33.5 | sections, blocks, tables, cells, links |
| xbrl_instance | 40,764 | 33.0 | XBRL tables |
| exhibit (html) | 103,365 | 16.1 | links, exhibit index; text not parsed |
| generated binary (zip/xlsx) | 70,567 | 13.0 | metadata only |
| generated json | 40,764 | 11.6 | metadata only |
| XBRL linkbases + schema | 153,770 | 23.2 | via Arelle DTS |
| other (generated xml/text, binary exhibits) | 122,784 | 2.5 | metadata only |

EX-99 exhibits (press releases, investor material): 25,373 documents, 9.9 GB. Their text is not in Silver yet.

**Narrative** (primary documents):

- Sections: 183,989 items, 23,066 parts, 40,010 signatures, 6,447 tables of contents, 4,310 exhibit indexes, plus 40,819 document roots.
- Content blocks: 8,849,842 paragraphs (2.06 G characters), 1,255,505 tables (549 M characters), 226,432 headings.
- Tables: 1,255,505 tables, 115,469,043 visible cells (~92 per table).
- Links: 46,899,422; exhibit index entries: 243,188.

**XBRL**:

- Instances: 40,763 (56 filings have no XBRL).
- Facts: 18,682,391, of which 16,462,857 numeric and 2,219,534 non-numeric.
  - 16,352,861 have a numeric value; 279,295 are hidden.
  - 626,452 (3.4%) keep only `raw_value`: `ixt-sec` format, no Arelle registry.
  - Fact → HTML location: 15,414,899 to a table cell (83%), 2,321,051 to a block.
- Contexts: 5,124,254; units: 102,417; dimension members: 9,412,779; fact footnotes: 296,960.
- Concepts: 1,602,549, of which 1,439,998 are filer extensions and 162,551 standard taxonomy.
- Labels: 17,902,707; networks: 1,570,598; relationships: 38,062,724.

## 4. Quality (500 companies, integrity 2026-10-26)

266 checks over all 268,991,711 rows: **261 PASS, 4 WARN, 0 FAIL, 1 SKIP** (`source_hashes`, opt-in). No snapshot changed during the audit.

WARNs are properties of SEC data, not defects:

| Check | Count | Cause |
|---|---|---|
| `public_document_count` | 17,854 filings | SEC header overcounts `<DOCUMENT>` blocks |
| `resolved_links` | 270,771 links | links to older filings (incorporated by reference) |
| `repeated_section_code` | 368 sections | filings repeat an item heading under one part |
| `root_only_document` | 51 documents | primary document with no detectable item headings |

## 5. Known gaps before Gold / RAG

- EX-99 exhibit text (25,373 documents) not parsed.
- No `companies` dimension table (name, ticker, sector).
- 3.4% of facts (`ixt-sec`) have no normalized value.
- The current code (rules versions, DuckDB audit) has not yet re-audited the 500-company Silver: run `build_silver.py --check-only` after `git pull` on `homepc`.
