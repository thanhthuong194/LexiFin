# Gold: native RAG chunks (`native_rag.chunks`)

One Iceberg table of retrieval chunks built from the Silver narrative by `scripts/native_rag/build_chunks.py`. Source of truth is the code: schema and row checks in `contracts/native_rag/chunks.py`, table spec in `storage/native_rag/chunks.py`, chunking rules in `transforms/native_rag/`, batching and metadata in `pipelines/native_rag/`, audit in `quality/native_rag/audit.py`. Design and decisions: `docs/plans/native_rag_chunking_draft.md`.

Numbers are from `data/` (343 filings, 523 narrative documents), 2026-11-16.

## 1. Storage

- Own catalog: `<gold_dir>/_catalog/gold.db` (PyIceberg `SqlCatalog`, SQLite), files under `<gold_dir>/gold/native_rag/chunks/`. `gold_dir` defaults to the Silver `--data-dir`. Purging Gold never touches the Silver catalog.
- Partition `year(filing_date)`. Grain: one chunk of one narrative document (primary HTML or EX-99 exhibit). Key `chunk_id` = `{document_id}:chunk:{NNNNNN}`.
- Table properties pin what rows were built with; a run whose values differ writes nothing (`rejected`): `lexifin.rules-version`, `lexifin.tokenizer` (`Qwen/Qwen3-Embedding-0.6B@<revision>`), `lexifin.chunk-params` (all tunables), `lexifin.silver-rules` (rules versions of every Silver table read: filings, documents, filing_entities, sections, content_blocks, document_tables, document_table_cells, exhibits, xbrl_instances, xbrl_contexts, xbrl_concepts, xbrl_facts). Change any of them: purge `native_rag.chunks` with `open_gold_catalog`, then build.

## 2. Columns

| Use | Columns |
|---|---|
| Keys, links | `chunk_id`, `filing_id`, `document_id`, `section_id`, `table_id` (table chunks), `block_ids` (Silver blocks whose text the chunk holds, in order) |
| Filter | `cik`, `tickers`, `form`, `fiscal_year`, `fiscal_period` (FY/Q1–Q3; null for 8-K), `period_end_date`, `filing_date`, `section_topic`, `document_type`, `chunk_kind` (text/table) |
| Citation | `company_name`, `accession_number`, `sec_url`, `section_title`, `subheading`, `table_title` |
| Context expansion | `chunk_ordinal` (document order), `document_chunk_count`, `part`/`part_count` (pieces of one paragraph or one table) |
| Content | `context` (prefix), `text` (body) |
| Control | `token_count`, `chunk_sha256`, `processed_at` |

- Embedded string = `context + "\n\n" + text`; `token_count` counts it with the Qwen3-Embedding tokenizer including `<|endoftext|>`; `chunk_sha256` hashes it. Cap 512 tokens.
- `context`: `Company | form | period | section path › subheading[ | Table: title (part i/n)]`. Period: `FY2025 (period ended …)`, `FY2025 Q2 (…)`, 8-K `filed … (event …)`.
- `company_name`: dei `EntityRegistrantName` of non-dimensional contexts (a multi-registrant filing names each subsidiary under `LegalEntityAxis`), one spelling per CIK when names differ only in case/punctuation/suffix: the earliest filing's, mixed case first, so new filings never change it; fallback the filer's conformed name. `tickers`: dei `TradingSymbol` of contexts whose `Security12bTitle` is not debt, `^[A-Z][A-Z.\-]{0,9}$`. `period_end_date`, fiscal fields: dei of non-dimensional contexts, fallback `report_date`.
- `section_topic`: `cover`, `business`, `risk_factors`, `cybersecurity`, `properties`, `legal_proceedings`, `market_equity`, `mdna`, `market_risk`, `financial_statements`, `controls`, `governance`, `exhibits`, `signatures`, `8k_event`, `ex99`, `other`; 10-Q items keyed by parent PART. Known mislabel: financial pages filed under 10-K ITEM 15 are `exhibits`.
- `section_title`: last element of the path; primary-document root = `Cover page`, prose kept from a TOC section = `Front matter`, EX-99 root = `EX-99.1: {Exhibit Index description}`.

## 3. Chunking rules

Structure-aware, deterministic, no LLM. Per document: drop noise → read tables → per run of blocks of one section: text between tables in topic units, each data table its own chunk.

- Noise (counted in `dropped_blocks`): TOC tables and lines ≤150 chars in a TOC section; the block opening a detected section when it is at most 40 chars longer than the title (the title goes to the path; an 8-K item laid out as one table stays); empty; no letters/digits; page numbers (`12`, `Page 3`, `xii`, `F-12`, `23.`); running headers (≤100 chars, not a sentence, same text with digits as `#` ≥4 times in ≥3 sections or ≥8 times); TOC-like tables (no `$`, most rows start with PART/ITEM or are a label of 20+ chars and a page number) and signature tables anywhere.
- Tables: TOC/signature → dropped; ≤2 non-empty columns or no numeric cell → paragraphs (one per row); data table with block text <100 chars → one inline paragraph; otherwise rendered from cells as `label | value | value` with an inferred header (years/dates are labels), `$`/`%`/`)` glued to numbers (lowercase `x`/`bps` only right after a value; uppercase `X` is a check mark), units rows as notes; title = last short paragraph or a lead-in ending in `:` plus up to 3 title-like lines above; up to 5 following footnotes; split by rows with the header repeated (a row over the cap is cut by tokens; parts are checked against their real `(part i/n)` prefix). Known weakness: a very wide table (15 columns, ~400-token header) gives one row per part, mostly header.
- Text: a title-like line (≤120 chars, ≤12 words, no final `.,;:` except a company suffix) opens a unit; unit ≤ cap → one chunk; longer → fewest balanced parts between paragraphs; a paragraph over the cap → balanced parts of sentences (a sentence over a third of the budget cut by tokens), each repeating the previous part's last piece when it is at most a quarter of the budget (rules version 2: v1 let 5 documents of the 500-company Silver reach 513–514 tokens); unit <64 tokens merges into the next unit, else the previous, when it fits; never across sections or tables.

## 4. Operation

- Incremental by document: complete documents (ordinals 1..N = declared count) are skipped; documents with no chunks (image-only exhibits) are re-examined each run; an incomplete stored document rejects the run. Batches by filing date of at most 200 documents or 300k cells, one verified commit each. A document whose chunks fail (contract, cap) is skipped and logged; the run ends `partial` (exit 1) and retries it next time. A layout table's rows may continue in the next chunk; every other block is in exactly one chunk or a split chain.
- Audit (`--check-only`, or after every build): definition and properties, contract on every row, keys, ordinals, token cap, hashes, FKs (document, section, each block in Silver and in the chunk's section), each non-table block in one chunk outside split chains, documents without chunks and >30% of characters outside chunks (WARN), exact rebuild of the newest 10-K/10-Q/8-K plus 5 documents by hash with full coverage (every block in a chunk or dropped, not both). Reads one filing year at a time.
- `data/` (2026-11-16): build 80 s, peak RSS ~0.7 GB, rerun 0.8 s; 26,658 chunks, tokens mean 287, max 512; audit PASS (17 checks).
