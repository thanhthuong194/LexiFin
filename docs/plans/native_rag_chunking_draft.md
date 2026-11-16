# Native RAG: chunking spec

Status 2026-11-16: implemented (as-built reference: `docs/data/native_rag_model.md`). Scope is chunking only (Silver -> `native_rag.chunks`). Embedding, Qdrant, retrieval, eval and GraphRAG come later; LLM-as-judge is built in a separate session. Teacher-facing summary: section B4 of `docs/reports/silver_improvements_report.html`.

## 1. Decisions

- Method: structure-aware chunking + deterministic context prefix. No LLM, no embedding model at chunk time.
- Contextual retrieval (LLM-written context), propositions, embedding-based semantic chunking: later experiments only, measured against this method; LLM context would go to a separate `chunk_contexts` table so `chunks` stays deterministic.
- Gold is organized by use case: namespace `native_rag`, data under `data/gold/native_rag/`.
- Separate Iceberg catalog `data/_catalog/gold.db` (PyIceberg `SqlCatalog` on SQLite, like Silver); purge/rebuild of Gold never touches the Silver catalog. DuckDB only reads.
- Token counts use the Qwen3-Embedding-0.6B tokenizer (`tokenizers` library + `tokenizer.json` only), +1 for the EOS token the embedder appends.

## 2. Data facts (local `data/`, 5 companies, 343 filings, 2026-11-12)

| Fact | Value | Consequence |
|---|---|---|
| 10-K/10-Q item length | p50 450–1,600 chars, p90 ~55k, max 130k (~40k tokens) | split inside sections |
| 8-K item length | p50 643 chars, p90 2,800 | usually one chunk per item |
| Paragraph length | median 76 chars, p90 607, p99 1,629, max 5,082; 40% < 40 chars | pack paragraphs; ~1% exceed the cap |
| Boilerplate | 41% of chars of paragraphs >= 200 chars repeat in >= 3 filings | period in prefix |
| Tables | 11,960; 508 empty; 2,661 < 100 chars; 31 > 3.2k chars; 43% have XBRL cells | drop empty, merge small, rebuild from cells |
| Table cells | `$` and number in separate cells; ~69% padding; `is_header` always false | infer header rows, merge by header spans |

## 3. Method

1. Never cross a section. Within a section, a topic unit = one subheading (heading block or title-like line: short, no final period, standalone) + the paragraphs under it until the next subheading. A section without subheadings is one unit.
2. Cap 512 tokens including prefix (upper bound; tune 256/512/1024 by eval).
3. Unit <= cap -> one chunk.
4. Unit > cap -> balanced parts (1,100 tokens -> 3 x ~370, not 512+512+76), cut at the weakest remaining boundary: title-like line > paragraph. No overlap between paragraphs.
5. Unit < 64 tokens (tunable) -> merged whole, with its subheading kept in the text, into the next unit of the same section; if it is the last unit or the merge exceeds the cap, into the previous unit; if neither fits, kept as a short chunk. Never merged across sections.
6. Paragraph > cap -> split at sentence boundaries into balanced parts; each part repeats the previous part's last sentence; parts share `block_ids` and carry `part`/`part_count`, so retrieval can fetch and join all parts. A single sentence over the cap is split by tokens.
7. Sentence splitter: rule-based; no split after Inc., Corp., Co., No., U.S., e.g., i.e., Mr., Ms., Dr., vs., or inside decimals.
8. Noise filter (counted per reason): empty/decoration-only text; page numbers (`12`, `Page 12`, `xii`, `F-12`); running headers (<= 100 chars, no final period, same text with digits as `#` >= 4 times in >= 3 sections of a document); `table_of_contents` sections; heading-only sections (title goes to children's path). "None." / "Not applicable." kept.
9. Tables:
   - table block that is a heading -> heading text;
   - text < 100 chars -> merged into the text flow as `a | b | c`;
   - otherwise one chunk per table from the cell grid: header rows = top rows before the first numeric row with text in value columns (`Net Sales:` in the label column only is a group label); output columns = header cell spans (merges `$`, `)`, `%` into the number, drops padding); multi-level headers joined (`Three Months Ended March 31, 2025`); empty rows/columns dropped; numbers verbatim;
   - title = preceding heading or paragraph <= 150 chars in the same section (the lead-in sentence goes with the table); up to 5 following footnotes (`(1)`, `(a)`, `*`, `†`, <= 600 chars each);
   - > 800 tokens -> split by rows, each part repeats title and header rows (`part`/`part_count`).
10. Section path from root (root omitted), titles cut at 80 chars; primary document root = `Cover page`; EX-99 path `EX-99.1: {description}`; SIGNATURES and EXHIBIT INDEX kept.
11. Prefix on every chunk, embedded with the text:
    `{company_name} | {form} | {period} | {section_path}[ › {subheading}][ | Table: {title} (part i/n)]`
    period: 10-K `FY2025 (period ended …)`, 10-Q `FY2025 Q2 (period ended …)` from dei facts (fallback `report_date`), 8-K `filed … (event …)`.

## 4. Flow

1. Pick filings with no complete chunks in Gold; batches of ~200 filings by `filing_date`.
2. Read the batch from Silver via `scan_batches`: filings, filing_entities, documents, dei facts, sections, content_blocks, document_tables, document_table_cells.
3. Per narrative document, per section in order: drop noise.
4. Build topic units -> text chunks (rules 3–6); tables -> table chunks (rule 9).
5. Add prefix and metadata, count tokens, compute `chunk_sha256`.
6. Validate rows against the contract, append to Gold, verify, commit per batch.
7. Audit.

## 5. Silver inputs

- Content: `content_blocks`, `document_tables`, `document_table_cells`.
- Boundaries + metadata: `sections`, `documents`, `filings`, `filing_entities`, dei facts from `xbrl_facts` + `xbrl_concepts` (`DocumentFiscalYearFocus`, `DocumentFiscalPeriodFocus`, `TradingSymbol`; one pre-pass).
- Not chunked: other XBRL tables, `document_links`, `exhibits` (index already chunked as a table).

## 6. `native_rag.chunks` (partition `year(filing_date)`)

Rule: keep only fields used to filter, cite or expand context; anything else is joined from Silver by `filing_id`/`section_id`.

| Use | Columns |
|---|---|
| Keys, links | `chunk_id` = `{document_id}:chunk:{NNNN}`, `filing_id`, `document_id`, `section_id`, `table_id`, `block_ids` |
| Filter | `cik`, `tickers`, `form`, `fiscal_year`, `fiscal_period` (FY/Q1/Q2/Q3/null), `period_end_date`, `filing_date`, `section_topic`, `document_type`, `chunk_kind` (text/table) |
| Citation | `company_name`, `accession_number`, `sec_url`, `section_title`, `subheading`, `table_title` |
| Context expansion | `chunk_ordinal` (document order), `part`, `part_count` |
| Content | `context` (prefix), `text` |
| Control | `token_count`, `chunk_sha256` |

Table properties: `lexifin.rules-version`, tokenizer revision + SHA-256, Silver rules versions read (mismatch -> refuse, rebuild).

`section_topic` maps form + raw section code to one label (10-K ITEM 7 and 10-Q PART I ITEM 2 -> `mdna`; ITEM 1A and PART II ITEM 1A -> `risk_factors`). Vocabulary: `cover`, `business`, `risk_factors`, `properties`, `legal_proceedings`, `market_equity`, `mdna`, `market_risk`, `financial_statements`, `controls`, `governance`, `exhibits`, `signatures`, `8k_event`, `press_release`, `other`.

## 7. Operation

- `uv run python scripts/native_rag/build_chunks.py --data-dir <dir>`; stage list in `pipelines/native_rag/gold_stages.py`.
- Incremental: skip documents with complete chunks; documents with zero chunks re-examined each run.
- Logs via `src/utils/logger.py` (`native_rag_chunks_started`, `_batch_committed`, `_document_rejected`, `_completed`, with `run_id`); phases timed with `timed_operation`; final summary printed.
- Audit (`quality/native_rag/audit.py`): unique keys, complete ordinals and parts; FKs to Silver, one section per chunk; coverage (chunked ∪ noise blocks = all narrative blocks, split parts rebuild the block, all table rows present); token cap; hash recompute; exact rebuild of the latest 10-K/10-Q/8-K; WARN on chunks < 16 tokens, documents without chunks, > 30% filtered.
- Tests: unit per transform; integration fixture Bronze -> Silver -> chunks with pinned digest, rerun skip, audit pass, Silver rules change refused.

## 8. Layout

```
data/_catalog/gold.db
data/_cache/models/qwen3-embedding-0.6b/tokenizer.json
data/gold/native_rag/chunks/{data,metadata}/

src/data_pipeline/storage/iceberg.py                 + open_gold_catalog (+ read-only)
src/data_pipeline/contracts/native_rag/chunks.py     row schema + checks
src/data_pipeline/storage/native_rag/chunks.py       Iceberg table spec
src/data_pipeline/transforms/native_rag/
    tokenizer.py  section_topics.py  noise.py  sentences.py
    context.py    table_chunks.py    text_chunks.py
src/data_pipeline/pipelines/native_rag/{gold_stages,filing_metadata,chunks}.py
src/data_pipeline/quality/native_rag/audit.py
scripts/native_rag/build_chunks.py
tests/unit/native_rag/, tests/integration/native_rag/
docs/data/native_rag_model.md
```
`src/data_pipeline/transforms/gold/.gitkeep` to be removed (ask first).

## 9. Build order

1. Gold catalog: `open_gold_catalog` in `storage/iceberg.py`.
2. Contract + table spec of `chunks`.
3. Pure transforms, each with unit tests: tokenizer -> section_topics -> noise -> sentences -> context -> table_chunks -> text_chunks.
4. Pipeline: filing_metadata (company, filing, dei) -> chunks (read, chunk, write, incremental) -> gold_stages.
5. `scripts/native_rag/build_chunks.py`.
6. Audit.
7. Integration test with pinned digest.
8. Run on local `data/`, inspect sample chunks per form/topic/kind, adjust rules.
9. Docs (`native_rag_model.md`, AGENTS.md), then run on homepc 500 companies.

## 10. Downloads (each needs the user's approval)

- `uv add tokenizers` (Hugging Face, Rust).
- `tokenizer.json` (~11 MB) from Hugging Face `Qwen/Qwen3-Embedding-0.6B`, pinned revision, SHA-256 recorded in code; no model weights, no GPU.

## 11. Estimates

Chunking local: minutes; audit < 1 min. 500 companies: 1–3 h (exFAT reads); audit < 1 h.

## 12. Amendments after independent review (2026-11-16; override sections 3–7)

1. Layout tables -> text: a table block whose non-empty cells span <= 2 columns, or that has no numeric cell, is read as paragraphs (one per non-empty row, cells joined by spaces), not as a grid. Covers bullet lists laid out as tables (966 locally) and HP-style 8-K item tables.
2. Header detection: years (`2025`), dates (`March 31, 2025`) and period words are not numeric when finding header rows.
3. One cap for all chunks: 512 tokens including prefix; tables split by rows at that cap (the 800-token table cap is dropped).
4. Title-like line: non-table block, <= 120 chars, <= 12 words, does not end in `.`, `,`, `;` or `:`, contains a letter, not a running header. A section's opening block (found through `sections.start_block_ordinal`, a table block included) goes into the path, not the text.
5. Order: noise -> layout tables to text -> small tables (< 100 chars) inline as text -> topic units split at subheadings -> a table chunk cuts its unit (lead-in title and footnotes move into the table chunk; text after it keeps the unit's subheading) -> merge short units (merged chunk carries the first unit's subheading) -> balanced split of long units -> long paragraphs by sentence (parts are never packed with neighbours). Balanced split: choose k = ceil(tokens / cap) parts, then the cut points minimizing max part size, preferring a cut before a title-like line on ties; increase k if a part still exceeds the cap.
6. Every block ends in exactly one place, recorded for the audit: text chunk, table chunk (table, title, footnote), section path, or drop reason. Chunks record source blocks in `block_ids`; the repeated sentence is not counted as coverage.
7. Metadata sources: `company_name` and `tickers` from dei `EntityRegistrantName` / `TradingSymbol` of the filing's own context (tickers filtered to `^[A-Z][A-Z.\-]{0,9}$`, deduplicated, sorted), fallback the `filing_entities` filer's conformed name; `period_end_date` from dei `DocumentPeriodEndDate`, fallback `report_date`; `fiscal_year`/`fiscal_period` null for 8-Ks; `sec_url` = `https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession without dashes}/{documents.filename}`; EX-99 path title from `exhibits.description` via `resolved_document_id`, else `documents.description` when it differs from the type, else the type.
8. Incremental grain = document; add `document_chunk_count` (declared count, ordinals 1..N audited). `chunk_id` = `{document_id}:chunk:{NNNNNN}`.
9. Topics: 10-Q items keyed by parent PART code; new topic `ex99` for EX-99 exhibits (replaces `press_release`); 10-K ITEM 15 stays `exhibits` (financial pages filed there are a known mislabel).
10. Reproducibility: embedded text = `context + "\n\n" + text`; `token_count` counts it with special tokens; `chunk_sha256` = SHA-256 of the embedded text; all tunables stored as table properties and covered by the rules version; rows written sorted by `chunk_id`.
11. Noise additions: signature-page tables and TOC-like tables (mostly `Item`/`Part`/page-number rows) in any section are dropped; 8-K cover pages stay, tagged `cover`, so retrieval can exclude them.
12. Prefix rule: a paragraph is split when it exceeds the cap minus the prefix tokens.

## 13. Status

Implemented 2026-11-16 after the user's go. Deviations from sections 3–12 found while testing on `data/`: TOC sections keep prose over 150 chars (path `Front matter`); running headers also count at >= 8 repeats in one section and may end in a company suffix (`Alphabet Inc.`); table titles take a lead-in ending in `:` (<= 400 chars) plus up to 3 title-like lines; company names get one spelling per CIK; a lone title line keeps the subheading in force.
