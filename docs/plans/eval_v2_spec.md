# Eval set v2: spec (agreed 2026-11-24, before coding)

Goal: a frozen, hard question set on the 500-company corpus that measures
native RAG now and GraphRAG / agents later on the same questions. v1.x stays
the 5-company regression set.

## 1. Principles

- Sample from data, never from what the system gets wrong (no selection bias);
  tag difficulty and report per tag instead.
- Exact answers come from code wherever possible (XBRL values, computed
  growth/ratios, rankings); writers word questions only.
- Every answerable question has evidence (Silver `block_id` + quote), as v1.
- Two splits: `dev` (used while fixing the system) and `test` (run only for
  reports). Fixes are judged on `dev`; `test` numbers are the headline.
- Different families: Claude writes; the reviewer, answerer and judge are not
  Claude.

## 2. Corpus and sampling

- homepc Gold: chunks v8, 3,737,715 chunks, 500 companies; Silver XBRL facts.
  `as_of` fixed in `set.json`.
- Strata (least-used value first, as `sampling.py` does now): GICS sector
  (11; the `Sector` column of the Bronze `ivv_holdings.csv` of the universe
  snapshot, joined by ticker to the CIK in `universe.jsonl`; SIC from
  `filing_entities` for a company without a match), company size (IVV
  weight decile, so small companies are not crowded out), form (10-K, 10-Q,
  8-K, EX-99), fiscal year 2021–2026, period, section topic,
  chunk kind (text, table, split table part, XBRL fact), mention style (name,
  ticker, nickname, former name such as Facebook → Meta).
- Hard negatives recorded per task: chunks with the same concept or heading in
  another year, segment or company, so a run can report distractor errors.

## 3. Question types and minimum counts (first cut 150 tasks; 300 later)

| Type | New | Min | Answer source |
|---|---|---|---|
| fact (text) | | 12 | passage |
| numeric (XBRL) | | 12 | fact value |
| table_lookup | | 10 | table row |
| list / summary | | 10 | passage |
| compare_periods (same company) | | 10 | two facts / passages |
| compare_companies (same sector) | | 10 | two facts / passages |
| multi_hop (one filing) | | 8 | two passages |
| calc: growth, margin, share, CAGR | yes | 12 | code from XBRL facts |
| ranking: 3–4 companies, one metric | yes | 8 | code from XBRL facts |
| temporal_change: what changed in a section between two years (risk factors, segments) | yes | 8 | two passages |
| event_8k: officer change, M&A, results, by item code | yes | 10 | 8-K item passage |
| relation: supplier, customer, competitor, subsidiary, officer named in text | yes | 10 | passage (tag `graph`) |
| aggregation: which of N companies say X | yes | 6 | N passages (tag `graph`) |
| unanswerable: company/period/topic absent (companies: half from smaller countries, e.g. Vietnam, Thailand, Poland; half US-listed outside the 500; checked against `universe.jsonl`) | | 10 | refusal |
| false_premise: question assumes something the filing contradicts | yes | 6 | refusal + correction |

Minimums sum to 142; the other 8 go to the least-covered cells. At 8–12 per
type, per-type numbers have wide intervals (report them with the count); the
300 cut doubles every minimum.

Traps spread across types (tags, ≥ 8 each): fiscal ≠ calendar year,
relative time ("last quarter" vs `as_of`), balance-sheet date vs period,
former name / ticker only, split table part, small company, near-duplicate
distractor.

## 4. Languages

English and Vietnamese per question, as v1 (Vietnamese written as an investor
would, no calques). Optional informal variant (typos, abbreviations) for 10%.

## 5. Quality process

1. `sample_testset.py` on homepc writes tasks (code computes calc/ranking
   answers and hard negatives).
2. Claude writers draft per the brief (extended for new types).
3. `check_testset.py`: evidence, numbers (`numbers.py`), wording; widening as v1.2.
4. Non-Claude reviewer LLM reads every question with full passages:
   answerable from the passages, unambiguous period/company, answer correct.
5. A person reviews a stratified 20% sample (≥ 30) with a rubric; reviewer–person
   agreement reported.
6. Freeze `evals/sets/v2/` with `dev`/`test` split (40/60, stratified by type).

Reviewer model for step 4: chosen when an API source is found (steps 1–3 do
not need it).

## 6. Code changes

- `testset.py`: new `QuestionType`s, `split`, `tags`, `hard_negative_chunks`,
  `computed` (operation, value, unit, operands, ranking order).
- `sampling.py` / `sampling_data.py`: builders for calc, ranking,
  temporal_change, event_8k, relation, aggregation, false_premise; sector and
  size strata; reads homepc Gold/Silver (memory-bounded as now).
- `checks.py`: computed answers, false-premise checks.
- `retrieval_eval.py` / `answer_eval.py`: per-tag and per-split reports,
  distractor-hit metric, `answer.value` for computed answers.
- Scripts: `--split dev|test`.

## 7. Decisions (2026-11-24)

1. Size: 150 tasks first (≈ 300 rows), 300 after the process holds.
2. Sector: GICS from the Bronze IVV holdings (no new source, deterministic);
   SIC fallback. Reused later by GraphRAG layer 3 (groups by sector).
3. Reviewer model: open until an API source is found.
4. Split: 40% `dev`, 60% `test`.
5. Outside companies: v1's list (Exxon, Coca-Cola…) is inside the 500;
   v2 draws them from smaller countries and from US-listed companies outside
   the universe, never from `universe.jsonl`.
6. Writers: Claude Sonnet in a separate session, prompt in
   `docs/plans/eval_v2_writer_prompt.md`.
