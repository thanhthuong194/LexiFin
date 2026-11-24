# RAG evaluation

Retrieval is scored against a frozen question set. Answer generation and its
judging come later and reuse the same set.

## Layout

- `evals/sets/<v>/` (tracked): `set.json` (`SetManifest`: seed, `as_of`),
  `tasks.jsonl` (sampled `EvalTask`s), `batches/` (tasks split per writer),
  `drafts/` (writers' `DraftQuestion`s), `questions.jsonl` (checked
  `EvalQuestion`s), `check_report.md`, `coverage.json`.
- `evals/runs/`, `evals/cache/` (gitignored): eval runs, query-model replies.
- Code: `src/evals/` (`testset` schema, `sampling`, `checks`,
  `retrieval_eval`), scripts in `scripts/evals/`.

## Flow

1. `uv run python scripts/evals/sample_testset.py --set-id v1 [--total 100] [--seed 1]`:
   picks chunks and XBRL facts per coverage cell (company, form, section topic,
   text/table/split part, year, period, question type, mention style), checked
   against the data; metadata never comes from writers.
2. Writers (Claude agents, one batch each) answer every task of their batch
   following the brief below, into `drafts/<batch>.jsonl`; a second agent
   reviews the drafts.
3. `uv run python scripts/evals/check_testset.py --set-dir evals/sets/v1.1 --data-dir data`:
   maps quotes to Silver blocks, checks evidence against Silver and Gold,
   flags wording problems; writes `questions.jsonl` only without errors.
   A person (or a non-Claude LLM) reviews a sample, shown the full passages,
   not only the quotes; then the set is frozen (never edited; a change is a
   new version).
4. `uv run python scripts/evals/eval_retrieval.py [--no-rerank] [--no-llm]`:
   runs the retriever on each question in English and Vietnamese and writes
   `evals/runs/<time>_<label>/` (`report.md`, per-question results, ranx
   runs); `--compare <run> <run>` tests differences.

Evidence is a Silver `block_id` plus a verbatim quote, so a set outlives chunk
rule changes. A retrieved chunk is relevant when it holds the block and shows
the quote. Query understanding is scored by whether the filters the retriever
used admit the evidence chunks, which covers fiscal vs calendar years.

Model roles stay in different families: Claude writes questions; the answer
model (later) and the judge (later) must not be Claude.

## Writer brief

Input: one `EvalTask` per line (passages with `context` and `text`, companies
with tickers and aliases, `persona`, `mention_style`, `question_type`, `fact`,
`hint`). Output: one `DraftQuestion` JSON per task, same `task_id`.

- Ask what a real `persona` would type, not a paraphrase of the passage: no
  run of 6+ passage words; use everyday terms ("how much cash did they burn"
  over "net cash used in operating activities") where natural.
- Name the company as `mention_style` says: `name` (plain name, no "Inc."),
  `ticker`, or `nickname` (one of `aliases`), the nickname only when it fits
  the matter (no "Google" for an Alphabet board vote).
- Always state the period the passage is about: fiscal year or quarter as the
  company reports it ("fiscal 2024", "Q2 FY2025"), or the 8-K's month/year.
  Never "latest" unless the passage is the newest one of its kind.
- The passage alone must answer the question (for paired types, each passage
  answers its half). Nothing outside the passage, no yes/no questions.
- Ask exactly what the passage answers: not "what makes up X" when it gives
  the drivers of a change in X.
- `question_vi`: the same question as a Vietnamese investor would write it,
  natural and with diacritics; same company, period and scope as English.
  No parenthetical calques ("dài hạn (không ngắn hạn)"); keep English terms
  Vietnamese investors use (goodwill, fulfillment) only where natural.
- `reference_answer`: English, 1–3 sentences, only facts, numbers and dates
  in the passages (no outside knowledge, even if true). Numeric: the fact's
  value with unit and period, rounded as shown in the passage. Unanswerable:
  a short refusal saying what is missing. The check warns
  (`answer_fact_not_in_passage`) on an amount or full date the passages'
  text or context do not show, unless it is a sum, difference, ratio or
  change of two numbers in the quotes or the answer.
- `evidence_quotes`: one group per passage (paired types: one per passage);
  each quote copied verbatim from that passage's `text` (a table row or a
  sentence, at most 300 characters) that holds the answer. Prefer the single
  line with the value over headings.
- By type: `numeric`/`table_lookup` one value; `list` 3+ items;
  `summary` gist of the passage; `compare_periods` the same item across both
  periods (only periods a passage alone does not already cover);
  `compare_companies` the same item for both companies; `multi_hop` needs
  both passages; `unanswerable`: ask naturally about `hint` (the company,
  period or topic the corpus lacks), `evidence_quotes` empty.
- If a task cannot yield a sensible question, still return a draft and start
  `reference_answer` with `SKIP:` and the reason; the check reports it.
