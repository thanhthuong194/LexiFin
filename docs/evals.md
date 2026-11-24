# RAG evaluation

Retrieval and answers are scored against the same frozen question set.

## Layout

- `evals/sets/<v>/` (tracked): `set.json` (`SetManifest`: seed, `as_of`),
  `tasks.jsonl` (sampled `EvalTask`s), `batches/` (tasks split per writer),
  `drafts/` (writers' `DraftQuestion`s), `questions.jsonl` (checked
  `EvalQuestion`s), `check_report.md`, `coverage.json`.
- `evals/runs/`, `evals/cache/` (gitignored): eval runs, query-model replies.
- Code: `src/evals/` (`testset` schema, `sampling`, `checks`, `numbers`,
  `retrieval_eval`, `answer_eval`, `answer_judge`), scripts in `scripts/evals/`.

## Flow

1. `uv run python scripts/evals/sample_testset.py --set-id v1 [--total 100] [--seed 1]`:
   picks chunks and XBRL facts per coverage cell (company, form, section topic,
   text/table/split part, year, period, question type, mention style), checked
   against the data; metadata never comes from writers.
2. Writers (Claude agents, one batch each) answer every task of their batch
   following the brief below, into `drafts/<batch>.jsonl`; a second agent
   reviews the drafts.
3. `uv run python scripts/evals/check_testset.py --set-dir evals/sets/v1.2 --data-dir data`:
   maps quotes to Silver blocks, checks evidence against Silver and Gold,
   flags wording problems; writes `questions.jsonl` only without errors.
   Every evidence group quoting a table row is widened to the same row of
   the group's company elsewhere (same label without note references, same
   answer value: 10-Q/10-K twin of a release table, later comparatives); a
   value with under 4 significant digits (EPS, ratios) also needs another
   number of the row. Widened evidence (`widened`) counts for retrieval, not
   for filter admission. v2 checks per type (errors unless noted): `calc`
   reference states `computed.value` (a percent within half its first
   decimal; an amount to 2+ significant digits that round to it) with the
   direction of a growth, CAGR or difference (rose/fell, tăng/giảm, or a
   sign), every operand quoted with its value (each operand's group is
   widened to its other fact blocks and same-label rows); `ranking` names
   every company of `computed.order` in order, each placed by its mention
   nearest its value, reversed only when the question or reference asks for
   the lowest; `false_premise` with `computed` as calc (1..passages groups),
   without it one group; warning `premise_not_denied`; `temporal_change` one
   group per passage, distinct fiscal years (warning when both quote the
   same text); `aggregation` one group per passage of a company not in
   `cell.without_topic` (`|`-separated CIKs; quoting one is an error), a
   warning when the reference omits a company or names a `without_topic`
   one without a negation in that sentence; `event_8k` passages are 8-K
   items (`8k_event`). Warning `relative_time` when the tag and relative
   wording ("last quarter", "năm ngoái") disagree. Hard negatives whose text
   shows an evidence quote or the fact/operand value are dropped (report
   lists them). `--assign-splits` (v2+) sets `split` per question: `dev`
   when sha256 of `{seed}:{task_id}` (`set.json` seed) is below 0.4·2^256,
   else `test`, so adding or dropping questions moves no other; the report
   shows the balance per type and tag. v1.x sets keep `split` null and
   their `questions.jsonl` byte-identical (v2 fields at defaults are left
   out).
   A person (or a non-Claude LLM) reviews a sample, shown the full passages,
   not only the quotes; then the set is frozen (never edited; a change is a
   new version).
4. `uv run python scripts/evals/eval_retrieval.py [--no-rerank] [--no-llm]`:
   runs the retriever on each question in English and Vietnamese and writes
   `evals/runs/<time>_<label>/` (`report.md`, per-question results, ranx
   runs); `--compare <run> <run>` tests differences. `--split dev|test`
   scores one split (v2; fixes are judged on `dev`, `test` is the headline).
   v2 reports add tables per tag and per split, and `post.hard_negative@8`:
   share of answerable rows with hard negatives where one ranks above the
   first evidence chunk in the delivered top 8 (shown only when a question
   has hard negatives).

Evidence is a Silver `block_id` plus a verbatim quote, so a set outlives chunk
rule changes. A retrieved chunk is relevant when it holds the block and shows
the quote. Query understanding is scored by whether the filters the retriever
used admit the evidence chunks, which covers fiscal vs calendar years, and by
`query.search_text`: every corpus searched in its own language (a Vietnamese
question with no English text searches English chunks in Vietnamese; v1.1
baseline: 77/81 such rows, silent before this metric).

v1.2, reranker on (2026-11-24): hit@8 en 0.964 -> 0.976, vi 0.880 -> 0.976
after per-language search texts (baseline v1.1 rescored on v1.2); 13 -> 4
failures (t0042 en reads Sept 30 as a 10-K; t0042 vi, t0060 vi ranking;
t0077 split table loses its title).

## Answers

5. `uv run python scripts/evals/eval_answers.py [--judge] [--limit N]`:
   retrieves each question in English and Vietnamese, answers with
   `chat.generator` and scores `answers.jsonl` (metrics in the
   `src/evals/answer_eval.py` docstring): the expected value stated (XBRL
   fact exactly, else the reference's first amount at its precision, any
   sign; `calc` and computed `false_premise`: the computed value with the
   direction of a change; other `false_premise`: the premise denied and the
   reference amount stated), `answer.order` (`ranking`: companies in order,
   reversed only when the question or answer asks for the lowest), refusals of unanswerable questions, false refusals, citations,
   a cited passage holding evidence, answer language; split by whether
   retrieval found every evidence group, with tables per question type and
   per tag. `--split dev|test` as for retrieval. `--judge` adds `chat.judge`:
   correctness against the English reference (correct/partial/incorrect,
   across languages and rounding) and faithfulness against every passage
   shown (hallucination; citation precision is the deterministic metric).
   `--rejudge RUN` judges stored answers again.

Model roles stay in different families: Claude writes questions; the answer
model (gpt-oss) and the judge (Mistral) must not be Claude.

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

### v2 additions

Full prompt: `docs/plans/eval_v2_writer_prompt.md` (keep in sync).

- `mention_style` `former_name`: one of the company's `former_names`.
- Tag `relative_time`: phrase the period relatively ("last quarter",
  "năm ngoái"), read against `as_of` in `set.json`; without the tag, never.
- `calc`: ask for the derived figure (growth, margin, share, CAGR) without
  the operands; the reference states `computed.value` (percent to 1
  decimal) with its direction (rose/fell) and its operands; one group per
  passage quoting each operand's row with its value.
- `ranking`: ask for the highest or the lowest; name all listed companies;
  the reference lists every one with its value, in `computed.order`
  (highest first), or lowest first when the question asks for the lowest;
  one group per passage.
- `temporal_change`: one group per passage (two fiscal years, different
  quotes); the reference names the change and both years.
- `event_8k`: the event of the 8-K item, with the 8-K's month/year.
- `relation`: `hint` holds the matched snippet; ask about that
  relationship; the reference states it as the passage does, both parties.
- `aggregation`: the topic is in `hint`; the reference says yes/no for
  every listed company (companies in `cell.without_topic` are "no", with a
  negation in their sentence); one group per company that says it.
- `false_premise`: ask naturally assuming `hint`; the reference denies it
  ("No, ...", "did not") and states the true fact with its direction (and
  `computed.value` if given); with `computed`, quote each operand's row
  (one group per passage used), else one group.
