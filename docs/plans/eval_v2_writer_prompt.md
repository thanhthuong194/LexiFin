# Eval v2: writer prompt (paste into a separate Claude Sonnet session)

Run it only after `sample_testset.py --set-id v2` has written
`evals/sets/v2/tasks.jsonl` and `evals/sets/v2/batches/*.jsonl`. The session
writes only `evals/sets/v2/drafts/`; it never edits code, other sets or data.

---

You are writing questions for a frozen evaluation set of LexiFin, a RAG system
over SEC 10-K, 10-Q and 8-K filings of 500 US companies. Repository:
`/home/bee/Workspace/LexiFin`. Read first: `AGENTS.md` (RAG eval line),
`docs/evals.md` (section "Writer brief", the rules below extend it),
`docs/plans/eval_v2_spec.md`, and `src/evals/testset.py` (`EvalTask`,
`DraftQuestion`).

## Task

Input: `evals/sets/v2/batches/<batch>.jsonl`, one `EvalTask` per line.
Output: `evals/sets/v2/drafts/<batch>.jsonl`, one `DraftQuestion` per task,
same `task_id`, same order, valid JSON per line. Never skip a task: if a task
cannot give a sensible question, still write the draft with
`reference_answer` starting `SKIP:` and the reason.

Work batch by batch. For each batch, use one sub-agent to write and a second,
fresh sub-agent to review the drafts against the passages (checklist below);
fix what the reviewer finds, then write the file. Do not batch-edit drafts
with scripts; every question is read and written by hand.

## Rules for every question (from the v1 brief, keep them)

- Ask what the `persona` would really type, not a paraphrase of the passage:
  no run of 6+ consecutive passage words; everyday terms where natural.
- Name companies as `mention_style` says (`name` without "Inc.", `ticker`,
  `nickname` from `aliases`, `former_name` when given).
- Always state the period as the company reports it ("fiscal 2024",
  "Q2 FY2025", or the 8-K's month/year). "Latest" only when the passage is the
  newest of its kind. When the task tags `relative_time`, phrase the period
  relatively ("last quarter", "năm ngoái"), read against `as_of` in
  `set.json`; never without the tag (the check warns either way).
- The passages alone must answer the question; nothing from outside.
- `question_vi`: the same question as a Vietnamese investor would write it,
  natural, with diacritics, same company/period/scope; no parenthetical
  calques; keep English terms Vietnamese investors use only where natural.
- `reference_answer`: English, 1–3 sentences, only facts in the passages (or
  the task's `computed`), numbers rounded as the passage shows them,
  with unit and period.
- `evidence_quotes`: one group per passage the answer needs; each quote
  copied verbatim from that passage's `text` (a table row or sentence,
  <= 300 characters) holding the answer; prefer the line with the value.

## Rules per question type

- `fact`, `numeric`, `table_lookup`, `list` (3+ items), `summary`,
  `compare_periods`, `compare_companies`, `multi_hop`, `unanswerable`: as in
  the v1 brief.
- `calc`: the task gives `computed` (value, unit, operation, operands). Ask
  for the derived figure (growth, margin, share, CAGR) without stating the
  operands; the reference states the result as `computed` gives it (a
  percent to 1 decimal: `12.3%`), with its direction for a growth, CAGR or
  difference ("rose", "fell"), and the operands it comes from. One quote
  group per passage; quote the row of every operand, with its value.
- `ranking`: ask which of the listed companies is highest, or lowest, on
  the metric for the period; name all of them; the reference names every
  company with its value in `computed.order` (highest first), or lowest
  first when the question asks for the lowest; one quote group per passage.
- `temporal_change`: two passages of the same section in two fiscal years;
  ask what changed (added, removed or reworded item); the reference names
  the change and both years; one quote group per passage, quoting the
  differing text (not the same sentence twice).
- `event_8k`: ask about the event of the 8-K item (who left or joined and in
  which role, what was acquired, the date); the 8-K's month/year in the
  question.
- `relation`: `hint` holds the snippet that matched; ask about the
  relationship it states (supplier, customer, competitor, subsidiary,
  officer and role); the reference names both parties and the relation
  exactly as stated, nothing inferred.
- `aggregation`: the topic is in `hint`; ask which of the listed companies
  say something about it; the reference names every listed company with
  yes/no (companies in `cell.without_topic` are "no", said with a negation
  in their sentence); one quote group per company that says it (from its
  own passage), none for the `without_topic` ones.
- `false_premise`: the question assumes `hint`, which the passage
  contradicts (revenue "fell" when it rose; a CEO who did not leave). Ask
  it naturally; the reference denies the premise in words ("No, ...",
  "did not") and states the true fact with its direction ("rose 12.3%"),
  and `computed.value` when given. With `computed`, quote the row of each
  operand (one group per passage used); else one group holding the fact.
- `unanswerable`: ask naturally about `hint` (a company outside the corpus,
  often from a smaller country, a period not covered, or an absent topic);
  `evidence_quotes` empty.

## Review checklist (second sub-agent)

For each draft: the passages alone answer it; company, period and scope are
unambiguous; the reference matches the passages (every number and date
appears there or in `computed`); quotes are verbatim; the Vietnamese question
means the same as the English one; the type rules hold; no 6+ word copy.

## When done

Do not run `check_testset.py`: it needs the 500-company Silver, which is on
another machine; its report comes back to you as a list of drafts to fix.
Validate every draft line yourself with
`uv run python -c "import sys; from src.evals.testset import DraftQuestion; [DraftQuestion.model_validate_json(l) for l in open(sys.argv[1]) if l.strip()]" evals/sets/v2/drafts/<batch>.jsonl`.
Report: drafts written per batch, SKIPs with reasons. Do not commit.
