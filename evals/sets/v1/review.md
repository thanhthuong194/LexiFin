# v1 human review (2026-11-24)

25-question sample (3 per type) read by three outside LLM reviewers (Gemini,
ChatGPT, Grok) and checked against the task passages. The set stays frozen;
all 97 drafts were then reviewed by agents and fixed in `../v1.1/` (19 drafts).

Known issues:

- t0016: question asks which cost categories make up R&D; the NVIDIA half of
  the answer lists the drivers of the R&D increase, not its composition.
- t0064 (vi): "dài hạn (không ngắn hạn)" is a stiff calque; "dài hạn" is
  enough.
- t0031 (en): "what does Alphabet say it needs from users…" reads ambiguously.
- t0030: nickname "Google" for an Alphabet board matter.
- t0026: the answer compares contractual principal with carrying value
  without saying so.

Not issues: "answer not supported by evidence" flags on t0005, t0008, t0011,
t0032, t0041, t0056, t0080 came from a sample that showed at most two quotes;
every flagged fact is in the passage (t0025's June 30, 2024 is in a table
header). Evidence quotes only locate the chunk
for retrieval scoring. Table quotes without column headers are fine for the
same reason; t0042 column order was checked.

For v2: check that numbers and dates in `reference_answer` appear in the
passages; tell writers to avoid parenthetical calques in Vietnamese and to
match the question to what the passage answers; review samples show the full
passage.
