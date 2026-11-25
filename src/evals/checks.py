"""Merge writer drafts with their tasks into checked `EvalQuestion`s.

Each evidence quote is mapped to the Silver blocks of its passage whose text
contains it, and must show in a current Gold chunk holding that block.
Errors block the set (missing drafts, quotes not found, en/vi asking about
different periods, a computed answer or ranking order missing, v2 type
shape); warnings are for a human pass (company, period or mention style
missing, copied wording, answer numbers or dates not in the passages, a
false premise not denied, near duplicates, skipped drafts).
"""

from __future__ import annotations

import hashlib
import itertools
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from difflib import SequenceMatcher
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
from pydantic import ValidationError
from pyiceberg.expressions import And

from src.data_pipeline.storage.iceberg import (
    filing_dates_filter,
    open_gold_catalog_read_only,
    open_silver_catalog_read_only,
    read_rows_in,
    scan_batches,
    values_filter,
)
from src.data_pipeline.storage.native_rag.chunks import CHUNKS_TABLE
from src.data_pipeline.storage.sec_filings.content_blocks import CONTENT_BLOCKS_TABLE
from src.evals.numbers import (
    VALUE_MIN_DIGITS,
    dates,
    derived,
    numbers,
    states,
    states_computed,
    states_direction,
    value_tokens,
    without_dates,
)
from src.evals.testset import (
    CompanyRef,
    Computed,
    DraftQuestion,
    EvalQuestion,
    EvalTask,
    Evidence,
    ExpectedFilters,
    Split,
    TaskPassage,
    XbrlFact,
    normalize,
    whole_number,
)
from src.rag_engine.retrieval.companies import normalize_name
from src.rag_engine.retrieval.query import rule_plan

QUOTE_MAX_CHARS = 300
QUOTE_MIN_CHARS = 12
# A quote no single block holds may span this many consecutive blocks.
SPAN_MAX_BLOCKS = 20
# Lexical copying, over content words (no stopwords, numbers, period words,
# company names): a shared run this long, or this share of the question's
# content words found in the passages (with at least COPY_MIN_WORDS of them).
COPY_RUN = 5
COPY_SHARE = 0.8
COPY_MIN_WORDS = 6
NEAR_DUPLICATE_JACCARD = 0.8
SKIP_PREFIX = "SKIP:"
# Same-company rows that widen a table-row evidence group must carry a
# specific label: at least LABEL_MIN_CHARS (normalized) and not generic,
# unless the value has VALUE_SPECIFIC_DIGITS significant digits.
LABEL_MIN_CHARS = 6
GENERIC_LABELS = frozenset({"total", "subtotal", "net", "other", "totals"})
VALUE_SPECIFIC_DIGITS = 4
# Note references and footnote marks in a row label: "(Note 12)", "(1)", "(a)".
_LABEL_REFERENCE = re.compile(r"\((?:note\s*\d+[a-z]?|\d{1,2}|[a-z])\)", re.IGNORECASE)

PAIRED_TYPES = frozenset({"compare_periods", "compare_companies", "multi_hop"})
# One evidence group per task passage, each group on its own passage.
GROUP_PER_PASSAGE = PAIRED_TYPES | {"calc", "ranking", "temporal_change"}
# Questions whose operands (`Computed.operands`) must each be quoted.
OPERAND_TYPES = frozenset({"calc", "ranking", "false_premise"})
# Operations whose sign the answer must state (a direction word or sign).
SIGNED_OPERATIONS = frozenset({"growth", "cagr", "difference"})
# A ranking listed lowest first: the question or answer asks for the lowest.
LOWEST = re.compile(
    r"\b(lowest|smallest|least|fewest|bottom)\b|thấp nhất|nhỏ nhất|ít nhất",
    re.IGNORECASE,
)
# Period wording relative to `as_of` (tag `relative_time`).
RELATIVE_TIME = re.compile(
    r"\b(last|this|past) (fiscal )?(quarter|year)\b|"
    r"\b(latest|most recent|previous|prior) (fiscal )?(quarter|year)\b|"
    r"\b(one|two|three|four|five|\d+) (fiscal )?(quarters?|years?) ago\b|"
    r"\bbefore the (latest|most recent)\b|"
    r"năm ngoái|năm nay|quý này|vừa qua|vừa rồi|cách đây|"
    r"(quý|năm)( tài chính)? gần nhất|"
    r"\b(một|hai|ba|bốn|năm|\d+) (quý|năm)( tài chính)? trước\b",
    re.IGNORECASE,
)
# Company naming: tickers of at least TICKER_MIN letters, distinctive name
# words of at least NAME_WORD_MIN letters not in GENERIC_NAME_WORDS.
TICKER_MIN = 2
NAME_WORD_MIN = 4
GENERIC_NAME_WORDS = frozenset(
    {"american", "america", "international", "national", "united", "general",
     "first", "financial", "energy", "corporation", "company", "industries",
     "technologies", "technology", "systems", "services", "global", "bank",
     "bancorp", "bancshares", "foods", "food", "electric", "power", "capital",
     "communications", "health", "healthcare", "resources", "brands",
     "partners", "trust", "realty", "properties", "pharmaceuticals",
     "insurance", "motors", "airlines", "materials", "products", "solutions",
     "entertainment", "worldwide", "north", "south", "west", "east",
     "pacific", "southern", "northern", "western", "eastern", "platforms",
     "group", "holdings", "enterprises", "investment", "investments",
     "management", "networks", "software", "semiconductor", "devices",
     "scientific", "medical", "therapeutics", "petroleum", "chemical",
     "chemicals", "manufacturing", "markets", "stores", "restaurants",
     "hotels", "resorts", "water", "utilities", "data", "digital", "life",
     "home", "homes", "auto", "automotive", "mobile", "media", "royal",
     "standard", "street", "state", "city", "river", "lake", "mountain",
     "union", "federal", "mutual", "republic", "interactive", "labs",
     "laboratories", "instruments", "aerospace", "industrial", "logistics",
     "income", "real", "estate", "equity", "fund", "funds", "news", "cruise",
     "waste", "light", "gas", "oil"}
)  # fmt: skip
_YEAR = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
# SEC conformed names end in a state code: `CORP/TX`, `CO/`, `INC /DE/`.
_STATE_SUFFIX = re.compile(r"\s*/[A-Za-z]{0,3}/?\s*$")
# A reference (or answer) refusing a false premise says so.
PREMISE_DENIAL = re.compile(
    r"\b(not|no|never|rather|instead|actually|contrary|incorrect|false|wrong|"
    r"however|in fact|did ?n[o']t|was ?n[o']t|is ?n[o']t)\b|"
    r"\b(không|chưa|thực tế|ngược lại|sai)\b",
    re.IGNORECASE,
)
# `--assign-splits`: a question is `dev` when sha256(`{seed}:{task_id}`) is
# below this share of 2**256 (each question on its own: stable as sets grow).
DEV_SHARE = 0.4
# EvalQuestion fields added for v2, left out of a question's JSON at their
# defaults so v1.x `questions.jsonl` stay byte-identical.
V2_FIELDS = ("computed", "tags", "hard_negative_chunks", "split")
V2_COMPANY_FIELDS = ("former_names", "sector")
STOPWORDS = frozenset(
    {"a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does", "for",
     "from", "had", "has", "have", "how", "in", "is", "it", "its", "of", "on",
     "or", "that", "the", "their", "this", "to", "was", "were", "what", "when",
     "which", "who", "why", "with", "during", "over", "per", "than", "any",
     "all", "much", "many", "each", "about", "into", "under", "they", "them"}
)  # fmt: skip
PERIOD_WORDS = frozenset(
    {"january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december", "ended", "ending", "month",
     "months", "fiscal", "year", "years", "quarter", "quarters", "annual"}
)  # fmt: skip


@dataclass(frozen=True, slots=True)
class Issue:
    task_id: str
    check: str
    message: str


@dataclass(frozen=True, slots=True)
class Block:
    text: str  # `normalize`d
    block_type: str

    @classmethod
    def of(cls, text: str, block_type: str) -> Block:
        return cls(normalize(text), block_type)


@dataclass(slots=True)
class CheckResult:
    questions: list[EvalQuestion] = field(default_factory=list)
    errors: list[Issue] = field(default_factory=list)
    warnings: list[Issue] = field(default_factory=list)
    coverage: dict[str, Counter[str]] = field(default_factory=dict)
    skipped: Counter[str] = field(default_factory=Counter)
    # Question id -> evidence added from other filings (`widen_evidence`).
    widened: dict[str, int] = field(default_factory=dict)
    # Question id -> hard negatives dropped (`prune_hard_negatives`).
    pruned: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CompanyChunk:
    cik: str
    filing_date: date
    block_ids: tuple[str, ...]
    text: str  # raw


BlockLoader = Callable[[set[str], set[date]], Mapping[str, Block]]
CompanyChunkLoader = Callable[[set[str]], Sequence[CompanyChunk]]


@dataclass(slots=True)
class _Found:
    errors: list[tuple[str, str]] = field(default_factory=list)
    warnings: list[tuple[str, str]] = field(default_factory=list)


def words(text: str) -> list[str]:
    return re.findall(r"\w+", unicodedata.normalize("NFC", text).casefold())


def load_drafts(paths: Iterable[Path]) -> tuple[list[DraftQuestion], list[Issue]]:
    """Drafts from files or directories of `*.jsonl`; bad lines become errors."""

    drafts: list[DraftQuestion] = []
    errors: list[Issue] = []
    for path in paths:
        for file in sorted(path.glob("*.jsonl")) if path.is_dir() else [path]:
            lines = file.read_text(encoding="utf-8").splitlines()
            for number, line in enumerate(lines, 1):
                if not line.strip():
                    continue
                try:
                    drafts.append(DraftQuestion.model_validate_json(line))
                except ValidationError as error:
                    summary = "; ".join(
                        f"{'.'.join(map(str, e['loc']))}: {e['msg']}"
                        for e in error.errors()
                    )
                    issue = Issue("-", "draft_invalid", f"{file}:{number}: {summary}")
                    errors.append(issue)
    return drafts, errors


def check_testset(
    set_id: str,
    tasks: Sequence[EvalTask],
    drafts: Sequence[DraftQuestion],
    blocks: Mapping[str, Block],
    chunks: Mapping[str, Sequence[str]],
) -> CheckResult:
    """Pair drafts with tasks, check them and build `EvalQuestion`s.

    `blocks`: Silver block_id -> block; `chunks`: block_id -> raw text of
    each current Gold chunk holding it (see `load_blocks`, `load_chunks`).
    """

    by_task: dict[str, list[DraftQuestion]] = defaultdict(list)
    for draft in drafts:
        by_task[draft.task_id].append(draft)
    result = CheckResult(errors=pairing_errors(tasks, by_task))
    accepted: list[EvalTask] = []
    for task in tasks:
        if task.task_id not in by_task:
            continue
        draft = by_task[task.task_id][0]
        if draft.reference_answer.startswith(SKIP_PREFIX):
            reason = draft.reference_answer.removeprefix(SKIP_PREFIX).strip()
            result.warnings.append(Issue(task.task_id, "skipped", reason))
            result.skipped[task.question_type] += 1
            continue
        found = _Found()
        groups = evidence(task, draft, blocks, chunks, found)
        wording(task, draft, found)
        answer_facts(task, draft, found)
        computed_answer(task, draft, found)
        relative_wording(task, draft, found)
        result.errors += [Issue(task.task_id, c, m) for c, m in found.errors]
        result.warnings += [Issue(task.task_id, c, m) for c, m in found.warnings]
        if found.errors:
            continue
        try:
            result.questions.append(build_question(set_id, task, draft, groups))
        except ValidationError as error:
            result.errors.append(Issue(task.task_id, "question_invalid", str(error)))
            continue
        accepted.append(task)
    result.warnings += near_duplicates(result.questions)
    result.coverage = coverage(accepted)
    return result


def pairing_errors(
    tasks: Sequence[EvalTask], by_task: Mapping[str, list[DraftQuestion]]
) -> list[Issue]:
    task_ids = Counter(t.task_id for t in tasks)
    return (
        [Issue(t, "task_duplicate", f"{n} tasks") for t, n in task_ids.items() if n > 1]
        + [Issue(t, "draft_missing", "no draft") for t in task_ids if t not in by_task]
        + [
            Issue(t, "draft_unknown_task", "no such task")
            for t in by_task
            if t not in task_ids
        ]
        + [
            Issue(t, "draft_duplicate", f"{len(found)} drafts")
            for t, found in by_task.items()
            if len(found) > 1
        ]
    )


# Evidence


def evidence(
    task: EvalTask,
    draft: DraftQuestion,
    blocks: Mapping[str, Block],
    chunks: Mapping[str, Sequence[str]],
    found: _Found,
) -> tuple[tuple[Evidence, ...], ...]:
    """Quotes mapped to (block, quote) evidence, group by group."""

    quotes = draft.evidence_quotes
    if task.question_type == "unanswerable":
        if quotes:
            found.errors.append(("evidence_unexpected", "unanswerable with quotes"))
        return ()
    if not _group_count_fits(task, quotes, found):
        return ()
    passages = _assign(task, quotes, found)
    if passages is None:
        return ()
    groups = tuple(
        tuple(e for q in group for e in _quote_evidence(q, p, blocks, chunks, found))
        for group, p in zip(quotes, passages, strict=True)
    )
    _shape(task, passages, quotes, found)
    if task.fact is not None and not found.errors:
        groups = _with_fact(task.fact, groups, blocks, chunks, found)
    computed = task.computed if task.question_type in OPERAND_TYPES else None
    if computed is not None and not found.errors:
        groups = _with_operands(computed, groups, blocks, chunks, found)
    return groups


def _group_count_fits(
    task: EvalTask, quotes: tuple[tuple[str, ...], ...], found: _Found
) -> bool:
    """One group per passage (GROUP_PER_PASSAGE); a computed false premise
    1 to one per passage; aggregation one per passage of a company that
    says it (not in `cell["without_topic"]`); else one."""

    allowed = range(1, 2)
    count = len(task.passages)
    if task.question_type in GROUP_PER_PASSAGE:
        allowed = range(count, count + 1)
    elif task.question_type == "false_premise" and task.computed is not None:
        allowed = range(1, count + 1)
    elif task.question_type == "aggregation":
        saying = count - len(without_topic(task))
        allowed = range(saying, saying + 1)
    if len(quotes) in allowed and all(quotes):
        return True
    sizes = [len(g) for g in quotes]
    expected = "-".join(dict.fromkeys(map(str, (allowed[0], allowed[-1]))))
    found.errors.append(("evidence_groups", f"groups {sizes}, expected {expected}"))
    return False


def without_topic(task: EvalTask) -> frozenset[str]:
    """CIKs of an aggregation's companies that do not say it (`cell`
    `without_topic`, `|`-separated)."""

    return frozenset(c for c in task.cell.get("without_topic", "").split("|") if c)


def _shape(
    task: EvalTask,
    passages: Sequence[TaskPassage],
    quotes: tuple[tuple[str, ...], ...],
    found: _Found,
) -> None:
    """v2 types: the passages and quotes the groups use fit the type."""

    kind = task.question_type
    if kind == "temporal_change":
        years = {p.fiscal_year for p in passages} - {None}
        if len(years) != len(passages):
            message = f"fiscal years {[p.fiscal_year for p in passages]}"
            found.errors.append(("temporal_same_year", message))
        if len({normalize(" ".join(g)) for g in quotes}) < len(quotes):
            found.warnings.append(("temporal_same_quote", "groups quote the same text"))
    elif kind == "aggregation":
        ciks = [p.cik for p in passages]
        if len(set(ciks)) != len(ciks):
            found.errors.append(("aggregation_company_repeated", ", ".join(ciks)))
        if quoted := sorted(without_topic(task).intersection(ciks)):
            found.errors.append(("aggregation_without_topic_quoted", ", ".join(quoted)))
    elif kind == "event_8k":
        for p in passages:
            if p.form != "8-K" or p.section_topic != "8k_event":
                message = f"{p.chunk_id}: {p.form} {p.section_topic}"
                found.errors.append(("event_not_8k_item", message))


def _assign(
    task: EvalTask, quotes: tuple[tuple[str, ...], ...], found: _Found
) -> list[TaskPassage] | None:
    """The passage of each group: one-to-one for paired types, any otherwise."""

    texts = [normalize(p.text) for p in task.passages]
    fits = [
        {i for i, t in enumerate(texts) if all(normalize(q) in t for q in g)}
        for g in quotes
    ]
    for order in itertools.permutations(range(len(texts)), len(quotes)):
        if all(i in fit for i, fit in zip(order, fits, strict=True)):
            return [task.passages[i] for i in order]
    lost = [q for g in quotes for q in g if not any(normalize(q) in t for t in texts)]
    message = (
        f"not in passage text: {lost[0][:80]!r}" if lost else "groups share a passage"
    )
    found.errors.append(("quote_not_in_passage", message))
    return None


def _quote_evidence(
    quote: str,
    passage: TaskPassage,
    blocks: Mapping[str, Block],
    chunks: Mapping[str, Sequence[str]],
    found: _Found,
) -> list[Evidence]:
    text, short = normalize(quote), quote[:60]
    if len(quote) > QUOTE_MAX_CHARS:
        found.errors.append(("quote_too_long", f"{len(quote)} chars: {short!r}"))
    if len(text) < QUOTE_MIN_CHARS:
        found.warnings.append(("quote_short", repr(quote)))
    absent = [b for b in passage.block_ids if b not in blocks]
    if absent:
        found.errors.append(("block_not_in_silver", absent[0]))
    hits = [b for b in passage.block_ids if b in blocks and text in blocks[b].text]
    if not hits:
        # A short table a chunk writes inline joins its rows with `; `.
        rows = normalize(quote.replace(";", " "))
        hits = [b for b in passage.block_ids if b in blocks and rows in blocks[b].text]
    if not hits:
        hits = _spanning_blocks(text, passage.block_ids, blocks)[:1]
        if not hits:
            found.errors.append(("quote_no_block", f"no block run holds {short!r}"))
            return []
        found.warnings.append(("quote_spans_blocks", repr(short)))
    if text == normalize(passage.section_title) or all(
        blocks[b].block_type == "heading" for b in hits
    ):
        found.warnings.append(("quote_heading", repr(short)))
    packed = {c: normalize(c) for b in hits for c in chunks.get(b, ())}
    shown = [b for b in hits if any(text in packed[c] for c in chunks.get(b, ()))]
    if not shown:
        found.errors.append(("quote_not_in_chunk", f"no Gold chunk shows {short!r}"))
    return [Evidence(block_id=b, quote=quote) for b in shown]


def _spanning_blocks(
    text: str, block_ids: Sequence[str], blocks: Mapping[str, Block]
) -> list[str]:
    """The shortest run (up to SPAN_MAX_BLOCKS) of consecutive non-table
    blocks whose joined text holds `text` (a `normalize`d quote); [] when
    none. A sentence of a document whose inline runs are separate blocks
    spans several."""

    best: list[str] = []
    for start in range(len(block_ids)):
        joined = ""
        for end in range(start, min(start + SPAN_MAX_BLOCKS, len(block_ids))):
            block = blocks.get(block_ids[end])
            if block is None or block.block_type == "table":
                break
            joined += block.text
            if text in joined:
                if not best or end + 1 - start < len(best):
                    best = list(block_ids[start : end + 1])
                break
    return best


def _with_fact(
    fact: XbrlFact,
    groups: tuple[tuple[Evidence, ...], ...],
    blocks: Mapping[str, Block],
    chunks: Mapping[str, Sequence[str]],
    found: _Found,
) -> tuple[tuple[Evidence, ...], ...]:
    """The fact's group, plus its value in every other block showing the fact."""

    group = groups[0]
    if not any(e.block_id == fact.block_id for e in group):
        found.errors.append(("evidence_without_fact", f"no quote on {fact.block_id}"))
        return groups
    values = _fact_values(fact, group, {fact.block_id})
    if not values:
        found.errors.append(("quote_without_value", f"no {fact.value} in the quote"))
        return groups
    return (_with_alternatives(fact, group, values[0], blocks, chunks), *groups[1:])


def _with_operands(
    computed: Computed,
    groups: tuple[tuple[Evidence, ...], ...],
    blocks: Mapping[str, Block],
    chunks: Mapping[str, Sequence[str]],
    found: _Found,
) -> tuple[tuple[Evidence, ...], ...]:
    """Each operand quoted with its value on one of its blocks; its group
    gains the value in every other block showing it (as `_with_fact`)."""

    grown = list(groups)
    for operand in computed.operands:
        for index, group in enumerate(grown):
            if values := _fact_values(operand, group, set(operand.block_ids)):
                grown[index] = _with_alternatives(
                    operand, group, values[0], blocks, chunks
                )
                break
        else:
            message = f"no quote shows {operand.concept} {operand.value}"
            found.errors.append(("operand_not_quoted", message))
    return tuple(grown)


def _fact_values(
    fact: XbrlFact, group: Sequence[Evidence], on: Collection[str]
) -> list[str]:
    """The fact's value as written in the group's quotes on blocks `on`."""

    return [
        v
        for e in group
        if e.block_id in on
        for v in value_tokens(e.quote, fact.value, fact.decimals, any_sign=True)
    ]


def _with_alternatives(
    fact: XbrlFact,
    group: tuple[Evidence, ...],
    token: str,
    blocks: Mapping[str, Block],
    chunks: Mapping[str, Sequence[str]],
) -> tuple[Evidence, ...]:
    present = {e.block_id for e in group}
    return group + tuple(
        Evidence(block_id=b, quote=line)
        for b in fact.block_ids
        if b not in present and b in blocks
        for line in _value_line(token, blocks[b], chunks.get(b, ()))
    )


def _value_line(token: str, block: Block, chunks: Sequence[str]) -> list[str]:
    """The first chunk line of `block` showing `token` as a whole number (not
    `12.5` for `2.5`, nor cells `1 | 558` for `1,558`), as a list of 0 or 1."""

    pattern = whole_number(token)
    for chunk in chunks:
        for line in chunk.splitlines():
            if pattern.search(line) and normalize(line) in block.text:
                return [line.strip()]
    return []


def _row(line: str) -> tuple[str, str]:
    """A table line's first cell (label: normalized, without note references
    or footnote marks) and its other cells."""

    label, _, rest = line.partition("|")
    return normalize(_LABEL_REFERENCE.sub("", label)), rest


def _significant(token: str) -> int:
    """Significant digits as written: `1,287` 4, `100` 1, `56.6` 3."""

    return len(re.sub(r"\D", "", token).strip("0"))


@dataclass(frozen=True, slots=True)
class _Seed:
    """A group's evidence row as widening matches it."""

    cik: str
    label: str
    token: str  # the answer value
    others: tuple[str, ...]  # the row's other numbers

    def matches(self, cells: str) -> bool:
        """`cells` show the value; a value with few significant digits (EPS,
        a ratio) also needs one of the row's other numbers, so the same
        figure of another period or share class does not match."""

        if not whole_number(self.token).search(cells):
            return False
        return _significant(self.token) >= VALUE_SPECIFIC_DIGITS or any(
            whole_number(o).search(cells) for o in self.others
        )


_CELL_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _seed(
    group: Sequence[Evidence],
    answer: str,
    fact: XbrlFact | None,
    cik: str,
    taken: Collection[str] = (),
) -> _Seed | None:
    """A group's evidence row and its answer value: with a fact, the cell
    showing it with the most significant digits; else the cell number the
    reference `answer` states first (the value asked; the next ones are
    mostly comparatives), skipping values earlier groups `taken`. At least
    VALUE_MIN_DIGITS significant digits (VALUE_SPECIFIC_DIGITS under a short
    or generic label)."""

    for evidence in group:
        label, cells = _row(evidence.quote)
        if not label or not cells:
            continue
        row_numbers = _CELL_NUMBER.findall(cells)
        if fact is not None:
            tokens = value_tokens(cells, fact.value, fact.decimals, any_sign=True)
            best = max(tokens, key=_significant, default="")
        else:
            stated = {
                t: m.start()
                for t in row_numbers
                if t not in taken and (m := whole_number(t).search(answer))
            }
            best = min(stated, key=stated.__getitem__, default="")
        generic = len(label) < LABEL_MIN_CHARS or label in GENERIC_LABELS
        least = VALUE_SPECIFIC_DIGITS if generic else VALUE_MIN_DIGITS
        if best and _significant(best) >= least:
            others = tuple(dict.fromkeys(n for n in row_numbers if n != best))
            return _Seed(cik, label, best, others)
    return None


def _table_rows(
    chunks: Iterable[CompanyChunk],
) -> dict[str, dict[str, list[tuple[CompanyChunk, str, str]]]]:
    """CIK -> table line label -> (chunk, stripped line, other cells), in order."""

    rows: dict[str, dict[str, list[tuple[CompanyChunk, str, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for chunk in chunks:
        for line in chunk.text.splitlines():
            if "|" in line:  # a line without cells holds no value
                label, rest = _row(line)
                rows[chunk.cik][label].append((chunk, line.strip(), rest))
    return rows


def widen_evidence(
    result: CheckResult,
    tasks: Sequence[EvalTask],
    load_company: CompanyChunkLoader,
    read: BlockLoader,
) -> None:
    """Add, to each evidence group quoting a table row, every chunk row of
    the group's company with that row's label and the answer value
    (comparatives in later filings, the 10-Q/10-K twin of an 8-K release
    table, the same row in a note), as (block, row) evidence. The XBRL fact
    gives the value of a numeric question's first group. `load_company`
    gets the CIKs of the groups that have a seed row."""

    by_task = {t.task_id: t for t in tasks}
    seeds: dict[tuple[str, int], _Seed] = {}  # (question ID, group index)
    for question in result.questions:
        task = by_task[question.task_id]
        taken: set[str] = set()
        for index, group in enumerate(question.evidence_groups):
            fact = _group_fact(task, index, group)
            cik = _group_cik(group, task)
            answer = question.reference_answer
            if seed := _seed(group, answer, fact, cik, taken):
                seeds[question.id, index] = seed
                taken.add(seed.token)
    rows = _table_rows(load_company({s.cik for s in seeds.values()}))
    found: dict[tuple[str, int], list[tuple[CompanyChunk, str]]] = {}
    for key, seed in seeds.items():
        found[key] = [
            (chunk, line)
            for chunk, line, rest in rows[seed.cik][seed.label]
            if seed.matches(rest)
        ]
    hits = [hit for matched in found.values() for hit in matched]
    blocks = read(
        {b for chunk, _ in hits for b in chunk.block_ids},
        {c.filing_date for c, _ in hits},
    )
    for position, question in enumerate(result.questions):
        groups = [list(g) for g in question.evidence_groups]
        for index, grown in enumerate(groups):
            present = {e.block_id for e in grown}
            for chunk, line in found.get((question.id, index), ()):
                text = normalize(line)
                held = [
                    b for b in chunk.block_ids if b in blocks and text in blocks[b].text
                ]
                if held and held[0] not in present:
                    grown.append(Evidence(block_id=held[0], quote=line, widened=True))
                    present.add(held[0])
        before = sum(len(g) for g in question.evidence_groups)
        if added := sum(len(g) for g in groups) - before:
            result.widened[question.id] = added
            update = {"evidence_groups": tuple(tuple(g) for g in groups)}
            result.questions[position] = question.model_copy(update=update)


def prune_hard_negatives(result: CheckResult, texts: Mapping[str, str]) -> None:
    """Drop from each question's `hard_negative_chunks` the chunks whose text
    (`texts`: chunk ID -> raw text) shows an evidence quote or the answer
    value (the fact, or an operand of `computed`): those are evidence."""

    for position, question in enumerate(result.questions):
        if not question.hard_negative_chunks:
            continue
        kept = tuple(
            c
            for c in question.hard_negative_chunks
            if not _shows_evidence(question, texts.get(c, ""))
        )
        if dropped := len(question.hard_negative_chunks) - len(kept):
            result.pruned[question.id] = dropped
            update = {"hard_negative_chunks": kept}
            result.questions[position] = question.model_copy(update=update)


def _shows_evidence(question: EvalQuestion, text: str) -> bool:
    """`text` shows one of the question's quotes, its fact or an operand."""

    packed = normalize(text)
    if any(normalize(e.quote) in packed for g in question.evidence_groups for e in g):
        return True
    values = [question.fact] if question.fact is not None else []
    if question.computed is not None:
        values += question.computed.operands
    return any(
        value_tokens(line, v.value, v.decimals, any_sign=True)
        for v in values
        for line in text.splitlines()
    )


def _group_fact(
    task: EvalTask, index: int, group: Sequence[Evidence]
) -> XbrlFact | None:
    """The fact giving a group's widening value: a numeric question's fact
    (first group), else the first operand quoted in the group."""

    if task.fact is not None:
        return task.fact if index == 0 else None
    if task.computed is None or task.question_type not in OPERAND_TYPES:
        return None
    held = {e.block_id for e in group}
    return next((o for o in task.computed.operands if held & set(o.block_ids)), None)


def _group_cik(group: Sequence[Evidence], task: EvalTask) -> str:
    """CIK of the passage holding the group's first block (each compared
    company's own passage)."""

    for passage in task.passages:
        if group[0].block_id in passage.block_ids:
            return passage.cik
    return task.passages[0].cik


# Wording


def wording(task: EvalTask, draft: DraftQuestion, found: _Found) -> None:
    english, vietnamese = draft.question_en.strip(), draft.question_vi.strip()
    if english.casefold() == vietnamese.casefold():
        found.errors.append(("question_untranslated", "question_vi equals question_en"))
    texts = {"en": english, "vi": vietnamese}
    plans = {k: rule_plan(t, {}) for k, t in texts.items()}
    periods = {
        k: (p.fiscal_years or _written_years(texts[k]), p.fiscal_periods)
        for k, p in plans.items()
    }
    if periods["en"] != periods["vi"]:
        message = f"en {periods['en']} vs vi {periods['vi']}"
        # Only a period kind (FY, Q) on one side, no year on either: the
        # parser reads "năm tài chính" as FY; a human pass decides.
        kind_only = not periods["en"][0] and not periods["vi"][0]
        (found.warnings if kind_only else found.errors).append(
            ("period_mismatch", message)
        )
    for language, plan in plans.items():
        relative = RELATIVE_TIME.search(texts[language]) is not None
        if not (
            plan.has_period or plan.wants_latest or relative or _no_period_needed(task)
        ):
            found.warnings.append(
                ("no_period", f"{language}: no year, quarter or recency")
            )
    if not any(not c.isascii() and c.isalpha() for c in vietnamese):
        found.warnings.append(
            ("vi_no_diacritics", "question_vi has no Vietnamese letters")
        )
    if task.question_type == "unanswerable":
        _hint_warnings(task, texts, found)
    else:
        _company_warnings(task, texts, found)
    if task.passages:
        run, share = copy_scores(english, task)
        if run >= COPY_RUN or share >= COPY_SHARE:
            message = f"shared run {run} content words, share {share:.2f}"
            found.warnings.append(("lexical_copy", message))
    fact = task.fact
    if fact and not value_tokens(
        draft.reference_answer, fact.value, fact.decimals, sign_words=True
    ):
        found.warnings.append(("answer_without_value", f"fact value {fact.value}"))


def _written_years(text: str) -> tuple[int, ...]:
    """Years written as `6/2026` or `2026` that the rule parser skipped
    (only used when it found none)."""

    return tuple(sorted({int(y) for y in _YEAR.findall(text)}))


def _no_period_needed(task: EvalTask) -> bool:
    """Unanswerable by company or topic: the question needs no period."""

    return task.unanswerable_reason in ("company_outside_corpus", "topic_absent")


def _company_warnings(task: EvalTask, texts: Mapping[str, str], found: _Found) -> None:
    for company in task.companies:
        for language, text in texts.items():
            if not names_company(text, company):
                found.warnings.append(("no_company", f"{language}: {company.name}"))
        if not uses_style(texts["en"], company, task.mention_style):
            message = f"en: no {task.mention_style} of {company.name}"
            found.warnings.append(("mention_style", message))


def _hint_warnings(task: EvalTask, texts: Mapping[str, str], found: _Found) -> None:
    hint = task.hint or ""
    for language, text in texts.items():
        if task.unanswerable_reason == "company_outside_corpus":
            if not _has_phrase(text, normalize_name(hint)):
                found.warnings.append(("hint_missing", f"{language}: {hint}"))
        elif task.unanswerable_reason == "period_not_covered":
            years = set(rule_plan(hint, {}).fiscal_years)
            if not years <= set(rule_plan(text, {}).fiscal_years):
                found.warnings.append(("hint_missing", f"{language}: {hint}"))


def _ascii_words(text: str) -> str:
    folded = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return " " + " ".join(re.sub(r"[^a-z0-9]+", " ", folded.lower()).split()) + " "


def _has_phrase(text: str, phrase: str) -> bool:
    return bool(phrase) and f" {phrase} " in _ascii_words(text)


def _has_ticker(text: str, ticker: str) -> bool:
    pattern = rf"(?<![A-Za-z0-9]){re.escape(ticker)}(?![A-Za-z0-9])"
    return re.search(pattern, text) is not None


def company_name(name: str) -> str:
    """`normalize_name` after an SEC state suffix: `VALERO ENERGY CORP/TX`
    -> `valero energy`, `WALT DISNEY CO/` -> `walt disney`."""

    return normalize_name(_STATE_SUFFIX.sub("", name))


def name_phrases(company: CompanyRef) -> list[str]:
    """Phrases naming the company: each name, alias and former name
    (`company_name`), and each distinctive word of them (4+ letters, not in
    GENERIC_NAME_WORDS): `hormel` for HORMEL FOODS, `disney` for WALT DISNEY."""

    phrases: list[str] = []
    for name in (company.name, *company.aliases, *company.former_names):
        if phrase := company_name(name):
            phrases.append(phrase)
            phrases += [
                w
                for w in phrase.split()
                if len(w) >= NAME_WORD_MIN
                and w not in GENERIC_NAME_WORDS
                and not w.isdigit()
            ]
    return list(dict.fromkeys(phrases))


def names_company(text: str, company: CompanyRef) -> bool:
    """Named by a ticker of 2+ letters, a name phrase (`name_phrases`)."""

    return bool(mention_positions(text, company))


def uses_style(text: str, company: CompanyRef, style: str) -> bool:
    """The question names the company as `mention_style` says; a style the
    company has no value for (no ticker, alias) is not checked."""

    if style == "ticker" and company.tickers:
        return any(_has_ticker(text, t) for t in company.tickers)
    if style == "nickname":
        return any(_has_phrase(text, company_name(a)) for a in company.aliases)
    if style == "former_name":
        return any(_has_phrase(text, company_name(n)) for n in company.former_names)
    return True


def _fold(text: str) -> str:
    """`text` lowercased and ASCII-folded char by char (same length)."""

    return "".join(
        (unicodedata.normalize("NFKD", c).encode("ascii", "ignore").decode() or " ")[:1]
        for c in text.lower()
    )


def mention_positions(text: str, company: CompanyRef) -> list[int]:
    """Offsets in `text` where it names the company (as `names_company`)."""

    folded = _fold(text)
    found: list[int] = []
    for phrase in name_phrases(company):
        joined = r"[^a-z0-9]*".join(map(_name_word, phrase.split()))
        pattern = rf"(?<![a-z0-9]){joined}(?![a-z0-9])"
        found += [m.start() for m in re.finditer(pattern, folded)]
    for ticker in company.tickers:
        if len(ticker) >= TICKER_MIN:
            pattern = rf"(?<![A-Za-z0-9]){re.escape(ticker)}(?![A-Za-z0-9])"
            found += [m.start() for m in re.finditer(pattern, text)]
    return sorted(set(found))


def _name_word(word: str) -> str:
    """Regex of a name word: `and` also as `&` (AT&T), a final `s` also
    after an apostrophe (Lowe's for `lowes`)."""

    if word == "and":
        return "(?:and|&)"
    if len(word) > 2 and word.endswith("s"):
        return re.escape(word[:-1]) + "[' ]?s"
    return re.escape(word)


def _company_position(
    text: str, company: CompanyRef, fact: XbrlFact | None
) -> int | None:
    """Where `text` places the company: its mention nearest the first
    statement of its value (`fact`), else its last mention."""

    mentions = mention_positions(text, company)
    if not mentions:
        return None
    tokens = (
        []
        if fact is None
        else value_tokens(text, fact.value, fact.decimals, any_sign=True)
    )
    value = next(
        (m.start() for t in tokens if (m := whole_number(t).search(text))), None
    )
    if value is None:
        return mentions[-1]
    return min(mentions, key=lambda m: abs(m - value))


def asks_lowest(*texts: str) -> bool:
    """A question or answer about the lowest (en/vi): a ranking may then
    list companies lowest first."""

    return any(LOWEST.search(t) for t in texts)


def in_order(
    text: str,
    companies: Sequence[CompanyRef],
    facts: Sequence[XbrlFact | None] = (),
    *,
    reverse: bool = False,
) -> bool:
    """`text` names every company in this order (each placed by its mention
    nearest its value, `facts` parallel to `companies`), or in the reverse
    order when `reverse` (the question asks for the lowest)."""

    values = [*facts, *[None] * len(companies)][: len(companies)]
    found = [
        _company_position(text, c, f) for c, f in zip(companies, values, strict=True)
    ]
    positions = [p for p in found if p is not None]
    if len(set(positions)) < len(found):
        return False
    return positions == sorted(positions) or (
        reverse and positions == sorted(positions, reverse=True)
    )


def ranking_holds(
    computed: Computed, companies: Sequence[CompanyRef], text: str, *asked: str
) -> bool:
    """`text` ranks every company of `computed.order` in order (reversed
    only when `text` or the question `asked` wants the lowest). Operands are
    parallel to `order` when as many."""

    ranked = ranked_companies(computed, companies)
    if len(ranked) != len(computed.order):
        return False
    facts = computed.operands if len(computed.operands) == len(ranked) else ()
    return in_order(text, ranked, facts, reverse=asks_lowest(text, *asked))


def ranked_companies(
    computed: Computed, companies: Sequence[CompanyRef]
) -> list[CompanyRef]:
    """`computed.order` (CIKs) as companies; CIKs not listed are dropped."""

    by_cik = {c.cik: c for c in companies}
    return [by_cik[cik] for cik in computed.order if cik in by_cik]


def denies_premise(text: str) -> bool:
    """`text` refuses a premise: a negation or correction word (en/vi)."""

    return PREMISE_DENIAL.search(text) is not None


def states_computed_answer(computed: Computed, text: str) -> bool | None:
    """`text` states a computed value (None for a ranking), with the sign of
    a growth, CAGR or difference (`states_direction`)."""

    if computed.value is None:
        return None
    if not states_computed(text, computed.value, computed.unit):
        return False
    signed = computed.operation in SIGNED_OPERATIONS
    return not signed or states_direction(text, computed.value)


def _denied_near(text: str, company: CompanyRef) -> bool:
    """Every sentence naming the company has a negation (`denies_premise`)."""

    sentences = re.split(r"(?<=[.;!?])\s+|\n", text)
    named = [t for t in sentences if mention_positions(t, company)]
    return all(denies_premise(t) for t in named)


def copy_scores(question: str, task: EvalTask) -> tuple[int, float]:
    """Longest content-word run shared with the passages, and share of content
    words found in them (0 for numeric and table_lookup)."""

    names = {
        w
        for c in task.companies
        for n in (c.name, *c.aliases, *c.tickers)
        for w in words(n)
    }
    asked = content_words(question, names)
    passage = content_words(" ".join(p.text for p in task.passages), names)
    matcher = SequenceMatcher(None, asked, passage, autojunk=False)
    run = matcher.find_longest_match().size
    if len(asked) < COPY_MIN_WORDS or task.question_type in ("numeric", "table_lookup"):
        return run, 0.0
    vocabulary = set(passage)
    return run, sum(w in vocabulary for w in asked) / len(asked)


def content_words(text: str, exclude: Collection[str] = ()) -> list[str]:
    return [
        w
        for w in words(text)
        if len(w) > 2
        and not any(c.isdigit() for c in w)
        and w not in STOPWORDS
        and w not in PERIOD_WORDS
        and w not in exclude
    ]


def answer_facts(task: EvalTask, draft: DraftQuestion, found: _Found) -> None:
    """Warn on numbers and full dates of the reference answer that the
    passages (text or context) do not show. A number may also be a sum,
    difference, ratio or change of two numbers of the evidence quotes or of
    the answer itself."""

    if task.question_type == "unanswerable":
        return
    passages = "\n".join(f"{p.context}\n{p.text}" for p in task.passages)
    answer = draft.reference_answer
    known_dates = set(dates(passages).values()) - {None}
    missing = [w for w, d in dates(answer).items() if d not in known_dates]
    stated = numbers(without_dates(answer))
    shown = [n.value for n in numbers(without_dates(passages))]
    unknown = [n.written for n in stated if n.checked and not states(n.written, shown)]
    if unknown:
        quoted = " ".join(q for group in draft.evidence_quotes for q in group)
        known = [n for n in stated if n.written not in unknown]
        operands = {
            n.value for n in [*numbers(without_dates(quoted)), *known] if n.checked
        }
        missing += [w for w in unknown if not derived(w, operands)]
    if missing:
        message = "not in passages: " + "; ".join(dict.fromkeys(missing))
        found.warnings.append(("answer_fact_not_in_passage", message))


def computed_answer(task: EvalTask, draft: DraftQuestion, found: _Found) -> None:
    """Computed answers state the value with its direction; a `ranking`
    names its companies in order; a `false_premise` reference denies the
    premise; an `aggregation` names every company, those without the topic
    with a negation."""

    answer, computed = draft.reference_answer, task.computed
    if computed is not None and states_computed_answer(computed, answer) is False:
        message = f"{computed.operation} {computed.value} {computed.unit or ''}"
        found.errors.append(("answer_without_computed", message.strip()))
    if task.question_type == "ranking" and computed is not None:
        asked = (draft.question_en, draft.question_vi)
        if not ranking_holds(computed, task.companies, answer, *asked):
            names = ", ".join(
                c.name for c in ranked_companies(computed, task.companies)
            )
            found.errors.append(("ranking_order", f"expected order {names}"))
    if task.question_type == "false_premise" and not denies_premise(answer):
        found.warnings.append(("premise_not_denied", task.hint or ""))
    if task.question_type == "aggregation":
        absent = without_topic(task)
        for company in task.companies:
            if not names_company(answer, company):
                found.warnings.append(("answer_company_missing", company.name))
            elif company.cik in absent and not _denied_near(answer, company):
                found.warnings.append(("without_topic_affirmed", company.name))


def relative_wording(task: EvalTask, draft: DraftQuestion, found: _Found) -> None:
    """Warn when the `relative_time` tag and relative period wording
    ("last quarter", "năm ngoái") disagree."""

    tagged = "relative_time" in task.tags
    for language, text in (("en", draft.question_en), ("vi", draft.question_vi)):
        if tagged != bool(RELATIVE_TIME.search(text)):
            state = "no relative period" if tagged else "relative period, untagged"
            found.warnings.append(("relative_time", f"{language}: {state}"))


# Questions and report


def build_question(
    set_id: str,
    task: EvalTask,
    draft: DraftQuestion,
    groups: tuple[tuple[Evidence, ...], ...],
) -> EvalQuestion:
    outside = (
        task.hint if task.unanswerable_reason == "company_outside_corpus" else None
    )
    return EvalQuestion(
        id=f"{set_id}-{task.task_id}",
        task_id=task.task_id,
        question_en=draft.question_en.strip(),
        question_vi=draft.question_vi.strip(),
        reference_answer=draft.reference_answer.strip(),
        answerable=task.question_type != "unanswerable",
        question_type=task.question_type,
        data_kind=task.data_kind,
        persona=task.persona,
        mention_style=task.mention_style,
        companies=task.companies,
        evidence_groups=groups,
        fact=task.fact,
        expected=ExpectedFilters(
            ciks=tuple(dict.fromkeys(c.cik for c in task.companies)),
            unknown_companies=(outside,) if outside else (),
        ),
        cell=task.cell,
        computed=task.computed,
        tags=task.tags,
        hard_negative_chunks=task.hard_negative_chunks,
    )


def assign_splits(questions: Sequence[EvalQuestion], seed: int) -> list[EvalQuestion]:
    """Each question in `dev` when sha256 of `{seed}:{task_id}` is below
    DEV_SHARE of 2**256, else `test`: decided per question, so adding or
    dropping questions never moves another one (report shows the balance)."""

    bound = int(DEV_SHARE * 2**256)

    def split(question: EvalQuestion) -> Split:
        digest = hashlib.sha256(f"{seed}:{question.task_id}".encode()).digest()
        return "dev" if int.from_bytes(digest) < bound else "test"

    return [q.model_copy(update={"split": split(q)}) for q in questions]


def question_json(question: EvalQuestion) -> str:
    """A question's JSON line, without v2 fields at their defaults (V2_FIELDS,
    V2_COMPANY_FIELDS), so a v1.x set rewrites byte-identical."""

    return question.model_dump_json(exclude=_unset_fields(question))


def write_questions(path: Path, questions: Iterable[EvalQuestion]) -> int:
    """`questions.jsonl`, as `testset.write_jsonl` (temp file, then rename)
    with `question_json` lines; returns the count."""

    lines = [question_json(q) + "\n" for q in questions]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(lines), encoding="utf-8")
    temporary.replace(path)
    return len(lines)


def _unset_fields(question: EvalQuestion) -> dict[str, bool | dict[int, set[str]]]:
    defaults = EvalQuestion.model_fields
    exclude: dict[str, bool | dict[int, set[str]]] = {
        name: True
        for name in V2_FIELDS
        if getattr(question, name)
        == defaults[name].get_default(call_default_factory=True)
    }
    company_defaults = CompanyRef.model_fields
    companies = {
        index: {
            name
            for name in V2_COMPANY_FIELDS
            if getattr(company, name) == company_defaults[name].get_default()
        }
        for index, company in enumerate(question.companies)
    }
    if any(companies.values()):
        exclude["companies"] = {i: f for i, f in companies.items() if f}
    return exclude


def near_duplicates(questions: Sequence[EvalQuestion]) -> list[Issue]:
    sets = [set(words(q.question_en)) for q in questions]
    issues: list[Issue] = []
    for (i, a), (j, b) in itertools.combinations(enumerate(sets), 2):
        score = len(a & b) / len(a | b) if a | b else 1.0
        if score >= NEAR_DUPLICATE_JACCARD:
            message = f"close to {questions[j].task_id} (Jaccard {score:.2f})"
            issues.append(Issue(questions[i].task_id, "near_duplicate", message))
    return issues


def coverage(tasks: Iterable[EvalTask]) -> dict[str, Counter[str]]:
    """Accepted tasks per value of each dimension (once per task per value)."""

    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for task in tasks:
        values: dict[str, Iterable[object]] = {
            "question_type": [task.question_type],
            "data_kind": [task.data_kind],
            "persona": [task.persona],
            "mention_style": [task.mention_style],
            "company": {c.name for c in task.companies} or {"(none)"},
            **{
                key: {getattr(p, key) for p in task.passages}
                for key in ("form", "section_topic", "chunk_kind", "fiscal_year")
            },
            **{f"cell.{k}": [v] for k, v in task.cell.items()},
        }
        if task.tags:  # v2; v1.x reports keep their dimensions
            values["tag"] = task.tags
        for dimension, found in values.items():
            counts[dimension].update(str(v) for v in found)
    return dict(counts)


def render_report(result: CheckResult) -> str:
    lines = [
        "# Eval set check",
        "",
        f"- questions: {len(result.questions)}",
        f"- errors: {len(result.errors)}",
        f"- warnings: {len(result.warnings)}",
        f"- skipped: {sum(result.skipped.values())}"
        + "".join(f", {t} {n}" for t, n in sorted(result.skipped.items())),
    ]
    for title, issues in (("Errors", result.errors), ("Warnings", result.warnings)):
        lines += ["", f"## {title}", ""]
        if not issues:
            lines.append("None.")
            continue
        by_check = Counter(i.check for i in issues)
        lines.append(", ".join(f"`{c}` {n}" for c, n in sorted(by_check.items())))
        lines += ["", "| task | check | detail |", "|---|---|---|"]
        lines += [
            f"| {i.task_id} | {i.check} | {i.message.replace('|', '/')} |"
            for i in sorted(issues, key=lambda i: (i.task_id, i.check))
        ]
    lines += ["", "## Evidence from other filings", ""]
    if result.widened:
        lines += ["| question | rows added |", "|---|---|"]
        lines += [f"| {q} | {n} |" for q, n in sorted(result.widened.items())]
    else:
        lines.append("None.")
    if any(q.split for q in result.questions):
        lines += ["", "## Splits"]
        for title in ("question type", "tag"):
            splits = Counter(
                (v, q.split)
                for q in result.questions
                for v in (q.tags if title == "tag" else (q.question_type,))
            )
            lines += ["", f"| {title} | dev | test |", "|---|---|---|"]
            lines += [
                f"| {v} | {splits[v, 'dev']} | {splits[v, 'test']} |"
                for v in sorted({v for v, _ in splits})
            ]
    if result.pruned:
        lines += ["", "## Hard negatives dropped (show evidence)", ""]
        lines += ["| question | dropped |", "|---|---|"]
        lines += [f"| {q} | {n} |" for q, n in sorted(result.pruned.items())]
    lines += ["", "## Coverage (accepted questions)"]
    for dimension, counts in result.coverage.items():
        lines += ["", f"### {dimension}", "", "| value | questions |", "|---|---|"]
        lines += [f"| {v} | {n} |" for v, n in sorted(counts.items())]
    return "\n".join(lines) + "\n"


# Silver and Gold


def _needed(tasks: Iterable[EvalTask]) -> tuple[set[str], set[str], set[date]]:
    """Block IDs, document IDs and filing dates the tasks' evidence can use."""

    block_ids: set[str] = set()
    dates: set[date] = set()
    for task in tasks:
        for passage in task.passages:
            block_ids.update(passage.block_ids)
            dates.add(passage.filing_date)
        if task.fact is not None:
            block_ids.update(task.fact.block_ids)
    documents = {b.rsplit(":block:", 1)[0] for b in block_ids}
    return block_ids, documents, dates


def load_blocks(data_dir: Path, tasks: Iterable[EvalTask]) -> dict[str, Block]:
    """Silver `content_blocks` rows of the tasks' blocks (read-only, 3 columns)."""

    wanted, _, dates = _needed(tasks)
    return read_blocks(data_dir, wanted, dates)


def read_blocks(data_dir: Path, wanted: set[str], dates: set[date]) -> dict[str, Block]:
    """Silver `content_blocks` rows of `wanted` filed on `dates` (3 columns)."""

    if not wanted:
        return {}
    files = And(values_filter("block_id", wanted), filing_dates_filter(dates))
    with open_silver_catalog_read_only(data_dir) as catalog:
        table = catalog.load_table(CONTENT_BLOCKS_TABLE)
        rows = read_rows_in(
            table, "block_id", wanted, columns=("block_type", "text"), files=files
        )
    return {r["block_id"]: Block.of(r["text"], r["block_type"]) for r in rows}


def load_chunks(gold_dir: Path, tasks: Iterable[EvalTask]) -> dict[str, list[str]]:
    """block_id -> raw text of each Gold chunk holding it, for the tasks'
    documents only (read-only, 3 columns)."""

    wanted, documents, dates = _needed(tasks)
    found: dict[str, list[str]] = defaultdict(list)
    if not wanted:
        return found
    files = And(values_filter("document_id", documents), filing_dates_filter(dates))
    with open_gold_catalog_read_only(gold_dir) as catalog:
        table = catalog.load_table(CHUNKS_TABLE)
        rows = read_rows_in(
            table, "document_id", documents, columns=("block_ids", "text"), files=files
        )
    for row in rows:
        for block_id in wanted.intersection(row["block_ids"]):
            found[block_id].append(row["text"])
    return dict(found)


def load_company_chunks(gold_dir: Path, ciks: set[str]) -> list[CompanyChunk]:
    """Gold table chunks (with ` | ` rows) of the given companies (4 columns)."""

    found: list[CompanyChunk] = []
    if not ciks:
        return found
    wanted = pa.array(sorted(ciks), type=pa.string())
    columns = ("cik", "filing_date", "block_ids", "text")
    with open_gold_catalog_read_only(gold_dir) as catalog:
        table = catalog.load_table(CHUNKS_TABLE)
        for batch in scan_batches(table, columns, files=values_filter("cik", ciks)):
            keep = pc.and_(
                pc.is_in(batch["cik"], value_set=wanted),
                pc.match_substring(batch["text"], " | "),
            )
            found += [
                CompanyChunk(
                    r["cik"], r["filing_date"], tuple(r["block_ids"]), r["text"]
                )
                for r in batch.filter(keep).to_pylist()
            ]
    return found


def load_chunk_texts(gold_dir: Path, chunk_ids: Collection[str]) -> dict[str, str]:
    """Raw text of the given Gold chunks (read-only, 2 columns)."""

    ids = sorted(chunk_ids)
    if not ids:
        return {}
    with open_gold_catalog_read_only(gold_dir) as catalog:
        table = catalog.load_table(CHUNKS_TABLE)
        rows = read_rows_in(
            table,
            "chunk_id",
            ids,
            columns=("text",),
            files=values_filter("chunk_id", ids),
        )
    return {r["chunk_id"]: r["text"] for r in rows}
