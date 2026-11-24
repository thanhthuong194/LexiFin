"""Merge writer drafts with their tasks into checked `EvalQuestion`s.

Each evidence quote is mapped to the Silver blocks of its passage whose text
contains it, and must show in a current Gold chunk holding that block.
Errors block the set (missing drafts, quotes not found, en/vi asking about
different periods); warnings are for a human pass (company, period or
mention style missing, copied wording, answer numbers or dates not in the
passages, near duplicates, skipped drafts).
"""

from __future__ import annotations

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
    value_tokens,
    without_dates,
)
from src.evals.testset import (
    CompanyRef,
    DraftQuestion,
    EvalQuestion,
    EvalTask,
    Evidence,
    ExpectedFilters,
    TaskPassage,
    XbrlFact,
    normalize,
    whole_number,
)
from src.rag_engine.retrieval.companies import normalize_name
from src.rag_engine.retrieval.query import rule_plan

QUOTE_MAX_CHARS = 300
QUOTE_MIN_CHARS = 12
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
    expected = len(task.passages) if task.question_type in PAIRED_TYPES else 1
    if len(quotes) != expected or not all(quotes):
        sizes = [len(g) for g in quotes]
        found.errors.append(("evidence_groups", f"groups {sizes}, expected {expected}"))
        return ()
    passages = _assign(task, quotes, found)
    if passages is None:
        return ()
    groups = tuple(
        tuple(e for q in group for e in _quote_evidence(q, p, blocks, chunks, found))
        for group, p in zip(quotes, passages, strict=True)
    )
    if task.fact is not None and not found.errors:
        groups = _with_fact(task.fact, groups, blocks, chunks, found)
    return groups


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
        found.errors.append(("quote_no_block", f"no single block holds {short!r}"))
        return []
    if text == normalize(passage.section_title) or all(
        blocks[b].block_type == "heading" for b in hits
    ):
        found.warnings.append(("quote_heading", repr(short)))
    packed = {c: normalize(c) for b in hits for c in chunks.get(b, ())}
    shown = [b for b in hits if any(text in packed[c] for c in chunks.get(b, ()))]
    if not shown:
        found.errors.append(("quote_not_in_chunk", f"no Gold chunk shows {short!r}"))
    return [Evidence(block_id=b, quote=quote) for b in shown]


def _with_fact(
    fact: XbrlFact,
    groups: tuple[tuple[Evidence, ...], ...],
    blocks: Mapping[str, Block],
    chunks: Mapping[str, Sequence[str]],
    found: _Found,
) -> tuple[tuple[Evidence, ...], ...]:
    """The fact's group, plus its value in every other block showing the fact."""

    group = groups[0]
    on_fact = [e.quote for e in group if e.block_id == fact.block_id]
    if not on_fact:
        found.errors.append(("evidence_without_fact", f"no quote on {fact.block_id}"))
        return groups
    values = [
        v
        for q in on_fact
        for v in value_tokens(q, fact.value, fact.decimals, any_sign=True)
    ]
    if not values:
        found.errors.append(("quote_without_value", f"no {fact.value} in the quote"))
        return groups
    present = {e.block_id for e in group}
    alternatives = tuple(
        Evidence(block_id=b, quote=line)
        for b in fact.block_ids
        if b not in present and b in blocks
        for line in _value_line(values[0], blocks[b], chunks.get(b, ()))
    )
    return (group + alternatives, *groups[1:])


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
            fact = task.fact if index == 0 else None
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
    plans = {"en": rule_plan(english, {}), "vi": rule_plan(vietnamese, {})}
    periods = {k: (p.fiscal_years, p.fiscal_periods) for k, p in plans.items()}
    if periods["en"] != periods["vi"]:
        found.errors.append(
            ("period_mismatch", f"en {periods['en']} vs vi {periods['vi']}")
        )
    for language, plan in plans.items():
        if not (plan.has_period or plan.wants_latest):
            found.warnings.append(
                ("no_period", f"{language}: no year, quarter or recency")
            )
    if not any(not c.isascii() and c.isalpha() for c in vietnamese):
        found.warnings.append(
            ("vi_no_diacritics", "question_vi has no Vietnamese letters")
        )
    texts = {"en": english, "vi": vietnamese}
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


def names_company(text: str, company: CompanyRef) -> bool:
    """Named by a ticker of 3+ letters, its name without legal suffixes, or an alias."""

    names = (company.name, *company.aliases)
    return any(_has_ticker(text, t) for t in company.tickers if len(t) > 2) or any(
        _has_phrase(text, normalize_name(n)) for n in names
    )


def uses_style(text: str, company: CompanyRef, style: str) -> bool:
    if style == "ticker":
        return any(_has_ticker(text, t) for t in company.tickers)
    if style == "nickname":
        return any(_has_phrase(text, normalize_name(a)) for a in company.aliases)
    return True


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
    )


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
