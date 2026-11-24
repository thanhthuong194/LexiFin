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
from decimal import Decimal, InvalidOperation
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
# A written number matches a fact when it rounds to it and has at least
# VALUE_MIN_DIGITS significant digits or is within VALUE_TOLERANCE.
VALUE_MIN_DIGITS = 3
VALUE_TOLERANCE = Decimal("0.005")
SKIP_PREFIX = "SKIP:"
# Same-company rows that widen numeric/table_lookup evidence must carry a
# specific label: at least LABEL_MIN_CHARS (normalized) and not generic,
# unless the value has VALUE_SPECIFIC_DIGITS significant digits.
LABEL_MIN_CHARS = 6
GENERIC_LABELS = frozenset({"total", "subtotal", "net", "other", "totals"})
VALUE_SPECIFIC_DIGITS = 4
WIDENED_TYPES = frozenset({"numeric", "table_lookup"})

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
_NUMBER = re.compile(
    r"\$?([(\-−])?\$?(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?\)?\s*"
    r"(%|percent\b|thousand\b|million\b|billion\b|trillion\b|bn\b|[kmb]\b)?",
    re.IGNORECASE,
)
_SCALES = {
    "%": Decimal("0.01"), "percent": Decimal("0.01"), "k": Decimal(10) ** 3,
    "thousand": Decimal(10) ** 3, "m": Decimal(10) ** 6,
    "million": Decimal(10) ** 6, "b": Decimal(10) ** 9, "bn": Decimal(10) ** 9,
    "billion": Decimal(10) ** 9, "trillion": Decimal(10) ** 12,
}  # fmt: skip
# Answer facts: numbers checked only when they carry `$`, a unit, a fraction or
# a thousands separator, or are at least ANSWER_MIN_NUMBER and not a year.
ANSWER_MIN_NUMBER = 100
_YEARS = range(1900, 2101)
_ANSWER_SCALES = tuple(Decimal(10) ** e for e in (0, 3, 6, 9, 12, -2))
_MONTHS = {
    m[:3]: i
    for i, m in enumerate(
        ("january", "february", "march", "april", "may", "june", "july",
         "august", "september", "october", "november", "december"),
        1,
    )
}  # fmt: skip
_DATE = re.compile(
    r"\b(?:(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?[ \t\n]+"
    r"(\d{1,2}),?[ \t\n]+(\d{4})|(\d{4})-(\d{2})-(\d{2}))\b",
    re.IGNORECASE,
)
_NEGATIVE = re.compile(
    r"\b(loss(es)?|negative|decreas\w*|declin\w*|outflows?|used|deficit)\b|\blỗ\b",
    re.IGNORECASE,
)


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
    """A table line's normalized first cell (label) and its other cells."""

    label, _, rest = line.partition("|")
    return normalize(label), rest


def _significant(token: str) -> int:
    """Significant digits as written: `1,287` 4, `100` 1, `56.6` 3."""

    return len(re.sub(r"\D", "", token).strip("0"))


def _seed(question: EvalQuestion, task: EvalTask) -> tuple[str, str] | None:
    """(label, value) of the question's evidence row: the value with the most
    significant digits, at least VALUE_MIN_DIGITS (VALUE_SPECIFIC_DIGITS
    under a short or generic label)."""

    for evidence in question.evidence_groups[0]:
        label, cells = _row(evidence.quote)
        if not label or not cells:
            continue
        if task.fact is not None:
            fact = task.fact
            tokens = value_tokens(cells, fact.value, fact.decimals, any_sign=True)
        else:  # table_lookup: a cell number the reference answer repeats
            found = re.findall(r"\d[\d,]*(?:\.\d+)?", cells)
            tokens = [
                t for t in found if whole_number(t).search(question.reference_answer)
            ]
        generic = len(label) < LABEL_MIN_CHARS or label in GENERIC_LABELS
        least = VALUE_SPECIFIC_DIGITS if generic else VALUE_MIN_DIGITS
        best = max(tokens, key=_significant, default="")
        if best and _significant(best) >= least:
            return label, best
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
    """Add, to numeric/table_lookup questions, every same-company chunk row with
    the evidence row's label and the answer value (comparatives in later
    filings, 8-K release tables), as (block, row) evidence. `load_company`
    gets the CIKs of the questions that have a seed row."""

    by_task = {t.task_id: t for t in tasks}
    seeds: dict[str, tuple[str, str, str]] = {}  # question ID -> cik, label, token
    for question in result.questions:
        task = by_task[question.task_id]
        if question.question_type in WIDENED_TYPES and (seed := _seed(question, task)):
            seeds[question.id] = (task.passages[0].cik, *seed)
    rows = _table_rows(load_company({cik for cik, _, _ in seeds.values()}))
    found: dict[str, list[tuple[CompanyChunk, str]]] = {}
    for question_id, (cik, label, token) in seeds.items():
        pattern = whole_number(token)
        found[question_id] = [
            (chunk, line)
            for chunk, line, rest in rows[cik][label]
            if pattern.search(rest)
        ]
    hits = [hit for matched in found.values() for hit in matched]
    blocks = read(
        {b for chunk, _ in hits for b in chunk.block_ids},
        {c.filing_date for c, _ in hits},
    )
    for index, question in enumerate(result.questions):
        if question.id not in found:
            continue
        group = list(question.evidence_groups[0])
        present = {e.block_id for e in group}
        for chunk, line in found[question.id]:
            text = normalize(line)
            held = [
                b for b in chunk.block_ids if b in blocks and text in blocks[b].text
            ]
            if held and held[0] not in present:
                group.append(Evidence(block_id=held[0], quote=line))
                present.add(held[0])
        if added := len(group) - len(question.evidence_groups[0]):
            result.widened[question.id] = added
            groups = (tuple(group), *question.evidence_groups[1:])
            update = {"evidence_groups": groups}
            result.questions[index] = question.model_copy(update=update)


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


def value_tokens(
    text: str,
    value: str,
    decimals: str | None,
    *,
    sign_words: bool = False,
    any_sign: bool = False,
) -> list[str]:
    """Numbers in `text` (as written) that state `value`, rounded and scaled.

    Sign must agree: a negative value is written `(1,558)` / `-1,558`, or, with
    `sign_words`, stated positive next to "loss", "decrease", "used"...
    `any_sign` skips this, for table rows whose sign is presentation (an
    outflow fact, positive in XBRL, shown as `(28,107)` in a cash-flow row).
    """

    try:
        target = Decimal(value)
    except InvalidOperation:
        return [value] if value in text else []
    implicit = [Decimal(1)]
    if decimals and decimals.lstrip("-").isdigit() and int(decimals) < 0:
        implicit.append(Decimal(10) ** -int(decimals))
    worded = sign_words and _NEGATIVE.search(text) is not None
    tokens: list[str] = []
    for sign, whole, fraction, unit in _NUMBER.findall(text):
        written = whole.replace(",", "") + fraction
        number = Decimal(written)
        if (number == 0) != (target == 0):
            continue
        if (
            number
            and not any_sign
            and (bool(sign) if target > 0 else not (sign or worded))
        ):
            continue
        digits = len(fraction) - 1 if fraction else 0
        significant = len(written.replace(".", "").lstrip("0"))
        for scale in [_SCALES[unit.casefold()]] if unit else implicit:
            error = abs(number * scale - abs(target))
            rounded = error <= Decimal(5) * Decimal(10) ** (-digits - 1) * scale
            close = significant >= VALUE_MIN_DIGITS or error <= VALUE_TOLERANCE * abs(
                target
            )
            if rounded and close:
                tokens.append(whole + fraction)
                break
    return tokens


def _dates(text: str) -> dict[str, date | None]:
    """Full dates written in `text` (`June 30, 2024`, `Sept. 28, 2024`,
    `2024-06-30`; not split over table cells), as written -> date."""

    found: dict[str, date | None] = {}
    for match in _DATE.finditer(text):
        month, day, year, iso_year, iso_month, iso_day = match.groups()
        try:
            if month:
                when = date(int(year), _MONTHS[month.casefold()], int(day))
            else:
                when = date(int(iso_year), int(iso_month), int(iso_day))
        except ValueError:
            when = None
        found[match.group(0)] = when
    return found


def _numbers(text: str) -> list[tuple[str, Decimal, bool]]:
    """Numbers in `text` not inside a word (`Q2`, `FY2024`): (as written,
    value without unit scale, whether it states an amount worth checking)."""

    found: list[tuple[str, Decimal, bool]] = []
    for match in _NUMBER.finditer(text):
        start = match.start()
        if start and (text[start - 1].isalnum() or text[start - 1] in ".,"):
            continue
        _, whole, fraction, unit = match.groups()
        value = Decimal(whole.replace(",", "") + (fraction or ""))
        marked = "$" in match.group(0) or bool(unit or fraction) or "," in whole
        checked = value != 0 and (
            marked or (value >= ANSWER_MIN_NUMBER and int(value) not in _YEARS)
        )
        found.append((match.group(0).strip(), value, checked))
    return found


def _states(written: str, values: Iterable[Decimal]) -> bool:
    """`written` (one answer number) states one of `values`, at any usual
    scale (units to trillions, or percent points)."""

    return any(
        value_tokens(written, str(abs(v) * scale), None, any_sign=True)
        for v in values
        for scale in _ANSWER_SCALES
    )


def _derived(written: str, operands: Collection[Decimal]) -> bool:
    """`written` is a sum or difference of two operands, or (a percentage) a
    ratio or change of two, to its shown rounding."""

    pairs = list(itertools.permutations(operands, 2))
    if _states(written, {a + b for a, b in pairs} | {a - b for a, b in pairs}):
        return True
    percent = re.search(r"(\d[\d,]*)(\.\d+)?\s*(%|percent)", written)
    if percent is None:
        return False
    shown = Decimal(percent.group(1).replace(",", "") + (percent.group(2) or ""))
    half = Decimal(5) * Decimal(10) ** -(len(percent.group(2) or "."))
    for a, b in pairs:
        if b and any(abs(abs(r) * 100 - shown) <= half for r in (a / b, (a - b) / b)):
            return True
    return False


def answer_facts(task: EvalTask, draft: DraftQuestion, found: _Found) -> None:
    """Warn on numbers and full dates of the reference answer that the
    passages (text or context) do not show. A number may also be a sum,
    difference, ratio or change of two numbers of the evidence quotes or of
    the answer itself."""

    if task.question_type == "unanswerable":
        return
    passages = "\n".join(f"{p.context}\n{p.text}" for p in task.passages)
    answer = draft.reference_answer
    known_dates = set(_dates(passages).values()) - {None}
    missing = [w for w, d in _dates(answer).items() if d not in known_dates]
    numbers = _numbers(_DATE.sub(" ", answer))
    shown = [v for _, v, _ in _numbers(_DATE.sub(" ", passages))]
    unknown = [w for w, _, checked in numbers if checked and not _states(w, shown)]
    if unknown:
        quoted = " ".join(q for group in draft.evidence_quotes for q in group)
        known = [(w, v, ok) for w, v, ok in numbers if w not in unknown]
        operands = {v for _, v, ok in [*_numbers(_DATE.sub(" ", quoted)), *known] if ok}
        missing += [w for w in unknown if not _derived(w, operands)]
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
