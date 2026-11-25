"""Eval question set: sampled tasks, drafted questions, final questions.

Flow: `sampling` writes `EvalTask`s (passages chosen per coverage cell, with
their metadata from Gold/Silver); writers (agents) answer each task with a
`DraftQuestion` (wording only); `checks` merges both into `EvalQuestion`s.
Evidence is a Silver `block_id` plus a verbatim quote of the answer, never a
chunk ID, so a set outlives chunk rule changes: a retrieved chunk is relevant
when it holds the block and shows the quote (`normalize` on both sides), which
pins the right part of a table split over several chunks.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

QuestionType = Literal[
    "fact",  # one fact stated in text
    "numeric",  # one XBRL-tagged number
    "table_lookup",  # a value read from a table (not XBRL-tagged)
    "list",  # several items from one passage
    "summary",  # gist of a passage
    "compare_periods",  # one company, two periods
    "compare_companies",  # two companies, one period
    "multi_hop",  # two passages of one filing
    "unanswerable",  # the corpus has no answer
    # v2 types
    "calc",  # a figure derived from XBRL facts (growth, margin, share, CAGR)
    "ranking",  # order of 3-4 companies on one XBRL metric and period
    "temporal_change",  # what changed in one section between two years
    "event_8k",  # the event of an 8-K item
    "relation",  # a relationship the text states (supplier, officer...)
    "aggregation",  # which of N companies say something about a topic
    "false_premise",  # the question assumes what the filing contradicts
]
DataKind = Literal["text", "table", "xbrl", "none"]
MentionStyle = Literal["name", "ticker", "nickname", "former_name"]
Source = Literal["passage", "topic"]
UnanswerableReason = Literal[
    "company_outside_corpus", "period_not_covered", "topic_absent"
]
Language = Literal["en", "vi"]
LANGUAGES: tuple[Language, ...] = ("en", "vi")
Split = Literal["dev", "test"]
Operation = Literal["growth", "margin", "share", "cagr", "difference", "ranking"]
_SPACES = re.compile(r"[\s|]+")


def normalize(text: str) -> str:
    """Text as compared for quotes: lowercase, without whitespace or table
    pipes, so a row `a | b` matches cells `a\nb`, and `$ (9,605 )` split over
    cells matches `$(9,605)`."""

    return _SPACES.sub("", text).lower()


def whole_number(token: str) -> re.Pattern[str]:
    """`token` not inside a longer number (`2.5` in `12.5`, `558` in `1,558`)."""

    return re.compile(rf"(?<![\d.,]){re.escape(token)}(?!\d|[.,]\d)")


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# Keys of `EvalTask.cell` / `EvalQuestion.cell` (coverage dimensions).
CELL_KEYS = (
    "cik",
    "form",
    "document_type",
    "section_topic",
    "chunk_kind",
    "fiscal_year",
    "fiscal_period",
    "reason",  # unanswerable tasks
)


class SetManifest(_Model):
    """`set.json` of a question set: how it was sampled, and the date its
    relative questions ("latest", "last year") are read against."""

    set_id: str
    as_of: date  # passed to the retriever as `today` in every eval run
    seed: int
    total: int
    chunks_rules_version: int | None

    @classmethod
    def read(cls, set_dir: Path) -> SetManifest | None:
        path = set_dir / "set.json"
        if not path.exists():
            return None
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


class CompanyRef(_Model):
    cik: str = Field(pattern=r"^\d{10}$")
    name: str
    tickers: tuple[str, ...] = ()
    # Other names users call it by (`Google` for Alphabet).
    aliases: tuple[str, ...] = ()
    # Names it filed under before (SEC header `FORMER COMPANY`), newest first.
    former_names: tuple[str, ...] = ()
    sector: str | None = None  # GICS sector (IVV holdings), else None


class TaskPassage(_Model):
    """One Gold chunk shown to the writer, with its Silver blocks."""

    chunk_id: str
    document_id: str
    cik: str
    company_name: str
    form: str
    document_type: str
    section_topic: str
    section_title: str
    chunk_kind: str
    fiscal_year: int | None
    fiscal_period: str | None
    period_end_date: date | None
    filing_date: date
    context: str
    text: str
    block_ids: tuple[str, ...] = Field(min_length=1)


class XbrlFact(_Model):
    """The XBRL fact a numeric question asks for (exact answer)."""

    fact_id: str
    concept: str  # local name, e.g. `Revenues`
    value: str  # `numeric_value` as stored (no scale applied by us)
    unit: str | None  # e.g. `iso4217:USD`, `iso4217:USD/xbrli:shares`
    decimals: str | None
    period_start: date | None
    period_end: date | None
    instant: date | None
    block_id: str  # the occurrence shown in the task passage
    cell_id: str | None
    # Every block showing this fact (statement and notes): any one answers.
    block_ids: tuple[str, ...] = Field(min_length=1)


class Computed(_Model):
    """An answer code derives from XBRL facts (`calc`, `ranking`,
    `false_premise`): writers word it, never compute it."""

    operation: Operation
    # Result as a plain decimal string (growth/margin/share/cagr in percent,
    # difference in the operands' unit); None for `ranking`.
    value: str | None
    unit: str | None  # `percent`, or the operands' unit
    operands: tuple[XbrlFact, ...] = Field(min_length=1)
    # `ranking`: CIKs from highest to lowest value.
    order: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _shape(self) -> Computed:
        if (self.operation == "ranking") != (self.value is None):
            raise ValueError("ranking has an order and no value; others a value")
        if (self.operation == "ranking") != bool(self.order):
            raise ValueError("only ranking has an order")
        return self


class EvalTask(_Model):
    """What a writer gets: a coverage cell and the passages to ask about."""

    task_id: str
    question_type: QuestionType
    data_kind: DataKind
    persona: str
    mention_style: MentionStyle
    companies: tuple[CompanyRef, ...]
    passages: tuple[TaskPassage, ...] = ()
    fact: XbrlFact | None = None
    unanswerable_reason: UnanswerableReason | None = None
    # Free-text hint for unanswerable tasks (e.g. "Tesla", "FY2015").
    hint: str | None = None
    # Coverage dimensions: CELL_KEYS, plus type-specific ones.
    cell: dict[str, str] = Field(default_factory=dict)
    computed: Computed | None = None
    # Traps and groups (`fiscal_calendar`, `relative_time`, `graph`...).
    tags: tuple[str, ...] = ()
    # Chunks easy to confuse with the evidence (same concept or heading in
    # another year, segment or company): a run reports hits on them.
    hard_negative_chunks: tuple[str, ...] = ()
    # Fixed split (tasks added to grow `dev` only); None: by hash.
    split: Split | None = None

    @model_validator(mode="after")
    def _consistent(self) -> EvalTask:
        unanswerable = self.question_type == "unanswerable"
        if unanswerable != (self.unanswerable_reason is not None):
            raise ValueError("unanswerable tasks need a reason, others none")
        if not unanswerable and not self.passages:
            raise ValueError("answerable tasks need passages")
        if (self.question_type == "numeric") != (self.fact is not None):
            raise ValueError("numeric tasks, and only they, carry a fact")
        computed = self.question_type in ("calc", "ranking")
        if computed and self.computed is None:
            raise ValueError("calc and ranking tasks need `computed`")
        if self.question_type == "false_premise" and not self.hint:
            raise ValueError("false_premise tasks state the premise in `hint`")
        return self


class DraftQuestion(_Model):
    """A writer's answer to one task: wording only, no metadata."""

    task_id: str
    question_en: str = Field(min_length=10)
    question_vi: str = Field(min_length=10)
    # English; for unanswerable tasks, what a correct refusal says.
    reference_answer: str = Field(min_length=1)
    # One group per passage the answer needs: verbatim quotes (a line or
    # sentence, <= 300 chars) copied from that passage's `text` that hold the
    # answer; any one quote finds the group. Empty for unanswerable tasks.
    evidence_quotes: tuple[tuple[str, ...], ...] = ()


class Evidence(_Model):
    block_id: str
    quote: str = Field(min_length=1)  # verbatim; compared after `normalize`
    # Added by `checks.widen_evidence` (same row elsewhere): counts for
    # retrieval, not for whether the filters admit the question's evidence.
    widened: bool = False


class ExpectedFilters(_Model):
    """What the query model should read from the question. Periods are not
    listed: the eval checks that the filters the retriever used admit the
    evidence chunks, which also covers fiscal vs calendar years."""

    ciks: tuple[str, ...] = ()
    # Company names the corpus lacks (unanswerable by company).
    unknown_companies: tuple[str, ...] = ()


class EvalQuestion(_Model):
    """A checked question of a frozen set."""

    id: str
    task_id: str
    source: Source = "passage"
    question_en: str
    question_vi: str
    reference_answer: str
    answerable: bool
    question_type: QuestionType
    data_kind: DataKind
    persona: str
    mention_style: MentionStyle
    companies: tuple[CompanyRef, ...] = ()
    # A group is found when a retrieved chunk matches any of its evidence.
    evidence_groups: tuple[tuple[Evidence, ...], ...]
    fact: XbrlFact | None = None
    expected: ExpectedFilters
    cell: dict[str, str] = Field(default_factory=dict)
    computed: Computed | None = None
    tags: tuple[str, ...] = ()
    hard_negative_chunks: tuple[str, ...] = ()
    split: Split | None = None  # v2+: `dev` or `test`; None for v1.x

    @model_validator(mode="after")
    def _evidence(self) -> EvalQuestion:
        if self.answerable != bool(self.evidence_groups):
            raise ValueError("answerable questions, and only they, have evidence")
        if any(not group for group in self.evidence_groups):
            raise ValueError("empty evidence group")
        return self

    def text(self, language: Language) -> str:
        return self.question_en if language == "en" else self.question_vi

    @property
    def evidence_blocks(self) -> frozenset[str]:
        return frozenset(e.block_id for group in self.evidence_groups for e in group)


# `--assign-splits`: a question is `dev` when sha256(`{seed}:{task_id}`) is
# below this share of 2**256 (each question on its own: stable as sets grow).
DEV_SHARE = 0.4


def hash_split(seed: int, task_id: str) -> Split:
    """A task's split by hash (`assign_splits`), unless the task fixes one."""

    digest = hashlib.sha256(f"{seed}:{task_id}".encode()).digest()
    return "dev" if int.from_bytes(digest) < int(DEV_SHARE * 2**256) else "test"


def read_jsonl[M: BaseModel](path: Path, model: type[M]) -> list[M]:
    with path.open(encoding="utf-8") as handle:
        return [model.model_validate_json(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[BaseModel]) -> int:
    """Write rows atomically (temp file, then rename); returns the count."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(row.model_dump_json() + "\n")
            count += 1
    temporary.replace(path)
    return count


def dump_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
