from __future__ import annotations

from collections import Counter
from datetime import date
from typing import Any

import pytest

from src.evals.sampling import (
    DEFAULT_PLAN,
    Picker,
    PlannedTask,
    assemble,
    company_pairs,
    fact_passage,
    filing_pairs,
    is_eligible,
    outside_companies,
    period_pairs,
    plan_tasks,
    plan_unanswerable,
    scale_plan,
)
from src.evals.sampling_data import (
    OUTSIDE_COMPANIES,
    TOPIC_PROBES,
    ChunkMeta,
    ChunkText,
    CorpusStats,
    FactCandidate,
    FilingInfo,
    Occurrence,
    cell_block,
    company_aliases,
    company_ref,
    fact_candidate,
    keep_text,
    readable_unit,
    record,
)

pytestmark = pytest.mark.unit

COMPANIES = {
    "0000000001": ("Alpha Corp", ("ALP",)),
    "0000000002": ("Beta Inc.", ("BET",)),
    "0000000003": ("Gamma Co", ("GAM",)),
}
FORMS = (("10-K", "10-K", "FY"), ("10-Q", "10-Q", "Q2"), ("8-K", "EX-99.1", None))
TOPICS = ("risk_factors", "mdna", "business", "properties")


def chunk(number: int, **fields: Any) -> ChunkMeta:
    cik = fields.pop("cik", "0000000001")
    name, tickers = COMPANIES[cik]
    values: dict[str, Any] = {
        "chunk_id": f"c{number:04d}",
        "filing_id": f"f-{cik}-{number % 7}",
        "document_id": f"d{number}",
        "block_ids": (f"b{number}",),
        "cik": cik,
        "company_name": name,
        "tickers": tickers,
        "form": "10-K",
        "document_type": "10-K",
        "section_topic": "risk_factors",
        "section_title": "Item 1A. Risk Factors",
        "subheading": f"Heading number {number}",
        "table_title": None,
        "chunk_kind": "text",
        "fiscal_year": 2024,
        "fiscal_period": "FY",
        "period_end_date": date(2024, 12, 31),
        "filing_date": date(2025, 2, 1),
        "part": None,
        "part_count": None,
        "chars": 2000,
        "list_lines": 4,
        "text_hash": number,
    }
    values.update(fields)
    return ChunkMeta(**values)


def corpus() -> list[ChunkMeta]:
    """3 companies x 3 forms x 4 topics x 2 kinds x 3 years."""

    chunks: list[ChunkMeta] = []
    for cik in COMPANIES:
        for form, document_type, period in FORMS:
            for topic in TOPICS:
                for kind in ("text", "table"):
                    for year in (2023, 2024, 2025):
                        chunks.append(
                            chunk(
                                len(chunks),
                                cik=cik,
                                form=form,
                                document_type=document_type,
                                fiscal_period=period,
                                fiscal_year=None if period is None else year,
                                period_end_date=None
                                if period is None
                                else date(year, 12, 31),
                                filing_date=date(year + 1, 2, 1),
                                section_topic="ex99" if period is None else topic,
                                chunk_kind=kind,
                                subheading=f"{topic} {kind} heading",
                                table_title=f"{topic} statement"
                                if kind == "table"
                                else None,
                            )
                        )
    return chunks


def test_scale_plan_keeps_proportions_and_total() -> None:
    assert scale_plan(100) == DEFAULT_PLAN
    assert scale_plan(300) == {q: 3 * n for q, n in DEFAULT_PLAN.items()}
    assert sum(scale_plan(37).values()) == 37


def test_exclusions() -> None:
    assert not is_eligible(chunk(1, section_topic="cover"))
    assert not is_eligible(chunk(1, chars=120))
    assert is_eligible(chunk(1, chunk_kind="table", chars=250))
    long_text = "We compete with many firms. " * 80
    assert keep_text("text", "mdna", "Competition", long_text)
    assert not keep_text("text", "governance", "", "is set forth in the Proxy. " * 80)
    assert not keep_text("text", "mdna", "", "Information is set forth in the Proxy.")
    assert not keep_text("text", "controls", "", "Our controls were reviewed.")
    assert not keep_text("text", "mdna", "Forward-Looking Statements", long_text)
    assert not keep_text("text", "mdna", "ir@alpha.com", long_text)
    assert not keep_text("table", "8k_event", "Broker Non-Votes 3,206,621,218", "x")


def test_titles_reject_bullets_rows_and_long_headings() -> None:
    assert chunk(1, subheading="Competition").title == "competition"
    assert chunk(1, subheading="• we sell chips; and").title == ""
    assert chunk(1, subheading="Total revenue 12,345").title == ""
    assert chunk(1, subheading="Word " * 30).title == ""
    table = chunk(1, chunk_kind="table", table_title="Alpha Corp BALANCE SHEETS")
    assert table.title == "balance sheets"


def test_topic_weights_favour_frequent_topics() -> None:
    picker = Picker(seed=7)
    picked = picker.chunks(corpus(), "fact", 18)
    topics = Counter(c.section_topic for c in picked)
    assert topics["risk_factors"] > topics["properties"]
    assert Counter(c.cik for c in picked) == {cik: 6 for cik in COMPANIES}
    # A company's heading is asked once per type.
    assert len({(c.cik, c.title) for c in picked}) == len(picked)


def test_plan_is_deterministic_and_unique() -> None:
    quotas = {**dict.fromkeys(DEFAULT_PLAN, 3), "numeric": 0, "unanswerable": 0}

    def ids(seed: int) -> list[tuple[str, ...]]:
        planned = plan_tasks(corpus(), quotas, Picker(seed))
        return [tuple(c.chunk_id for c in task.chunks) for task in planned]

    planned = ids(1)
    assert planned == ids(1)
    assert planned != ids(2)
    flat = [chunk_id for task in planned for chunk_id in task]
    assert len(flat) == len(set(flat))


def test_period_pairs_take_nearest_year_and_same_part() -> None:
    def table(number: int, year: int, part: int | None, text_hash: int) -> ChunkMeta:
        return chunk(
            number,
            chunk_kind="table",
            table_title="Alpha Corp Cash Flows",
            fiscal_year=year,
            part=part,
            part_count=2 if part else None,
            text_hash=text_hash,
        )

    pool = [
        table(1, 2021, None, 1),
        table(2, 2023, None, 2),
        table(3, 2024, None, 3),
        table(4, 2025, None, 3),  # same text as 2024: no comparison
        table(5, 2024, 1, 4),  # a part never pairs with a whole table
    ]
    pairs = period_pairs(pool, seed=1)
    assert [(p.first.chunk_id, p.second.chunk_id, p.penalty) for p in pairs] == [
        ("c0001", "c0002", 5),
        ("c0002", "c0003", 0),
    ]


def test_company_pairs_need_title_period_and_part() -> None:
    nvidia_like = chunk(
        1,
        cik="0000000002",
        fiscal_year=2025,
        period_end_date=date(2025, 1, 26),
        subheading="Income Taxes",
    )
    december = chunk(2, fiscal_year=2024, subheading="Income Taxes")
    other_title = chunk(3, cik="0000000003", subheading="Competition")
    other_part = chunk(
        4, cik="0000000003", subheading="Income Taxes", part=1, part_count=2
    )
    pairs = company_pairs([nvidia_like, december, other_title, other_part], seed=1)
    assert {(p.first.chunk_id, p.second.chunk_id) for p in pairs} in (
        {("c0001", "c0002")},
        {("c0002", "c0001")},
    )


def test_filing_pairs_are_linked() -> None:
    item = chunk(
        1,
        form="8-K",
        document_type="8-K",
        section_topic="8k_event",
        exhibit_refs=("99.1",),
        filing_id="f8",
    )
    exhibit = chunk(
        2, form="8-K", document_type="EX-99.01", section_topic="ex99", filing_id="f8"
    )
    other_exhibit = chunk(
        3, form="8-K", document_type="EX-99.2", section_topic="ex99", filing_id="f8"
    )
    mdna = chunk(
        4,
        section_topic="mdna",
        subheading="Income Taxes",
        note_refs=("9",),
        filing_id="fk",
    )
    note = chunk(
        5,
        section_topic="financial_statements",
        subheading="Income Taxes",
        note="9",
        filing_id="fk",
    )
    unrelated = chunk(6, section_topic="risk_factors", filing_id="fk")
    pairs = filing_pairs([item, exhibit, other_exhibit, mdna, note, unrelated], seed=1)
    found = {(p.first.chunk_id, p.second.chunk_id, p.kind) for p in pairs}
    assert found == {
        ("c0001", "c0002", "8k_exhibit"),
        ("c0004", "c0005", "mdna_note"),
        ("c0004", "c0005", "note_ref"),
    }


def test_cell_block_and_units() -> None:
    assert cell_block("doc:block:000012:table:cell:000004") == "doc:block:000012"
    assert cell_block(None) is None
    assert readable_unit("{http://www.xbrl.org/2003/iso4217}USD") == "iso4217:USD"
    assert (
        readable_unit(
            "({http://www.xbrl.org/2003/iso4217}USD)/({http://www.xbrl.org/2003/instance}shares)"
        )
        == "iso4217:USD/xbrli:shares"
    )


def fact(
    *occurrences: Occurrence, current: bool = True, concept: str = "Revenues"
) -> FactCandidate:
    filing = FilingInfo(
        "0000000001", "10-K", 2024, "FY", date(2024, 12, 31), date(2025, 2, 1)
    )
    row = {
        "fact_id": f"x-{concept}-{current}",
        "concept": concept,
        "numeric_value": "1234000000",
        "unit": "iso4217:USD",
        "decimals": "-6",
        "block_id": None,
        "cell_id": None,
    }
    end = date(2024, 12, 31) if current else date(2023, 12, 31)
    return fact_candidate(row, occurrences, "f1", filing, (date(2024, 1, 1), end, None))


def test_fact_candidate_keeps_every_occurrence() -> None:
    candidate = fact(
        Occurrence("b1", "b1:table:cell:000001", "1,234"), Occurrence("b2", None, "1.2")
    )
    assert candidate.is_current
    assert candidate.fact.block_ids == ("b1", "b2")
    assert not fact(Occurrence("b1", None, "1"), current=False).is_current


def test_fact_passage_prefers_statement_then_part_showing_value() -> None:
    note = chunk(
        1,
        block_ids=("b1",),
        chunk_kind="table",
        table_title="Revenue by product",
        note="3",
    )
    statement = [
        chunk(
            2,
            block_ids=("b2",),
            part=1,
            part_count=2,
            chunk_kind="table",
            table_title="CONSOLIDATED STATEMENTS OF OPERATIONS",
        ),
        chunk(
            3,
            block_ids=("b2",),
            part=2,
            part_count=2,
            chunk_kind="table",
            table_title="CONSOLIDATED STATEMENTS OF OPERATIONS",
        ),
    ]
    chunks_of = {"b1": [note], "b2": statement}
    texts = {
        "c0001": ChunkText("ctx", "Revenue | 1,234"),
        "c0002": ChunkText("ctx", "Cost | 11,234"),
        "c0003": ChunkText("ctx", "Revenue | 1,234"),
    }
    candidate = fact(Occurrence("b1", None, "1,234"), Occurrence("b2", None, "1,234"))
    found = fact_passage(candidate, chunks_of, texts)
    assert found is not None and found[0].chunk_id == "c0003"
    assert found[1].block_id == "b2"
    assert fact_passage(fact(Occurrence("b9", None, "1")), chunks_of, texts) is None


def test_facts_prefer_current_and_set_the_passage_block() -> None:
    current = fact(Occurrence("b1", None, "1"), Occurrence("b2", None, "1"))
    prior = fact(Occurrence("b2", None, "1"), current=False, concept="Assets")
    passages = {"b2": chunk(2, block_ids=("b2",))}

    def passage_of(candidate: FactCandidate) -> tuple[ChunkMeta, Occurrence] | None:
        for occurrence in candidate.occurrences:
            if occurrence.block_id in passages:
                return passages[occurrence.block_id], occurrence
        return None

    picked = Picker(seed=1).facts([prior, current], 2, passage_of)
    # The current fact takes chunk c0002; the prior one shares it and is skipped.
    assert [(f.concept, f.block_id, f.block_ids) for f, _ in picked] == [
        ("Revenues", "b2", ("b1", "b2"))
    ]


def stats() -> CorpusStats:
    found = CorpusStats()
    found.companies = {
        "0000000001": company_ref("0000000001", "Tesla, Inc.", ("TSLA",)),
        "0000000002": company_ref("0000000002", "Alphabet Inc.", ("GOOGL",)),
    }
    found.first_year = {"0000000001": 2021, "0000000002": 2022}
    found.years["0000000001"] = {2015, 2014}
    found.probe_hits["0000000001"] = set(range(len(TOPIC_PROBES) - 1))
    return found


def test_outside_names_drop_corpus_companies_and_mentions() -> None:
    corpus_stats = stats()
    row = {
        "chunk_id": "c1",
        "filing_id": "f1",
        "cik": "0000000002",
        "company_name": "Alphabet Inc.",
        "tickers": ["GOOGL"],
        "form": "10-K",
        "fiscal_year": 2022,
        "fiscal_period": "FY",
        "period_end_date": None,
        "filing_date": date(2023, 2, 1),
        "chunk_kind": "text",
        "text": "We compete with Netflix and ExxonMobil.",
    }
    record(corpus_stats, row)
    names = outside_companies(corpus_stats)
    assert "Tesla" not in names  # a corpus company
    assert "Netflix" not in names and "Exxon Mobil" not in names  # mentioned
    assert "Coca-Cola" in names
    assert len(names) == len(OUTSIDE_COMPANIES) - 3


def test_unanswerable_tasks_come_from_data() -> None:
    corpus_stats = stats()
    planned = plan_unanswerable(corpus_stats, 9, Picker(seed=3))
    assert Counter(t.unanswerable_reason for t in planned) == {
        "company_outside_corpus": 3,
        "period_not_covered": 3,
        "topic_absent": 3,
    }
    for task in planned:
        if task.unanswerable_reason == "period_not_covered":
            cik = task.companies[0].cik
            year = int(str(task.hint).rsplit("FY", 1)[1])
            assert year <= corpus_stats.first_year[cik] - 6
            assert not {year, year - 1} & corpus_stats.years.get(cik, set())
        elif task.unanswerable_reason == "topic_absent":
            probe = [name for name, _ in TOPIC_PROBES].index(str(task.hint))
            assert probe not in corpus_stats.probe_hits.get(
                task.companies[0].cik, set()
            )
    tasks = assemble(planned, {})
    assert all(t.mention_style == "name" for t in tasks if not t.companies)


def test_assemble_rotates_styles_and_fills_cells() -> None:
    assert company_aliases("ALPHABET INC.") == ("Google",)
    assert company_aliases("AMAZON.COM, INC.") == ()  # AWS is a segment
    assert company_aliases("Apple Inc.") == ()
    pool = corpus()
    old, new = pool[0], pool[2]
    planned = [
        PlannedTask("fact", (pool[1],)),
        PlannedTask("compare_periods", (old, new), pair_kind="table"),
        PlannedTask("fact", (pool[3],)),
        PlannedTask("fact", (pool[4],)),
    ]
    texts = {c.chunk_id: ChunkText("ctx", "text") for t in planned for c in t.chunks}
    tasks = assemble(planned, texts)
    assert [t.task_id for t in tasks] == ["t0001", "t0002", "t0003", "t0004"]
    assert [t.question_type for t in tasks][:2] == ["fact", "compare_periods"]
    assert tasks[1].cell["fiscal_year"] == "2023|2025"
    assert "ask" in tasks[1].cell
    # No company of the corpus fixture has an alias: never `nickname`.
    assert {t.mention_style for t in tasks} <= {"name", "ticker"}
