from __future__ import annotations

from collections import Counter
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from src.evals.sampling import (
    V2_PLAN,
    V2DEV_PLAN,
    Exclusion,
    NegativeIndex,
    Picker,
    PlannedTask,
    TagContext,
    aggregation_groups,
    aim_relative,
    alternate,
    assemble,
    calendar_fiscal,
    event_items,
    finish_v2,
    min_relative,
    outside_name,
    plan_aggregation,
    plan_extra,
    plan_quotas,
    plan_spread,
    read_exclusion,
    relabel,
    reworded,
    task_json,
    task_tags,
    temporal_pairs,
    write_tasks,
)
from src.evals.sampling_data import (
    AGGREGATION_PROBES,
    ChunkMeta,
    ChunkText,
    CompanyHeader,
    CorpusStats,
    FactCandidate,
    FilingInfo,
    Occurrence,
    company_ref,
    find_relation,
    former_names,
    record,
    sic_sector,
)
from src.evals.testset import (
    Computed,
    EvalQuestion,
    Evidence,
    ExpectedFilters,
    SetManifest,
    Split,
    XbrlFact,
    dump_json,
    write_jsonl,
)
from src.evals.universe import OutsideCompany
from tests.unit.evals.test_sampling import chunk, fact

pytestmark = pytest.mark.unit

ALPHA, BETA, GAMMA = "0000000001", "0000000002", "0000000003"
STRATA = {ALPHA: ("Energy", "1"), BETA: ("Energy", "9"), GAMMA: ("Utilities", "5")}


def test_v2_quotas_are_the_spec_minimums_plus_extra() -> None:
    assert sum(V2_PLAN.values()) == 142
    assert plan_quotas(150, "v2") == (V2_PLAN, 8)
    quotas, extra = plan_quotas(300, "v2")
    assert quotas == {q: 2 * n for q, n in V2_PLAN.items()} and extra == 16
    assert plan_quotas(100, "v1")[1] == 0


def test_v2dev_quotas_are_exact_and_scale() -> None:
    assert plan_quotas(120, "v2dev") == (V2DEV_PLAN, 0)
    quotas, extra = plan_quotas(60, "v2dev")
    assert sum(quotas.values()) == 60 and extra == 0
    assert quotas["calc"] == 12 or quotas["calc"] == 13
    assert min_relative(120, "v2dev") == 20 and min_relative(240, "v2dev") == 40
    assert min_relative(150, "v2") == 0 and min_relative(100, "v1") == 0


def _kind(text: str) -> str | None:
    found = find_relation(text)
    return found[0] if found else None


def test_relation_cues_need_a_named_party() -> None:
    assert _kind("Our competitors include Intel and AMD.") == "competitor"
    assert _kind("Our competitors include many firms.") is None
    assert _kind("We rely on sole suppliers such as TSMC.") == "supplier"
    officer = "Revenue grew. On May 1, Jane Doe was appointed Chief Financial Officer."
    assert find_relation(officer) == (
        "officer",
        "On May 1, Jane Doe was appointed Chief Financial Officer.",
    )
    assert _kind("Our Chief Executive was appointed Chief Operating Officer.") is None
    assert _kind("We appointed Chief Officers last year.") is None
    assert _kind("We entered into an agreement with Sales Agents.") is None
    assert _kind("a strategic alliance with the Company Program") is None
    assert _kind("Google LLC, a wholly-owned subsidiary of Alphabet") == "subsidiary"


def test_item_code_and_heading() -> None:
    item = chunk(
        1,
        section_topic="8k_event",
        section_title="ITEM 5.02. DEPARTURE OF DIRECTORS",
        subheading=None,
    )
    assert item.item_code == "5.02" and item.heading == "item 5.02"
    assert chunk(2).item_code is None


def test_temporal_pairs_match_openings_in_consecutive_years() -> None:
    def text(number: int, year: int, head: int, body: int) -> ChunkMeta:
        return chunk(
            number,
            subheading="Competition",
            section_topic="business",
            fiscal_year=year,
            head_hash=head,
            text_hash=body,
        )

    pool = [
        text(1, 2022, 7, 1),
        text(2, 2023, 8, 2),  # another part of the section
        text(3, 2023, 7, 3),  # the 2022 passage reworded
        text(4, 2025, 7, 4),  # 2024 missing: not consecutive
        replace(text(5, 2024, 9, 5), part=1, part_count=2),
        text(6, 2025, 9, 6),  # not the same split part as c0005
    ]
    pairs = temporal_pairs(pool, seed=1)
    assert [(p.first.chunk_id, p.second.chunk_id, p.penalty) for p in pairs] == [
        ("c0001", "c0003", 0)
    ]


def test_reworded_needs_enough_new_words() -> None:
    old = " ".join(f"word{chr(97 + i)}x" for i in range(26)) + " 2023"
    assert not reworded(old, old.replace("2023", "2024"))  # digits only
    new = old + " " + " ".join(f"fresh{chr(97 + i)}y" for i in range(16))
    assert reworded(old, new)
    assert not reworded(old, old + " freshay freshby")


def test_event_items_keep_the_dated_or_first_chunk() -> None:
    def item(number: int, ordinal: int, *, dated: bool = False) -> ChunkMeta:
        return chunk(
            number,
            form="8-K",
            document_type="8-K",
            document_id="d8",
            section_topic="8k_event",
            section_title="Item 5.02. Departure",
            chunk_ordinal=ordinal,
            dated_event=dated,
        )

    assert [c.chunk_id for c in event_items([item(1, 3), item(2, 5)])] == ["c0001"]
    picked = event_items([item(1, 3), item(2, 5, dated=True)])
    assert [c.chunk_id for c in picked] == ["c0002"]


def test_plan_spread_spreads_over_item_codes() -> None:
    codes = ("5.02", "5.02", "5.02", "2.01", "8.01")
    pool = [
        chunk(
            n,
            form="8-K",
            document_type="8-K",
            section_topic="8k_event",
            section_title=f"Item {code}. Event",
            subheading=None,
            filing_id=f"f{n}",
        )
        for n, code in enumerate(codes)
    ]
    planned = plan_spread(pool, "event_8k", 3, Picker(seed=1), lambda c: c.item_code)
    assert sorted(dict(t.cell_extra)["item_code"] for t in planned) == [
        "2.01",
        "5.02",
        "8.01",
    ]


def _stats(*sections: tuple[str, str, int, set[int]]) -> CorpusStats:
    stats = CorpusStats()
    for cik in STRATA:
        stats.companies[cik] = company_ref(cik, f"Company {cik[-1]}", (f"T{cik[-1]}",))
    for cik, topic, year, found in sections:
        stats.section_probes[(cik, topic, year)] = found
    return stats


def test_aggregation_groups_need_a_company_without_the_topic() -> None:
    probe = len(AGGREGATION_PROBES) - 1
    delta = "0000000009"
    pool = [
        chunk(n, cik=cik, section_topic="risk_factors", probes=((probe, "key"),))
        for n, cik in enumerate((ALPHA, BETA, GAMMA, ALPHA))
    ]
    without = replace(chunk(9, section_topic="risk_factors"), cik=delta)
    sectors = {ALPHA: "Energy", BETA: "Energy", GAMMA: "Energy", delta: "Energy"}
    stats = _stats((delta, "risk_factors", 2024, {0}))
    stats.companies[delta] = company_ref(delta, "Delta", ("DLT",))
    groups = aggregation_groups([*pool, without], stats, sectors, seed=1)
    assert len(groups) == 1
    group = groups[0]
    assert {c.cik for c in group.chunks} == {ALPHA, BETA, GAMMA, delta}
    assert group.absent == (delta,) and group.keys == ("key",)
    planned = plan_aggregation(groups, 2, Picker(seed=1), stats)
    assert len(planned) == 1
    task = planned[0]
    assert [c.cik for c in task.companies] == [c.cik for c in task.chunks]
    assert task.hint == AGGREGATION_PROBES[probe][0]
    cell = dict(task.cell_extra)
    assert cell["without_topic"] == delta and cell["probe_keys"] == "key"
    # Without a chunk of its section, no company is shown lacking it: no group.
    assert not aggregation_groups(pool, stats, sectors, seed=1)


def test_picker_strata_spread_companies_over_sectors() -> None:
    pool = [
        chunk(n, cik=cik, subheading=f"Heading {'abcdefghijklmnopqr'[n] * 3}")
        for n, cik in enumerate(list(STRATA) * 6)
    ]
    picker = Picker(seed=3, strata=STRATA)
    picked = picker.chunks(pool, "fact", 4)
    sectors = Counter(STRATA[c.cik][0] for c in picked)
    assert sectors == {"Energy": 2, "Utilities": 2}
    assert picker.overall[("size", "9")] == 1


def test_negatives_same_heading_older_year_first_never_evidence() -> None:
    passage = chunk(1, fiscal_year=2024, subheading="Competition")
    older = chunk(2, fiscal_year=2022, subheading="Competition")
    newer = chunk(3, fiscal_year=2025, subheading="Competition")
    twin = chunk(4, fiscal_year=2024, subheading="Competition")  # same period
    shared = chunk(5, fiscal_year=2021, subheading="Competition", block_ids=("b1",))
    peer = chunk(6, cik=BETA, subheading="Competition", head_hash=6)
    other_sector = chunk(7, cik=GAMMA, subheading="Competition")
    index = NegativeIndex(
        [passage, older, newer, twin, shared, peer, other_sector],
        {cik: sector for cik, (sector, _) in STRATA.items()},
    )
    found = index.negatives([passage], seed=1, numeric=False)
    assert found == ("c0002", "c0003", "c0006")
    assert index.near([passage], found)
    assert not index.near([passage], ("c0006",))  # another company
    # Table/XBRL tasks: later years show the passage year as a comparative.
    assert index.negatives([passage], seed=1, numeric=True) == ("c0002", "c0006")


def test_former_names_drop_state_suffix_variants() -> None:
    block = (
        "FILER:\n\tCOMPANY DATA:\n\t\tCOMPANY CONFORMED NAME:\tAPPLE INC\n"
        "\tFORMER COMPANY:\n\t\tFORMER CONFORMED NAME:\tAPPLE COMPUTER INC\n"
        "\t\tDATE OF NAME CHANGE:\t19970808\n"
        "\tFORMER COMPANY:\n\t\tFORMER CONFORMED NAME:\tAPPLE INC/CA\n"
        "\t\tDATE OF NAME CHANGE:\t19990101\n"
        "\tFORMER COMPANY:\n\t\tFORMER CONFORMED NAME:\tFRUIT CO\n"
        "\t\tDATE OF NAME CHANGE:\t20010101\n"
    )
    assert former_names(block, "Apple Inc.", since="") == (
        "FRUIT CO",
        "APPLE COMPUTER INC",
    )
    assert former_names(block, "Apple Inc.", since="20000101") == ("FRUIT CO",)
    assert former_names(block, "Apple Inc.") == ()  # all before 2015
    assert former_names("", "Apple Inc.") == ()


def test_sic_sector_and_fiscal_calendar() -> None:
    assert sic_sector("1311") == "Energy"
    assert sic_sector("2834") == "Health Care"
    assert sic_sector("3571") == "Information Technology"
    assert sic_sector("6798") == "Real Estate"
    assert sic_sector("6021") == "Financials"
    assert sic_sector(None) is None
    assert not calendar_fiscal(CompanyHeader(fiscal_year_end="0926"), [])
    assert calendar_fiscal(CompanyHeader(fiscal_year_end="0102"), [])
    nvidia = FilingInfo(ALPHA, "10-K", 2025, "FY", date(2025, 1, 26), date(2025, 3, 1))
    assert not calendar_fiscal(None, [nvidia])


def test_record_keeps_section_probes_of_10k_sections() -> None:
    stats = CorpusStats()
    row: dict[str, Any] = {
        "cik": ALPHA,
        "company_name": "Alpha",
        "tickers": ["ALP"],
        "fiscal_year": 2024,
        "filing_id": "f1",
        "form": "10-K",
        "fiscal_period": "FY",
        "period_end_date": date(2024, 12, 31),
        "filing_date": date(2025, 2, 1),
        "chunk_kind": "text",
        "section_topic": "risk_factors",
        "text": "New tariffs could hurt us.",
    }
    record(stats, row)
    record(stats, row | {"section_topic": "mdna", "text": "Sales grew."})
    tariffs = [name for name, _ in AGGREGATION_PROBES].index(
        "tariffs on imported goods"
    )
    assert stats.section_probes[(ALPHA, "risk_factors", 2024)] == {tariffs}
    assert stats.section_probes[(ALPHA, "mdna", 2024)] == set()


def _fact(instant: date | None) -> XbrlFact:
    return XbrlFact(
        fact_id="x1",
        concept="Assets",
        value="10",
        unit="iso4217:USD",
        decimals="-6",
        period_start=None,
        period_end=None,
        instant=instant,
        block_id="b1",
        cell_id=None,
        block_ids=("b1",),
    )


def test_task_tags() -> None:
    context = TagContext(
        off_calendar=frozenset({BETA}),
        size_deciles={ALPHA: 1, BETA: 9},
        newest={(ALPHA, "10-K"): "f-0000000001-1"},
        as_of=date(2026, 1, 1),
    )
    newest = chunk(1, filing_id="f-0000000001-1", part=2, part_count=3)
    assert set(task_tags(PlannedTask("fact", (newest,)), context)) == {"relative_time"}
    part = replace(newest, chunk_kind="table", table_title="Revenue")
    assert "split_table" in task_tags(PlannedTask("table_lookup", (part,)), context)
    small = chunk(2, cik=BETA, chunk_kind="table", table_title="Balance Sheets")
    assert set(task_tags(PlannedTask("table_lookup", (small,)), context)) == {
        "fiscal_calendar",
        "small_company",
        "balance_sheet_date",
    }
    computed = Computed(
        operation="growth",
        value="5",
        unit="percent",
        operands=(_fact(date(2024, 12, 31)),),
    )
    calc = PlannedTask("calc", (chunk(3),), computed=computed, tags=("x",))
    assert task_tags(calc, context) == ("balance_sheet_date",)
    assert "graph" in task_tags(PlannedTask("relation", (chunk(4),)), context)


def test_finish_keeps_existing_negatives() -> None:
    context = TagContext(frozenset(), {}, {}, None)
    passage = chunk(1, fiscal_year=2024, subheading="Competition")
    older = chunk(2, fiscal_year=2023, subheading="Competition")
    index = NegativeIndex([passage, older], {})
    given = PlannedTask("fact", (passage,), hard_negative_chunks=("zz",))
    fresh = PlannedTask("fact", (passage,))
    kept, found = finish_v2([given, fresh], context, index, seed=1)
    assert kept.hard_negative_chunks == ("zz",)
    assert found.hard_negative_chunks == ("c0002",)
    assert "near_duplicate" in found.tags


def test_task_json_omits_v2_defaults() -> None:
    planned = [PlannedTask("fact", (chunk(1),))]
    texts = {"c0001": ChunkText("ctx", "text")}
    (plain,) = assemble(planned, texts)
    line = task_json(plain)
    for name in ("computed", "tags", "hard_negative_chunks", "former_names", "sector"):
        assert f'"{name}"' not in line
    tagged = plain.model_copy(update={"tags": ("graph",)})
    assert '"tags":["graph"]' in task_json(tagged)


def test_assemble_v2_directory_styles_and_cells() -> None:
    ref = company_ref(ALPHA, "Alpha Corp", ("ALP",)).model_copy(
        update={"former_names": ("Old Alpha",), "sector": "Energy"}
    )
    planned = [PlannedTask("fact", (chunk(n),), tags=("t",)) for n in range(1, 9)]
    texts = {c.chunk_id: ChunkText("ctx", "text") for t in planned for c in t.chunks}
    tasks = assemble(planned, texts, {ALPHA: ref}, {ALPHA: 9})
    styles = {t.mention_style for t in tasks}
    assert "former_name" in styles
    for task in tasks:
        assert task.companies == (ref,)
        assert task.cell["sector"] == "Energy" and task.cell["size_decile"] == "9"
        assert ("former_name" in task.tags) == (task.mention_style == "former_name")


def test_plan_extra_fills_the_least_used_types() -> None:
    pool = [
        chunk(n, subheading=f"Heading {letter * 3}")
        for n, letter in enumerate("abcdefgh")
    ]
    planned = [PlannedTask("fact", (pool[0],))]
    picker = Picker(seed=1, used={pool[0].chunk_id})
    added = plan_extra(pool, planned, 3, picker)
    assert [t.question_type for t in added] == ["list", "summary", "fact"]


def test_outside_names_alternate_and_keys() -> None:
    company = OutsideCompany("NVIDIA CORP", ("NVDA",), (), "US-listed")
    assert outside_name(company) == ("NVIDIA CORP", ("NVDA",), ("NVIDIA", "Nvidia"))
    assert alternate(["a", "b", "x", "c"], {"a", "b", "c"}) == ["a", "x", "b", "c"]


def test_exhibit_number_keeps_suffixes() -> None:
    from src.evals.sampling_data import exhibit_number

    assert [exhibit_number(n) for n in ("99.01", "99.1", "99", "99.3(II)")] == [
        "99.1",
        "99.1",
        "99.0",
        "99.3(II)",
    ]


def _question(task_id: str, cik: str, split: str, **fields: Any) -> EvalQuestion:
    return EvalQuestion.model_validate(
        {
            "id": f"v2-{task_id}",
            "task_id": task_id,
            "question_en": "What changed?",
            "question_vi": "Điều gì thay đổi?",
            "reference_answer": "Something.",
            "answerable": True,
            "question_type": "fact",
            "data_kind": "text",
            "persona": "equity analyst",
            "mention_style": "name",
            "companies": [company_ref(cik, "Company", ("T",)).model_dump()],
            "evidence_groups": [[Evidence(block_id="b1", quote="q").model_dump()]],
            "expected": ExpectedFilters(ciks=(cik,)).model_dump(),
            "split": split,
            **fields,
        }
    )


def test_read_exclusion_takes_test_companies_and_every_chunk(tmp_path: Path) -> None:
    planned = [
        PlannedTask("fact", (chunk(1, cik=ALPHA),), hard_negative_chunks=("n1",)),
        PlannedTask("fact", (chunk(2, cik=BETA),)),
        PlannedTask(
            "unanswerable",
            unanswerable_reason="company_outside_corpus",
            hint="ING Groep N.V.",
        ),
        # A test task without a question yet (dropped draft): still excluded.
        PlannedTask("fact", (chunk(5, cik=GAMMA),)),
    ]
    texts = {c.chunk_id: ChunkText("ctx", "text") for t in planned for c in t.chunks}
    splits: list[Split] = ["test", "dev", "dev", "test"]
    tasks = [
        t.model_copy(update={"split": split})
        for t, split in zip(assemble(planned, texts), splits, strict=True)
    ]
    write_tasks(tmp_path / "tasks.jsonl", tasks)
    dump_json(
        tmp_path / "set.json",
        SetManifest(
            set_id="v2",
            as_of=date(2026, 11, 24),
            seed=1,
            total=4,
            chunks_rules_version=8,
        ).model_dump(mode="json"),
    )
    write_jsonl(
        tmp_path / "questions.jsonl",
        [
            _question("t0001", ALPHA, "test", hard_negative_chunks=("n2",)),
            _question("t0002", BETA, "dev"),
        ],
    )
    exclusion = read_exclusion(tmp_path)
    assert exclusion.ciks == {ALPHA, GAMMA}  # BETA only in a dev question
    assert exclusion.chunks == {"c0001", "c0002", "c0005", "n1", "n2"}
    assert exclusion.outside == {"ing groep"}
    assert not exclusion.allows(chunk(3, cik=ALPHA))
    assert not exclusion.allows(chunk(2, cik=BETA))
    assert exclusion.allows(chunk(4, cik=BETA))


def test_exclusion_drops_companies_from_candidates_stats_and_blocks() -> None:
    stats = _stats(
        (ALPHA, "risk_factors", 2024, {0}), (BETA, "risk_factors", 2024, {1})
    )
    stats.first_year.update({ALPHA: 2021, BETA: 2022})
    stats.filings["f1"] = FilingInfo(ALPHA, "10-K", 2024, "FY", None, date(2025, 2, 1))
    stats.filings["f2"] = FilingInfo(BETA, "10-K", 2024, "FY", None, date(2025, 2, 1))
    exclusion = Exclusion(frozenset({ALPHA}), frozenset({"c0002"}))
    pool = [chunk(1, cik=ALPHA), chunk(2, cik=BETA), chunk(3, cik=BETA)]
    kept, left = exclusion.apply(pool, stats)
    assert [c.chunk_id for c in kept] == ["c0003"]
    assert ALPHA not in left.companies and BETA in left.companies
    assert set(left.filings) == {"f2"} and set(left.first_year) == {BETA}
    assert [key[0] for key in left.section_probes] == [BETA]
    assert ALPHA in stats.companies  # the corpus stats stay whole
    blocks = exclusion.blocks({"b": pool})
    assert [c.chunk_id for c in blocks["b"]] == ["c0003"]


def test_relabel_sets_prefix_and_split() -> None:
    planned = [PlannedTask("fact", (chunk(n),)) for n in (1, 2)]
    texts = {c.chunk_id: ChunkText("ctx", "text") for t in planned for c in t.chunks}
    tasks = assemble(planned, texts)
    assert relabel(tasks, "t", None) == tasks
    dev = relabel(tasks, "x", "dev")
    assert [t.task_id for t in dev] == ["x0001", "x0002"]
    assert all(t.split == "dev" for t in dev)
    assert '"split":"dev"' in task_json(dev[0])


def test_picker_takes_newest_filings_until_relative_needed() -> None:
    pool = [
        chunk(
            n,
            filing_id="new" if n < 3 else "old",
            subheading=f"Heading {'abcdefghi'[n] * 3}",
        )
        for n in range(1, 9)
    ]
    picker = Picker(seed=1, newest=frozenset({"new"}), relative_needed=2)
    picked = picker.chunks(pool, "table_lookup", 4)
    assert len(picked) == 4 and {c.filing_id for c in picked[:2]} == {"new"}
    assert picker.relative_needed == 0
    # Without a need, the same pool picks as a plain picker does.
    plain = Picker(seed=1).chunks(pool, "table_lookup", 4)
    assert plain == Picker(seed=1, newest=frozenset({"new"})).chunks(
        pool, "table_lookup", 4
    )


def test_picker_facts_prefer_newest_filing() -> None:
    occurrence = Occurrence("b1", None, "1")
    old = fact(occurrence)
    new = replace(fact(occurrence, concept="Assets"), filing_id="f-new")
    passages = {"f1": chunk(1), "f-new": chunk(2)}

    def passage_of(candidate: FactCandidate) -> tuple[ChunkMeta, Occurrence]:
        return passages[candidate.filing_id], occurrence

    picker = Picker(seed=1, newest=frozenset({"f-new"}), relative_needed=1)
    picked = picker.facts([old, new], 1, passage_of)
    assert [f.concept for f, _ in picked] == ["Assets"]
    assert picker.relative_needed == 0


def test_aim_relative_counts_tasks_already_tagged() -> None:
    context = TagContext(
        frozenset(), {}, {(ALPHA, "10-K"): "f-new"}, as_of=date(2026, 1, 1)
    )
    tagged = PlannedTask("calc", (chunk(1, filing_id="f-new"),))
    picker = Picker(seed=1)
    aim_relative(picker, [tagged], context, 3)
    assert picker.newest == {"f-new"} and picker.relative_needed == 2
    unset = Picker(seed=1)
    aim_relative(unset, [tagged], replace(context, as_of=None), 3)
    assert unset.relative_needed == 0
