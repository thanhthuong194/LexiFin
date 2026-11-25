from __future__ import annotations

from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from src.evals.checks import (
    Block,
    CheckResult,
    CompanyChunk,
    _Found,
    _value_line,
    answer_facts,
    assign_splits,
    check_testset,
    copy_scores,
    load_drafts,
    names_company,
    prune_hard_negatives,
    question_json,
    ranking_holds,
    render_report,
    uses_style,
    widen_evidence,
    wording,
    write_questions,
)
from src.evals.testset import (
    CompanyRef,
    Computed,
    DraftQuestion,
    EvalQuestion,
    EvalTask,
    Evidence,
    ExpectedFilters,
    TaskPassage,
    XbrlFact,
)

pytestmark = pytest.mark.unit

MSFT = CompanyRef(cik="0000789019", name="Microsoft Corporation", tickers=("MSFT",))
GOOG = CompanyRef(
    cik="0001652044",
    name="ALPHABET INC.",
    tickers=("GOOGL", "GOOG"),
    aliases=("Google",),
)
SENTENCE = "Revenue increased due to growth in cloud services across regions."
TEXT = f"{SENTENCE}\nNet income | 72,361 | 88,136"
BLOCKS = {
    "b1": Block.of(SENTENCE, "paragraph"),
    "b2": Block.of("Net income\n72,361\n88,136", "table"),
    "b3": Block.of("Net income was $88,136 million.", "paragraph"),
    "b5": Block.of("Net income\n188,136", "table"),
    "b6": Block.of("Net income\n88\n136", "table"),
    "b9": Block.of("Other passage text about capital expenditures.", "paragraph"),
}
CHUNKS = {
    "b1": [TEXT],
    "b2": [TEXT],
    "b3": ["Net income was $88,136 million."],
    "b5": ["Net income | 188,136"],
    "b6": ["Net income | 88 | 136"],
    "b9": ["Other passage text about capital expenditures."],
}


def passage(blocks: tuple[str, ...] = ("b1", "b2"), **changes: Any) -> TaskPassage:
    values: dict[str, Any] = {
        "chunk_id": f"chunk-{blocks[0]}",
        "document_id": "doc-1",
        "cik": MSFT.cik,
        "company_name": MSFT.name,
        "form": "10-K",
        "document_type": "10-K",
        "section_topic": "mdna",
        "section_title": "ITEM 7",
        "chunk_kind": "text",
        "fiscal_year": 2024,
        "fiscal_period": "FY",
        "period_end_date": date(2024, 6, 30),
        "filing_date": date(2024, 7, 30),
        "context": "",
        "text": TEXT,
        "block_ids": blocks,
    }
    return TaskPassage(**(values | changes))


def task(task_id: str = "t1", **changes: Any) -> EvalTask:
    values: dict[str, Any] = {
        "task_id": task_id,
        "question_type": "fact",
        "data_kind": "text",
        "persona": "analyst",
        "mention_style": "name",
        "companies": (MSFT,),
        "passages": (passage(),),
        "cell": {"form": "10-K"},
    }
    return EvalTask(**(values | changes))


def draft(task_id: str = "t1", **changes: Any) -> DraftQuestion:
    values: dict[str, Any] = {
        "task_id": task_id,
        "question_en": "What pushed Microsoft's sales up in FY2024?",
        "question_vi": "Điều gì làm doanh thu của Microsoft tăng trong năm tài chính 2024?",
        "reference_answer": "Cloud growth.",
        "evidence_quotes": ((SENTENCE,),),
    }
    return DraftQuestion(**(values | changes))


def run(tasks: list[EvalTask], drafts: list[DraftQuestion], **kw: Any) -> CheckResult:
    blocks = kw.get("blocks", BLOCKS)
    return check_testset("v1", tasks, drafts, blocks, kw.get("chunks", CHUNKS))


def checks(issues: list[Any]) -> set[str]:
    return {issue.check for issue in issues}


def test_clean_question() -> None:
    result = run([task("t2"), task("t1")], [draft("t1"), draft("t2")])

    assert not result.errors
    assert [q.id for q in result.questions] == ["v1-t2", "v1-t1"]
    question = result.questions[0]
    assert question.evidence_groups == ((Evidence(block_id="b1", quote=SENTENCE),),)
    assert question.expected.ciks == (MSFT.cik,)
    assert result.coverage["company"]["Microsoft Corporation"] == 2
    assert "near_duplicate" in checks(result.warnings)


def test_table_row_quote_maps_to_cells() -> None:
    quote = "Net income | 72,361 | 88,136"
    result = run([task()], [draft(evidence_quotes=((quote,),))])
    assert not result.errors
    assert result.questions[0].evidence_blocks == {"b2"}

    # Silver keeps `$`, `(9,605` and `)` as cells; the chunk row joins them.
    row = "Revenue | $(9,605) | Earnings"
    split = task(passages=(passage(("b4",), text=row),))
    blocks = {"b4": Block.of("Revenue\n$\n(9,605\n)\nEarnings", "table")}
    chunks = {"b4": [row]}
    quoted = draft(evidence_quotes=((row,),))
    result = run([split], [quoted], blocks=blocks, chunks=chunks)
    assert not result.errors


@pytest.mark.parametrize(
    ("task_changes", "quotes", "kw", "expected"),
    [
        ({}, ((),), {}, "evidence_groups"),
        ({}, ((SENTENCE,), (SENTENCE,)), {}, "evidence_groups"),
        ({}, (("not in the passage at all",),), {}, "quote_not_in_passage"),
        ({}, (("across regions. Net income",),), {}, "quote_no_block"),
        ({}, ((SENTENCE,),), {"chunks": {}}, "quote_not_in_chunk"),
        ({}, ((SENTENCE,),), {"blocks": {"b1": BLOCKS["b1"]}}, "block_not_in_silver"),
        ({"passages": (passage(text=TEXT + " x" * 200),)}, (("x " * 160,),), {}, "quote_too_long"),
        (
            {"question_type": "multi_hop", "passages": (passage(), passage())},
            ((SENTENCE,), (SENTENCE, "Other passage")),
            {},
            "quote_not_in_passage",
        ),
        (
            {"question_type": "unanswerable", "passages": (), "unanswerable_reason": "topic_absent"},
            ((SENTENCE,),),
            {},
            "evidence_unexpected",
        ),
    ],
)  # fmt: skip
def test_evidence_errors(
    task_changes: dict[str, Any],
    quotes: tuple[tuple[str, ...], ...],
    kw: dict[str, Any],
    expected: str,
) -> None:
    result = run([task(**task_changes)], [draft(evidence_quotes=quotes)], **kw)
    assert expected in checks(result.errors)
    assert not result.questions


def test_paired_groups_any_order() -> None:
    other = passage(("b9",), text="Other passage text about capital expenditures.")
    paired = task(question_type="multi_hop", passages=(passage(), other))
    quotes = (("capital expenditures",), (SENTENCE,))
    result = run([paired], [draft(evidence_quotes=quotes)])
    assert not result.errors
    assert [g[0].block_id for g in result.questions[0].evidence_groups] == ["b9", "b1"]


def numeric_task(block_id: str = "b2") -> EvalTask:
    fact = XbrlFact(
        fact_id="f1",
        concept="NetIncomeLoss",
        value="88136000000",
        unit="iso4217:USD",
        decimals="-6",
        period_start=date(2023, 7, 1),
        period_end=date(2024, 6, 30),
        instant=None,
        block_id=block_id,
        cell_id=None,
        block_ids=(block_id, "b3", "b5", "b6"),
    )
    return task(question_type="numeric", data_kind="xbrl", fact=fact)


def test_numeric_evidence_adds_other_fact_blocks() -> None:
    answer = "Net income was $88.1 billion in fiscal 2024."
    quote = draft(
        reference_answer=answer, evidence_quotes=(("Net income | 72,361 | 88,136",),)
    )
    result = run([numeric_task()], [quote])
    assert not result.errors and not result.warnings
    # b5 shows 188,136 and b6 the cells 88 | 136: no alternative.
    group = result.questions[0].evidence_groups[0]
    assert group[1:] == (
        Evidence(block_id="b3", quote="Net income was $88,136 million."),
    )

    result = run([numeric_task()], [draft(evidence_quotes=(("Net income | 72,361",),))])
    assert {"quote_without_value"} == checks(result.errors)
    assert "answer_without_value" in checks(result.warnings)
    result = run([numeric_task()], [draft()])
    assert "evidence_without_fact" in checks(result.errors)


def test_value_line_needs_whole_number() -> None:
    block = Block.of("Rate\n12.5\nOther\n2.5", "table")
    assert _value_line("2.5", block, ["Rate | 12.5", "Other | 2.5"]) == ["Other | 2.5"]
    assert _value_line("2.5", block, ["Rate | 12.5"]) == []


def company_chunk(text: str, block: str, cik: str = MSFT.cik) -> CompanyChunk:
    return CompanyChunk(cik, date(2025, 7, 30), ("h1", block), text)


def test_widen_evidence_with_same_label_and_value() -> None:
    later = {
        "c1": Block.of("Net income\n90,001\n88,136", "table"),
        "c2": Block.of("Net income\n188,136", "table"),
        "c3": Block.of("Net income attributable\n88,136", "table"),
        "c4": Block.of("Net income\n88,136", "table"),
    }
    chunks = [
        company_chunk("FY2025\nNet income | 90,001 | 88,136", "c1"),
        company_chunk("Net income | 188,136", "c2"),
        company_chunk("Net income attributable | 88,136", "c3"),
        company_chunk("Net income | 88,136", "c4", cik=GOOG.cik),
    ]
    quote = draft(evidence_quotes=(("Net income | 72,361 | 88,136",),))
    result = run([numeric_task()], [quote])
    asked: list[set[str]] = []
    loaded: list[set[str]] = []

    def load(ciks: set[str]) -> list[CompanyChunk]:
        loaded.append(ciks)
        return chunks

    def read(ids: set[str], dates: set[date]) -> dict[str, Block]:
        asked.append(ids)
        return later

    widen_evidence(result, [numeric_task()], load, read)
    added = result.questions[0].evidence_groups[0][-1]
    assert added == Evidence(
        block_id="c1", quote="Net income | 90,001 | 88,136", widened=True
    )
    assert result.widened == {"v1-t1": 1} and asked == [{"h1", "c1"}]
    assert loaded == [{MSFT.cik}]
    assert "| v1-t1 | 1 |" in render_report(result)


@pytest.mark.parametrize(
    ("row", "widened"), [("Total | 25", False), ("Total | 4,512", True)]
)
def test_widen_evidence_generic_label_needs_specific_value(
    row: str, widened: bool
) -> None:
    lookup = task(
        question_type="table_lookup",
        data_kind="table",
        passages=(passage(("b4",), text=row),),
    )
    blocks = BLOCKS | {"b4": Block.of(row, "table"), "c1": Block.of(row, "table")}
    quote = draft(
        reference_answer=f"It was {row[-5:].strip(' |')}.", evidence_quotes=((row,),)
    )
    result = check_testset("v1", [lookup], [quote], blocks, {"b4": [row]})
    chunks = [company_chunk(row, "c1")]
    widen_evidence(result, [lookup], lambda c: chunks, lambda ids, dates: blocks)
    assert bool(result.widened) is widened


def test_widen_evidence_uses_the_most_specific_value() -> None:
    row = "Interest rate | 100 basis point increase | (1,287) | Fair Value"
    later = "Interest rate | 100 basis point increase | (1,569) | Fair Value"
    lookup = task(
        question_type="table_lookup",
        data_kind="table",
        passages=(passage(("b4",), text=row),),
    )
    blocks = BLOCKS | {"b4": Block.of(row, "table"), "c1": Block.of(later, "table")}
    answer = "A 100 basis point rise cuts fair value by $1,287 million."
    quote = draft(reference_answer=answer, evidence_quotes=((row,),))
    result = check_testset("v1", [lookup], [quote], blocks, {"b4": [row]})
    chunks = [company_chunk(later, "c1")]
    widen_evidence(result, [lookup], lambda c: chunks, lambda i, d: blocks)
    assert not result.widened


def test_widen_evidence_per_group_with_each_company_and_note_references() -> None:
    equity = "Total stockholders’ equity | 93,404 | 120,564"
    eps = "Diluted net income per share (Note 12) | $2.12 | $7.99"
    compare = task(
        question_type="compare_companies",
        data_kind="table",
        companies=(MSFT, GOOG),
        passages=(
            passage(("b4",), text=equity),
            passage(("g4",), text=eps, cik=GOOG.cik, company_name=GOOG.name),
        ),
    )
    question = EvalQuestion(
        id="v1-t1",
        task_id="t1",
        question_en="Compare Microsoft's equity with Alphabet's diluted EPS.",
        question_vi="So sánh vốn chủ sở hữu của Microsoft với EPS pha loãng của Alphabet.",
        reference_answer="Microsoft: $120,564 million; Alphabet: $7.99 per share.",
        answerable=True,
        question_type="compare_companies",
        data_kind="table",
        persona="analyst",
        mention_style="name",
        companies=(MSFT, GOOG),
        evidence_groups=(
            (Evidence(block_id="b4", quote=equity),),
            (Evidence(block_id="g4", quote=eps),),
        ),
        expected=ExpectedFilters(ciks=(MSFT.cik, GOOG.cik)),
    )
    rows = {
        "c1": equity,  # the 10-Q twin of the release row
        # Without "(Note 12)": the same row in the release.
        "c2": "Diluted net income per share | $2.12 | $5.90 | $7.99",
        "c3": "Diluted net income per share | $2.12 | $7.99",  # another company
        # $7.99 alone (3 significant digits): another period or share class.
        "c4": "Diluted net income per share | $8.01 | $7.99",
    }
    blocks = {b: Block.of(text, "table") for b, text in rows.items()}
    chunks = [
        company_chunk(rows["c1"], "c1"),
        company_chunk(rows["c2"], "c2", cik=GOOG.cik),
        company_chunk(rows["c3"], "c3"),
        company_chunk(rows["c4"], "c4", cik=GOOG.cik),
    ]
    result = CheckResult(questions=[question])
    widen_evidence(result, [compare], lambda c: chunks, lambda i, d: blocks)
    first, second = result.questions[0].evidence_groups
    assert [e.block_id for e in first] == ["b4", "c1"]
    assert [e.block_id for e in second] == ["g4", "c2"]
    assert result.widened == {"v1-t1": 2}


def test_skip_draft() -> None:
    skip = draft(reference_answer="SKIP: passage is a heading only")
    result = run([task()], [skip])
    assert not result.errors and not result.questions and not result.coverage
    assert result.warnings[0].message == "passage is a heading only"
    assert result.skipped == {"fact": 1}
    assert "skipped: 1, fact 1" in render_report(result)


def test_pairing_errors() -> None:
    result = run([task("t1"), task("t2")], [draft("t1"), draft("t1"), draft("t9")])
    assert checks(result.errors) == {
        "draft_missing",
        "draft_duplicate",
        "draft_unknown_task",
    }


@pytest.mark.parametrize(
    ("english", "vietnamese", "error"),
    [
        ("Microsoft revenue in FY2024?", "Doanh thu Microsoft năm tài chính 2024?", False),
        ("Microsoft revenue in Q2 2024?", "Doanh thu Microsoft quý II năm 2024?", False),
        ("Microsoft revenue in Q2 2024?", "Doanh thu Microsoft quý hai năm 2024?", False),
        ("Microsoft revenue in fiscal 2024?", "Doanh thu Microsoft năm 2023?", True),
        ("Microsoft revenue in Q3 2024?", "Doanh thu Microsoft quý 2 năm 2024?", True),
    ],
)  # fmt: skip
def test_period_mismatch(english: str, vietnamese: str, error: bool) -> None:
    result = run([task()], [draft(question_en=english, question_vi=vietnamese)])
    assert ("period_mismatch" in checks(result.errors)) is error


def test_wording_warnings() -> None:
    bare = draft(
        question_en="What drove the company's revenue growth overall?",
        question_vi="What drove the company's revenue growth, overall?",
    )
    result = run([task(mention_style="ticker")], [bare])
    found = checks(result.warnings)
    assert {"no_company", "no_period", "vi_no_diacritics", "mention_style"} <= found

    nickname = draft(
        question_en="How did Google's cloud do in FY2024?",
        question_vi="Mảng đám mây của Google năm tài chính 2024 ra sao?",
    )
    result = run([task(companies=(GOOG,), mention_style="nickname")], [nickname])
    assert not checks(result.warnings) & {"no_company", "mention_style"}

    copied = draft(
        question_en="Why has revenue increased due to growth in cloud services in FY2024?"
    )
    assert "lexical_copy" in checks(run([task()], [copied]).warnings)


def test_hint_warnings() -> None:
    def unanswerable(reason: str, hint: str) -> EvalTask:
        return task(
            question_type="unanswerable",
            data_kind="none",
            companies=(),
            passages=(),
            unanswerable_reason=reason,
            hint=hint,
        )

    question = draft(
        question_en="What was Tesla's revenue in 2016?",
        question_vi="Doanh thu của hãng xe điện năm 2016 là bao nhiêu?",
        evidence_quotes=(),
    )
    result = run([unanswerable("company_outside_corpus", "Tesla, Inc.")], [question])
    assert [w.message for w in result.warnings] == ["vi: Tesla, Inc."]
    assert result.questions[0].expected.unknown_companies == ("Tesla, Inc.",)
    result = run([unanswerable("period_not_covered", "FY2015")], [question])
    assert checks(result.warnings) == {"hint_missing"}


@pytest.mark.parametrize(
    ("text", "company", "named"),
    [
        ("What did Microsoft report?", MSFT, True),
        ("MSFT capex in 2024?", MSFT, True),
        ("MSFTX capex?", MSFT, False),
        ("Doanh thu của Google năm 2024?", GOOG, True),
        ("Alphabet's revenue?", GOOG, True),
        ("A report on growth", CompanyRef(cik="0001090872", name="Agilent", tickers=("A",)), False),
    ],
)  # fmt: skip
def test_names_company(text: str, company: CompanyRef, named: bool) -> None:
    assert names_company(text, company) is named


def test_copy_scores_skip_numbers_names_and_period_words() -> None:
    run_length, share = copy_scores(
        "Microsoft revenue for the fiscal year ended June 2024?", task()
    )
    assert (run_length, share) == (1, 0.0)
    numeric = numeric_task()
    question = "Revenue increased due to growth in cloud services across regions?"
    assert copy_scores(question, numeric)[1] == 0.0
    assert copy_scores(question, task()) == (8, 1.0)


def test_load_drafts_reports_bad_lines(tmp_path: Path) -> None:
    folder = tmp_path / "drafts"
    folder.mkdir()
    (folder / "a.jsonl").write_text(
        draft().model_dump_json() + "\n{}\n", encoding="utf-8"
    )
    drafts, errors = load_drafts([folder])
    assert len(drafts) == 1 and errors[0].message.startswith(f"{folder / 'a.jsonl'}:2:")


def answer_warnings(answer: str, *passages: TaskPassage) -> list[str]:
    found = _Found()
    question = task(passages=passages or (passage(),))
    answer_facts(question, draft(reference_answer=answer), found)
    return [m for c, m in found.warnings if c == "answer_fact_not_in_passage"]


def test_answer_date_not_in_passages() -> None:  # v1 t0025
    apple = passage(
        ("b1",),
        text=" | September 28, 2024 | September 30, 2023\nTotal assets | $364,980",
    )
    msft = passage(
        ("b2",), text="June 30, | 2024 | 2023\nTotal assets | $512,163 | $411,976"
    )
    answer = (
        "Microsoft had more: total assets of $512,163 million as of June 30, 2024, "
        "versus Apple's $364,980 million as of September 28, 2024."
    )
    assert answer_warnings(answer, apple, msft) == ["not in passages: June 30, 2024"]
    labeled = msft.model_copy(update={"context": "FY2024 (period ended 2024-06-30)"})
    assert answer_warnings(answer, apple, labeled) == []


def test_answer_derived_numbers() -> None:  # v1 t0026
    principal = passage(("b1",), text="Principal payments | $2,750 | $52,761 | $55,511")
    debt = passage(("b2",), text="Total debt | 49,781 | 58,146")
    quotes = (
        ("Principal payments | $2,750 | $52,761 | $55,511",),
        ("Total debt | 49,781",),
    )
    found = _Found()
    answer = (
        "Principal due of $55,511 million exceeds total debt of $49,781 million "
        "by $5,730 million, a gap of 11.5%; debt fell 14.4% from $58,146 million."
    )
    answer_facts(
        task(passages=(principal, debt)),
        draft(reference_answer=answer, evidence_quotes=quotes),
        found,
    )
    assert found.warnings == []
    assert answer_warnings(
        "Debt was $5,731 million and $9,999 million.", principal, debt
    ) == ["not in passages: $5,731 million; $9,999 million"]


@pytest.mark.parametrize(
    ("answer", "text", "missing"),
    [
        ("Revenue was $391.0 billion in fiscal 2024.", "Net sales | 391,035", None),
        ("Revenue was $391 billion.", "Net sales | 391,035", None),
        ("Margin was 12.5% in Q2.", "Gross margin 12.5%", None),
        ("Margin was 12.5%.", "Gross margin | 12.5 | 11.0", None),
        ("Capex was $28,107 million.", "Additions | (28,107) | (23,886)", None),
        ("It sees over 78 trillion signals a day.", "over 78 trillion signals", None),
        ("Item 1A lists 3 risks for FY2024 and fiscal 2023.", "Risks.", None),
        ("Revenue was $47 billion.", "Revenue | 46,752", "$47 billion"),
        ("It ended on 2024-06-30.", "As of June 30, 2024", None),
        ("It ended on Sept. 28, 2024.", "As of 2024-09-28", None),
        ("It ended on June 29, 2024.", "As of June 30, 2024", "June 29, 2024"),
    ],
)  # fmt: skip
def test_answer_number_forms(answer: str, text: str, missing: str | None) -> None:
    expected = [] if missing is None else [f"not in passages: {missing}"]
    assert answer_warnings(answer, passage(text=text)) == expected


def test_answer_facts_skip_unanswerable_and_skip_drafts() -> None:
    unanswerable = task(
        question_type="unanswerable",
        data_kind="none",
        companies=(),
        passages=(),
        unanswerable_reason="company_outside_corpus",
        hint="Tesla",
    )
    found = _Found()
    refusal = draft(reference_answer="No filing shows Tesla's $9,999 million.")
    answer_facts(unanswerable, refusal, found)
    assert found.warnings == []
    skip = draft(reference_answer="SKIP: no figure of $9,999 million")
    assert "answer_fact_not_in_passage" not in checks(run([task()], [skip]).warnings)


# v2 types

AAPL = CompanyRef(cik="0000320193", name="Apple Inc.", tickers=("AAPL",))
META = CompanyRef(
    cik="0001326801",
    name="Meta Platforms, Inc.",
    tickers=("META",),
    former_names=("Facebook Inc",),
)
OTHER = "Other passage text about capital expenditures."


def operand(value: str, block_ids: tuple[str, ...] = ("b2", "b3")) -> XbrlFact:
    return XbrlFact(
        fact_id=f"f{value}",
        concept="NetIncomeLoss",
        value=value,
        unit="iso4217:USD",
        decimals="-6",
        period_start=None,
        period_end=None,
        instant=None,
        block_id=block_ids[0],
        cell_id=None,
        block_ids=block_ids,
    )


def calc_task() -> EvalTask:
    computed = Computed(
        operation="growth",
        value="21.8006",
        unit="percent",
        operands=(operand("88136000000"), operand("72361000000", ("b2",))),
    )
    return task(question_type="calc", data_kind="xbrl", computed=computed)


ROW = "Net income | 72,361 | 88,136"


def test_calc_needs_value_and_every_operand_quoted() -> None:
    answer = (
        "Net income grew 21.8% in fiscal 2024, from $72,361 million to $88,136 million."
    )
    good = draft(reference_answer=answer, evidence_quotes=((ROW,),))
    result = run([calc_task()], [good])
    assert not result.errors
    question = result.questions[0]
    assert question.computed == calc_task().computed
    # 88,136 also shows in b3: alternative evidence for that operand.
    assert [e.block_id for e in question.evidence_groups[0]] == ["b2", "b3"]
    rounded = draft(
        reference_answer="Net income grew 22% in fiscal 2024.",
        evidence_quotes=((ROW,),),
    )
    assert "answer_without_computed" in checks(run([calc_task()], [rounded]).errors)
    unquoted = draft(reference_answer=answer)
    assert "operand_not_quoted" in checks(run([calc_task()], [unquoted]).errors)
    # A growth needs its direction: a word or a sign.
    flat = draft(
        reference_answer="Net income moved 21.8% in fiscal 2024.",
        evidence_quotes=((ROW,),),
    )
    assert "answer_without_computed" in checks(run([calc_task()], [flat]).errors)
    fell = draft(
        reference_answer="Net income fell 21.8% in fiscal 2024.",
        evidence_quotes=((ROW,),),
    )
    assert "answer_without_computed" in checks(run([calc_task()], [fell]).errors)


def ranking_task() -> EvalTask:
    rows = {"b2": ROW, "g4": "Net income | 100,118", "a4": "Net income | 93,736"}
    passages = tuple(
        passage((b,), text=text, cik=c.cik, company_name=c.name, chunk_id=f"k{b}")
        for (b, text), c in zip(rows.items(), (MSFT, GOOG, AAPL), strict=True)
    )
    computed = Computed(
        operation="ranking",
        value=None,
        unit="iso4217:USD",
        operands=(
            operand("88136000000", ("b2",)),
            operand("100118000000", ("g4",)),
            operand("93736000000", ("a4",)),
        ),
        order=(GOOG.cik, AAPL.cik, MSFT.cik),
    )
    return task(
        question_type="ranking",
        data_kind="xbrl",
        companies=(MSFT, GOOG, AAPL),
        passages=passages,
        computed=computed,
    )


@pytest.mark.parametrize(
    ("answer", "error"),
    [
        ("Alphabet led with $100,118 million, then Apple ($93,736 million), then Microsoft ($88,136 million).", False),
        # Reversed only when the lowest is asked for.
        ("From the lowest: MSFT $88,136 million, AAPL $93,736 million, Google $100,118 million.", False),
        ("MSFT $88,136 million, AAPL $93,736 million, Google $100,118 million.", True),
        ("Apple ($93,736 million) led Alphabet ($100,118 million) and Microsoft ($88,136 million).", True),
        ("Alphabet led with $100,118 million, then Apple with $93,736 million.", True),
        # Each company placed by the mention nearest its value.
        ("Unlike Microsoft, Alphabet led with $100,118 million; Apple had $93,736 million and Microsoft $88,136 million.", False),
    ],
)  # fmt: skip
def test_ranking_order(answer: str, error: bool) -> None:
    blocks = BLOCKS | {
        "g4": Block.of("Net income\n100,118", "table"),
        "a4": Block.of("Net income\n93,736", "table"),
    }
    chunks = CHUNKS | {"g4": ["Net income | 100,118"], "a4": ["Net income | 93,736"]}
    quotes = ((ROW,), ("Net income | 100,118",), ("Net income | 93,736",))
    result = run(
        [ranking_task()],
        [draft(reference_answer=answer, evidence_quotes=quotes)],
        blocks=blocks,
        chunks=chunks,
    )
    assert ("ranking_order" in checks(result.errors)) is error
    assert checks(result.errors) <= {"ranking_order"}


def test_false_premise_reference_denies_it() -> None:
    premise = task(question_type="false_premise", hint="sales fell in FY2024")
    denied = draft(reference_answer="Sales did not fall; cloud growth pushed them up.")
    assert not run([premise], [denied]).warnings
    affirmed = draft(reference_answer="Cloud growth pushed sales up.")
    assert "premise_not_denied" in checks(run([premise], [affirmed]).warnings)
    two = draft(evidence_quotes=((SENTENCE,), (SENTENCE,)))
    assert "evidence_groups" in checks(run([premise], [two]).errors)


def test_computed_false_premise_like_calc() -> None:
    computed = calc_task().computed
    premise = task(
        question_type="false_premise", hint="net income fell", computed=computed
    )
    answer = "No, net income rose 21.8% in fiscal 2024, to $88,136 million from $72,361 million."
    assert not run(
        [premise], [draft(reference_answer=answer, evidence_quotes=((ROW,),))]
    ).errors
    wrong_way = "No, net income fell 21.8% in fiscal 2024."
    result = run(
        [premise], [draft(reference_answer=wrong_way, evidence_quotes=((ROW,),))]
    )
    assert "answer_without_computed" in checks(result.errors)
    result = run([premise], [draft(reference_answer=answer)])
    assert "operand_not_quoted" in checks(result.errors)


@pytest.mark.parametrize(
    ("years", "error"), [((2024, 2023), False), ((2024, 2024), True)]
)
def test_temporal_change_needs_two_years(years: tuple[int, int], error: bool) -> None:
    passages = (
        passage(fiscal_year=years[0]),
        passage(("b9",), text=OTHER, fiscal_year=years[1]),
    )
    change = task(question_type="temporal_change", passages=passages)
    quotes = ((SENTENCE,), ("capital expenditures",))
    result = run([change], [draft(evidence_quotes=quotes)])
    assert ("temporal_same_year" in checks(result.errors)) is error
    assert "temporal_same_quote" not in checks(result.warnings)


def test_temporal_change_same_quote_warns() -> None:
    same = passage(("b1",), fiscal_year=2023, chunk_id="k-old")
    change = task(question_type="temporal_change", passages=(passage(), same))
    result = run([change], [draft(evidence_quotes=((SENTENCE,), (SENTENCE,)))])
    assert "temporal_same_quote" in checks(result.warnings)


def test_aggregation_one_group_per_company_that_says_it() -> None:
    google = passage(("b9",), text=OTHER, cik=GOOG.cik, company_name=GOOG.name)

    def aggregate(absent: str = "", *passages: TaskPassage) -> EvalTask:
        return task(
            question_type="aggregation",
            companies=(MSFT, GOOG),
            passages=passages or (passage(), google),
            hint="cloud growth",
            cell={"without_topic": absent},
        )

    both = ((SENTENCE,), ("capital expenditures",))
    answer = "Microsoft says so, and Alphabet says so too."
    assert not run(
        [aggregate()], [draft(reference_answer=answer, evidence_quotes=both)]
    ).errors
    # Alphabet lacks the topic: one group, and the reference must deny it.
    denied = "Microsoft says so; Alphabet does not."
    result = run([aggregate(GOOG.cik)], [draft(reference_answer=denied)])
    assert not result.errors
    assert not checks(result.warnings) & {
        "answer_company_missing",
        "without_topic_affirmed",
    }
    affirmed = run([aggregate(GOOG.cik)], [draft(reference_answer=answer)])
    assert "without_topic_affirmed" in checks(affirmed.warnings)
    result = run(
        [aggregate(GOOG.cik)], [draft(reference_answer=denied, evidence_quotes=both)]
    )
    assert "evidence_groups" in checks(result.errors)
    absent_quoted = run([aggregate(MSFT.cik)], [draft(reference_answer=denied)])
    assert "aggregation_without_topic_quoted" in checks(absent_quoted.errors)
    missing = run([aggregate(GOOG.cik)], [draft(reference_answer="Microsoft says so.")])
    assert "answer_company_missing" in checks(missing.warnings)
    same = aggregate("", passage(), passage(("b9",), text=OTHER))
    result = run([same], [draft(reference_answer=answer, evidence_quotes=both)])
    assert "aggregation_company_repeated" in checks(result.errors)


@pytest.mark.parametrize(
    ("form", "topic", "error"),
    [("8-K", "8k_event", False), ("10-K", "8k_event", True), ("8-K", "ex99", True)],
)
def test_event_8k_passage_is_an_item(form: str, topic: str, error: bool) -> None:
    event = task(
        question_type="event_8k", passages=(passage(form=form, section_topic=topic),)
    )
    result = run([event], [draft()])
    assert ("event_not_8k_item" in checks(result.errors)) is error


def test_former_name_style() -> None:
    text = "What did Facebook spend on capex in 2021?"
    assert names_company(text, META) and uses_style(text, META, "former_name")
    assert not uses_style("What did Meta spend in 2021?", META, "former_name")


def test_tags_and_hard_negatives_reach_questions_and_coverage() -> None:
    tagged = task(tags=("fiscal_calendar",), hard_negative_chunks=("n1",))
    result = run([tagged], [draft()])
    question = result.questions[0]
    assert question.tags == ("fiscal_calendar",)
    assert question.hard_negative_chunks == ("n1",)
    assert result.coverage["tag"] == {"fiscal_calendar": 1}
    assert "tag" not in run([task()], [draft()]).coverage


def test_assign_splits_per_question_and_stable() -> None:
    tagged = [task(f"f{i}", tags=("graph",) if i % 2 else ()) for i in range(40)]
    questions = run(tagged, [draft(f"f{i}") for i in range(40)]).questions
    split = assign_splits(questions, seed=1)
    assert [q.id for q in split] == [q.id for q in questions]
    dev = sum(q.split == "dev" for q in split)
    assert 8 <= dev <= 24  # about 40%
    assert assign_splits(questions, seed=1) == split
    # Each question decided alone: dropping others moves none.
    assert assign_splits(questions[5:], seed=1) == split[5:]
    other = assign_splits(questions, seed=2)
    assert [q.split for q in other] != [q.split for q in split]
    report = render_report(CheckResult(questions=split))
    tests = sum(q.split == "test" for q in split)
    assert f"| fact | {dev} | {tests} |" in report
    graph = Counter(q.split for q in split if q.tags)
    assert f"| graph | {graph['dev']} | {graph['test']} |" in report
    assert "## Splits" not in render_report(CheckResult(questions=questions))


def test_prune_hard_negatives() -> None:
    tagged = task(hard_negative_chunks=("n1", "n2", "n3"))
    result = run([tagged], [draft()])
    texts = {
        "n1": f"Intro. {SENTENCE}",
        "n2": "Revenue fell in other regions.",
        "n3": "",
    }
    prune_hard_negatives(result, texts)
    assert result.questions[0].hard_negative_chunks == ("n2", "n3")
    assert result.pruned == {"v1-t1": 1}
    assert "| v1-t1 | 1 |" in render_report(result)
    numeric = numeric_task().model_copy(update={"hard_negative_chunks": ("n4", "n5")})
    quote = draft(
        evidence_quotes=(("Net income | 72,361 | 88,136",),),
        reference_answer="Net income was $88,136 million in fiscal 2024.",
    )
    result = run([numeric], [quote])
    prune_hard_negatives(
        result, {"n4": "Net income | 88,136 | 99,000", "n5": "Net income | 90,000"}
    )
    assert result.questions[0].hard_negative_chunks == ("n5",)


def test_relative_time_tag_and_wording() -> None:
    relative = draft(
        question_en="How much did Microsoft's revenue grow last quarter?",
        question_vi="Doanh thu của Microsoft quý vừa qua tăng bao nhiêu?",
    )
    assert "relative_time" in checks(run([task()], [relative]).warnings)
    tagged = task(tags=("relative_time",))
    assert "relative_time" not in checks(run([tagged], [relative]).warnings)
    assert "relative_time" in checks(run([tagged], [draft()]).warnings)


def test_question_json_leaves_out_v2_defaults(tmp_path: Path) -> None:
    v1 = run([task()], [draft()]).questions[0]
    line = question_json(v1)
    for key in (
        "computed",
        "tags",
        "hard_negative_chunks",
        "split",
        "former_names",
        "sector",
    ):
        assert f'"{key}"' not in line
    assert EvalQuestion.model_validate_json(line) == v1
    v2 = run([task(companies=(META,), mention_style="name")], [draft()]).questions[0]
    v2 = v2.model_copy(update={"split": "test"})
    line = question_json(v2)
    assert '"split":"test"' in line and '"former_names":["Facebook Inc"]' in line
    assert '"sector"' not in line
    assert EvalQuestion.model_validate_json(line) == v2
    path = tmp_path / "questions.jsonl"
    assert write_questions(path, [v1, v2]) == 2
    assert path.read_text(encoding="utf-8").splitlines() == [question_json(v1), line]


# v2 check fixes (strings from the homepc v2 drafts)


def company(cik: str, name: str, *tickers: str) -> CompanyRef:
    return CompanyRef(cik=cik, name=name, tickers=tickers)


def eps(value: str, decimals: str = "2") -> XbrlFact:
    return operand(value).model_copy(update={"decimals": decimals})


def test_ranking_with_tickers_and_short_names() -> None:  # t0121, t0027
    ivz = company("0000914208", "Invesco Ltd.", "IVZ")
    syf = company("0001601712", "SYNCHRONY FINANCIAL", "SYF")
    bx = company("0001393818", "Blackstone Inc", "BX")
    computed = Computed(
        operation="ranking",
        value=None,
        unit="iso4217:USD/xbrli:shares",
        operands=(eps("6.19"), eps("2.36"), eps("1.5")),
        order=(syf.cik, bx.cik, ivz.cik),
    )
    answer = (
        "SYF was highest with basic EPS of $6.19 in fiscal 2022, followed by BX "
        "at $2.36 and IVZ at $1.50."
    )
    assert ranking_holds(computed, (ivz, syf, bx), answer)
    hrl = company("0000048465", "HORMEL FOODS CORPORATION", "HRL")
    pm = company("0001413329", "Philip Morris International Inc.", "PM")
    adm = company("0000007084", "ARCHER-DANIELS-MIDLAND CO", "ADM")
    goodwill = Computed(
        operation="ranking",
        value=None,
        unit="iso4217:USD",
        operands=(
            eps("6680000000", "-6"),
            eps("4929102000", "-3"),
            eps("4182000000", "-6"),
        ),
        order=(pm.cik, hrl.cik, adm.cik),
    )
    answer = (
        "Philip Morris International ranked highest with goodwill of $6,680 "
        "million at December 31, 2021, followed by Hormel at $4,929,102 thousand "
        "(October 31, 2021) and Archer-Daniels-Midland at $4,182 million "
        "(December 31, 2021)."
    )
    assert ranking_holds(goodwill, (hrl, pm, adm), answer)


@pytest.mark.parametrize(
    ("text", "name", "tickers"),
    [
        ("How much cash did Valero generate", "VALERO ENERGY CORP/TX", ()),
        ("Disney does not; its passage covers FCC regulation", "WALT DISNEY CO/", ("DIS",)),
        ("Regions (RF) yes, listing FDIC insurance assessments", "REGIONS FINANCIAL CORP", ("RF",)),
    ],
)  # fmt: skip
def test_names_company_sec_names(
    text: str, name: str, tickers: tuple[str, ...]
) -> None:
    assert names_company(text, company("0000000001", name, *tickers))


def test_generic_name_words_do_not_name() -> None:
    aep = company("0000004904", "AMERICAN ELECTRIC POWER COMPANY, INC.")
    assert not names_company("Which American utility grew fastest?", aep)
    assert names_company("American Electric Power's 2024 liabilities", aep)


def test_ticker_style_without_tickers() -> None:  # t0091 Mosaic
    mosaic = company("0001285785", "MOSAIC CO")
    assert uses_style(
        "Of ALB, Mosaic, CTVA and IP, which earned most?", mosaic, "ticker"
    )


def wording_issues(
    english: str, vietnamese: str, **changes: Any
) -> tuple[set[str], set[str]]:
    found = _Found()
    wording(task(**changes), draft(question_en=english, question_vi=vietnamese), found)
    return {c for c, _ in found.errors}, {c for c, _ in found.warnings}


RELATIVE = ("relative_time",)


@pytest.mark.parametrize(
    ("english", "vietnamese"),
    [
        ("What gross margin did AutoZone earn in the fiscal year before the latest one?",
         "Biên lợi nhuận gộp của AutoZone trong năm tài chính trước năm tài chính gần nhất là bao nhiêu?"),
        ("Two quarters ago, what share of its revenue did RDDT spend on R&D?",
         "Hai quý trước, RDDT chi bao nhiêu phần trăm doanh thu cho R&D?"),
        ("Why did CRM's stock buybacks jump last quarter compared with the same quarter a year earlier?",
         "Vì sao khoản mua lại cổ phiếu của CRM trong quý vừa rồi tăng mạnh so với cùng kỳ năm trước?"),
        ("What net income did DHI report for its last fiscal year?",
         "DHI đã báo cáo lợi nhuận ròng là bao nhiêu trong năm tài chính vừa rồi?"),
        ("Two quarters ago, what were the main results CRH highlighted for its second quarter?",
         "Cách đây hai quý, CRH nêu những kết quả chính nào của quý 2?"),
    ],
)  # fmt: skip
def test_relative_wording_read_in_both_languages(english: str, vietnamese: str) -> None:
    errors, warnings = wording_issues(english, vietnamese, tags=RELATIVE)
    assert not errors  # t0010: "năm tài chính" alone is a period kind, no year
    assert not warnings & {"no_period", "relative_time"}


def test_relative_tag_with_explicit_date_still_warns() -> None:  # t0056
    explicit = draft(
        question_en="In the 8-K Microsoft filed in August 2026 for its second quarter results, which exhibit holds the release?",
        question_vi="Trong 8-K Microsoft nộp hồi tháng 8 năm 2026 về kết quả quý 2, thông cáo nằm ở phụ lục nào?",
    )  # fmt: skip
    assert "relative_time" in checks(run([task(tags=RELATIVE)], [explicit]).warnings)
    relative = draft(
        question_en="What net income did Microsoft report for its last fiscal year?",
        question_vi="Microsoft đã báo cáo lợi nhuận ròng là bao nhiêu trong năm tài chính vừa rồi?",
    )  # fmt: skip
    assert "relative_time" not in checks(
        run([task(tags=RELATIVE)], [relative]).warnings
    )


def test_period_mismatch_cases() -> None:
    errors, _ = wording_issues(  # t0055: vi writes the year as 6/2026
        "How many broker non-votes did Lowe's and Zimmer Biomet each report in the director elections disclosed in their 8-Ks from May and June 2026?",
        "Lowe's và Zimmer Biomet mỗi công ty ghi nhận bao nhiêu phiếu broker non-votes trong cuộc bầu giám đốc được công bố trong các báo cáo 8-K tháng 5 và tháng 6/2026?",
    )  # fmt: skip
    assert "period_mismatch" not in errors
    errors, _ = wording_issues(
        "Microsoft revenue in fiscal 2024?", "Doanh thu Microsoft năm 2023?"
    )
    assert "period_mismatch" in errors


def test_topic_absent_needs_no_period() -> None:  # t0132
    _, warnings = wording_issues(
        "How much coal mine reclamation liability does Procter & Gamble report?",
        "Procter & Gamble ghi nhận bao nhiêu khoản nợ phải trả cho việc hoàn nguyên mỏ than?",
        question_type="unanswerable",
        data_kind="none",
        companies=(),
        passages=(),
        unanswerable_reason="topic_absent",
        hint="coal mine reclamation liabilities",
    )
    assert "no_period" not in warnings


def test_inline_table_rows_and_spanning_blocks() -> None:
    # t0021: a short table the chunk writes inline, rows joined by "; ".
    inline = "Commodity Derivative | 10% Change | 25% Change; Natural gas | $6 | $15; Other commodities | $14 | $36"
    cells = "Commodity Derivative\n10% Change\n25% Change\nNatural gas\n$6\n$15\nOther commodities\n$14\n$36"
    quote = "Natural gas | $6 | $15; Other commodities | $14 | $36"
    blocks = {"t1": Block.of(cells, "table")}
    result = run(
        [task(passages=(passage(("t1",), text=inline),))],
        [draft(evidence_quotes=((quote,),))],
        blocks=blocks,
        chunks={"t1": [inline]},
    )
    assert not result.errors
    # t0046: a sentence whose inline runs are separate blocks.
    sentence = "We compete with private, public and state-owned companies in all facets of the E&P business."
    runs = {
        "r1": "We compete with private, public",
        "r2": "and state-owned companies in all",
        "r3": "facets of the E&P business.",
    }
    blocks = {b: Block.of(t, "paragraph") for b, t in runs.items()}
    result = run(
        [task(passages=(passage(tuple(runs), text=sentence),))],
        [draft(evidence_quotes=((sentence,),))],
        blocks=blocks,
        chunks={b: [sentence] for b in runs},
    )
    assert not result.errors
    assert "quote_spans_blocks" in checks(result.warnings)
    assert result.questions[0].evidence_blocks == {"r1"}


def test_assign_splits_keeps_a_fixed_task_split() -> None:
    fixed = [task(f"x{i}", split="dev") for i in range(20)]
    questions = run(fixed, [draft(f"x{i}") for i in range(20)]).questions
    assert {q.split for q in questions} == {"dev"}
    assert {q.split for q in assign_splits(questions, seed=1)} == {"dev"}


def test_a_final_country_word_is_optional() -> None:  # x0086
    tmus = company("0001283699", "T-MOBILE US, INC.")
    assert names_company("followed by T-Mobile at $5,409 million", tmus)
