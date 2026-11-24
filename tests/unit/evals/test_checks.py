from __future__ import annotations

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
    check_testset,
    copy_scores,
    load_drafts,
    names_company,
    render_report,
    widen_evidence,
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
