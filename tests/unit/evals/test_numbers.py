from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from src.evals.numbers import (
    Expected,
    dates,
    derived,
    english_numbers,
    expected_amount,
    numbers,
    states,
    states_computed,
    states_direction,
    states_expected,
    text_states,
    value_tokens,
    without_dates,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("text", "value", "decimals", "found"),
    [
        ("Revenue was $245,122 million.", "245122000000", "-6", True),
        ("Revenue was $245.1 billion.", "245122000000", "-6", True),
        ("Revenue was 245,122 (in millions).", "245122000000", "-6", True),
        ("Revenue was about $245 billion.", "245122000000", "-6", True),
        ("Revenue was about $47 billion.", "46752000000", "-6", False),
        ("Diluted EPS was $11.80.", "11.8", "2", True),
        ("The tax rate was 18%.", "0.18", "2", True),
        ("Accrued expenses | (1,558)", "-1558000000", "-6", True),
        ("Accrued expenses fell to 1,558", "-1558000000", "-6", False),
        ("Accrued expenses | (1,558)", "1558000000", "-6", False),
        ("Repurchases | — | 0", "0", "-6", True),
    ],
)  # fmt: skip
def test_value_tokens(text: str, value: str, decimals: str, found: bool) -> None:
    assert bool(value_tokens(text, value, decimals)) is found


def test_value_sign_words() -> None:
    answer = "A net loss of $1,558 million."
    assert value_tokens(answer, "-1558000000", "-6", sign_words=True) == ["1,558"]


def test_quote_value_ignores_table_sign() -> None:
    row = "Additions to property and equipment | (28,107) | (23,886)"
    assert value_tokens(row, "28107000000", "-6") == []
    assert value_tokens(row, "28107000000", "-6", any_sign=True) == ["28,107"]


def test_numbers_skip_words_and_mark_checked() -> None:
    found = numbers("Q2 FY2024: $1.2 billion, 2023, 45 units, 18%")
    assert [(n.written, n.value, n.checked, n.scale) for n in found] == [
        ("$1.2 billion", Decimal("1.2"), True, Decimal(10) ** 9),
        ("2023", Decimal(2023), False, Decimal(1)),
        ("45", Decimal(45), False, Decimal(1)),
        ("18%", Decimal(18), True, Decimal("0.01")),
    ]


def test_dates() -> None:
    text = "On Sept. 28, 2024 and 2024-02-30."
    assert dates(text) == {"Sept. 28, 2024": date(2024, 9, 28), "2024-02-30": None}
    assert numbers(without_dates(text)) == []


def test_states_and_derived() -> None:
    assert states("$1.5 billion", [Decimal(1500)])
    assert not states("$1.5 billion", [Decimal(1700)])
    assert derived("$2,000", [Decimal(1500), Decimal(500)])
    assert derived("25%", [Decimal(500), Decimal(400)])  # change of two
    assert not derived("7%", [Decimal(500), Decimal(400)])


def test_english_numbers() -> None:
    assert english_numbers("46,8 tỷ USD, 1.287 triệu, giảm 5 phần trăm") == (
        "46.8 billion USD, 1,287 million, decrease 5 percent"
    )


M = Decimal(10) ** 6


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Amazon had $46,752 million on December 31, 2022.", Expected(Decimal(46752) * M, "-6", M / 2)),
        ("About $22.5 billion remained.", Expected(Decimal("22.5") * 1000 * M, "-9", 50 * M)),
        ("Net income was 56.6% of revenue, up from 55.2%.", Expected(Decimal("0.566"), None, Decimal("0.0005"))),
        ("Up to 375,439 shares, adopted on 6/16/2026.", Expected(Decimal(375439), None, Decimal("0.5"))),
        ("Bradford L. Smith (age 64) as of July 27, 2023.", None),
    ],
)  # fmt: skip
def test_expected_amount(text: str, expected: Expected | None) -> None:
    assert expected_amount(text) == expected


@pytest.mark.parametrize(
    ("answer", "value", "decimals", "found"),
    [
        ("Cost of revenues was $33.7 billion [1].", "33712000000", "-6", True),
        ("Cost of revenues was $33,712 million [1].", "33712000000", "-6", True),
        ("Cost of revenues was $34 billion [1].", "33712000000", "-6", False),
        ("Giá vốn hàng bán là 33,7 tỷ USD [1].", "33712000000", "-6", True),
        ("Giá vốn hàng bán là 33.712 triệu USD [1].", "33712000000", "-6", True),
        # Any sign: an outflow fact, positive in XBRL, written in parentheses.
        ("Additions were $(28,107) million [1].", "28107000000", "-6", True),
        ("Chi đầu tư là (28.107) triệu USD [1].", "28107000000", "-6", True),
        ("Net income of $1,558 million [2].", "-1558000000", "-6", True),
        ("A net loss of $1.6 billion [2].", "-1558000000", "-6", False),
        ("Diluted EPS was $5.12.", "5.12", "2", True),
    ],
)  # fmt: skip
def test_text_states(answer: str, value: str, decimals: str, found: bool) -> None:
    assert text_states(answer, value, decimals) is found


@pytest.mark.parametrize(
    ("answer", "reference", "found"),
    [
        ("Cost was $33,712 million [1].", "Cost was $33.7 billion.", True),
        ("Giá vốn là 33.712 triệu USD [1].", "Cost was $33.7 billion.", True),
        ("Giá vốn là 33,71 tỷ USD [1].", "Cost was $33.7 billion.", True),
        ("Net sales were $391,035 million.", "Net sales were $391.0 billion.", True),
        ("Cost was $33.7 billion.", "Cost was $33,712 million.", True),
        ("Khoản lỗ 1.287 triệu USD", "A fair value loss of $1,287 million.", True),
        ("a loss of 1,287 (in millions)", "A fair value loss of $1,287 million.", True),
        ("Cost was $33,812 million.", "Cost was $33.7 billion.", False),
        ("$1.3 million", "A fair value loss of $1,287 million.", False),
        ("It was 56.6% [1].", "Net income was 56.6% of revenue.", True),
        ("It was 57% [1].", "Net income was 56.6% of revenue.", False),
    ],
)  # fmt: skip
def test_states_expected(answer: str, reference: str, found: bool) -> None:
    expected = expected_amount(reference)
    assert expected is not None
    assert states_expected(answer, expected) is found


@pytest.mark.parametrize(
    ("text", "value", "unit", "found"),
    [
        ("Revenue grew 12.3% in FY2024.", "12.3456", "percent", True),
        ("Revenue grew 12.35 percent.", "12.3456", "percent", True),
        ("Doanh thu tăng 12,3%.", "12.3456", "percent", True),
        ("Revenue fell 4.2%.", "-4.2", "percent", True),
        ("Revenue grew 12%.", "12.3456", "percent", False),
        ("Revenue grew 12.4%.", "12.3456", "percent", False),
        ("Revenue grew 12.3 times.", "12.3456", "percent", False),
        ("The gap was $1.23 billion.", "1234000000", "iso4217:USD", True),
        # Two significant digits suffice when they round right.
        ("The gap was $1.2 billion.", "1234000000", "iso4217:USD", True),
        ("The gap was $1 billion.", "1234000000", "iso4217:USD", False),
        ("The gap was $1.3 billion.", "1234000000", "iso4217:USD", False),
    ],
)  # fmt: skip
def test_states_computed(text: str, value: str, unit: str, found: bool) -> None:
    assert states_computed(text, value, unit) is found


@pytest.mark.parametrize(
    ("text", "value", "found"),
    [
        ("Revenue rose 12.3%.", "12.3", True),
        ("Doanh thu tăng 12,3%.", "12.3", True),
        ("Revenue grew +12.3%.", "12.3", True),
        ("Revenue fell 12.3%.", "12.3", False),
        ("Revenue fell 4.2%.", "-4.2", True),
        ("Doanh thu giảm 4,2%.", "-4.2", True),
        ("Revenue changed -4.2% in 2021-2022.", "-4.2", True),
        ("Revenue changed 4.2% in 2021-2022.", "-4.2", False),
        ("Debt changed by $5,042 million.", "-5042000000", False),
        ("Flat at 0.0%.", "0", True),
    ],
)  # fmt: skip
def test_states_direction(text: str, value: str, found: bool) -> None:
    assert states_direction(text, value) is found


@pytest.mark.parametrize(
    ("text", "value", "token"),
    [  # v2 t0081, t0091, t0120: decimals -5 shown with one decimal in millions
        ("TOTAL LIABILITIES | 76,054.1 | 71,355.6", "76054100000", "76,054.1"),
        ("Net earnings attributable to Mosaic | $540.7 | $174.9 | $1,164.9", "1164900000", "1,164.9"),
        ("Total assets | $ 14,234.6 | $ 13,682.6", "13682600000", "13,682.6"),
    ],
)  # fmt: skip
def test_value_tokens_fraction_digits_shift_the_scale(
    text: str, value: str, token: str
) -> None:
    assert value_tokens(text, value, "-5", any_sign=True) == [token]
