"""Numbers and dates as written in answers and passages.

Reads `$33.7 billion`, `(1,558)`, `-4.2%`, `12 thousand` and tells whether
text states a value to its shown rounding, at the scale its unit gives
(`value_tokens`), at any usual scale (`states`), or as a sum, difference,
ratio or change of two others (`derived`). Vietnamese text (`46,8 tỷ`,
`1.287 triệu`) is read through `english_numbers`.
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Collection, Iterable
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import NamedTuple

# A written number matches a value when it rounds to it and has at least
# VALUE_MIN_DIGITS significant digits or is within VALUE_TOLERANCE.
VALUE_MIN_DIGITS = 3
VALUE_TOLERANCE = Decimal("0.005")
# A computed percent (growth, margin, share, CAGR) is stated when written
# within half its first decimal: 12.3 by `12.3%` or `12.25%`, not `12%`.
PERCENT_HALF = Decimal("0.05")
# A computed amount (a difference) rounded this coarsely still states it.
COMPUTED_MIN_DIGITS = 2
_UP = re.compile(
    r"\b(rose|rise[sn]?|increas\w*|grew|grow(s|n|th)?|up|higher|gain(s|ed)?|"
    r"jump\w*|climb\w*|expand\w*)\b|\btăng\b",
    re.IGNORECASE,
)
_DOWN = re.compile(
    r"\b(fell|fall(s|en)?|decreas\w*|declin\w*|drop\w*|down|lower|shr[au]nk|"
    r"reduc\w*|contract\w*)\b|\bgiảm\b|\bsụt\b",
    re.IGNORECASE,
)
_PLUS = re.compile(r"\+\s*\$?\d")
_MINUS = re.compile(r"(?<![\w-])[-−]\s*\$?\d|\(\$?\d[\d,.]*\)?\s*%")
# Numbers worth checking carry `$`, a unit, a fraction or a thousands
# separator, or are at least ANSWER_MIN_NUMBER and not a year.
ANSWER_MIN_NUMBER = 100
_YEARS = range(1900, 2101)
_ANSWER_SCALES = tuple(Decimal(10) ** e for e in (0, 3, 6, 9, 12, -2))
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
_NEGATIVE = re.compile(
    r"\b(loss(es)?|negative|decreas\w*|declin\w*|outflows?|used|deficit)\b|\blỗ\b",
    re.IGNORECASE,
)
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
# Vietnamese: `1.287,5` (dot thousands, comma decimals) and scale words.
_VI_NUMBER = re.compile(r"(?<![\d.,])(\d{1,3}(?:\.\d{3})+(?:,\d+)?|\d+,\d+)(?![\d.,])")
_VI_WORDS = (
    (re.compile(r"\bnghìn tỷ\b|\bngàn tỷ\b|\bnghìn tỉ\b", re.IGNORECASE), "trillion"),
    (re.compile(r"\btỷ\b|\btỉ\b", re.IGNORECASE), "billion"),
    (re.compile(r"\btriệu\b", re.IGNORECASE), "million"),
    (re.compile(r"\bnghìn\b|\bngàn\b", re.IGNORECASE), "thousand"),
    (re.compile(r"\bphần trăm\b", re.IGNORECASE), "percent"),
    (re.compile(r"\bgiảm\b", re.IGNORECASE), "decrease"),
)


class Number(NamedTuple):
    written: str  # as written, stripped
    value: Decimal  # without unit scale
    checked: bool  # an amount worth checking (see ANSWER_MIN_NUMBER)
    scale: Decimal  # of its unit, 1 without one


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
    A number without unit may also be in units of `10 ** -decimals`, shifted
    by its shown fraction digits: `76,054.1` for decimals -5 is in millions.
    """

    try:
        target = Decimal(value)
    except InvalidOperation:
        return [value] if value in text else []
    precision = int(decimals) if decimals and decimals.lstrip("-").isdigit() else None
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
        implicit = [Decimal(1)]
        if precision is not None and precision < 0:
            implicit.append(Decimal(10) ** (digits - precision))
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


def dates(text: str) -> dict[str, date | None]:
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


def without_dates(text: str) -> str:
    """`text` with each full date (see `dates`) replaced by a space."""

    return _DATE.sub(" ", text)


def numbers(text: str) -> list[Number]:
    """Numbers in `text` not inside a word (`Q2`, `FY2024`)."""

    found: list[Number] = []
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
        scale = _SCALES[unit.casefold()] if unit else Decimal(1)
        found.append(Number(match.group(0).strip(), value, checked, scale))
    return found


def states(written: str, values: Iterable[Decimal]) -> bool:
    """`written` (one answer number) states one of `values`, at any usual
    scale (units to trillions, or percent points)."""

    return any(
        value_tokens(written, str(abs(v) * scale), None, any_sign=True)
        for v in values
        for scale in _ANSWER_SCALES
    )


def derived(written: str, operands: Collection[Decimal]) -> bool:
    """`written` is a sum or difference of two operands, or (a percentage) a
    ratio or change of two, to its shown rounding."""

    pairs = list(itertools.permutations(operands, 2))
    if states(written, {a + b for a, b in pairs} | {a - b for a, b in pairs}):
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


def english_numbers(text: str) -> str:
    """Vietnamese number formats and scale words in English form:
    `46,8 tỷ` -> `46.8 billion`, `1.287 triệu` -> `1,287 million`. English
    numbers in `1,287.5` form are misread; check both texts."""

    def swap(match: re.Match[str]) -> str:
        return match.group(1).translate(str.maketrans(".,", ",."))

    converted = _VI_NUMBER.sub(swap, text)
    for pattern, word in _VI_WORDS:
        converted = pattern.sub(word, converted)
    return converted


class Expected(NamedTuple):
    """A reference answer's value: `amount` (unit scale applied), the
    `decimals` letting a bare number in the unit's scale state it, and
    `half` its shown precision (half its last digit, scaled)."""

    amount: Decimal
    decimals: str | None
    half: Decimal


def expected_amount(text: str) -> Expected | None:
    """The first checked number of `text` (the value a reference answer
    states first: the one asked; later ones are mostly comparatives):
    `$33.7 billion` -> 33,700,000,000, decimals `-9`, half 50,000,000."""

    for number in numbers(without_dates(text)):
        if number.checked:
            exponent = number.scale.adjusted()
            decimals = str(-exponent) if exponent >= 3 else None
            shown = number.value.as_tuple().exponent
            digits = -shown if isinstance(shown, int) and shown < 0 else 0
            half = Decimal(5) * Decimal(10) ** (-digits - 1) * number.scale
            return Expected(number.value * number.scale, decimals, half)
    return None


def _readings(text: str) -> tuple[str, ...]:
    """`text` as written and with Vietnamese number formats in English form."""

    return tuple(dict.fromkeys((text, english_numbers(text))))


def text_states(text: str, value: Decimal | str, decimals: str | None) -> bool:
    """`text` (an answer, English or Vietnamese) states `value` to its own
    shown rounding, in any sign (sign errors are left to a judge)."""

    shown = format(value, "f") if isinstance(value, Decimal) else value
    return any(value_tokens(t, shown, decimals, any_sign=True) for t in _readings(text))


def states_expected(text: str, expected: Expected) -> bool:
    """`text` states a reference value: rounded as `text` shows it
    (`text_states`), or more precisely, within the reference's `half`
    (`$33,712 million` for `$33.7 billion`), in any sign."""

    if text_states(text, expected.amount, expected.decimals):
        return True
    bare = [Decimal(1)]
    if expected.decimals is not None:
        bare.append(Decimal(10) ** -int(expected.decimals))
    return any(
        abs(n.value * scale - expected.amount) <= expected.half
        for t in _readings(text)
        for n in numbers(t)
        if n.value
        for scale in ([n.scale] if n.scale != 1 else bare)
    )


def states_computed(text: str, value: str, unit: str | None) -> bool:
    """`text` states a computed answer (`testset.Computed`) in any sign: a
    percent within PERCENT_HALF points, written with `%` or `percent`; any
    other unit as `text_states`, or rounded to at least COMPUTED_MIN_DIGITS
    significant digits (`$5.0 billion` for 5,042 million). The sign is
    `states_direction`'s."""

    target = abs(Decimal(value))
    if unit != "percent":
        if text_states(text, value, None):
            return True
        return any(
            len(n.value.as_tuple().digits) >= COMPUTED_MIN_DIGITS
            and abs(n.value * n.scale - target) <= _half(n)
            for t in _readings(text)
            for n in numbers(t)
            if n.value
        )
    return any(
        n.scale == _SCALES["%"] and abs(n.value - target) <= PERCENT_HALF
        for t in _readings(text)
        for n in numbers(t)
    )


def _half(number: Number) -> Decimal:
    """Half the last shown digit of `number`, at its unit's scale."""

    exponent = number.value.as_tuple().exponent
    digits = -exponent if isinstance(exponent, int) and exponent < 0 else 0
    return Decimal(5) * Decimal(10) ** (-digits - 1) * number.scale


def states_direction(text: str, value: str) -> bool:
    """`text` gives the sign of a change: a direction word (rose, fell; tăng,
    giảm) or a signed number agreeing with `value`; a zero needs none."""

    number = Decimal(value)
    if number == 0:
        return True
    words, signed = (_UP, _PLUS) if number > 0 else (_DOWN, _MINUS)
    return words.search(text) is not None or signed.search(text) is not None
