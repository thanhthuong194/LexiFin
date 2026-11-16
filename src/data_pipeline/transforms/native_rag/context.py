from __future__ import annotations

from dataclasses import dataclass
from datetime import date

SECTION_TITLE_MAX_CHARS = 80
SEPARATOR = "\n\n"


@dataclass(frozen=True)
class FilingContext:
    """What every chunk of one filing says about where it comes from."""

    company_name: str
    form: str
    filing_date: date
    period_end_date: date | None
    fiscal_year: int | None
    fiscal_period: str | None


def period_label(filing: FilingContext) -> str:
    """`FY2025 (period ended 2025-09-27)`, `FY2025 Q2 (...)`, or for 8-Ks
    `filed 2025-01-29 (event 2025-01-28)`."""

    ended = filing.period_end_date
    if filing.form == "8-K":
        event = f" (event {ended.isoformat()})" if ended else ""
        return f"filed {filing.filing_date.isoformat()}{event}"
    label = f"FY{filing.fiscal_year}" if filing.fiscal_year else ""
    if filing.fiscal_period and filing.fiscal_period != "FY":
        label = f"{label} {filing.fiscal_period}".strip()
    if ended:
        label = f"{label} (period ended {ended.isoformat()})".strip()
    return label or f"filed {filing.filing_date.isoformat()}"


def short_title(title: str) -> str:
    """A section title cut to 80 characters (at a word when possible)."""

    title = " ".join(title.split())
    if len(title) <= SECTION_TITLE_MAX_CHARS:
        return title
    cut = title[: SECTION_TITLE_MAX_CHARS - 1].rsplit(" ", 1)[0]
    return f"{cut}…"


def chunk_context(
    filing: FilingContext,
    path: tuple[str, ...],
    *,
    subheading: str | None = None,
    table: str | None = None,
) -> str:
    """The prefix embedded before every chunk's text.

    `Company | form | period | Part › Item[ › subheading][ | Table: title]`;
    `table` is None for text chunks; for table chunks it holds the title and
    `(part i/n)` when split, or "" for an untitled table.
    """

    location = " › ".join([*path, *([subheading] if subheading else [])])
    fields = [filing.company_name, filing.form, period_label(filing)]
    if location:
        fields.append(location)
    if table is not None:
        fields.append(f"Table: {table}" if table else "Table")
    return " | ".join(fields)


def embedded_text(context: str, text: str) -> str:
    """The exact string embedded, counted and hashed for a chunk."""

    return f"{context}{SEPARATOR}{text}"
