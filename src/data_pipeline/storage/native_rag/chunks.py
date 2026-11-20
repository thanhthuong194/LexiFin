from __future__ import annotations

from src.data_pipeline.contracts.native_rag.chunks import CHUNKS_ARROW_SCHEMA
from src.data_pipeline.storage.iceberg import SilverTableSpec
from src.data_pipeline.storage.sec_filings.filings import FILING_YEAR_PARTITION

CHUNKS_TABLE = "native_rag.chunks"

# Version of the chunking rules (method, caps, noise rules, rendering). Bump
# it whenever the same Silver rows would give different chunks, then purge
# and rebuild the table.
# Version 2 (2026-11-17): long paragraphs cut in thirds of the budget with a
# short overlap, so no part exceeds the cap (v1 rejected 5 documents at 513-514
# tokens on the 500-company Silver).
# Version 3 (2026-11-17): tables inside a table-of-contents section are
# classified (data tables kept: Silver puts whole statements there when it finds
# no item heading); a section's opening block is dropped only when at most
# min(title, 200) + 40 chars (HP 8-K items whose title is the whole table).
# Version 4 (2026-11-17): a repeated line is a running header only near page
# breaks or with a page counter (row labels of tables laid out as text stay);
# word-fragmented documents (55 on 500-company Silver) are joined into
# paragraphs.
# Version 5 (2026-11-17): short lines right after a table (statement footers
# such as "See accompanying Notes...") join the table chunk when followed by a
# table, a title-like line or the section end.
# Version 6 (2026-11-17): a repeated line is a running header only next to a
# page break with nothing but noise between (v5 dropped repeated segment
# headings); lone footnote markers are never headers or subheadings and join
# the line after them; a TOC section drops only TOC-shaped lines and 8-K
# contents tables are TOC; lower header rows split table columns; a table's
# subheading is its title line closest to it, never a company name line; a
# title-only unit joins the next unit; EX-99 labels are shortened in the
# prefix; company names come from the earliest filing only. v7: a layout table
# of notes after a table is never split across chunks. v8: a lone marker never
# takes the first row of a layout table.
CHUNKS_SPEC = SilverTableSpec(
    CHUNKS_TABLE,
    CHUNKS_ARROW_SCHEMA,
    FILING_YEAR_PARTITION,
    rules_version=8,
    layer="gold",
)
