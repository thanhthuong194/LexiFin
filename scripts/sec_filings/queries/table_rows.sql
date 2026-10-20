-- One HTML table rebuilt row by row: the non-empty cells of each row, left to right.
-- $table_id: e.g. sec:0000320193-25-000079:document:0001:block:000399:table (Apple's
--   2025 income statement); find tables with
--   select table_id, row_count, column_count from document_tables where filing_id = '...'
-- SEC tables pad layout with empty cells (about 69% of all cells), which are left out.
select
    row_ordinal,
    string_agg(text, ' | ' order by column_ordinal) as cells
from document_table_cells
where table_id = $table_id
  and text <> ''
group by row_ordinal
order by row_ordinal;
