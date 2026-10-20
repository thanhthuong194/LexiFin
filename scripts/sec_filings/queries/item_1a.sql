-- The text of Item 1A (Risk Factors) of one filing, block by block in reading order.
-- $filing_id: e.g. sec:0000320193-25-000079 (Apple's 2025 10-K); list filings with
--   select filing_id, cik, form, filing_date from filings order by filing_date
-- A section owns the blocks with start_block_ordinal <= block_ordinal < end_block_ordinal.
select
    s.title as section,
    b.block_ordinal,
    b.block_type,
    b.text
from sections s
join content_blocks b
  on b.document_id = s.document_id
 and b.block_ordinal >= s.start_block_ordinal
 and b.block_ordinal < s.end_block_ordinal
where s.filing_id = $filing_id
  and s.section_code = 'ITEM 1A'
order by s.section_ordinal, b.block_ordinal;
