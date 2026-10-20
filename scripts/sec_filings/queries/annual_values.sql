-- Full-year values of one XBRL concept from each 10-K, per company and fiscal year.
-- $concept: the concept's local name, e.g. RevenueFromContractWithCustomerExcludingAssessedTax
-- (revenue) or NetIncomeLoss; concepts measured at an instant (Assets) have no start date.
-- Only the company-wide value counts: contexts with dimensions (by product, segment...)
-- are left out. xbrl_facts keeps every place a value appears, hence DISTINCT.
select distinct
    f.cik,
    e.company_name,
    x.start_date,
    x.end_date,
    u.normalized_expression as unit,
    cast(fa.numeric_value as double) as value
from xbrl_facts fa
join xbrl_concepts c on c.concept_id = fa.concept_id
join xbrl_contexts x on x.context_key = fa.context_key
left join xbrl_units u on u.unit_key = fa.unit_key
join xbrl_instances i on i.instance_id = fa.instance_id
join filings f on f.filing_id = i.filing_id
join filing_entities e on e.filing_id = f.filing_id and e.entity_ordinal = 1
where f.form = '10-K'
  and c.local_name = $concept
  and x.segment_xml is null
  and x.scenario_xml is null
  and x.end_date = f.report_date                          -- the year the 10-K reports on
  and date_diff('day', x.start_date, x.end_date) > 350    -- a full year, not a quarter
order by f.cik, x.end_date;
