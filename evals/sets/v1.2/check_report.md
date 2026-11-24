# Eval set check

- questions: 97
- errors: 0
- warnings: 5
- skipped: 3, compare_companies 2, multi_hop 1

## Errors

None.

## Warnings

`lexical_copy` 1, `quote_short` 1, `skipped` 3

| task | check | detail |
|---|---|---|
| t0021 | lexical_copy | shared run 5 content words, share 0.00 |
| t0034 | skipped | the two passages share no line item (Alphabet's part shows financing cash flow and cash balances; Amazon's part shows only supplemental lease, interest and tax payments), so no like-for-like company comparison is possible. |
| t0035 | skipped | both passages (MD&A and the financial-statement note) state the same facts about the Arm termination (reason and $1.35 billion charge), so neither half needs the other; not multi-hop. |
| t0043 | skipped | the Alphabet passage only introduces its repurchase table and gives no repurchase figures, so the two companies' buybacks cannot be compared. |
| t0080 | quote_short | 'July 27, 2023' |

## Evidence from other filings

| question | rows added |
|---|---|
| v1.2-t0001 | 10 |
| v1.2-t0006 | 6 |
| v1.2-t0007 | 10 |
| v1.2-t0010 | 3 |
| v1.2-t0017 | 2 |
| v1.2-t0019 | 5 |
| v1.2-t0021 | 3 |
| v1.2-t0024 | 7 |
| v1.2-t0025 | 19 |
| v1.2-t0026 | 4 |
| v1.2-t0028 | 5 |
| v1.2-t0037 | 8 |
| v1.2-t0042 | 2 |
| v1.2-t0045 | 19 |
| v1.2-t0053 | 5 |
| v1.2-t0060 | 3 |
| v1.2-t0062 | 3 |
| v1.2-t0065 | 29 |
| v1.2-t0067 | 4 |
| v1.2-t0071 | 1 |
| v1.2-t0075 | 10 |
| v1.2-t0077 | 1 |
| v1.2-t0083 | 13 |
| v1.2-t0086 | 3 |
| v1.2-t0089 | 9 |
| v1.2-t0091 | 1 |
| v1.2-t0093 | 9 |
| v1.2-t0095 | 1 |
| v1.2-t0097 | 7 |
| v1.2-t0099 | 5 |

## Coverage (accepted questions)

### question_type

| value | questions |
|---|---|
| compare_companies | 6 |
| compare_periods | 8 |
| fact | 20 |
| list | 8 |
| multi_hop | 3 |
| numeric | 20 |
| summary | 6 |
| table_lookup | 12 |
| unanswerable | 14 |

### data_kind

| value | questions |
|---|---|
| none | 14 |
| table | 22 |
| text | 41 |
| xbrl | 20 |

### persona

| value | questions |
|---|---|
| Vietnamese retail investor | 17 |
| compliance officer | 14 |
| equity analyst | 17 |
| finance student | 15 |
| financial journalist | 16 |
| retail investor | 18 |

### mention_style

| value | questions |
|---|---|
| name | 49 |
| nickname | 3 |
| ticker | 45 |

### company

| value | questions |
|---|---|
| (none) | 5 |
| ALPHABET INC. | 18 |
| AMAZON.COM, INC. | 19 |
| Apple Inc. | 21 |
| Microsoft Corporation | 21 |
| NVIDIA CORP | 19 |

### form

| value | questions |
|---|---|
| 10-K | 38 |
| 10-Q | 33 |
| 8-K | 12 |

### section_topic

| value | questions |
|---|---|
| 8k_event | 7 |
| business | 7 |
| controls | 3 |
| cybersecurity | 3 |
| ex99 | 6 |
| exhibits | 2 |
| financial_statements | 26 |
| governance | 1 |
| legal_proceedings | 4 |
| market_equity | 2 |
| market_risk | 3 |
| mdna | 9 |
| other | 3 |
| properties | 2 |
| risk_factors | 7 |

### chunk_kind

| value | questions |
|---|---|
| table | 42 |
| text | 41 |

### fiscal_year

| value | questions |
|---|---|
| 2021 | 12 |
| 2022 | 14 |
| 2023 | 10 |
| 2024 | 15 |
| 2025 | 12 |
| 2026 | 11 |
| 2027 | 3 |
| None | 12 |

### cell.cik

| value | questions |
|---|---|
| 0000320193 | 17 |
| 0000320193|0000789019 | 1 |
| 0000789019 | 19 |
| 0001018724 | 16 |
| 0001018724|0000320193 | 2 |
| 0001018724|0000789019 | 1 |
| 0001045810 | 17 |
| 0001045810|0000320193 | 1 |
| 0001652044 | 17 |
| 0001652044|0001045810 | 1 |

### cell.form

| value | questions |
|---|---|
| 10-K | 38 |
| 10-Q | 33 |
| 8-K | 12 |

### cell.document_type

| value | questions |
|---|---|
| 10-K | 38 |
| 10-Q | 33 |
| 8-K | 6 |
| 8-K|EX-99.2 | 1 |
| EX-99.01|EX-99.1 | 1 |
| EX-99.1 | 4 |

### cell.section_topic

| value | questions |
|---|---|
| 8k_event | 6 |
| 8k_event|ex99 | 1 |
| business | 7 |
| controls | 3 |
| cybersecurity | 3 |
| ex99 | 5 |
| exhibits | 2 |
| financial_statements | 25 |
| governance | 1 |
| legal_proceedings | 4 |
| market_equity | 2 |
| market_risk | 3 |
| mdna | 8 |
| mdna|financial_statements | 1 |
| other | 3 |
| properties | 2 |
| risk_factors | 7 |

### cell.chunk_kind

| value | questions |
|---|---|
| table | 42 |
| text | 41 |

### cell.fiscal_year

| value | questions |
|---|---|
| 2021 | 11 |
| 2021|2022 | 1 |
| 2022 | 13 |
| 2023 | 9 |
| 2023|2024 | 1 |
| 2024 | 13 |
| 2024|2025 | 1 |
| 2025 | 10 |
| 2025|2026 | 1 |
| 2026 | 8 |
| 2026|2027 | 1 |
| 2027 | 1 |
| 2027|2026 | 1 |

### cell.fiscal_period

| value | questions |
|---|---|
| FY | 38 |
| Q1 | 12 |
| Q2 | 8 |
| Q2|Q3 | 1 |
| Q3 | 12 |

### cell.part

| value | questions |
|---|---|
| 1/2 | 11 |
| 1/3 | 1 |
| 2/2 | 13 |
| 2/2|1/4 | 1 |
| 2/3 | 4 |
| 2/4 | 1 |
| 3/3 | 6 |
| 3/4 | 1 |
| whole | 44 |
| whole|4/4 | 1 |

### cell.concept

| value | questions |
|---|---|
| AllocatedShareBasedCompensationExpense | 1 |
| Assets | 1 |
| CostOfRevenue | 1 |
| EarningsPerShareDiluted | 1 |
| GrossProfit | 1 |
| IncomeTaxExpenseBenefit | 1 |
| InventoryNet | 1 |
| Liabilities | 1 |
| LongTermDebtNoncurrent | 1 |
| NetCashProvidedByUsedInOperatingActivities | 1 |
| NetIncomeLoss | 1 |
| OperatingExpenses | 1 |
| OperatingIncomeLoss | 1 |
| PaymentsForRepurchaseOfCommonStock | 1 |
| PaymentsOfDividends | 1 |
| PaymentsToAcquirePropertyPlantAndEquipment | 1 |
| ResearchAndDevelopmentExpense | 1 |
| RevenueFromContractWithCustomerExcludingAssessedTax | 1 |
| ShareBasedCompensation | 1 |
| StockholdersEquity | 1 |

### cell.pair_kind

| value | questions |
|---|---|
| 8k_exhibit | 1 |
| note_ref | 2 |
| table | 8 |
| text | 6 |

### cell.ask

| value | questions |
|---|---|
| values only one passage shows (each table's latest column) | 4 |

### cell.reason

| value | questions |
|---|---|
| company_outside_corpus | 5 |
| period_not_covered | 5 |
| topic_absent | 4 |
