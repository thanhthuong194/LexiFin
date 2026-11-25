# Eval set check

- questions: 140
- errors: 0
- warnings: 24
- skipped: 10, aggregation 1, calc 1, compare_companies 3, compare_periods 3, multi_hop 1, temporal_change 1

## Errors

None.

## Warnings

`answer_fact_not_in_passage` 4, `mention_style` 1, `period_mismatch` 5, `quote_spans_blocks` 3, `relative_time` 1, `skipped` 10

| task | check | detail |
|---|---|---|
| t0010 | period_mismatch | en ((), ()) vs vi ((), ('FY',)) |
| t0022 | mention_style | en: no ticker of DUKE ENERGY CORPORATION |
| t0024 | skipped | the two passages share no sensible item; the 8-K item covers board resignations while the exhibit passage is only a partial non-GAAP adjustments table, so no natural question needs both. |
| t0028 | skipped | the fiscal 2023 passage is a chunk that ends after Use of Estimates (the fiscal 2024 chunk is longer and cut mid-sentence), so the apparent addition of Revenue Recognition is a chunk-boundary artifact, not a real change between the two 10-Ks. |
| t0036 | period_mismatch | en ((), ()) vs vi ((), ('FY',)) |
| t0046 | quote_spans_blocks | 'We compete with private, public and state-owned companies in' |
| t0046 | quote_spans_blocks | 'Some of our competitors are larger and have greater resource' |
| t0046 | quote_spans_blocks | 'Each of our segments is highly competitive, with no single c' |
| t0056 | relative_time | en: no relative period |
| t0058 | answer_fact_not_in_passage | not in passages: December 31, 2023 |
| t0070 | skipped | the two passages are different parts of the risk factor (privacy laws such as CCPA and LGPD versus fines on Drivers and delivery fee caps) and share no item to compare. |
| t0071 | skipped | the two passages cover unrelated boilerplate (WTW's forward-looking statements disclaimer vs Kimco's proxy-solicitation notice about RPT), with no shared item to compare. |
| t0074 | answer_fact_not_in_passage | not in passages: 8.5% |
| t0079 | skipped | the passages mention nuclear fleets, NRC rules and nuclear licenses, but none mentions decommissioning trusts, so the hint topic cannot be answered from them. |
| t0086 | skipped | the 2022 passage's columns are also nine months ended October 3, 2021 (Core Illumina/GRAIL split), so the two passages share no distinct period to compare. |
| t0090 | skipped | the net income operand (NetIncomeLoss of $30 million) is not shown in the passage, which ends at income from discontinued operations. |
| t0097 | period_mismatch | en ((), ()) vs vi ((), ('FY',)) |
| t0099 | period_mismatch | en ((), ()) vs vi ((), ('FY',)) |
| t0103 | skipped | the WEC passage is a forward-looking risk-factor list while the Kroger passage is about a debt tender offer for ACI notes, so there is nothing sensible to compare. |
| t0120 | answer_fact_not_in_passage | not in passages: December 31, 2021 |
| t0126 | period_mismatch | en ((), ()) vs vi ((), ('FY',)) |
| t0133 | answer_fact_not_in_passage | not in passages: 5.7% |
| t0139 | skipped | the passages are unlabeled table fragments with no column headers or units, so the fourth quarter columns cannot be identified from the text alone. |
| t0140 | skipped | the passages are unrelated pieces (TPL's presentation footnotes on revenue adjustments and well counts, versus PNW's forward-looking-statement risk language) with no shared item. |

## Evidence from other filings

| question | rows added |
|---|---|
| v2-t0001 | 1 |
| v2-t0007 | 2 |
| v2-t0010 | 4 |
| v2-t0011 | 4 |
| v2-t0016 | 3 |
| v2-t0022 | 2 |
| v2-t0026 | 6 |
| v2-t0027 | 7 |
| v2-t0032 | 6 |
| v2-t0033 | 3 |
| v2-t0039 | 1 |
| v2-t0040 | 8 |
| v2-t0042 | 5 |
| v2-t0043 | 6 |
| v2-t0048 | 15 |
| v2-t0049 | 5 |
| v2-t0059 | 17 |
| v2-t0064 | 9 |
| v2-t0065 | 9 |
| v2-t0074 | 6 |
| v2-t0075 | 11 |
| v2-t0080 | 13 |
| v2-t0081 | 3 |
| v2-t0083 | 5 |
| v2-t0087 | 11 |
| v2-t0088 | 8 |
| v2-t0091 | 30 |
| v2-t0097 | 1 |
| v2-t0104 | 3 |
| v2-t0106 | 3 |
| v2-t0107 | 32 |
| v2-t0111 | 3 |
| v2-t0116 | 10 |
| v2-t0120 | 10 |
| v2-t0125 | 7 |
| v2-t0127 | 7 |
| v2-t0133 | 8 |
| v2-t0136 | 3 |
| v2-t0142 | 3 |
| v2-t0145 | 7 |
| v2-t0147 | 2 |
| v2-t0148 | 6 |
| v2-t0150 | 9 |

## Splits

| question type | dev | test |
|---|---|---|
| aggregation | 3 | 2 |
| calc | 2 | 9 |
| compare_companies | 3 | 4 |
| compare_periods | 2 | 5 |
| event_8k | 4 | 6 |
| fact | 6 | 6 |
| false_premise | 1 | 5 |
| list | 5 | 4 |
| multi_hop | 2 | 5 |
| numeric | 4 | 8 |
| ranking | 3 | 5 |
| relation | 2 | 8 |
| summary | 3 | 6 |
| table_lookup | 4 | 6 |
| temporal_change | 2 | 5 |
| unanswerable | 3 | 7 |

| tag | dev | test |
|---|---|---|
| balance_sheet_date | 4 | 5 |
| fiscal_calendar | 10 | 18 |
| former_name | 1 | 1 |
| graph | 5 | 10 |
| near_duplicate | 21 | 47 |
| relative_time | 7 | 7 |
| small_company | 19 | 28 |
| split_table | 19 | 37 |

## Hard negatives dropped (show evidence)

| question | dropped |
|---|---|
| v2-t0004 | 3 |
| v2-t0015 | 1 |
| v2-t0023 | 1 |
| v2-t0026 | 1 |
| v2-t0038 | 2 |
| v2-t0048 | 1 |
| v2-t0050 | 1 |
| v2-t0053 | 1 |
| v2-t0056 | 2 |
| v2-t0074 | 1 |
| v2-t0085 | 1 |
| v2-t0088 | 2 |
| v2-t0092 | 1 |
| v2-t0095 | 1 |
| v2-t0104 | 1 |
| v2-t0107 | 1 |
| v2-t0118 | 1 |
| v2-t0120 | 2 |
| v2-t0127 | 1 |
| v2-t0147 | 1 |

## Coverage (accepted questions)

### question_type

| value | questions |
|---|---|
| aggregation | 5 |
| calc | 11 |
| compare_companies | 7 |
| compare_periods | 7 |
| event_8k | 10 |
| fact | 12 |
| false_premise | 6 |
| list | 9 |
| multi_hop | 7 |
| numeric | 12 |
| ranking | 8 |
| relation | 10 |
| summary | 9 |
| table_lookup | 10 |
| temporal_change | 7 |
| unanswerable | 10 |

### data_kind

| value | questions |
|---|---|
| none | 10 |
| table | 24 |
| text | 69 |
| xbrl | 37 |

### persona

| value | questions |
|---|---|
| Vietnamese retail investor | 23 |
| compliance officer | 23 |
| equity analyst | 21 |
| finance student | 26 |
| financial journalist | 23 |
| retail investor | 24 |

### mention_style

| value | questions |
|---|---|
| former_name | 2 |
| name | 72 |
| ticker | 66 |

### company

| value | questions |
|---|---|
| (none) | 4 |
| ABBVIE INC. | 1 |
| ADVANCED MICRO DEVICES, INC. | 1 |
| AIR PRODUCTS AND CHEMICALS, INC. | 1 |
| ALBEMARLE CORPORATION | 2 |
| ALEXANDRIA REAL ESTATE EQUITIES, INC. | 1 |
| ALIGN TECHNOLOGY, INC. | 1 |
| ALPHABET INC. | 1 |
| AMAZON.COM, INC. | 1 |
| AMERICAN ELECTRIC POWER COMPANY, INC. | 1 |
| AMERICAN EXPRESS CO | 1 |
| AMERICAN TOWER CORP /MA/ | 2 |
| APA Corp | 2 |
| ARCHER-DANIELS-MIDLAND CO | 1 |
| ARES MANAGEMENT CORPORATION | 1 |
| AT&T INC. | 1 |
| AUTOZONE, INC. | 1 |
| Amgen Inc. | 1 |
| Arthur J. Gallagher & Co. | 2 |
| Autodesk, Inc. | 2 |
| BERKSHIRE HATHAWAY INC | 1 |
| BRISTOL MYERS SQUIBB CO | 2 |
| BROADRIDGE FINANCIAL SOLUTIONS, INC. | 1 |
| BROWN & BROWN, INC. | 1 |
| BUNGE GLOBAL SA | 1 |
| BXP, INC. | 1 |
| Baker Hughes Company | 1 |
| Blackstone Inc | 1 |
| Block, Inc. | 1 |
| Brown-Forman Corporation | 1 |
| C.H. ROBINSON WORLDWIDE, INC. | 1 |
| CASEY'S GENERAL STORES, INC. | 1 |
| CBRE GROUP, INC. | 1 |
| CHEVRON CORP | 1 |
| CHURCH & DWIGHT CO INC /DE/ | 1 |
| CMS ENERGY CORPORATION | 1 |
| COLGATE-PALMOLIVE COMPANY | 1 |
| CONSTELLATION BRANDS, INC. | 1 |
| COSTAR GROUP, INC. | 1 |
| CRH public limited company | 1 |
| CROWN CASTLE INC. | 1 |
| Charter Communications, Inc. | 1 |
| ConocoPhillips | 1 |
| Corteva, Inc. | 1 |
| D.R. Horton, Inc. | 1 |
| DIAMONDBACK ENERGY, INC. | 1 |
| DIGITAL REALTY TRUST, INC. | 2 |
| DOMINION ENERGY, INC | 1 |
| DOORDASH, INC. | 1 |
| DUKE ENERGY CORPORATION | 1 |
| ECOLAB INC. | 1 |
| EDISON INTERNATIONAL | 1 |
| ELI LILLY AND COMPANY | 1 |
| ENTERGY CORPORATION | 1 |
| EOG RESOURCES, INC | 1 |
| EQT CORPORATION | 1 |
| EQUIFAX INC. | 1 |
| EXPAND ENERGY CORPORATION | 1 |
| EXTRA SPACE STORAGE INC. | 1 |
| ExxonMobil Holdings Corporation | 1 |
| F5, Inc. | 1 |
| FEDEX CORP | 1 |
| FLEX LTD. | 2 |
| FRANKLIN RESOURCES, INC. | 1 |
| Fox Corp | 1 |
| Freeport-McMoRan Inc. | 1 |
| GE HealthCare Technologies Inc. | 1 |
| Generac Holdings Inc. | 1 |
| General Motors Co | 1 |
| HCA HEALTHCARE, INC. | 1 |
| HOME DEPOT, INC. | 1 |
| HORMEL FOODS CORPORATION | 1 |
| HOWMET AEROSPACE INC. | 1 |
| HUNTINGTON INGALLS INDUSTRIES, INC. | 1 |
| Healthpeak Properties, Inc. | 2 |
| INTERNATIONAL FLAVORS & FRAGRANCES INC | 1 |
| Ingersoll Rand Inc. | 1 |
| Interactive Brokers Group, Inc. | 1 |
| International Paper Company | 1 |
| Invesco Ltd. | 1 |
| J.B. HUNT TRANSPORT SERVICES, INC. | 1 |
| KINDER MORGAN, INC. | 1 |
| LAS VEGAS SANDS CORP. | 1 |
| LENNAR CORP /NEW/ | 1 |
| LOWES COMPANIES INC | 1 |
| Live Nation Entertainment, Inc. | 2 |
| Lumentum Holdings Inc. | 1 |
| M&T BANK CORP | 1 |
| MGM Resorts International | 1 |
| MICROCHIP TECHNOLOGY INCORPORATED | 1 |
| MID-AMERICA APARTMENT COMMUNITIES, INC. | 1 |
| MOSAIC CO | 2 |
| Marsh & McLennan Companies, Inc. | 1 |
| Martin Marietta Materials, Inc. | 1 |
| McCormick & Company, Inc. | 1 |
| McKESSON CORPORATION | 1 |
| Medtronic plc | 1 |
| Meta Platforms, Inc. | 1 |
| Mondelēz International, Inc. | 1 |
| Monster Beverage Corporation | 2 |
| Motorola Solutions, Inc. | 1 |
| NETFLIX INC | 2 |
| NEXTERA ENERGY, INC. | 1 |
| NRG ENERGY, INC. | 1 |
| NUCOR CORP | 1 |
| NVIDIA CORP | 1 |
| NetApp, Inc. | 1 |
| PRICE T ROWE GROUP INC | 1 |
| PRINCIPAL FINANCIAL GROUP, INC. | 1 |
| PROCTER & GAMBLE CO | 1 |
| Philip Morris International Inc. | 2 |
| Phillips 66 | 1 |
| Prologis, Inc. | 1 |
| RAYMOND JAMES FINANCIAL, INC. | 1 |
| REGIONS FINANCIAL CORP | 1 |
| ROLLINS, INC. | 1 |
| ROYAL CARIBBEAN CRUISES LTD. | 1 |
| RTX Corp | 1 |
| Ralph Lauren Corporation | 1 |
| Reddit, Inc. | 2 |
| Republic Services, Inc. | 1 |
| SHERWIN WILLIAMS CO | 1 |
| SOUTHERN CALIFORNIA GAS COMPANY | 2 |
| SOUTHWEST AIRLINES CO. | 1 |
| STERIS plc | 1 |
| SUPER MICRO COMPUTER, INC. | 1 |
| SYNCHRONY FINANCIAL | 1 |
| SYNOPSYS INC | 1 |
| Salesforce, Inc. | 2 |
| Starbucks Corporation | 1 |
| Sysco Corporation | 1 |
| TAKE-TWO INTERACTIVE SOFTWARE, INC. | 1 |
| TEXAS PACIFIC LAND CORPORATION | 1 |
| THE AES CORPORATION | 1 |
| THE COOPER COMPANIES, INC. | 1 |
| TRACTOR SUPPLY CO /DE/ | 1 |
| Target Corporation | 1 |
| Teledyne Technologies Inc | 1 |
| Tesla, Inc. | 1 |
| TransDigm Group Incorporated | 1 |
| UBER TECHNOLOGIES, INC. | 1 |
| UNION PACIFIC CORPORATION | 1 |
| UNITED AIRLINES HOLDINGS, INC. | 1 |
| UNITEDHEALTH GROUP INCORPORATED | 1 |
| VALERO ENERGY CORP/TX | 1 |
| VICI Properties Inc. | 1 |
| VULCAN MATERIALS COMPANY | 1 |
| Veralto Corp | 1 |
| Verizon Communications Inc. | 2 |
| WALT DISNEY CO/ | 1 |
| WEYERHAEUSER COMPANY | 1 |
| WILLIAMS COMPANIES, INC. | 1 |
| WYNN RESORTS, LIMITED | 1 |
| Warner Bros. Discovery, Inc. | 1 |
| Welltower Inc. | 1 |
| Williams-Sonoma, Inc. | 1 |
| XCEL ENERGY INC | 1 |
| YUM! BRANDS, INC. | 1 |
| ZIMMER BIOMET HOLDINGS, INC. | 1 |
| Zoetis Inc. | 1 |
| eBay Inc. | 1 |

### form

| value | questions |
|---|---|
| 10-K | 60 |
| 10-Q | 45 |
| 8-K | 25 |

### section_topic

| value | questions |
|---|---|
| 8k_event | 18 |
| business | 8 |
| controls | 2 |
| cover | 5 |
| cybersecurity | 4 |
| ex99 | 10 |
| exhibits | 2 |
| financial_statements | 39 |
| governance | 2 |
| legal_proceedings | 5 |
| market_equity | 5 |
| market_risk | 5 |
| mdna | 18 |
| other | 4 |
| properties | 4 |
| risk_factors | 9 |
| signatures | 3 |

### chunk_kind

| value | questions |
|---|---|
| table | 63 |
| text | 69 |

### fiscal_year

| value | questions |
|---|---|
| 2021 | 19 |
| 2022 | 17 |
| 2023 | 22 |
| 2024 | 21 |
| 2025 | 22 |
| 2026 | 13 |
| 2027 | 5 |
| None | 25 |

### cell.cik

| value | questions |
|---|---|
| 0000002488 | 1 |
| 0000002969 | 1 |
| 0000004281 | 1 |
| 0000004904 | 1 |
| 0000004962 | 1 |
| 0000014272 | 1 |
| 0000014693 | 1 |
| 0000016918 | 1 |
| 0000021665 | 1 |
| 0000027419 | 1 |
| 0000031462 | 1 |
| 0000033185 | 1 |
| 0000038777 | 1 |
| 0000048465|0001413329|0000007084 | 1 |
| 0000051253 | 1 |
| 0000059478 | 1 |
| 0000060667|0001136869 | 1 |
| 0000062709 | 1 |
| 0000063754 | 1 |
| 0000065984 | 1 |
| 0000068505 | 1 |
| 0000072903 | 1 |
| 0000073309 | 1 |
| 0000080424 | 1 |
| 0000084839|0001048911|0001543151|0001501585 | 1 |
| 0000089800 | 1 |
| 0000092380 | 1 |
| 0000093410 | 1 |
| 0000096021 | 1 |
| 0000100517 | 1 |
| 0000100885 | 1 |
| 0000101829 | 1 |
| 0000106535 | 1 |
| 0000313927 | 1 |
| 0000318154 | 1 |
| 0000354190 | 2 |
| 0000354950 | 1 |
| 0000711404 | 1 |
| 0000715957|0000753308|0001013871|0000827052 | 1 |
| 0000719955 | 1 |
| 0000726958 | 1 |
| 0000728535 | 1 |
| 0000731766 | 1 |
| 0000732712 | 2 |
| 0000732717 | 1 |
| 0000765880 | 1 |
| 0000765880|0001289490|0001297996|0001057352 | 1 |
| 0000766704 | 1 |
| 0000769397 | 1 |
| 0000811156 | 1 |
| 0000827054 | 1 |
| 0000829224 | 1 |
| 0000831259 | 1 |
| 0000849395 | 1 |
| 0000860730 | 1 |
| 0000865752 | 2 |
| 0000866374 | 1 |
| 0000866787 | 1 |
| 0000874761 | 1 |
| 0000882184 | 1 |
| 0000883241 | 1 |
| 0000895126 | 1 |
| 0000912595 | 1 |
| 0000914208|0001601712|0001393818 | 1 |
| 0000915913 | 1 |
| 0000915913|0001285785|0001755672|0000051434 | 1 |
| 0000916076 | 1 |
| 0000916365 | 1 |
| 0000920760 | 1 |
| 0000927653 | 1 |
| 0000946581 | 1 |
| 0001018724 | 1 |
| 0001032208 | 2 |
| 0001035002 | 1 |
| 0001035443 | 1 |
| 0001037540 | 1 |
| 0001041061 | 1 |
| 0001043277 | 1 |
| 0001045609 | 1 |
| 0001045810|0001613103 | 1 |
| 0001048695 | 1 |
| 0001051470|0000884887 | 1 |
| 0001053507 | 2 |
| 0001060391|0000769397 | 1 |
| 0001065280 | 2 |
| 0001067983 | 1 |
| 0001094285 | 1 |
| 0001097149|0001551152|0000014272|0001555280 | 1 |
| 0001103982 | 1 |
| 0001108524 | 1 |
| 0001126328 | 1 |
| 0001138118 | 1 |
| 0001163165 | 1 |
| 0001176948 | 1 |
| 0001260221 | 1 |
| 0001281761|0000720005|0000079282|0000036270 | 1 |
| 0001285785|0001108524 | 1 |
| 0001297996 | 1 |
| 0001300514|0000789570|0001037038|0001174922 | 1 |
| 0001318605|0001065088|0001792789 | 1 |
| 0001326160 | 1 |
| 0001335258 | 2 |
| 0001375365 | 1 |
| 0001381197 | 1 |
| 0001383312 | 1 |
| 0001396009 | 1 |
| 0001413329 | 1 |
| 0001437107 | 1 |
| 0001467858 | 1 |
| 0001474735 | 1 |
| 0001506307 | 1 |
| 0001512673|0001113169 | 1 |
| 0001534701 | 1 |
| 0001539838 | 1 |
| 0001633978|0000866374|0001002047 | 1 |
| 0001652044|0001713445 | 1 |
| 0001699150 | 1 |
| 0001701605 | 1 |
| 0001705696 | 1 |
| 0001713445 | 1 |
| 0001744489|0001326801|0001091667|0001754301 | 1 |
| 0001757898 | 1 |
| 0001811074 | 1 |
| 0001841666 | 1 |
| 0001841666|0000821189|0000033213|0000107263 | 1 |
| 0001932393 | 1 |
| 0001967680 | 1 |
| 0001996862 | 1 |
| 0002115436 | 1 |

### cell.form

| value | questions |
|---|---|
| 10-K | 60 |
| 10-Q | 45 |
| 8-K | 25 |

### cell.document_type

| value | questions |
|---|---|
| 10-K | 60 |
| 10-Q | 45 |
| 8-K | 15 |
| 8-K|EX-99.1 | 2 |
| 8-K|EX-99.2 | 1 |
| EX-99.1 | 5 |
| EX-99.2 | 2 |

### cell.section_topic

| value | questions |
|---|---|
| 8k_event | 15 |
| 8k_event|ex99 | 3 |
| business | 8 |
| controls | 2 |
| cover | 3 |
| cybersecurity | 4 |
| ex99 | 7 |
| financial_statements | 31 |
| financial_statements|cover | 1 |
| financial_statements|exhibits|cover | 1 |
| financial_statements|other | 1 |
| financial_statements|signatures | 1 |
| financial_statements|signatures|exhibits | 1 |
| governance | 2 |
| legal_proceedings | 5 |
| market_equity | 5 |
| market_risk | 5 |
| mdna | 15 |
| mdna|financial_statements | 3 |
| other | 3 |
| properties | 4 |
| risk_factors | 9 |
| signatures | 1 |

### cell.chunk_kind

| value | questions |
|---|---|
| table | 61 |
| text | 67 |
| text|table | 2 |

### cell.fiscal_year

| value | questions |
|---|---|
| 2021 | 14 |
| 2021|2022 | 3 |
| 2021|2025 | 1 |
| 2022 | 12 |
| 2022|2021 | 1 |
| 2022|2026 | 1 |
| 2023 | 19 |
| 2023|2024 | 2 |
| 2023|2025 | 1 |
| 2024 | 17 |
| 2024|2025 | 2 |
| 2025 | 16 |
| 2025|2026 | 2 |
| 2026 | 9 |
| 2026|2027 | 1 |
| 2027 | 4 |

### cell.fiscal_period

| value | questions |
|---|---|
| FY | 60 |
| Q1 | 12 |
| Q2 | 17 |
| Q2|Q1 | 1 |
| Q3 | 15 |

### cell.part

| value | questions |
|---|---|
| 1/2 | 18 |
| 1/2|2/2 | 2 |
| 1/3 | 2 |
| 1/3|1/2|whole | 1 |
| 1/6|1/5 | 1 |
| 2/2 | 9 |
| 2/2|1/2 | 2 |
| 2/2|3/3 | 1 |
| 2/2|4/4 | 1 |
| 2/3 | 5 |
| 2/4|2/3 | 1 |
| 3/3 | 7 |
| 3/4 | 1 |
| 3/5 | 1 |
| 3/5|2/9|2/4|2/3 | 1 |
| 3/6 | 1 |
| 4/5 | 1 |
| 4/7 | 1 |
| 5/5 | 1 |
| 6/6 | 1 |
| whole | 67 |
| whole|1/2 | 1 |
| whole|1/3 | 1 |
| whole|2/2 | 3 |

### cell.concept

| value | questions |
|---|---|
| AccountsReceivableNetCurrent | 1 |
| AllocatedShareBasedCompensationExpense | 1 |
| Assets | 1 |
| CashAndCashEquivalentsAtCarryingValue | 1 |
| CostOfGoodsAndServicesSold | 1 |
| CostOfRevenue | 1 |
| EarningsPerShareBasic | 1 |
| EarningsPerShareDiluted | 2 |
| Goodwill | 1 |
| GrossProfit | 1 |
| GrossProfit|Revenues | 1 |
| IncomeTaxExpenseBenefit | 2 |
| Liabilities | 2 |
| NetCashProvidedByUsedInOperatingActivities | 2 |
| NetIncomeLoss | 2 |
| OperatingExpenses | 2 |
| OperatingIncomeLoss | 2 |
| OperatingIncomeLoss|RevenueFromContractWithCustomerExcludingAssessedTax | 1 |
| PaymentsForRepurchaseOfCommonStock | 2 |
| PaymentsOfDividends | 1 |
| PaymentsToAcquirePropertyPlantAndEquipment | 2 |
| ResearchAndDevelopmentExpense | 1 |
| ResearchAndDevelopmentExpense|RevenueFromContractWithCustomerExcludingAssessedTax | 1 |
| ResearchAndDevelopmentExpense|Revenues | 1 |
| RevenueFromContractWithCustomerExcludingAssessedTax | 1 |
| Revenues | 1 |
| ShareBasedCompensation | 1 |
| StockholdersEquity | 1 |

### cell.sector

| value | questions |
|---|---|
| Communication | 12 |
| Consumer Discretionary | 12 |
| Consumer Discretionary|Health Care | 1 |
| Consumer Staples | 15 |
| Energy | 12 |
| Financials | 12 |
| Health Care | 10 |
| Industrials | 14 |
| Industrials|Information Technology | 1 |
| Information Technology | 11 |
| Information Technology|Health Care | 1 |
| Materials | 11 |
| Materials|Information Technology | 1 |
| Real Estate | 13 |
| Real Estate|Consumer Discretionary | 1 |
| Utilities | 9 |

### cell.size_decile

| value | questions |
|---|---|
| 1 | 14 |
| 10 | 9 |
| 10|1|2|7 | 1 |
| 10|1|5 | 1 |
| 10|2 | 1 |
| 10|4|2 | 1 |
| 10|4|6|3 | 1 |
| 10|5|8 | 1 |
| 1|2 | 1 |
| 1|6|4 | 1 |
| 1|9 | 1 |
| 2 | 11 |
| 2|1|10|9 | 1 |
| 3 | 13 |
| 3|6 | 1 |
| 3|8 | 1 |
| 4 | 14 |
| 5 | 12 |
| 5|2|8 | 1 |
| 6 | 8 |
| 6|5 | 1 |
| 6|8 | 1 |
| 7 | 12 |
| 7|4 | 1 |
| 7|8|6 | 1 |
| 8 | 13 |
| 9 | 11 |
| 9|7|3 | 1 |
| 9|7|4|10 | 1 |

### tag

| value | questions |
|---|---|
| balance_sheet_date | 9 |
| fiscal_calendar | 28 |
| former_name | 2 |
| graph | 15 |
| near_duplicate | 68 |
| relative_time | 14 |
| small_company | 47 |
| split_table | 56 |

### cell.pair_kind

| value | questions |
|---|---|
| 8k_exhibit | 3 |
| business | 2 |
| mdna | 2 |
| mdna_note | 3 |
| note_ref | 1 |
| risk_factors | 3 |
| table | 10 |
| text | 4 |

### cell.ask

| value | questions |
|---|---|
| values only one passage shows (each table's latest column) | 5 |

### cell.reason

| value | questions |
|---|---|
| company_outside_corpus | 4 |
| period_not_covered | 3 |
| topic_absent | 3 |

### cell.operation

| value | questions |
|---|---|
| cagr | 2 |
| difference | 2 |
| growth | 9 |
| margin | 2 |
| ranking | 8 |
| share | 2 |

### cell.item_code

| value | questions |
|---|---|
| 1.01 | 1 |
| 1.02 | 1 |
| 1.05 | 1 |
| 2.01 | 1 |
| 2.02 | 1 |
| 2.03 | 1 |
| 5.02 | 1 |
| 5.03 | 1 |
| 5.07 | 1 |
| 8.01 | 1 |

### cell.relation

| value | questions |
|---|---|
| competitor | 2 |
| customer | 2 |
| officer | 2 |
| partner | 1 |
| subsidiary | 1 |
| supplier | 2 |

### cell.probe

| value | questions |
|---|---|
| FDIC deposit insurance assessments | 1 |
| artificial intelligence | 1 |
| casino gaming licenses in Macau | 1 |
| labor unions and collective bargaining | 1 |
| proved oil and gas reserves | 1 |

### cell.probe_keys

| value | questions |
|---|---|
| artificial intelligence | 1 |
| casino|macau | 1 |
| collective bargaining | 1 |
| fdic | 1 |
| proved reserves | 1 |

### cell.without_topic

| value | questions |
|---|---|
| 0000079282 | 1 |
| 0000107263 | 1 |
| 0001037038 | 1 |
| 0001543151 | 1 |
| 0001744489 | 1 |
