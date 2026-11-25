# Eval set check

- questions: 246
- errors: 0
- warnings: 65
- skipped: 23, aggregation 5, calc 1, compare_companies 6, compare_periods 8, multi_hop 1, numeric 1, temporal_change 1

## Errors

None.

## Warnings

`answer_fact_not_in_passage` 11, `lexical_copy` 5, `mention_style` 1, `no_company` 4, `no_period` 4, `period_mismatch` 6, `quote_spans_blocks` 3, `relative_time` 8, `skipped` 23

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
| x0003 | skipped | the two passages are different parts of the earnings-presentation disclosures (income tax methodology and Adjusted Net Income - Normalized in 2023, other glossary terms in 2024), with no shared item to compare. |
| x0004 | skipped | the Citizens passage is a non-GAAP other operating expense table, the Aflac passage defines adjusted book value and adjusted return on equity; no shared item to compare. |
| x0013 | skipped | the two passages share no common item; the 2023 filing's passage is cut before any 2022 comparison in the earlier one, and the earlier passage shows only expense lines. |
| x0014 | skipped | the passages cover different topics (Exelon transmission investment vs Constellation debt balances), so there is no common item to compare. |
| x0016 | lexical_copy | shared run 2 content words, share 0.83 |
| x0018 | no_period | en: no year, quarter or recency |
| x0018 | relative_time | en: no relative period |
| x0020 | relative_time | en: no relative period |
| x0020 | relative_time | vi: no relative period |
| x0030 | no_period | en: no year, quarter or recency |
| x0030 | period_mismatch | en ((), ()) vs vi ((), ('FY',)) |
| x0030 | relative_time | en: no relative period |
| x0030 | relative_time | vi: no relative period |
| x0038 | no_period | en: no year, quarter or recency |
| x0038 | no_period | vi: no year, quarter or recency |
| x0038 | relative_time | en: no relative period |
| x0038 | relative_time | vi: no relative period |
| x0039 | lexical_copy | shared run 3 content words, share 0.83 |
| x0043 | skipped | the two passages are mid-list fragments of the forward-looking-statement risk lists with different item numbering, so no shared item can be compared. |
| x0045 | answer_fact_not_in_passage | not in passages: 5.8%; 5.7732% |
| x0047 | lexical_copy | shared run 3 content words, share 0.80 |
| x0048 | skipped | the hint is FDIC deposit insurance assessments but no passage discusses assessments (NTRS uninsured deposits, FITB insurance coverage, CFG FDIC only in a LIBOR sentence); a reworded topic would not follow the hint. |
| x0058 | lexical_copy | shared run 5 content words, share 0.00 |
| x0059 | answer_fact_not_in_passage | not in passages: October 31, 2022 |
| x0061 | skipped | the passages cover unrelated items (KKR segment operating earnings vs Tyson ROIC), with no shared item to compare. |
| x0064 | skipped | no passage states that the company operates or sells in China (Avery cites COVID in China as supply disruption, CF discusses Chinese competitors, IFF US-China tension); topic does not fit. |
| x0066 | skipped | the passage shows only the top of the cash flow statement and has no row for share repurchases, so the figure is not answerable from it. |
| x0068 | skipped | the two passages cover different topics (PISA/RESRAM cost recovery in 2022, MISO capacity auction prices in 2023), so there is no shared item to compare. |
| x0071 | no_company | en: REALTY INCOME CORPORATION |
| x0071 | no_company | vi: REALTY INCOME CORPORATION |
| x0072 | skipped | the hint is Medicare Advantage star ratings but only CVS discusses star ratings; UNH and CNC mention Medicare Advantage without them, so yes/no on the hint is not supported by the passages. |
| x0078 | answer_fact_not_in_passage | not in passages: December 31, 2021 |
| x0080 | skipped | the Ulta passage is listed as without the topic but its risk factor is about AI (abbreviated), so yes/no is ambiguous. |
| x0084 | skipped | the two slides cover different metrics (Erica digital usage in 3Q21, net interest yield and income in 3Q22), the chart text is jumbled and no single item is shared across periods. |
| x0085 | answer_fact_not_in_passage | not in passages: -4.5% |
| x0099 | no_company | en: REALTY INCOME CORPORATION |
| x0099 | no_company | vi: REALTY INCOME CORPORATION |
| x0108 | answer_fact_not_in_passage | not in passages: 6.1% |
| x0111 | lexical_copy | shared run 2 content words, share 0.83 |
| x0114 | answer_fact_not_in_passage | not in passages: 4.1%; -4.0692% |
| x0119 | answer_fact_not_in_passage | not in passages: -12.7% |

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
| v2-t0059 | 16 |
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
| v2-x0001 | 1 |
| v2-x0006 | 9 |
| v2-x0007 | 10 |
| v2-x0010 | 5 |
| v2-x0012 | 2 |
| v2-x0016 | 3 |
| v2-x0017 | 28 |
| v2-x0026 | 4 |
| v2-x0027 | 2 |
| v2-x0030 | 5 |
| v2-x0031 | 1 |
| v2-x0032 | 2 |
| v2-x0034 | 6 |
| v2-x0037 | 27 |
| v2-x0040 | 2 |
| v2-x0041 | 1 |
| v2-x0045 | 10 |
| v2-x0046 | 62 |
| v2-x0049 | 5 |
| v2-x0050 | 1 |
| v2-x0054 | 17 |
| v2-x0055 | 8 |
| v2-x0057 | 3 |
| v2-x0063 | 3 |
| v2-x0065 | 11 |
| v2-x0069 | 3 |
| v2-x0070 | 1 |
| v2-x0071 | 39 |
| v2-x0073 | 2 |
| v2-x0078 | 9 |
| v2-x0079 | 36 |
| v2-x0081 | 5 |
| v2-x0086 | 41 |
| v2-x0088 | 7 |
| v2-x0090 | 4 |
| v2-x0091 | 3 |
| v2-x0092 | 13 |
| v2-x0094 | 5 |
| v2-x0097 | 4 |
| v2-x0098 | 18 |
| v2-x0102 | 1 |
| v2-x0103 | 38 |
| v2-x0105 | 4 |
| v2-x0106 | 1 |
| v2-x0107 | 15 |
| v2-x0108 | 8 |
| v2-x0109 | 25 |
| v2-x0110 | 9 |
| v2-x0111 | 7 |
| v2-x0112 | 6 |
| v2-x0113 | 4 |
| v2-x0114 | 5 |
| v2-x0116 | 3 |
| v2-x0117 | 2 |
| v2-x0119 | 9 |

## Splits

| question type | dev | test |
|---|---|---|
| aggregation | 11 | 2 |
| calc | 27 | 9 |
| compare_companies | 9 | 4 |
| compare_periods | 12 | 5 |
| event_8k | 9 | 6 |
| fact | 6 | 6 |
| false_premise | 13 | 5 |
| list | 5 | 4 |
| multi_hop | 2 | 5 |
| numeric | 13 | 8 |
| ranking | 18 | 5 |
| relation | 2 | 8 |
| summary | 3 | 6 |
| table_lookup | 16 | 6 |
| temporal_change | 2 | 5 |
| unanswerable | 7 | 7 |

| tag | dev | test |
|---|---|---|
| balance_sheet_date | 20 | 5 |
| fiscal_calendar | 35 | 18 |
| former_name | 4 | 1 |
| graph | 13 | 10 |
| near_duplicate | 85 | 47 |
| relative_time | 26 | 7 |
| small_company | 61 | 28 |
| split_table | 88 | 37 |

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
| v2-x0026 | 1 |
| v2-x0030 | 1 |
| v2-x0039 | 1 |
| v2-x0045 | 1 |
| v2-x0073 | 1 |
| v2-x0077 | 1 |
| v2-x0078 | 1 |
| v2-x0085 | 2 |
| v2-x0087 | 1 |
| v2-x0093 | 1 |
| v2-x0096 | 2 |
| v2-x0102 | 2 |
| v2-x0108 | 1 |
| v2-x0115 | 2 |

## Coverage (accepted questions)

### question_type

| value | questions |
|---|---|
| aggregation | 13 |
| calc | 36 |
| compare_companies | 13 |
| compare_periods | 17 |
| event_8k | 15 |
| fact | 12 |
| false_premise | 18 |
| list | 9 |
| multi_hop | 7 |
| numeric | 21 |
| ranking | 23 |
| relation | 10 |
| summary | 9 |
| table_lookup | 22 |
| temporal_change | 7 |
| unanswerable | 14 |

### data_kind

| value | questions |
|---|---|
| none | 14 |
| table | 45 |
| text | 89 |
| xbrl | 98 |

### persona

| value | questions |
|---|---|
| Vietnamese retail investor | 41 |
| compliance officer | 41 |
| equity analyst | 37 |
| finance student | 44 |
| financial journalist | 41 |
| retail investor | 42 |

### mention_style

| value | questions |
|---|---|
| former_name | 5 |
| name | 122 |
| ticker | 119 |

### company

| value | questions |
|---|---|
| (none) | 6 |
| 3M COMPANY | 2 |
| ABBVIE INC. | 3 |
| ADVANCED MICRO DEVICES, INC. | 1 |
| AIR PRODUCTS AND CHEMICALS, INC. | 1 |
| AKAMAI TECHNOLOGIES, INC. | 1 |
| ALBEMARLE CORPORATION | 2 |
| ALEXANDRIA REAL ESTATE EQUITIES, INC. | 1 |
| ALIGN TECHNOLOGY, INC. | 3 |
| ALPHABET INC. | 1 |
| ALTRIA GROUP, INC. | 3 |
| AMAZON.COM, INC. | 1 |
| AMERICAN ELECTRIC POWER COMPANY, INC. | 2 |
| AMERICAN EXPRESS CO | 1 |
| AMERICAN INTERNATIONAL GROUP, INC. | 1 |
| AMERICAN TOWER CORP /MA/ | 2 |
| AMERIPRISE FINANCIAL, INC. | 1 |
| APA Corp | 2 |
| ARCHER-DANIELS-MIDLAND CO | 1 |
| ARES MANAGEMENT CORPORATION | 1 |
| AT&T INC. | 2 |
| ATMOS ENERGY CORP | 1 |
| AUTOZONE, INC. | 1 |
| Amgen Inc. | 1 |
| Applovin Corp | 1 |
| Aptiv PLC | 3 |
| Arthur J. Gallagher & Co. | 2 |
| Autodesk, Inc. | 2 |
| BERKSHIRE HATHAWAY INC | 1 |
| BRISTOL MYERS SQUIBB CO | 2 |
| BROADRIDGE FINANCIAL SOLUTIONS, INC. | 1 |
| BROWN & BROWN, INC. | 2 |
| BUNGE GLOBAL SA | 1 |
| BXP, INC. | 1 |
| Baker Hughes Company | 1 |
| Blackstone Inc | 1 |
| Block, Inc. | 1 |
| Booking Holdings Inc. | 1 |
| Brown-Forman Corporation | 1 |
| C.H. ROBINSON WORLDWIDE, INC. | 1 |
| CAMDEN PROPERTY TRUST | 1 |
| CASEY'S GENERAL STORES, INC. | 1 |
| CBRE GROUP, INC. | 1 |
| CDW Corp | 3 |
| CENTERPOINT ENERGY INC | 1 |
| CHEVRON CORP | 1 |
| CHURCH & DWIGHT CO INC /DE/ | 1 |
| CMS ENERGY CORPORATION | 1 |
| COCA COLA CO | 3 |
| COLGATE-PALMOLIVE COMPANY | 1 |
| CONSTELLATION BRANDS, INC. | 2 |
| CONSTELLATION ENERGY CORPORATION | 1 |
| COPART, INC. | 1 |
| CORNING INC NY | 3 |
| COSTAR GROUP, INC. | 1 |
| CRH public limited company | 1 |
| CROWN CASTLE INC. | 2 |
| CUMMINS INC. | 1 |
| Cardinal Health, Inc. | 1 |
| Charter Communications, Inc. | 1 |
| Cintas Corp | 1 |
| Comcast Corporation | 1 |
| ConocoPhillips | 1 |
| Corteva, Inc. | 1 |
| CrowdStrike Holdings, Inc. | 1 |
| D.R. Horton, Inc. | 1 |
| DEVON ENERGY CORP/DE | 1 |
| DIAMONDBACK ENERGY, INC. | 1 |
| DIGITAL REALTY TRUST, INC. | 2 |
| DOLLAR GENERAL CORPORATION | 4 |
| DOMINION ENERGY, INC | 2 |
| DOORDASH, INC. | 1 |
| DOVER CORPORATION | 1 |
| DUKE ENERGY CORPORATION | 1 |
| Dell Technologies Inc. | 1 |
| Dow Inc. | 1 |
| ECOLAB INC. | 1 |
| EDISON INTERNATIONAL | 2 |
| ELI LILLY AND COMPANY | 1 |
| ENTERGY CORPORATION | 1 |
| EOG RESOURCES, INC | 1 |
| EQT CORPORATION | 1 |
| EQUIFAX INC. | 1 |
| EQUINIX, INC. | 1 |
| ERIE INDEMNITY COMPANY | 1 |
| EVERSOURCE ENERGY | 1 |
| EXPAND ENERGY CORPORATION | 5 |
| EXPEDIA GROUP, INC. | 1 |
| EXTRA SPACE STORAGE INC. | 1 |
| ExxonMobil Holdings Corporation | 2 |
| F5, Inc. | 1 |
| FASTENAL CO | 1 |
| FEDEX CORP | 1 |
| FERGUSON ENTERPRISES INC. | 1 |
| FIRSTENERGY CORP | 1 |
| FLEX LTD. | 2 |
| FORD MOTOR CO | 2 |
| FORTINET, INC. | 3 |
| FRANKLIN RESOURCES, INC. | 1 |
| Fox Corp | 1 |
| Freeport-McMoRan Inc. | 1 |
| GE HealthCare Technologies Inc. | 1 |
| GENERAL DYNAMICS CORPORATION | 1 |
| Gen Digital Inc. | 1 |
| Generac Holdings Inc. | 1 |
| General Motors Co | 1 |
| HALLIBURTON COMPANY | 1 |
| HCA HEALTHCARE, INC. | 1 |
| HOME DEPOT, INC. | 1 |
| HORMEL FOODS CORPORATION | 1 |
| HOST HOTELS & RESORTS, INC. | 3 |
| HOWMET AEROSPACE INC. | 1 |
| HP Inc. | 1 |
| HUNTINGTON INGALLS INDUSTRIES, INC. | 1 |
| Healthpeak Properties, Inc. | 2 |
| INTERNATIONAL FLAVORS & FRAGRANCES INC | 3 |
| IRON MOUNTAIN INC | 2 |
| Ingersoll Rand Inc. | 1 |
| Interactive Brokers Group, Inc. | 1 |
| International Paper Company | 1 |
| Invesco Ltd. | 1 |
| J.B. HUNT TRANSPORT SERVICES, INC. | 1 |
| KINDER MORGAN, INC. | 3 |
| L3HARRIS TECHNOLOGIES, INC. | 1 |
| LAS VEGAS SANDS CORP. | 1 |
| LENNAR CORP /NEW/ | 1 |
| LENNOX INTERNATIONAL INC | 1 |
| LOWES COMPANIES INC | 1 |
| LYONDELLBASELL INDUSTRIES N.V. | 1 |
| Live Nation Entertainment, Inc. | 2 |
| Lumentum Holdings Inc. | 1 |
| M&T BANK CORP | 1 |
| MGM Resorts International | 1 |
| MICROCHIP TECHNOLOGY INCORPORATED | 1 |
| MID-AMERICA APARTMENT COMMUNITIES, INC. | 1 |
| MODERNA, INC. | 2 |
| MOSAIC CO | 2 |
| MSCI Inc. | 2 |
| Marathon Petroleum Corporation | 3 |
| Marsh & McLennan Companies, Inc. | 1 |
| Martin Marietta Materials, Inc. | 1 |
| Mastercard Incorporated | 2 |
| McCormick & Company, Inc. | 4 |
| McKESSON CORPORATION | 1 |
| Medtronic plc | 1 |
| Meta Platforms, Inc. | 4 |
| Mettler-Toledo International Inc. | 3 |
| Mondelēz International, Inc. | 1 |
| Monster Beverage Corporation | 2 |
| Motorola Solutions, Inc. | 1 |
| NETFLIX INC | 2 |
| NEWS CORPORATION | 1 |
| NEXTERA ENERGY, INC. | 4 |
| NRG ENERGY, INC. | 2 |
| NUCOR CORP | 1 |
| NVIDIA CORP | 2 |
| NVR, Inc. | 1 |
| NetApp, Inc. | 1 |
| O Reilly Automotive Inc | 3 |
| OCCIDENTAL PETROLEUM CORPORATION | 1 |
| ONEOK INC /NEW/ | 1 |
| PACKAGING CORP OF AMERICA | 1 |
| PAYCHEX, INC. | 1 |
| PG&E CORPORATION | 1 |
| PPL Corporation | 1 |
| PRICE T ROWE GROUP INC | 1 |
| PRINCIPAL FINANCIAL GROUP, INC. | 1 |
| PROCTER & GAMBLE CO | 1 |
| PTC Inc. | 3 |
| PULTEGROUP, INC. | 1 |
| Paramount Skydance Corporation | 1 |
| PayPal Holdings, Inc. | 1 |
| PepsiCo, Inc. | 1 |
| Philip Morris International Inc. | 2 |
| Phillips 66 | 1 |
| Prologis, Inc. | 1 |
| Public Storage | 2 |
| QUALCOMM INC/DE | 1 |
| RAYMOND JAMES FINANCIAL, INC. | 3 |
| REALTY INCOME CORPORATION | 2 |
| REGENCY CENTERS CORPORATION | 1 |
| REGENERON PHARMACEUTICALS, INC. | 1 |
| REGIONS FINANCIAL CORP | 1 |
| ROLLINS, INC. | 1 |
| ROYAL CARIBBEAN CRUISES LTD. | 4 |
| RTX Corp | 1 |
| Ralph Lauren Corporation | 1 |
| Reddit, Inc. | 2 |
| Republic Services, Inc. | 1 |
| SHERWIN WILLIAMS CO | 3 |
| SLB LIMITED/NV | 2 |
| SOUTHERN CALIFORNIA GAS COMPANY | 2 |
| SOUTHWEST AIRLINES CO. | 1 |
| STEEL DYNAMICS, INC. | 3 |
| STERIS plc | 1 |
| SUPER MICRO COMPUTER, INC. | 1 |
| SYNCHRONY FINANCIAL | 1 |
| SYNOPSYS INC | 1 |
| Salesforce, Inc. | 2 |
| Sandisk Corp | 1 |
| Seagate Technology Holdings plc | 1 |
| Smurfit WestRock plc | 1 |
| Starbucks Corporation | 1 |
| Sysco Corporation | 1 |
| T-MOBILE US, INC. | 2 |
| TAKE-TWO INTERACTIVE SOFTWARE, INC. | 1 |
| TERADYNE, INC | 1 |
| TEXAS PACIFIC LAND CORPORATION | 1 |
| TEXTRON INC. | 1 |
| THE AES CORPORATION | 1 |
| THE CLOROX COMPANY | 2 |
| THE COOPER COMPANIES, INC. | 1 |
| TRACTOR SUPPLY CO /DE/ | 1 |
| TYSON FOODS, INC. | 1 |
| Target Corporation | 1 |
| Teledyne Technologies Inc | 1 |
| Tesla, Inc. | 1 |
| The Southern Company | 1 |
| TransDigm Group Incorporated | 1 |
| UBER TECHNOLOGIES, INC. | 2 |
| UDR, Inc. | 3 |
| UNION PACIFIC CORPORATION | 1 |
| UNITED AIRLINES HOLDINGS, INC. | 1 |
| UNITEDHEALTH GROUP INCORPORATED | 1 |
| UNIVERSAL HEALTH SERVICES, INC. | 2 |
| United Rentals, Inc. | 1 |
| VALERO ENERGY CORP/TX | 1 |
| VERTIV HOLDINGS CO | 2 |
| VICI Properties Inc. | 1 |
| VISA INC. | 1 |
| VISTRA CORP. | 1 |
| VULCAN MATERIALS COMPANY | 1 |
| Veralto Corp | 1 |
| Verizon Communications Inc. | 3 |
| WALT DISNEY CO/ | 3 |
| WATERS CORP /DE/ | 3 |
| WEYERHAEUSER COMPANY | 1 |
| WILLIAMS COMPANIES, INC. | 1 |
| WYNN RESORTS, LIMITED | 1 |
| Warner Bros. Discovery, Inc. | 4 |
| Welltower Inc. | 1 |
| Williams-Sonoma, Inc. | 1 |
| XCEL ENERGY INC | 1 |
| YUM! BRANDS, INC. | 1 |
| ZIMMER BIOMET HOLDINGS, INC. | 1 |
| Zoetis Inc. | 2 |
| eBay Inc. | 1 |

### form

| value | questions |
|---|---|
| 10-K | 107 |
| 10-Q | 85 |
| 8-K | 40 |

### section_topic

| value | questions |
|---|---|
| 8k_event | 30 |
| business | 11 |
| controls | 2 |
| cover | 8 |
| cybersecurity | 4 |
| ex99 | 14 |
| exhibits | 4 |
| financial_statements | 94 |
| governance | 4 |
| legal_proceedings | 5 |
| market_equity | 8 |
| market_risk | 8 |
| mdna | 25 |
| other | 7 |
| properties | 6 |
| risk_factors | 13 |
| signatures | 12 |

### chunk_kind

| value | questions |
|---|---|
| table | 143 |
| text | 92 |

### fiscal_year

| value | questions |
|---|---|
| 2021 | 36 |
| 2022 | 31 |
| 2023 | 34 |
| 2024 | 36 |
| 2025 | 40 |
| 2026 | 32 |
| 2027 | 9 |
| None | 40 |

### cell.cik

| value | questions |
|---|---|
| 0000002488 | 1 |
| 0000002969 | 1 |
| 0000004281 | 1 |
| 0000004904 | 2 |
| 0000004962 | 1 |
| 0000005272 | 1 |
| 0000014272 | 1 |
| 0000014693 | 1 |
| 0000016918 | 2 |
| 0000021076 | 1 |
| 0000021076|0000077476|0000029534|0000063754 | 1 |
| 0000021344 | 1 |
| 0000021344|0000029534|0000764180 | 1 |
| 0000021665 | 1 |
| 0000024741 | 1 |
| 0000026172|0000040533|0000900075|0000217346 | 1 |
| 0000027419 | 1 |
| 0000029534 | 1 |
| 0000029534|0000764180|0000063754|0000021344 | 1 |
| 0000029905|0001674101|0000066740|0000815556 | 1 |
| 0000031462 | 1 |
| 0000033185 | 1 |
| 0000037996|0001521332|0000884887|0000898173 | 1 |
| 0000038777 | 1 |
| 0000045012|0001510295|0000087347|0000895126 | 1 |
| 0000048465|0001413329|0000007084 | 1 |
| 0000051253 | 2 |
| 0000059478 | 1 |
| 0000060667|0001136869 | 1 |
| 0000062709 | 1 |
| 0000063754 | 2 |
| 0000065984 | 1 |
| 0000066740 | 1 |
| 0000068505 | 1 |
| 0000072741 | 1 |
| 0000072903 | 1 |
| 0000073309 | 1 |
| 0000074208 | 1 |
| 0000074208|0000726728|0001070750 | 1 |
| 0000075677 | 1 |
| 0000079282 | 1 |
| 0000080424 | 1 |
| 0000084839|0001048911|0001543151|0001501585 | 1 |
| 0000087347 | 1 |
| 0000089800 | 2 |
| 0000092122 | 1 |
| 0000092380 | 1 |
| 0000093410 | 1 |
| 0000096021 | 1 |
| 0000097210 | 1 |
| 0000100493 | 1 |
| 0000100517 | 1 |
| 0000100885 | 1 |
| 0000101829 | 1 |
| 0000106535 | 1 |
| 0000202058|0001543151 | 1 |
| 0000313927 | 1 |
| 0000318154 | 1 |
| 0000352915 | 2 |
| 0000354190 | 2 |
| 0000354950 | 1 |
| 0000711404 | 1 |
| 0000715957|0000731802|0000753308|0001868275 | 1 |
| 0000715957|0000753308|0001013871|0000827052 | 1 |
| 0000719955 | 1 |
| 0000720005 | 1 |
| 0000720005|0001633917|0001141391|0001408198 | 1 |
| 0000721371 | 1 |
| 0000723254 | 1 |
| 0000723531 | 1 |
| 0000726728|0000074208|0000906345|0001070750 | 1 |
| 0000726958 | 1 |
| 0000728535 | 1 |
| 0000731766 | 1 |
| 0000732712 | 3 |
| 0000732717 | 1 |
| 0000753308 | 1 |
| 0000764180 | 1 |
| 0000765880 | 1 |
| 0000765880|0001289490|0001297996|0001057352 | 1 |
| 0000766704 | 1 |
| 0000769397 | 1 |
| 0000804328|0001571996|0001086222|0000047217 | 1 |
| 0000811156 | 1 |
| 0000820027 | 1 |
| 0000827054 | 1 |
| 0000829224 | 1 |
| 0000831259 | 1 |
| 0000849395 | 1 |
| 0000849399 | 1 |
| 0000857005 | 1 |
| 0000860730 | 1 |
| 0000865752 | 2 |
| 0000866374 | 1 |
| 0000866787 | 1 |
| 0000872589 | 1 |
| 0000874761 | 1 |
| 0000882184 | 1 |
| 0000883241 | 1 |
| 0000884887|0001555280 | 1 |
| 0000895126 | 2 |
| 0000895126|0000822416 | 1 |
| 0000895126|0001510295|0000797468|0001506307 | 1 |
| 0000898173 | 1 |
| 0000898173|0000037996|0001521332|0000884887 | 1 |
| 0000906163 | 1 |
| 0000912595 | 1 |
| 0000914208|0001601712|0001393818 | 1 |
| 0000915913 | 1 |
| 0000915913|0001285785|0001755672|0000051434 | 1 |
| 0000916076 | 1 |
| 0000916365 | 1 |
| 0000920760 | 1 |
| 0000922224 | 1 |
| 0000922621 | 1 |
| 0000927653 | 1 |
| 0000946581 | 1 |
| 0001000697 | 1 |
| 0001000697|0001097149|0001037646|0001551152 | 1 |
| 0001013871 | 1 |
| 0001018724 | 1 |
| 0001020569 | 2 |
| 0001022671 | 2 |
| 0001032208 | 2 |
| 0001035002 | 1 |
| 0001035443 | 1 |
| 0001037540 | 1 |
| 0001037646 | 1 |
| 0001039684 | 1 |
| 0001041061 | 1 |
| 0001043277 | 1 |
| 0001045609 | 1 |
| 0001045810|0001613103 | 1 |
| 0001048695 | 1 |
| 0001051470|0000884887 | 1 |
| 0001051470|0000910606|0001393311|0001070750 | 1 |
| 0001053507 | 2 |
| 0001060391|0000769397 | 1 |
| 0001065280 | 2 |
| 0001067701 | 1 |
| 0001067983 | 1 |
| 0001069202 | 1 |
| 0001075531 | 1 |
| 0001090012 | 1 |
| 0001094285 | 1 |
| 0001097149|0001000697|0001682852|0001037646 | 1 |
| 0001097149|0001551152|0000014272|0001555280 | 1 |
| 0001101239 | 1 |
| 0001103982 | 1 |
| 0001108524 | 1 |
| 0001126328 | 1 |
| 0001130310|0001004980|0000827052|0000753308 | 1 |
| 0001137789 | 1 |
| 0001138118 | 1 |
| 0001141391 | 1 |
| 0001163165 | 1 |
| 0001176948 | 1 |
| 0001260221 | 1 |
| 0001262039 | 1 |
| 0001262039|0001402057|0000024741|0000857005 | 1 |
| 0001262039|0001402057|0000857005|0000024741 | 1 |
| 0001281761|0000720005|0000079282|0000036270 | 1 |
| 0001283699 | 1 |
| 0001283699|0001166691|0001326801 | 1 |
| 0001285785|0001108524 | 1 |
| 0001297996 | 1 |
| 0001300514|0000789570|0001037038|0001174922 | 1 |
| 0001318605|0001065088|0001792789 | 1 |
| 0001324424|0001031296 | 1 |
| 0001326160 | 1 |
| 0001326801 | 2 |
| 0001335258 | 2 |
| 0001375365 | 1 |
| 0001381197 | 1 |
| 0001383312 | 1 |
| 0001393311 | 1 |
| 0001396009 | 1 |
| 0001402057 | 1 |
| 0001403161 | 1 |
| 0001408198 | 1 |
| 0001413329 | 1 |
| 0001437107 | 2 |
| 0001437107|0000732717|0001744489|0001751008 | 1 |
| 0001467858 | 1 |
| 0001474735 | 1 |
| 0001489393 | 1 |
| 0001506307 | 2 |
| 0001510295 | 1 |
| 0001512673|0001113169 | 1 |
| 0001521332 | 1 |
| 0001534701 | 1 |
| 0001535527|0001045810 | 1 |
| 0001539838 | 1 |
| 0001551152 | 1 |
| 0001564708 | 1 |
| 0001633978|0000866374|0001002047 | 1 |
| 0001652044|0001713445 | 1 |
| 0001674101 | 1 |
| 0001682852 | 1 |
| 0001692819 | 1 |
| 0001699150 | 1 |
| 0001701605 | 1 |
| 0001705696 | 1 |
| 0001713445 | 1 |
| 0001744489 | 1 |
| 0001744489|0001326801|0001091667|0001754301 | 1 |
| 0001751788|0000051253|0001022671|0000089800 | 1 |
| 0001757898 | 1 |
| 0001811074 | 1 |
| 0001841666 | 1 |
| 0001841666|0000821189|0000033213|0000107263 | 1 |
| 0001932393 | 1 |
| 0001967680 | 1 |
| 0001996862 | 1 |
| 0002005951 | 1 |
| 0002011641 | 1 |
| 0002023554 | 1 |
| 0002041610|0001437107 | 1 |
| 0002115436 | 2 |

### cell.form

| value | questions |
|---|---|
| 10-K | 107 |
| 10-Q | 85 |
| 8-K | 40 |

### cell.document_type

| value | questions |
|---|---|
| 10-K | 106 |
| 10-Q | 85 |
| 8-K | 27 |
| 8-K|EX-99.1 | 2 |
| 8-K|EX-99.2 | 1 |
| EX-99.1 | 6 |
| EX-99.2 | 3 |
| EX-99.2|EX-99.1 | 1 |
| EX-99.3 | 1 |

### cell.section_topic

| value | questions |
|---|---|
| 8k_event | 27 |
| 8k_event|ex99 | 3 |
| business | 11 |
| controls | 2 |
| cover | 5 |
| cybersecurity | 4 |
| ex99 | 11 |
| financial_statements | 76 |
| financial_statements|cover | 2 |
| financial_statements|exhibits | 2 |
| financial_statements|exhibits|cover | 1 |
| financial_statements|other | 1 |
| financial_statements|signatures | 6 |
| financial_statements|signatures|exhibits | 1 |
| governance | 4 |
| legal_proceedings | 5 |
| market_equity | 8 |
| market_risk | 8 |
| mdna | 22 |
| mdna|financial_statements | 3 |
| other | 6 |
| properties | 6 |
| risk_factors | 13 |
| signatures | 3 |
| signatures|financial_statements | 2 |

### cell.chunk_kind

| value | questions |
|---|---|
| table | 140 |
| text | 89 |
| text|table | 3 |

### cell.fiscal_year

| value | questions |
|---|---|
| 2021 | 25 |
| 2021|2022 | 5 |
| 2021|2025 | 5 |
| 2022 | 24 |
| 2022|2021 | 1 |
| 2022|2026 | 1 |
| 2023 | 28 |
| 2023|2024 | 2 |
| 2023|2025 | 2 |
| 2023|2026 | 1 |
| 2024 | 30 |
| 2024|2025 | 2 |
| 2024|2026 | 1 |
| 2025 | 28 |
| 2025|2023|2024 | 1 |
| 2025|2026 | 2 |
| 2026 | 25 |
| 2026|2027 | 2 |
| 2027 | 7 |

### cell.fiscal_period

| value | questions |
|---|---|
| FY | 107 |
| Q1 | 25 |
| Q2 | 32 |
| Q2|Q1 | 1 |
| Q3 | 27 |

### cell.part

| value | questions |
|---|---|
| 1/2 | 33 |
| 1/2|1/3 | 1 |
| 1/2|2/2 | 2 |
| 1/2|2/3 | 2 |
| 1/3 | 11 |
| 1/3|1/2|whole | 1 |
| 1/3|2/3 | 1 |
| 1/3|2/3|1/2 | 1 |
| 1/4 | 1 |
| 1/5|1/4 | 1 |
| 1/6|1/5 | 1 |
| 2/2 | 18 |
| 2/2|1/2 | 4 |
| 2/2|1/2|2/3 | 1 |
| 2/2|3/3 | 1 |
| 2/2|4/4 | 1 |
| 2/3 | 13 |
| 2/4|2/3 | 1 |
| 3/3 | 11 |
| 3/3|2/2 | 1 |
| 3/4 | 6 |
| 3/4|2/3|2/2 | 1 |
| 3/4|3/5 | 1 |
| 3/5 | 1 |
| 3/5|1/2|3/3 | 1 |
| 3/5|2/9|2/4|2/3 | 1 |
| 3/6 | 1 |
| 4/4 | 1 |
| 4/4|3/3 | 1 |
| 4/5 | 2 |
| 4/7 | 1 |
| 5/5 | 1 |
| 5/6 | 1 |
| 6/6 | 1 |
| whole | 97 |
| whole|1/2 | 2 |
| whole|1/3 | 2 |
| whole|1/3|2/2 | 1 |
| whole|2/2 | 4 |

### cell.concept

| value | questions |
|---|---|
| AccountsReceivableNetCurrent | 3 |
| AllocatedShareBasedCompensationExpense | 4 |
| Assets | 3 |
| CashAndCashEquivalentsAtCarryingValue | 3 |
| CostOfGoodsAndServicesSold | 3 |
| CostOfRevenue | 2 |
| EarningsPerShareBasic | 3 |
| EarningsPerShareDiluted | 4 |
| Goodwill | 3 |
| GrossProfit | 3 |
| GrossProfit|RevenueFromContractWithCustomerExcludingAssessedTax | 1 |
| GrossProfit|Revenues | 1 |
| IncomeTaxExpenseBenefit | 5 |
| InventoryNet | 1 |
| Liabilities | 3 |
| LongTermDebtNoncurrent | 3 |
| NetCashProvidedByUsedInOperatingActivities | 4 |
| NetIncomeLoss | 4 |
| NetIncomeLoss|RevenueFromContractWithCustomerExcludingAssessedTax | 1 |
| NetIncomeLoss|Revenues | 1 |
| OperatingExpenses | 4 |
| OperatingIncomeLoss | 4 |
| OperatingIncomeLoss|RevenueFromContractWithCustomerExcludingAssessedTax | 2 |
| OperatingIncomeLoss|Revenues | 1 |
| PaymentsForRepurchaseOfCommonStock | 4 |
| PaymentsOfDividends | 3 |
| PaymentsToAcquirePropertyPlantAndEquipment | 4 |
| ResearchAndDevelopmentExpense | 2 |
| ResearchAndDevelopmentExpense|RevenueFromContractWithCustomerExcludingAssessedTax | 4 |
| ResearchAndDevelopmentExpense|Revenues | 3 |
| RevenueFromContractWithCustomerExcludingAssessedTax | 3 |
| RevenueFromContractWithCustomerExcludingAssessedTax|Revenues | 1 |
| Revenues | 2 |
| ShareBasedCompensation | 3 |
| StockholdersEquity | 3 |

### cell.sector

| value | questions |
|---|---|
| Communication | 22 |
| Consumer Discretionary | 18 |
| Consumer Discretionary|Health Care | 2 |
| Consumer Discretionary|Utilities | 1 |
| Consumer Staples | 25 |
| Energy | 21 |
| Energy|Consumer Discretionary | 1 |
| Financials | 21 |
| Health Care | 20 |
| Industrials | 24 |
| Industrials|Information Technology | 1 |
| Information Technology | 23 |
| Information Technology|Health Care | 1 |
| Materials | 19 |
| Materials|Information Technology | 1 |
| Real Estate | 21 |
| Real Estate|Consumer Discretionary | 1 |
| Utilities | 18 |

### cell.size_decile

| value | questions |
|---|---|
| 1 | 23 |
| 10 | 16 |
| 10|1|2|7 | 1 |
| 10|1|5 | 1 |
| 10|2 | 1 |
| 10|3 | 1 |
| 10|4|2 | 1 |
| 10|4|6|3 | 1 |
| 10|5 | 1 |
| 10|5|8 | 1 |
| 10|6|4 | 1 |
| 1|2 | 1 |
| 1|6|4 | 1 |
| 1|7|2 | 1 |
| 1|9 | 1 |
| 2 | 19 |
| 2|10|7 | 1 |
| 2|1|10|9 | 1 |
| 3 | 21 |
| 3|1 | 1 |
| 3|2 | 1 |
| 3|6 | 1 |
| 3|8 | 1 |
| 3|8|9 | 1 |
| 3|9|2 | 2 |
| 4 | 22 |
| 4|5|10 | 1 |
| 4|7 | 1 |
| 5 | 20 |
| 5|10 | 1 |
| 5|10|4 | 1 |
| 5|2 | 1 |
| 5|2|8 | 1 |
| 5|7|2|3 | 1 |
| 6 | 17 |
| 6|10|1 | 1 |
| 6|5 | 1 |
| 6|8 | 1 |
| 7 | 16 |
| 7|2|4|8 | 1 |
| 7|2|9|1 | 1 |
| 7|3|4 | 1 |
| 7|4 | 1 |
| 7|5|1|6 | 1 |
| 7|8 | 1 |
| 7|8|2 | 1 |
| 7|8|6 | 1 |
| 7|9|5|10 | 1 |
| 8 | 21 |
| 8|2|6|5 | 1 |
| 9 | 19 |
| 9|2|7 | 1 |
| 9|7|3 | 1 |
| 9|7|4|10 | 1 |
| 9|8|6|4 | 1 |

### tag

| value | questions |
|---|---|
| balance_sheet_date | 25 |
| fiscal_calendar | 53 |
| former_name | 5 |
| graph | 23 |
| near_duplicate | 132 |
| relative_time | 33 |
| small_company | 89 |
| split_table | 125 |

### cell.pair_kind

| value | questions |
|---|---|
| 8k_exhibit | 3 |
| business | 2 |
| mdna | 2 |
| mdna_note | 3 |
| note_ref | 1 |
| risk_factors | 3 |
| table | 19 |
| text | 11 |

### cell.ask

| value | questions |
|---|---|
| values only one passage shows (each table's latest column) | 11 |

### cell.reason

| value | questions |
|---|---|
| company_outside_corpus | 6 |
| period_not_covered | 4 |
| topic_absent | 4 |

### cell.operation

| value | questions |
|---|---|
| cagr | 7 |
| difference | 7 |
| growth | 26 |
| margin | 7 |
| ranking | 23 |
| share | 7 |

### cell.item_code

| value | questions |
|---|---|
| 1.01 | 2 |
| 1.02 | 2 |
| 1.05 | 1 |
| 2.01 | 1 |
| 2.02 | 1 |
| 2.03 | 2 |
| 5.02 | 1 |
| 5.03 | 2 |
| 5.07 | 1 |
| 8.01 | 2 |

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
| climate change | 1 |
| foreign currency exchange rates | 1 |
| labor unions and collective bargaining | 2 |
| nuclear plant decommissioning trusts | 1 |
| pandemics such as COVID-19 | 1 |
| proved oil and gas reserves | 2 |
| share repurchase programs | 1 |
| tariffs on imported goods | 1 |

### cell.probe_keys

| value | questions |
|---|---|
| artificial intelligence | 1 |
| casino|macau | 1 |
| climate change | 1 |
| collective bargaining | 1 |
| collective bargaining|labor union | 1 |
| covid-19 | 1 |
| exchange rate | 1 |
| fdic | 1 |
| nuclear|decommissioning trust | 1 |
| proved reserves | 1 |
| proved reserves|oil and gas reserves | 1 |
| repurchase program | 1 |
| tariff | 1 |

### cell.without_topic

| value | questions |
|---|---|
| 0000029534 | 1 |
| 0000040533 | 1 |
| 0000079282 | 1 |
| 0000107263 | 1 |
| 0000731802 | 1 |
| 0000804328 | 1 |
| 0000906345 | 1 |
| 0001037038 | 1 |
| 0001051470 | 1 |
| 0001510295 | 1 |
| 0001543151 | 1 |
| 0001744489 | 1 |
| 0001751008 | 1 |
