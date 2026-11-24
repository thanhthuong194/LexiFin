# v1.2 review (2026-11-24)

Retrieval failures of the v1.1 baseline included rows where the top 8 held
the answer row in another chunk (10-Q twin of an 8-K table, the same row in
a note). v1.2 widens table-row evidence to every group and question type;
an agent checked the 84 added rows: 83 valid, 1 invalid (t0071: $5.12 was
Class A for another quarter), removed by requiring another number of the row
when the value has under 4 significant digits. Valid by value but not the
named source: t0007 (asks for the earnings releases; 10-Q/10-K rows added)
and t0026 group 2 (asks for the fiscal 2022 debt note; FY23 comparatives).
Not widened (other label or table): t0069 (8-K EPS row), t0026 group 1.
