# Note Delta Table

## Silver: Đầy đủ semantic-layer

- Filing metadata
- Các document và exhibit
- PART, ITEM, note, heading
- Paragraph, list, signature
- Tất cả bảng và cell
- XBRL fact, context, dimension, unit
- Concept, label, calculation/presentation/definition relationships
- Link và provenance quay về bronze

---

## Nhóm submission và định danh - 6 bảng

| | | | |
| --- | --- | --- | --- |
| # | Delta table | Grain | Vì sao cần |
| 1 | companies | Một CIK | Định danh ổn định của doanh nghiệp |
| 2 | filings | Một accession number | Đại diện cho một lần nộp 10-K/10-Q/8-K |
| 3 | filing_entities | Một entity-role trong một filing | Giữ thông tin doanh nghiệp tại đúng thời điểm filing |
| 4 | documents | <DOCUMENT> trong full submission | Inventory toàn bộ primary document, exhibit, XBRL, graphic, generated files |
| 5 | exhibits | Một dòng Exhibit Index | Exhibit được liệt kê không phải lúc nào cũng có payload đính kèm |
| 6 | document_links | Một hyperlink/anchor | Giữ TOC links, internal anchors, exhibit links và external links |

---

### 1. companies

| | | | | | | |
| --- | --- | --- | --- | --- | --- | --- |
| cik (pid) | latest_company_name | ticker_aliases | sic_code | sic_name | first_seen_date | latest_seen_date |

### 2. filings 

| | | | | | | | | | | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| filing_id | form | filing_data | report_date | accepted_at | public_document_count | fiscal_year | fiscal_period | amendent_flag | full_submission_path | full_submission_sha256 | primary_document_path | primary_document_sha_256 | header_extra_json | ingestion_run_id |
| # accession number | # 10-K, 10-Q, 8-K | | | | | | | | | | | | | |

### 3. filing_entities

| | | | | | | | | | | | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| filing_entity_id | filing_id | cik | entity_role | entity_order | conformed_name | sic_code | irs_number | state_of_incorporation | fiscal_year_end |sec_act | sec_file_number | business_address | mail_address | former_names | extra_json |
| | | | # filer, subject_company,... | | | | | | | | | # struct | #s struct | # array<struct> | |

### 4. documents

**Là bảng đặc biệt quan trọng:**
- 10-K có 90 document blocks
- 10-Q có 62
- 8-K có 13
- Sequence có thể bị nhảy số, nên không được tự đánh giá lại.

| | | | | | | | | | | | | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| document_id | filing_id | sequence | sec_document_type | filename | description | document_role | content_kind | media_type | payload_size_bytes | payload_sha256 | submission_start_offset | submission_end_offset | standalone_path | is_primary | is_canonical_content | parse_status |

**document_role**

| | | | | | | | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| primary_document | exhibit | xbrl_schema | sbrl_calculation | xbrl_definition | xbrl_label | xbrl_presentation | xbrl_instance | xbrl_instance | graphic | generated | other |

> Primary document trong full-submission.txt và primary-document.html là 2 representation của cùng một logical document -> chỉ tạo một **document_id**, không tạo hai dòng duplicate.

### 5. exhibits

> SEC xác nhận Exhibit INdex có thể trỏ đến exhibit nộp trong filing hiện tại hoặc exhibit của filing trước đó được incorporated by reference. -> exhitbits và documents có 2 grain khác nhau.
### 6. document_links

## Nhóm nội dung tài liệu - 4 bảng

| | | | |
| --- | --- | --- | --- |
| # | Delta table | Grain | Nội dung | 
| 7 | sections | một section occurrence | PART, ITEM, note, statement, heading |
| 8 | content_blocks | một semantic block theo thứ tự | Paragraph, list item, table reference, image, signature |
| 9 | document_tables | Một bảng HTML | Metadata và cấu trúc bảng |
| 10 | document_table_cells | Một source cell | Toàn bộ giá trị và vị trí của cell |

### 7. sections 

| | | | | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| section_id | document_id | filing_id | parent_section_id | section_kind | section_code | section_title | section_order | level | source_xpath |

### 8. content_blocks

| | | | | | | | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| block_id | document_id | section_id | parent_block_id | block_order | block_type | text | normalized_text | language | depth | attributes_json | source_xpath |

**block_type** có thể là:

| | | | | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| paragraph | list_item | heading | table_ref | image_ref | signature_line | page_break | other |

**Đây là nguồn tổng quát để sau này:** 

> Ghép thành section text
> Tạo RAG chunks
> Extract entities/relations
> Phần tích risk factors
> So sánh nội dung giữa các năm

### 9. document_tables

| | | | | | | | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| table_id | document_id | section_id | block_id | table_order | table_kind | caption | title | row_count | classification_confidence | source_xpath |

**table_kind** 

| | | | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- |
| financial_statement | note_schedule | cover_metadata | table_of_contents | exhibit_index | layout | other |

> Table 3 trong 8-K thực chất là bảng dùng để trình bày heading của Item 5.02, không phải bảng dữ liệu.

### 10. document_table_cells

| | | | | | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cell_id | table_id | row_index | column_index | row_span | column_span | is_header | cell_text | normalized_text | source_xpath |

## Nhóm bảng XBRL instance - 6 bảng

> XBRL không chỉ có `metric + value`. Một fact cần concept, context và unit nếu là numeric. Context chứa entity, period và dimensions -> đây là cấu trúc cốt lõi của XBRL.

| | | | |
| --- | --- | --- | --- |
| # | Delta table | Grain | Nội dung |
| 11 | xbrl_instances | Một logical XBRL instance | Quản lý instance/target độc lập |
| 12 | xrbl_facts | một fact occurrence | Giá trị tài chính hoặc text được tag |
| 13 | xbrl_contexts | Một context trong instance | Entity và reporting period |
| 14 | xbrl_dimensions | Một dimension-member trong context | Segment, product, geography, debt type, ... |
| 15 | xbrl_units | Một unit trong instance | USD, shares, USD/share, ... |
| 16 | xbrl_fact_footnotes | Một fact-footnote relation | Footnote gắn trực tiếp vào fact |

### 11. xbrl_instances

| | | | | | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| instance_id | filing_id | target_name | is_inline | source_document_ids | schema_refs | namespace_map | fact_count | context_count | unit_count | taxonomy_version |

> Một submission có thể chứa nhiều logical XBRL instance hoặc target khác nhau. Context ID và unit ID chỉ unique bên trong instance.

### 12. xbrl_facts

| | | | | | | | | | | | | | | | | | | | | | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fact_id | instance_id | filing_id | document_id | concept_id | context_key | unit_key | section_id | block_id | cell_id | source_fact_id | fact_kind | raw_value | normalized_value | numeric_value | is_nil | is_hidden | decimals | precision | scale | sign | format | xml_language | continuation_refs | source_xpath | fact_order |

> 10-K có hơn 1000 fact nhưng không tạo hơn 1000 columns
> `unit_key` nullable vì 8-K có 40 nonnumeric facts và `unit_count=0`
> Fact nằm trong table cell thì liên kết bằng `cell_id`
> Fact nằm trong paragraph thì liên kết bằng `block_id`
>`ix:continuation` phải được nối thành một logical value trước khi ghi 


### 13. xbrl_contexts

| | | | | | | | | | | | | 
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|context_key | instance_id | context_ref | entity_identifier_scheme | entity_identifier | period_type | start_date | end_date | instant_date | segment_xml | scenario_xml | context_hash |

### 14. xbrl_dimensions

| | | | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dimension_id | context_key | context_element | dimension_concept_id | member_type | member_concept_id | typed_member_value | dimension_order |

### 15. xrl_units

| | | | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- |
| unit_key | instance_id | unit_ref | unit_kind | numerator_measures | denominator_measures | normalized expression |

### 16. xbrl_fact_footnotes

| | | | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- |
| footenote_relation_id | fact_id | footnote_text | role_uri | arcole_uri | language | source_xpath |

## Nhóm XBRL taxonomy và linkbase

| | | | |
| --- | --- | --- | --- |
| # | Delta table | Grain | Nội dung |
| 17 | xbrl_concepts | Một expaned QName | Định nghĩa metric/axis/member |
| 18 | xrbrl_labels | Một label của concept | Human-readable tables | 
| 19 | xbrl_networks | Một linkbase network | Một report/statement/disclosure network | 
| 20 | xbrl_relationships | Một acc trong network | Quan hệ presentation/calculation/definition |

### 17. xbrl_concepts

| | | | | | | | | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| concep_id | namespace_uri | local_name | qname | taxonomy_version | data_type | subsitution_group | period_type | balance | is_abstract | is_nillable | is_extension | source_document_id |

`concept_id` nên dựa trên: `namespace_uri + local_name`

### 18. xbrl_labels

| | | | | | |
| --- | --- | --- | --- | --- | --- |
| label_id | concept_id | label_role | language | label_text | source_document_id |

Một concept có thể có:

- Standard label
- Terse label
- Total label
- Negated label
- Period-start/period-end label

### 19. xbrl_networks

| | | | | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| network_id | filing_id | linkbase_type | role_uri | role_definition | report_order | short_name | long_name | source_document_id |

### 20. xbrl_relationships

| | | | | | | | | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| relationship_id | network_id | source_concept_id | target_concept_id | arcrole_uri | relationship_weight | preferred_label_role | target_role_uri | is_closed | is_usable | context_element | source_document_id |

Bảng này cover:
- Presentation tree
- Calculation relationships
- Definition/dimensional relationships
- Hypercube -> dimension -> domain -> member

> Arelle phù hợp hơn tự parse XBRL bằng regex/lxml vì nó đã mô hình hóa fact, Inline fact, context, dimension, unit và relationship sets.