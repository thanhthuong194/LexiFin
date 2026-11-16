# GraphRAG cho LexiFin: thảo luận và kế hoạch (13/11/2026)

Trạng thái: thảo luận thiết kế, chưa có code. Giả định: tài nguyên LLM/GPU không phải giới hạn (thuê GPU hoặc dùng GPU công ty), ưu tiên chất lượng cao nhất. Các con số quy mô lấy từ `sec_filings_silver_stats.md` (Silver 500 công ty); các con số khác là ước tính, chưa đo.

Liên quan: native RAG (chunking) ở `docs/plans/native_rag_chunking_draft.md` và mục B4 của `silver_improvements_report.html`.

## 1. GraphRAG là gì, khác native RAG ở đâu

**Native RAG** cắt báo cáo thành các đoạn nhỏ (chunk). Khi có câu hỏi, hệ thống tìm vài đoạn giống câu hỏi nhất rồi đưa cho LLM đọc và trả lời. Cách này làm tốt khi câu trả lời nằm gọn trong 1–2 đoạn:

- "Doanh thu AWS của Amazon quý 1/2026 là bao nhiêu?": tìm đúng bảng doanh thu.
- "Apple nêu rủi ro gì về chuỗi cung ứng?": tìm đúng đoạn trong Risk Factors.

Native RAG làm kém hai loại câu hỏi:

| Loại câu hỏi | Ví dụ | Vì sao native RAG vướng |
|---|---|---|
| Nối nhiều bước (multi-hop) | "Những công ty nào cùng phụ thuộc vào TSMC?" | Mỗi công ty nhắc TSMC ở một báo cáo riêng. Top-8 chỉ lấy được vài đoạn, không nối được hết |
| Tổng hợp toàn cục | "Rủi ro phổ biến nhất của nhóm ngân hàng là gì?" | Câu trả lời rải trên hàng trăm báo cáo; không có đoạn nào "giống" câu hỏi |

**Đồ thị** gồm các thực thể (node) và các quan hệ giữa chúng (cạnh):

```
(Apple) ──[nhà cung cấp là]──▶ (TSMC) ◀──[nhà cung cấp là]── (NVIDIA)
(Apple) ──[CEO là]──▶ (Tim Cook)
(Apple) ──[nêu rủi ro]──▶ (Phụ thuộc chuỗi cung ứng châu Á)
```

Câu hỏi "công ty nào phụ thuộc TSMC" khi đó chỉ cần đi theo các cạnh từ node TSMC.

**GraphRAG** là RAG dùng thêm đồ thị, gồm ba việc:
1. Dựng đồ thị thực thể và quan hệ từ dữ liệu.
2. Khi có câu hỏi, tìm trên đồ thị: đi theo quan hệ, hoặc lấy cụm thực thể liên quan.
3. Đưa kết quả kèm các đoạn văn bằng chứng cho LLM trả lời.

Microsoft GraphRAG làm thêm một bước: chia đồ thị thành các cụm (community) và nhờ LLM viết tóm tắt từng cụm, để trả lời câu hỏi tổng hợp.

GraphRAG **không thay thế** native RAG. Hai cách bổ sung cho nhau:

| Câu hỏi | Dùng |
|---|---|
| "Doanh thu AWS quý 1/2026?" | Native RAG hoặc tra SQL trên XBRL |
| "Apple nói gì về rủi ro tỷ giá?" | Native RAG |
| "Công ty nào cùng phụ thuộc TSMC?" | GraphRAG |
| "Rủi ro phổ biến của nhóm ngân hàng?" | GraphRAG (tóm tắt nhóm) |
| "Ai thay CFO của Intel năm 2024, trước đó người đó làm ở đâu?" | GraphRAG (đi qua nhiều bước, có yếu tố thời gian) |

## 2. Quy mô thật của data (500 công ty)

| | Số lượng | Ghi chú |
|---|---|---|
| Filing | 40.819 | 76% là 8-K, thường 1–5 trang. 10-K (khoảng 100 trang) chỉ có 2.445 |
| Văn bản có nội dung chữ | ~66.000 | 40.819 văn bản chính + 25.373 thông cáo EX-99 |
| Chữ | ~2,6 tỷ ký tự ≈ 0,8 tỷ token | |
| Chunk (native RAG) | ~2,4 triệu | |
| XBRL | 18,7 triệu fact, 38 triệu quan hệ chỉ tiêu, 9,4 triệu thành viên dimension | |

"Vài trăm nghìn tài liệu" (1,6 triệu `<DOCUMENT>`) là tính cả ảnh, file XBRL, các trang R do SEC sinh. Phần chữ cần đọc là khoảng 66 nghìn văn bản.

- **Vector DB:** 2,4 triệu vector là quy mô vừa với Qdrant (int8 khoảng 2,6 GB RAM). Cái khó là **thời gian embed**: khoảng 85–130 giờ trên GTX 1650; GPU thuê nhanh hơn nhiều lần.
- **GraphRAG:** trích thực thể bằng LLM tốn hơn embed rất nhiều. Thiết kế dựa vào ba lợi thế của data:
  - một phần lớn đồ thị đã có sẵn trong XBRL, không cần LLM;
  - biết chính xác thông tin nằm ở section nào (section rules v3);
  - 41% đoạn dài là boilerplate lặp lại, nên mỗi đoạn chỉ cần trích một lần.

## 3. Mục tiêu

GraphRAG bổ sung bốn loại câu hỏi:

| Loại | Ví dụ |
|---|---|
| Quan hệ, nhiều bước | "Những công ty nào trong S&P 500 cùng phụ thuộc TSMC?" |
| Tổng hợp toàn cục | "Rủi ro được nêu nhiều nhất trong ngành ngân hàng năm 2024?" |
| Theo thời gian | "Ai là CFO của Intel năm 2023? Ai thay người đó?" |
| So sánh có cấu trúc | "So sánh tỷ trọng doanh thu Trung Quốc của Apple, NVIDIA, Qualcomm qua các năm" |

## 4. Nguyên tắc chất lượng

1. Cái gì đã có dạng cấu trúc thì lấy tất định, không qua LLM: XBRL, header SEC, mã item 8-K.
2. Mọi quan hệ do LLM trích phải kèm **câu trích nguyên văn**, và code kiểm tra câu đó có thật trong chunk. Không có hoặc không khớp thì loại.
3. Schema và danh mục cố định; LLM không được tự đặt loại quan hệ hay tên rủi ro.
4. Lưu **từng khẳng định gắn với filing và thời gian**, không chỉ cạnh cuối cùng. "Theo 10-K FY2024, Apple nói X" khác "Apple luôn X".
5. Gold (Iceberg) là nguồn gốc, Neo4j chỉ là index nạp lại được, giống quan hệ Gold và Qdrant.
6. Đo bằng số: bộ chuẩn chấm tay các cạnh, và bộ câu hỏi so sánh native RAG, GraphRAG và kết hợp.

## 5. Lớp 1: đồ thị tất định từ Silver (cả 500 công ty, không LLM)

| Node / quan hệ | Ví dụ | Nguồn Silver |
|---|---|---|
| Company, Filing, Document, Section | Apple → nộp → 10-K FY2025 → gồm → Item 1A | `filings`, `filing_entities`, `documents`, `sections` |
| Phân khúc sản phẩm / kinh doanh, có doanh thu theo năm | Apple → iPhone: doanh thu FY2025 | `xbrl_dimensions` + `xbrl_facts` |
| Thị trường địa lý, có doanh thu | Apple → Greater China | như trên |
| Khách hàng tập trung | Công ty X → khách hàng lớn chiếm 12% doanh thu | fact `ConcentrationRiskPercentage` + trục khách hàng |
| Chỉ tiêu tài chính chính, khoảng 30–50 chỉ tiêu chuẩn theo năm | Revenue, NetIncome, Assets, R&D | `xbrl_facts`, `xbrl_contexts` |
| Cấu trúc chỉ tiêu, gộp trùng giữa các filing | Revenue = Products + Services | `xbrl_relationships` (38 triệu → ước tính 1–3 triệu cạnh) |
| Sự kiện 8-K theo mã item | 8-K → Item 5.02 (thay đổi lãnh đạo), 2.01 (hoàn tất M&A) | `sections` (mã item) |
| Dẫn chiếu giữa các filing | 10-K → dẫn chiếu → hợp đồng tín dụng trong 8-K cũ | `document_links`, `exhibits` |
| Công ty đồng nộp | Exelon ↔ Constellation | `filing_entities` |

- 18,7 triệu fact không đưa hết vào đồ thị. Câu hỏi số liệu chi tiết đi qua công cụ SQL trên Iceberg, nhờ đó đồ thị gọn và nhanh.
- **Đề xuất bổ sung data: EX-21** (danh sách công ty con, bắt buộc kèm 10-K). File có trong Bronze nhưng Silver chưa parse text. Parse EX-21 cho ra đồ thị công ty mẹ–con chính xác tuyệt đối, thay vì để LLM đoán. Việc này cần mở lại Silver đang đóng băng.

## 6. Lớp 2: LLM trích thực thể và quan hệ

### 6.1 Phạm vi

Đọc mọi chunk có nội dung thực:
- 10-K: Item 1, 1A, 3, 7, 7A, 8 (thuyết minh: M&A, nợ, kiện tụng, cam kết);
- 10-Q: MD&A, Legal Proceedings, Risk Factors;
- toàn bộ item 8-K và thông cáo EX-99.

Chỉ bỏ những phần không có thông tin quan hệ: trang bìa, mục lục, chữ ký, danh mục exhibit (đã có ở lớp 1), đoạn kiểm soát nội bộ mang tính thủ tục.

Gộp trùng theo `text_sha256` trước khi trích. Mục đích chính là **nhất quán**: cùng một đoạn văn luôn ra cùng một kết quả, và khẳng định được gắn vào mọi filing chứa đoạn đó. Khối lượng ước tính khoảng 0,5–0,6 tỷ token sau gộp trùng; với 3 lượt xử lý thành khoảng 1,5–2 tỷ token.

### 6.2 Schema cố định

| Node | Định danh chuẩn |
|---|---|
| Company | CIK cho công ty đã đăng ký với SEC; ID nội bộ cho công ty tư nhân hoặc nước ngoài |
| Person | tên + công ty + chức vụ |
| Product / Segment | gắn với công ty; ưu tiên khớp với phân khúc XBRL ở lớp 1 |
| Geography | danh mục quốc gia / vùng chuẩn |
| RiskCategory | danh mục khoảng 40–60 loại: dựng một lần từ các tiêu đề risk factor (LLM gom nhóm, người duyệt), rồi cố định |
| Event | loại sự kiện bám theo mã item 8-K, thêm một số loại bổ sung |
| Agreement | hợp đồng trọng yếu: tín dụng, sáp nhập, cung ứng |

Quan hệ: `SUPPLIER_OF`, `CUSTOMER_OF`, `COMPETES_WITH`, `PARTNER_OF`, `SUBSIDIARY_OF`, `ACQUIRED` / `DIVESTED`, `OFFICER_OF` / `DIRECTOR_OF` (kèm chức vụ), `OPERATES_IN`, `EXPOSED_TO` (rủi ro), `PARTY_TO` (kiện tụng, hợp đồng), `LENDER_TO`.

### 6.3 Quy trình trích, 3 lượt

1. **Trích xuất.** Dùng model mạnh nhất: open-weights loại Qwen3-235B-A22B chạy vLLM trên GPU thuê, hoặc model frontier qua API. Đầu vào gồm chunk, các chunk lân cận (parent), và metadata (công ty, kỳ, section), để "we" / "the Company" được gán đúng về CIK của công ty nộp. Đầu ra là JSON theo schema (guided decoding của vLLM). Mỗi quan hệ kèm câu trích nguyên văn.
2. **Gleaning.** Hỏi lại model "còn sót thực thể hay quan hệ nào không?" (kỹ thuật của Microsoft GraphRAG), để tăng recall.
3. **Kiểm chứng.**
   - Code kiểm tra câu trích có xuất hiện nguyên văn trong chunk; không có thì loại.
   - Một model khác kiểm tra câu trích có thực sự nói lên quan hệ đó không (kiểm tra suy diễn), rồi cho điểm tin cậy. Hai model độc lập ít khi cùng sai một chỗ.

Prompt, model và tham số được cố định và gắn phiên bản (`extractor_version`), giống `rules_version` của Silver.

### 6.4 Hợp nhất thực thể (khâu quyết định chất lượng)

Ví dụ: "TSMC", "Taiwan Semiconductor Manufacturing Company Limited" và "Taiwan Semiconductor" phải là một node.

| Tầng | Cách làm |
|---|---|
| 1. Tất định | Công ty trong danh sách 500: khớp về CIK theo tên, ticker và tên cũ trong `filing_entities` (Facebook → Meta). Công ty Mỹ niêm yết khác: khớp với danh sách CIK/ticker công khai của SEC (một nguồn Bronze nhỏ mới) |
| 2. Gom ứng viên | Chuẩn hóa tên (bỏ Inc./Ltd./Corp.), so độ giống bằng embedding để gom các cặp có thể trùng |
| 3. LLM phân xử | Model mạnh quyết định từng cặp có cùng thực thể không, dựa trên ngữ cảnh (ngành, quốc gia) |
| 4. Người duyệt | Duyệt tay các lần gộp ảnh hưởng nhiều cạnh nhất (top vài trăm) |

Mọi tên gốc được giữ trong bảng alias, nên gộp sai thì tách lại được.

### 6.5 Thời gian

- Gold lưu từng **khẳng định (claim)**: chủ thể, quan hệ, đối tượng, `chunk_id`, câu trích, filing, năm tài chính, độ tin cậy, phiên bản extractor.
- **Cạnh tổng hợp** (nạp vào Neo4j) có `first_seen`, `last_seen`, số filing ủng hộ. Ví dụ: Apple `SUPPLIER_OF` TSMC, xuất hiện FY2021–FY2025 trong 5 báo cáo.
- Chức vụ: Person `OFFICER_OF` (CFO) có khoảng thời gian lấy từ 8-K 5.02 và chữ ký 10-K/10-Q, nên trả lời được "ai là CFO năm 2023".

## 7. Lớp 3: tóm tắt cho câu hỏi tổng hợp

- **Nhóm tự nhiên:** ngành × loại rủi ro hoặc sự kiện × năm (ví dụ "Ngân hàng × rủi ro lãi suất × 2024"). Model mạnh viết tóm tắt từ các claim và bằng chứng của nhóm, có trích dẫn, rồi kiểm chứng lại như lượt 3. Số nhóm có hạn (vài nghìn) và dễ giải thích.
- **Cụm Leiden** (cách của Microsoft GraphRAG) làm song song để so sánh. Leiden có thể phát hiện cụm không theo ngành, ví dụ chuỗi cung ứng bán dẫn bao gồm cả Apple.
- **Ngành:** SIC có sẵn trong Silver; nên thêm GICS từ dữ liệu iShares (bảng `companies` ở Gold) vì phân ngành này quen thuộc hơn.

## 8. Trả lời câu hỏi

```
Câu hỏi
 → nhận diện thực thể và thời gian ("Apple" → CIK, "2023" → fiscal year)
 → bộ định tuyến (LLM) chọn công cụ:
     tra cứu đoạn văn       → native RAG (Qdrant)
     số liệu chính xác      → SQL trên XBRL (Iceberg/DuckDB)
     quan hệ, nhiều bước    → Cypher trên Neo4j (LLM viết, có schema + ví dụ mẫu,
                              chạy EXPLAIN kiểm tra trước, sai thì dùng mẫu truy vấn có sẵn)
     tổng hợp toàn cục      → tóm tắt nhóm (lớp 3)
 → gom chunk bằng chứng qua chunk_id → LLM trả lời, bắt buộc trích dẫn
```

GraphRAG dùng lại chunk của native RAG làm đầu vào và làm bằng chứng, nên chunk phải có trước. Trích đồ thị chỉ cần chunk, không cần chờ embed xong, nên hai việc chạy song song được.

## 9. Lưu trữ

### 9.1 Gold (Iceberg, nguồn gốc): `data/gold/graph_rag/`

| Bảng | Nội dung |
|---|---|
| `company_segments`, `company_metrics`, `concept_edges`, `filing_events`, `filing_references` | Lớp 1 |
| `entities`, `entity_aliases` | Thực thể chuẩn và các tên gốc |
| `mentions` | Thực thể xuất hiện ở chunk nào, vị trí nào |
| `claims` | Từng khẳng định kèm câu trích, filing, thời gian, độ tin cậy, phiên bản |
| `edges` | Cạnh tổng hợp từ claims (`first_seen` / `last_seen`, số filing ủng hộ) |
| `risk_taxonomy` | Danh mục rủi ro cố định |
| `group_summaries`, `communities` | Lớp 3 |
| `eval_triples`, `eval_questions` | Bộ chuẩn chấm tay và bộ câu hỏi |

Neo4j chỉ nạp những gì cần cho truy vấn: entities, edges, một ít bằng chứng tiêu biểu, tóm tắt. Bảng `claims` đầy đủ ở lại Iceberg.

### 9.2 Neo4j hay NebulaGraph

Quy mô ước tính trong đồ thị phục vụ: khoảng 2–5 triệu node và 10–30 triệu cạnh (claims ở Iceberg).

| Tiêu chí | Neo4j (Community) | NebulaGraph |
|---|---|---|
| Quy mô phù hợp | Một máy, tới hàng trăm triệu cạnh | Cụm nhiều máy, hàng tỷ cạnh trở lên |
| Vận hành | 1 container | Tối thiểu 3 dịch vụ (graphd, metad, storaged) |
| Ngôn ngữ truy vấn | Cypher, phổ biến nhất; LLM viết Cypher tốt hơn hẳn nGQL (quan trọng cho text-to-Cypher) | nGQL, hỗ trợ openCypher một phần |
| Hệ sinh thái GraphRAG | Thư viện `neo4j-graphrag` chính chủ; LangChain và LlamaIndex tích hợp sẵn; có vector index | Ít hơn |
| Thuật toán đồ thị | GDS (Leiden, PageRank), bản miễn phí giới hạn 4 lõi | nebula-algorithm chạy qua Spark |
| Giấy phép | GPLv3, không có cluster (đủ cho dự án này) | Apache 2.0 |
| Trong repo | Đã có trong `docker-compose.yml` | Chưa có |

**Chọn Neo4j.** Quy mô LexiFin nhỏ hơn ngưỡng cần NebulaGraph hàng chục lần. NebulaGraph chỉ đáng chọn khi có hàng tỷ cạnh, cần cụm nhiều máy, hoặc ghi liên tục với tốc độ rất cao.

Lưu ý phần cứng:
- Đặt dữ liệu Neo4j và Qdrant trên ổ ext4/SSD trong máy, **không đặt trên ổ USB exFAT**: exFAT không có phân quyền và khóa file kiểu Linux, và ổ USB HDD chậm cho truy cập ngẫu nhiên.
- RAM 15 GB của homepc phải chia cho Neo4j (heap 2–4 GB + page cache 4–6 GB) và Qdrant (~3–4 GB). Nếu không đủ thì đặt Neo4j trên máy GPU thuê hoặc máy công ty khi demo.
- `docker-compose.yml` đang hardcode mật khẩu Neo4j và dùng image `latest`; cần sửa trước khi dùng thật.

## 10. Đánh giá

| Đo cái gì | Cách đo |
|---|---|
| Chất lượng trích xuất | Bộ chuẩn khoảng 500 cạnh chấm tay (lấy mẫu phân tầng theo loại quan hệ): precision, recall, tỷ lệ câu trích không khớp |
| Hợp nhất thực thể | Khoảng 300 cặp chấm tay: tỷ lệ gộp nhầm và gộp sót |
| Trả lời câu hỏi | Bộ khoảng 300 câu theo 4 loại ở mục 3, mở rộng từ bộ của native RAG. So native RAG / GraphRAG / kết hợp: độ đúng, tỷ lệ trích dẫn đúng, faithfulness (RAGAS + chấm tay mẫu) |
| Hệ thống | Độ trễ, chi phí trích xuất, thời gian dựng lại Neo4j từ Gold |

Ngưỡng để chuyển giai đoạn: precision cạnh ≥ 90% và tỷ lệ gộp nhầm thực thể ≤ 2% trên bộ chuẩn.

## 11. Lộ trình

| Giai đoạn | Việc | Phụ thuộc |
|---|---|---|
| 0 | Native RAG bước 1 (chunks) | Silver, đã xong |
| 1 | Lớp 1 cho 500 công ty; (tùy chọn) parse EX-21 | Silver; chạy song song với 0 |
| 2 | Danh mục rủi ro và schema; prompt; pilot lớp 2 trên 5 công ty; chấm bộ chuẩn và tinh chỉnh | chunks |
| 3 | Lát 50 công ty (2–3 ngành có chuỗi cung ứng đan xen: bán dẫn, phần cứng, ngân hàng); entity resolution đầy đủ; đo câu hỏi nhiều bước | giai đoạn 2 đạt ngưỡng |
| 4 | Toàn bộ 500 công ty trên GPU thuê | giai đoạn 3 đạt ngưỡng |
| 5 | Lớp 3 tóm tắt; bộ định tuyến và truy vấn; đánh giá so sánh | giai đoạn 4 + native RAG |

## 11b. Bottleneck phần cứng và LLM (ước tính, 500 công ty)

Khâu đắt nhất là lớp 2. Thời gian nằm chủ yếu ở số token LLM sinh ra (decode), không ở số token đọc vào.

| Khâu | Khối lượng | Cách tính |
|---|---|---|
| Chọn chunk | 2,5 tr → ~0,85 tr | bỏ 1,1 tr chunk bảng (số liệu đã có ở lớp 1); giữ section cần đọc; gộp đoạn trùng |
| Lượt 1: trích | vào ~1 tỉ, ra ~0,35 tỉ token | mỗi chunk ~1.100 token mới (chunk + lân cận + metadata; prompt được cache), ~400 token JSON |
| Lượt 2: bổ sung | ra ~0,1 tỉ | ~100 token mỗi chunk |
| Lượt 3: kiểm chứng | ~3–4 tr quan hệ | ~200 token vào, ~20 ra mỗi quan hệ; model khác, nhỏ hơn được |
| Hợp nhất thực thể | ~50–200 nghìn cặp | chỉ các cặp embedding thấy giống nhau mới gửi LLM |
| Lớp 3: tóm tắt | vào ~0,1–0,2 tỉ | vài nghìn nhóm × 20–50 nghìn token |

| Phần cứng | Model | Lượt 1+2 | Dùng cho |
|---|---|---|---|
| Laptop (không GPU NVIDIA) | — | — | code, chunk, test |
| homepc GTX 1650 4 GB | ≤ 4B nén 4-bit, ~20–30 token/s | 6–12 tháng | embedding native RAG (85–130 h), Qdrant, Neo4j |
| 1 × 24 GB (L4 / 4090) | Qwen3-8B, ~1–2 nghìn token/s | 3–5 ngày | kiểm chứng |
| 1 × H100 80 GB | Qwen3-32B FP8, ~2–3 nghìn token/s | 2–3 ngày | pilot, lát 50 công ty |
| 8 × H100 80 GB | Qwen3-235B-A22B FP8 (~240 GB trọng số), ~5–8 nghìn token/s | 1–2 ngày (~200–400 GPU-giờ, ~400–1.200 USD ở 2–3 USD/GPU-giờ) | chạy chính |

Vai trò LLM:
- Trích: Qwen3-235B-A22B tự host bằng vLLM (cố định phiên bản, ép JSON theo schema, tính tiền theo giờ GPU).
- Kiểm chứng: model khác họ hoặc frontier API.
- Danh mục rủi ro, phân xử thực thể khó, tóm tắt: frontier API.
- Trả lời câu hỏi: Gemini.

Bottleneck khác:
- Neo4j cần ~8–12 GB RAM (homepc có 15 GB, dùng chung với Qdrant): nâng lên 32 GB hoặc dùng máy riêng; lưu trên SSD ext4.
- Embedding native RAG trên GTX 1650 mất 85–130 h; thuê 1 GPU 24 GB còn ~10–20 h.
- Đọc Silver trên exFAT ~34 MB/s.
- Hợp nhất thực thể: nút thắt là người duyệt.

Cách giảm khối lượng: bỏ chunk bảng, gộp đoạn trùng, prefix caching, gom chunk cùng section, FP8, guided decoding. Pilot 5 công ty (~1% khối lượng, ~1 giờ trên 1×H100) để đo throughput thật trước khi thuê 8×H100.

## 12. Câu hỏi còn mở

1. Có parse EX-21 (công ty con) vào Silver không, dù phải mở lại data đang đóng băng.
2. Có thêm danh sách CIK/ticker công khai của SEC và GICS từ iShares làm nguồn phụ không.
3. Báo cáo HTML cho thầy (`docs/reports/graphrag_plan.html`, cùng phong cách các báo cáo trước): thực trạng data → vì sao cần GraphRAG → thiết kế 3 lớp → lưu trữ → đánh giá → lộ trình → kết quả dự kiến (ước tính).
