# Native RAG: kết quả (24/11/2026)

Trạng thái: chunk, embedding, Qdrant, retrieval, sinh câu trả lời và eval đã có code (commit `e59baad` → `ca091ed`, 16–24/11/2026). Kết quả đo trên `data/` (5 công ty). Ở 500 công ty mới xong chunk và embedding; Qdrant đang nạp.

Nguồn số liệu: `evals/runs/<run>/report.md` (gitignored, máy local), `docs/data/native_rag_model.md`, `AGENTS.md`, `evals/sets/v1*/`. Dự đoán trước đó: mục B2, B3, B4 của `silver_improvements_report.html` (07–13/11/2026, "ước tính, chưa đo"). Số đánh dấu "ước tính" là suy ra, không phải đo.

## 1. Tóm tắt

- Retrieval trên bộ eval v1.2 (97 câu × en/vi, 5 công ty): post `hit_rate@8` en 0,976, vi 0,976; `mrr@10` en 0,682, vi 0,718; còn 4/166 dòng lỗi (run `20261124-100952_batch2_translate`).
- Cải thiện lớn nhất không đến từ mô hình mà từ việc sửa lỗi im lặng: gpt-oss trả câu tìm kiếm tiếng Việt cho 77/81 câu hỏi tiếng Việt, nên chunk tiếng Anh bị tìm bằng tiếng Việt. Sửa schema + dịch dự phòng: vi `hit_rate@8` 0,880 → 0,976 (khác biệt có ý nghĩa, paired t-test p < 0,05).
- Reranker là thành phần đáng giá nhất: tắt reranker làm vi `hit_rate@8` 0,843 → 0,651, `mrr@10` (all) 0,646 → 0,544 (bộ v1).
- Một phần "lỗi retrieval" là lỗi của bộ eval: bằng chứng chỉ gắn một filing trong khi cùng dòng bảng có ở filing khác (10-Q/10-K lặp bảng của thông cáo). Bộ v1.2 mở rộng bằng chứng theo luật; chấm lại cùng run: lỗi 15 → 13.
- Sinh câu trả lời: thử 10 câu × 2 ngôn ngữ: đúng giá trị 6/6, từ chối đúng 2/2, từ chối nhầm 0/18, có trích dẫn 18/18, trích dẫn chứa bằng chứng 17/18, đúng ngôn ngữ 20/20. Chưa có judge (key NVIDIA bị 401).
- 500 công ty: 3.737.715 chunk, 3.737.715 vector (Qwen3-Embedding-0.6B, 1024 chiều), audit embedding toàn PASS. Số chunk cao hơn dự đoán ~2,5 tr khoảng 50%.
- Chưa đo: chất lượng ở 500 công ty, độ chính xác câu trả lời bằng judge, faithfulness, baseline chunk cố định, thí nghiệm chunking A–E.

## 2. Pipeline và cấu hình

```
Silver → native_rag.chunks (Gold, Iceberg) → native_rag.embeddings → Qdrant (dense + BM25)
câu hỏi → query model → danh bạ công ty → hybrid (RRF) → gộp trùng → rerank → recency
        → quota theo công ty/năm → mở rộng ngữ cảnh → generator trả lời có [n]
```

### 2.1 Mô hình (`config/llm_profiles.toml`)

| Vai trò | Mô hình | Chạy ở đâu | Ghi chú |
|---|---|---|---|
| `chat.query` | gpt-oss-120b, `effort = low` | Cerebras (từ 24/11) | Trước đó Groq free, 200k token/ngày/key: hết quota giữa eval |
| `chat.generator` | gpt-oss-120b, `temperature 0` | Cerebras | Key riêng với query |
| `chat.judge` | Mistral Large (`mistral-large-4-1t-a52b`) | NVIDIA API | Key bị 401; đang chọn judge khác |
| `embedding.default` | Qwen3-Embedding-0.6B, 1024 chiều | vLLM Kaggle 2×T4, Cloudflare tunnel | Cùng tokenizer dùng để giới hạn chunk |
| `rerank.default` | Qwen3-Reranker-0.6B, `prompt_format = "qwen3"` | vLLM Kaggle | Không có chat template thì điểm gần ngẫu nhiên (0,32 vs 0,26; có template: 0,995 vs 0,0006) |

Claude viết câu hỏi eval, nên không bao giờ là mô hình trả lời hay judge.

### 2.2 Tham số

| Phần | Giá trị | Nguồn |
|---|---|---|
| Chunk | Theo cấu trúc, không LLM; trần 512 token tính cả tiền tố; bảng dựng từ ô, tách theo hàng, lặp header | `native_rag_model.md` §3 |
| Tiền tố | `Company \| form \| period \| section › subheading [\| Table: title (part i/n)]` | §2 |
| Rules version chunk | v8 (v6 → v7 → v8 sửa lỗi bảng ghi chú, 18–20/11) | git log |
| Qdrant | HNSW m 16, ef_construct 200, `hnsw_ef` 128; `--large`: int8 trong RAM, rescoring, oversampling 2,0 | §6 |
| BM25 | Sparse, từ thường + số, bỏ stopword tiếng Anh, IDF do Qdrant | §6 |
| `RetrieverSettings` | `prefetch` 100 (mỗi nhánh), `candidates` 50, `top_k` 8, `context_tokens` 6000, `min_results` 8, `neighbor_window` 1, `small_section_chunks` 4, `recency_weight` 0,05, `latest_weight` 0,25, `per_company` 2, `max_budget_factor` 2,0, `rerank_cooldown_s` 120 | `retriever.py` |
| Lọc cứng | `cik`, form, kỳ (fiscal hoặc calendar, cửa sổ 8-K theo ngày nộp); dưới 8 ứng viên thì nới kỳ, rồi form, không bao giờ nới `cik` | §7 |

## 3. Dữ liệu và chi phí build

| | 5 công ty (`data/`) | 500 công ty (homepc) |
|---|---|---|
| Filing | 343 | 40.819 |
| Văn bản có chunk | 523 | 65.999 |
| Chunk | 25.928 (v6; v7, v8 giống hệt) | 3.737.715 (v8) |
| Build chunk | ~85 s | 3 h 13 phút (v7); RSS đỉnh 3,6 GB (v6) |
| Embedding | Đủ cho chunk v6 (= v8) | 3.737.715 vector, xong 24/11 16:11; audit PASS, re-embed mẫu cosine ≥ 0,999 |
| Qdrant | `sec_filings__chunks_v8__index_v1__qwen_qwen3_embedding_0_6b` | Đang chạy (bắt đầu 24/11 18:20, Qdrant 1.20.2, ~72k point/phút, lưu trên NVMe ext4) |

### 3.1 Dự đoán (B2/B3) và thực tế

| Hạng mục | Dự đoán | Thực tế |
|---|---|---|
| Số chunk 500 công ty | ~2,5 tr (1,4 tr văn bản + 1,1 tr bảng) | 3,74 tr |
| Token cần embed | ~0,85 tỷ | Chưa đo |
| Thời gian embed | ~4 ngày GTX 1650 | Chạy trên Kaggle 2×T4; tổng thời gian chưa ghi lại. Colab T4: 19,6 chunk/s |
| RAM vector Qdrant (int8) | ~2,6 GB | Chưa đo (đang nạp) |
| Dựng index Qdrant | ~1 h | Upsert ước tính ~52 phút (3,74 tr ÷ 72k/phút); HNSW chưa xong |
| Chunk văn bản | 400–512 token, không overlap | Trần 512 tính cả tiền tố; đoạn dài chia cân bằng; câu lặp nhỏ giữa các phần của một đoạn quá dài |
| Chunk bảng | 1 bảng/chunk, tách > 800 token | Tách ở 512 (bỏ trần 800, sửa 16/11) |
| Truy hồi | dense top-50 + BM25 top-50 → RRF → rerank → top-8 | prefetch 100 mỗi nhánh → RRF → 50 ứng viên → rerank → top 8 + quota + mở rộng |
| Sinh câu trả lời | Gemini, 8 chunk (~4k token) | gpt-oss-120b (Cerebras), 8 chunk mở rộng đến 6.000 token |
| Bộ eval | ~300 câu, 150 câu số liệu nhãn tự động từ XBRL | 97 câu × en/vi, Claude viết, bằng chứng kiểm bằng code |
| Baseline chunk cố định 512, chỉ dense; thí nghiệm A–E (B4) | Có | Chưa làm |

## 4. Kết quả retrieval

Bộ câu hỏi: 100 task lấy mẫu theo ô phủ (công ty, form, section, năm, kỳ, loại câu), 3 bị bỏ (`SKIP`), còn 97: 83 có bằng chứng, 14 không trả lời được. Mỗi câu chạy en và vi. Chấm theo nhóm bằng chứng với ranx; `post` = thứ tự giao cho LLM, `pre` = thứ tự RRF trước rerank.

| Phiên bản bộ | Thay đổi |
|---|---|
| v1 (23/11) | 97 câu |
| v1.1 (24/11) | Sửa 19 bản nháp sau review ngoài (Gemini, ChatGPT, Grok đọc mẫu 25 câu) và agent đọc cả 97: lệch câu hỏi/đáp án, số ngoài đoạn, calque tiếng Việt |
| v1.2 (24/11) | Mở rộng bằng chứng dòng bảng cho mọi nhóm: 84 dòng thêm (83 hợp lệ, 1 loại bởi luật < 4 chữ số có nghĩa). Tổng bằng chứng từ filing khác: 125 dòng/20 câu (v1.1) → 207 dòng/30 câu (v1.2) |

### 4.1 Lịch sử run (post, 83 dòng mỗi ngôn ngữ)

| Run | Thay đổi | Bộ | hit@8 en | hit@8 vi | mrr@10 en | mrr@10 vi | recall@8 en / vi | Lỗi |
|---|---|---|---|---|---|---|---|---|
| `20261123-152425_baseline_full` | Baseline | v1 | 0,904 | 0,843 | 0,676 | 0,615 | 0,880 / 0,801 | 21 |
| `20261123-170838_no_rerank` | Tắt reranker | v1 | 0,904 | 0,651 | 0,613 | 0,476 | 0,867 / 0,639 | 37 |
| `20261123-173043_no_llm` | Không query model (luật) | v1 | 0,807 | 0,723 | 0,617 | 0,578 | 0,789 / 0,699 | 39 |
| `20261124-013535_batch1_periods` | Đọc kỳ (fiscal/calendar, nửa năm, 9 tháng, ngày), lọc theo tháng nộp | v1 | 0,952 | 0,867 | 0,672 | 0,621 | 0,928 / 0,831 | 15 |
| `20261124-035410_v1.1_baseline` | Như batch 1 | v1.1 | 0,952 | 0,867 | 0,666 | 0,629 | 0,928 / 0,825 | 15 |
| ↳ chấm lại trên v1.2 | Không chạy lại, chỉ đổi bằng chứng | v1.2 | 0,964 | 0,880 | 0,678 | 0,646 | 0,940 / 0,843 | 13 |
| `20261124-100952_batch2_translate` | Câu tìm kiếm riêng cho từng ngôn ngữ corpus | v1.2 | 0,976 | 0,976 | 0,682 | 0,718 | 0,952 / 0,946 | 4 |

- Kiểm định paired t-test (p < 0,05, `--compare`): batch 2 hơn v1.1 chấm lại ở vi `hit@8` và vi `ndcg@10` (0,689 → 0,765); vi `mrr@10` và en không khác có ý nghĩa. Batch 1 so với baseline: không khác có ý nghĩa. Tắt reranker: kém có ý nghĩa ở vi (hit@8, mrr, ndcg) và en `ndcg@10`.
- Batch 2: `query.search_text` = 1,000 (97/97 câu vi có câu tìm kiếm tiếng Anh; trước đó 4/81). `pre.hit_rate@8` vi 0,711 → 0,952: lợi ích nằm ở tập ứng viên, không chỉ ở rerank.
- Hiểu câu hỏi (batch 2): `query.company` 1,000; `query.unknown_flagged` 1,000 (5 công ty ngoài corpus); `query.filter_admits_evidence` en 0,970, vi 0,982.
- Theo loại (baseline v1): yếu nhất là bảng (`chunk_kind=table` hit@8 en 0,857, vi 0,738; mrr 0,495/0,440) và EX-99 (hit@8 en 0,667, vi 0,333); text đạt 0,951/0,951.

### 4.2 Độ trễ (giây, mean / p95)

| Run | Query model | Embed | Qdrant | Rerank | Tổng |
|---|---|---|---|---|---|
| baseline v1 (en) | 1,50 / 2,47 | 28,79 / 34,95 | 0,12 / 0,29 | 3,05 / 4,08 | 33,51 / 40,03 |
| batch 2 (en) | 7,88 / 56,91 | 0,43 / 0,86 | 0,03 / 0,06 | 2,87 / 3,97 | 11,23 / 61,20 |
| batch 2 (vi) | 10,01 / 56,89 | 0,43 / 0,89 | 0,03 / 0,05 | 2,90 / 4,17 | 13,39 / 61,20 |

- Embed trung bình 20–29 s ở các run 23–24/11 sáng, 0,43 s ở batch 2 (cùng loại server Kaggle qua tunnel); nguyên nhân chưa tách.
- Query model p95 ~57 s ở batch 2 (Cerebras; có thêm lời gọi dịch dự phòng); nguyên nhân chưa tách.
- Không tính query model, batch 2 mất ~3,4 s trung bình, phần lớn là rerank (p95 rerank một mình 4,0–4,2 s). Dự đoán B3 "< 3 s p95" chưa đạt; rerank đang chạy trên T4 qua tunnel.

## 5. Sinh câu trả lời (thử 10 câu)

Code: `src/rag_engine/generation/answer.py` (một lời gọi structured tới `chat.generator`, trích dẫn `[n]`, chuẩn hoá `【n†…】` và khoảng `[1-3]`, bỏ số ngoài phạm vi), `scripts/rag/ask.py`; eval `src/evals/answer_eval.py`, `answer_judge.py`, `numbers.py`.

Run `20261124-105158_answers_try10` (10 câu đầu v1.2 × en/vi; retrieval tìm đủ bằng chứng ở mọi câu có đáp án):

| Chỉ số | en | vi | Tổng |
|---|---|---|---|
| `answer.value` (đúng giá trị XBRL / số đầu tiên của đáp án) | 3/3 | 3/3 | 6/6 |
| `answer.refusal` (từ chối câu không trả lời được) | 1/1 | 1/1 | 2/2 |
| `answer.false_refusal` (thấp là tốt) | 0/9 | 0/9 | 0/18 |
| `answer.cited` | 9/9 | 9/9 | 18/18 |
| `answer.citation_evidence` | 9/9 | 8/9 | 17/18 |
| `answer.language` | 10/10 | 10/10 | 20/20 |
| `judge.correct`, `judge.faithful` | – | – | Chưa đo (401) |

- Dòng lỗi duy nhất: t0006 vi trả đúng số ($14.728 tr so với Q3 FY2022) nhưng đoạn được trích không chứa nhóm bằng chứng.
- 10 câu quá ít để kết luận; chưa đo độ trễ sinh câu trả lời. Dự đoán B3/B4 (đúng ≥ 80–87,5%, faithfulness ≥ 90–92%, trễ p95 < 10 s): chưa đo.

### 5.1 Chỉ số dự đoán và thực tế

| Chỉ số | Dự đoán B3 | Dự đoán B4 | Thực tế (5 công ty, v1.2) |
|---|---|---|---|
| Recall@10 hybrid + rerank | ≥ 0,85 | 0,89 | Không đo đúng chỉ số này. Gần nhất: `post.recall@8` en 0,952, vi 0,946 |
| Recall@10 hybrid không rerank | 0,75 | – | `post.recall@8` en 0,867, vi 0,639 (v1, run no_rerank) |
| Recall@10 dense / BM25 riêng | 0,65 / 0,55 | – | Chưa đo |
| MRR@10 | ≥ 0,70 | 0,775 | en 0,682, vi 0,718 |
| Trả lời đúng | ≥ 80% | 87,5% | 6/6 giá trị số (10 câu); judge chưa chạy |
| Faithfulness | ≥ 90% | 92% | Chưa đo; trích dẫn chứa bằng chứng 17/18 |
| Trễ truy hồi p95 (không LLM) | < 3 s | – | ~4 s chỉ riêng rerank (batch 2) |
| Trễ trả lời p95 | < 10 s | – | Chưa đo; retrieval p95 kể cả query model 61 s |

Lưu ý: dự đoán tính cho 500 công ty và ~300 câu; thực tế là 5 công ty, 97 câu. Kết quả 500 công ty sẽ thấp hơn (nhiều văn bản giống nhau giữa công ty, nhiều kỳ hơn).

## 6. Lỗi gặp phải và cách khắc phục

| # | Vấn đề | Nguyên nhân | Cách sửa | Kết quả |
|---|---|---|---|---|
| 1 | Câu hỏi tiếng Việt kém hẳn tiếng Anh (vi `pre.hit@8` 0,711 vs en 0,964) | gpt-oss trả câu tìm kiếm tiếng Việt cho 77/81 câu vi; chunk tiếng Anh bị tìm bằng tiếng Việt, không báo lỗi | Schema động: mỗi ngôn ngữ corpus một trường bắt buộc; câu thiếu hoặc vẫn tiếng Việt thì gọi dịch lần hai; thêm chỉ số `query.search_text` | vi hit@8 0,880 → 0,976, lỗi 13 → 4 (`ca091ed`) |
| 2 | Lỗi giả: top 8 có dòng đáp án nhưng ở chunk khác | Bằng chứng chỉ gắn một filing; cùng dòng bảng có trong 10-Q/10-K (bản sao bảng của thông cáo, số so sánh năm sau) | v1.2: mở rộng bằng chứng sang dòng cùng nhãn (bỏ số chú thích), cùng giá trị; giá trị < 4 chữ số có nghĩa cần thêm một số khác của dòng; bằng chứng mở rộng không tính cho kiểm tra filter | 84 dòng thêm, agent kiểm: 83 hợp lệ; lỗi 15 → 13 khi chấm lại |
| 3 | Đọc sai kỳ: năm tài chính vs năm dương lịch, nửa năm, 9 tháng, 8-K | Query model trả kỳ không ổn định; cửa sổ 8-K nhận cả filing công ty khác | `checked_plan` bổ sung kỳ từ chữ trong câu; `period_basis`; `filed_months`; cửa sổ 8-K chỉ nhận filing của chính công ty (`42437e9`) | Batch 1: en hit@8 0,904 → 0,952, vi 0,843 → 0,867, lỗi 21 → 15 (không có ý nghĩa thống kê) |
| 4 | Reranker đẩy chunk đúng xuống (t0071, t0086) | Hạn chế của Qwen3-Reranker-0.6B | Không sửa riêng | Hai câu này không còn trong 4 lỗi của batch 2 |
| 5 | Bảng bị tách mất tiêu đề ở phần sau (t0077) | Các phần sau của một bảng tách theo hàng không giữ tiêu đề bảng | Cần đổi luật chunk và build lại; hoãn | Còn lỗi |
| 6 | Dòng `degraded` giữa eval | Groq free hết 200k token/ngày/key | Eval có `--resume` chạy lại dòng lỗi/degraded; chuyển query model sang Cerebras | v1.1 baseline chạy xong bằng Cerebras, 0 dòng degraded |
| 7 | Không gọi được Cerebras | Gõ sai host trong base URL | Sửa base URL | Chạy được |
| 8 | Qdrant không lưu được trên ổ DATA | Ổ exFAT không giữ được storage của Qdrant | Đặt storage trên NVMe ext4 | Đang nạp 500 công ty |
| 9 | Audit chunk 500 công ty FAIL `block_in_one_chunk` | Bảng ghi chú sau bảng bị tách giữa hai chunk (v6); dấu `(ii)` đứng riêng nhận hàng đầu của bảng bố cục (v7, 2 block) | Rules v7 (`dd520d4`), v8 (`cb441db`) | Chunk `data/` giống hệt qua v6–v8; 500 công ty v8: 3.737.715 chunk |
| 10 | Judge không chạy | Key NVIDIA bị từ chối (401) | Chưa sửa; đang chọn mô hình judge | 20/20 dòng chờ chấm lại (`--rejudge`) |

## 7. Đã thử và loại bỏ

| Phương án | Kết quả | Quyết định |
|---|---|---|
| Tắt reranker | v1: vi hit@8 0,843 → 0,651, mrr (all) 0,646 → 0,544, lỗi 21 → 37 | Giữ reranker |
| Bỏ query model, chỉ dùng luật | v1: hit@8 en 0,807, vi 0,723, lỗi 39 | Giữ query model; luật chỉ là dự phòng |
| Trộn điểm reranker với thứ tự RRF | Không tốt hơn (phân tích trên run baseline, không có run riêng) | Loại |
| Nới bộ lọc kỳ/form | Không tốt hơn (phân tích trên run baseline, không có run riêng) | Loại |
| `top_k` 10 | Không tốt hơn (phân tích trên run baseline, không có run riêng) | Giữ 8 |
| Sửa từng câu trong 4 lỗi còn lại | Bộ 97 câu quá nhỏ, dễ overfit | Gom lại, xử lý cùng bộ v2 / run 500 công ty |

## 8. Còn mở

| Việc | Ghi chú |
|---|---|
| Nạp xong Qdrant 500 công ty, audit, chuyển alias | Đang chạy; đo RAM int8, thời gian HNSW, `ann_recall` |
| Eval retrieval trên 500 công ty | Chạy v1.2 trước (bằng chứng có trong Silver 500 công ty): đo mức nhiễu khi thêm 495 công ty; rồi bộ v2 lấy mẫu từ cả 500 |
| Bộ eval v2 | Lớn hơn 97 câu; người hoặc LLM không phải Claude review mẫu với đoạn đầy đủ; kiểm số/ngày trong đáp án có trong đoạn |
| Chọn judge, chạy eval câu trả lời đầy đủ | Judge khác họ với gpt-oss và Claude; `--rejudge` dùng lại câu trả lời đã lưu |
| 4 lỗi còn lại | t0042 en (30/9 bị đọc là 10-K, thật ra là Q3), t0042 vi và t0060 vi (xếp hạng), t0077 en (bảng tách mất tiêu đề) |
| Tiêu đề bảng ở các phần sau | Đổi luật chunk → bump rules version → build lại chunk, embedding, Qdrant |
| Độ trễ | Rerank ~3 s/câu trên T4 qua tunnel; query model p95 ~57 s khi phải dịch |
| Baseline và thí nghiệm chunking (B4: A, B+O, C, D, E) | Chưa làm; cần để chứng minh lợi ích của chunk theo cấu trúc |
