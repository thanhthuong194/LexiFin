# Native RAG: kết quả (26/11/2026)

Trạng thái: chunk, embedding, Qdrant, retrieval, sinh câu trả lời và eval đã có code (commit `e59baad` → `b4e358e`, 16–26/11/2026). Bộ v1.x đo trên `data/` (5 công ty); bộ v2 đo trên 500 công ty (homepc, Qdrant 3.737.715 điểm).

Nguồn số liệu: `evals/runs/<run>/report.md` (gitignored, máy local), `docs/data/native_rag_model.md`, `AGENTS.md`, `evals/sets/v1*/`. Dự đoán trước đó: mục B2, B3, B4 của `silver_improvements_report.html` (07–13/11/2026, "ước tính, chưa đo"). Số đánh dấu "ước tính" là suy ra, không phải đo.

## 1. Tóm tắt

- Retrieval trên bộ eval v1.2 (97 câu × en/vi, 5 công ty): post `hit_rate@8` en 0,976, vi 0,976; `mrr@10` en 0,682, vi 0,718; còn 4/166 dòng lỗi (run `20261124-100952_batch2_translate`).
- Cải thiện lớn nhất không đến từ mô hình mà từ việc sửa lỗi im lặng: gpt-oss trả câu tìm kiếm tiếng Việt cho 77/81 câu hỏi tiếng Việt, nên chunk tiếng Anh bị tìm bằng tiếng Việt. Sửa schema + dịch dự phòng: vi `hit_rate@8` 0,880 → 0,976 (khác biệt có ý nghĩa, paired t-test p < 0,05).
- Reranker là thành phần đáng giá nhất: tắt reranker làm vi `hit_rate@8` 0,843 → 0,651, `mrr@10` (all) 0,646 → 0,544 (bộ v1).
- Một phần "lỗi retrieval" là lỗi của bộ eval: bằng chứng chỉ gắn một filing trong khi cùng dòng bảng có ở filing khác (10-Q/10-K lặp bảng của thông cáo). Bộ v1.2 mở rộng bằng chứng theo luật; chấm lại cùng run: lỗi 15 → 13.
- Sinh câu trả lời: thử 10 câu × 2 ngôn ngữ: đúng giá trị 6/6, từ chối đúng 2/2, từ chối nhầm 0/18, có trích dẫn 18/18, trích dẫn chứa bằng chứng 17/18, đúng ngôn ngữ 20/20. Judge chưa chạy (đã chọn Gemini 3.6 Flash, 26/11).
- 500 công ty: 3.737.715 chunk, 3.737.715 vector (Qwen3-Embedding-0.6B, 1024 chiều), audit embedding toàn PASS. Số chunk cao hơn dự đoán ~2,5 tr khoảng 50%.
- 500 công ty, bộ v2 (246 câu: dev 155 gồm 106 câu dev bổ sung, test 91): test `hit@8` en 0,905, vi 0,869; `mrr@10` 0,748 / 0,696 (run `20261126-033541_s500_v2_test_b5_rr4b`, con số chính thức; lần test trước 0,810 / 0,810). Dev sau đợt 4–5 và reranker 4B: 0,811 / 0,804 → 0,912 / 0,905 (mục 4.3).
- Reranker 4B thay 0.6B (cùng code, dev): hit@8 0,885 / 0,858 → 0,912 / 0,905, `mrr@10` có ý nghĩa; rerank 2,5 → 8,8 s/câu.
- Câu trả lời (500 công ty, dev, judge Qwen 3.8 27B): 116/310 dòng sạch, đúng 80%, đúng một phần 12%, sai 8%, faithful 76% (mục 5.2). Phần còn lại chờ hạn mức token ngày của Groq/Cerebras.
- Chưa đo: eval câu trả lời đủ dev và test, baseline chunk cố định, thí nghiệm chunking A–E.

## 2. Pipeline và cấu hình

```
Silver → native_rag.chunks (Gold, Iceberg) → native_rag.embeddings → Qdrant (dense + BM25)
câu hỏi → query model → danh bạ công ty → hybrid (RRF) → gộp trùng → rerank → recency
        → quota theo công ty/năm → mở rộng ngữ cảnh → generator trả lời có [n]
```

### 2.1 Mô hình (`config/llm_profiles.toml`)

| Vai trò | Mô hình | Chạy ở đâu | Ghi chú |
|---|---|---|---|
| `chat.query` | gpt-oss-120b, `effort = low` | Cerebras (Groq sáng 26/11: hết 200k token/ngày giữa eval) | Groq free: 8k token/phút (câu chậm ~50 s là chờ 429), 200k token/ngày |
| `chat.generator` | gpt-oss-120b, `temperature 0` | Cerebras | Cerebras cần `served_model = "gpt-oss-120b"` |
| `chat.judge` | Qwen 3.8 27B (`qwen-3.8-27b`), `max_tokens` 16384 | Cerebras | Khác họ model với generator. Gemini 3.6 Flash giữ ở `chat.judge_gemini` để soi mẫu (free: 20 lượt/ngày); trên 25 câu hai judge khớp 8/8 dòng soi |
| `embedding.default` | Qwen3-Embedding-0.6B, 1024 chiều | vLLM Kaggle 2×T4, Cloudflare tunnel | Cùng tokenizer dùng để giới hạn chunk |
| `rerank.default` | Qwen3-Reranker-4B (0.6B đến 25/11), `prompt_format = "qwen3"` | vLLM Kaggle 2×T4, fp16 | Không có chat template thì điểm gần ngẫu nhiên (0,32 vs 0,26; có template: 0,995 vs 0,0006) |

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

### 4.3 500 công ty, bộ v2

Bộ v2 (`evals/sets/v2/`, spec `docs/plans/eval_v2_spec.md`): 150 task lấy mẫu trên 500 công ty theo ngành GICS, cỡ công ty, form, năm, loại câu (16 loại, thêm calc, ranking, temporal_change, event_8k, relation, aggregation, false_premise); 10 bị bỏ, còn 140. Chia theo hash câu hỏi: dev 49, test 91. Chỉ sửa theo dev; test chạy một lần để báo cáo.

v1.2 trên 500 công ty cho đúng số của 5 công ty (0,976 / 0,976, cùng 4 lỗi): câu hỏi v1 nêu tên công ty, lọc theo CIK loại 495 công ty còn lại.

| Run | Thay đổi | Split | hit@8 en / vi | mrr@10 en / vi | recall@8 en / vi | Đúng công ty en / vi | Lỗi |
|---|---|---|---|---|---|---|---|
| `20261125-035116_s500_v2_dev` | Như batch 2 | dev | 0,870 / 0,891 | 0,602 / 0,636 | 0,793 / 0,866 | 0,898 / 0,959 | 11 |
| `20261125-082441_s500_v2_dev_b3` | Đợt 3: mã chứng khoán viết trong câu thắng đoán của query model; tháng + năm → quý chứa tháng | dev | 0,913 / 0,913 | 0,667 / 0,664 | 0,848 / 0,888 | 0,980 / 0,980 | 8 |
| `20261125-092400_s500_v2_dev_b3b` | Tháng của thương vụ/mua lại là tháng nộp 8-K; "8-Ks" số nhiều | dev | 0,935 / 0,935 | 0,672 / 0,670 | 0,870 / 0,909 | 0,980 / 0,980 | 6 |
| `20261125-093610_s500_v2_test_b3` | Như dòng trên | test | **0,810 / 0,810** | **0,546 / 0,545** | 0,745 / 0,750 | 0,945 / 0,956 | 32 |

- Dev hơn test ~0,12: dev đã dùng để tìm và sửa lỗi; mỗi split ~45–85 câu có bằng chứng, khoảng tin cậy ±0,08. Dev trước đợt 3 (0,870 / 0,891) cũng cao hơn test.
- Theo loại trên test (hit@8 en / vi): calc 0,556 / 0,556; false_premise 0,400 / 0,800; compare_periods 0,800 / 0,400; table_lookup 0,667 / 0,833; ranking 0,800 / 0,800 (recall 0,450 / 0,433); aggregation recall 0,167 / 0,167. Tốt: summary, multi_hop, temporal_change, relation, event_8k (0,83–1,0). Mỗi loại 2–9 câu.
- 5 lỗi dev còn lại (ngoài t0099 vi "COO") đã đúng công ty, năm và form; đoạn đúng không vào top 8 (bảng XBRL, MD&A của câu aggregation).
- Không mở từng câu test để sửa. Lỗi để sửa tiếp phải tìm trên dev; dev còn 6 lỗi, không đủ. Hướng: sinh thêm câu dev cho các loại yếu (mục 8).

Dev bổ sung (25/11, `evals/sets/v2/extra`, chỉ vào dev): 106 câu cho loại yếu, ≥ 20 câu thời gian tương đối, không dùng lại công ty/chunk của test. Dev thành 155 câu (148 dòng có bằng chứng mỗi ngôn ngữ).

| Run | Thay đổi | Split | hit@8 en / vi | mrr@10 en / vi | recall@8 en / vi | Đúng công ty en / vi | Lỗi |
|---|---|---|---|---|---|---|---|
| `20261125-142636_s500_v2dev_extra` | Như đợt 3 | dev 155 | 0,811 / 0,804 | 0,562 / 0,544 | 0,713 / 0,721 | 0,948 / 0,974 | 57 |
| `20261125-160055_s500_v2dev_b4c` | Đợt 4: thời gian tương đối (năm nay, năm ngoái, năm tài chính này theo lịch từng công ty, N tháng trước); ngày viết trong câu thắng kỳ của model; bỏ form câu không nhắc; mã đọc nhầm (UHS, BRO, REG/REGN) | dev 155 | 0,872 / 0,851 | 0,614 / 0,581 | 0,771 / 0,768 | 0,981 / 0,987 | 41 |
| `20261125-165557_s500_v2dev_b5b` | Đợt 5: năm lấy từ câu khi model không trả năm; CAGR/thay đổi bao nhiêu chỉ giữ 2 năm đầu–cuối; thêm `context.answer_figures` | dev 155 | 0,885 / 0,858 | 0,619 / 0,593 | 0,781 / 0,775 | 0,981 / 0,987 | 38 |
| `20261126-022741_s500_v2dev_b5_rr4b` | Reranker Qwen3-Reranker-4B | dev 155 | 0,912 / 0,905 | 0,682 / 0,686 | 0,819 / 0,822 | 0,981 / 0,987 | 27 |
| `20261126-033541_s500_v2_test_b5_rr4b` | Như dòng trên | test | **0,905 / 0,869** | **0,748 / 0,696** | 0,848 / 0,829 | 0,978 / 0,989 | 19 |

- `context.answer_figures` (ngữ cảnh chứa số của đáp án, ở bất kỳ đoạn nào, không chỉ đoạn bằng chứng): dev 0,875 / 0,911 (0.6B) → 0,938 / 0,922 (4B); test 0,862 / 0,847.
- 4B so với 0.6B trên dev: sửa 17 dòng, hỏng 6; 5/6 dòng hỏng là trượt giả (top 8 vẫn có đoạn trả lời được, chỉ không trùng đoạn bằng chứng đã ghi).
- Hai agent độc lập soi các lỗi còn lại: khoảng 12–15 lỗi là trượt giả; tìm riêng theo từng kỳ (A) không giúp; nhãn dòng bảng (B) đúng một phần, hoãn.

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
| `judge.correct`, `judge.faithful` | – | – | Chưa đo |

- Dòng lỗi duy nhất: t0006 vi trả đúng số ($14.728 tr so với Q3 FY2022) nhưng đoạn được trích không chứa nhóm bằng chứng.
- 10 câu quá ít để kết luận; chưa đo độ trễ sinh câu trả lời. Dự đoán B3/B4 (đúng ≥ 80–87,5%, faithfulness ≥ 90–92%, trễ p95 < 10 s): chưa đo.

### 5.1 Chỉ số dự đoán và thực tế

| Chỉ số | Dự đoán B3 | Dự đoán B4 | Thực tế (5 công ty, v1.2; câu trả lời: 500 công ty, mục 5.2) |
|---|---|---|---|
| Recall@10 hybrid + rerank | ≥ 0,85 | 0,89 | Không đo đúng chỉ số này. Gần nhất: `post.recall@8` en 0,952, vi 0,946 |
| Recall@10 hybrid không rerank | 0,75 | – | `post.recall@8` en 0,867, vi 0,639 (v1, run no_rerank) |
| Recall@10 dense / BM25 riêng | 0,65 / 0,55 | – | Chưa đo |
| MRR@10 | ≥ 0,70 | 0,775 | en 0,682, vi 0,718 |
| Trả lời đúng | ≥ 80% | 87,5% | Judge: 80% đúng, 12% đúng một phần (116 dòng dev sạch, 500 công ty) |
| Faithfulness | ≥ 90% | 92% | Judge: 76% (105 dòng không từ chối); trích dẫn chứa bằng chứng 82% |
| Trễ truy hồi p95 (không LLM) | < 3 s | – | ~4 s chỉ riêng rerank (batch 2) |
| Trễ trả lời p95 | < 10 s | – | Chưa đo; retrieval p95 kể cả query model 61 s |

Lưu ý: dự đoán tính cho 500 công ty và ~300 câu; retrieval ở bảng này là 5 công ty, 97 câu (500 công ty: mục 4.3).

### 5.2 Eval câu trả lời, 500 công ty, dev

Run `20261126-051638_answers_s500_v2dev_answers` (155 câu × en/vi; generator gpt-oss-120b, judge Qwen 3.8 27B, cả hai trên Cerebras). Chưa xong: Groq hết 200k token/ngày ở khoảng dòng 100 (query model rơi về luật), Cerebras hết token/ngày ở dòng 254; dòng hỏng chạy lại bằng `--resume`. Số dưới đây chỉ tính 116 dòng sạch (en 60, vi 56), chấm lại bằng `answer.value` đã sửa.

| Chỉ số | en | vi | Tổng |
|---|---|---|---|
| `judge.correct` | 0,783 | 0,821 | 0,802 |
| `judge.partial` | 0,133 | 0,107 | 0,121 |
| `judge.incorrect` | 0,083 | 0,071 | 0,078 |
| `judge.faithful` | 0,764 | 0,760 | 0,762 |
| `answer.value` | 0,812 (16) | 0,929 (14) | 0,867 (30) |
| `answer.false_refusal` | 0,000 | 0,038 | 0,019 |
| `answer.citation_evidence` | 0,800 | 0,840 | 0,819 |

- Dòng chỉ dùng luật đọc câu hỏi (Groq hết hạn mức, 145 dòng): đúng 57%, so với 79% ở dòng sạch; query model đáng giá với câu trả lời, không chỉ với retrieval.
- Theo retrieval (cả 256 dòng có chấm): đúng 84–85% khi top 8 có đủ bằng chứng, 17–22% khi thiếu.
- Rubric judge (26/11): "đúng" khi trả lời mọi điều câu hỏi hỏi; ngữ cảnh thêm của đáp án mẫu mà câu không hỏi được phép thiếu. Trên 25 câu thử: đúng 0,80, khớp Gemini 8/8 dòng soi.
- `answer.value` sửa 26/11: bỏ số lượng cổ phiếu khi câu không hỏi "bao nhiêu cổ phiếu" (t0051, t0067 hỏi giá nhưng đáp án ghi số cổ phiếu trước); phần trăm được làm tròn như câu trả lời viết (7% cho 7,4%). 116 dòng sạch: 0,667 → 0,867; 4 dòng còn trượt đều bị judge chấm sai.

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
| 8 | Qdrant không lưu được trên ổ DATA | Ổ exFAT không giữ được storage của Qdrant | Đặt storage trên NVMe ext4 | Nạp xong: 3.737.715 điểm, nạp 61 phút, audit PASS, `ann_recall` 1,000 |
| 9 | Audit chunk 500 công ty FAIL `block_in_one_chunk` | Bảng ghi chú sau bảng bị tách giữa hai chunk (v6); dấu `(ii)` đứng riêng nhận hàng đầu của bảng bố cục (v7, 2 block) | Rules v7 (`dd520d4`), v8 (`cb441db`) | Chunk `data/` giống hệt qua v6–v8; 500 công ty v8: 3.737.715 chunk |
| 10 | Judge không chạy | Key NVIDIA bị từ chối (401) | Đổi judge sang Gemini 3.6 Flash (26/11) | 20/20 dòng chờ chấm lại (`--rejudge`) |
| 11 | Sai công ty khi câu chỉ ghi mã (500 công ty): EXE → Exelon, MKC → McKesson, bỏ sót RCL và 4 mã ngân hàng | Query model đoán tên từ mã; tên thắng mã khi tra danh mục | `resolve_in_question`: mã viết trong câu thắng; mã lệch 1 chữ so với mã viết trong câu và câu không nhắc tên → đọc nhầm; chỉ thêm mã rõ là mã (`$`, sở hữu, ngoặc sau tên, danh sách, từ 2 mã); từ viết tắt trùng mã (COO, RF, FICO…) cần `$` hoặc query model xác nhận. Agent review 2 vòng (`32200be`) | Dev: đúng công ty en 0,898 → 0,980; COO vẫn trượt (cố ý) |
| 12 | "February 2022" không ra kỳ (t0067) | Tháng + năm không gắn sự kiện bị bỏ qua | Tháng + năm → quý dương lịch chứa tháng, chỉ khi câu không hỏi năm/ngày khác; bỏ qua sau since/from/due/maturing… (`32200be`) | t0067 qua. Công ty có quý kết thúc tháng 1/4/7/10 có thể trượt tháng đầu quý |
| 13 | "January 2026 deal" đọc thành Q1 2026 (t0078) | Thương vụ không nằm trong từ chỉ sự kiện | deal/acquisition/merger/thương vụ/sáp nhập: tháng là tháng nộp 8-K; nhận "8-Ks", "10-Qs" (`aba868b`) | Dev lỗi 8 → 6 |
| 14 | Thời gian tương đối, form và kỳ do model tự thêm (dev bổ sung) | Model đoán năm/form; "năm ngoái", "năm tài chính này" phụ thuộc ngày hỏi và lịch từng công ty | Đợt 4 (`673766e`): đọc thời gian tương đối từ câu, năm tài chính theo filing mới nhất của công ty, bỏ form câu không nhắc (giữ 8-K cho chủ đề sự kiện) | Dev lỗi 57 → 41 |
| 15 | CAGR/thay đổi lọc cả các năm giữa; model không trả năm | Model trả dải năm hoặc bỏ năm | Đợt 5 (`533f994`): chỉ giữ năm đầu–cuối; năm lấy từ câu (bỏ năm sau since/until/due và năm của sự kiện) | Dev lỗi 41 → 38 |
| 16 | Query model có câu chậm ~50–57 s | Groq free 8k token/phút: thời gian chờ retry sau 429 | `requests_per_minute` cho profile chat (`cc8fdbb`) | Hiểu nguyên nhân; chưa bật cho Groq |
| 17 | Judge Gemini dừng giữa chừng | Gemini 3.6 Flash free chỉ 20 lượt/ngày | Judge chính sang Qwen 3.8 27B trên Cerebras (`b4e358e`) | 2/50 dòng bị cắt ở 8192 token → `max_tokens` 16384; chấm đủ 50 |
| 18 | Judge chấm "đúng một phần" khi câu trả lời đúng điều được hỏi | Rubric đòi đủ mọi chi tiết của đáp án mẫu | Rubric chấm theo điều câu hỏi hỏi (`b4e358e`) | 25 câu: đúng 0,80 |
| 19 | `answer.value` trượt khi câu trả lời đúng | Lấy số đầu tiên của đáp án (số cổ phiếu thay vì giá); phần trăm phải đúng 1 chữ số thập phân | Bỏ số cổ phiếu khi câu không hỏi số lượng; phần trăm theo độ làm tròn của câu trả lời | 0,667 → 0,867, khớp judge |
| 20 | 199/310 dòng eval câu trả lời hỏng | Groq 200k token/ngày, Cerebras hết token/ngày cho gpt-oss-120b | `--resume` khi hạn mức hồi; query và generator dùng hai nhà cung cấp khác nhau | Chờ |

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
| Review bộ v2 trước khi đóng băng | LLM không phải Claude đọc cả 140 câu với đoạn đầy đủ; người review ~30 câu |
| Chạy eval câu trả lời đầy đủ | Dev: `--resume` run `20261126-051638_answers_s500_v2dev_answers` khi hạn mức hồi (cần embedding + reranker); sau đó test. Nếu thiếu bằng chứng là nguyên nhân chính: nới bằng chứng theo luật (bản sao 8-K EX-99, câu MD&A) |
| Build lại index | BM25 có tách gốc từ; sửa nhãn mục sai; nhãn dòng bảng (B); tra XBRL (D) |
| 4 lỗi còn lại (v1.2) | t0042 en (30/9 bị đọc là 10-K, thật ra là Q3), t0042 vi và t0060 vi (xếp hạng), t0077 en (bảng tách mất tiêu đề) |
| Tiêu đề bảng ở các phần sau | Đổi luật chunk → bump rules version → build lại chunk, embedding, Qdrant |
| Độ trễ | Rerank 4B ~8,8 s/câu trên 2×T4 qua tunnel (0.6B: 2,5 s); query model chậm khi Groq trả 429 |
| Baseline và thí nghiệm chunking (B4: A, B+O, C, D, E) | Chưa làm; cần để chứng minh lợi ích của chunk theo cấu trúc |
