# Báo cáo Silver: chuẩn hoá, lỗi, scale và tối ưu (27/10/2026)

Trả lời 6 câu hỏi: dữ liệu được chuẩn hoá từ raw ra sao, lỗi gặp phải, scale 5 → 500 công ty, tối ưu đã làm, tối ưu tiếp theo, kỹ năng data engineering.

- `silver_scale_report.html`: báo cáo đầy đủ, biểu đồ tương tác. Mở trực tiếp bằng trình duyệt, không cần mạng.
- `silver_scale_slides.html`: 14 slide kèm ghi chú thuyết trình (font tải từ Google Fonts, offline thì dùng font hệ thống).
- `silver_improvements_report.html` (07/11/2026): A. cải thiện sau báo cáo scale trên Silver 500 công ty (audit 18,4 → 4,8 h, sửa OOM, EX-99, section); B. kế hoạch RAG: thông số kỹ thuật và kết quả dự kiến (ước tính, chưa đo). B4 (13/11/2026): cải tiến chunking sau khảo sát dữ liệu, bộ thí nghiệm A–E, chỉ số dự kiến. C (13/11/2026): kế hoạch GraphRAG (3 lớp, Neo4j, đánh giá, lộ trình, kết quả dự kiến).
- `graphrag.md` (13/11/2026): GraphRAG là gì, quy mô data, thiết kế 3 lớp (tất định từ Silver, LLM trích có kiểm chứng, tóm tắt nhóm), Neo4j hay NebulaGraph, đánh giá, lộ trình. Bản thảo luận, chưa có code.
- `native_rag_results.md` (24/11/2026): kết quả native RAG: pipeline và cấu hình, chi phí build 5/500 công ty, lịch sử eval retrieval (v1 → v1.2, hit@8 en/vi 0,976), thử sinh câu trả lời, lỗi và cách sửa, so với dự đoán ở mục B.

Số liệu lấy từ `scripts/sec_filings/benchmark_silver.py` (code `ef316fa`):

| Mức | Filing | Row Silver | Build | RAM đỉnh | Integrity |
|---|---|---|---|---|---|
| 5 công ty | 343 | 2,2 tr | 10 phút | 2,6 GB | 0 FAIL |
| 50 công ty | 3.850 | 28,2 tr | 2,7 giờ | 6,3 GB | 0 FAIL |
| 100 công ty | 7.977 | 57,3 tr | 5,8 giờ | 7,4 GB | 0 FAIL |
| 200 công ty | 16.141 | 113,3 tr | 12,3 giờ | 6,5 GB | 0 FAIL |
| 500 công ty (code cũ) | 40.824 | 269,0 tr | ~41 giờ máy | — | 0 FAIL |

Tối ưu (mẫu 2% của 500 công ty, 769 filing, cùng 4.716.499 row): build 34,1 → 21,5 phút (−37%), chạy lại 5,0 phút → 101 s.
