# Sổ thí nghiệm OCR (mốc 8): model đọc chữ tiếng Việt

Mọi model chấm bằng **cùng một script** (`tools/ocr_eval.py`): CER = tổng khoảng cách sửa / tổng ký tự đáp án;
"CER bỏ dấu" = như trên sau khi xoá hết dấu (tách lỗi dấu khỏi lỗi chữ); "đúng" = % dòng / từ đọc đúng hoàn toàn.

## Bộ dữ liệu

| Bộ | Vai trò | Nội dung | Tạo bằng |
|---|---|---|---|
| `synth_train_v1` | train | 60.000 dòng tổng hợp (Wikipedia + cụm từ giấy tờ), 8 họ font, nền sáng, dòng ≤ 25 ký tự | `colab/ocr_buoc3_data.ipynb` (`tools/ocr_synth.py`) |
| `synth_v1` | valid (tổng hợp) | 1.500 dòng tổng hợp, font NotoSerif (không có trong train) | `tools/make_ocr_eval.py` |
| `vintext/train` | train (thật) | 24.200 từ, ảnh chữ ngoài đời (VinText, VinAI CVPR 2021) | `tools/make_vintext_lines.py` |
| `vintext/valid` | **valid (thật)** | 6.841 từ (VinText test_image) | như trên |
| `vintext/unseen` | test phụ (thật) | 9.624 từ (VinText unseen_test_images) | như trên |
| ảnh scan bằng app | **test chính (thật)** | chưa có: user scan ~20 tài liệu, gõ đáp án | (sẽ làm) |

VinText: chỉ dùng nghiên cứu / học tập, không phân phối lại, trích dẫn: Nguyen et al., "Dictionary-guided Scene Text Recognition", CVPR 2021.

## Kết quả

| # | Ngày | Model | Thay đổi | synth_v1 CER | VinText valid CER (đúng) | Ghi chú |
|---|---|---|---|---|---|---|
| R0 | 15/10 | latin v5 RapidOCR (503 ký tự) | (tham khảo) | 12.27% | – | Từ điển thiếu chữ có dấu thanh |
| B0 | 15/10 | latin_PP-OCRv5_mobile_rec_pretrained (836 ký tự) | **baseline** = điểm xuất phát fine-tune | 8.90% | **23.74%** (42.0%) | |
| FT1 | 15/10 | B0 + từ điển 236 ký tự Việt | fine-tune 3 epoch trên `synth_train_v1` (Colab T4, batch 64, lr 2e-4) | **0.51%** | 35.58% (35.1%) ❌ | **Overfit ảnh tổng hợp**: tốt trên synth, tệ hơn B0 trên ảnh thật |

## Bài học
- Đánh giá chỉ trên ảnh tổng hợp cùng kiểu với train là **đánh lừa** (FT1: 0.51% tổng hợp nhưng 35.6% thật). Quyết định dựa trên **valid thật** (VinText valid, sau này ảnh scan).
- Phần wiki và phần giấy tờ của `synth_v1` cho CER như nhau (0.51%): không phải học thuộc cụm từ, mà là quen **kiểu ảnh** tổng hợp.
