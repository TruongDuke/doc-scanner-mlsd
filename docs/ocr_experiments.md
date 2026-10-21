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
| `myscans_test` | **test thật** (18/10, chỉ chấm 1 lần) | 233 dòng / 11.230 ký tự từ 6 trang mới chụp bằng app trên Bee 3: 2 trang báo An Ninh, 4 trang báo cáo kiểm toán VDAC (có đoạn tiếng Anh). Bỏ 71 dòng: 64 dòng thân bài trang báo 2 (chụp cả 2 mặt báo, chữ cao ~10 px, người cũng không đọc chắc), logo, dòng có dấu "•". Dấu đầu dòng bỏ khỏi đáp án. Claude gõ đáp án **không so với chữ model đọc**, user duyệt | `tools/make_myscans_lines.py --dir data/ocr/myscans_test` |
| `govdocs/clean`, `govdocs/cam` | **test (văn bản in, giả lập)** | 1.990 dòng / 94.692 ký tự từ 19 văn bản (49 trang) nhà nước (PDF điện tử có sẵn chữ -> đáp án chính xác), vẽ 150 DPI; `cam` = mô phỏng ảnh chụp camera | `tools/make_govdocs_test.py` |
| `myscans` | **valid thật** (từ 16/10; trước đó là test) | 266 dòng từ 8 trang user scan bằng app trên Bee 3 (phong bì EMS, đơn xin, 2 trang sách, bìa sách, mẫu CT01 2 trang, bảng + nhận xét đề thi). Dòng do DBNet PP-OCRv5 tìm; đáp án do Claude đọc ảnh gõ, user duyệt; 19 mẩu vụn / logo / ký hiệu ngoài từ điển bỏ (`###`) | `tools/make_myscans_lines.py` |

Phạm vi (15/10): app OCR **văn bản in** (A4, công văn, giấy tờ). Chữ viết tay, chữ ngoài đường, bảng phức tạp: ngoài phạm vi.
Chấm văn bản in dùng `--width 0` (ảnh rộng theo độ dài dòng; dòng A4 dài trung vị 69 ký tự, ảnh 320 px chỉ có 40 lát).

VinText: chỉ dùng nghiên cứu / học tập, không phân phối lại, trích dẫn: Nguyen et al., "Dictionary-guided Scene Text Recognition", CVPR 2021.

## Kết quả

| # | Ngày | Model | Thay đổi | synth_v1 CER | VinText valid CER (đúng) | Ghi chú |
|---|---|---|---|---|---|---|
| R0 | 15/10 | latin v5 RapidOCR (503 ký tự) | (tham khảo) | 12.27% | – | Từ điển thiếu chữ có dấu thanh |
| B0 | 15/10 | latin_PP-OCRv5_mobile_rec_pretrained (836 ký tự) | **baseline** = điểm xuất phát fine-tune | 8.90% | **23.74%** (42.0%) | |
| FT1 | 15/10 | B0 + từ điển 236 ký tự Việt | fine-tune 3 epoch trên `synth_train_v1` (Colab T4, batch 64, lr 2e-4) | **0.51%** | 35.58% (35.1%) ❌ | Tệ hơn B0 trên chữ ngoài đường (quên chữ không dấu, số, in hoa, nền tối) |

Văn bản in (`govdocs`, 1.990 dòng, `--width 0`), CER / CER bỏ dấu / dòng đúng:

| # | clean | cam (mô phỏng camera) | Ghi chú |
|---|---|---|---|
| B0 | 12.02% / 11.29% / 9.1% | 12.65% / 11.20% / 8.7% | |
| FT1 | **3.59%** / 0.68% / 29.5% | **4.05%** / 0.90% / 26.4% | Tốt hơn B0 ~3 lần trên đúng loại tài liệu cần. ~3/4 lỗi còn lại là nhầm dấu thanh giống nhau (hỏi/ngã/sắc: chuẩn->chuấn, bổ->bồ, số->sổ), vài lỗi số (5->s) |

Ảnh chụp thật bằng app (`myscans`, 266 dòng, `--width 0`), CER / CER bỏ dấu / dòng đúng:

| # | myscans | Ghi chú |
|---|---|---|
| B0 | 25.94% / 23.50% / 28.9% | |
| FT1 | **6.87%** / 3.50% / 30.5% | Tốt hơn B0 ~3.8 lần, nhưng kém govdocs cam (4.05%) |

FT1 theo từng trang (CER / bỏ dấu): sách 2 trang 3.5% / 1.4% và 5.7% / 3.5%; mẫu CT01 phần chú thích 6.0% / 2.3%; đề thi 8.2% / 3.1%;
đơn xin 9.1% / 4.6%; phong bì EMS 8.4% / 6.5%; **phần đầu mẫu CT01 15.1% / 11.4%** (dòng ngắn, chữ đậm in hoa sát nhau, số chú thích nhỏ `(1)`, dấu chấm điền chỗ trống).
- Mô phỏng `govdocs cam` hơi "dễ" hơn ảnh thật (4.05% so với 6.87%): ảnh thật có thêm chữ đậm in hoa, chữ nghiêng, nền xám tối, mực in mờ.
- Lỗi bỏ dấu (3.5%) cao hơn govdocs (0.9%): ngoài nhầm dấu thanh còn có lỗi chữ thật (`l` thừa trước `h`/`k`: lkhông, lHọ; ô số `0` đọc thành `(`; `1` thành `I`; chữ bị dính / mất ký tự đầu).
- FT2 nên thêm: chữ đậm / nghiêng / in hoa, ô số đứng riêng 1 ký tự, dòng ngắn trong bảng / biểu mẫu, nền tối và mực mờ.

FT1 theo độ dài dòng (CER): govdocs cam 1-5 ký tự 4.8%, 6-25: 4.8%, 26-45: 5.0%, 46-65: 3.9%, 66-85: 3.5%, >85: 8.3% (51 dòng);
myscans 1-5: **17.8%**, 6-25: **14.2%**, 26-45: 6.4%, 46-65: 5.1%, 66-85: 6.2%, >85: 6.9%.
-> Train dòng <= 25 ký tự ở rộng 320 vẫn đọc tốt dòng dài (suy luận rộng tự động). Điểm yếu là dòng NGẮN (ô bảng, biểu mẫu):
FT2 giữ cấu hình train như FT1, không cần train dòng dài.

### FT2 (kế hoạch, 16/10)
- Dữ liệu bản 2 (`tools/ocr_synth.py` phần v2, `tools/make_ocr_train_v2.py`, xem trước bằng `tools/ocr_synth_preview.py`):
  10% ô bảng 1-5 ký tự, 15% dòng biểu mẫu (hàng chấm, số chú thích nhỏ), 15% IN HOA; font 55% thường / 25% đậm / 15% nghiêng /
  5% đậm nghiêng, 40% kiểu Times (Liberation Serif); 50% dòng (trừ ô bảng) có nét dòng trên / dưới lọt vào mép; nền 80% xám /
  xám xanh, 30% tối (110-170); 20% mực mờ đứt nét; 10% đường kẻ sát chữ; ép ngang 40% dòng IN HOA / 10% dòng thường.
- Học tiếp từ FT1, lr 1e-4, 5 epoch, lưu mỗi epoch; chọn epoch bằng `myscans` (valid), báo cáo `govdocs` (test).
  Notebook `colab/ocr_ft2.ipynb` sinh bằng `tools/make_colab_ft2.py` (chép nguyên văn code từ repo).
- Chạy thử trên Mac CPU (512 dòng, batch 16, 48 bước): nạp FT1 đủ mọi lớp, loss 12.3 -> 5.6; FT1 lúc đầu chỉ đúng ~38% dòng
  của dữ liệu mới (dữ liệu bản 1: 96%) -> dữ liệu mới có nhiều kiểu FT1 chưa biết.
- Lưu ý: `myscans` đã dùng để quyết định dữ liệu bản 2 -> giờ là tập valid. Cần thêm ảnh chụp mới làm test cuối.

### FT2 kết quả (16/10)
Colab T4, 5 epoch (bị ngắt 2 lần, chạy tiếp bằng `latest`; mỗi epoch ~21 phút). CER (CER bỏ dấu), `--width 0`:

| | FT1 | FT2 e1 | e2 | e3 | **e4** | best (giữa e4) |
|---|---|---|---|---|---|---|
| myscans (valid) | 6.87% (3.50) | 4.64% (1.49) | 4.60% (1.33) | 4.69% (1.15) | **4.07% (1.15)** | 4.08% (1.13) |
| myscans dòng <= 25 ký tự | 14.82% | 7.41% | 7.41% | 7.63% | **6.98%** | 6.98% |
| myscans: ô "0"/"0%" đọc sai | 15 | 0 | 0 | 0 | 0 | 0 |
| myscans: "l" thừa (lkhông, lHọ) | 4 | 0 | 0 | 0 | 0 | 0 |
| govdocs clean (test) | 3.59% (0.68) | 3.27% | 3.10% | 3.43% | 3.38% (0.50) | 3.47% |
| govdocs cam (test) | 4.05% (0.90) | 3.54% | 3.54% | 3.74% | 3.70% (0.70) | 3.59% |

myscans theo trang, FT1 -> FT2 e4: CT01 đầu trang 15.1 -> 7.6; đơn xin 9.1 -> 4.4; phong bì EMS 8.4 -> 3.4; đề thi 8.2 -> 5.5;
CT01 chú thích 6.0 -> 3.8; sách 3.5 / 5.7 -> 2.3 / 3.2; bìa sách 3.6 -> 2.2. Không trang nào kém đi.
- Chọn **FT2 e4** (tốt nhất trên valid). Trên test govdocs tốt hơn FT1 ~9%, trên valid ~41% (valid lạc quan hơn vì đã dùng để thiết kế dữ liệu).
- Mỗi lỗi nhắm tới đều hết: ô số "0", "l" thừa. Lỗi chữ (bỏ dấu) giảm 3.5% -> 1.15%; còn lại ~70% lỗi là **dấu thanh**.
- Epoch 1 đã được gần hết lợi ích; e1-e5 dao động trong khoảng nhiễu của 266 dòng. e5: myscans 4.08% (1.15), dòng ngắn 7.20%,
  govdocs clean 3.42%, cam 3.66% -> bằng e4; giữ e4 theo quy tắc chọn trên valid.

### Test cuối trên ảnh chụp mới (`myscans_test`, 18/10, chấm 1 lần)

| | CER | CER bỏ dấu | dòng đúng | dòng <= 25 ký tự |
|---|---|---|---|---|
| B0 | 45.34% | 44.03% | 9.9% | 14.29% |
| FT1 | 12.68% | 8.73% | 14.6% | 14.14% |
| **FT2 e4** | **8.33%** | **4.40%** | 15.9% | 8.25% |

Theo trang (FT1 -> FT2): báo trang 1: 4.5 -> 5.0; báo trang 2 (8 dòng tiêu đề) 26.1 -> 10.1; VDAC báo cáo 4.4 -> 3.8;
VDAC rủi ro (Việt + Anh) 18.1 -> 9.7; VDAC đề xuất 17.5 -> 12.2; VDAC tồn tại / kế hoạch 11.1 -> 7.9.
- FT2 tốt hơn FT1 ~34% trên ảnh chưa từng thấy -> cải thiện của FT2 là thật, không chỉ do "ôn đúng đề".
- Lỗi tuyệt đối cao hơn valid (8.3% so với 4.1%) vì trang test khó hơn: dòng rất dài (86+ ký tự: 10.9%) và **chữ rất nhỏ**:
  ảnh dòng cao 29-39 px cho dòng ~100 ký tự (trang A4 sau nắn chỉ ~990 x 1400 px). Tiếng Anh bị đọc kiểu tiếng Việt
  (i -> l, u -> ư). Dấu đầu dòng "•" (không có trong đáp án) bị đọc thành "-".
- Hướng rẻ nhất tiếp theo có lẽ không phải train: tăng độ phân giải ảnh chụp / ảnh nắn trong app.

## Quantize model đọc chữ (FT2 e4), PTQ (18-19/10)

Đường đi: ONNX -> onnxsim (cố định 1x3x48xW) -> onnx2tf -> TFLite, 4 cỡ độ rộng 320 / 640 / 1280 / 2000 (mỗi dòng dùng cỡ nhỏ nhất
đủ rộng, đệm 0; đo trên ONNX không đổi CER) (`tools/export_ocr_tflite.py`, ghi thêm signature vào flatbuffer vì
ai_edge_quantizer cần). FP32 TFLite 4 cỡ: myscans 4.04% / govdocs cam 3.71%; FP16 như FP32.
Calibration: 200 dòng máy vẽ bản 2 mỗi cỡ, đúng độ rộng (`tools/make_ocr_calib.py`). Quantize: `tools/quantize_ocr.py`
(ai_edge_quantizer static_wi8_ai8, tự tính "thước" từ min / max từng ảnh). Trước đó đo khoảng giá trị
(`tools/ocr_act_ranges.py`): lớp cuối an toàn (int8 riêng lớp cuối: CER 4.07 -> 4.10), attention gọn, ngoại lai ở cuối CNN.

| Bản (INT8, CER myscans / govdocs cam) | | Ghi chú |
|---|---|---|
| v1 mặc định = trung bình trượt 0.95 của min/max từng ảnh | 8.21% / 6.41% | đầu vào chỉ phủ -0.50..0.47 |
| min/max toàn bộ | 69.6% / 66.3% | ngoại lai kéo thước dài -> hỏng |
| p99 / p95 / p90 | 54.5 / 16.1 / 9.5% | |
| p75 / p50 | 7.20 / 7.71% (gov 5.08 / 5.25) | p75 tốt nhất trong các cách 1 quy tắc |
| p75 + giữ float lớp cuối / SVTR / cuối CNN / cả CNN | 7.20 / 7.37 / 6.27 / **4.05%** | lỗi nằm hết trong CNN |
| chỉ quantize CNN đầu (0-53) / giữa / cuối | 5.88 / 4.83 / 4.36% (gov 5.12 / 3.79 / 3.64) | thủ phạm: CNN đầu |
| p75in (đầu vào -1..1), chỉ quantize khối b0 (op 0-2) / b2 (op 6-9) / khối khác | 4.71 / 5.05 / 3.99-4.12% | 2 khối gây hại |
| **p75in + giữ float b0, b2 (7/241 op, 2.27 MB)** | **6.08% / 3.94%** | |
| p75in + giữ float b0, b2, cuối CNN (4.36 MB) | 5.34% / 3.80% | |

**Chẩn đoán và chữa tận gốc (19/10)**, mô phỏng làm tròn trên ONNX (`tools/ocr_sim_quant.py`, khớp TFLite: b0 4.85 vs 4.71,
b2 5.11 vs 5.05): lỗi do ACTIVATION, không phải trọng số (chỉ trọng số b0 / b2: 4.08 / 4.04%; chỉ activation: 4.88 / 5.21%).
- b0: đầu ra conv đầu + BN (16 kênh, kênh lớn nhất / trung vị 4.1x) gây hết lỗi (riêng tensor này 4.90%). Giữa nó và depthwise
  không có hàm phi tuyến -> **cân bằng kênh** (nhân kênh c của BN với s_c, chia trọng số depthwise kênh c cho s_c; FP32 không đổi,
  `data/models/ocr/vi_ft2_e4_cle.onnx`): b0 còn 4.08%.
- b2: tensor trước HARD_SWISH ("Add.9", thước -40..44, kênh 6.2x) riêng nó 4.47%. hard_swish(x) = 0 với x <= -3 -> **đầu dưới thước = -3**
  không mất gì; áp cho cả 28 tensor vào HARD_SWISH. Chênh kênh ở đây không cân bằng được (sau nó là hàm phi tuyến).
- **Thước MSE riêng từng tensor** (`tools/mse_ruler.py`: histogram 4096 ô, chọn (lo, hi) ít sai số làm tròn + cắt nhất).

| INT8 HOÀN TOÀN (không giữ lớp float), 2.27 MB | myscans | govdocs cam |
|---|---|---|
| p75 + đầu vào -1..1 | 7.58% | 4.75% |
| + cân bằng kênh khối đầu | 6.90% | 4.38% |
| + cắt dưới -3 trước HARD_SWISH | 6.53% | 4.22% |
| **cân bằng kênh + thước MSE (+ -3) + đầu vào -1..1** | **5.12%** | **4.10%** |
| (giữ float b0, b2, cách "né") | 6.08% | 3.94% |
| FP32 | 4.04% | 3.71% |

- Bài học: thước "thường gặp" (trung bình trượt) tốt hơn hẳn min/max vì có ngoại lai; nhưng không 1 quy tắc nào hợp mọi lớp.

**Chốt (19/10): INT8 = cân bằng kênh + thước MSE + đầu vào -1..1** (`ft2cle_w*_msein_int8.tflite`), chấp nhận khoảng cách còn lại.
Chấm bộ TEST (`myscans_test`, 233 dòng) 1 lần sau khi chốt, TFLite 4 cỡ:

| | valid (myscans) | govdocs cam | **test (myscans_test)** | dung lượng / cỡ |
|---|---|---|---|---|
| FP32 | 4.04% | 3.71% | **7.73%** | 7.7 MB |
| FP16 | 4.04% | 3.72% | 7.75% | 3.9 MB |
| INT8 v1 mặc định | 8.21% | 6.41% | 13.73% | 2.27 MB |
| **INT8 chốt** | 5.12% | 4.10% | **10.44%** | 2.27 MB |

- Trên test, cách chữa lấy lại ~55% phần INT8 làm mất (valid ~74%): một phần do đã chọn cách chữa nhìn valid, một phần do trang
  test chữ nhỏ hơn (nhạy với làm tròn hơn). Khoảng cách INT8 - FP32 trên test: 2.7 điểm.
- (Bản ONNX rộng tự động trên test: 8.33%; TFLite 4 cỡ 7.73%: đệm theo cỡ hơi tốt hơn.)

**Tốc độ trên Bee 3** (19/10, `benchmark_model` chính thức bản android_arm, 4 luồng CPU, XNNPACK, lần nhanh nhất trong 15 lần;
trung vị chênh 1-5%, có lúc máy bận làm trung vị tăng tới 1.9 lần), ms / 1 dòng:

| cỡ | FP32 | FP16 | INT8 chốt | INT8 nhanh hơn |
|---|---|---|---|---|
| 320 | 185 | 191 | 86 | 2.2x |
| 640 | 395 | 380 | 179 | 2.2x |
| 1280 | 781 | 780 | 358 | 2.2x |
| 2000 | 1228 | 1191 | 569 | 2.2x |

- FP16 không nhanh hơn FP32 trên CPU (chỉ nhỏ file). INT8 nhanh 2.2 lần (M-LSD trên Bee 3: 1.7 lần).
- Ước tính 1 trang A4 hành chính (~41 dòng, phân bố cỡ của govdocs): FP32 ~23 s, INT8 ~10.5 s (chưa tính bước tìm dòng DBNet).

**Trong app trên Bee 3** (20/10, nút "Kiểm tra OCR", `OcrRecognizer.kt` + `OcrGoldenCheck.kt`, 52 dòng valid từ
`tools/make_ocr_golden.py`): độ chính xác như Mac (CER máy 5.19% / Mac 5.15%, 42/52 dòng chữ trùng hệt; 30/52 trùng mọi lát:
bộ tính int8 ARMv7 làm tròn hơi khác). Tốc độ riêng model, ms / dòng (cỡ 320 / 640 / 1280 / 2000):
- LiteRT 1.4.2, API Interpreter: 661 / 1337 / 2698 / 4248 (chậm 7.7x so với benchmark_model)
- LiteRT 2.3.0, API Interpreter (+ setUseXNNPACK): 230 / 484 / 1024 / 1665
- **LiteRT 2.3.0, API CompiledModel (CPU 4 luồng): 81 / 172 / 387 / 659** = bằng benchmark_model -> 1 trang A4 ~11-12 s.
- Nâng LiteRT 2.3.0 cần Kotlin 2.3.21 (thư viện biên dịch bằng Kotlin 2.4); GpuDelegate kiểu cũ bị bỏ (GPU chỉ qua
  CompiledModel) -> tạm tắt GPU trong "Đo tốc độ" của M-LSD. Golden M-LSD vẫn (A) 6/6, (B) 6/6 PASS.
- Ảnh thật (myscans) nhạy hơn govdocs: còn cách FP32 1.3-2 điểm. Chưa đo tốc độ trên Bee 3.

## Đọc cả trang trong app (21/10)

Tìm dòng: DBNet PP-OCRv5 mobile det (RapidOCR, FP32 TFLite cỡ cố định 1408x992 trang dọc / 992x1408 trang ngang).
Bee 3 (benchmark_model): 2.0 s ở 1408x992, 1.2 s ở 1024x736 (cỡ nhỏ tìm lại kém hơn: 96.7% so với 98.9%).
Xử lý sau DBNet đơn giản để viết lại bằng Kotlin (`tools/det_simple.py` -> `OcrPage.kt`): vùng liền nhau + khung xoay theo
hướng chính (PCA), nới theo công thức unclip của RapidOCR, cắt dòng bằng affine. Trên 11 trang dọc đã chụp:
- khung thẳng + điểm TB trên cả khung: tìm lại 64% dòng của RapidOCR (dòng nghiêng nhẹ bị điểm thấp -> bỏ)
- điểm TB trên các điểm của vùng: 98.9%; nhưng khung thẳng ôm luôn dòng bên cạnh (trang lệch ~1 độ) -> CER cả chuỗi 46%
- khung xoay (PCA) + nghiêng > 20 độ coi như ngang (ô số "1"): CER 5.81% so với 5.07% khi dùng khung RapidOCR
  (phần chênh chủ yếu do cách chấm: gộp "sự sống" + "của" thành 1 dòng, hàng chấm điền chỗ trống)
App (Bee 3, "Kiểm tra OCR trang", 3 trang): khung trùng Mac 30/30, 38/38, 58/58; chữ cả trang lệch Mac 2.3-5.9% (Android co / cắt
ảnh hơi khác cv2; chưa chấm theo đáp án). Thời gian 1 trang 7-15 s (tìm dòng ~2.3 s, đọc chữ 4.7-12.8 s).
Lỗi đã gặp: CompiledModel.create không truyền Environment -> môi trường tạm bị GC huỷ giữa chừng -> treo "Đang đọc chữ...".
Sửa: 1 Environment dùng chung, giữ suốt đời OcrRecognizer.

## Bài học
- Chọn bộ test đúng loại tài liệu: cùng model FT1 mà VinText (chữ ngoài đường) cho kết luận "tệ hơn B0", văn bản in cho "tốt hơn B0 3 lần".
- Đánh giá chỉ trên ảnh tổng hợp cùng kiểu với train là **đánh lừa** (FT1: 0.51% tổng hợp nhưng 35.6% thật). Quyết định dựa trên **valid thật** (VinText valid, sau này ảnh scan).
- Phần wiki và phần giấy tờ của `synth_v1` cho CER như nhau (0.51%): không phải học thuộc cụm từ, mà là quen **kiểu ảnh** tổng hợp.
