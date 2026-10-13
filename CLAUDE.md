# Hướng dẫn cho agent

Trả lời bằng tiếng Việt.

## Dự án

App scan tài liệu chạy offline trên Android, dùng M-LSD tiny 512 (phát hiện đoạn thẳng) để tìm 4 góc tờ giấy, rồi so sánh các bản model FP32 / FP16 / INT8 và CPU / GPU trên điện thoại thật. Tính năng và lộ trình 7 mốc: xem `README.md`.

Đây là dự án cá nhân để học, nối tiếp phần luyện tập ở repo https://github.com/TruongDuke/Quantization (thư mục `5-mlsd/`: convert M-LSD sang TFLite, quantize, và `5-mlsd/ALGORITHM.md` là phần giải thích thuật toán post-processing do user tự viết).

## Cách làm việc với user (quan trọng)

- Ban đầu user định tự viết phần cốt lõi (post-processing, sửa quantization), agent chỉ đưa khung `???` và review. Thực tế user đã đổi ý: `predLines`, `predSquares` do agent viết theo yêu cầu rõ, user review lại. Với phần sửa quantization (mốc 5), hỏi user muốn tự viết hay để agent viết trước khi bắt đầu.
- Phần phụ trợ (công cụ sinh dữ liệu test, đọc file / so sánh, Gradle, giao diện Android, CameraX, code vẽ): agent sửa thẳng vào repo, tự build (`./gradlew installDebug` trong `android/`) và chạy thử trên máy ảo (adb), rồi báo kết quả kèm giải thích từng phần để user review. Chưa commit khi user chưa xem, trừ khi user bảo làm luôn.
- User chưa vững Kotlin/Android, giải thích đơn giản, từng bước nhỏ.

## Tình trạng hiện tại: mốc 4

Mốc 1, 2, 3 xong.
- `postprocess/src/` (`package mlsd`, app Android dùng chung qua sourceSets trong `android/app/build.gradle.kts`): `PredLines.kt`, `PredSquares.kt` (sau bước lọc dùng `Double` vì numpy là float64), `ResizeArea.kt` (chép lại `cv2.INTER_AREA` của OpenCV 4.10: nhánh thu nhỏ số nguyên lần, tỉ lệ lẻ, phóng to; khớp từng điểm ảnh).
- `postprocess/test/`: golden test trên máy tính. Test squares so các tứ giác bằng điểm như một tập vì `np.argsort` không ổn định với mảng lớn.
- App `android/` (Compose, package `com.truongduke.docscanner`, minSdk 26, LiteRT 1.0.1, model trong `app/src/main/assets/`): chọn ảnh (Photo Picker) -> `bitmapToRgb` -> `resizeArea` -> model (`MlsdModel.kt`) -> `predLines` / `predSquares` -> vẽ (`Drawing.kt`). Nút "Kiểm tra golden" (`GoldenCheck.kt`) chạy cả chuỗi trên ảnh mẫu: trên máy ảo khớp tuyệt đối 6/6 (giải mã JPEG, đầu vào, output model, đoạn, tứ giác).
- Hạn chế đã biết: ảnh cạnh dài > 2048 bị `loadBitmap` thu nhỏ bằng `inSampleSize` nên sẽ khác Python chạy trên ảnh gốc; `predSquares` chậm khi nhiều tứ giác (ảnh đầu đĩa: 2920 tứ giác), cần đo lại trên máy thật.

Điện thoại thật của user: **Vsmart Bee 3** (MediaTek MT6739, 4x Cortex-A53, **32-bit armeabi-v7a**, RAM 2 GB, Android 9, cổng micro-USB; serial adb `AIE00201260`, cần `ANDROID_SERIAL` khi máy ảo cũng chạy). Trên máy này Logcat không hiện `Log.d` của app, đọc kết quả qua màn hình (`uiautomator dump`). Kết quả "Kiểm tra golden" (13/10): (A) 6/6, (B) 6/6 PASS (model lệch score ~1e-6, vmap ~1e-2 nhưng đoạn / tứ giác khớp hết); (C) giải mã JPEG của máy khác cv2 (lệch tới 32 mức) nên kết quả từ file JPEG khác Python, chỉ để tham khảo. Thời gian FP32: model ~2.05 s, tiền xử lý ảnh 1920x1440 ~1.8 s (cần tối ưu `bitmapToRgb` + `resizeArea`), post-processing 35 ms (ảnh nhiều tứ giác 372 ms).

Mốc 4 (chưa bắt đầu): CameraX chạy liên tục, chụp, nắn thẳng, chỉnh góc, lưu ảnh.

Chạy golden test từ thư mục gốc repo:
```bash
kotlinc postprocess/src/PredLines.kt postprocess/test/GoldenLines.kt -include-runtime -d build/predlines.jar && java -jar build/predlines.jar
kotlinc postprocess/src/PredSquares.kt postprocess/test/GoldenSquares.kt -include-runtime -d build/predsquares.jar && java -jar build/predsquares.jar
kotlinc postprocess/src/ResizeArea.kt postprocess/test/GoldenResize.kt -include-runtime -d build/resize.jar && java -jar build/resize.jar   # cần thêm: python tools/make_resize_golden.py
```
Kiểm tra trên máy ảo / điện thoại: cài app (`cd android && ./gradlew installDebug`), chạy `tools/push_golden.sh` (chép golden + ảnh mẫu vào `files/golden` của app bằng `run-as`, chỉ bản debug), rồi bấm "Kiểm tra golden". adb nằm ở `~/Library/Android/sdk/platform-tools/adb` (không có trong PATH).

## Dữ liệu golden (`golden/`, không đưa lên git)

Sinh bằng `tools/make_golden.py` từ model chính thức `third_party/mlsd-official/tflite_models/M-LSD_512_tiny_fp32.tflite` và hàm `pred_lines` của repo gốc (`third_party/mlsd-official/utils.py`, dòng 12-47). Mỗi ảnh trong `samples/` + 1 ảnh trơn `blank`:
- `{name}_pts.bin`: int32 little-endian, (200, 2), thứ tự **(y, x)**, lưới 256x256
- `{name}_scores.bin`: float32, (200,)
- `{name}_vmap.bin`: float32, (256, 256, 4) làm phẳng: `vmap[(y * 256 + x) * 4 + c]`, c = (dx_start, dy_start, dx_end, dy_end)
- `{name}_lines.bin`: float32, (n, 4) = (x1, y1, x2, y2) trên ảnh gốc, kết quả mong đợi
- `{name}_sq_segments.bin`: float64, (n, 4), đoạn đã gộp của `pred_squares` trên ảnh gốc
- `{name}_squares.bin`: int32, (m, 4, 2), tứ giác xếp theo điểm giảm dần, góc theo thứ tự 0 trên-trái, 1 trên-phải, 2 dưới-phải, 3 dưới-trái
- `{name}_sq_scores.bin`: float64, (m,)
- `{name}_inter.bin`: int32, (k, 2), các góc hợp lệ
- `{name}_rgb.bin`: uint8, (h, w, 3) RGB, ảnh sau khi cv2 giải mã
- `{name}_input.bin`: uint8, (512, 512, 3), `cv2.resize(..., INTER_AREA)`, đầu vào model
- `meta.json`: tên, h, w, số đoạn, số đoạn gộp / tứ giác / góc
- `golden/resize/` (sinh bằng `tools/make_resize_golden.py`): ảnh nhiễu nhiều kích thước + kết quả `INTER_AREA`, để test `resizeArea`

Tiêu chí PASS: cùng số đoạn, lệch tối đa < 0.01 px.

## Môi trường trên máy user (macOS, Apple M1 Pro)

- `kotlinc` (Homebrew), Java 21, Android Studio có máy ảo `Medium_Phone_API_36.0` (arm64). User có điện thoại Android thật (dùng cho đo tốc độ, máy ảo chậm hơn máy thật rất nhiều).
- Python để chạy `tools/make_golden.py` cần TensorFlow 2.15 + numpy 1.26 + opencv 4.10 (`tools/requirements.txt`). Trên máy user hiện đang dùng venv `/Users/duc/Neural Network Quantization/huy-lab/.venv-huy/bin/python`. Nếu tạo venv riêng cho repo này thì nên để ở `.venv/` (đã có trong `.gitignore`).
- `third_party/mlsd-official`: clone của https://github.com/navervision/mlsd (không đưa lên git).
- `samples/`: ảnh thử (không đưa lên git vì bản quyền). Hiện là `frame_1.jpg` (từ repo lhwcv/mlsd_pytorch) và 4 ảnh Imagenette. Nên thay dần bằng ảnh tài liệu user tự chụp.

## Điểm dễ sai khi port post-processing (rút ra từ bản Python)

- `pts` là (y, x), còn `vmap` và đoạn thẳng là (x, y).
- Bản Python crash khi không có đoạn nào (`np.array([])` rồi index cột). Bản Kotlin phải trả list rỗng.
- Điều kiện giữ đoạn: `score > score_thr` và `độ dài > dist_thr` (dấu `>`, không phải `>=`).
- Đổi toạ độ: nhân 2 (lưới 256 lên 512), rồi nhân `imgW / 512f` cho x, `imgH / 512f` cho y. Chú ý chia số thực trong Kotlin.
- `pred_squares` (mốc 2): numpy cho phép chỉ số âm (tự quay vòng) ở bước gộp đoạn, Kotlin sẽ lỗi, phải kiểm tra biên; đoạn song song làm `det` gần 0; bước ghép tứ giác có 4 vòng lặp lồng nhau; khối chấm điểm dùng `try/except: pass` nuốt lỗi.

## Dữ liệu tài liệu thật (cho đánh giá chất lượng và calibration INT8)

- `data/smartdoc15/` (không đưa lên git): SmartDoc 2015 Challenge 1, bản v2.0.0 từ github.com/jchazalon/smartdoc15-ch1-dataset, giấy phép CC BY 4.0 (trích dẫn bài ICDAR 2015 SmartDoc của Burie et al.). 24.889 khung hình 1920x1080 cắt từ video, 30 tài liệu (6 loại: datasheet, letter, magazine, paper, patent, tax) trên 5 nền (background05 có đồ vật lộn xộn). `metadata.csv.gz`: đường dẫn ảnh + 4 góc đáp án `tl, bl, br, tr` (x, y). Khung hình liền nhau gần giống hệt nhau: chia calibration / đánh giá **theo tài liệu**, không chia ngẫu nhiên theo khung hình.
- Chia tập: `tools/make_doc_split.py` -> `data/calib.txt` (360 khung, 18 tài liệu, cho calibration INT8) và `data/eval.csv` (120 khung, 12 tài liệu khác, có đáp án).
- Đo chất lượng: `tools/eval_quads.py [--model x.tflite]` (model + `pred_squares` repo gốc, tứ giác tốt nhất so với đáp án: IoU, % khung IoU>=0.9, sai số góc px). Mức trần FP32 (13/10): IoU TB 0.786, đúng 56.7%, sai số góc trung vị 14 px; nền 1-4: IoU 0.85-0.89, đúng 62-79%; nền 5 (lộn xộn): IoU 0.47, đúng 8%, thường chọn nhầm quyển tạp chí bìa xanh. Lỗi hay gặp ở nền trắng: chọn mép khối chữ bên trong thay cho mép giấy.
- `samples/docs/`: ảnh tài liệu user tự chụp (to, có EXIF xoay), không có đáp án.
- Dự phòng nếu cần đa dạng hơn: Mendeley "A New Image Dataset for Document Corner Localization" (1.111 ảnh, CC BY 4.0).

## Bối cảnh quantization (cho mốc 5)

**Bước 1 xong (13/10), `tools/sim_quant_head.py`**: mô phỏng int8 chỉ ở output head (tensor 156 `Decoder/resize`, 256x256x16; kênh 7 = center -> sigmoid -> maxpool 3x3 -> top-k 200; kênh 8-11 = vmap; 11 kênh còn lại không dùng), các layer khác FP32, đo trên 120 khung SmartDoc:
- 1 scale chung 16 kênh, min/max (như TFLite): IoU 0.000, giữ 1.8% đoạn -> tái hiện đúng lỗi INT8.
- Nguyên nhân thật gồm 2 phần: (1) dùng chung scale, (2) **ngoại lai**: vmap ở chỗ không có đoạn thẳng lên tới ±1100 (tại điểm center chỉ ±111), logit center xuống -47 (điểm được giữ chỉ từ -2.75). Min/max lấy ngoại lai nên scale khoảng 8.5 ô lưới / mức.
- Tách center / vmap **và** cắt khoảng (center -8..max, vmap ±128): giữ 99.6% đoạn, IoU 0.776 (FP32 0.786), đúng 56.7% (bằng FP32). Chỉ cắt khoảng mà vẫn chung scale: IoU 0.691. Chỉ per-channel min/max: 0.705.
- Hướng sửa model thật: tách head thành 2 output + chặn khoảng giá trị (clip) trước khi quantize, hoặc giữ head FP32.

**Bước 2-3 (13/10), model thật**: không có mã nguồn bản NAVER (chỉ có .tflite) nên dùng bản PyTorch lhwcv (`third_party/mlsd-pytorch` là symlink tới `/Users/duc/Neural Network Quantization/5-mlsd/mlsd-pytorch`), convert với vào/ra giống hệt NAVER (`tools/export_lhwcv.py`). Môi trường: `/Users/duc/Neural Network Quantization/3-deploy/.venv` (torch, litert_torch 0.9.4, ai_edge_quantizer 0.9.0, ai_edge_litert; không có TF). Đo: `tools/run_raw.py <model>` (runtime mới, lưu output thô `data/raw/*.npz`, tự đổi số vào/ra int8) rồi `tools/eval_quads.py --raw ...` (môi trường TF 2.15). Model ở `data/models/` (không lên git), tạo bằng `tools/quantize_lhwcv.py` (calibration 360 khung SmartDoc).
Kết quả trên 120 khung SmartDoc (IoU TB / đúng IoU>=0.9 / Mac 4 thread):
- NAVER FP32: 0.786 / 56.7% / 34 ms. lhwcv FP32: 0.802 / 55.0% / 33 ms (tách head, cắt ngoại lai không đổi FP32).
- V1 INT8 mặc định: 0.395 / 36.7% (66/120 khung không tìm được tứ giác) / 10 ms.
- V2 + tách head: 0.734 / 44.2%. V3 + cắt ngoại lai: 0.751 / 51.7% / 9.7 ms. V4 + giữ 3 layer nhạy float: 0.744. V5 + giữ head float: 0.755 / 51.7% / 10.5 ms.
- Trên tập này giữ 3 layer nhạy ở float gần như không giúp (khác thí nghiệm 7-mlsd-lowbit đo theo % đoạn thẳng). Phần mất còn lại chủ yếu ở nền 5 (lộn xộn): 0.53 -> 0.34.
- Trên **Bee 3** (13/10, nút "Đo tốc độ" trong app, frame_1, 4 thread, lấy lần nhanh nhất vì máy hay bận/nóng): NAVER FP32 ~1.97 s, lhwcv FP32 ~2.0 s, INT8 mặc định ~1.19 s, INT8 đã sửa ~1.18 s -> INT8 chỉ nhanh ~1.7 lần (Mac 3.4 lần), có lẽ do Cortex-A53 32-bit không có lệnh dot-product int8. LiteRT 1.4.2 không nhanh hơn 1.0.1 (app đã nâng lên 1.4.2, golden vẫn 6/6). App có 4 model trong assets, chọn bằng radio; golden luôn chạy bằng NAVER.
- Model do ai_edge_quantizer tạo có vào/ra **int8** (vào scale 0.988 zp -128; scores scale 1/273; vmap scale 1.0 ở V3-V5, 3.7 ở V1-V2): app phải đổi số. Chưa thử trên LiteRT 1.0.1 của app / Bee 3.


Từ thí nghiệm ở repo Quantization (`5-mlsd/BAO_CAO.md`): quantize INT8 M-LSD nhanh hơn khoảng 3.6 lần nhưng chỉ giữ được 37-60% đoạn thẳng. Nguyên nhân chính: 9 kênh output dùng chung một scale (kênh displacement khoảng ±131 lấn kênh center khoảng -17..3). Activation INT16 không giúp và chậm hơn 20 lần (không có XNNPACK). Hướng sửa dự kiến: giữ layer cuối FP32, tách output head, calibration bằng ảnh tài liệu.
