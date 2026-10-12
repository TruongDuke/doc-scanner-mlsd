# Hướng dẫn cho agent

Trả lời bằng tiếng Việt.

## Dự án

App scan tài liệu chạy offline trên Android, dùng M-LSD tiny 512 (phát hiện đoạn thẳng) để tìm 4 góc tờ giấy, rồi so sánh các bản model FP32 / FP16 / INT8 và CPU / GPU trên điện thoại thật. Tính năng và lộ trình 7 mốc: xem `README.md`.

Đây là dự án cá nhân để học, nối tiếp phần luyện tập ở repo https://github.com/TruongDuke/Quantization (thư mục `5-mlsd/`: convert M-LSD sang TFLite, quantize, và `5-mlsd/ALGORITHM.md` là phần giải thích thuật toán post-processing do user tự viết).

## Cách làm việc với user (quan trọng)

- **User tự viết phần cốt lõi**: post-processing (`predLines`, `predSquares`), phần sửa quantization. Agent KHÔNG viết thay phần này. Agent chỉ: chia nhỏ bước, đưa khung có chỗ trống `???`, gợi ý, review code user viết và chỉ ra chỗ sai.
- Agent được viết phần phụ trợ: công cụ sinh dữ liệu test, phần đọc file / so sánh kết quả, cấu hình build Gradle, giao diện Android, CameraX, code vẽ.
- Đưa code để user tự dán và tự chạy, kèm giải thích từng phần. Không tự chạy thí nghiệm thay user trừ khi user yêu cầu rõ ("chạy luôn đi").
- User chưa vững Kotlin/Android, giải thích đơn giản, từng bước nhỏ.

## Tình trạng hiện tại: mốc 2

Mốc 1 xong: `predLines` trong `postprocess/PredLines.kt` đạt **6/6 PASS** golden test (lệch max 0.0001 px). Lần này agent viết thân hàm theo yêu cầu rõ của user, user review lại.

Mốc 2: viết `predSquares` (port `pred_squares`, `third_party/mlsd-official/utils.py` từ dòng 50). Việc trước tiên: mở rộng `tools/make_golden.py` để sinh golden cho `pred_squares`. Chưa chốt ai viết `predSquares` (mặc định vẫn theo quy tắc: user tự viết, agent đưa khung), hỏi user trước khi bắt đầu.

Lưu ý: `pred_squares` KHÔNG gọi `pred_lines`. Nó lặp lại bước lọc nhưng với score **0.06** (dist vẫn 20), giữ toạ độ đoạn trên **lưới 256** (chưa đổi về ảnh gốc), và lưu thêm `junc_list` (các trung điểm (x, y)). Nên `predLines` hiện tại chưa dùng lại nguyên được, cần tách hàm hoặc thêm tham số.

Chạy golden test từ thư mục gốc repo:
```bash
kotlinc postprocess/PredLines.kt -include-runtime -d build/predlines.jar && java -jar build/predlines.jar
```

## Dữ liệu golden (`golden/`, không đưa lên git)

Sinh bằng `tools/make_golden.py` từ model chính thức `third_party/mlsd-official/tflite_models/M-LSD_512_tiny_fp32.tflite` và hàm `pred_lines` của repo gốc (`third_party/mlsd-official/utils.py`, dòng 12-47). Mỗi ảnh trong `samples/` + 1 ảnh trơn `blank`:
- `{name}_pts.bin`: int32 little-endian, (200, 2), thứ tự **(y, x)**, lưới 256x256
- `{name}_scores.bin`: float32, (200,)
- `{name}_vmap.bin`: float32, (256, 256, 4) làm phẳng: `vmap[(y * 256 + x) * 4 + c]`, c = (dx_start, dy_start, dx_end, dy_end)
- `{name}_lines.bin`: float32, (n, 4) = (x1, y1, x2, y2) trên ảnh gốc, kết quả mong đợi
- `meta.json`: tên, h, w, số đoạn

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

## Bối cảnh quantization (cho mốc 5)

Từ thí nghiệm ở repo Quantization (`5-mlsd/BAO_CAO.md`): quantize INT8 M-LSD nhanh hơn khoảng 3.6 lần nhưng chỉ giữ được 37-60% đoạn thẳng. Nguyên nhân chính: 9 kênh output dùng chung một scale (kênh displacement khoảng ±131 lấn kênh center khoảng -17..3). Activation INT16 không giúp và chậm hơn 20 lần (không có XNNPACK). Hướng sửa dự kiến: giữ layer cuối FP32, tách output head, calibration bằng ảnh tài liệu.
