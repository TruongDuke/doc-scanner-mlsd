# doc-scanner-mlsd

App scan tài liệu chạy offline trên Android, dùng model M-LSD (phát hiện đoạn thẳng) để tìm 4 góc tờ giấy, và so sánh các bản model FP32 / FP16 / INT8 trên điện thoại thật.

Dự án cá nhân, nối tiếp phần luyện tập quantization ở repo [Quantization](https://github.com/TruongDuke/Quantization).

## Tính năng dự kiến

App:
1. Mở camera, tự tìm tờ giấy và vẽ khung 4 góc theo thời gian thực.
2. Chụp và nắn thẳng tài liệu (perspective transform).
3. Chỉnh 4 góc bằng tay nếu app tìm sai.
4. Lưu ảnh vào máy.
5. Chạy offline hoàn toàn.

Phần kỹ thuật:
6. Chọn model trong app: FP32, FP16, INT8, INT8 đã sửa.
7. Chọn nơi chạy: CPU (XNNPACK), GPU, NNAPI.
8. Hiện thời gian model, thời gian post-processing, FPS ngay trên màn hình.
9. Chế độ benchmark trên bộ ảnh mẫu, xuất bảng so sánh tốc độ và độ chính xác.

## Lộ trình

| Mốc | Việc | Trạng thái |
|---|---|---|
| 1 | `pred_lines` bằng Kotlin, kiểm tra với kết quả Python (golden test) | Xong |
| 2 | `pred_squares` bằng Kotlin, có golden test | Xong |
| 3 | App Android: chọn ảnh có sẵn, detect box, vẽ lên | Xong |
| 4 | Camera chạy liên tục, chụp, nắn thẳng, chỉnh góc, lưu ảnh | Đang làm (đã có camera + khung trực tiếp, chụp, chỉnh góc, nắn thẳng, lưu) |
| 5 | Sửa quantization M-LSD để bản INT8 không bị hỏng | Xong phần chính (INT8 đã sửa: IoU 0.751 so với FP32 0.802) |
| 6 | Chọn model / CPU / GPU, hiện số đo, chế độ benchmark | |
| 7 | Viết lại quá trình: đã thử gì, sai ở đâu, sửa thế nào | |

## Cấu trúc

```
postprocess/src/           pred_lines, pred_squares, resize INTER_AREA viết bằng Kotlin (app Android dùng chung code này)
postprocess/test/          golden test: so kết quả Kotlin với Python
android/                   app Android (Jetpack Compose + LiteRT)
tools/make_golden.py       sinh dữ liệu golden từ model chính thức (output thô + kết quả pred_lines / pred_squares của repo gốc)
tools/make_resize_golden.py  dữ liệu kiểm tra hàm resize
tools/push_golden.sh       chép golden vào app trên điện thoại, cho nút "Kiểm tra golden"
samples/                   ảnh thử
golden/                    dữ liệu golden (sinh ra, không đưa lên git)
third_party/mlsd-official  clone repo M-LSD chính thức (không đưa lên git)
```

## Chạy golden test (mốc 1, 2)

```bash
# 1. Lấy repo M-LSD chính thức
git clone https://github.com/navervision/mlsd.git third_party/mlsd-official

# 2. Sinh dữ liệu golden (cần Python 3.11 + tensorflow + opencv-python)
pip install -r tools/requirements.txt
python tools/make_golden.py

# 3. Dịch và chạy bản Kotlin
kotlinc postprocess/src/PredLines.kt postprocess/test/GoldenLines.kt -include-runtime -d build/predlines.jar
java -jar build/predlines.jar
kotlinc postprocess/src/PredSquares.kt postprocess/test/GoldenSquares.kt -include-runtime -d build/predsquares.jar
java -jar build/predsquares.jar

# 4. Kiểm tra trên điện thoại / máy ảo (app bản debug đã cài): rồi bấm "Kiểm tra golden" trong app
tools/push_golden.sh
```

## Nguồn

- Model và code gốc: [navervision/mlsd](https://github.com/navervision/mlsd) (Apache License 2.0)
- Paper: [Towards Light-weight and Real-time Line Segment Detection](https://arxiv.org/abs/2106.00186)
