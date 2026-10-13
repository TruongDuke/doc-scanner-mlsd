# Đo chất lượng tìm khung tài liệu trên tập đánh giá (data/eval.csv, sinh bằng tools/make_doc_split.py):
# chạy model + pred_squares của repo gốc, lấy tứ giác tốt nhất, so với đáp án 4 góc.
#
# Chạy từ thư mục gốc repo:
#   python tools/eval_quads.py                                  # model FP32 mặc định
#   python tools/eval_quads.py --model path/to/model_int8.tflite
# Ra: bảng tổng kết + data/eval_results/<tên model>.csv (kết quả từng khung)
#
# Thước đo:
#   IoU: diện tích phần chung / diện tích phần hợp của tứ giác tìm được và tứ giác đáp án (0..1),
#        giống thước đo của cuộc thi SmartDoc. Không tìm được tứ giác nào -> IoU = 0.
#   Sai số góc: trung bình khoảng cách 4 góc tới 4 góc đáp án (px, ảnh 1920x1080),
#        thử mọi cách xoay vòng thứ tự góc và lấy cách khớp nhất.
import argparse
import csv
import os
import sys
import time
from collections import defaultdict

import cv2
import numpy as np
import tensorflow as tf

sys.path.insert(0, "third_party/mlsd-official")
from utils import pred_squares

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="third_party/mlsd-official/tflite_models/M-LSD_512_tiny_fp32.tflite")
parser.add_argument("--eval", default="data/eval.csv")
parser.add_argument("--good-iou", type=float, default=0.9, help="IoU từ mức này trở lên coi là tìm đúng")
args = parser.parse_args()

interpreter = tf.lite.Interpreter(model_path=args.model)
interpreter.allocate_tensors()
inp = interpreter.get_input_details()
out = interpreter.get_output_details()


def iou(q1, q2, h, w):
    # Vẽ 2 tứ giác thành mặt nạ trên ảnh rồi đếm điểm ảnh: đơn giản, đúng cả với tứ giác lõm / tự cắt
    m1 = np.zeros((h, w), np.uint8)
    m2 = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m1, [np.round(q1).astype(np.int32)], 1)
    cv2.fillPoly(m2, [np.round(q2).astype(np.int32)], 1)
    union = np.count_nonzero(m1 | m2)
    return np.count_nonzero(m1 & m2) / union if union else 0.0


def corner_error(q, gt):
    # Thứ tự góc của pred_squares có thể lệch vòng so với đáp án: thử 4 cách xoay vòng (và chiều ngược lại)
    best = np.inf
    for qq in (q, q[::-1]):
        for s in range(4):
            best = min(best, np.linalg.norm(np.roll(qq, s, axis=0) - gt, axis=1).mean())
    return best


rows = list(csv.DictReader(open(args.eval)))
results = []
t0 = time.time()
for k, r in enumerate(rows):
    img = cv2.cvtColor(cv2.imread(r["image_path"]), cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    gt = np.array([[float(r[f"{c}_x"]), float(r[f"{c}_y"])] for c in ["tl", "tr", "br", "bl"]])
    try:
        _, squares, _, _ = pred_squares(img, interpreter, inp, out, input_shape=[512, 512])
    except IndexError:                       # repo gốc crash khi không có đoạn nào
        squares = []
    if len(squares) == 0:
        res = {"iou": 0.0, "corner_err": np.nan, "found": 0}
    else:
        q = np.asarray(squares[0], dtype=np.float64).reshape(4, 2)
        res = {"iou": iou(q, gt, h, w), "corner_err": corner_error(q, gt), "found": 1}
    res.update({"image_path": r["image_path"], "bg_name": r["bg_name"], "model_name": r["model_name"],
                "n_squares": len(squares)})
    results.append(res)
    if (k + 1) % 20 == 0:
        print(f"  {k + 1}/{len(rows)} khung...", flush=True)

# Lưu kết quả từng khung
os.makedirs("data/eval_results", exist_ok=True)
name = os.path.splitext(os.path.basename(args.model))[0]
with open(f"data/eval_results/{name}.csv", "w") as f:
    wr = csv.DictWriter(f, fieldnames=["image_path", "bg_name", "model_name", "found", "n_squares", "iou", "corner_err"])
    wr.writeheader()
    wr.writerows(results)


def summary(rs):
    ious = np.array([r["iou"] for r in rs])
    errs = np.array([r["corner_err"] for r in rs if r["found"]])
    return (f"{len(rs):4d} khung | IoU TB {ious.mean():.3f} | dung (IoU>={args.good_iou}) {np.mean(ious >= args.good_iou) * 100:5.1f}% "
            f"| khong tim thay {sum(1 - r['found'] for r in rs):3d} | sai so goc TB {np.mean(errs) if len(errs) else float('nan'):6.1f} px "
            f"(trung vi {np.median(errs) if len(errs) else float('nan'):5.1f})")


print(f"\nModel: {args.model}  ({time.time() - t0:.0f} s)")
print("Tat ca    : " + summary(results))
by_bg = defaultdict(list)
for r in results:
    by_bg[r["bg_name"]].append(r)
for bg in sorted(by_bg):
    print(f"{bg}: " + summary(by_bg[bg]))
print(f"Chi tiet tung khung: data/eval_results/{name}.csv")
