# Chọn trọng số chấm điểm tứ giác của pred_squares cho ảnh tài liệu.
# Trọng số gốc (repo M-LSD) chọn cho ảnh nội thất; với tài liệu, app hay chọn tứ giác nhỏ bên trong tờ giấy
# (cạnh là đường kẻ / mép khối chữ). Thử lưới các giá trị w_area, w_overlap:
#   - CHỌN trên tập tune (data/tune.csv: 360 khung của 18 tài liệu calibration)
#   - KIỂM TRA trên tập đánh giá (data/eval.csv: 120 khung của 12 tài liệu khác) -> tránh "học tủ".
# Cần output thô của model trên cả 2 tập (tools/run_raw.py <model> [data/tune.csv]).
#
# Chạy từ thư mục gốc repo (môi trường TF 2.15):
#   python tools/tune_square_weights.py --model lhwcv_v3_splitclip_int8
# Nhẹ RAM: không giữ ảnh trong bộ nhớ. pred_squares chỉ cần kích thước ảnh (output model đã có sẵn trong .npz),
# nên đưa vào một ảnh đen cùng kích thước, tạo mới mỗi lần gọi.
import argparse
import csv
import itertools
import sys

import cv2
import numpy as np
import tensorflow as tf

sys.path.insert(0, "third_party/mlsd-official")
from utils import pred_squares

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="lhwcv_v3_splitclip_int8")
args = parser.parse_args()

it = tf.lite.Interpreter(model_path="third_party/mlsd-official/tflite_models/M-LSD_512_tiny_fp32.tflite")
it.allocate_tensors()
inp, out = it.get_input_details(), it.get_output_details()

BASE = {"score": 0.06, "outside_ratio": 0.28, "inside_ratio": 0.45, "w_overlap": 0.0,
        "w_degree": 1.95, "w_length": 0.0, "w_area": 1.86, "w_center": 0.14}
GRID = {"w_area": [1.86, 2.5, 3.0, 4.0, 5.0, 7.0], "w_overlap": [0.0, 0.5, 1.0]}


class FakeInterpreter:
    def __init__(self, pts, scores, vmap):
        self.t = {out[0]["index"]: pts[None], out[1]["index"]: scores[None], out[2]["index"]: vmap[None]}
    def set_tensor(self, *a): pass
    def invoke(self): pass
    def get_tensor(self, i): return self.t[i]


def iou(q1, q2, h, w):
    m1 = np.zeros((h, w), np.uint8)
    m2 = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m1, [np.round(q1).astype(np.int32)], 1)
    cv2.fillPoly(m2, [np.round(q2).astype(np.int32)], 1)
    u = np.count_nonzero(m1 | m2)
    return np.count_nonzero(m1 & m2) / u if u else 0.0


def load(split):
    rows = list(csv.DictReader(open(f"data/{split}.csv")))
    raw = np.load(f"data/raw/{args.model}{'' if split == 'eval' else '_' + split}.npz")
    # Đọc mỗi mảng 1 lần (raw["vmap"] mỗi lần truy cập sẽ đọc lại cả file .npz ~380 MB)
    paths, pts, scores, vmap = raw["image_path"], raw["pts"], raw["scores"], raw["vmap"]
    idx = {p: k for k, p in enumerate(paths)}
    h, w = cv2.imread(rows[0]["image_path"]).shape[:2]   # SmartDoc: mọi khung 1920x1080
    data = []
    for r in rows:
        k = idx[r["image_path"]]
        gt = np.array([[float(r[f"{c}_x"]), float(r[f"{c}_y"])] for c in ["tl", "tr", "br", "bl"]])
        data.append(((h, w), FakeInterpreter(pts[k], scores[k], vmap[k]), gt, r["bg_name"]))
    return data


def evaluate(data, params):
    v = []
    for (h, w), fi, gt, _ in data:
        try:
            _, sq, _, _ = pred_squares(np.zeros((h, w, 3), np.uint8), fi, inp, out, params=params)
        except IndexError:
            sq = []
        v.append(iou(np.asarray(sq[0], float).reshape(4, 2), gt, h, w) if len(sq) else 0.0)
    return np.array(v)


tune, ev = load("tune"), load("eval")
results = []
for wa, wo in itertools.product(GRID["w_area"], GRID["w_overlap"]):
    p = {**BASE, "w_area": wa, "w_overlap": wo}
    v = evaluate(tune, p)
    results.append((v.mean(), wa, wo))
    print(f"tune  w_area {wa:4.2f} w_overlap {wo:3.1f}: IoU TB {v.mean():.3f} | dung {np.mean(v >= 0.9) * 100:5.1f}%", flush=True)

best = max(results)
print(f"\nTot nhat tren tap tune: w_area {best[1]}, w_overlap {best[2]}")
bg = np.array([d[3] for d in ev])
for name, p in [("goc", BASE), ("da chon", {**BASE, "w_area": best[1], "w_overlap": best[2]})]:
    v = evaluate(ev, p)
    print(f"eval  {name:8s}: IoU TB {v.mean():.3f} | dung {np.mean(v >= 0.9) * 100:5.1f}% | nen 1-5: "
          + " ".join(f"{v[bg == b].mean():.2f}" for b in sorted(set(bg))))
