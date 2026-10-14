# Thử cách chọn lại tứ giác bằng độ tương phản ở mép (cải tiến riêng cho tài liệu, không có trong repo gốc).
#
# Vấn đề: pred_squares chấm điểm tứ giác chỉ bằng hình dạng (góc, diện tích, vị trí), nên hay chọn
# đường kẻ / mép khối chữ BÊN TRONG tờ giấy làm cạnh. Mép giấy thật thì 2 bên khác màu (giấy sáng,
# bàn tối); đường bên trong thì 2 bên đều là giấy.
# Cách thử: lấy K tứ giác điểm cao nhất của pred_squares, đo độ sáng ngay trong / ngay ngoài từng cạnh,
# chọn lại theo điểm mới. Đo IoU trên tập đánh giá SmartDoc so với cách chọn gốc.
#
# Chạy từ thư mục gốc repo (môi trường TF 2.15):
#   python tools/rerank_quads.py --raw data/raw/lhwcv_v3_splitclip_int8.npz
import argparse
import csv
import sys

import cv2
import numpy as np

sys.path.insert(0, "third_party/mlsd-official")
sys.path.insert(0, "tools")
from utils import pred_squares

parser = argparse.ArgumentParser()
parser.add_argument("--raw", default="data/raw/lhwcv_v3_splitclip_int8.npz")
parser.add_argument("--topk", type=int, default=20)
args = parser.parse_args()

import tensorflow as tf
_it = tf.lite.Interpreter(model_path="third_party/mlsd-official/tflite_models/M-LSD_512_tiny_fp32.tflite")
_it.allocate_tensors()
inp, out = _it.get_input_details(), _it.get_output_details()


class FakeInterpreter:
    def __init__(self, pts, scores, vmap):
        self.t = {out[0]["index"]: pts[None], out[1]["index"]: scores[None], out[2]["index"]: vmap[None]}
    def set_tensor(self, *a): pass
    def invoke(self): pass
    def get_tensor(self, i): return self.t[i]


def edge_contrast(gray, q, offset, n=24):
    # Với mỗi cạnh: lấy n điểm dọc cạnh, so độ sáng trung bình ngay trong và ngay ngoài (cách cạnh `offset` px)
    # trên ảnh đã làm mờ (bớt nhiễu do chữ). Trả về độ chênh của 4 cạnh (0..1).
    h, w = gray.shape
    c = q.mean(0)
    res = []
    for i in range(4):
        a, b = q[i], q[(i + 1) % 4]
        d = b - a
        nrm = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-9)
        if np.dot(nrm, (a + b) / 2 - c) < 0:      # pháp tuyến hướng ra ngoài tứ giác
            nrm = -nrm
        t = np.linspace(0.15, 0.85, n)[:, None]
        p = a + t * d
        pin = np.clip(np.round(p - nrm * offset), 0, [w - 1, h - 1]).astype(int)
        pout = np.clip(np.round(p + nrm * offset), 0, [w - 1, h - 1]).astype(int)
        diff = gray[pin[:, 1], pin[:, 0]] - gray[pout[:, 1], pout[:, 0]]
        res.append(np.median(np.abs(diff)) / 255.0)
    return np.array(res)


def iou(q1, q2, h, w):
    m1 = np.zeros((h, w), np.uint8)
    m2 = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m1, [np.round(q1).astype(np.int32)], 1)
    cv2.fillPoly(m2, [np.round(q2).astype(np.int32)], 1)
    u = np.count_nonzero(m1 | m2)
    return np.count_nonzero(m1 & m2) / u if u else 0.0


raw = np.load(args.raw)
idx = {p: k for k, p in enumerate(raw["image_path"])}
rows = list(csv.DictReader(open("data/eval.csv")))

# Mỗi cách chọn: hàm (danh sách tứ giác đã xếp theo điểm gốc, độ tương phản 4 cạnh của từng cái) -> chỉ số được chọn
METHODS = {
    "goc (pred_squares)": lambda cons: 0,
    "canh yeu nhat":      lambda cons: int(np.argmax([c.min() for c in cons])),
    "trung binh 4 canh":  lambda cons: int(np.argmax([c.mean() for c in cons])),
    "canh yeu nhat + thu hang": lambda cons: int(np.argmax([c.min() - 0.002 * k for k, c in enumerate(cons)])),
}
ious = {m: [] for m in METHODS}
for r in rows:
    bgr = cv2.imread(r["image_path"])
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]
    gray = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (0, 0), 3).astype(np.float32)
    gt = np.array([[float(r[f"{c}_x"]), float(r[f"{c}_y"])] for c in ["tl", "tr", "br", "bl"]])
    k = idx[r["image_path"]]
    try:
        _, squares, _, _ = pred_squares(img, FakeInterpreter(raw["pts"][k], raw["scores"][k], raw["vmap"][k]), inp, out)
    except IndexError:
        squares = []
    if len(squares) == 0:
        for m in METHODS:
            ious[m].append(0.0)
        continue
    cand = [np.asarray(s, np.float64).reshape(4, 2) for s in squares[:args.topk]]
    offset = 0.01 * np.hypot(W, H)               # khoảng 1% đường chéo ảnh (~22 px với ảnh 1920x1080)
    cons = [edge_contrast(gray, q, offset) for q in cand]
    for m, f in METHODS.items():
        ious[m].append(iou(cand[f(cons)], gt, H, W))

print(f"{args.raw}, top {args.topk} tu giac, {len(rows)} khung:")
bg = np.array([r["bg_name"] for r in rows])
for m, v in ious.items():
    v = np.array(v)
    per_bg = " ".join(f"{np.mean(v[bg == b]):.2f}" for b in sorted(set(bg)))
    print(f"  {m:28s} IoU TB {v.mean():.3f} | dung (>=0.9) {np.mean(v >= 0.9) * 100:5.1f}% | theo nen 1-5: {per_bg}")
