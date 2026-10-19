# Trước khi quantize model đọc chữ: đo khoảng giá trị (activation) của TỪNG lớp trên 2 nguồn ảnh, tìm chỗ đáng lo.
#   - ảnh máy vẽ bản 2 (ứng viên calibration) và ảnh thật bộ valid (chỉ để chẩn đoán, KHÔNG dùng làm calibration)
#   - mỗi lớp: min / max, p99.99 của |x|, độ "ngoại lai" = max|x| / p99.99|x| (lớn = vài giá trị rất xa làm khoảng
#     min/max rộng ra -> 256 mức int8 thô đi)
#   - lớp cuối (điểm 238 ký tự, trước softmax): mô phỏng làm tròn int8 riêng lớp này (min/max đo trên ảnh máy vẽ),
#     chấm CER / CER bỏ dấu trên ảnh thật: lỗi dấu có tăng không
# Chạy (thư mục gốc repo):  .venv-tf/bin/python tools/ocr_act_ranges.py [--n 300]
import argparse
import os
import sys
import unicodedata

import cv2
import numpy as np
import onnx
import onnxruntime as ort

sys.path.insert(0, "tools")
from ocr_eval import ctc_decode, levenshtein, load_dict, prep, strip_marks

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="data/models/ocr/vi_ft2_e4.onnx")
ap.add_argument("--n", type=int, default=300, help="số ảnh máy vẽ")
args = ap.parse_args()

SKIP = {"Identity", "Shape", "Squeeze", "Unsqueeze", "Transpose", "Reshape", "Concat", "Slice"}   # chỉ đổi hình dạng
m = onnx.shape_inference.infer_shapes(onnx.load(args.model))
vi = {v.name: v for v in list(m.graph.value_info) + list(m.graph.output)}
watch = []                                        # (thứ tự node, phần, loại, tên tensor)
for i, n in enumerate(m.graph.node):
    if n.op_type in SKIP or n.output[0] not in vi or vi[n.output[0]].type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
        continue
    part = "CNN" if i < 380 else ("SVTR" if i < 545 else "cuoi")
    watch.append((i, part, n.op_type, n.output[0]))
LOGITS = "Add.227"                 # đầu ra lớp FC cuối, trước softmax
del m.graph.output[:]
m.graph.output.extend([vi[w[3]] for w in watch])
sess = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
names = [w[3] for w in watch]

rng = np.random.default_rng(0)


def run(images):
    st = {k: [np.inf, -np.inf, []] for k in names}
    logits = []
    for im in images:
        outs = sess.run(names, {"x": prep(im, 0)})
        for k, o in zip(names, outs):
            s = st[k]
            s[0] = min(s[0], float(o.min())); s[1] = max(s[1], float(o.max()))
            flat = np.abs(o.ravel())
            s[2].append(flat[rng.integers(0, flat.size, min(flat.size, 300))])
        logits.append(outs[names.index(LOGITS)][0])
    res = {}
    for k, (lo, hi, samp) in st.items():
        p = float(np.percentile(np.concatenate(samp), 99.99))
        res[k] = (lo, hi, p, max(abs(lo), abs(hi)) / max(p, 1e-6))
    return res, logits


syn_lines = [l.rstrip("\n").split("\t", 1) for l in open("data/ocr_train_v2_mini/train.txt", encoding="utf-8")][:args.n]
syn = [cv2.imread("data/ocr_train_v2_mini/" + p) for p, _ in syn_lines]
real_lines = [l.rstrip("\n").split("\t", 1) for l in open("data/ocr/myscans/labels.txt", encoding="utf-8")]
real = [cv2.imread("data/ocr/myscans/" + p) for p, _ in real_lines]
rs, _ = run(syn)
rr, real_logits = run(real)

print(f"{len(watch)} lop do | anh may ve: {len(syn)} | anh that (valid): {len(real)}\n")
print("10 lop ngoai lai nhat (do tren anh that); 'khop' = khoang min/max anh may ve / anh that")
order = sorted(names, key=lambda k: -rr[k][3])
idx = {w[3]: w for w in watch}
print(f"{'#node':>5} {'phan':5} {'loai':18} {'min..max (that)':>20} {'p99.99|x|':>9} {'ngoai lai':>9} {'min..max (may ve)':>20}")
for k in order[:10]:
    i, part, op, _ = idx[k]; lo, hi, p, r = rr[k]; slo, shi, _, _ = rs[k]
    print(f"{i:5d} {part:5} {op:18} {lo:9.2f}..{hi:<9.2f} {p:9.2f} {r:8.1f}x {slo:9.2f}..{shi:<9.2f}")
for part in ["CNN", "SVTR", "cuoi"]:
    ks = [w[3] for w in watch if w[1] == part]
    rat = [rr[k][3] for k in ks]
    cover = [max(abs(rr[k][0]), abs(rr[k][1])) / max(abs(rs[k][0]), abs(rs[k][1]), 1e-6) for k in ks]
    print(f"\n{part}: {len(ks)} lop | ngoai lai trung vi {np.median(rat):.1f}x, max {max(rat):.1f}x | "
          f"max|x| anh that / anh may ve: trung vi {np.median(cover):.2f}, max {max(cover):.2f}")

# Lớp cuối: làm tròn int8 riêng lớp này (khoảng min/max đo trên ảnh máy vẽ), đọc lại ảnh thật
chars = load_dict(None, "data/ocr_eval/vi_dict.txt")
lo, hi = rs[LOGITS][0], rs[LOGITS][1]
step = (hi - lo) / 255
print(f"\nLop cuoi (diem 238 ky tu): khoang may ve {lo:.2f}..{hi:.2f} -> 1 muc int8 = {step:.3f}")
gaps = []
for lg in real_logits:
    top = np.sort(lg, axis=-1)
    keep = lg.argmax(-1) != 0                      # bỏ lát "trống"
    gaps += list((top[:, -1] - top[:, -2])[keep])
gaps = np.array(gaps)
print(f"Lat co ky tu: {len(gaps)} | chenh lech diem ky tu dung va ky tu thu 2: trung vi {np.median(gaps):.2f}, "
      f"< 1 muc int8: {(gaps < step).mean() * 100:.2f}%, < 2 muc: {(gaps < 2 * step).mean() * 100:.2f}%")


def score(fake_quant):
    e = n = enm = nnm = 0
    for lg, (_, gt) in zip(real_logits, real_lines):
        x = np.clip(np.round((lg - lo) / step), 0, 255) * step + lo if fake_quant else lg
        pr = unicodedata.normalize("NFC", ctc_decode(x, chars))
        e += levenshtein(pr, gt); n += len(gt)
        enm += levenshtein(strip_marks(pr), strip_marks(gt)); nnm += len(strip_marks(gt))
    return e / n * 100, enm / nnm * 100


for name, fq in [("float", False), ("int8 rieng lop cuoi", True)]:
    c, cn = score(fq)
    print(f"  {name:22s} CER {c:.2f}% | bo dau {cn:.2f}%")
