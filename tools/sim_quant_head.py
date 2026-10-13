# Mốc 5, bước 1: mô phỏng quantize INT8 riêng ở output head của M-LSD, để kiểm tra giả thuyết
# "các kênh output dùng chung một scale nên kênh center bị mất độ chính xác".
#
# Cách làm: chạy model FP32, lấy tensor head 16 kênh (256x256x16, tensor "Decoder/resize", ngay trước
# bước cắt kênh), ép tensor đó về int8 rồi đổi ngược lại (fake quantization) theo nhiều cách chọn scale,
# tự tính lại pts / scores / vmap bằng numpy giống hệt phần cuối của model, rồi đưa qua pred_lines /
# pred_squares của repo gốc. Mọi layer khác giữ FP32 -> chỉ đo đúng ảnh hưởng của việc quantize head.
#
# Phần cuối của model (đọc từ file .tflite):
#   center = head[..., 7]   -> sigmoid -> max_pool 3x3 (giữ điểm cực đại) -> top-k 200 -> pts, scores
#   vmap   = head[..., 8:12]
#
# Chạy từ thư mục gốc repo (cần data/calib.txt, data/eval.csv từ tools/make_doc_split.py):
#   python tools/sim_quant_head.py
import csv
import sys

import cv2
import numpy as np
import tensorflow as tf

sys.path.insert(0, "third_party/mlsd-official")
from utils import pred_lines, pred_squares

MODEL = "third_party/mlsd-official/tflite_models/M-LSD_512_tiny_fp32.tflite"
HEAD = 156          # chỉ số tensor head 16 kênh trong file .tflite
CENTER, VMAP = 7, slice(8, 12)
N_CALIB = 120       # số khung calibration dùng để đo min/max (đủ ổn định, chạy nhanh)

interp = tf.lite.Interpreter(model_path=MODEL, experimental_preserve_all_tensors=True)
interp.allocate_tensors()
inp = interp.get_input_details()
outs = interp.get_output_details()


def load(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def run_fp32(img):
    # Giống pred_lines: resize INTER_AREA về 512, thêm kênh 1. Trả về head + 3 output thật của model
    x = np.concatenate([cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA),
                        np.ones([512, 512, 1])], axis=-1)[None].astype(np.float32)
    interp.set_tensor(inp[0]["index"], x)
    interp.invoke()
    return interp.get_tensor(HEAD)[0].copy(), [interp.get_tensor(o["index"]) for o in outs]


def decode(head, quant_sigmoid=False):
    # Phần cuối của model viết bằng numpy: head (256, 256, 16) -> pts (200, 2), scores (200,), vmap (256, 256, 4)
    s = 1 / (1 + np.exp(-head[..., CENTER].astype(np.float32)))
    if quant_sigmoid:
        # Trong model INT8 thật, output sigmoid luôn có scale 1/256 (TFLite cố định)
        s = np.round(s * 256) / 256
    pad = np.pad(s, 1, constant_values=-np.inf)
    mp = np.max([pad[dy:dy + 256, dx:dx + 256] for dy in range(3) for dx in range(3)], axis=0)   # max_pool 3x3 'same'
    s = (s * (s == mp)).reshape(-1)
    idx = np.argsort(-s, kind="stable")[:200]            # top-k: bằng nhau thì chỉ số nhỏ trước (giống TF)
    pts = np.stack([idx // 256, idx % 256], axis=1).astype(np.int32)
    return pts, s[idx].astype(np.float32), head[..., VMAP].astype(np.float32)


class FakeInterpreter:
    # pred_lines / pred_squares của repo gốc tự gọi interpreter; giả lập để chúng dùng output đã quantize
    def __init__(self, pts, scores, vmap):
        self.t = {outs[0]["index"]: pts[None], outs[1]["index"]: scores[None], outs[2]["index"]: vmap[None]}
    def set_tensor(self, *a): pass
    def invoke(self): pass
    def get_tensor(self, i): return self.t[i]


def fake_quant(x, lo, hi):
    # int8 bất đối xứng như TFLite: scale = (hi - lo) / 255, khoảng luôn chứa 0
    lo, hi = np.minimum(lo, 0), np.maximum(hi, 0)
    scale = (hi - lo) / 255
    zp = np.round(-128 - lo / scale)
    q = np.clip(np.round(x / scale) + zp, -128, 127)
    return ((q - zp) * scale).astype(np.float32)


# ---------- 1. Calibration: đo min / max từng kênh của head trên ảnh tài liệu ----------
calib_paths = [l.strip() for l in open("data/calib.txt")][::360 // N_CALIB]
lo = np.full(16, np.inf)
hi = np.full(16, -np.inf)
for p in calib_paths:
    h, _ = run_fp32(load(p))
    lo = np.minimum(lo, h.reshape(-1, 16).min(0))
    hi = np.maximum(hi, h.reshape(-1, 16).max(0))
print(f"Calibration tren {len(calib_paths)} khung SmartDoc:")
print(f"  kenh center (7): {lo[7]:7.2f} .. {hi[7]:6.2f}")
print(f"  kenh vmap (8-11): {lo[8:12].min():7.2f} .. {hi[8:12].max():6.2f}")
print(f"  ca 16 kenh:       {lo.min():7.2f} .. {hi.max():6.2f}")

used = list(range(7, 12))
# Khoảng có ý nghĩa (đo ở bước khám phá): vmap tại các điểm center chỉ khoảng ±111, logit center của các điểm
# được giữ (score > 0.06) chỉ từ -2.75 trở lên. Các giá trị lớn hơn là ngoại lai ở chỗ không có đoạn thẳng
# nhưng lại kéo giãn min/max -> cắt bớt khoảng (clip) trước khi tính scale.
C_LO, V_R = -8.0, 128.0


def split_q(h, c_lo, c_hi, v_lo, v_hi):
    return np.concatenate([h[..., :7], fake_quant(h[..., 7:8], c_lo, c_hi),
                           fake_quant(h[..., 8:12], v_lo, v_hi), h[..., 12:]], -1)


# Các cách chọn scale cho head: (tên, hàm quantize)
VARIANTS = [
    ("FP32 (khong quantize)",               lambda h: h, False),
    ("1 scale chung 16 kenh (nhu TFLite)",   lambda h: fake_quant(h, lo.min(), hi.max()), True),
    ("1 scale chung, chi 5 kenh dung",       lambda h: fake_quant(h, lo[used].min(), hi[used].max()), True),
    ("tach: center 1 scale, vmap 1 scale",   lambda h: split_q(h, lo[7], hi[7], lo[8:12].min(), hi[8:12].max()), True),
    ("moi kenh 1 scale (per-channel)",       lambda h: fake_quant(h, lo, hi), True),
    ("1 scale chung, cat khoang +-128",      lambda h: fake_quant(h, -V_R, V_R), True),
    ("tach + cat khoang (c -8..max, v +-128)", lambda h: split_q(h, C_LO, hi[7], -V_R, V_R), True),
]


# ---------- 2. Đánh giá trên tập đánh giá ----------
def match_ratio(ref, got, tol):
    # Tỉ lệ đoạn FP32 còn được tìm thấy (2 đầu mút lệch < tol px, cho phép đảo chiều)
    if len(ref) == 0:
        return 1.0
    if len(got) == 0:
        return 0.0
    ref, got = np.asarray(ref)[:, None, :], np.asarray(got)[None, :, :]
    d1 = np.maximum(np.hypot(*(ref[..., :2] - got[..., :2]).transpose(2, 0, 1)), np.hypot(*(ref[..., 2:] - got[..., 2:]).transpose(2, 0, 1)))
    d2 = np.maximum(np.hypot(*(ref[..., :2] - got[..., 2:]).transpose(2, 0, 1)), np.hypot(*(ref[..., 2:] - got[..., :2]).transpose(2, 0, 1)))
    return float(np.mean(np.minimum(d1, d2).min(1) < tol))


def iou(q1, q2, h, w):
    m1 = np.zeros((h, w), np.uint8)
    m2 = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m1, [np.round(q1).astype(np.int32)], 1)
    cv2.fillPoly(m2, [np.round(q2).astype(np.int32)], 1)
    u = np.count_nonzero(m1 | m2)
    return np.count_nonzero(m1 & m2) / u if u else 0.0


def safe(f, *a, **k):
    try:
        return f(*a, **k)
    except IndexError:          # repo gốc crash khi không có đoạn nào
        return None


rows = list(csv.DictReader(open("data/eval.csv")))
stats = {name: {"keep": [], "iou": [], "nlines": []} for name, _, _ in VARIANTS}
checked = False
for r in rows:
    img = load(r["image_path"])
    H, W = img.shape[:2]
    gt = np.array([[float(r[f"{c}_x"]), float(r[f"{c}_y"])] for c in ["tl", "tr", "br", "bl"]])
    head, real = run_fp32(img)
    if not checked:
        # Kiểm tra phần decode bằng numpy khớp với phần cuối thật của model
        p, s, v = decode(head)
        assert (p == real[0][0]).all() and np.allclose(s, real[1][0]) and np.allclose(v, real[2][0]), "decode khong khop model!"
        checked = True
    ref_lines = None
    for name, qf, qsig in VARIANTS:
        fi = FakeInterpreter(*decode(qf(head), quant_sigmoid=qsig))
        lines = safe(pred_lines, img, fi, inp, outs, input_shape=[512, 512], score_thr=0.1, dist_thr=20.0)
        lines = [] if lines is None else lines
        if ref_lines is None:
            ref_lines = lines
        sq = safe(pred_squares, img, fi, inp, outs, input_shape=[512, 512])
        squares = [] if sq is None else sq[1]
        st = stats[name]
        st["nlines"].append(len(lines))
        # ngưỡng: 1.5 ô lưới 256 tính ra px ảnh gốc (1 ô = 2 * W / 512 px, ảnh 1920 rộng -> 7.5 px)
        st["keep"].append(match_ratio(ref_lines, lines, tol=1.5 * 2 * max(W, H) / 512))
        st["iou"].append(iou(np.asarray(squares[0], float).reshape(4, 2), gt, H, W) if len(squares) else 0.0)

print(f"\nKet qua tren {len(rows)} khung danh gia (chi quantize head, cac layer khac FP32):")
print(f"{'cach chon scale':38s} | so doan TB | giu doan FP32 | IoU TB | dung (IoU>=0.9)")
for name, _, _ in VARIANTS:
    st = stats[name]
    print(f"{name:38s} | {np.mean(st['nlines']):10.1f} | {np.mean(st['keep']) * 100:12.1f}% | {np.mean(st['iou']):.3f} | {np.mean(np.array(st['iou']) >= 0.9) * 100:5.1f}%")
