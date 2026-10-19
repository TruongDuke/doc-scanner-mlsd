# Mô phỏng làm tròn INT8 trên model ONNX đọc chữ (fake quant, tính bằng float), bật / tắt riêng TRỌNG SỐ hoặc ACTIVATION
# của từng lớp, để chẩn đoán lớp nào / phía nào gây lỗi khi quantize (giống tools/sim_quant_head.py của M-LSD).
#   - trọng số: per-channel đối xứng [-127, 127] (như ai_edge_quantizer / TFLite)
#   - activation: bất đối xứng 256 mức, thước = quy tắc p75 (percentile 75 của min / max từng ảnh calibration) hoặc
#     khoảng cho sẵn; chèn các node Div / Round / Clip / Mul ngay sau tensor -> mọi lớp sau nhận giá trị đã làm tròn
#   - ảnh: mỗi dòng đệm tới cỡ độ rộng như TFLite (320 / 640 / 1280 / 2000); calibration = data/ocr_calib/calib_w*.npy
# Tensor ONNX ứng với op TFLite (đầu model, TFLite gộp BN / bias / hệ số LAB trước activation vào conv):
#   op0-2 (khối b0): đầu vào x, conv đầu + BN "p2o.pd_op.batch_norm_.0.0", depthwise + bias + LAB "Add.3"; trọng số conv2d_0, conv2d_136
#   op6-9 (khối b2): conv 1x1 + bias + LAB "Add.9", hardswish "p2o.pd_op.hardswish.1.0", LAB sau "Mul.7", "Add.11"; trọng số conv2d_137
import sys
import unicodedata

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper

sys.path.insert(0, "tools")
from ocr_eval import ctc_decode, levenshtein, load_dict, prep, strip_marks

MODEL = "data/models/ocr/vi_ft2_e4.onnx"
BUCKETS = [320, 640, 1280, 2000]
CHARS = load_dict(None, "data/ocr_eval/vi_dict.txt")
B0 = dict(acts=["x", "p2o.pd_op.batch_norm_.0.0", "Add.3"], weights=["conv2d_0.w_0", "conv2d_136.w_0"])
B2 = dict(acts=["Add.9", "p2o.pd_op.hardswish.1.0", "Mul.7", "Add.11"], weights=["conv2d_137.w_0"])


def bucket(bgr):
    need = int(np.ceil(48 * bgr.shape[1] / bgr.shape[0]))
    return next((b for b in BUCKETS if b >= need), BUCKETS[-1])


def fq_weight(w):
    # per-channel (trục 0 = kênh ra) đối xứng int8
    s = np.abs(w.reshape(w.shape[0], -1)).max(1) / 127
    s = np.where(s == 0, 1, s).reshape((-1,) + (1,) * (w.ndim - 1))
    return (np.clip(np.round(w / s), -127, 127) * s).astype(np.float32)


def act_stats(names, n=200):
    # min / max từng ảnh calibration (mọi cỡ) ở các tensor cần đo -> (mins, maxs) mảng [số ảnh]
    m = onnx.shape_inference.infer_shapes(onnx.load(MODEL))
    vi = {v.name: v for v in list(m.graph.value_info) + list(m.graph.input) + list(m.graph.output)}
    del m.graph.output[:]
    m.graph.output.extend([vi[k] for k in names])
    sess = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
    st = {k: ([], []) for k in names}
    for b in BUCKETS:
        x = np.load(f"data/ocr_calib/calib_w{b}.npy", mmap_mode="r")[:n]
        for i in range(len(x)):
            outs = sess.run(names, {"x": np.ascontiguousarray(np.array(x[i:i + 1]).transpose(0, 3, 1, 2))})
            for k, o in zip(names, outs):
                st[k][0].append(float(o.min())); st[k][1].append(float(o.max()))
    return {k: (np.array(a), np.array(b)) for k, (a, b) in st.items()}


def rule(mins, maxs, how="p75"):
    if how == "minmax":
        return float(mins.min()), float(maxs.max())
    q = float(how[1:])
    return float(np.percentile(mins, 100 - q)), float(np.percentile(maxs, q))


def build(acts=None, weights=()):
    # acts: {tên tensor: (lo, hi)} -> chèn fake quant activation; weights: tên initializer -> fake quant trọng số
    m = onnx.load(MODEL)
    for init in m.graph.initializer:
        if init.name in weights:
            init.CopyFrom(numpy_helper.from_array(fq_weight(numpy_helper.to_array(init)), init.name))
    new_nodes = []
    for k, (lo, hi) in (acts or {}).items():
        lo, hi = min(lo, 0.0), max(hi, 0.0)
        s = (hi - lo) / 255
        zp = float(np.round(-128 - lo / s))
        c = lambda name, v: numpy_helper.from_array(np.array(v, np.float32), name)
        p = f"fq_{k}"
        m.graph.initializer.extend([c(p + "_s", s), c(p + "_zp", zp), c(p + "_lo", -128.0), c(p + "_hi", 127.0)])
        for node in m.graph.node:                       # mọi node đang dùng tensor k -> dùng bản đã làm tròn
            for j, inp in enumerate(node.input):
                if inp == k:
                    node.input[j] = p + "_out"
        new_nodes += [helper.make_node("Div", [k, p + "_s"], [p + "_a"]), helper.make_node("Add", [p + "_a", p + "_zp"], [p + "_b"]),
                      helper.make_node("Round", [p + "_b"], [p + "_c"]), helper.make_node("Clip", [p + "_c", p + "_lo", p + "_hi"], [p + "_d"]),
                      helper.make_node("Sub", [p + "_d", p + "_zp"], [p + "_e"]), helper.make_node("Mul", [p + "_e", p + "_s"], [p + "_out"])]
    # chèn node fake quant ngay sau node tạo ra tensor (hoặc đầu đồ thị nếu là đầu vào)
    nodes = list(m.graph.node)
    for k in (acts or {}):
        grp = [n for n in new_nodes if n.output[0].startswith(f"fq_{k}_")]
        pos = 0 if k == "x" else next(i for i, n in enumerate(nodes) if k in n.output) + 1
        nodes[pos:pos] = grp
    del m.graph.node[:]
    m.graph.node.extend(nodes)
    return ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])


_DATA = {}


def evaluate(sess, data="myscans"):
    if data not in _DATA:
        root, lab = {"myscans": ("data/ocr/myscans", "labels.txt"), "gov": ("data/ocr/govdocs", "cam.txt")}[data]
        items = [l.rstrip("\n").split("\t", 1) for l in open(f"{root}/{lab}", encoding="utf-8")]
        _DATA[data] = [(cv2.imread(f"{root}/{p}"), gt) for p, gt in items]
    e = n = enm = nnm = 0
    for im, gt in _DATA[data]:
        pr = unicodedata.normalize("NFC", ctc_decode(sess.run(None, {"x": prep(im, bucket(im))})[0][0], CHARS))
        e += levenshtein(pr, gt); n += len(gt)
        enm += levenshtein(strip_marks(pr), strip_marks(gt)); nnm += len(strip_marks(gt))
    return e / n * 100, enm / nnm * 100


if __name__ == "__main__":
    st = act_stats(B0["acts"] + B2["acts"])
    rng = {k: rule(*v) for k, v in st.items()}
    rng["x"] = (-1.0, 1.0)
    for k, (lo, hi) in rng.items():
        print(f"  thuoc p75 {k:28s} {lo:8.2f} .. {hi:8.2f}")
    exps = [("FP32 (khong lam tron gi)", {}, ()),
            ("b0: trong so + activation", {k: rng[k] for k in B0["acts"]}, B0["weights"]),
            ("b0: CHI trong so", {}, B0["weights"]),
            ("b0: CHI activation", {k: rng[k] for k in B0["acts"]}, ()),
            ("b2: trong so + activation", {k: rng[k] for k in B2["acts"]}, B2["weights"]),
            ("b2: CHI trong so", {}, B2["weights"]),
            ("b2: CHI activation", {k: rng[k] for k in B2["acts"]}, ())]
    for name, acts, ws in exps:
        c, cn = evaluate(build(acts, ws))
        print(f"{name:30s} CER {c:5.2f}% | bo dau {cn:5.2f}%", flush=True)
