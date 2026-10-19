# Cân bằng kênh (cross-layer equalization) ở đầu model đọc chữ, trước khi quantize INT8. Kết quả FP32 KHÔNG đổi.
# Chẩn đoán (tools/ocr_sim_quant.py, 19/10): đầu ra conv đầu + BN có 16 kênh chênh nhau 4.1 lần (±1.3 .. ±13.7), dùng chung
# 1 thước int8 -> kênh nhỏ bị vạch thô; riêng tensor này làm CER 4.04 -> 4.90%. Giữa nó và depthwise không có hàm phi tuyến:
#   BN gamma_c, beta_c  x s_c   (đầu ra kênh c nhân s_c)
#   trọng số depthwise kênh c  / s_c   (bù lại -> đầu ra depthwise y hệt)
#   s_c = r_max / r_c, r_c = max|x| kênh c (p75 qua 100 ảnh calibration mỗi cỡ) -> mọi kênh có khoảng như kênh lớn nhất.
# Trọng số depthwise làm tròn theo từng kênh nên chia s_c không hại trọng số.
# Chạy (thư mục gốc repo, .venv-tf):  .venv-tf/bin/python tools/ocr_cle.py --onnx data/models/ocr/vi_ft2_e4.onnx --out data/models/ocr/vi_ft2_e4_cle.onnx
import argparse

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

BUCKETS = [320, 640, 1280, 2000]
BN_OUT, BN_W, BN_B, DW_W = "p2o.pd_op.batch_norm_.0.0", "batch_norm2d_0.w_0", "batch_norm2d_0.b_0", "conv2d_136.w_0"

ap = argparse.ArgumentParser()
ap.add_argument("--onnx", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=100, help="số ảnh calibration mỗi cỡ")
args = ap.parse_args()

m = onnx.load(args.onnx)
users = [n.op_type for n in m.graph.node if BN_OUT in n.input]
assert users == ["Conv"], f"dau ra BN phai chi vao 1 conv (depthwise), thay {users}"

# Khoảng từng kênh của đầu ra BN
mm = onnx.shape_inference.infer_shapes(onnx.load(args.onnx))
vi = {v.name: v for v in mm.graph.value_info}
del mm.graph.output[:]
mm.graph.output.extend([vi[BN_OUT]])
sess = ort.InferenceSession(mm.SerializeToString(), providers=["CPUExecutionProvider"])
per = []
for b in BUCKETS:
    x = np.load(f"data/ocr_calib/calib_w{b}.npy", mmap_mode="r")[:args.n]
    for i in range(len(x)):
        o = sess.run(None, {"x": np.ascontiguousarray(np.array(x[i:i + 1]).transpose(0, 3, 1, 2))})[0]
        per.append(np.abs(o[0]).reshape(o.shape[1], -1).max(1))
r = np.percentile(np.stack(per), 75, axis=0)
s = r.max() / r
print("he so can bang tung kenh:", np.round(s, 2))

init = {i.name: i for i in m.graph.initializer}
get = lambda k: numpy_helper.to_array(init[k])
put = lambda k, a: init[k].CopyFrom(numpy_helper.from_array(a.astype(np.float32), k))
put(BN_W, get(BN_W) * s)
put(BN_B, get(BN_B) * s)
put(DW_W, get(DW_W) / s.reshape(-1, 1, 1, 1))
onnx.save(m, args.out)

# Kiểm tra: đầu ra FP32 không đổi
xin = np.ascontiguousarray(np.load("data/ocr_calib/calib_w640.npy", mmap_mode="r")[:4].transpose(0, 3, 1, 2))
a = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"]).run(None, {"x": xin})[0]
c = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"]).run(None, {"x": xin})[0]
print(f"-> {args.out} | lech dau ra so voi ban goc: {np.abs(a - c).max():.1e}")
