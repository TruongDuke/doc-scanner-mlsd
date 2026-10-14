# Chạy một model .tflite trên tập đánh giá (data/eval.csv) bằng runtime LiteRT mới (ai_edge_litert),
# lưu output thô (pts, scores, vmap) ra data/raw/<tên model>.npz để tools/eval_quads.py --raw chấm điểm.
# Tách bước này ra vì môi trường TensorFlow 2.15 (dùng cho pred_squares) có thể đọc sai model
# do công cụ mới (litert-torch, ai-edge-quantizer) tạo ra.
#
# Chạy từ thư mục gốc repo:  <python có ai_edge_litert> tools/run_raw.py data/models/lhwcv_fp32.tflite [danh sách .csv]
# Mặc định chạy trên data/eval.csv; nếu đưa danh sách khác (vd. data/tune.csv) thì tên file ra thêm hậu tố _tune.
import csv
import os
import sys
import time

import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter

model = sys.argv[1]
rows = list(csv.DictReader(open(sys.argv[2] if len(sys.argv) > 2 else "data/eval.csv")))

it = Interpreter(model_path=model, num_threads=4)
it.allocate_tensors()
inp = it.get_input_details()[0]
outs = it.get_output_details()

def to_input(x, d):
    # Model INT8 do ai_edge_quantizer tạo có đầu vào / ra int8: đổi số theo scale, zero_point của tensor
    if d["dtype"] == np.float32:
        return x
    s, z = d["quantization"]
    return np.clip(np.round(x / s) + z, -128, 127).astype(d["dtype"])


def from_output(y, d):
    if d["dtype"] != np.int8:
        return y
    s, z = d["quantization"]
    return ((y.astype(np.float32) - z) * s).astype(np.float32)


print("vao:", inp["dtype"].__name__, inp["quantization"], "| ra:", [(d["dtype"].__name__, d["quantization"]) for d in outs])
pts, scores, vmap = [], [], []
times = []
for r in rows:
    img = cv2.cvtColor(cv2.imread(r["image_path"]), cv2.COLOR_BGR2RGB)
    x = np.concatenate([cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA),
                        np.ones([512, 512, 1])], axis=-1)[None].astype(np.float32)
    it.set_tensor(inp["index"], to_input(x, inp))
    t0 = time.perf_counter()
    it.invoke()
    times.append((time.perf_counter() - t0) * 1000)
    o = [from_output(it.get_tensor(d["index"]), d) for d in outs]
    pts.append(o[0][0])
    scores.append(o[1][0])
    vmap.append(o[2][0])

os.makedirs("data/raw", exist_ok=True)
name = os.path.splitext(os.path.basename(model))[0] + ("" if len(sys.argv) < 3 else "_" + os.path.splitext(os.path.basename(sys.argv[2]))[0])
np.savez(f"data/raw/{name}.npz", image_path=np.array([r["image_path"] for r in rows]),
         pts=np.array(pts), scores=np.array(scores), vmap=np.array(vmap))
print(f"Da luu data/raw/{name}.npz ({len(rows)} khung, model {np.median(times):.1f} ms/khung tren Mac, 4 thread)")
