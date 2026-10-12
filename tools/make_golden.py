import os
import sys
import json
import glob
import numpy as np
import cv2
import tensorflow as tf

sys.path.insert(0, "third_party/mlsd-official")
from utils import pred_lines, pred_squares

MODEL = "third_party/mlsd-official/tflite_models/M-LSD_512_tiny_fp32.tflite"
interpreter = tf.lite.Interpreter(model_path=MODEL)
interpreter.allocate_tensors()
inp = interpreter.get_input_details()
out = interpreter.get_output_details()

def raw_outputs(img):
    # chạy model giống hệt pred_lines, lấy 3 output thô
    x = np.concatenate([cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA),
                        np.ones([512, 512, 1])], axis=-1)[None].astype(np.float32)
    interpreter.set_tensor(inp[0]["index"], x)
    interpreter.invoke()
    return [interpreter.get_tensor(o["index"])[0] for o in out]

# Ảnh thử: mọi ảnh trong samples/, cộng 1 ảnh trơn (không có đoạn thẳng nào)
cases = {}
for p in sorted(glob.glob("samples/*")):
    if p.lower().endswith((".jpg", ".jpeg", ".png")):
        cases[os.path.splitext(os.path.basename(p))[0]] = cv2.imread(p)
cases["blank"] = np.full((300, 400, 3), 128, dtype=np.uint8)

meta = []
for name, bgr in cases.items():
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pts, scores, vmap = raw_outputs(img)
    try:
        lines = pred_lines(img, interpreter, inp, out, input_shape=[512, 512], score_thr=0.1, dist_thr=20.0)
    except IndexError:                       # hàm của repo crash khi không có đoạn nào
        lines = np.zeros((0, 4))
    img.tofile(f"golden/{name}_rgb.bin")                                    # (h, w, 3) uint8 RGB, ảnh sau khi cv2 giải mã
    cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA).tofile(f"golden/{name}_input.bin")  # (512, 512, 3) uint8, đầu vào model
    pts.astype("<i4").tofile(f"golden/{name}_pts.bin")          # (200, 2) int32, thứ tự (y, x)
    scores.astype("<f4").tofile(f"golden/{name}_scores.bin")    # (200,) float32
    vmap.astype("<f4").tofile(f"golden/{name}_vmap.bin")        # (256, 256, 4) float32
    np.asarray(lines, dtype="<f4").tofile(f"golden/{name}_lines.bin")  # (n, 4) float32

    # pred_squares (mốc 2), tham số mặc định của repo gốc
    try:
        sq_segments, squares, sq_scores, inter = pred_squares(img, interpreter, inp, out, input_shape=[512, 512])
    except IndexError:                       # crash khi không có đoạn nào qua được bước lọc
        sq_segments, squares, sq_scores, inter = [], [], [], []
    np.asarray(sq_segments, dtype="<f8").reshape(-1, 4).tofile(f"golden/{name}_sq_segments.bin")  # (n, 4) float64, đoạn đã gộp
    np.asarray(squares, dtype="<i4").reshape(-1, 4, 2).tofile(f"golden/{name}_squares.bin")      # (m, 4, 2) int32, xếp theo điểm giảm dần
    np.asarray(sq_scores, dtype="<f8").reshape(-1).tofile(f"golden/{name}_sq_scores.bin")       # (m,) float64
    np.asarray(inter, dtype="<i4").reshape(-1, 2).tofile(f"golden/{name}_inter.bin")            # (k, 2) int32, các góc tìm được

    h, w = img.shape[:2]
    meta.append({"name": name, "h": h, "w": w, "n_lines": len(lines),
                 "n_sq_segments": len(sq_segments), "n_squares": len(squares), "n_inter": len(inter)})
    print(f"{name:28s} anh {w}x{h}  ->  {len(lines)} doan thang | squares: {len(sq_segments)} doan gop, "
          f"{len(inter)} goc, {len(squares)} tu giac")

with open("golden/meta.json", "w") as f:
    json.dump(meta, f, indent=1)
