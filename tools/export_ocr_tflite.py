# Đổi model đọc chữ ONNX -> TFLite FP32 (+ FP16), mỗi "cỡ" độ rộng ảnh vào một file (TFLite có kích thước cố định).
# Lúc chạy: dòng cần rộng w -> dùng cỡ nhỏ nhất >= w, đệm 0 bên phải (đo trên ONNX: CER không đổi so với rộng tự động).
#   1. onnxsim cố định kích thước 1x3x48xW (không cố định thì onnx2tf đổi sai thứ tự chiều ở 1 lớp Conv)
#   2. onnx2tf -> saved_model + <out>/<tên>_w<W>/..._float32.tflite, ..._float16.tflite
#   3. File của onnx2tf KHÔNG có signature (ai_edge_quantizer bắt buộc phải có). Cờ -osd không giúp và làm treo ở cỡ
#      1280; đổi lại saved_model bằng tf.lite.TFLiteConverter thì hỏng lớp PAD. -> giữ nguyên file của onnx2tf, chỉ ghi
#      thêm bảng signature "serving_default" (vào "x", ra "y") vào flatbuffer.
# onnx2tf tự kiểm tra kết quả bằng file ảnh mẫu tải từ GitHub (đã bị xoá, lỗi 404) -> tự tạo file đó bằng ảnh dòng chữ.
# Cần .venv-tf (tensorflow 2.19, onnx2tf, onnxsim, opencv).  Chạy (thư mục gốc repo):
#   .venv-tf/bin/python tools/export_ocr_tflite.py --onnx data/models/ocr/vi_ft2_e4.onnx --name ft2
import argparse
import glob
import os
import shutil
import subprocess
import sys

import cv2
import numpy as np
import flatbuffers
from ai_edge_litert import schema_py_generated as schema
from ai_edge_litert.interpreter import Interpreter

sys.path.insert(0, "tools")
from ocr_eval import prep

BUCKETS = [320, 640, 1280, 2000]


def add_signature(path):
    # Ghi thêm SignatureDef "serving_default" (vào "x", ra "y") cho subgraph 0; nội dung tính toán không đổi
    m = schema.ModelT.InitFromObj(schema.Model.GetRootAs(open(path, "rb").read(), 0))
    sg = m.subgraphs[0]
    sig = schema.SignatureDefT()
    sig.signatureKey, sig.subgraphIndex = "serving_default", 0
    sig.inputs, sig.outputs = [], []
    for name, idx, lst in [("x", sg.inputs[0], sig.inputs), ("y", sg.outputs[0], sig.outputs)]:
        tm = schema.TensorMapT()
        tm.name, tm.tensorIndex = name, idx
        lst.append(tm)
    m.signatureDefs = [sig]
    b = flatbuffers.Builder(1024)
    b.Finish(m.Pack(b), file_identifier=b"TFL3")
    open(path, "wb").write(b.Output())
    assert list(Interpreter(model_path=path).get_signature_list()) == ["serving_default"]

ap = argparse.ArgumentParser()
ap.add_argument("--onnx", required=True)
ap.add_argument("--name", required=True)
ap.add_argument("--out", default="data/models/ocr/tflite")
ap.add_argument("--buckets", default=",".join(map(str, BUCKETS)))
args = ap.parse_args()
bin_dir = os.path.dirname(sys.executable)
onnx_path = os.path.abspath(args.onnx)
lines = [l.split("\t")[0] for l in open("data/ocr/myscans/labels.txt")][::13][:20]

for w in map(int, args.buckets.split(",")):
    d = os.path.abspath(f"{args.out}/{args.name}_w{w}")
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    x = np.concatenate([prep(cv2.imread("data/ocr/myscans/" + p), w) for p in lines])
    np.save(f"{d}/calibration_image_sample_data_20x128x128x3_float32.npy", (x * 0.5 + 0.5).transpose(0, 2, 3, 1).astype(np.float32))
    sim = f"{d}/{args.name}_w{w}.onnx"
    subprocess.run([f"{bin_dir}/onnxsim", onnx_path, sim, "--overwrite-input-shape", f"x:1,3,48,{w}"],
                   check=True, capture_output=True)
    r = subprocess.run([f"{bin_dir}/onnx2tf", "-i", sim, "-o", ".", "-n"], cwd=d, capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-2000:], r.stderr[-2000:])
        raise SystemExit(f"onnx2tf loi o co {w}")
    for t in ("float32", "float16"):
        add_signature(f"{d}/{args.name}_w{w}_{t}.tflite")
    for f in ["saved_model.pb", "fingerprint.pb", "calibration_image_sample_data_20x128x128x3_float32.npy"]:
        if os.path.exists(f"{d}/{f}"):
            os.remove(f"{d}/{f}")
    shutil.rmtree(f"{d}/variables", ignore_errors=True)
    shutil.rmtree(f"{d}/assets", ignore_errors=True)
    print(w, sorted(os.path.basename(p) for p in glob.glob(f"{d}/*.tflite")))
