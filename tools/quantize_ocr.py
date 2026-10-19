# Quantize model đọc chữ (TFLite FP32, mỗi cỡ độ rộng 1 file, tools/export_ocr_tflite.py) bằng ai_edge_quantizer,
# static INT8 (trọng số int8 per-channel, activation int8). Các bản khác nhau ở cách chọn "thước" (khoảng min..max được
# chia 256 mức) cho activation của từng tensor:
#   v1      mặc định ai_edge_quantizer: trung bình trượt (hệ số 0.95) của min / max TỪNG ảnh = mức "thường gặp"
#           (đo 18/10: đầu vào chỉ phủ -0.50..0.47 trong khi ảnh trải -1..1 -> sáng / tối bị cắt)
#   ema     tự tính lại đúng như v1 từ min / max từng ảnh (kiểm tra cách tự tính thước cho ra cùng kết quả v1)
#   minmax  min / max toàn bộ ảnh calibration (không cắt gì)
#   pXX     bỏ ảnh cực đoan: max = percentile XX của max từng ảnh, min = percentile (100-XX) của min từng ảnh
#   mse     thước riêng từng tensor, ít sai số làm tròn nhất (tools/mse_ruler.py -> data/ocr_calib/mse_<tên>_w<W>.pkl;
#           tensor vào HARD_SWISH đã có lo >= -3); tensor không có trong file thì dùng p75
#   +in     (hậu tố, vd. p99in) đầu vào đặt cố định -1..1 (biết chắc khoảng: ảnh đã chuẩn hoá)
#   +hs     (hậu tố, vd. p75inhs) tensor chỉ đi vào HARD_SWISH: đầu dưới thước = -3 (hard_swish(x) = 0 với mọi x <= -3
#           -> cắt không mất thông tin, phần dương mịn hơn). Chẩn đoán 19/10 (tools/ocr_sim_quant.py)
# --keep-float a:b[,c:d]  giữ float các op thứ a..b (đếm trong file FP32, không delegate) để khoanh vùng lớp gây lỗi.
#   Ranh giới (cỡ 320): CNN 0..128 (cuối CNN, tầng 480 kênh + AVERAGE_POOL: 107..128), SVTR 129..238
#   (conv 129..134, attention 138..238), lớp cuối 239..240
# Cách làm: đo min / max từng ảnh ở mọi tensor (calibrator của ai_edge_quantizer, mỗi lần 1 ảnh), tự tính thước,
# ghi đè vào kết quả calibration rồi quantize. Kết quả đo lưu data/ocr_calib/qsv_<tên>_w<W>.pkl để dùng lại.
# Calibration: data/ocr_calib/calib_w<W>.npy (ảnh máy vẽ, tools/make_ocr_calib.py), đúng cỡ của từng model.
# Cần môi trường có ai_edge_quantizer (vd. "/Users/duc/Neural Network Quantization/3-deploy/.venv").
# Chạy (thư mục gốc repo):  <python đó> tools/quantize_ocr.py --version minmax [--n-calib 200]
import argparse
import os
import pickle
import re

import numpy as np
from ai_edge_litert.interpreter import Interpreter, OpResolverType
from ai_edge_quantizer import algorithm_manager, calibrator, quantizer, recipe

BUCKETS = [320, 640, 1280, 2000]
ap = argparse.ArgumentParser()
ap.add_argument("--name", default="ft2")
ap.add_argument("--version", default="v1", help="v1 | ema | minmax | pXX | <cách>in")
ap.add_argument("--n-calib", type=int, default=200)
ap.add_argument("--buckets", default=",".join(map(str, BUCKETS)))
ap.add_argument("--keep-float", default="", help="khoảng op giữ float, vd. 107:128,239:240")
ap.add_argument("--tag", default="", help="hậu tố tên file, vd. keepcnn")
args = ap.parse_args()


def keep_float_rules(fp32, spec):
    # Mỗi op trong khoảng -> 1 luật NO_QUANTIZE; tên op trong ai_edge_quantizer = tên tensor đầu ra + ";"
    if not spec:
        return []
    it = Interpreter(model_path=fp32, experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    it.allocate_tensors()
    td, ops = it.get_tensor_details(), it._get_ops_details()
    idx = [i for r in spec.split(",") for i in range(int(r.split(":")[0]), int(r.split(":")[1]) + 1)]
    return [{"regex": "^" + re.escape(td[ops[i]["outputs"][0]]["name"]) + ";$", "operation": "*",
             "algorithm_key": algorithm_manager.AlgorithmName.NO_QUANTIZE, "op_config": {}} for i in idx]


def hardswish_inputs(fp32):
    # Tensor mà MỌI op dùng nó đều là HARD_SWISH (không bị lớp khác đọc giá trị âm)
    it = Interpreter(model_path=fp32, experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    it.allocate_tensors()
    td, ops = it.get_tensor_details(), it._get_ops_details()
    users = {}
    for o in ops:
        for t in o["inputs"]:
            users.setdefault(t, []).append(o["op_name"])
    outs = [t for o in ops for t in o["outputs"]]
    return [td[t]["name"] for t in outs if users.get(t) and all(u == "HARD_SWISH" for u in users[t])]


def per_sample_qsvs(qt, sig_name, sig_in, x, cache):
    # min / max của TỪNG ảnh ở mọi tensor: calibrator với hàm cập nhật "lấy giá trị mới" -> sau mỗi ảnh đọc ra
    if os.path.exists(cache):
        return pickle.load(open(cache, "rb"))
    calib = calibrator.Calibrator(qt._float_model_buffer, num_threads=8, qsv_update_func=lambda old, new: new)
    per = []
    for i in range(len(x)):
        calib.calibrate({sig_name: [{sig_in: np.array(x[i:i + 1])}]}, qt._recipe_manager)
        per.append({k: (np.array(v["min"]), np.array(v["max"])) for k, v in calib.get_model_qsvs().items()})
    pickle.dump(per, open(cache, "wb"))
    return per


def ruler(per, how):
    # Tự tính thước cho mọi tensor từ min / max từng ảnh
    out = {}
    for k in per[0]:
        mins = np.stack([p[k][0] for p in per])
        maxs = np.stack([p[k][1] for p in per])
        if how == "ema":
            lo, hi = mins[0], maxs[0]
            for a, b in zip(mins[1:], maxs[1:]):
                lo, hi = 0.95 * lo + 0.05 * a, 0.95 * hi + 0.05 * b
        elif how == "minmax":
            lo, hi = mins.min(0), maxs.max(0)
        elif how.startswith("p"):
            q = float(how[1:].replace("_", "."))
            lo, hi = np.percentile(mins, 100 - q, axis=0), np.percentile(maxs, q, axis=0)
        else:
            raise SystemExit(f"khong biet cach {how}")
        out[k] = {"min": np.asarray(lo, np.float32).reshape(mins.shape[1:]), "max": np.asarray(hi, np.float32).reshape(maxs.shape[1:])}
    return out


for w in map(int, args.buckets.split(",")):
    d = f"data/models/ocr/tflite/{args.name}_w{w}"
    fp32 = f"{d}/{args.name}_w{w}_float32.tflite"
    it = Interpreter(model_path=fp32)
    sig_name, sig = list(it.get_signature_list().items())[0]
    x = np.load(f"data/ocr_calib/calib_w{w}.npy", mmap_mode="r")[:args.n_calib]
    qt = quantizer.Quantizer(fp32, recipe.static_wi8_ai8() + keep_float_rules(fp32, args.keep_float))
    if args.version == "v1":
        cal = qt.calibrate({sig_name: [{sig["inputs"][0]: np.array(x[i:i + 1])} for i in range(len(x))]})
    else:
        per = per_sample_qsvs(quantizer.Quantizer(fp32, recipe.static_wi8_ai8()), sig_name, sig["inputs"][0], x,
                              f"data/ocr_calib/qsv_{args.name}_w{w}_n{len(x)}.pkl")
        v = args.version
        hs = v.endswith("hs")
        v = v[:-2] if hs else v
        fix_in = v.endswith("in")
        base = v[:-2] if fix_in else v
        if base == "mse":
            cal = ruler(per, "p75")
            for k, (lo, hi) in pickle.load(open(f"data/ocr_calib/mse_{args.name}_w{w}.pkl", "rb")).items():
                if k in cal:
                    cal[k] = {"min": np.full_like(cal[k]["min"], lo), "max": np.full_like(cal[k]["max"], hi)}
        else:
            cal = ruler(per, base)
        if hs:
            n = 0
            for name in hardswish_inputs(fp32):
                if name in cal:
                    cal[name]["min"] = np.maximum(cal[name]["min"], -3.0).astype(np.float32); n += 1
            print(f"  cat duoi -3: {n} tensor vao HARD_SWISH")
        if fix_in:
            in_name = it.get_input_details()[0]["name"]
            cal[in_name] = {"min": np.full_like(cal[in_name]["min"], -1.0), "max": np.full_like(cal[in_name]["max"], 1.0)}
    out = f"{d}/{args.name}_w{w}_{args.version}{args.tag}_int8.tflite"
    qt.quantize(cal).export_model(out, overwrite=True)
    q = Interpreter(model_path=out).get_input_details()[0]["quantization"]
    print(f"{out}: {os.path.getsize(out) / 1e6:.2f} MB | dau vao scale {q[0]:.5f} zp {q[1]} -> {(-128 - q[1]) * q[0]:.2f}..{(127 - q[1]) * q[0]:.2f}", flush=True)
