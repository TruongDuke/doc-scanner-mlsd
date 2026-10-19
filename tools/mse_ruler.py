# "Thước" MSE cho từng tensor activation: chọn khoảng [lo, hi] để làm tròn 256 mức ít sai nhất (trung bình bình phương sai
# số) so với giá trị thật trên ảnh calibration, thay cho 1 quy tắc chung (p75) cho mọi tensor.
#   1. Chạy model TFLite FP32 trên ảnh calibration, giữ mọi tensor trung gian, đếm giá trị vào histogram (4096 ô) mỗi tensor
#      (khoảng ô = min..max từng gặp, lấy từ data/ocr_calib/qsv_<tên>_w<W>_n<N>.pkl của tools/quantize_ocr.py)
#   2. Với mỗi tensor thử nhiều cặp (lo, hi) (cắt dần 2 đầu theo phân vị), sai số = sai số làm tròn trong khoảng
#      (bước^2 / 12) + sai số bị cắt ngoài khoảng; chọn cặp nhỏ nhất. Tensor chỉ đi vào HARD_SWISH: lo >= -3 (không mất gì)
# Ra: data/ocr_calib/mse_<tên>_w<W>.pkl  {tên tensor: (lo, hi)}; dùng bằng  tools/quantize_ocr.py --version mse...
# Cần môi trường có ai_edge_quantizer (đọc qsv) + ai_edge_litert.  Chạy (thư mục gốc repo):
#   <python 3-deploy> tools/mse_ruler.py --name ft2cle
import argparse
import pickle

import numpy as np
from ai_edge_litert.interpreter import Interpreter, OpResolverType

BUCKETS = [320, 640, 1280, 2000]
NB = 4096
ap = argparse.ArgumentParser()
ap.add_argument("--name", default="ft2cle")
ap.add_argument("--n-calib", type=int, default=200)
ap.add_argument("--buckets", default=",".join(map(str, BUCKETS)))
args = ap.parse_args()


def best_range(hist, edges, lo_floor=None):
    # Thử cắt 2 đầu theo phân vị; trả (lo, hi) có sai số nhỏ nhất
    centers = (edges[:-1] + edges[1:]) / 2
    cdf = np.cumsum(hist) / max(hist.sum(), 1)
    qs = [0, 1e-6, 1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 2e-2, 5e-2, 0.1]
    cand_lo = sorted({float(edges[np.searchsorted(cdf, q)]) for q in qs})
    cand_hi = sorted({float(edges[min(np.searchsorted(cdf, 1 - q) + 1, len(edges) - 1)]) for q in qs})
    best = None
    for lo in cand_lo:
        lo = min(lo, 0.0)
        if lo_floor is not None:
            lo = max(lo, lo_floor)
        for hi in cand_hi:
            hi = max(hi, 0.0)
            if hi - lo <= 0:
                continue
            step = (hi - lo) / 255
            inside = (centers >= lo) & (centers <= hi)
            err = (hist[inside].sum() * step * step / 12
                   + (hist[centers < lo] * (lo - centers[centers < lo]) ** 2).sum()
                   + (hist[centers > hi] * (centers[centers > hi] - hi) ** 2).sum())
            if lo_floor is not None:                       # dưới -3 qua hard_swish đều ra 0: cắt ở đó không tính sai số
                err -= (hist[centers < lo] * (lo - centers[centers < lo]) ** 2).sum() if lo <= -3 + 1e-6 else 0
            if best is None or err < best[0]:
                best = (err, lo, hi)
    return best[1], best[2]


for w in map(int, args.buckets.split(",")):
    d = f"data/models/ocr/tflite/{args.name}_w{w}"
    fp32 = f"{d}/{args.name}_w{w}_float32.tflite"
    per = pickle.load(open(f"data/ocr_calib/qsv_{args.name}_w{w}_n{args.n_calib}.pkl", "rb"))
    names = list(per[0])
    gmin = {k: float(min(p[k][0].min() for p in per)) for k in names}
    gmax = {k: float(max(p[k][1].max() for p in per)) for k in names}
    it = Interpreter(model_path=fp32, experimental_preserve_all_tensors=True,
                     experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    it.allocate_tensors()
    td = it.get_tensor_details()
    ops = it._get_ops_details()
    users = {}
    for o in ops:
        for t in o["inputs"]:
            users.setdefault(t, []).append(o["op_name"])
    idx = {t["name"]: t["index"] for t in td}
    hs_in = {td[t]["name"] for o in ops for t in o["outputs"] if users.get(t) and all(u == "HARD_SWISH" for u in users[t])}
    keep = [k for k in names if k in idx]
    hists = {k: np.zeros(NB) for k in keep}
    edges = {k: np.linspace(gmin[k], gmax[k] if gmax[k] > gmin[k] else gmin[k] + 1e-6, NB + 1) for k in keep}
    x = np.load(f"data/ocr_calib/calib_w{w}.npy", mmap_mode="r")[:args.n_calib]
    din = it.get_input_details()[0]["index"]
    for i in range(len(x)):
        it.set_tensor(din, np.array(x[i:i + 1]))
        it.invoke()
        for k in keep:
            hists[k] += np.histogram(it.get_tensor(idx[k]), edges[k])[0]
    out = {k: best_range(hists[k], edges[k], -3.0 if k in hs_in else None) for k in keep}
    pickle.dump(out, open(f"data/ocr_calib/mse_{args.name}_w{w}.pkl", "wb"))
    print(f"co {w}: {len(out)} tensor ({len(hs_in & set(keep))} vao HARD_SWISH)", flush=True)
