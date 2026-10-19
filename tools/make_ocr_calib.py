# Ảnh calibration cho quantize model đọc chữ: ảnh MÁY VẼ theo công thức bản 2 (tools/ocr_synth.py, giống dữ liệu train
# FT2), KHÔNG dùng ảnh thật của bộ valid / test. Mỗi cỡ độ rộng (model TFLite riêng) cần dòng có độ rộng rơi đúng vào cỡ
# đó, đệm 0 giống lúc chạy thật: cỡ 320 = dòng cần rộng <= 320, cỡ 640 = 321..640, ...
# Cỡ rộng cần dòng DÀI (tới ~110 ký tự) -> vẽ đoạn văn dài hơn 25 ký tự (calibration không cần đáp án).
# Đo tools/ocr_act_ranges.py (18/10): khoảng giá trị trên ảnh máy vẽ gần như trùng ảnh thật (tỉ lệ trung vị 0.97).
# Ra: data/ocr_calib/calib_w<W>.npy, NHWC float32 đã chuẩn hoá [-1, 1] (đúng đầu vào model TFLite FP32).
# Chạy (thư mục gốc repo):  .venv/bin/python tools/make_ocr_calib.py [--n 200]
import argparse
import glob
import os
import random
import sys

import numpy as np

sys.path.insert(0, "tools")
from ocr_eval import prep
from ocr_synth import (chunk_text, clean, group_fonts, label_of, make_pool_v2, pick_font, render_line_v2, sample_text_v2,
                       vietnamese_dict)

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=200, help="số dòng mỗi cỡ")
ap.add_argument("--buckets", default="320,640,1280,2000")
ap.add_argument("--seed", type=int, default=7)
args = ap.parse_args()

rng = random.Random(args.seed)
fonts = [f for f in glob.glob("/System/Library/Fonts/Supplemental/*.ttf")
         if os.path.basename(f).startswith(("Times New Roman", "Arial")) and not any(k in f for k in ["Narrow", "Black", "Rounded", "Unicode"])]
groups = group_fonts(fonts)
paras = [clean(l, set(vietnamese_dict())) for l in open("data/ocr_eval/wiki_vi.txt", encoding="utf-8") if len(l) > 60]
pool = make_pool_v2(paras, rng, 2000)
buckets = [int(b) for b in args.buckets.split(",")]
os.makedirs("data/ocr_calib", exist_ok=True)


def long_text(n_chars):
    # Đoạn văn dài ~n_chars ký tự, cắt theo từ; 15% viết hoa như dữ liệu train
    p = rng.choice([q for q in paras if len(q) > n_chars] or paras)
    start = rng.randint(0, max(0, len(p) - n_chars))
    t = " ".join(p[start:start + n_chars].split(" ")[1:-1]) or p[:n_chars]
    return t.upper() if rng.random() < 0.15 else t


for i, b in enumerate(buckets):
    lo = buckets[i - 1] if i else 0
    xs, tries = [], 0
    while len(xs) < args.n:
        tries += 1
        text = sample_text_v2(pool, rng) if b <= 320 else long_text(rng.randint(max(5, lo // 16), b // 9))
        nb = (rng.choice(pool), rng.choice(pool)) if rng.random() < 0.5 and len(label_of(text)) > 5 else None
        img = np.asarray(render_line_v2(text, pick_font(groups, rng), rng, neighbors=nb))[:, :, ::-1]   # RGB -> BGR
        if not (lo < img.shape[1] <= b):                         # ảnh đã cao 48: rộng = độ rộng cần thiết
            continue
        xs.append(prep(np.ascontiguousarray(img), b)[0].transpose(1, 2, 0))
    x = np.stack(xs).astype(np.float32)
    np.save(f"data/ocr_calib/calib_w{b}.npy", x)
    print(f"co {b}: {len(xs)} dong (thu {tries} lan), {x.shape}, {x.nbytes / 1e6:.0f} MB")
