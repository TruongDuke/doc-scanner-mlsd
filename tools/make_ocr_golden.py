# Dữ liệu golden cho nút "Kiểm tra OCR" trong app (OcrGoldenCheck.kt): model đọc chữ INT8 chạy trên điện thoại có ra cùng
# kết quả như trên Mac không, đọc đúng bao nhiêu, mất bao lâu mỗi dòng.
# Chọn dòng từ bộ valid (data/ocr/myscans), chia đủ 4 cỡ độ rộng. Mỗi dòng:
#   {tên}_input.bin  float32 (48, W, 3) NHWC, đã chuẩn hoá [-1, 1] và đệm 0 tới cỡ W (đúng đầu vào model)
#   {tên}_idx.bin    int32 (W/8,): ký tự điểm cao nhất ở từng lát, model INT8 chạy trên Mac (ai_edge_litert)
# meta.json: [{name, w, gt (đáp án), mac_int8, mac_fp32 (chữ đọc ra trên Mac)}]; vi_dict.txt: từ điển 236 ký tự
# Ra: golden/ocr/ (không đưa lên git). Chép vào máy: tools/push_golden.sh
# Chạy (thư mục gốc repo):  .venv-tf/bin/python tools/make_ocr_golden.py [--per-bucket 15]
import argparse
import json
import os
import shutil
import sys
import unicodedata

import cv2
import numpy as np

sys.path.insert(0, "tools")
from ocr_eval import BucketRecognizer, ctc_decode, load_dict, prep

BUCKETS = [320, 640, 1280, 2000]
INT8 = "data/models/ocr/tflite/ft2cle_w{w}/ft2cle_w{w}_msein_int8.tflite"
FP32 = "data/models/ocr/tflite/ft2_w{w}/ft2_w{w}_float32.tflite"
ap = argparse.ArgumentParser()
ap.add_argument("--per-bucket", type=int, default=15)
args = ap.parse_args()

OUT = "golden/ocr"
shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT)
chars = load_dict(None, "data/ocr_eval/vi_dict.txt")
q, f = BucketRecognizer(INT8, BUCKETS), BucketRecognizer(FP32, BUCKETS)
items = [l.rstrip("\n").split("\t", 1) for l in open("data/ocr/myscans/labels.txt", encoding="utf-8")]
by_w = {b: [] for b in BUCKETS}
for p, gt in items:
    im = cv2.imread("data/ocr/myscans/" + p)
    by_w[q.width(im)].append((im, gt))

meta = []
for b in BUCKETS:
    pick = by_w[b][::max(1, len(by_w[b]) // args.per_bucket)][:args.per_bucket]
    for im, gt in pick:
        name = f"l{len(meta):03d}"
        x = prep(im, b)                                            # (1, 3, 48, b)
        prob = q.recs[b](x)
        x.transpose(0, 2, 3, 1).astype("<f4").tofile(f"{OUT}/{name}_input.bin")
        prob.argmax(-1).astype("<i4").tofile(f"{OUT}/{name}_idx.bin")
        meta.append({"name": name, "w": b, "gt": gt,
                     "mac_int8": unicodedata.normalize("NFC", ctc_decode(prob, chars)),
                     "mac_fp32": unicodedata.normalize("NFC", ctc_decode(f.recs[b](x), chars))})
    print(f"co {b}: {len(pick)} dong (co {len(by_w[b])})")
json.dump(meta, open(f"{OUT}/meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
shutil.copy("data/ocr_eval/vi_dict.txt", f"{OUT}/vi_dict.txt")
print(len(meta), "dong ->", OUT)
