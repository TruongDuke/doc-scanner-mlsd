# Golden cho nút "Kiểm tra OCR trang" (OcrPage.kt): cả chuỗi đọc trang trên Mac (tools/det_simple.py: DBNet FP32 ->
# khung xoay -> cắt dòng -> model đọc chữ INT8) cho vài trang ảnh chụp bằng app, để so với bản Kotlin trên điện thoại.
#   {tên}_rgb.bin  uint8 (h, w, 3) RGB: điểm ảnh trang (đưa thẳng vào app, không qua bộ giải mã JPEG của máy)
#   meta.json      [{name, h, w, boxes: [[8 số: 4 góc]], texts: [chữ từng dòng]}]
# Ra: golden/ocr_page/ (không đưa lên git). Chép vào máy: tools/push_golden.sh
# Chạy (thư mục gốc repo, .venv):  .venv/bin/python tools/make_ocr_page_golden.py
import json
import os
import shutil
import sys
import unicodedata

import cv2

sys.path.insert(0, "tools")
from ai_edge_litert.interpreter import Interpreter
from det_simple import boxes_from_prob, crop_line, prob_map
from ocr_eval import BucketRecognizer, ctc_decode, load_dict

PAGES = ["data/ocr/myscans/raw/scan_1792078082275.jpg",          # trang sách (dọc)
         "data/ocr/myscans/raw/scan_1792078242093.jpg",          # tờ khai CT01 (dọc, biểu mẫu)
         "data/ocr/myscans_test/raw/scan_1792314532393.jpg"]     # trang báo (ngang)
OUT = "golden/ocr_page"
shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT)
dets = {}
rec = BucketRecognizer("data/models/ocr/tflite/ft2cle_w{w}/ft2cle_w{w}_msein_int8.tflite", [320, 640, 1280, 2000])
chars = load_dict(None, "data/ocr_eval/vi_dict.txt")
meta = []
for k, path in enumerate(PAGES):
    img = cv2.imread(path)
    h, w = img.shape[:2]
    mh, mw = (1408, 992) if h >= w else (992, 1408)
    if (mh, mw) not in dets:
        dets[(mh, mw)] = Interpreter(model_path=f"data/models/ocr/det/det_{mh}x{mw}/det_float32.tflite")
        dets[(mh, mw)].allocate_tensors()
    boxes = boxes_from_prob(prob_map(dets[(mh, mw)], img, mh, mw), w, h)
    texts = [unicodedata.normalize("NFC", ctc_decode(rec.read(crop_line(img, b[5])), chars)).strip() for b in boxes]
    name = f"p{k}"
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB).tofile(f"{OUT}/{name}_rgb.bin")
    meta.append({"name": name, "h": h, "w": w, "boxes": [[float(v) for v in b[5].ravel()] for b in boxes],
                 "texts": texts})
    print(f"{name} {os.path.basename(path)} {w}x{h}: {len(boxes)} dong | {texts[:2]}")
json.dump(meta, open(f"{OUT}/meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
