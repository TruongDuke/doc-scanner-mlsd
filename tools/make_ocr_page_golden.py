# Golden cho nút "Kiểm tra OCR trang" (OcrPage.kt): cả chuỗi đọc trang trên Mac (tools/det_simple.py: DBNet FP32 ->
# khung xoay -> cắt dòng -> model đọc chữ INT8) cho vài trang ảnh chụp bằng app, để so với bản Kotlin trên điện thoại.
#   {tên}_rgb.bin  uint8 (h, w, 3) RGB: điểm ảnh trang (đưa thẳng vào app, không qua bộ giải mã JPEG của máy)
#   meta.json      [{name, h, w, boxes: [[8 số: 4 góc]], texts: [chữ từng dòng], gts: [đáp án từng dòng hoặc null]}]
#   gts: chỉ với trang thuộc bộ valid (data/ocr/myscans): ghép khung với khung RapidOCR đã có đáp án (cùng thứ tự với
#   tools/make_myscans_lines.py, IoU >= 0.5) -> app chấm CER máy và Mac theo đáp án trên đúng các dòng đó
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
from det_simple import boxes_from_prob, crop_line, iou, prob_map
from rapidocr import LangDet, ModelType, OCRVersion, RapidOCR
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
ref = RapidOCR(params={"Global.use_cls": False, "Global.use_rec": False, "Global.log_level": "error",
                       "Det.ocr_version": OCRVersion.PPOCRV5, "Det.model_type": ModelType.MOBILE, "Det.lang_type": LangDet.CH})
labels = dict(l.rstrip("\n").split("\t", 1) for l in open("data/ocr/myscans/labels.txt", encoding="utf-8"))


def page_gts(img, path, boxes):
    # Đáp án cho từng khung (None nếu không có): khung RapidOCR xếp như make_myscans_lines -> nhãn lines/{doc}_{k:03d}.jpg
    if "/myscans/" not in path:
        return [None] * len(boxes)
    doc = os.path.basename(path)[5:-4]
    rb = sorted(ref(img).boxes, key=lambda b: (round(min(p[1] for p in b) / 12), min(p[0] for p in b)))
    rb = [([min(p[0] for p in b), min(p[1] for p in b), max(p[0] for p in b), max(p[1] for p in b)], labels.get(f"lines/{doc}_{k:03d}.jpg"))
          for k, b in enumerate(rb)]
    out = []
    for b in boxes:
        r, gt = max(rb, key=lambda x: iou(x[0], b))
        out.append(gt if gt is not None and iou(r, b) >= 0.5 else None)
    return out


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
    gts = page_gts(img, path, boxes)
    meta.append({"name": name, "h": h, "w": w, "boxes": [[float(v) for v in b[5].ravel()] for b in boxes],
                 "texts": texts, "gts": gts})
    from ocr_eval import levenshtein
    pairs = [(t, g) for t, g in zip(texts, gts) if g is not None]
    cer = sum(levenshtein(t, g) for t, g in pairs) / max(1, sum(len(g) for _, g in pairs)) * 100
    print(f"{name} {os.path.basename(path)} {w}x{h}: {len(boxes)} dong, {len(pairs)} dong co dap an, CER Mac {cer:.2f}%")
json.dump(meta, open(f"{OUT}/meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
