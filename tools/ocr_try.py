# Mốc 8, bước 8.1: thử nhanh OCR tiếng Việt bằng các model PaddleOCR (bản ONNX đóng gói trong RapidOCR)
# trước khi quyết định đưa vào app. So 3 cấu hình (tìm dòng + đọc chữ), in chữ đọc được và thời gian.
#
# Môi trường: .venv của repo (pip install rapidocr onnxruntime). Chạy từ thư mục gốc repo:
#   .venv/bin/python tools/ocr_try.py data/ocr_try/*.png data/ocr_try/*.jpg
# Nhẹ RAM: mỗi lần chỉ nạp 1 cấu hình, dùng xong giải phóng.
import gc
import os
import sys
import time

import cv2
from rapidocr import EngineType, LangDet, LangRec, ModelType, OCRVersion, RapidOCR

CONFIGS = {
    "A: v5 mobile + latin v5 mobile": {
        "Det.ocr_version": OCRVersion.PPOCRV5, "Det.model_type": ModelType.MOBILE, "Det.lang_type": LangDet.CH,
        "Rec.ocr_version": OCRVersion.PPOCRV5, "Rec.model_type": ModelType.MOBILE, "Rec.lang_type": LangRec.LATIN,
    },
    "B: v6 tiny + v6 tiny": {
        "Det.ocr_version": OCRVersion.PPOCRV6, "Det.model_type": ModelType.TINY,
        "Rec.ocr_version": OCRVersion.PPOCRV6, "Rec.model_type": ModelType.TINY,
    },
    "C: v6 small + v6 small": {
        "Det.ocr_version": OCRVersion.PPOCRV6, "Det.model_type": ModelType.SMALL,
        "Rec.ocr_version": OCRVersion.PPOCRV6, "Rec.model_type": ModelType.SMALL,
    },
}

images = sys.argv[1:]
for name, cfg in CONFIGS.items():
    params = {"Global.use_cls": False, "Global.log_level": "warning",
              "Det.engine_type": EngineType.ONNXRUNTIME, "Rec.engine_type": EngineType.ONNXRUNTIME, **cfg}
    engine = RapidOCR(params=params)
    print(f"\n==================== {name} ====================")
    for path in images:
        img = cv2.imread(path)
        engine(img)                      # chạy làm nóng 1 lần
        t0 = time.perf_counter()
        res = engine(img)
        ms = (time.perf_counter() - t0) * 1000
        print(f"--- {os.path.basename(path)} ({img.shape[1]}x{img.shape[0]}): {ms:.0f} ms, {len(res.txts or [])} dong")
        for txt, score in zip(res.txts or [], res.scores or []):
            print(f"   [{score:.2f}] {txt}")
    del engine
    gc.collect()
