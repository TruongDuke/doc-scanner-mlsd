#!/bin/bash
# Xuất model đọc chữ PaddleOCR (.pdparams sau khi train) -> model suy luận Paddle -> ONNX, chạy trên Mac.
# Cần: .venv-paddle (Python 3.11: paddlepaddle==3.2.0, paddle2onnx==2.1.0) và third_party/PaddleOCR (tag v3.7.0).
# Dùng:  tools/export_ocr_rec.sh <file .pdparams (bỏ đuôi)> <từ điển .txt> <tên ra>
#   vd.  tools/export_ocr_rec.sh data/models/ocr/vi_ft3_best_accuracy data/ocr_eval/vi_dict.txt vi_ft3
#   -> data/models/ocr/vi_ft3_infer/ (Paddle) và data/models/ocr/vi_ft3.onnx
set -e
ROOT=$(cd "$(dirname "$0")/.." && pwd)
W=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
DICT=$(cd "$(dirname "$2")" && pwd)/$(basename "$2")
OUT=$ROOT/data/models/ocr/$3
cd "$ROOT/third_party/PaddleOCR"
"$ROOT/.venv-paddle/bin/python" tools/export_model.py -c configs/rec/PP-OCRv5/multi_language/latin_PP-OCRv5_mobile_rec.yml -o \
  Global.pretrained_model="$W" Global.character_dict_path="$DICT" Global.save_inference_dir="${OUT}_infer" Global.use_gpu=false 2>&1 | grep -E "load pretrain|saved"
"$ROOT/.venv-paddle/bin/paddle2onnx" --model_dir "${OUT}_infer" --model_filename inference.json --params_filename inference.pdiparams \
  --save_file "$OUT.onnx" --opset_version 14 2>&1 | grep -E "saved"
