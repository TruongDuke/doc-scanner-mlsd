# Chấm điểm model đọc chữ (OCR recognition) trên một bộ ảnh dòng chữ có đáp án, CÙNG MỘT CÁCH cho mọi model
# (baseline latin v5 gốc, model fine-tune tiếng Việt, bản INT8...), để so sánh công bằng.
#
# Bộ dữ liệu: thư mục ảnh + file nhãn kiểu PaddleOCR, mỗi dòng "đường_dẫn_tương_đối<TAB>chữ đúng".
# Model: file .onnx (vào NCHW) hoặc .tflite (vào NHWC), ảnh dòng cao 48, đệm cho đủ rộng 320, chuẩn hoá (x/255-0.5)/0.5.
# Từ điển: lấy từ metadata của .onnx (model RapidOCR) hoặc file --dict (mỗi dòng 1 ký tự); tự thêm khoảng trắng cuối
# và ký tự trống (blank) ở đầu, giống PaddleOCR use_space_char.
#
# Chỉ số:
#   CER          = tổng khoảng cách sửa (Levenshtein) / tổng số ký tự đáp án
#   CER bỏ dấu   = như trên nhưng xoá hết dấu tiếng Việt ở cả 2 bên (đ -> d): tách lỗi dấu khỏi lỗi chữ
#   dòng đúng    = % dòng đọc đúng hoàn toàn
#
# Chạy từ thư mục gốc repo (môi trường .venv):
#   .venv/bin/python tools/ocr_eval.py --model x.onnx --data data/ocr_eval/synth --labels eval.txt [--dict vi_dict.txt] [--show 10]
import argparse
import os
import unicodedata

import cv2
import numpy as np


def load_dict(model_path, dict_path):
    if dict_path:
        chars = [l.rstrip("\n") for l in open(dict_path, encoding="utf-8") if l.rstrip("\n")]
    else:
        import onnxruntime as ort
        meta = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"]).get_modelmeta().custom_metadata_map
        chars = meta["character"].splitlines()
    return ["<blank>"] + chars + [" "]


class Recognizer:
    def __init__(self, path):
        self.kind = os.path.splitext(path)[1]
        if self.kind == ".onnx":
            import onnxruntime as ort
            self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            self.inp = self.sess.get_inputs()[0].name
        else:
            from ai_edge_litert.interpreter import Interpreter
            self.it = Interpreter(model_path=path, num_threads=4)
            self.it.allocate_tensors()
            self.din, self.dout = self.it.get_input_details()[0], self.it.get_output_details()[0]

    def __call__(self, x_nchw):
        if self.kind == ".onnx":
            return self.sess.run(None, {self.inp: x_nchw})[0][0]
        self.it.set_tensor(self.din["index"], x_nchw.transpose(0, 2, 3, 1).copy())
        self.it.invoke()
        return self.it.get_tensor(self.dout["index"])[0]


def prep(bgr, width=320, height=48):
    # Giống RapidOCR / PaddleOCR khi suy luận: co về cao 48 giữ tỉ lệ (tối đa rộng 320), chuẩn hoá, đệm 0 bên phải
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    nw = min(width, int(np.ceil(height * w / h)))
    im = cv2.resize(rgb, (nw, height)).astype(np.float32)
    im = (im / 255.0 - 0.5) / 0.5
    out = np.zeros((height, width, 3), np.float32)
    out[:, :nw] = im
    return out.transpose(2, 0, 1)[None]


def ctc_decode(prob, chars):
    idx = prob.argmax(-1)
    res, prev = [], -1
    for k in idx:
        if k != prev and k != 0:
            res.append(chars[k])
        prev = k
    return "".join(res)


def levenshtein(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def strip_marks(s):
    s = s.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", s) if not unicodedata.combining(c))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="thư mục gốc chứa ảnh")
    ap.add_argument("--labels", required=True, help="file nhãn trong thư mục data")
    ap.add_argument("--dict", help="file từ điển (bỏ trống: lấy từ metadata .onnx)")
    ap.add_argument("--show", type=int, default=10, help="in bao nhiêu dòng sai đầu tiên")
    args = ap.parse_args()

    chars = load_dict(args.model, args.dict)
    rec = Recognizer(args.model)
    n_err = n_err_nm = n_chars = n_chars_nm = n_ok = n = 0
    wrong = []
    for line in open(os.path.join(args.data, args.labels), encoding="utf-8"):
        path, gt = line.rstrip("\n").split("\t", 1)
        gt = unicodedata.normalize("NFC", gt)
        pred = unicodedata.normalize("NFC", ctc_decode(rec(prep(cv2.imread(os.path.join(args.data, path)))), chars))
        e = levenshtein(pred, gt)
        n_err += e
        n_chars += len(gt)
        n_err_nm += levenshtein(strip_marks(pred), strip_marks(gt))
        n_chars_nm += len(strip_marks(gt))
        n_ok += pred == gt
        n += 1
        if e and len(wrong) < args.show:
            wrong.append((gt, pred))
    print(f"{os.path.basename(args.model)} | {n} dong | CER {n_err / n_chars * 100:.2f}% | CER bo dau {n_err_nm / n_chars_nm * 100:.2f}% "
          f"| dong dung {n_ok / n * 100:.1f}%")
    for gt, pred in wrong:
        print(f"   dung: {gt!r:32}  doc: {pred!r}")


if __name__ == "__main__":
    main()
