# Chấm điểm model đọc chữ (OCR recognition) trên một bộ ảnh dòng chữ có đáp án, CÙNG MỘT CÁCH cho mọi model
# (baseline latin v5 gốc, model fine-tune tiếng Việt, bản INT8...), để so sánh công bằng.
#
# Bộ dữ liệu: thư mục ảnh + file nhãn kiểu PaddleOCR, mỗi dòng "đường_dẫn_tương_đối<TAB>chữ đúng".
# Model: file .onnx (vào NCHW) hoặc .tflite (vào NHWC), ảnh dòng cao 48, đệm cho đủ rộng 320, chuẩn hoá (x/255-0.5)/0.5.
#   Nhiều cỡ độ rộng (TFLite kích thước cố định, tools/export_ocr_tflite.py): --model có "{w}", vd.
#   data/models/ocr/tflite/ft2_w{w}/ft2_w{w}_float32.tflite --buckets 320,640,1280,2000 -> mỗi dòng dùng cỡ nhỏ nhất đủ rộng.
#   TFLite vào / ra int8 (do ai_edge_quantizer tạo) được tự đổi số theo scale / zero_point.
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
        x = x_nchw.transpose(0, 2, 3, 1)
        if self.din["dtype"] == np.int8:
            s, z = self.din["quantization"]
            x = np.clip(np.round(x / s + z), -128, 127).astype(np.int8)
        self.it.set_tensor(self.din["index"], np.ascontiguousarray(x))
        self.it.invoke()
        y = self.it.get_tensor(self.dout["index"])[0]
        if self.dout["dtype"] == np.int8:
            s, z = self.dout["quantization"]
            y = (y.astype(np.float32) - z) * s
        return y


class BucketRecognizer:
    # Nhiều file TFLite, mỗi file 1 độ rộng cố định: dòng cần rộng w -> cỡ nhỏ nhất >= w (dài hơn cỡ lớn nhất thì bị ép)
    def __init__(self, pattern, buckets):
        self.buckets = sorted(buckets)
        self.recs = {b: Recognizer(pattern.replace("{w}", str(b))) for b in self.buckets}

    def width(self, bgr, height=48):
        need = int(np.ceil(height * bgr.shape[1] / bgr.shape[0]))
        return next((b for b in self.buckets if b >= need), self.buckets[-1])

    def read(self, bgr):
        w = self.width(bgr)
        return self.recs[w](prep(bgr, w))


def prep(bgr, width=320, height=48):
    # Giống RapidOCR / PaddleOCR khi suy luận: co về cao 48 giữ tỉ lệ (tối đa rộng `width`), chuẩn hoá, đệm 0 bên phải.
    # width = 0: rộng tự động theo tỉ lệ ảnh (làm tròn lên bội của 8, tối thiểu 320) -> đọc được dòng dài;
    #            chỉ dùng được với model .onnx (TFLite có kích thước cố định).
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    if width == 0:
        width = max(320, int(np.ceil(height * w / h / 8)) * 8)
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
    ap.add_argument("--width", type=int, default=320, help="độ rộng ảnh vào; 0 = tự động theo độ dài dòng (chỉ .onnx)")
    ap.add_argument("--buckets", help="các cỡ độ rộng, vd. 320,640,1280,2000 (khi --model có {w})")
    args = ap.parse_args()

    chars = load_dict(args.model, args.dict)
    if args.buckets:
        br = BucketRecognizer(args.model, [int(b) for b in args.buckets.split(",")])
        read = br.read
    else:
        rec = Recognizer(args.model)
        read = lambda bgr: rec(prep(bgr, args.width))
    n_err = n_err_nm = n_chars = n_chars_nm = n_ok = n = 0
    wrong = []
    for line in open(os.path.join(args.data, args.labels), encoding="utf-8"):
        path, gt = line.rstrip("\n").split("\t", 1)
        gt = unicodedata.normalize("NFC", gt)
        pred = unicodedata.normalize("NFC", ctc_decode(read(cv2.imread(os.path.join(args.data, path))), chars))
        e = levenshtein(pred, gt)
        n_err += e
        n_chars += len(gt)
        n_err_nm += levenshtein(strip_marks(pred), strip_marks(gt))
        n_chars_nm += len(strip_marks(gt))
        n_ok += pred == gt
        n += 1
        if e and len(wrong) < args.show:
            wrong.append((gt, pred))
    print(f"{os.path.basename(args.model)} | rong {('co ' + args.buckets) if args.buckets else (args.width or 'tu dong')} | {n} dong | CER {n_err / n_chars * 100:.2f}% | CER bo dau {n_err_nm / n_chars_nm * 100:.2f}% "
          f"| dong dung {n_ok / n * 100:.1f}%")
    for gt, pred in wrong:
        print(f"   dung: {gt!r:32}  doc: {pred!r}")


if __name__ == "__main__":
    main()
