# Tìm dòng chữ (DBNet PP-OCRv5 mobile det, bản TFLite cỡ cố định) + xử lý sau ĐƠN GIẢN, để viết lại bằng Kotlin trong app.
# Thay đường viền + khung xoay nhỏ nhất (cv2.findContours, minAreaRect, pyclipper của RapidOCR) bằng cách dễ viết lại:
# vùng liền nhau (8 hướng) -> khung chữ nhật XOAY theo hướng chính của vùng (PCA trên toạ độ các điểm).
# (Khung thẳng không được: trang chụp lệch ~1 độ thì dòng dài 750 px nghiêng cả chục px, khung thẳng + nới ra ôm luôn dòng
#  bên cạnh -> đọc ra chữ lộn xộn, CER 46% so với 5% của RapidOCR, đo 20/10.)
#   1. ảnh trang co (không giữ tỉ lệ) về H x W cố định của model, chuẩn hoá (x/255 - 0.5)/0.5, giữ thứ tự kênh BGR như RapidOCR
#   2. mask = xác suất > 0.3, nở 2x2 (như RapidOCR use_dilation)
#   3. mỗi vùng: tâm + hướng chính (PCA), chiếu các điểm lên 2 trục -> dài L, dày T; bỏ nếu cạnh ngắn < 3 hoặc điểm TB
#      (xác suất trên CÁC ĐIỂM của vùng) < 0.5
#   4. nới khung: d = L*T * 1.6 / (2(L+T)) (công thức unclip của RapidOCR) ra mỗi phía; bỏ nếu cạnh ngắn < 5
#   5. đổi về toạ độ ảnh gốc (4 góc), sắp thứ tự đọc (trên -> dưới; cùng dòng (lệch y < 10 px) thì trái -> phải)
#   Cắt dòng: crop_line() nắn khung xoay về ảnh ngang (biến đổi affine)
# So với RapidOCR (khung đã dùng để cắt dòng bộ myscans / myscans_test): tools/det_simple.py --compare
import argparse
import os
import sys

import cv2
import numpy as np

THRESH, BOX_THRESH, UNCLIP, MIN_SIZE = 0.3, 0.5, 1.6, 3
MAX_TILT = np.deg2rad(20)          # dòng trên trang đã nắn không nghiêng quá 20 độ


def prob_map(interp, bgr, h, w):
    x = (cv2.resize(bgr, (w, h)).astype(np.float32) / 255 - 0.5) / 0.5
    d = interp.get_input_details()[0]
    interp.set_tensor(d["index"], x[None])
    interp.invoke()
    p = interp.get_tensor(interp.get_output_details()[0]["index"])
    return p[0, :, :, 0] if p.shape[-1] == 1 else p[0, 0]


def boxes_from_prob(p, src_w, src_h):
    # Trả về danh sách khung: [x0, y0, x1, y1 (khung thẳng bao ngoài, để sắp thứ tự / so sánh), score, 4 góc (4x2, ảnh gốc)]
    h, w = p.shape
    sx, sy = src_w / w, src_h / h
    mask = cv2.dilate((p > THRESH).astype(np.uint8), np.ones((2, 2), np.uint8))
    n, lab = cv2.connectedComponents(mask, connectivity=8)
    out = []
    for k in range(1, n):
        ys, xs = np.nonzero(lab == k)
        if len(xs) < MIN_SIZE * MIN_SIZE:
            continue
        score = float(p[ys, xs].mean())
        if score < BOX_THRESH:
            continue
        cx, cy = xs.mean(), ys.mean()
        dx, dy = xs - cx, ys - cy
        sxx, syy, sxy = (dx * dx).mean(), (dy * dy).mean(), (dx * dy).mean()
        ang = 0.5 * np.arctan2(2 * sxy, sxx - syy)          # hướng chính (trải dài nhất)
        if abs(ang) > MAX_TILT:                              # vùng cao hơn rộng (ô số "1", "0"): trang đã nắn -> coi như nằm ngang
            ang = 0.0
        ux, uy = np.cos(ang), np.sin(ang)
        u = dx * ux + dy * uy
        v = -dx * uy + dy * ux
        u0, u1, v0, v1 = u.min() - 0.5, u.max() + 0.5, v.min() - 0.5, v.max() + 0.5   # +-0.5: điểm ảnh có bề rộng 1
        L, T = u1 - u0, v1 - v0
        if min(L, T) < MIN_SIZE:
            continue
        d = L * T * UNCLIP / (2 * (L + T))
        u0, u1, v0, v1 = u0 - d, u1 + d, v0 - d, v1 + d
        if min(u1 - u0, v1 - v0) < MIN_SIZE + 2:
            continue
        corners = np.array([[cx + a * ux - b * uy, cy + a * uy + b * ux] for a, b in [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]])
        corners[:, 0] = corners[:, 0] * sx
        corners[:, 1] = corners[:, 1] * sy
        corners[:, 0] = corners[:, 0].clip(0, src_w); corners[:, 1] = corners[:, 1].clip(0, src_h)
        out.append([corners[:, 0].min(), corners[:, 1].min(), corners[:, 0].max(), corners[:, 1].max(), score, corners])
    return reading_order(out)


def crop_line(img, corners):
    # Nắn khung xoay (4 góc: trên-trái, trên-phải, dưới-phải, dưới-trái theo hướng dòng) về ảnh ngang
    tl, tr, br, bl = corners
    if tl[0] > tr[0]:                                      # hướng chính ngược chiều -> đổi để chữ không bị lật
        tl, tr, br, bl = br, bl, tl, tr
    wd = int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
    ht = int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))
    m = cv2.getAffineTransform(np.float32([tl, tr, bl]), np.float32([[0, 0], [wd, 0], [0, ht]]))
    return cv2.warpAffine(img, m, (max(wd, 1), max(ht, 1)), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def reading_order(boxes):
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    for i in range(len(boxes) - 1):                      # như sorted_boxes của RapidOCR: cùng dòng thì trái trước
        for j in range(i, -1, -1):
            if abs(boxes[j + 1][1] - boxes[j][1]) < 10 and boxes[j + 1][0] < boxes[j][0]:
                boxes[j], boxes[j + 1] = boxes[j + 1], boxes[j]
            else:
                break
    return boxes


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args()
    sys.path.insert(0, "tools")
    from ai_edge_litert.interpreter import Interpreter
    from rapidocr import LangDet, ModelType, OCRVersion, RapidOCR
    ref = RapidOCR(params={"Global.use_cls": False, "Global.use_rec": False, "Global.log_level": "error",
                           "Det.ocr_version": OCRVersion.PPOCRV5, "Det.model_type": ModelType.MOBILE, "Det.lang_type": LangDet.CH})
    ROTATE = {"scan_1792075114813.jpg": cv2.ROTATE_90_CLOCKWISE}
    sizes = [(1408, 992), (1024, 736)]
    its = {s: Interpreter(model_path=f"data/models/ocr/det/det_{s[0]}x{s[1]}/det_float32.tflite") for s in sizes}
    for it in its.values():
        it.allocate_tensors()
    tot = {s: [0, 0, 0] for s in sizes}                  # [khung RapidOCR, tìm lại được (IoU>=0.5), khung thừa]
    for d in ["data/ocr/myscans/raw", "data/ocr/myscans_test/raw"]:
        for name in sorted(os.listdir(d)):
            img = cv2.imread(f"{d}/{name}")
            if name in ROTATE:
                img = cv2.rotate(img, ROTATE[name])
            H0, W0 = img.shape[:2]
            rb = ref(img).boxes
            rb = [] if rb is None else [[float(b[:, 0].min()), float(b[:, 1].min()), float(b[:, 0].max()), float(b[:, 1].max())] for b in np.asarray(rb)]
            line = f"{name}: RapidOCR {len(rb)} dong"
            for s in sizes:
                if W0 > H0:                              # trang nằm ngang: đổi chiều model cho đỡ méo
                    mine = boxes_from_prob(prob_map(its[s], cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), *s), H0, W0)
                    mine = []                            # (chưa hỗ trợ trang ngang) -> không tính
                    continue
                mine = boxes_from_prob(prob_map(its[s], img, *s), W0, H0)
                hit = sum(any(iou(r, m) >= 0.5 for m in mine) for r in rb)
                extra = sum(not any(iou(m, r) >= 0.5 for r in rb) for m in mine)
                tot[s][0] += len(rb); tot[s][1] += hit; tot[s][2] += extra
                line += f" | {s[0]}x{s[1]}: {len(mine)} dong, tim lai {hit}/{len(rb)}, thua {extra}"
            print(line, flush=True)
    for s, (n, hit, extra) in tot.items():
        print(f"TONG {s[0]}x{s[1]}: tim lai {hit}/{n} = {hit / max(n, 1) * 100:.1f}% dong cua RapidOCR, khung thua {extra}")
