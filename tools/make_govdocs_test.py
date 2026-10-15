# Bộ test OCR "văn bản in thật": văn bản hành chính / pháp luật của nhà nước (vanban.chinhphu.vn), bản PDF điện tử
# có sẵn chữ -> lấy CHÍNH XÁC từng dòng chữ + vị trí, vẽ trang ra ảnh, cắt từng dòng. Không cần gõ đáp án.
# (Văn bản quy phạm pháp luật, văn bản hành chính không thuộc đối tượng bảo hộ quyền tác giả - Luật SHTT, Điều 15.)
#
# Hai phiên bản cùng các dòng:
#   clean: vẽ trang sạch ở 150 DPI (A4 ~ 1240x1754, gần độ phân giải ảnh chụp của app 1200x1600)
#   cam  : cả trang được "làm như ảnh chụp camera" trước khi cắt: mờ, ánh sáng không đều, ám màu, nhiễu, thu nhỏ rồi
#          phóng lại, nén JPEG. Hình học (nghiêng, phối cảnh) không làm vì app đã nắn thẳng trang trước khi OCR.
# Danh sách URL lưu ở data/ocr/govdocs/urls.txt để tạo lại đúng bộ cũ.
#
# Chạy từ thư mục gốc repo (.venv có pymupdf):  .venv/bin/python tools/make_govdocs_test.py
import os
import random
import re
import sys
import unicodedata
import urllib.request

import cv2
import numpy as np
import pymupdf

sys.path.insert(0, "tools")
from ocr_synth import vietnamese_dict

ROOT = "data/ocr/govdocs"
N_DOCS, PAGES_PER_DOC, MAX_PAGES = 25, 3, 40
DPI = 150
CS = set(vietnamese_dict())
UA = {"User-Agent": "Mozilla/5.0 (doc-scanner-mlsd personal study)"}


def get(url, path):
    if not os.path.exists(path):
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r, open(path, "wb") as f:
            f.write(r.read())


def camera_like(page, rng):
    # Ảnh chụp bằng điện thoại sau khi đã nắn thẳng: mờ nhẹ, sáng không đều, ám vàng / xám, nhiễu, mất chi tiết, JPEG
    img = page.astype(np.float32)
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ang = rng.uniform(0, 2 * np.pi)
    grad = (np.cos(ang) * xx / w + np.sin(ang) * yy / h)
    light = rng.uniform(0.72, 0.85) + rng.uniform(0.12, 0.25) * (grad - grad.min()) / (np.ptp(grad) + 1e-6)
    img *= light[..., None]
    img *= np.array([rng.uniform(0.85, 0.95), rng.uniform(0.93, 1.0), 1.0])[None, None, :]   # BGR: ám vàng
    img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.5, 1.1))
    f = rng.uniform(0.6, 0.85)
    img = cv2.resize(cv2.resize(img, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA), (w, h), interpolation=cv2.INTER_LINEAR)
    img += np.random.default_rng(rng.randint(0, 1 << 30)).normal(0, rng.uniform(3, 7), img.shape)
    ok, enc = cv2.imencode(".jpg", img.clip(0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, rng.randint(55, 85)])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def main():
    os.makedirs(f"{ROOT}/pdf", exist_ok=True)
    url_file = f"{ROOT}/urls.txt"
    if not os.path.exists(url_file):
        html = urllib.request.urlopen(urllib.request.Request("https://vanban.chinhphu.vn/he-thong-van-ban", headers=UA), timeout=60).read().decode("utf-8", "ignore")
        urls = sorted(set(re.findall(r"https://datafiles\.chinhphu\.vn/[^\"]+\.pdf", html)))
        open(url_file, "w").write("\n".join(urls) + "\n")
    urls = open(url_file).read().split()
    rng = random.Random(2027)
    rng.shuffle(urls)

    for v in ["clean", "cam"]:
        os.makedirs(f"{ROOT}/{v}", exist_ok=True)
    labels = {v: open(f"{ROOT}/{v}.txt", "w", encoding="utf-8") for v in ["clean", "cam"]}
    docs = n_lines = 0
    for url in urls:
        if docs >= N_DOCS:
            break
        name = os.path.basename(url)
        path = f"{ROOT}/pdf/{name}"
        try:
            get(url, path)
            d = pymupdf.open(path)
        except Exception as e:
            print("bo", name, e)
            continue
        if d.page_count > MAX_PAGES or len(d[0].get_text()) < 200:      # bỏ văn bản quá dài / bản scan không có chữ
            continue
        docs += 1
        pages = sorted(rng.sample(range(d.page_count), min(PAGES_PER_DOC, d.page_count)))
        for pi in pages:
            page = d[pi]
            zoom = DPI / 72
            pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
            clean = cv2.cvtColor(np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, 3), cv2.COLOR_RGB2BGR)
            cam = camera_like(clean, rng)
            for li, (text, bb) in enumerate(page_lines(page)):
                x0, y0, x1, y1 = [v * zoom for v in bb]
                pad_y = 0.15 * (y1 - y0)                     # chừa chỗ cho dấu phía trên / dưới (0.25 thì lấn sang dòng bên cạnh)
                x0, x1 = max(0, int(x0 - 4)), min(pix.w, int(x1 + 4))
                y0, y1 = max(0, int(y0 - pad_y)), min(pix.h, int(y1 + pad_y))
                if x1 - x0 < 8 or y1 - y0 < 8:
                    continue
                rel = f"{docs:02d}_{pi:03d}_{li:03d}.jpg"
                for v, img in [("clean", clean), ("cam", cam)]:
                    cv2.imwrite(f"{ROOT}/{v}/{rel}", img[y0:y1, x0:x1], [cv2.IMWRITE_JPEG_QUALITY, 95])
                    labels[v].write(f"{v}/{rel}\t{text}\n")
                n_lines += 1
        print(f"{docs:2d}. {name} ({d.page_count} trang) -> trang {pages}")
    for f in labels.values():
        f.close()
    print(f"\n{docs} van ban, {n_lines} dong -> {ROOT}/clean.txt, {ROOT}/cam.txt")


def page_lines(page):
    # Các dòng chữ trên trang: nối các đoạn (span) cùng dòng; bỏ dòng có ký tự ngoài từ điển, dòng xoay, chữ quá nhỏ,
    # dòng chỉ toàn ký tự trang trí (____, ....)
    for b in page.get_text("dict")["blocks"]:
        for l in b.get("lines", []):
            if abs(l["dir"][1]) > 0.01:
                continue
            spans = [s for s in l["spans"] if s["text"].strip()]
            if not spans or max(s["size"] for s in spans) < 7:
                continue
            raw = "".join(s["text"] for s in l["spans"])
            # Có >= 3 dấu cách liền nhau = ô để trống, điền sau khi ký (số văn bản, ngày tháng): chữ trên ảnh có thể
            # khác chữ trong PDF -> bỏ, tránh chấm oan
            if "   " in raw.strip():
                continue
            text = unicodedata.normalize("NFC", raw).strip()
            text = " ".join(text.split())
            if len(text) < 2 or any(c not in CS and c != " " for c in text):
                continue
            if sum(c.isalnum() for c in text) < 0.3 * len(text):
                continue
            x0 = min(s["bbox"][0] for s in spans); x1 = max(s["bbox"][2] for s in spans)
            y0 = min(s["bbox"][1] for s in spans); y1 = max(s["bbox"][3] for s in spans)
            yield text, (x0, y0, x1, y1)


if __name__ == "__main__":
    main()
