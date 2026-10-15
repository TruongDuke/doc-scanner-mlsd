# Cắt VinText (VinAI, CVPR 2021; chỉ dùng nghiên cứu / học tập, không phân phối lại, phải trích dẫn) thành ảnh
# TỪNG TỪ kèm nhãn, định dạng PaddleOCR ("đường_dẫn<TAB>chữ"), để train / đánh giá model đọc chữ.
#   - Nhãn VinText: mỗi dòng "x1,y1,x2,y2,x3,y3,x4,y4,CHỮ" (mức từ); "###" = không đọc được -> bỏ
#   - Mỗi từ được nắn thẳng theo 4 góc (biến đổi phối cảnh) rồi lưu JPEG
#   - Bỏ từ có ký tự ngoài từ điển 236 ký tự (tools/ocr_synth.vietnamese_dict), từ quá nhỏ, từ viết dọc
#   - Giữ cách chia của tác giả: train_images -> train, test_image -> valid, unseen_test_images -> unseen
# Chạy từ thư mục gốc repo (cần data/vintext/vietnamese đã giải nén):  .venv/bin/python tools/make_vintext_lines.py
import os
import sys
import unicodedata

import cv2
import numpy as np

sys.path.insert(0, "tools")
from ocr_synth import vietnamese_dict

SRC = "data/vintext/vietnamese"
OUT = "data/ocr/vintext"
SPLITS = {"train_images": "train", "test_image": "valid", "unseen_test_images": "unseen"}
CS = set(vietnamese_dict())


def crop(img, q):
    # Nắn tứ giác về hình chữ nhật: rộng = cạnh trên/dưới dài hơn, cao = cạnh trái/phải dài hơn
    w = int(max(np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[3])))
    h = int(max(np.linalg.norm(q[3] - q[0]), np.linalg.norm(q[2] - q[1])))
    if w < 4 or h < 8:
        return None
    m = cv2.getPerspectiveTransform(q.astype(np.float32), np.float32([[0, 0], [w, 0], [w, h], [0, h]]))
    return cv2.warpPerspective(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


stats = {}
for folder, split in SPLITS.items():
    os.makedirs(f"{OUT}/{split}", exist_ok=True)
    kept = skipped = 0
    with open(f"{OUT}/{split}.txt", "w", encoding="utf-8") as lab:
        for name in sorted(os.listdir(f"{SRC}/{folder}")):
            k = int(name[2:6])                                   # im0001.jpg -> gt_1.txt
            img = cv2.imread(f"{SRC}/{folder}/{name}")
            gt = f"{SRC}/labels/gt_{k}.txt"
            if img is None or not os.path.exists(gt):
                continue
            for j, line in enumerate(open(gt, encoding="utf-8-sig")):
                parts = line.strip().split(",", 8)
                if len(parts) < 9:
                    continue
                text = unicodedata.normalize("NFC", parts[8].strip())
                if text == "###" or not text or any(c not in CS for c in text):
                    skipped += 1
                    continue
                q = np.array(list(map(float, parts[:8]))).reshape(4, 2)
                c = crop(img, q)
                if c is None or (c.shape[0] > 1.5 * c.shape[1] and len(text) > 1):   # quá nhỏ / chữ viết dọc
                    skipped += 1
                    continue
                p = f"{split}/{k:04d}_{j:03d}.jpg"
                cv2.imwrite(f"{OUT}/{p}", c, [cv2.IMWRITE_JPEG_QUALITY, 95])
                lab.write(f"{p}\t{text}\n")
                kept += 1
    stats[split] = (kept, skipped)
    print(f"{split}: giu {kept} tu, bo {skipped}")
