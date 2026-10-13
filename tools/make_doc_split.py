# Chia SmartDoc 2015 (data/smartdoc15/) thành tập calibration (cho INT8) và tập đánh giá (có đáp án 4 góc).
# Khung hình cắt từ video nên các khung liền nhau gần giống hệt nhau -> chia THEO TÀI LIỆU:
# mỗi loại (datasheet, letter, ...) có 5 tài liệu: 3 tài liệu cho calibration, 2 cho đánh giá.
# Trong mỗi video (1 tài liệu trên 1 nền) lấy vài khung cách đều nhau.
#
# Chạy từ thư mục gốc repo: python tools/make_doc_split.py
# Ra: data/calib.txt (đường dẫn ảnh), data/eval.csv (đường dẫn ảnh + 4 góc đáp án tl, tr, br, bl)
import csv
import gzip
import random
from collections import defaultdict

ROOT = "data/smartdoc15"
CALIB_PER_VIDEO = 4   # 18 tài liệu x 5 nền x 4 = 360 khung
EVAL_PER_VIDEO = 2    # 12 tài liệu x 5 nền x 2 = 120 khung

rows = list(csv.DictReader(gzip.open(f"{ROOT}/metadata.csv.gz", "rt")))

# 1. Chọn tài liệu cho từng tập (cố định hạt giống để lần nào chạy cũng ra như nhau)
rng = random.Random(0)
docs_by_type = defaultdict(set)
for r in rows:
    docs_by_type[r["modeltype_name"]].add(r["model_name"])
calib_docs, eval_docs = set(), set()
for t in sorted(docs_by_type):
    docs = sorted(docs_by_type[t])
    rng.shuffle(docs)
    calib_docs.update(docs[:3])
    eval_docs.update(docs[3:])

# 2. Gom khung theo video (nền, tài liệu), xếp theo thứ tự thời gian
videos = defaultdict(list)
for r in rows:
    videos[(r["bg_name"], r["model_name"])].append(r)
for v in videos.values():
    v.sort(key=lambda r: int(r["frame_index"]))

def pick(frames, n):
    # n khung cách đều nhau, tránh 2 đầu video (lúc mới bắt đầu / sắp kết thúc quay)
    return [frames[int(len(frames) * (k + 1) / (n + 1))] for k in range(n)]

calib, evals = [], []
for (bg, doc), frames in sorted(videos.items()):
    if doc in calib_docs:
        calib += pick(frames, CALIB_PER_VIDEO)
    else:
        evals += pick(frames, EVAL_PER_VIDEO)

with open("data/calib.txt", "w") as f:
    for r in calib:
        f.write(f"{ROOT}/{r['image_path']}\n")
with open("data/eval.csv", "w") as f:
    w = csv.writer(f)
    w.writerow(["image_path", "bg_name", "model_name", "tl_x", "tl_y", "tr_x", "tr_y", "br_x", "br_y", "bl_x", "bl_y"])
    for r in evals:
        w.writerow([f"{ROOT}/{r['image_path']}", r["bg_name"], r["model_name"]] +
                   [r[f"{c}_{a}"] for c in ["tl", "tr", "br", "bl"] for a in ["x", "y"]])

print(f"Calibration: {len(calib)} khung tu {len(calib_docs)} tai lieu: {sorted(calib_docs)}")
print(f"Danh gia:    {len(evals)} khung tu {len(eval_docs)} tai lieu: {sorted(eval_docs)}")
