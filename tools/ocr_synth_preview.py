# Xem trước dữ liệu tổng hợp bản 2 (FT2) cạnh ảnh dòng thật user chụp, trước khi sinh 60.000 dòng và train.
# Cả hai đều thu về cao 48 px (đúng cỡ model nhìn thấy). Font: Times New Roman / Arial có sẵn trên macOS
# (trên Colab dùng Liberation Serif / Sans, cùng hình dáng). Văn bản: data/ocr_eval/wiki_vi.txt + cụm từ giấy tờ.
# Chạy từ thư mục gốc repo (.venv):  .venv/bin/python tools/ocr_synth_preview.py [--n 80]
#   -> data/ocr_eval/preview_v2/index.html
import argparse
import glob
import html
import os
import random
import sys

import cv2

sys.path.insert(0, "tools")
from ocr_synth import (group_fonts, label_of, make_pool_v2, pick_font, render_line_v2, sample_text_v2,
                       vietnamese_dict)

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=80, help="số ảnh tổng hợp")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

OUT = "data/ocr_eval/preview_v2"
os.makedirs(OUT, exist_ok=True)
rng = random.Random(args.seed)
cs = set(vietnamese_dict()) | {" "}
fonts = [f for f in glob.glob("/System/Library/Fonts/Supplemental/*.ttf")
         if os.path.basename(f).startswith(("Times New Roman", "Arial")) and "Narrow" not in f and "Black" not in f
         and "Rounded" not in f and "Unicode" not in f]
groups = group_fonts(fonts)
pool = make_pool_v2([l.strip() for l in open("data/ocr_eval/wiki_vi.txt", encoding="utf-8") if l.strip()], rng, 2000)

syn = []
for k in range(args.n):
    text = sample_text_v2(pool, rng)
    nb = (rng.choice(pool), rng.choice(pool)) if rng.random() < 0.5 and len(label_of(text)) > 5 else None   # ô bảng không có dòng bên cạnh
    font = pick_font(groups, rng)
    img = render_line_v2(text, font, rng, neighbors=nb)
    lab = label_of(text)
    assert all(c in cs for c in lab), lab
    p = f"syn_{k:03d}.jpg"
    img.save(f"{OUT}/{p}", quality=92)
    syn.append((p, lab, f"{os.path.basename(font)[:-4]}{' | có dòng bên cạnh' if nb else ''}"))

real = [l.rstrip("\n").split("\t", 1) for l in open("data/ocr/myscans/labels.txt", encoding="utf-8")]
real = rng.sample(real, min(40, len(real)))
rows_real = []
for k, (p, lab) in enumerate(real):
    im = cv2.imread(f"data/ocr/myscans/{p}")
    im = cv2.resize(im, (max(8, round(im.shape[1] * 48 / im.shape[0])), 48), interpolation=cv2.INTER_LINEAR)
    q = f"real_{k:03d}.jpg"
    cv2.imwrite(f"{OUT}/{q}", im)
    rows_real.append((q, lab, ""))


def section(title, rows):
    items = "\n".join(f'<div class="it"><img src="{p}"><div class="l">{html.escape(lab)}'
                      f'<span class="m">{html.escape(m)}</span></div></div>' for p, lab, m in rows)
    return f"<h2>{title}</h2>{items}"


page = f"""<!doctype html><html lang="vi"><head><meta charset="utf-8"><title>Xem trước dữ liệu FT2</title>
<style>body{{font-family:-apple-system,Arial,sans-serif;margin:16px;background:#fafafa}}
.it{{margin:10px 0}} img{{height:48px;max-width:100%;border:1px solid #ccc}} .l{{font-size:14px;color:#333}}
.m{{color:#999;margin-left:10px;font-size:12px}} .cols{{display:grid;grid-template-columns:1fr 1fr;gap:24px}}
@media(max-width:900px){{.cols{{grid-template-columns:1fr}}}}</style></head><body>
<p>Cả hai cột đều đã thu về cao 48 px như model nhìn thấy. Dòng chữ dưới mỗi ảnh là nhãn (đáp án).</p>
<div class="cols"><div>{section(f"Ảnh tổng hợp mới ({len(syn)})", syn)}</div>
<div>{section(f"Ảnh thật bạn chụp ({len(rows_real)}, lấy ngẫu nhiên)", rows_real)}</div></div></body></html>"""
open(f"{OUT}/index.html", "w", encoding="utf-8").write(page)
print(f"{len(syn)} anh tong hop + {len(rows_real)} anh that -> {OUT}/index.html")
