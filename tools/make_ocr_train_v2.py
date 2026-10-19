# Sinh dữ liệu train bản 2 (FT2) theo định dạng PaddleOCR: <out>/{train,eval}/*.jpg + train.txt, eval.txt, vi_dict.txt
# Dùng chung cho Colab (sinh 60.000 dòng) và Mac (sinh vài trăm dòng để chạy thử train trước khi lên Colab).
#   --fonts: các file font train (Colab: Liberation / DejaVu / Noto / Roboto / OpenSans; Mac: Times New Roman / Arial)
#   --eval-fonts: font chỉ dùng cho tập eval (chưa gặp khi train)
#   --text: file văn bản, mỗi dòng 1 đoạn (Wikipedia tiếng Việt)
# Ví dụ trên Mac (thư mục gốc repo):
#   .venv/bin/python tools/make_ocr_train_v2.py --out data/ocr_train_v2_mini --n 512 --n-eval 64 \
#     --text data/ocr_eval/wiki_vi.txt --fonts "/System/Library/Fonts/Supplemental/Times New Roman*.ttf" \
#     "/System/Library/Fonts/Supplemental/Arial.ttf" "/System/Library/Fonts/Supplemental/Arial B*.ttf" \
#     "/System/Library/Fonts/Supplemental/Arial I*.ttf" --eval-fonts "data/ocr_eval/fonts/*.ttf"
import argparse
import glob
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ocr_synth import group_fonts, label_of, make_pool_v2, pick_font, render_line_v2, sample_text_v2, vietnamese_dict

NEIGHBOR_P = 0.5          # 50% dòng (trừ ô bảng) vẽ kèm dòng trên / dưới


def expand(patterns):
    return sorted({f for p in patterns for f in glob.glob(p, recursive=True)})


def make(split, n, groups, pool, rng, out):
    os.makedirs(f"{out}/{split}", exist_ok=True)
    cs = set(vietnamese_dict()) | {" "}
    with open(f"{out}/{split}.txt", "w", encoding="utf-8") as lab:
        for k in range(n):
            text = sample_text_v2(pool, rng)
            nb = (rng.choice(pool), rng.choice(pool)) if rng.random() < NEIGHBOR_P and len(label_of(text)) > 5 else None
            img = render_line_v2(text, pick_font(groups, rng), rng, neighbors=nb)
            label = label_of(text)
            assert label.strip() and all(c in cs for c in label), repr(label)
            p = f"{split}/{k:06d}.jpg"
            img.save(f"{out}/{p}", quality=92)
            lab.write(f"{p}\t{label}\n")
            if k % 10000 == 0:
                print(split, k, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=60000)
    ap.add_argument("--n-eval", type=int, default=1500)
    ap.add_argument("--text", required=True)
    ap.add_argument("--fonts", nargs="+", required=True)
    ap.add_argument("--eval-fonts", nargs="+", required=True)
    ap.add_argument("--seed", type=int, default=2)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    fonts, eval_fonts = expand(args.fonts), expand(args.eval_fonts)
    groups, eval_groups = group_fonts(fonts), group_fonts(eval_fonts)
    print("font train:", {s: len(v) for s, v in groups.items()}, "| font eval:", {s: len(v) for s, v in eval_groups.items()})
    paragraphs = [l.strip() for l in open(args.text, encoding="utf-8") if l.strip()]
    pool = make_pool_v2(paragraphs, rng, n_doc=max(1000, args.n // 3))
    print(len(pool), "dong van ban")
    os.makedirs(args.out, exist_ok=True)
    make("train", args.n, groups, pool, rng, args.out)
    make("eval", args.n_eval, eval_groups, pool, random.Random(args.seed + 1000), args.out)
    with open(f"{args.out}/vi_dict.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(vietnamese_dict()) + "\n")
    print("xong ->", args.out)


if __name__ == "__main__":
    main()
