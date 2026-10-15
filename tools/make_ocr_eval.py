# Tạo bộ đánh giá OCR cố định data/ocr_eval/synth_v1 (1.500 dòng, có đáp án) để chấm MỌI model bằng tools/ocr_eval.py:
#   - 750 dòng văn bản Wikipedia tiếng Việt (60 bài ngẫu nhiên, lưu sẵn ở data/ocr_eval/wiki_vi.txt) + 750 dòng cụm từ giấy tờ
#   - font NotoSerif thường / nghiêng (KHÔNG có trong dữ liệu train), làm xấu ảnh giống dữ liệu train (tools/ocr_synth.py)
#   - hạt giống cố định: chạy lại cho ra đúng bộ cũ (khi wiki_vi.txt đã có)
# Font: data/ocr_eval/fonts/NotoSerif[wdth,wght].ttf, NotoSerif-Italic[wdth,wght].ttf (github.com/google/fonts, ofl/notoserif)
# Chạy từ thư mục gốc repo:  .venv/bin/python tools/make_ocr_eval.py
import json, os, random, sys, time, urllib.error, urllib.parse, urllib.request
sys.path.insert(0, "tools")
from ocr_synth import vietnamese_dict, clean, chunk_text, doc_lines, render_line

CS = set(vietnamese_dict())
cache = "data/ocr_eval/wiki_vi.txt"
if not os.path.exists(cache):
    q = urllib.parse.urlencode({"action": "query", "format": "json", "generator": "random", "grnnamespace": 0,
                                "grnlimit": 1, "prop": "extracts", "explaintext": 1})
    paras, got = [], 0
    while got < 60:          # mỗi lần 1 bài, nghỉ giữa các lần để Wikipedia không chặn (HTTP 429)
        req = urllib.request.Request("https://vi.wikipedia.org/w/api.php?" + q, headers={"User-Agent": "doc-scanner-mlsd/0.1 (personal study project)"})
        try:
            d = json.load(urllib.request.urlopen(req, timeout=30))
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(20)
                continue
            raise
        for pg in d["query"]["pages"].values():
            paras += [p for p in pg.get("extract", "").split("\n") if len(p) > 40]
        got += 1
        time.sleep(1.5)
    open(cache, "w").write("\n".join(paras))
paras = open(cache).read().split("\n")
rng = random.Random(2026)
wiki = [t for p in paras for t in chunk_text(clean(p, CS), rng) if len(t) >= 3]
rng.shuffle(wiki)
doc = [l for _ in range(900) for l in doc_lines(rng)]
texts = wiki[:750] + doc[:750]
fonts = ["data/ocr_eval/fonts/NotoSerif[wdth,wght].ttf", "data/ocr_eval/fonts/NotoSerif-Italic[wdth,wght].ttf"]
out = "data/ocr_eval/synth_v1"
os.makedirs(out + "/img", exist_ok=True)
with open(out + "/labels.txt", "w") as f:
    for k, t in enumerate(texts):
        render_line(t, fonts[0] if rng.random() < 0.7 else fonts[1], rng).save(f"{out}/img/{k:04d}.jpg", quality=92)
        f.write(f"img/{k:04d}.jpg\t{t}\n")
print(f"{len(texts)} dong -> {out}")
