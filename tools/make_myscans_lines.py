# Bộ test "ảnh chụp thật bằng app": tìm dòng chữ trên ảnh user scan (data/ocr/myscans/raw), cắt từng dòng,
# cho model đọc sẵn làm GỢI Ý, xuất trang HTML để user sửa đáp án (data/ocr/myscans/label.html).
#   - Tìm dòng: DBNet PP-OCRv5 mobile (qua RapidOCR, chỉ dùng phần tìm dòng)
#   - Gợi ý: model fine-tune tiếng Việt (ONNX), ảnh rộng theo độ dài dòng
#   - ROTATE: ảnh có chữ xoay dọc -> xoay cho chữ nằm ngang trước khi tìm dòng
#   - Đáp án nháp: data/ocr/myscans/my_transcripts.tsv (số_thứ_tự<TAB>U nếu chưa chắc<TAB>chữ), Claude đọc ảnh và gõ;
#     có file này thì trang HTML điền sẵn đáp án nháp (dòng chưa chắc tô vàng) thay cho gợi ý của model
# Chạy từ thư mục gốc repo (.venv):  .venv/bin/python tools/make_myscans_lines.py [--html-only] [--dir data/ocr/myscans_test]
#   --dir: thư mục bộ ảnh (có raw/ bên trong); mặc định data/ocr/myscans (bộ valid), myscans_test = bộ test (18/10)
import html
import json
import os
import sys

import cv2
import numpy as np
from rapidocr import LangDet, ModelType, OCRVersion, RapidOCR

sys.path.insert(0, "tools")
from ocr_eval import Recognizer, ctc_decode, load_dict, prep

OUT = sys.argv[sys.argv.index("--dir") + 1] if "--dir" in sys.argv else "data/ocr/myscans"
RAW = f"{OUT}/raw"
ROTATE = {"scan_1792075114813.jpg": cv2.ROTATE_90_CLOCKWISE}       # phong bì EMS: chữ chạy dọc
REC = ("data/models/ocr/vi_ft3.onnx", "data/ocr_eval/vi_dict.txt")

HTML_ONLY = "--html-only" in sys.argv     # chỉ làm lại trang HTML từ guesses.json, không tìm / cắt dòng lại

det = None if HTML_ONLY else RapidOCR(params={"Global.use_cls": False, "Global.use_rec": False, "Global.log_level": "error",
                       "Det.ocr_version": OCRVersion.PPOCRV5, "Det.model_type": ModelType.MOBILE, "Det.lang_type": LangDet.CH})
rec, chars = (None, None) if HTML_ONLY else (Recognizer(REC[0]), load_dict(*REC))


def crop_quad(img, q):
    q = np.asarray(q, np.float32)
    w = int(max(np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[3])))
    h = int(max(np.linalg.norm(q[3] - q[0]), np.linalg.norm(q[2] - q[1])))
    m = cv2.getPerspectiveTransform(q, np.float32([[0, 0], [w, 0], [w, h], [0, h]]))
    return cv2.warpPerspective(img, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


os.makedirs(f"{OUT}/lines", exist_ok=True)
items = []
for name in [] if HTML_ONLY else sorted(os.listdir(RAW)):
    img = cv2.imread(f"{RAW}/{name}")
    if name in ROTATE:
        img = cv2.rotate(img, ROTATE[name])
    boxes = det(img).boxes
    boxes = [] if boxes is None else sorted(boxes, key=lambda b: (round(min(p[1] for p in b) / 12), min(p[0] for p in b)))
    doc = name[5:-4]
    for k, b in enumerate(boxes):
        c = crop_quad(img, b)
        if c.shape[0] < 8 or c.shape[1] < 8:
            continue
        p = f"lines/{doc}_{k:03d}.jpg"
        cv2.imwrite(f"{OUT}/{p}", c, [cv2.IMWRITE_JPEG_QUALITY, 95])
        items.append({"img": p, "doc": doc, "guess": ctc_decode(rec(prep(c, 0)), chars).strip()})
    print(f"{name}: {len(boxes)} dong")
if HTML_ONLY:
    items = json.load(open(f"{OUT}/guesses.json"))
else:
    json.dump(items, open(f"{OUT}/guesses.json", "w"), ensure_ascii=False, indent=0)
draft, unsure = [it["guess"] for it in items], set()
if os.path.exists(f"{OUT}/my_transcripts.tsv"):
    for line in open(f"{OUT}/my_transcripts.tsv", encoding="utf-8"):
        n, u, text = line.rstrip("\n").split("\t")
        draft[int(n) - 1] = text
        if u:
            unsure.add(int(n) - 1)

# Trang HTML để sửa đáp án: mỗi dòng 1 ảnh + 1 ô chữ (điền sẵn gợi ý). Tự lưu nháp trong trình duyệt,
# nút "Tải file đáp án" -> labels.txt (đường_dẫn<TAB>chữ; dòng để trống hoặc ghi ### = bỏ qua).
rows = "\n".join(
    f'<div class="r{" u" if i in unsure else ""}"><div class="n">{i + 1}</div><div><img src="{html.escape(it["img"])}"><br>'
    f'<input data-img="{html.escape(it["img"])}" value="{html.escape(draft[i])}"></div></div>' for i, it in enumerate(items))
page = f"""<!doctype html><html lang="vi"><head><meta charset="utf-8"><title>Sửa đáp án OCR</title>
<style>
body{{font-family:-apple-system,Arial,sans-serif;margin:16px;background:#fafafa}}
.top{{position:sticky;top:0;background:#fff;padding:10px;border-bottom:1px solid #ccc;z-index:1}}
.r{{display:flex;gap:10px;margin:14px 0;align-items:flex-start}} .n{{width:40px;color:#888;padding-top:6px}}
img{{max-width:100%;max-height:70px;border:1px solid #ddd;background:#fff}}
input{{width:min(1000px,90vw);font-size:20px;padding:6px;margin-top:4px}}
button{{font-size:16px;padding:8px 14px}}
.u{{background:#fff3b0;padding:4px;border-radius:4px}}
</style></head><body>
<div class="top"><b>Duyệt đáp án:</b> ô đã điền sẵn đáp án Claude gõ; dòng <span class="u">tô vàng</span> là dòng Claude chưa chắc, xem kỹ. gõ lại đúng TỪNG KÝ TỰ chữ trong ảnh (hoa/thường, dấu, khoảng trắng, dấu câu).
Dòng không đọc được hoặc ảnh không phải chữ: để trống hoặc gõ <code>###</code>. Bài làm tự lưu nháp trong trình duyệt.
<button onclick="dl()">Tải file đáp án (labels.txt)</button> <span id="c"></span></div>
{rows}
<script>
const ins=[...document.querySelectorAll('input')];
const key='ocr-labels-v2-'+location.pathname;   // v2: bản nháp cũ (gợi ý của model) không đè lên đáp án mới
let saved={{}}; try{{saved=JSON.parse(localStorage.getItem(key)||'{{}}')}}catch(e){{}}
ins.forEach(i=>{{ if(saved[i.dataset.img]!==undefined) i.value=saved[i.dataset.img];
  i.addEventListener('input',()=>{{saved[i.dataset.img]=i.value; try{{localStorage.setItem(key,JSON.stringify(saved))}}catch(e){{}} }}); }});
document.getElementById('c').textContent=ins.length+' dòng';
function dl(){{ const t=ins.map(i=>i.dataset.img+'\\t'+i.value.trim()).join('\\n')+'\\n';
  const a=document.createElement('a'); a.href=URL.createObjectURL(new Blob([t],{{type:'text/plain'}})); a.download='labels.txt'; a.click(); }}
</script></body></html>"""
open(f"{OUT}/label.html", "w", encoding="utf-8").write(page)
print(f"\n{len(items)} dong -> {OUT}/label.html")
