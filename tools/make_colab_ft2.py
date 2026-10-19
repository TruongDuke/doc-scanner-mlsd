# Tạo notebook Colab cho FT2 (colab/ocr_ft2.ipynb): sinh dữ liệu bản 2 + fine-tune tiếp từ FT1.
# Code sinh dữ liệu được chép NGUYÊN VĂN từ tools/ocr_synth.py và tools/make_ocr_train_v2.py (ô %%writefile),
# nên sửa code trong repo xong chỉ cần chạy lại script này:  .venv/bin/python tools/make_colab_ft2.py
import json

FT2_EPOCHS = 5


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(True)}


def writefile(path, header):
    return code(f"%%writefile {path.split('/')[-1]}\n# {header} (chép nguyên văn {path} trong repo)\n" + open(path, encoding="utf-8").read())


TRAIN_ARGS = f"""  Global.character_dict_path=/content/data_v2/vi_dict.txt \\
  Global.save_model_dir=$OUT/output_ft2 \\
  Global.epoch_num={FT2_EPOCHS} Global.save_epoch_step=1 Global.eval_batch_step=[0,1000] Global.distributed=false \\
  Optimizer.lr.learning_rate=0.0001 Optimizer.lr.warmup_epoch=1 \\
  Train.dataset.data_dir=/content/data_v2/ Train.dataset.label_file_list=[/content/data_v2/train.txt] \\
  Train.sampler.first_bs=64 Train.loader.batch_size_per_card=64 Train.loader.num_workers=2 \\
  Eval.dataset.data_dir=/content/data_v2/ Eval.dataset.label_file_list=[/content/data_v2/eval.txt] \\
  Eval.loader.num_workers=2 2>&1 | grep --line-buffered -E "load pretrain|not matched|best metric|cur metric|Error|error|global_step: [0-9]*00,\""""

cells = [
    md(f"""
# Mốc 8: FT2 = fine-tune lần 2 model đọc chữ tiếng Việt (cần GPU T4)

**Trước khi chạy:** Runtime → Change runtime type → **T4 GPU** → Save. Rồi chạy lần lượt từng ô.

Khác FT1:
- **Dữ liệu bản 2** (sửa theo lỗi của FT1 trên ảnh chụp thật): ô bảng 1-5 ký tự ("0", "29%"), dòng biểu mẫu
  (hàng chấm, số chú thích nhỏ), chữ đậm / nghiêng / IN HOA, font kiểu Times, nét dòng trên / dưới lọt vào mép ảnh,
  nền xám / xám xanh và tối hơn, mực mờ, đường kẻ sát chữ, chữ bị ép ngang.
- **Học tiếp từ FT1** (đã biết tiếng Việt), không học lại từ model latin gốc. Learning rate nhỏ hơn (0.0001), {FT2_EPOCHS} epoch,
  lưu trọng số **sau mỗi epoch** (`iter_epoch_1` ... `iter_epoch_{FT2_EPOCHS}`) để chọn epoch tốt nhất trên ảnh chụp thật (làm trên Mac),
  không tin `best_accuracy` (chỉ chấm trên ảnh tổng hợp).

Mọi thứ lưu vào `MyDrive/doc-scanner-ocr/` (dữ liệu: `data_v2.zip`, trọng số: `output_ft2/`): Colab ngắt giữa chừng không mất.
Thời gian dự kiến: sinh dữ liệu 15-25 phút (chỉ lần đầu), train khoảng 40-60 phút.
**Nếu Colab bị ngắt khi đang train:** chạy ô 1, ô 6 (giải nén dữ liệu có sẵn), ô 7, rồi ô **8b**.
"""),
    code("""
# 1. Kiểm tra GPU, kết nối Drive, kiểm tra trọng số FT1 (điểm xuất phát của FT2)
!nvidia-smi --query-gpu=name,memory.total --format=csv
from google.colab import drive
drive.mount("/content/drive")
OUT = "/content/drive/MyDrive/doc-scanner-ocr"
!ls -la $OUT/output/best_accuracy.pdparams
!pip -q install fonttools datasets
"""),
    writefile("tools/ocr_synth.py", "2. Code sinh ảnh dòng chữ"),
    writefile("tools/make_ocr_train_v2.py", "3. Code sinh bộ dữ liệu train / eval"),
    code("""
# 4. Font: cài các bộ font có tiếng Việt, chỉ giữ font vẽ được ĐỦ 236 ký tự.
#    fonts-liberation2: Liberation Serif / Sans bản 2.x (đủ chữ Việt; Serif = hình dáng Times New Roman, Sans = Arial)
#    ~20% họ font để riêng cho tập eval, nhưng Liberation luôn ở tập train (là font giấy tờ quan trọng nhất)
!apt-get -qq install -y fonts-liberation fonts-liberation2 fonts-dejavu-core fonts-noto-core fonts-roboto-unhinted fonts-open-sans > /dev/null
import glob, random, collections
from fontTools.ttLib import TTFont
from ocr_synth import vietnamese_dict, font_style
CHARS = vietnamese_dict()
ok = []
for f in sorted(glob.glob("/usr/share/fonts/truetype/**/*.ttf", recursive=True)):
    name = f.split("/")[-1]
    if any(k in name for k in ["Noto", "Liberation", "DejaVu", "Roboto", "OpenSans"]) and not any(k in name for k in ["Mono", "Condensed", "Thin", "ExtraLight", "Display", "Symbols", "Math", "Emoji", "Arabic", "CJK"]):
        try:
            cmap = TTFont(f, fontNumber=0).getBestCmap()
        except Exception:
            continue
        if all(ord(c) in cmap for c in CHARS):
            ok.append(f)
fam = lambda f: f.split("/")[-1].split("-")[0]
families = sorted(set(fam(f) for f in ok))
rng = random.Random(0)
cands = [x for x in families if not x.startswith("Liberation")]
eval_fams = set(rng.sample(cands, min(len(cands), max(1, len(families) // 5))))
assert eval_fams, "khong co ho font nao de lam tap eval"
TRAIN_FONTS = [f for f in ok if fam(f) not in eval_fams]
EVAL_FONTS = [f for f in ok if fam(f) in eval_fams]
print(len(ok), "font du ky tu | ho font eval (chua gap khi train):", sorted(eval_fams))
print("train theo kieu chu:", collections.Counter(font_style(f) for f in TRAIN_FONTS))
times = [f.split("/")[-1] for f in TRAIN_FONTS if "LiberationSerif" in f]
print("font kieu Times:", times)
assert {font_style(f) for f in TRAIN_FONTS if "LiberationSerif" in f} == {"regular", "bold", "italic", "bolditalic"}, "thieu Liberation Serif"
open("/content/train_fonts.txt", "w").write("\\n".join(TRAIN_FONTS))
open("/content/eval_fonts.txt", "w").write("\\n".join(EVAL_FONTS))
"""),
    code("""
# 5. Văn bản: đoạn văn Wikipedia tiếng Việt (đọc dần, không tải cả bộ), lưu ra file, khoảng 3 triệu ký tự
from datasets import load_dataset
from ocr_synth import clean
CS = set(CHARS)
wiki = load_dataset("wikimedia/wikipedia", "20231101.vi", split="train", streaming=True)
n = 0
with open("/content/wiki_vi.txt", "w") as f:
    for art in wiki:
        for para in art["text"].split("\\n"):
            para = clean(para, CS)
            if len(para) > 30:
                f.write(para + "\\n"); n += len(para)
        if n > 3_000_000:
            break
print(n, "ky tu")
"""),
    code("""
# 6. Dữ liệu: nếu Drive đã có data_v2.zip (lần chạy trước) thì chỉ giải nén; chưa có thì sinh 60.000 dòng train
#    + 1.500 dòng eval (font chưa gặp) rồi nén lưu lên Drive. Sinh mất khoảng 15-25 phút.
import os, subprocess
def sh(cmd):                                        # chạy lệnh shell, lỗi thì dừng ngay (không chạy tiếp với dữ liệu hỏng)
    # (Colab không hiện chữ in ra trực tiếp từ subprocess -> lấy về rồi tự in)
    print("$", cmd[:200])
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(r.stdout[-3000:], r.stderr[-3000:])
    r.check_returncode()
if os.path.exists(f"{OUT}/data_v2.zip"):
    sh(f"cd /content && rm -rf data_v2 && unzip -q {OUT}/data_v2.zip")
else:
    sh("cd /content && python make_ocr_train_v2.py --out /content/data_v2 --n 60000 --n-eval 1500 --text /content/wiki_vi.txt"
       " --fonts $(cat /content/train_fonts.txt) --eval-fonts $(cat /content/eval_fonts.txt)")
    sh(f"cd /content && zip -q -r data_v2.zip data_v2 && cp data_v2.zip {OUT}/")
sh(f"wc -l /content/data_v2/train.txt /content/data_v2/eval.txt /content/data_v2/vi_dict.txt && ls -la {OUT} {OUT}/output_ft2")
# Xem thử 10 ảnh + đáp án
from PIL import Image
from IPython.display import display
for line in open("/content/data_v2/train.txt").readlines()[:10]:
    p, t = line.rstrip("\\n").split("\\t")
    print(repr(t)); display(Image.open("/content/data_v2/" + p))
"""),
    code("""
# 7. Cài PaddlePaddle bản GPU + mã nguồn PaddleOCR v3.7.0 (giống FT1, khoảng 3-5 phút)
!pip -q install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
!git clone -q --depth 1 --branch v3.7.0 https://github.com/PaddlePaddle/PaddleOCR.git /content/PaddleOCR
!pip -q install -r /content/PaddleOCR/requirements.txt
import paddle
paddle.utils.run_check()
"""),
    code(f"""
# 8. Train FT2: học tiếp từ trọng số FT1 ($OUT/output/best_accuracy), lưu vào $OUT/output_ft2
#   Không được thấy dòng "not matched" (FT1 và FT2 cùng từ điển 236 ký tự -> mọi lớp phải nạp được)
#   Theo dõi chi tiết: mở Terminal của Colab, gõ  tail -n 8 /content/drive/MyDrive/doc-scanner-ocr/output_ft2/train.log
%cd /content/PaddleOCR
!python tools/train.py -c configs/rec/PP-OCRv5/multi_language/latin_PP-OCRv5_mobile_rec.yml -o \\
  Global.pretrained_model=$OUT/output/best_accuracy \\
{TRAIN_ARGS}
"""),
    code(f"""
# 8b. TRAIN TIẾP sau khi Colab bị ngắt (chạy ô 1, 6, 7 trước; KHÔNG chạy lại ô 8)
%cd /content/PaddleOCR
!python tools/train.py -c configs/rec/PP-OCRv5/multi_language/latin_PP-OCRv5_mobile_rec.yml -o \\
  Global.checkpoints=$OUT/output_ft2/latest \\
{TRAIN_ARGS}
"""),
    code("""
# 9. Kiểm tra đã đủ trọng số từng epoch trên Drive
!ls -la $OUT/output_ft2/ | grep pdparams
"""),
    md("""
**Xong:** tải thư mục `MyDrive/doc-scanner-ocr/output_ft2/` (chỉ cần các file `iter_epoch_*.pdparams` và `best_accuracy.pdparams`)
về máy, báo Claude. Trên Mac sẽ đổi từng epoch sang ONNX rồi chấm trên ảnh chụp thật (`myscans`) và văn bản nhà nước (`govdocs`)
để chọn epoch tốt nhất.
"""),
]

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"}},
      "nbformat": 4, "nbformat_minor": 0}
json.dump(nb, open("colab/ocr_ft2.ipynb", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("-> colab/ocr_ft2.ipynb,", len(cells), "o")
