# Mốc 8, bước 3: sinh dữ liệu dòng chữ tiếng Việt để fine-tune model đọc chữ (PaddleOCR rec).
# Dùng chung cho notebook Colab (colab/ocr_buoc3_data.ipynb) và để thử nhanh trên Mac.
#   - vietnamese_dict(): từ điển ký tự đầy đủ (chữ có dấu hoa + thường, số, dấu câu)
#   - doc_phrases(): cụm từ hay gặp trên giấy tờ / hoá đơn (họ tên, địa chỉ, ngày, số tiền...)
#   - chunk_text(): cắt câu dài thành đoạn vừa 1 dòng (<= 25 ký tự, giới hạn của PaddleOCR rec)
#   - render_line(): vẽ 1 dòng chữ bằng 1 font rồi "làm xấu" cho giống ảnh chụp thật
# Bản 2 (cho FT2, giữ nguyên bản 1 ở trên để tạo lại đúng dữ liệu FT1), sửa theo lỗi của FT1 trên ảnh chụp thật:
#   - sample_text_v2(): thêm ô bảng 1-5 ký tự, dòng biểu mẫu (hàng chấm, số chú thích nhỏ), dòng in hoa
#   - group_fonts() / pick_font(): chọn font theo tỉ lệ thường / đậm / nghiêng, ưu tiên font kiểu Times
#   - render_line_v2(): nền xám / xám xanh và tối hơn, mực nhạt / đứt nét, nét của dòng trên / dưới lọt vào mép ảnh,
#     đường kẻ sát chữ, chữ bị ép ngang
import random
import unicodedata

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

MAX_LEN = 25

# ---------------- Từ điển ----------------
TONES = ["", "́", "̀", "̉", "̃", "̣"]   # không dấu, sắc, huyền, hỏi, ngã, nặng
VOWELS = "aăâeêioôơuưy"


def vietnamese_dict():
    chars = list("0123456789")
    chars += list("abcdefghijklmnopqrstuvwxyz") + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    for v in VOWELS:
        for t in TONES[1:] + ([""] if v in "ăâêôơư" else []):
            c = unicodedata.normalize("NFC", v + t)
            chars += [c, c.upper()]
    chars += ["đ", "Đ"]
    chars += list(".,:;!?-–/\\()[]{}'\"%&@#*+=_<>~^$|°·…“”‘’") + ["₫"]
    seen, out = set(), []
    for c in chars:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out          # khoảng trắng KHÔNG nằm trong file; PaddleOCR thêm khi use_space_char = True


# ---------------- Văn bản giống giấy tờ ----------------
HO = "Nguyễn Trần Lê Phạm Hoàng Huỳnh Phan Vũ Võ Đặng Bùi Đỗ Hồ Ngô Dương Lý Trương Đinh Lâm Mai".split()
DEM = "Văn Thị Hữu Đức Minh Thanh Ngọc Quốc Xuân Thu Hồng Gia Bảo Công Thái".split()
TEN = ("An Anh Bình Chi Cường Dũng Duy Giang Hà Hải Hạnh Hiếu Hoa Hùng Hương Khánh Lan Linh Long Mai Nam "
       "Ngân Nhung Phong Phúc Phương Quân Quang Sơn Tâm Thảo Thắng Trang Trung Tuấn Tùng Uyên Việt Vy Yến").split()
DUONG = ("Lê Lợi|Trần Hưng Đạo|Nguyễn Trãi|Hai Bà Trưng|Láng Hạ|Cầu Giấy|Kim Mã|Nguyễn Văn Cừ|Lý Thường Kiệt|"
         "Điện Biên Phủ|Phạm Văn Đồng|Võ Thị Sáu|Ngụy Như Kon Tum|Xuân Thủy|Hoàng Quốc Việt").split("|")
QUAN = ("Ba Đình|Hoàn Kiếm|Đống Đa|Cầu Giấy|Thanh Xuân|Hai Bà Trưng|Hoàng Mai|Long Biên|Tây Hồ|Hà Đông|"
        "Quận 1|Quận 3|Bình Thạnh|Gò Vấp|Thủ Đức|Hải Châu|Ninh Kiều").split("|")
TINH = "Hà Nội|TP. Hồ Chí Minh|Đà Nẵng|Hải Phòng|Cần Thơ|Huế|Nghệ An|Thanh Hóa|Quảng Ninh|Bắc Ninh".split("|")
NHAN = ("Họ và tên|Ngày sinh|Giới tính|Quốc tịch|Nơi thường trú|Số định danh cá nhân|Số điện thoại|Email|"
        "Tổng tiền|Thành tiền|Đơn giá|Số lượng|Thanh toán tiền mặt|Hóa đơn thanh toán|Thu ngân|Giờ vào|Giờ ra|"
        "Kính gửi|Nội dung đề nghị|Người khai|Xác nhận của cơ quan|Ngày cấp|Nơi cấp|Mã số thuế|Ghi chú").split("|")
CAU = ["CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM", "Độc lập - Tự do - Hạnh phúc", "TỜ KHAI THAY ĐỔI THÔNG TIN CƯ TRÚ",
       "HÓA ĐƠN GIÁ TRỊ GIA TĂNG", "Cảm ơn quý khách", "Hẹn gặp lại", "Trân trọng cảm ơn", "Chữ ký của người khai"]


def doc_phrases(rng):
    r = rng.random()
    name = f"{rng.choice(HO)} {rng.choice(DEM)} {rng.choice(TEN)}"
    if r < 0.2:
        return name if rng.random() < 0.5 else name.upper()
    if r < 0.4:
        return f"{rng.randint(1, 300)} {rng.choice(DUONG)}, {rng.choice(QUAN)}, {rng.choice(TINH)}"
    if r < 0.55:
        return f"{rng.choice(NHAN)}: {rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/{rng.randint(1950, 2026)}"
    if r < 0.7:
        return f"{rng.choice(NHAN)}: {rng.randint(1, 999) * 1000:,} đ".replace(",", ".")
    if r < 0.85:
        return f"{rng.choice(NHAN)}: {name}"
    return rng.choice(CAU)


def doc_lines(rng):
    # Cụm từ giấy tờ, cắt cho vừa 1 dòng
    t = doc_phrases(rng)
    return [t] if len(t) <= MAX_LEN else chunk_text(t, rng)


def clean(text, charset):
    text = unicodedata.normalize("NFC", " ".join(text.split()))
    return "".join(c for c in text if c in charset or c == " ").strip()


def chunk_text(text, rng, max_len=MAX_LEN):
    # Cắt theo từ thành đoạn ngẫu nhiên dài 5..max_len ký tự
    words, out, cur = text.split(), [], ""
    limit = rng.randint(8, max_len)
    for w in words:
        if len(cur) + len(w) + (1 if cur else 0) <= limit:
            cur = f"{cur} {w}" if cur else w
        else:
            if cur:
                out.append(cur)
            cur, limit = w[:max_len], rng.randint(8, max_len)
    if cur:
        out.append(cur)
    return out


# ---------------- Vẽ và làm xấu ảnh ----------------
def render_line(text, font_path, rng, height=48):
    size = rng.randint(26, 44)
    font = ImageFont.truetype(font_path, size)
    probe = ImageDraw.Draw(Image.new("L", (1, 1)))
    l, t, r, b = probe.textbbox((0, 0), text, font=font)
    pad_x, pad_y = rng.randint(2, 14), rng.randint(2, 10)
    w, h = r - l + 2 * pad_x, b - t + 2 * pad_y
    bg = rng.randint(170, 255)                      # nền giấy: trắng / xám / ngả màu
    tint = np.array([bg, bg - rng.randint(0, 20), bg - rng.randint(0, 35)]).clip(0, 255)
    img = Image.new("RGB", (w, h), tuple(int(x) for x in tint))
    ink = rng.randint(0, 90)                        # mực: đen / xám đậm / xanh đậm
    color = (ink, ink, ink) if rng.random() < 0.8 else (ink // 3, ink // 3, rng.randint(60, 140))
    ImageDraw.Draw(img).text((pad_x - l, pad_y - t), text, font=font, fill=color)
    if rng.random() < 0.5:                          # nghiêng nhẹ
        img = img.rotate(rng.uniform(-2.5, 2.5), resample=Image.BILINEAR, expand=True, fillcolor=tuple(int(x) for x in tint))
    arr = np.asarray(img).astype(np.float32)
    if rng.random() < 0.6:                          # nhiễu
        arr += np.random.default_rng(rng.randint(0, 1 << 30)).normal(0, rng.uniform(2, 12), arr.shape)
    if rng.random() < 0.4:                          # độ sáng / tương phản
        arr = (arr - 128) * rng.uniform(0.6, 1.2) + 128 + rng.uniform(-30, 30)
    img = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
    if rng.random() < 0.5:                          # mờ
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.3)))
    if rng.random() < 0.4:                          # mất chi tiết: thu nhỏ rồi phóng lại
        f = rng.uniform(0.4, 0.8)
        small = img.resize((max(1, int(img.width * f)), max(1, int(img.height * f))), Image.BILINEAR)
        img = small.resize(img.size, Image.BILINEAR)
    nw = max(8, round(img.width * height / img.height))
    return img.resize((nw, height), Image.BILINEAR)


# ======================= Bản 2 (FT2) =======================
MAX_LEN_V2 = 25          # giữ như FT1 (cấu hình train: ảnh rộng 320, tối đa 25 ký tự). Đo FT1 theo độ dài dòng (16/10):
                         # dòng 66-85 ký tự không kém hơn (govdocs cam 3.5%), dòng NGẮN mới yếu (myscans 1-25 ký tự: 14-18%)
SUP = "\x01"             # bao quanh đoạn chữ nhỏ đặt cao (số chú thích): "Kính gửi\x01(1)\x01:" -> nhãn "Kính gửi(1):"
FORM = NHAN + ("Họ, chữ đệm và tên|Ngày, tháng, năm sinh|Nơi ở hiện nay|Nghề nghiệp|Đơn vị công tác|Dân tộc|"
               "Tôn giáo|Quê quán|Số căn cước|Mối quan hệ với chủ hộ|Lý do|Địa chỉ liên hệ|Trình độ học vấn").split("|")


def label_of(text):
    return text.replace(SUP, "")


def make_pool_v2(paragraphs, rng, n_doc=20000):
    # Câu văn (bỏ ký tự ngoài từ điển, cắt dài 8..25 ký tự) + cụm từ giấy tờ (cắt cho vừa 1 dòng)
    cs = set(vietnamese_dict())
    pool = [c for p in paragraphs for c in chunk_text(clean(p, cs), rng, MAX_LEN_V2)]
    pool += [l for _ in range(n_doc) for l in doc_lines(rng)]
    return [t for t in pool if len(t) >= 2]


def cell_text(rng):
    # Ô bảng: 1-5 ký tự đứng riêng (ảnh thật: "0", "1", "17,5", "29%", "TT"; FT1 đọc "0" thành "(")
    r = rng.random()
    if r < 0.35:
        return str(rng.randint(0, 9))
    if r < 0.55:
        return str(rng.randint(10, 999))
    if r < 0.7:
        return f"{rng.randint(0, 100)}%"
    if r < 0.85:
        return f"{rng.randint(0, 99)},{rng.randint(0, 9)}"
    return rng.choice(["TT", "STT", "a)", "b)", "c)", "(1)", "(2)", "(3)", "I", "II", "III", "IV", "x", "%"])


def form_line(rng):
    # Dòng biểu mẫu: "3. Ngày, tháng, năm sinh(2):........", ".....ngày....tháng....năm...."
    dots = lambda: "." * rng.randint(1, 8)
    if rng.random() < 0.15:
        return f"{dots()}ngày{dots()}tháng{dots()}năm{dots()}"
    head = rng.choice(["", "", f"{rng.randint(1, 15)}. ", f"{rng.choice('abcdđe')}) ", "+ ", "- ", "– ", f"({rng.randint(1, 9)}) "])
    lab = rng.choice(FORM)
    if rng.random() < 0.25:
        lab += SUP + f"({rng.randint(1, 9)})" + SUP
    if rng.random() < 0.3:                                       # ô đã điền
        return f"{head}{lab}: {doc_phrases(rng).split(': ')[-1]}"
    return head + lab + rng.choice([":", ":", ""]) + "." * rng.choice([0, 0, rng.randint(1, 4), rng.randint(5, 40)])


def sample_text_v2(pool, rng):
    # 10% ô bảng, 15% dòng biểu mẫu, 15% dòng IN HOA, 60% câu văn thường
    r = rng.random()
    if r < 0.10:
        return cell_text(rng)
    if r < 0.25:
        t = form_line(rng)
    else:
        t = rng.choice(pool)
        t = t.upper() if r < 0.40 else t
    if len(label_of(t)) > MAX_LEN_V2:                            # dòng biểu mẫu dài: bớt hàng chấm, không thì cắt theo từ
        excess = len(label_of(t)) - MAX_LEN_V2
        t = t[:-excess] if len(label_of(t.rstrip("."))) <= MAX_LEN_V2 else chunk_text(label_of(t), rng)[0]
    return t


# ---------------- Font ----------------
STYLE_WEIGHTS = {"regular": 55, "bold": 25, "italic": 15, "bolditalic": 5}
TIMES_LIKE = ("liberationserif", "timesnewroman", "tinos")


def font_style(path):
    n = path.rsplit("/", 1)[-1].lower().replace(" ", "")
    s = ("bold" if "bold" in n else "") + ("italic" if "italic" in n or "oblique" in n else "")
    return s or "regular"


def group_fonts(paths):
    groups = {}
    for p in paths:
        groups.setdefault(font_style(p), []).append(p)
    return groups


def pick_font(groups, rng):
    # Bốc kiểu chữ theo STYLE_WEIGHTS, rồi 40% lấy font kiểu Times (giấy tờ Việt Nam hay dùng; "1" giống "I")
    styles = [s for s in STYLE_WEIGHTS if groups.get(s)]
    fs = groups[rng.choices(styles, weights=[STYLE_WEIGHTS[s] for s in styles])[0]]
    times = [f for f in fs if any(k in f.lower().replace(" ", "") for k in TIMES_LIKE)]
    return rng.choice(times) if times and rng.random() < 0.4 else rng.choice(fs)


# ---------------- Vẽ ----------------
def _draw_segments(draw, x, y, text, font, small):
    # Vẽ chữ; đoạn nằm giữa 2 dấu SUP vẽ bằng font nhỏ, cùng mép trên -> trông như số chú thích đặt cao
    for i, seg in enumerate(text.split(SUP)):
        f = small if i % 2 else font
        draw.text((x, y), seg, font=f, fill=255)
        x += f.getlength(seg)
    return x


def render_line_v2(text, font_path, rng, height=48, neighbors=None):
    # neighbors = (dòng trên, dòng dưới): vẽ như trên trang giấy (cách nhau 1.1-1.4 lần cỡ chữ) rồi cắt sát dòng chính
    # -> nét của dòng bên cạnh lọt vào mép ảnh như khi bộ tìm dòng cắt ảnh thật. Nhãn vẫn chỉ là dòng chính.
    size = rng.randint(26, 44)
    font = ImageFont.truetype(font_path, size)
    small = ImageFont.truetype(font_path, max(10, int(size * 0.6)))
    plain = label_of(text)
    l, t, r, b = font.getbbox(plain)
    if SUP in text:
        r = l + int(sum((small if i % 2 else font).getlength(s) for i, s in enumerate(text.split(SUP)))) + 2
    short = len(plain) <= 5
    if short:                                                    # ô bảng: cắt sát
        pad_x, pad_y, neighbors = rng.randint(1, 4), rng.randint(1, 5), None
    elif neighbors:
        pad_x, pad_y = rng.randint(2, 14), int((b - t) * rng.uniform(0.15, 0.4))
    else:
        pad_x, pad_y = rng.randint(2, 14), rng.randint(2, 10)
    w, h = r - l + 2 * pad_x, b - t + 2 * pad_y
    ox, oy = pad_x - l, pad_y - t

    # 1) Vẽ "mặt nạ mực" (255 = có mực), gồm cả dòng bên cạnh nếu có
    mask = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(mask)
    _draw_segments(d, ox, oy, text, font, small)
    if neighbors:
        gap = int(size * rng.uniform(1.1, 1.4))
        above, below = neighbors
        if above:
            d.text((ox, oy - gap), label_of(above), font=font, fill=255)
        if below:
            d.text((ox, oy + gap), label_of(below), font=font, fill=255)
    if rng.random() < 0.1:                                       # đường kẻ: gạch chân / viền ô bảng sát chữ
        text_bottom = pad_y + (b - t)
        y = rng.randint(min(text_bottom + 1, h - 1), h - 1) if rng.random() < 0.7 else rng.randint(0, max(0, pad_y - 1))
        x0, x1 = (0, w) if rng.random() < 0.6 else (rng.randint(0, w // 3), rng.randint(2 * w // 3, w))
        d.line((x0, y, x1, y), fill=255, width=rng.randint(1, 3))
    if size >= 32 and rng.random() < 0.1:                        # nét mảnh đi
        mask = mask.filter(ImageFilter.MinFilter(3))
    a = np.asarray(mask, np.float32) / 255
    g = np.random.default_rng(rng.randint(0, 1 << 30))
    if rng.random() < 0.2:                                       # mực in mờ loang lổ + đứt nét
        field = Image.fromarray((g.random((h // 6 + 1, w // 6 + 1)) * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
        lo = rng.uniform(0.25, 0.6)
        a *= lo + (1 - lo) * np.asarray(field, np.float32) / 255
        a *= g.random((h, w)) > rng.uniform(0, 0.15)

    # 2) Tô màu: nền (có cả nền xám tối) + mực (luôn tối hơn nền ít nhất 60 mức)
    bg = rng.randint(110, 170) if rng.random() < 0.3 else rng.randint(170, 255)
    if rng.random() < 0.8:                                       # xám / xám ngả xanh như camera điện thoại (RGB)
        tint = np.array([bg - rng.randint(0, 15), bg - rng.randint(0, 8), bg], np.float32)
    else:                                                        # hơi vàng: giấy cũ, đèn vàng
        tint = np.array([bg, bg - rng.randint(0, 15), bg - rng.randint(5, 35)], np.float32)
    tint = tint.clip(0, 255)
    ink = rng.randint(0, min(140, bg - 60))
    color = np.array((ink, ink, ink) if rng.random() < 0.8 else (ink // 3, ink // 3, rng.randint(60, 140)), np.float32)
    arr = tint * (1 - a[..., None]) + color * a[..., None]
    img = Image.fromarray(arr.clip(0, 255).astype(np.uint8))

    # Ép ngang cho chữ hẹp và sát nhau (tiêu đề in hoa đậm trong biểu mẫu); dòng in hoa hay bị hơn
    if rng.random() < (0.4 if plain.isupper() else 0.1):
        img = img.resize((max(4, int(w * rng.uniform(0.75, 0.9))), h), Image.BILINEAR)

    # 3) Làm xấu giống bản 1: nghiêng nhẹ, nhiễu, sáng / tương phản, mờ, mất chi tiết
    if rng.random() < 0.5:
        img = img.rotate(rng.uniform(-2.5, 2.5), resample=Image.BILINEAR, expand=True, fillcolor=tuple(int(x) for x in tint))
    arr = np.asarray(img).astype(np.float32)
    if rng.random() < 0.6:
        arr += g.normal(0, rng.uniform(2, 12), arr.shape)
    if rng.random() < 0.4:
        arr = (arr - 128) * rng.uniform(0.7, 1.2) + 128 + rng.uniform(-30, 30)
    img = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
    if rng.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.3)))
    if rng.random() < 0.4:
        f = rng.uniform(0.4, 0.8)
        lowres = img.resize((max(1, int(img.width * f)), max(1, int(img.height * f))), Image.BILINEAR)
        img = lowres.resize(img.size, Image.BILINEAR)
    nw = max(8, round(img.width * height / img.height))
    return img.resize((nw, height), Image.BILINEAR)
