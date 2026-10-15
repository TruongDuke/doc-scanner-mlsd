# Mốc 8, bước 3: sinh dữ liệu dòng chữ tiếng Việt để fine-tune model đọc chữ (PaddleOCR rec).
# Dùng chung cho notebook Colab (colab/ocr_buoc3_data.ipynb) và để thử nhanh trên Mac.
#   - vietnamese_dict(): từ điển ký tự đầy đủ (chữ có dấu hoa + thường, số, dấu câu)
#   - doc_phrases(): cụm từ hay gặp trên giấy tờ / hoá đơn (họ tên, địa chỉ, ngày, số tiền...)
#   - chunk_text(): cắt câu dài thành đoạn vừa 1 dòng (<= 25 ký tự, giới hạn của PaddleOCR rec)
#   - render_line(): vẽ 1 dòng chữ bằng 1 font rồi "làm xấu" cho giống ảnh chụp thật
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
