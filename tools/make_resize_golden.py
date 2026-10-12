# Sinh dữ liệu kiểm tra cho hàm resizeArea (Kotlin): ảnh nhiễu ngẫu nhiên nhiều kích thước,
# và kết quả cv2.resize(..., (512, 512), interpolation=cv2.INTER_AREA) tương ứng.
# Chạy từ thư mục gốc repo: python tools/make_resize_golden.py
import os
import json
import numpy as np
import cv2

os.makedirs("golden/resize", exist_ok=True)
rng = np.random.default_rng(0)
# (rộng, cao): đủ các nhánh của INTER_AREA
sizes = [
    (1024, 1024),  # thu nhỏ đúng 2 lần (nhánh "fast", 2x2)
    (1536, 1536),  # thu nhỏ đúng 3 lần (nhánh "fast")
    (4032, 3024),  # ảnh điện thoại 4:3, tỉ lệ lẻ
    (3024, 4032),  # ảnh chụp dọc
    (1920, 1080),  # khung hình video 16:9
    (2048, 1536),  # ảnh sau khi app thu nhỏ còn cạnh dài 2048
    (700, 1000),   # một chiều phóng to, một chiều thu nhỏ
    (300, 200),    # phóng to cả hai chiều
]
meta = []
for w, h in sizes:
    name = f"noise_{w}x{h}"
    img = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    img.tofile(f"golden/resize/{name}_rgb.bin")
    cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA).tofile(f"golden/resize/{name}_out.bin")
    meta.append({"name": name, "h": h, "w": w})
    print(name)
with open("golden/resize/meta.json", "w") as f:
    json.dump(meta, f, indent=1)
