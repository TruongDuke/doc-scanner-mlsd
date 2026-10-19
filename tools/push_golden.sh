#!/bin/bash
# Chép dữ liệu golden + ảnh mẫu vào bộ nhớ riêng của app (bản debug) trên điện thoại / máy ảo đang kết nối,
# cho nút "Kiểm tra golden" trong app. Đích: /data/data/com.truongduke.docscanner/files/golden/
# Chạy từ thư mục gốc repo, sau khi đã cài app ít nhất 1 lần: tools/push_golden.sh
# Nhiều thiết bị cùng cắm: chọn máy bằng biến ANDROID_SERIAL (xem tên bằng "adb devices"), ví dụ:
#   ANDROID_SERIAL=AIE00201260 tools/push_golden.sh
set -e
ADB=$(command -v adb || echo ~/Library/Android/sdk/platform-tools/adb)
PKG=com.truongduke.docscanner
TMP=/data/local/tmp/golden_push

FILES="golden/meta.json $(ls golden/*.bin) $(find samples -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \))"

# 1. Đẩy vào thư mục tạm (adb ghi được), 2. app tự chép sang bộ nhớ riêng bằng run-as (chạy với quyền của app)
"$ADB" shell "rm -rf $TMP && mkdir -p $TMP"
"$ADB" push $FILES "$TMP/" 2>&1 | tail -1
"$ADB" shell "run-as $PKG sh -c 'rm -rf files/golden && mkdir -p files/golden && cp $TMP/* files/golden/'"
"$ADB" shell "rm -rf $TMP"

# Golden OCR (tools/make_ocr_golden.py) -> files/golden/ocr/
if [ -d golden/ocr ]; then
    "$ADB" shell "rm -rf $TMP && mkdir -p $TMP"
    "$ADB" push golden/ocr/. "$TMP/" 2>&1 | tail -1
    "$ADB" shell "run-as $PKG sh -c 'mkdir -p files/golden/ocr && cp $TMP/* files/golden/ocr/'"
    "$ADB" shell "rm -rf $TMP"
    want=$(ls golden/ocr | wc -l | tr -d ' ')
    got=$("$ADB" shell "run-as $PKG ls files/golden/ocr" | wc -l | tr -d ' ')
    echo "OCR: $got/$want file tren may"
fi

# 3. Kiểm tra kích thước từng file trên máy khớp với file trên Mac
bad=0
for f in $FILES; do
    want=$(wc -c < "$f" | tr -d ' ')
    got=$("$ADB" shell "run-as $PKG stat -c %s files/golden/$(basename "$f")" 2>/dev/null | tr -d '\r')
    if [ "$want" != "$got" ]; then echo "SAI: $(basename "$f") tren may $got byte, tren Mac $want byte"; bad=1; fi
done
[ $bad = 0 ] && echo "Xong: $(echo $FILES | wc -w | tr -d ' ') file, kich thuoc khop het"
