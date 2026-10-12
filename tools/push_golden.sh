#!/bin/bash
# Chép dữ liệu golden + ảnh mẫu vào bộ nhớ riêng của app (bản debug) trên điện thoại / máy ảo đang kết nối,
# cho nút "Kiểm tra golden" trong app. Đích: /data/data/com.truongduke.docscanner/files/golden/
# Chạy từ thư mục gốc repo, sau khi đã cài app ít nhất 1 lần: tools/push_golden.sh
set -e
ADB=$(command -v adb || echo ~/Library/Android/sdk/platform-tools/adb)
PKG=com.truongduke.docscanner
"$ADB" shell run-as $PKG mkdir -p files/golden
n=0
for f in golden/meta.json golden/*.bin $(find samples -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \)); do
    # run-as chạy lệnh với quyền của app, nên ghi được vào bộ nhớ riêng của app
    "$ADB" exec-in run-as $PKG sh -c "cat > files/golden/$(basename "$f")" < "$f"
    n=$((n + 1))
done
echo "Da chep $n file vao files/golden cua app"
