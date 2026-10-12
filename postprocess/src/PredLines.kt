package mlsd

import kotlin.math.sqrt

// Một đoạn thẳng: 2 đầu mút (x1, y1) và (x2, y2), toạ độ trên ảnh gốc
data class Line(val x1: Float, val y1: Float, val x2: Float, val y2: Float)

/**
 * Viết lại pred_lines của repo M-LSD chính thức (mlsd-official/utils.py, dòng 12-47).
 *
 * pts:    200 điểm ứng viên làm center, lưu phẳng [y0, x0, y1, x1, ...]  (độ dài 400), toạ độ trên lưới 256x256
 * scores: độ tin cậy của 200 điểm (độ dài 200)
 * vmap:   bản đồ độ dời, lưu phẳng theo thứ tự (hàng y, cột x, kênh c): vmap[(y * 256 + x) * 4 + c]
 *         với c = 0..3 là (dx_start, dy_start, dx_end, dy_end)
 * imgH, imgW: kích thước ảnh gốc
 */
fun predLines(
    pts: IntArray,
    scores: FloatArray,
    vmap: FloatArray,
    imgH: Int,
    imgW: Int,
    scoreThr: Float = 0.1f,
    distThr: Float = 20.0f,
): List<Line> {
    val lines = mutableListOf<Line>()

    // Tỉ lệ từ ảnh 512x512 (đầu vào model) về ảnh gốc
    val wRatio = imgW / 512f
    val hRatio = imgH / 512f

    for (i in 0 until 200) {
        // 1. Lấy toạ độ điểm thứ i. pts lưu phẳng [y0, x0, y1, x1, ...]
        val y = pts[i * 2]
        val x = pts[i * 2 + 1]
        val score = scores[i]

        // 2. Đọc 4 giá trị độ dời của vmap tại ô (y, x)
        val base = (y * 256 + x) * 4
        val dxStart = vmap[base + 0]
        val dyStart = vmap[base + 1]
        val dxEnd = vmap[base + 2]
        val dyEnd = vmap[base + 3]

        // 3. Độ dài đoạn = khoảng cách giữa 2 độ dời (giống dist_map bên Python)
        val dist = sqrt((dxStart - dxEnd) * (dxStart - dxEnd) + (dyStart - dyEnd) * (dyStart - dyEnd))

        // 4. Lọc: Python dùng dấu > cho cả hai điều kiện
        if (score > scoreThr && dist > distThr) {
            // 5. Hai đầu mút trên lưới 256 = center + độ dời
            val x1 = x + dxStart
            val y1 = y + dyStart
            val x2 = x + dxEnd
            val y2 = y + dyEnd

            // 6. Đổi về ảnh gốc: nhân 2 (256 → 512) rồi nhân tỉ lệ, đúng thứ tự như Python
            lines.add(Line(2 * x1 * wRatio, 2 * y1 * hRatio, 2 * x2 * wRatio, 2 * y2 * hRatio))
        }
    }

    return lines
}
