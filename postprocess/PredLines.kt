import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs

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

    // TODO: tự viết phần này theo ALGORITHM.md
    // Gợi ý thứ tự:
    //   1. Với mỗi điểm i trong 200 điểm: lấy y, x từ pts, score từ scores
    //   2. Đọc 4 giá trị độ dời của vmap tại (y, x)
    //   3. Tính độ dài đoạn, bỏ nếu score <= scoreThr hoặc độ dài <= distThr
    //   4. Tính 2 đầu mút trên lưới 256, rồi đổi về ảnh gốc (nhân 2, nhân tỉ lệ)
    //   5. Thêm vào lines

    return lines
}

// ===================== Phần dưới: đọc dữ liệu golden và so sánh, không cần sửa =====================

fun readFloats(path: String): FloatArray {
    val buf = ByteBuffer.wrap(File(path).readBytes()).order(ByteOrder.LITTLE_ENDIAN)
    return FloatArray(buf.remaining() / 4) { buf.float }
}

fun readInts(path: String): IntArray {
    val buf = ByteBuffer.wrap(File(path).readBytes()).order(ByteOrder.LITTLE_ENDIAN)
    return IntArray(buf.remaining() / 4) { buf.int }
}

fun main() {
    val meta = File("golden/meta.json").readText()
    val entries = Regex("\"name\": \"([^\"]+)\",\\s*\"h\": (\\d+),\\s*\"w\": (\\d+)").findAll(meta)
    var passed = 0
    var total = 0
    for (m in entries) {
        val (name, h, w) = m.destructured
        val pts = readInts("golden/${name}_pts.bin")
        val scores = readFloats("golden/${name}_scores.bin")
        val vmap = readFloats("golden/${name}_vmap.bin")
        val expected = readFloats("golden/${name}_lines.bin")
            .toList().chunked(4).map { Line(it[0], it[1], it[2], it[3]) }

        val got = predLines(pts, scores, vmap, h.toInt(), w.toInt())

        // So sánh: cùng số đoạn, và từng đoạn lệch không quá 0.01 pixel
        val sameCount = got.size == expected.size
        val maxErr = if (sameCount && got.isNotEmpty()) got.zip(expected).maxOf { (a, b) ->
            maxOf(abs(a.x1 - b.x1), abs(a.y1 - b.y1), abs(a.x2 - b.x2), abs(a.y2 - b.y2))
        } else 0f
        val ok = sameCount && maxErr < 0.01f
        total++
        if (ok) passed++
        println("%-28s %s  (Kotlin %d doan, Python %d doan, lech max %.4f px)"
            .format(name, if (ok) "PASS" else "FAIL", got.size, expected.size, maxErr))
    }
    println("\nKet qua: $passed/$total PASS")
}
