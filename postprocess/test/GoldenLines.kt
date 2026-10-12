// Golden test cho predLines: so kết quả Kotlin với kết quả pred_lines của repo gốc (golden/).
// Chạy từ thư mục gốc repo:
//   kotlinc postprocess/src/PredLines.kt postprocess/test/GoldenLines.kt -include-runtime -d build/predlines.jar && java -jar build/predlines.jar

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs
import mlsd.Line
import mlsd.predLines

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
