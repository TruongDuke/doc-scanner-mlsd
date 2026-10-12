// Golden test cho predSquares: so kết quả Kotlin với kết quả pred_squares của repo gốc (golden/).
// Chạy từ thư mục gốc repo:
//   kotlinc postprocess/src/PredSquares.kt postprocess/test/GoldenSquares.kt -include-runtime -d build/predsquares.jar && java -jar build/predsquares.jar

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs
import mlsd.predSquares

fun readBuf(path: String): ByteBuffer = ByteBuffer.wrap(File(path).readBytes()).order(ByteOrder.LITTLE_ENDIAN)
fun readF32(path: String): FloatArray { val b = readBuf(path); return FloatArray(b.remaining() / 4) { b.float } }
fun readF64(path: String): DoubleArray { val b = readBuf(path); return DoubleArray(b.remaining() / 8) { b.double } }
fun readI32(path: String): IntArray { val b = readBuf(path); return IntArray(b.remaining() / 4) { b.int } }

fun main() {
    val meta = File("golden/meta.json").readText()
    val entries = Regex("\"name\": \"([^\"]+)\",\\s*\"h\": (\\d+),\\s*\"w\": (\\d+)").findAll(meta)
    var passed = 0
    var total = 0
    for (mt in entries) {
        val (name, h, w) = mt.destructured
        val g = "golden/$name"
        val r = predSquares(readI32("${g}_pts.bin"), readF32("${g}_scores.bin"), readF32("${g}_vmap.bin"), h.toInt(), w.toInt())

        val expSeg = readF64("${g}_sq_segments.bin").toList().chunked(4)
        val expSq = readI32("${g}_squares.bin").toList().chunked(8)
        val expScore = readF64("${g}_sq_scores.bin")
        val expInter = readI32("${g}_inter.bin").toList().chunked(2)

        // Đoạn gộp: lệch < 0.01 px. Góc và tứ giác: toạ độ nguyên phải khớp hẳn. Điểm: lệch < 1e-6
        val segOk = r.segments.size == expSeg.size &&
            r.segments.zip(expSeg).all { (a, b) -> a.indices.all { abs(a[it] - b[it]) < 0.01 } }
        val interOk = r.interPoints.size == expInter.size &&
            r.interPoints.zip(expInter).all { (a, b) -> a.toList() == b }
        // np.argsort không ổn định với mảng lớn: các tứ giác bằng điểm nhau có thể bị đảo chỗ.
        // Vì vậy so theo từng nhóm cùng điểm (như một tập), không so thứ tự bên trong nhóm.
        val sqOk = r.squares.size == expSq.size && expScore.size == expSq.size &&
            expScore.indices.groupBy { expScore[it] }.values.all { idx ->
                idx.map { r.squares[it].toList() }.sortedBy { it.toString() } == idx.map { expSq[it] }.sortedBy { it.toString() }
            }
        val scoreOk = r.scores.size == expScore.size &&
            r.scores.indices.all { abs(r.scores[it] - expScore[it]) < 1e-6 || (r.scores[it].isNaN() && expScore[it].isNaN()) }

        val ok = segOk && interOk && sqOk && scoreOk
        total++
        if (ok) passed++
        fun mark(b: Boolean) = if (b) "ok" else "SAI"
        println("%-28s %s  doan gop %d/%d %s | goc %d/%d %s | tu giac %d/%d %s | diem %s".format(
            name, if (ok) "PASS" else "FAIL",
            r.segments.size, expSeg.size, mark(segOk),
            r.interPoints.size, expInter.size, mark(interOk),
            r.squares.size, expSq.size, mark(sqOk), mark(scoreOk)))
        if (ok && r.squares.isNotEmpty()) println("    tu giac tot nhat: ${r.squares[0].toList()}")
    }
    println("\nKet qua: $passed/$total PASS")
}
