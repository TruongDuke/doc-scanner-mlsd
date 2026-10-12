// Golden test cho resizeArea: so với cv2.resize(..., (512, 512), interpolation=cv2.INTER_AREA).
// Cần chạy trước: python tools/make_golden.py và python tools/make_resize_golden.py
// Chạy từ thư mục gốc repo:
//   kotlinc postprocess/src/ResizeArea.kt postprocess/test/GoldenResize.kt -include-runtime -d build/resize.jar && java -jar build/resize.jar

import java.io.File
import kotlin.math.abs
import kotlin.math.max
import mlsd.resizeArea

fun main() {
    // (tên, đường dẫn ảnh gốc, đường dẫn kết quả cv2, rộng, cao)
    val cases = mutableListOf<List<Any>>()
    val metaRe = Regex("\"name\": \"([^\"]+)\",\\s*\"h\": (\\d+),\\s*\"w\": (\\d+)")
    for (m in metaRe.findAll(File("golden/meta.json").readText())) {
        val (name, h, w) = m.destructured
        cases.add(listOf(name, "golden/${name}_rgb.bin", "golden/${name}_input.bin", w.toInt(), h.toInt()))
    }
    for (m in metaRe.findAll(File("golden/resize/meta.json").readText())) {
        val (name, h, w) = m.destructured
        cases.add(listOf(name, "golden/resize/${name}_rgb.bin", "golden/resize/${name}_out.bin", w.toInt(), h.toInt()))
    }

    run {
        var passed = 0
        for ((name, rgbPath, outPath, w, h) in cases) {
            val src = File(rgbPath as String).readBytes()
            val expected = File(outPath as String).readBytes()
            val got = resizeArea(src, w as Int, h as Int, 512, 512)
            var nDiff = 0
            var maxDiff = 0
            for (k in expected.indices) {
                val d = abs((got[k].toInt() and 0xFF) - (expected[k].toInt() and 0xFF))
                if (d != 0) nDiff++
                maxDiff = max(maxDiff, d)
            }
            val ok = nDiff == 0
            if (ok) passed++
            println("%-26s %4dx%-4d %s  (lech %d / %d gia tri, lech max %d)".format(
                name, w, h, if (ok) "PASS" else "FAIL", nDiff, expected.size, maxDiff))
        }
        println("\nKet qua: $passed/${cases.size} PASS")
    }
}
