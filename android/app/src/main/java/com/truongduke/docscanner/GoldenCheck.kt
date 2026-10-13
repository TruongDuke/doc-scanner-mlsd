package com.truongduke.docscanner

import android.content.Context
import android.graphics.BitmapFactory
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs
import kotlin.math.max
import mlsd.predLines
import mlsd.predSquares

// Kiểm tra ngay trên điện thoại / máy ảo: app có ra cùng kết quả với bản Python (golden) không.
// Dữ liệu golden + ảnh mẫu được chép vào bộ nhớ riêng của app bằng tools/push_golden.sh, nằm ở
//   /data/data/com.truongduke.docscanner/files/golden/
//
// Với mỗi ảnh mẫu kiểm tra 3 việc:
//   (A) đưa vào model đúng ảnh 512x512 mà cv2 tạo ra ({name}_input.bin)
//       -> kiểm tra riêng model + predLines / predSquares trên máy
//   (B) từ ảnh RGB do cv2 giải mã ({name}_rgb.bin): resizeArea -> model -> post-processing
//       -> kiểm tra thêm resizeArea trên máy; đo thời gian từng bước
//   (C) từ file JPEG, giải mã bằng Android (như khi người dùng chọn ảnh)
//       -> chỉ để tham khảo: bộ giải mã JPEG mỗi máy có thể khác cv2 một chút (vd. chip MediaTek),
//          nên (C) không tính PASS/FAIL, chỉ báo mức lệch

private fun buf(f: File) = ByteBuffer.wrap(f.readBytes()).order(ByteOrder.LITTLE_ENDIAN)
private fun f32(f: File) = buf(f).let { b -> FloatArray(b.remaining() / 4) { b.float } }
private fun f64(f: File) = buf(f).let { b -> DoubleArray(b.remaining() / 8) { b.double } }
private fun i32(f: File) = buf(f).let { b -> IntArray(b.remaining() / 4) { b.int } }

fun goldenDir(context: Context) = File(context.filesDir, "golden")

fun runGoldenCheck(context: Context, model: MlsdModel): String {
    val dir = goldenDir(context)
    val metaFile = File(dir, "meta.json")
    if (!metaFile.exists()) return "Chưa có dữ liệu golden trong máy.\nChạy trên Mac: tools/push_golden.sh"

    val sb = StringBuilder()
    var passA = 0
    var passB = 0
    var total = 0
    var sameC = 0
    var totalC = 0
    val metaRe = Regex("\"name\": \"([^\"]+)\",\\s*\"h\": (\\d+),\\s*\"w\": (\\d+)")
    for (m in metaRe.findAll(metaFile.readText())) {
        val (name, hs, ws) = m.destructured
        val h = hs.toInt()
        val w = ws.toInt()
        fun g(suffix: String) = File(dir, "${name}_$suffix.bin")
        total++
        sb.append("■ $name ($w x $h)\n")

        val expLines = f32(g("lines")).toList().chunked(4)
        val expSq = i32(g("squares")).toList().chunked(8)
        val expScore = f64(g("sq_scores"))

        // So kết quả post-processing với golden, trả về (khớp?, mô tả)
        fun comparePost(pts: IntArray, scores: FloatArray, vmap: FloatArray): Pair<Boolean, String> {
            val lines = predLines(pts, scores, vmap, h, w)
            val linesErr = if (lines.size == expLines.size && lines.isNotEmpty()) lines.zip(expLines).maxOf { (a, b) ->
                maxOf(abs(a.x1 - b[0]), abs(a.y1 - b[1]), abs(a.x2 - b[2]), abs(a.y2 - b[3]))
            } else 0f
            val linesOk = lines.size == expLines.size && linesErr < 0.01f

            val sq = predSquares(pts, scores, vmap, h, w)
            // Tứ giác bằng điểm nhau có thể đổi chỗ (np.argsort không ổn định), nên so theo nhóm cùng điểm
            val sqOk = sq.squares.size == expSq.size && expScore.indices.groupBy { expScore[it] }.values.all { idx ->
                idx.map { sq.squares[it].toList() }.sortedBy { it.toString() } == idx.map { expSq[it] }.sortedBy { it.toString() }
            }
            val text = "đoạn ${lines.size}/${expLines.size} (lệch %.4f px), tứ giác ${sq.squares.size}/${expSq.size}"
                .format(linesErr)
            return Pair(linesOk && sqOk, text)
        }

        // ----- (A) model trên đúng đầu vào của cv2 -----
        val out = model.runRgb(g("input").readBytes(), 512, 512)
        val expPts = i32(g("pts"))
        val ptsSame = out.pts.indices.count { out.pts[it] == expPts[it] } / 2
        val expScores = f32(g("scores"))
        val scoreErr = out.scores.indices.maxOf { abs(out.scores[it] - expScores[it]) }
        val expVmap = f32(g("vmap"))
        var vmapErr = 0f
        for (k in expVmap.indices) vmapErr = max(vmapErr, abs(out.vmap[k] - expVmap[k]))
        val (okA, postA) = comparePost(out.pts, out.scores, out.vmap)
        if (okA) passA++
        sb.append("  (A) ${if (okA) "PASS" else "FAIL"}  pts trùng $ptsSame/200, lệch score %.1e, lệch vmap %.1e\n".format(scoreErr, vmapErr))
        sb.append("      $postA\n")

        // ----- (B) từ ảnh RGB của cv2: resizeArea + model + post-processing, kèm thời gian -----
        val rgbPy = g("rgb").readBytes()
        val outB = model.runRgb(rgbPy, w, h)
        val tPost0 = System.nanoTime()
        val (okB, postB) = comparePost(outB.pts, outB.scores, outB.vmap)
        val postMs = (System.nanoTime() - tPost0) / 1_000_000
        if (okB) passB++
        sb.append("  (B) ${if (okB) "PASS" else "FAIL"}  $postB\n")
        sb.append("      thời gian: tiền xử lý ${outB.prepMs} ms, model ${outB.timeMs} ms, post-processing $postMs ms\n")

        // ----- (C) từ file JPEG, giải mã bằng Android (tham khảo) -----
        val imgFile = dir.listFiles()?.firstOrNull {
            it.nameWithoutExtension == name && it.extension.lowercase() in listOf("jpg", "jpeg", "png")
        } ?: continue   // ảnh trơn "blank" do Python tự tạo, không có file ảnh
        val bmp = BitmapFactory.decodeFile(imgFile.path)
        val rgb = bitmapToRgb(bmp)
        var nDiff = 0
        var maxDiff = 0
        if (rgb.size == rgbPy.size) for (k in rgb.indices) {
            val d = abs((rgb[k].toInt() and 0xFF) - (rgbPy[k].toInt() and 0xFF))
            if (d != 0) nDiff++
            maxDiff = max(maxDiff, d)
        }
        val outC = model.runRgb(rgb, bmp.width, bmp.height)
        val (okC, postC) = comparePost(outC.pts, outC.scores, outC.vmap)
        if (okC) sameC++
        totalC++
        val decodeText = if (rgb.size != rgbPy.size) "kích thước ảnh khác Python!"
        else "giải mã lệch $nDiff/${rgb.size} giá trị (max $maxDiff)"
        sb.append("  (C) ${if (okC) "giống" else "khác"}  $decodeText\n      $postC\n")
    }
    sb.insert(0, "(A) model + post-processing:     $passA/$total PASS\n" +
        "(B) resize + model + post:        $passB/$total PASS\n" +
        "(C) từ JPEG (giải mã của máy):    $sameC/$totalC giống Python (tham khảo)\n\n")
    return sb.toString()
}
