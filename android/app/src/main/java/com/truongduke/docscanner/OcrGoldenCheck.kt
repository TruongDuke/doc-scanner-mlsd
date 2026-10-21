package com.truongduke.docscanner

import android.content.Context
import org.json.JSONArray
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.text.Normalizer

// Nút "Kiểm tra OCR": model đọc chữ INT8 chạy trên điện thoại có ra cùng kết quả như trên Mac không, đọc đúng bao nhiêu,
// mất bao lâu mỗi dòng. Dữ liệu: tools/make_ocr_golden.py -> golden/ocr/, chép vào máy bằng tools/push_golden.sh
//   (/data/data/com.truongduke.docscanner/files/golden/ocr/)
// Mỗi dòng: đưa vào model ĐÚNG đầu vào đã chuẩn bị trên Mac ({tên}_input.bin) -> so ký tự điểm cao nhất từng lát với
// Mac ({tên}_idx.bin), so chữ đọc ra, chấm CER theo đáp án (cả kết quả máy lẫn kết quả Mac).

private fun levenshtein(a: String, b: String): Int {
    var prev = IntArray(b.length + 1) { it }
    for (i in 1..a.length) {
        val cur = IntArray(b.length + 1)
        cur[0] = i
        for (j in 1..b.length) cur[j] = minOf(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + if (a[i - 1] == b[j - 1]) 0 else 1)
        prev = cur
    }
    return prev[b.length]
}

private fun nfc(s: String) = Normalizer.normalize(s, Normalizer.Form.NFC)

fun runOcrGoldenCheck(context: Context, rec: OcrRecognizer): String {
    val dir = File(goldenDir(context), "ocr")
    val metaFile = File(dir, "meta.json")
    if (!metaFile.exists()) return "Chưa có dữ liệu golden OCR trong máy.\nTrên Mac: tools/make_ocr_golden.py rồi tools/push_golden.sh"
    val meta = JSONArray(metaFile.readText())

    var sameIdx = 0
    var sameText = 0
    var errDev = 0
    var errMac = 0
    var nChars = 0
    val times = HashMap<Int, MutableList<Long>>()
    val modelTimes = HashMap<Int, MutableList<Long>>()
    val diffs = StringBuilder()
    for (i in 0 until meta.length()) {
        val m = meta.getJSONObject(i)
        val name = m.getString("name")
        val w = m.getInt("w")
        val gt = nfc(m.getString("gt"))
        val macText = nfc(m.getString("mac_int8"))
        val input = ByteBuffer.wrap(File(dir, "${name}_input.bin").readBytes()).order(ByteOrder.LITTLE_ENDIAN)
            .let { b -> FloatArray(b.remaining() / 4) { b.float } }
        val macIdx = ByteBuffer.wrap(File(dir, "${name}_idx.bin").readBytes()).order(ByteOrder.LITTLE_ENDIAN)
            .let { b -> IntArray(b.remaining() / 4) { b.int } }

        rec.runIdx(input, w)                       // lần đầu mỗi dòng: làm nóng (lần đầu mỗi cỡ còn tính cả nạp model)
        val t0 = System.nanoTime()
        val idx = rec.runIdx(input, w)
        times.getOrPut(w) { mutableListOf() }.add((System.nanoTime() - t0) / 1_000_000)
        modelTimes.getOrPut(w) { mutableListOf() }.add(rec.lastModelMs.toLong())
        val text = nfc(rec.decode(idx))

        if (idx.contentEquals(macIdx)) sameIdx++
        if (text == macText) sameText++ else if (diffs.length < 1500) diffs.append("$name Mac: $macText\n     máy: $text\n")
        errDev += levenshtein(text, gt)
        errMac += levenshtein(macText, gt)
        nChars += gt.length
    }
    val n = meta.length()
    val sb = StringBuilder("Kiểm tra OCR (INT8, ${n} dòng, CPU 4 luồng, API ${if (rec.compiled) "CompiledModel" else "Interpreter"})\n")
    sb.append("Chuỗi ký tự từng lát trùng Mac: $sameIdx/$n\n")
    sb.append("Chữ đọc ra trùng Mac: $sameText/$n\n")
    sb.append("CER theo đáp án: máy %.2f%% | Mac %.2f%%\n".format(errDev * 100.0 / nChars, errMac * 100.0 / nChars))
    sb.append("Thời gian 1 dòng, nhanh nhất / trung vị (riêng model: nhanh nhất):\n")
    for (w in times.keys.sorted()) {
        val t = times.getValue(w).sorted()
        val mt = modelTimes.getValue(w).sorted()
        sb.append("  rộng %4d: %4d / %4d ms (model %4d ms, %d dòng)\n".format(w, t.first(), t[t.size / 2], mt.first(), t.size))
    }
    if (diffs.isNotEmpty()) sb.append("\nDòng khác Mac:\n").append(diffs)
    return sb.toString()
}

// Nút "Kiểm tra OCR trang": cả chuỗi OcrPage (DBNet -> khung xoay -> cắt dòng -> đọc) trên điểm ảnh trang y hệt Mac
// (tools/make_ocr_page_golden.py -> golden/ocr_page/), so với kết quả bản Python (tools/det_simple.py).
// Không mong khớp tuyệt đối: co ảnh của Android (Bitmap.createScaledBitmap) và cv2.resize nội suy hơi khác nhau.
fun runOcrPageGoldenCheck(context: Context, page: OcrPage): String {
    val dir = File(goldenDir(context), "ocr_page")
    val metaFile = File(dir, "meta.json")
    if (!metaFile.exists()) return "Chưa có golden OCR trang.\nTrên Mac: tools/make_ocr_page_golden.py rồi tools/push_golden.sh"
    val meta = JSONArray(metaFile.readText())
    val sb = StringBuilder("Kiểm tra OCR trang (${meta.length()} trang)\n")
    fun bounds(c: FloatArray) = floatArrayOf(minOf(c[0], c[2], c[4], c[6]), minOf(c[1], c[3], c[5], c[7]),
        maxOf(c[0], c[2], c[4], c[6]), maxOf(c[1], c[3], c[5], c[7]))
    fun iou(a: FloatArray, b: FloatArray): Float {
        val ix = maxOf(0f, minOf(a[2], b[2]) - maxOf(a[0], b[0]))
        val iy = maxOf(0f, minOf(a[3], b[3]) - maxOf(a[1], b[1]))
        val inter = ix * iy
        return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-6f)
    }
    for (i in 0 until meta.length()) {
        val m = meta.getJSONObject(i)
        val h = m.getInt("h")
        val w = m.getInt("w")
        val rgb = File(dir, "${m.getString("name")}_rgb.bin").readBytes()
        val px = IntArray(w * h) { k ->
            (0xff shl 24) or ((rgb[3 * k].toInt() and 0xff) shl 16) or ((rgb[3 * k + 1].toInt() and 0xff) shl 8) or (rgb[3 * k + 2].toInt() and 0xff)
        }
        val bmp = android.graphics.Bitmap.createBitmap(px, w, h, android.graphics.Bitmap.Config.ARGB_8888)
        page.read(bmp)                                   // làm nóng (lần đầu còn tính nạp model)
        val r = page.read(bmp)
        val pyBoxes = m.getJSONArray("boxes").let { a -> List(a.length()) { k -> a.getJSONArray(k).let { b -> FloatArray(8) { b.getDouble(it).toFloat() } } } }
        val pyTexts = m.getJSONArray("texts").let { a -> List(a.length()) { a.getString(it) } }.filter { it.isNotEmpty() }
        val mine = r.lines.map { bounds(it.box.corners) }
        val matched = pyBoxes.count { pb -> mine.any { iou(bounds(pb), it) >= 0.5f } }
        val pyAll = nfc(pyTexts.joinToString("\n"))
        val myAll = nfc(r.text)
        sb.append("■ ${m.getString("name")} ($w x $h)\n")
        sb.append("  dòng: máy ${r.lines.size}, Mac ${pyTexts.size}; khung Mac tìm lại được $matched/${pyBoxes.size}\n")
        sb.append("  dòng chữ trùng hệt Mac: ${r.lines.count { it.text in pyTexts }}/${pyTexts.size}; " +
            "chữ cả trang lệch Mac %.2f%%\n".format(levenshtein(myAll, pyAll) * 100.0 / pyAll.length))
        // Chấm theo đáp án (trang có đáp án): mỗi khung Mac có đáp án -> dòng của máy trùng khung nhất (IoU >= 0.5)
        val gts = m.optJSONArray("gts")
        if (gts != null) {
            val pyAllTexts = m.getJSONArray("texts").let { a -> List(a.length()) { a.getString(it) } }
            var eDev = 0; var eMac = 0; var nGt = 0; var nLine = 0
            for (k in 0 until gts.length()) {
                if (gts.isNull(k)) continue
                val gt = nfc(gts.getString(k))
                val pb = bounds(pyBoxes[k])
                val best = r.lines.maxByOrNull { iou(bounds(it.box.corners), pb) }
                val dev = if (best != null && iou(bounds(best.box.corners), pb) >= 0.5f) best.text else ""
                eDev += levenshtein(dev, gt); eMac += levenshtein(nfc(pyAllTexts[k]), gt); nGt += gt.length; nLine++
            }
            if (nLine > 0) sb.append("  CER theo đáp án ($nLine dòng): máy %.2f%% | Mac %.2f%%\n".format(eDev * 100.0 / nGt, eMac * 100.0 / nGt))
        }
        sb.append("  thời gian: tìm dòng ${r.detMs} ms, xử lý khung ${r.postMs} ms, đọc chữ ${r.recMs} ms\n")
        sb.append("  3 dòng đầu: ${r.lines.take(3).joinToString(" | ") { it.text }}\n")
        bmp.recycle()
    }
    return sb.toString()
}
