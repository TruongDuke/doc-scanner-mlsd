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
