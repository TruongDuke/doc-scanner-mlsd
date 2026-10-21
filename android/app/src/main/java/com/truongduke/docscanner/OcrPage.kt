package com.truongduke.docscanner

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapShader
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Shader
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer
import java.text.Normalizer
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.ceil
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sin

// Đọc chữ cả trang (đã nắn thẳng): tìm dòng bằng DBNet (PP-OCRv5 mobile det, FP32) -> cắt từng dòng -> OcrRecognizer (INT8).
// Cùng các bước với tools/det_simple.py (bản Python, đã so với RapidOCR trên 14 trang: tìm lại 98.9% dòng; đọc cả chuỗi
// CER 5.81% so với 5.07% khi dùng khung của RapidOCR):
//   1. ảnh co (không giữ tỉ lệ) về cỡ cố định của model: trang dọc 1408 x 992 (cao x rộng), trang ngang 992 x 1408;
//      thứ tự kênh B, G, R như RapidOCR (cv2), chuẩn hoá (x/255 - 0.5)/0.5
//   2. mask = xác suất > 0.3, nở 2x2 (cv2.dilate, tâm kernel (1,1): điểm (x,y) = max của (x-1..x, y-1..y))
//   3. vùng liền nhau 8 hướng; bỏ vùng < 9 điểm hoặc xác suất TB trên các điểm < 0.5
//   4. hướng chính của vùng (PCA trên toạ độ điểm; nghiêng > 20 độ -> coi như nằm ngang, vd. ô số "1"), chiếu điểm lên
//      2 trục -> khung xoay dài L, dày T; nới ra d = L*T*1.6 / (2(L+T)) mỗi phía (unclip của RapidOCR)
//   5. đổi về toạ độ ảnh gốc, sắp thứ tự đọc (trên -> dưới; lệch y < 10 px thì trái -> phải)
//   6. nắn khung xoay về ảnh ngang (affine, mép ngoài lấy điểm ảnh gần nhất), cao 48 -> OcrRecognizer
class OcrPage(private val context: Context, val rec: OcrRecognizer = OcrRecognizer(context)) {
    // 4 góc trên ảnh gốc: trên-trái, trên-phải, dưới-phải, dưới-trái (theo hướng dòng)
    class Box(val corners: FloatArray, val score: Float) {
        val x0 get() = minOf(corners[0], corners[2], corners[4], corners[6])
        val y0 get() = minOf(corners[1], corners[3], corners[5], corners[7])
    }

    class Line(val box: Box, val text: String)
    class Result(val lines: List<Line>, val detMs: Long, val postMs: Long, val recMs: Long) {
        val text get() = lines.joinToString("\n") { it.text }
    }

    private class Det(val cm: CompiledModel, val inBufs: List<TensorBuffer>, val outBufs: List<TensorBuffer>)

    private val dets = HashMap<String, Det>()

    private fun det(h: Int, w: Int): Det = synchronized(dets) {
        dets.getOrPut("${h}x$w") {
            val opts = CompiledModel.Options(Accelerator.CPU)
            opts.cpuOptions = CompiledModel.CpuOptions(4, null, null)
            val cm = CompiledModel.create(context.assets, "ocr_det_${h}x$w.tflite", opts, rec.env)   // môi trường dùng chung, xem OcrRecognizer
            Det(cm, cm.createInputBuffers(), cm.createOutputBuffers())
        }
    }

    // Bước 1: bản đồ xác suất (H x W) của trang
    fun probMap(page: Bitmap): Triple<FloatArray, Int, Int> {
        val (h, w) = if (page.height >= page.width) 1408 to 992 else 992 to 1408
        val small = Bitmap.createScaledBitmap(page, w, h, true)
        val px = IntArray(w * h)
        small.getPixels(px, 0, w, 0, 0, w, h)
        if (small !== page) small.recycle()
        val x = FloatArray(w * h * 3)
        for (i in px.indices) {
            val c = px[i]
            x[3 * i] = ((c and 0xff) / 255f - 0.5f) / 0.5f               // B
            x[3 * i + 1] = (((c shr 8) and 0xff) / 255f - 0.5f) / 0.5f   // G
            x[3 * i + 2] = (((c shr 16) and 0xff) / 255f - 0.5f) / 0.5f  // R
        }
        val d = det(h, w)
        d.inBufs[0].writeFloat(x)
        d.cm.run(d.inBufs, d.outBufs)
        return Triple(d.outBufs[0].readFloat(), h, w)
    }

    // Bước 2-5: khung các dòng (toạ độ ảnh gốc srcW x srcH), đã sắp thứ tự đọc
    fun boxes(p: FloatArray, h: Int, w: Int, srcW: Int, srcH: Int): List<Box> {
        val raw = BooleanArray(w * h) { p[it] > 0.3f }
        val mask = BooleanArray(w * h)
        for (y in 0 until h) for (x in 0 until w) {
            val i = y * w + x
            mask[i] = raw[i] || (x > 0 && raw[i - 1]) || (y > 0 && raw[i - w]) || (x > 0 && y > 0 && raw[i - w - 1])
        }
        val seen = BooleanArray(w * h)
        val stack = IntArray(w * h)
        val pix = IntArray(w * h)
        val sx = srcW.toFloat() / w
        val sy = srcH.toFloat() / h
        val maxTilt = Math.toRadians(20.0)
        val out = ArrayList<Box>()
        for (start in 0 until w * h) {
            if (!mask[start] || seen[start]) continue
            // Vùng liền nhau 8 hướng (tô loang bằng ngăn xếp)
            var n = 0
            var top = 0
            stack[top++] = start
            seen[start] = true
            while (top > 0) {
                val i = stack[--top]
                pix[n++] = i
                val x = i % w
                val y = i / w
                for (dy in -1..1) for (dx in -1..1) {
                    val xx = x + dx
                    val yy = y + dy
                    if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue
                    val j = yy * w + xx
                    if (mask[j] && !seen[j]) { seen[j] = true; stack[top++] = j }
                }
            }
            if (n < 9) continue
            var sp = 0.0; var mx = 0.0; var my = 0.0
            for (k in 0 until n) { val i = pix[k]; sp += p[i]; mx += i % w; my += i / w }
            if (sp / n < 0.5) continue
            mx /= n; my /= n
            var sxx = 0.0; var syy = 0.0; var sxy = 0.0
            for (k in 0 until n) {
                val dx = pix[k] % w - mx
                val dy = pix[k] / w - my
                sxx += dx * dx; syy += dy * dy; sxy += dx * dy
            }
            var ang = 0.5 * atan2(2 * sxy / n, sxx / n - syy / n)
            if (abs(ang) > maxTilt) ang = 0.0
            val ux = cos(ang)
            val uy = sin(ang)
            var u0 = Double.MAX_VALUE; var u1 = -Double.MAX_VALUE; var v0 = Double.MAX_VALUE; var v1 = -Double.MAX_VALUE
            for (k in 0 until n) {
                val dx = pix[k] % w - mx
                val dy = pix[k] / w - my
                val u = dx * ux + dy * uy
                val v = -dx * uy + dy * ux
                u0 = min(u0, u); u1 = max(u1, u); v0 = min(v0, v); v1 = max(v1, v)
            }
            u0 -= 0.5; u1 += 0.5; v0 -= 0.5; v1 += 0.5
            val len = u1 - u0
            val thick = v1 - v0
            if (min(len, thick) < 3) continue
            val d = len * thick * 1.6 / (2 * (len + thick))
            u0 -= d; u1 += d; v0 -= d; v1 += d
            if (min(u1 - u0, v1 - v0) < 5) continue
            val c = FloatArray(8)
            for ((k, uv) in listOf(u0 to v0, u1 to v0, u1 to v1, u0 to v1).withIndex()) {
                val (a, b) = uv
                c[2 * k] = ((mx + a * ux - b * uy) * sx).toFloat().coerceIn(0f, srcW.toFloat())
                c[2 * k + 1] = ((my + a * uy + b * ux) * sy).toFloat().coerceIn(0f, srcH.toFloat())
            }
            out.add(Box(c, (sp / n).toFloat()))
        }
        // Thứ tự đọc (như sorted_boxes của RapidOCR)
        out.sortWith(compareBy<Box>({ it.y0 }, { it.x0 }))
        for (i in 0 until out.size - 1) {
            var j = i
            while (j >= 0 && abs(out[j + 1].y0 - out[j].y0) < 10 && out[j + 1].x0 < out[j].x0) {
                val t = out[j]; out[j] = out[j + 1]; out[j + 1] = t
                j--
            }
        }
        return out
    }

    // Bước 6: nắn khung xoay về ảnh dòng nằm ngang
    fun cropLine(page: Bitmap, box: Box): Bitmap {
        val c = box.corners
        var tl = 0; var tr = 1; var br = 2; var bl = 3
        if (c[0] > c[2]) { tl = 2; tr = 3; br = 0; bl = 1 }           // hướng chính ngược chiều: đổi để chữ không bị lật
        fun pt(k: Int) = c[2 * k] to c[2 * k + 1]
        val wd = max(1, max(hypot(pt(tr).first - pt(tl).first, pt(tr).second - pt(tl).second),
            hypot(pt(br).first - pt(bl).first, pt(br).second - pt(bl).second)).roundToInt())
        val ht = max(1, max(hypot(pt(bl).first - pt(tl).first, pt(bl).second - pt(tl).second),
            hypot(pt(br).first - pt(tr).first, pt(br).second - pt(tr).second)).roundToInt())
        val m = Matrix()
        m.setPolyToPoly(floatArrayOf(pt(tl).first, pt(tl).second, pt(tr).first, pt(tr).second, pt(bl).first, pt(bl).second), 0,
            floatArrayOf(0f, 0f, wd.toFloat(), 0f, 0f, ht.toFloat()), 0, 3)
        val out = Bitmap.createBitmap(wd, ht, Bitmap.Config.ARGB_8888)
        val paint = Paint(Paint.FILTER_BITMAP_FLAG)
        paint.shader = BitmapShader(page, Shader.TileMode.CLAMP, Shader.TileMode.CLAMP).also { it.setLocalMatrix(m) }
        Canvas(out).drawRect(0f, 0f, wd.toFloat(), ht.toFloat(), paint)
        return out
    }

    // Ảnh dòng -> đầu vào model đọc chữ: cao 48 giữ tỉ lệ, RGB, chuẩn hoá [-1, 1], đệm 0 tới cỡ độ rộng (như prep của Python)
    fun recInput(line: Bitmap): Pair<FloatArray, Int> {
        val need = ceil(48.0 * line.width / line.height).toInt()
        val bw = rec.bucketFor(need)
        val nw = max(1, min(bw, need))
        val small = Bitmap.createScaledBitmap(line, nw, 48, true)
        val px = IntArray(nw * 48)
        small.getPixels(px, 0, nw, 0, 0, nw, 48)
        if (small !== line) small.recycle()
        val x = FloatArray(48 * bw * 3)
        for (y in 0 until 48) for (xx in 0 until nw) {
            val c = px[y * nw + xx]
            val o = 3 * (y * bw + xx)
            x[o] = (((c shr 16) and 0xff) / 255f - 0.5f) / 0.5f
            x[o + 1] = (((c shr 8) and 0xff) / 255f - 0.5f) / 0.5f
            x[o + 2] = ((c and 0xff) / 255f - 0.5f) / 0.5f
        }
        return x to bw
    }

    fun read(page: Bitmap): Result {
        val t0 = System.nanoTime()
        val (p, h, w) = probMap(page)
        val t1 = System.nanoTime()
        val bs = boxes(p, h, w, page.width, page.height)
        val t2 = System.nanoTime()
        val lines = bs.map { b ->
            val crop = cropLine(page, b)
            val (x, bw) = recInput(crop)
            crop.recycle()
            Line(b, Normalizer.normalize(rec.decode(rec.runIdx(x, bw)), Normalizer.Form.NFC).trim())
        }.filter { it.text.isNotEmpty() }
        val t3 = System.nanoTime()
        return Result(lines, (t1 - t0) / 1_000_000, (t2 - t1) / 1_000_000, (t3 - t2) / 1_000_000)
    }

    fun close() = synchronized(dets) {
        dets.values.forEach { d -> d.inBufs.forEach { it.close() }; d.outBufs.forEach { it.close() }; d.cm.close() }
        dets.clear()
        rec.close()                                      // đóng cả môi trường dùng chung (sau khi đã đóng các model)
    }
}
