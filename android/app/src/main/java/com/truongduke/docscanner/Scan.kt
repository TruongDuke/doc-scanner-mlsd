package com.truongduke.docscanner

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import android.media.MediaScannerConnection
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import java.io.File
import kotlin.math.atan2
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.roundToInt
import mlsd.predSquares

// Các bước của mốc 4 (camera, chụp, chỉnh góc, nắn thẳng, lưu) không phụ thuộc giao diện.
// Một tứ giác = FloatArray(8): x0, y0, x1, y1, x2, y2, x3, y3 theo pixel của ảnh,
// thứ tự góc 0 trên-trái, 1 trên-phải, 2 dưới-phải, 3 dưới-trái.

class Detection(val quad: FloatArray?, val ms: Long)

// Tứ giác nhỏ hơn mức này (so với diện tích ảnh) không thể là tờ giấy cần scan, bỏ qua
const val MIN_QUAD_AREA = 0.05f

// Diện tích tứ giác (công thức shoelace)
fun quadArea(q: FloatArray): Float {
    var s = 0f
    for (i in 0 until 4) {
        val j = (i + 1) % 4
        s += q[i * 2] * q[j * 2 + 1] - q[j * 2] * q[i * 2 + 1]
    }
    return kotlin.math.abs(s) / 2
}

// Tìm tờ giấy trên ảnh: model + predSquares, lấy tứ giác điểm cao nhất mà không quá nhỏ.
// Interpreter không dùng chung được giữa nhiều luồng (camera phân tích liên tục + lúc chụp),
// nên khoá model trong lúc chạy.
fun detectQuad(model: MlsdModel, bmp: Bitmap): Detection {
    val t0 = System.nanoTime()
    val out = synchronized(model) { model.run(bmp) }
    val sq = predSquares(out.pts, out.scores, out.vmap, bmp.height, bmp.width)
    val minArea = MIN_QUAD_AREA * bmp.width * bmp.height
    val quad = sq.squares.asSequence()
        .map { s -> FloatArray(8) { s[it].toFloat() } }
        .firstOrNull { quadArea(it) >= minArea }
        ?.let { orderCorners(it) }
    return Detection(quad, (System.nanoTime() - t0) / 1_000_000)
}

// Khung mặc định khi không tìm được tứ giác: chừa 10% mỗi cạnh để người dùng tự kéo góc
fun defaultQuad(w: Int, h: Int): FloatArray {
    val dx = w * 0.1f
    val dy = h * 0.1f
    return floatArrayOf(dx, dy, w - dx, dy, w - dx, h - dy, dx, h - dy)
}

// Xếp lại 4 góc theo chiều kim đồng hồ, bắt đầu từ góc trên-trái (x + y nhỏ nhất).
// Cần vì người dùng có thể kéo góc lung tung, và nắn thẳng cần đúng thứ tự.
fun orderCorners(q: FloatArray): FloatArray {
    val cx = (q[0] + q[2] + q[4] + q[6]) / 4
    val cy = (q[1] + q[3] + q[5] + q[7]) / 4
    // Toạ độ ảnh có trục y hướng xuống, nên góc atan2 tăng dần = đi theo chiều kim đồng hồ
    val idx = (0 until 4).sortedBy { atan2(q[it * 2 + 1] - cy, q[it * 2] - cx) }
    val start = (0 until 4).minBy { q[idx[it] * 2] + q[idx[it] * 2 + 1] }
    return FloatArray(8) { k -> q[idx[(start + k / 2) % 4] * 2 + k % 2] }
}

// Nắn thẳng: kéo tứ giác về hình chữ nhật (biến đổi phối cảnh).
// Cạnh của ảnh kết quả = cạnh dài hơn trong 2 cạnh đối diện của tứ giác.
fun warpQuad(src: Bitmap, quad: FloatArray): Bitmap {
    val q = orderCorners(quad)
    fun d(i: Int, j: Int) = hypot(q[i * 2] - q[j * 2], q[i * 2 + 1] - q[j * 2 + 1])
    val w = max(d(0, 1), d(3, 2)).roundToInt().coerceAtLeast(1)
    val h = max(d(0, 3), d(1, 2)).roundToInt().coerceAtLeast(1)
    val dst = floatArrayOf(0f, 0f, w.toFloat(), 0f, w.toFloat(), h.toFloat(), 0f, h.toFloat())
    // setPolyToPoly với 4 cặp điểm tạo ra đúng phép biến đổi phối cảnh (homography)
    val m = Matrix().apply { setPolyToPoly(q, 0, dst, 0, 4) }
    val out = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
    Canvas(out).drawBitmap(src, m, Paint(Paint.FILTER_BITMAP_FLAG or Paint.ANTI_ALIAS_FLAG))
    return out
}

// Lưu ảnh vào thư viện, thư mục Pictures/DocScanner. Trả về đường dẫn để báo cho người dùng.
// Android 10+ dùng MediaStore (không cần quyền); Android 9 trở xuống (Bee 3) ghi file trực tiếp
// và cần quyền WRITE_EXTERNAL_STORAGE (xin ở màn hình kết quả).
fun saveToGallery(context: Context, bmp: Bitmap): String {
    val name = "scan_${System.currentTimeMillis()}.jpg"
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
        val values = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, name)
            put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
            put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/DocScanner")
            put(MediaStore.Images.Media.IS_PENDING, 1)
        }
        val resolver = context.contentResolver
        val uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
            ?: error("Không tạo được file ảnh")
        resolver.openOutputStream(uri)?.use { bmp.compress(Bitmap.CompressFormat.JPEG, 95, it) }
        values.clear()
        values.put(MediaStore.Images.Media.IS_PENDING, 0)
        resolver.update(uri, values, null, null)
        return "Pictures/DocScanner/$name"
    }
    @Suppress("DEPRECATION")
    val dir = File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES), "DocScanner")
    dir.mkdirs()
    val file = File(dir, name)
    file.outputStream().use { bmp.compress(Bitmap.CompressFormat.JPEG, 95, it) }
    // Báo cho thư viện ảnh biết có file mới
    MediaScannerConnection.scanFile(context, arrayOf(file.path), arrayOf("image/jpeg"), null)
    return file.path
}

// Xoay ảnh theo góc camera báo về (ảnh từ cảm biến thường nằm ngang)
fun Bitmap.rotated(degrees: Int): Bitmap =
    if (degrees == 0) this
    else Bitmap.createBitmap(this, 0, 0, width, height, Matrix().apply { postRotate(degrees.toFloat()) }, true)
