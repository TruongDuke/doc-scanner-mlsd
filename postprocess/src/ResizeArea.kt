package mlsd

import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.min

// cvRound: làm tròn nửa về số chẵn, rồi kẹp 0..255 (saturate_cast<uchar>)
private fun roundU8(v: Float): Int = Math.rint(v.toDouble()).toInt().coerceIn(0, 255)

/**
 * Thu / phóng ảnh RGB giống cv2.resize(img, (dw, dh), interpolation=cv2.INTER_AREA) của OpenCV 4.10
 * (imgproc/src/resize.cpp), để đầu vào model trong app giống hệt bản Python.
 *
 * src: ảnh RGB, 3 byte mỗi điểm ảnh, xếp theo hàng: src[(y * sw + x) * 3 + c]
 * Trả về ảnh dw x dh cùng định dạng.
 */
fun resizeArea(src: ByteArray, sw: Int, sh: Int, dw: Int, dh: Int): ByteArray {
    // OpenCV tính scale = 1 / (dsize / ssize), giữ đúng cách tính để sai số làm tròn giống nhau
    val invScaleX = dw.toDouble() / sw
    val invScaleY = dh.toDouble() / sh
    val scaleX = 1.0 / invScaleX
    val scaleY = 1.0 / invScaleY

    if (scaleX >= 1 && scaleY >= 1) {
        val isx = Math.rint(scaleX).toInt()
        val isy = Math.rint(scaleY).toInt()
        val eps = Math.ulp(1.0)   // DBL_EPSILON
        return if (abs(scaleX - isx) < eps && abs(scaleY - isy) < eps) areaFast(src, sw, dw, dh, isx, isy)
        else areaGeneric(src, sw, sh, dw, dh, scaleX, scaleY)
    }
    return areaUpscale(src, sw, sh, dw, dh, scaleX, scaleY, invScaleX, invScaleY)
}

// ===== Nhánh 1: thu nhỏ đúng số nguyên lần: trung bình cộng từng khối isx x isy =====
private fun areaFast(src: ByteArray, sw: Int, dw: Int, dh: Int, isx: Int, isy: Int): ByteArray {
    val dst = ByteArray(dw * dh * 3)
    val area = isx * isy
    val scale = 1f / area
    for (dy in 0 until dh) for (dx in 0 until dw) for (c in 0 until 3) {
        var sum = 0
        for (ky in 0 until isy) for (kx in 0 until isx) {
            sum += src[((dy * isy + ky) * sw + dx * isx + kx) * 3 + c].toInt() and 0xFF
        }
        // Khối 2x2 OpenCV dùng lệnh SIMD: (sum + 2) >> 2 (làm tròn nửa lên); còn lại: sum * scale rồi cvRound
        dst[(dy * dw + dx) * 3 + c] = (if (area == 4) (sum + 2) shr 2 else roundU8(sum * scale)).toByte()
    }
    return dst
}

// ===== Nhánh 2: thu nhỏ tỉ lệ lẻ: mỗi điểm ảnh đích là trung bình có trọng số các điểm nguồn nó phủ lên =====
private class Tab(val di: Int, val si: Int, val alpha: Float)

// computeResizeAreaTab: với mỗi điểm đích, các điểm nguồn nằm trong nó và tỉ lệ phủ (alpha)
private fun areaTab(ssize: Int, dsize: Int, scale: Double): List<Tab> {
    val tab = mutableListOf<Tab>()
    for (dx in 0 until dsize) {
        val fsx1 = dx * scale
        val fsx2 = fsx1 + scale
        val cellWidth = min(scale, ssize - fsx1)
        var sx2 = floor(fsx2).toInt()
        sx2 = min(sx2, ssize - 1)
        val sx1 = min(ceil(fsx1).toInt(), sx2)
        if (sx1 - fsx1 > 1e-3) tab.add(Tab(dx, sx1 - 1, ((sx1 - fsx1) / cellWidth).toFloat()))
        for (sx in sx1 until sx2) tab.add(Tab(dx, sx, (1.0 / cellWidth).toFloat()))
        if (fsx2 - sx2 > 1e-3) tab.add(Tab(dx, sx2, (min(min(fsx2 - sx2, 1.0), cellWidth) / cellWidth).toFloat()))
    }
    return tab
}

private fun areaGeneric(src: ByteArray, sw: Int, sh: Int, dw: Int, dh: Int, scaleX: Double, scaleY: Double): ByteArray {
    val xtab = areaTab(sw, dw, scaleX)
    val ytab = areaTab(sh, dh, scaleY)
    val dst = ByteArray(dw * dh * 3)
    val width = dw * 3
    val buf = FloatArray(width)   // một hàng nguồn đã gộp theo chiều ngang
    val sum = FloatArray(width)   // cộng dồn các hàng nguồn cho hàng đích hiện tại
    var prevDy = ytab[0].di

    for (yt in ytab) {
        val beta = yt.alpha
        val rowOfs = yt.si * sw * 3
        buf.fill(0f)
        // Gộp theo chiều ngang (thứ tự cộng giống OpenCV)
        for (xt in xtab) {
            val s = rowOfs + xt.si * 3
            val d = xt.di * 3
            for (c in 0 until 3) buf[d + c] += (src[s + c].toInt() and 0xFF) * xt.alpha
        }
        // Gộp theo chiều dọc: sang hàng đích mới thì ghi hàng cũ ra
        if (yt.di != prevDy) {
            for (k in 0 until width) {
                dst[prevDy * width + k] = roundU8(sum[k]).toByte()
                sum[k] = beta * buf[k]
            }
            prevDy = yt.di
        } else {
            for (k in 0 until width) sum[k] += beta * buf[k]
        }
    }
    for (k in 0 until width) dst[prevDy * width + k] = roundU8(sum[k]).toByte()
    return dst
}

// ===== Nhánh 3: phóng to (ít nhất 1 chiều): nội suy tuyến tính số nguyên, hệ số theo kiểu INTER_AREA =====
private const val COEF_SCALE = 2048   // INTER_RESIZE_COEF_SCALE = 1 << 11

private fun areaUpscale(
    src: ByteArray, sw: Int, sh: Int, dw: Int, dh: Int,
    scaleX: Double, scaleY: Double, invScaleX: Double, invScaleY: Double,
): ByteArray {
    // Hệ số theo chiều ngang
    val xofs = IntArray(dw)
    val ax0 = IntArray(dw)
    val ax1 = IntArray(dw)
    for (dx in 0 until dw) {
        var sx = floor(dx * scaleX).toInt()
        var fx = ((dx + 1) - (sx + 1) * invScaleX).toFloat()
        fx = if (fx <= 0) 0f else fx - floor(fx)
        if (sx < 0) { fx = 0f; sx = 0 }
        if (sx >= sw - 1) { fx = 0f; sx = sw - 1 }
        xofs[dx] = sx
        ax0[dx] = Math.rint(((1f - fx) * COEF_SCALE).toDouble()).toInt()
        ax1[dx] = Math.rint((fx * COEF_SCALE).toDouble()).toInt()
    }

    // Nội suy ngang một hàng nguồn -> mảng int (đã nhân 2048)
    fun hResize(sy: Int): IntArray {
        val row = IntArray(dw * 3)
        val ofs = sy * sw * 3
        for (dx in 0 until dw) {
            val sx = xofs[dx]
            val sx1 = min(sx + 1, sw - 1)   // khi sx là cột cuối thì ax1 = 0, không ảnh hưởng
            for (c in 0 until 3) {
                val p0 = src[ofs + sx * 3 + c].toInt() and 0xFF
                val p1 = src[ofs + sx1 * 3 + c].toInt() and 0xFF
                row[dx * 3 + c] = p0 * ax0[dx] + p1 * ax1[dx]
            }
        }
        return row
    }

    val dst = ByteArray(dw * dh * 3)
    for (dy in 0 until dh) {
        val sy = floor(dy * scaleY).toInt()
        var fy = ((dy + 1) - (sy + 1) * invScaleY).toFloat()
        fy = if (fy <= 0) 0f else fy - floor(fy)
        val b0 = Math.rint(((1f - fy) * COEF_SCALE).toDouble()).toInt()
        val b1 = Math.rint((fy * COEF_SCALE).toDouble()).toInt()
        val r0 = hResize(sy.coerceIn(0, sh - 1))
        val r1 = hResize((sy + 1).coerceIn(0, sh - 1))
        // Nội suy dọc, công thức số nguyên của OpenCV cho uchar (cả bản SIMD lẫn bản thường)
        for (k in 0 until dw * 3) {
            dst[dy * dw * 3 + k] = ((((b0 * (r0[k] shr 4)) shr 16) + ((b1 * (r1[k] shr 4)) shr 16) + 2) shr 2).toByte()
        }
    }
    return dst
}
