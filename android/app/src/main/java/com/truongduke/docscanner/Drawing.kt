package com.truongduke.docscanner

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import mlsd.Line

// Vẽ kết quả lên một bản sao của ảnh (ảnh gốc không bị sửa).
// Vẽ trực tiếp theo toạ độ pixel ảnh gốc, nên khớp đúng với số mà predLines / predSquares trả về.
//   - lines: các đoạn của predLines, nét mảnh màu cam
//   - square: tứ giác tốt nhất (8 số x0, y0, ..., x3, y3), tô xanh mờ, viền xanh, 4 góc đánh số 0-3
fun drawResult(src: Bitmap, lines: List<Line>, square: IntArray?): Bitmap {
    val out = src.copy(Bitmap.Config.ARGB_8888, true)
    val canvas = Canvas(out)
    // Độ dày nét tính theo kích thước ảnh, để ảnh to hay nhỏ nhìn đều rõ
    val s = maxOf(out.width, out.height) / 500f

    val linePaint = Paint().apply {
        color = Color.rgb(255, 140, 0)
        style = Paint.Style.STROKE
        strokeWidth = 1.5f * s
        isAntiAlias = true
    }
    for (l in lines) canvas.drawLine(l.x1, l.y1, l.x2, l.y2, linePaint)

    if (square != null) {
        val path = Path().apply {
            moveTo(square[0].toFloat(), square[1].toFloat())
            for (k in 1 until 4) lineTo(square[k * 2].toFloat(), square[k * 2 + 1].toFloat())
            close()
        }
        canvas.drawPath(path, Paint().apply { color = Color.argb(60, 0, 200, 80); style = Paint.Style.FILL })
        canvas.drawPath(path, Paint().apply {
            color = Color.rgb(0, 200, 80)
            style = Paint.Style.STROKE
            strokeWidth = 3f * s
            isAntiAlias = true
        })

        // 4 góc: chấm tròn + số thứ tự (0 trên-trái, 1 trên-phải, 2 dưới-phải, 3 dưới-trái)
        val dotPaint = Paint().apply { color = Color.rgb(0, 120, 255); isAntiAlias = true }
        val textPaint = Paint().apply {
            color = Color.WHITE
            textSize = 7f * s
            textAlign = Paint.Align.CENTER
            isFakeBoldText = true
            isAntiAlias = true
        }
        for (k in 0 until 4) {
            val x = square[k * 2].toFloat()
            val y = square[k * 2 + 1].toFloat()
            canvas.drawCircle(x, y, 6f * s, dotPaint)
            canvas.drawText("$k", x, y + textPaint.textSize / 3, textPaint)
        }
    }
    return out
}
