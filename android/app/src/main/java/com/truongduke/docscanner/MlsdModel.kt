package com.truongduke.docscanner

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

// Kết quả thô của model, cùng định dạng phẳng như file golden
class MlsdOutput(
    val pts: IntArray,      // 400 số: [y0, x0, y1, x1, ...] trên lưới 256
    val scores: FloatArray, // 200 số
    val vmap: FloatArray,   // 256 * 256 * 4 số: vmap[(y * 256 + x) * 4 + c]
    val timeMs: Long,       // thời gian chạy model (ms)
)

class MlsdModel(context: Context, modelFile: String = "M-LSD_512_tiny_fp32.tflite") {
    private val interpreter: Interpreter

    init {
        // Đọc file model trong assets/ (map thẳng vào bộ nhớ, không chép)
        val fd = context.assets.openFd(modelFile)
        val model = FileInputStream(fd.fileDescriptor).channel
            .map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
        interpreter = Interpreter(model, Interpreter.Options().setNumThreads(4))
    }

    // Mô tả các tensor vào/ra, để kiểm tra model đọc đúng
    fun describe(): String {
        val sb = StringBuilder()
        for (i in 0 until interpreter.inputTensorCount) {
            val t = interpreter.getInputTensor(i)
            sb.append("IN  $i ${t.name()} ${t.shape().contentToString()} ${t.dataType()}\n")
        }
        for (i in 0 until interpreter.outputTensorCount) {
            val t = interpreter.getOutputTensor(i)
            sb.append("OUT $i ${t.name()} ${t.shape().contentToString()} ${t.dataType()}\n")
        }
        return sb.toString()
    }

    fun run(bitmap: Bitmap): MlsdOutput {
        // 1. Thu ảnh về 512x512.
        //    Lưu ý: Python dùng cv2.INTER_AREA, ở đây tạm dùng bilinear nên kết quả sẽ hơi khác golden (sửa ở bước 6)
        val resized = Bitmap.createScaledBitmap(bitmap, 512, 512, true)
        val pixels = IntArray(512 * 512)
        resized.getPixels(pixels, 0, 512, 0, 0, 512, 512)

        // 2. Đầu vào (1, 512, 512, 4) float32: R, G, B (giữ nguyên 0..255) và kênh thứ 4 toàn số 1
        val input = ByteBuffer.allocateDirect(512 * 512 * 4 * 4).order(ByteOrder.nativeOrder())
        for (p in pixels) {
            input.putFloat(((p shr 16) and 0xFF).toFloat())  // R
            input.putFloat(((p shr 8) and 0xFF).toFloat())   // G
            input.putFloat((p and 0xFF).toFloat())           // B
            input.putFloat(1f)
        }
        input.rewind()

        // 3. Chỗ chứa 3 đầu ra
        val ptsBuf = ByteBuffer.allocateDirect(200 * 2 * 4).order(ByteOrder.nativeOrder())
        val scoresBuf = ByteBuffer.allocateDirect(200 * 4).order(ByteOrder.nativeOrder())
        val vmapBuf = ByteBuffer.allocateDirect(256 * 256 * 4 * 4).order(ByteOrder.nativeOrder())

        // 4. Chạy model, đo thời gian
        val t0 = System.nanoTime()
        interpreter.runForMultipleInputsOutputs(arrayOf(input), mapOf(0 to ptsBuf, 1 to scoresBuf, 2 to vmapBuf))
        val timeMs = (System.nanoTime() - t0) / 1_000_000

        // 5. Chép ra mảng phẳng
        ptsBuf.rewind(); scoresBuf.rewind(); vmapBuf.rewind()
        val pts = IntArray(400).also { ptsBuf.asIntBuffer().get(it) }
        val scores = FloatArray(200).also { scoresBuf.asFloatBuffer().get(it) }
        val vmap = FloatArray(256 * 256 * 4).also { vmapBuf.asFloatBuffer().get(it) }
        return MlsdOutput(pts, scores, vmap, timeMs)
    }
}
