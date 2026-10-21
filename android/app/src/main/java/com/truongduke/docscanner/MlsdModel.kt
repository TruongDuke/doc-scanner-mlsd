package com.truongduke.docscanner

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.TensorBuffer
import org.tensorflow.lite.Tensor
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.math.roundToInt
import mlsd.resizeArea

// Kết quả thô của model, cùng định dạng phẳng như file golden
class MlsdOutput(
    val pts: IntArray,      // 400 số: [y0, x0, y1, x1, ...] trên lưới 256
    val scores: FloatArray, // 200 số
    val vmap: FloatArray,   // 256 * 256 * 4 số: vmap[(y * 256 + x) * 4 + c]
    val timeMs: Long,       // thời gian chạy model (ms)
    val prepMs: Long,       // thời gian tiền xử lý: thu ảnh về 512x512, đổi sang float (ms)
)

// Lấy điểm ảnh của Bitmap thành mảng RGB: 3 byte mỗi điểm, xếp theo hàng (cùng định dạng ảnh numpy (h, w, 3))
fun bitmapToRgb(bitmap: Bitmap): ByteArray {
    val w = bitmap.width
    val h = bitmap.height
    val pixels = IntArray(w * h)
    bitmap.getPixels(pixels, 0, w, 0, 0, w, h)
    val rgb = ByteArray(w * h * 3)
    for (i in pixels.indices) {
        val p = pixels[i]
        rgb[i * 3] = (p shr 16).toByte()      // R
        rgb[i * 3 + 1] = (p shr 8).toByte()   // G
        rgb[i * 3 + 2] = p.toByte()           // B
    }
    return rgb
}

// Các model có trong assets/ (cùng đầu vào / đầu ra như bản NAVER):
// - INT8_FIXED: model app dùng (lhwcv, tách head + cắt ngoại lai, INT8, xem mốc 5 trong CLAUDE.md)
// - NAVER_FP32: để chạy "Kiểm tra golden" (golden sinh từ model này) và so tốc độ
// - NAVER_FP16: chỉ để so tốc độ CPU / GPU (mốc 6)
enum class ModelChoice(val label: String, val file: String) {
    INT8_FIXED("INT8 đã sửa", "lhwcv_v3_splitclip_int8.tflite"),
    NAVER_FP32("NAVER FP32", "M-LSD_512_tiny_fp32.tflite"),
    NAVER_FP16("NAVER FP16", "M-LSD_512_tiny_fp16.tflite"),
}

// useGpu: chạy bằng API CompiledModel của LiteRT 2.x trên GPU qua OpenGL (cho phép tính FP16; phép nào GPU không làm được thì
// về CPU). LiteRT 2.3.0 đã bỏ GpuDelegate kiểu cũ của mốc 6. Interpreter vẫn được tạo để đọc thông số vào / ra (scale,
// zero point, kích thước) và để chạy khi useGpu = false.
// Lưu ý: model chạy GPU nên được tạo và dùng trên cùng 1 luồng, dùng xong gọi close().
class MlsdModel(context: Context, modelFile: String = ModelChoice.INT8_FIXED.file, useGpu: Boolean = false) {
    private val interpreter: Interpreter
    // Đường GPU: môi trường LiteRT giữ suốt đời model (để thư viện tự tạo thì bộ dọn rác huỷ giữa chừng -> treo)
    private val env: Environment?
    private val cm: CompiledModel?
    private val cmIn: List<TensorBuffer>?
    private val cmOut: List<TensorBuffer>?

    init {
        // Đọc file model trong assets/ (map thẳng vào bộ nhớ, không chép)
        val fd = context.assets.openFd(modelFile)
        val model = FileInputStream(fd.fileDescriptor).channel
            .map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
        val options = Interpreter.Options().setNumThreads(4)
        interpreter = Interpreter(model, options)
        if (useGpu) {
            env = Environment.create(emptyMap())
            // GPU: cho phép tính bằng FP16 (nhanh hơn trên GPU điện thoại, sai số rất nhỏ), như mốc 6
            val opts = CompiledModel.Options(Accelerator.GPU, Accelerator.CPU)
            // Backend OpenGL: backend OpenCL (mặc định) truyền tuỳ chọn biên dịch "-cl-uniform-work-group-size" của OpenCL 2.0,
            // driver OpenCL 1.2 của PowerVR GE8100 (Bee 3) không hiểu -> "Failed to compile model" (21/10)
            opts.gpuOptions = CompiledModel.GpuOptions(precision = CompiledModel.GpuOptions.Precision.FP16,
                backend = CompiledModel.GpuOptions.Backend.OPENGL)
            cm = CompiledModel.create(context.assets, modelFile, opts, env)
            cmIn = cm.createInputBuffers()
            cmOut = cm.createOutputBuffers()
            check(cmOut.size == interpreter.outputTensorCount) { "số đầu ra CompiledModel ${cmOut.size} khác Interpreter" }
        } else {
            env = null; cm = null; cmIn = null; cmOut = null
        }
    }

    fun close() {
        cmIn?.forEach { it.close() }
        cmOut?.forEach { it.close() }
        cm?.close()
        env?.close()
        interpreter.close()
    }

    // Tìm đầu ra theo kích thước (thứ tự đầu ra mỗi model có thể khác nhau)
    private fun outIndex(vararg shape: Int) = (0 until interpreter.outputTensorCount)
        .first { interpreter.getOutputTensor(it).shape().contentEquals(shape) }
    private val ptsIdx = outIndex(1, 200, 2)
    private val scoresIdx = outIndex(1, 200)
    private val vmapIdx = outIndex(1, 256, 256, 4)

    // Mô tả các tensor vào/ra, để kiểm tra model đọc đúng
    fun describe(): String {
        val sb = StringBuilder()
        fun line(kind: String, i: Int, t: Tensor) = sb.append(
            "$kind $i ${t.name()} ${t.shape().contentToString()} ${t.dataType()} " +
                "scale ${t.quantizationParams().scale} zp ${t.quantizationParams().zeroPoint}\n")
        for (i in 0 until interpreter.inputTensorCount) line("IN ", i, interpreter.getInputTensor(i))
        for (i in 0 until interpreter.outputTensorCount) line("OUT", i, interpreter.getOutputTensor(i))
        return sb.toString()
    }

    fun run(bitmap: Bitmap): MlsdOutput {
        val t0 = System.nanoTime()
        val rgb = bitmapToRgb(bitmap)
        return runRgb(rgb, bitmap.width, bitmap.height, t0)
    }

    // rgb: ảnh RGB w x h (3 byte mỗi điểm). t0: lúc bắt đầu tiền xử lý, để tính prepMs
    fun runRgb(rgb: ByteArray, w: Int, h: Int, t0: Long = System.nanoTime()): MlsdOutput {
        // 1. Thu ảnh về 512x512 giống hệt cv2.resize(..., interpolation=cv2.INTER_AREA) của bản Python
        val small = resizeArea(rgb, w, h, 512, 512)

        // 2. Đầu vào (1, 512, 512, 4): R, G, B (giữ nguyên 0..255) và kênh thứ 4 toàn số 1.
        //    Model FP32: ghi số thực. Model INT8: đổi từng số sang int8 theo scale / zero_point của tensor vào
        val inT = interpreter.getInputTensor(0)
        val input = ByteBuffer.allocateDirect(inT.numBytes()).order(ByteOrder.nativeOrder())
        if (inT.dataType() == DataType.INT8) {
            val s = inT.quantizationParams().scale
            val z = inT.quantizationParams().zeroPoint
            // Tính sẵn 256 giá trị int8 cho 256 mức điểm ảnh, và cho số 1
            val lut = ByteArray(256) { v -> ((v / s).roundToInt() + z).coerceIn(-128, 127).toByte() }
            val one = ((1f / s).roundToInt() + z).coerceIn(-128, 127).toByte()
            for (i in 0 until 512 * 512) {
                input.put(lut[small[i * 3].toInt() and 0xFF])
                input.put(lut[small[i * 3 + 1].toInt() and 0xFF])
                input.put(lut[small[i * 3 + 2].toInt() and 0xFF])
                input.put(one)
            }
        } else {
            for (i in 0 until 512 * 512) {
                input.putFloat((small[i * 3].toInt() and 0xFF).toFloat())      // R
                input.putFloat((small[i * 3 + 1].toInt() and 0xFF).toFloat())  // G
                input.putFloat((small[i * 3 + 2].toInt() and 0xFF).toFloat())  // B
                input.putFloat(1f)
            }
        }
        input.rewind()
        val prepMs = (System.nanoTime() - t0) / 1_000_000

        // 3. Chỗ chứa các đầu ra (kích thước theo kiểu dữ liệu của từng tensor)
        val outBufs = (0 until interpreter.outputTensorCount).associateWith {
            ByteBuffer.allocateDirect(interpreter.getOutputTensor(it).numBytes()).order(ByteOrder.nativeOrder())
        }

        // 4. Chạy model, đo thời gian (GPU: CompiledModel; chép đầu ra về cùng chỗ chứa như đường CPU cho code sau dùng chung)
        val t1 = System.nanoTime()
        if (cm != null) {
            if (inT.dataType() == DataType.INT8) cmIn!![0].writeInt8(ByteArray(input.remaining()).also { input.get(it) })
            else cmIn!![0].writeFloat(FloatArray(input.remaining() / 4).also { input.asFloatBuffer().get(it) })
            cm.run(cmIn, cmOut!!)
            for ((i, b) in outBufs) {
                val t = interpreter.getOutputTensor(i)
                b.clear()
                when (t.dataType()) {
                    DataType.INT8 -> b.put(cmOut[i].readInt8())
                    DataType.INT32 -> b.asIntBuffer().put(cmOut[i].readInt())
                    else -> b.asFloatBuffer().put(cmOut[i].readFloat())
                }
            }
        } else {
            interpreter.runForMultipleInputsOutputs(arrayOf(input), outBufs)
        }
        val timeMs = (System.nanoTime() - t1) / 1_000_000

        // 5. Chép ra mảng phẳng; đầu ra INT8 thì đổi ngược về số thực: (q - zero_point) * scale
        fun floats(idx: Int, n: Int): FloatArray {
            val t = interpreter.getOutputTensor(idx)
            val b = outBufs.getValue(idx).also { it.rewind() }
            if (t.dataType() != DataType.INT8) return FloatArray(n).also { b.asFloatBuffer().get(it) }
            val s = t.quantizationParams().scale
            val z = t.quantizationParams().zeroPoint
            return FloatArray(n) { (b.get(it) - z) * s }
        }
        val pts = IntArray(400).also { outBufs.getValue(ptsIdx).rewind(); outBufs.getValue(ptsIdx).asIntBuffer().get(it) }
        val scores = floats(scoresIdx, 200)
        val vmap = floats(vmapIdx, 256 * 256 * 4)
        return MlsdOutput(pts, scores, vmap, timeMs, prepMs)
    }

    // Đo tốc độ riêng phần model: chạy runs lần trên cùng 1 ảnh (sau warmup lần chạy làm nóng), trả về các thời gian (ms)
    fun benchmark(rgb: ByteArray, w: Int, h: Int, warmup: Int = 2, runs: Int = 10): List<Long> {
        repeat(warmup) { runRgb(rgb, w, h) }
        return List(runs) { runRgb(rgb, w, h).timeMs }
    }
}
