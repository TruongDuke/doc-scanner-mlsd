package com.truongduke.docscanner

import android.content.Context
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.TensorBuffer
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.math.round

// Đọc chữ 1 dòng: model PP-OCRv5 mobile rec fine-tune tiếng Việt (FT2), bản INT8 đã chữa (cân bằng kênh + thước MSE,
// xem docs/ocr_experiments.md). Đầu vào ảnh dòng cao 48, rộng cố định -> 4 file model, mỗi file 1 "cỡ" độ rộng:
//   assets/ocr_int8_w320.tflite, _w640, _w1280, _w2000  (dòng cần rộng w -> cỡ nhỏ nhất >= w, đệm 0 bên phải)
// Đầu ra: mỗi lát dọc (rộng 8 px) chấm điểm 238 ký tự (0 = trống, 1..236 = từ điển, 237 = khoảng trắng)
// -> lấy ký tự điểm cao nhất từng lát -> giải mã CTC (gộp ký tự lặp, bỏ ký tự trống).
// Model nạp lúc dùng cỡ đó lần đầu (đỡ tốn RAM khi chỉ đọc dòng ngắn).
//
// Hai cách gọi LiteRT (cùng file model):
//   compiled = true  (mặc định): API mới CompiledModel của LiteRT 2.x, CPU 4 luồng
//   compiled = false: API cũ Interpreter. Đo trên Bee 3 (19/10): model cỡ 320 mất 230 ms, trong khi công cụ
//                     benchmark_model chỉ 86 ms -> thử API mới
class OcrRecognizer(
    private val context: Context,
    val buckets: List<Int> = listOf(320, 640, 1280, 2000),
    val compiled: Boolean = true,
) {
    // Một model đã nạp: đầu vào int8 (scale, zero point), đầu ra (số lát, số ký tự)
    private class Loaded(
        val interp: Interpreter?, val cm: CompiledModel?, val inBufs: List<TensorBuffer>?, val outBufs: List<TensorBuffer>?,
        val inInt8: Boolean, val scale: Float, val zero: Int, val steps: Int, val classes: Int,
    )

    private val models = HashMap<Int, Loaded>()

    // Môi trường LiteRT dùng chung cho mọi CompiledModel, giữ suốt đời đối tượng này. KHÔNG để CompiledModel.create tự tạo:
    // môi trường tạm không ai giữ tham chiếu -> bộ dọn rác (GC) huỷ giữa chừng -> nhóm luồng tính toán bị xoá, model đang
    // chạy chờ mãi (treo "Đang đọc chữ...", Bee 3 21/10)
    val env: Environment = Environment.create(emptyMap())

    // Thời gian riêng phần model (ms) của lần runIdx gần nhất, không tính đổi số đầu vào / chọn ký tự
    var lastModelMs = 0.0

    // Từ điển: vị trí 0 là ký tự trống của CTC, cuối là khoảng trắng (giống PaddleOCR use_space_char)
    val chars: List<String> = listOf("") +
        context.assets.open("ocr_vi_dict.txt").bufferedReader().readLines().filter { it.isNotEmpty() } + listOf(" ")

    fun bucketFor(needWidth: Int) = buckets.firstOrNull { it >= needWidth } ?: buckets.last()

    private fun model(w: Int): Loaded = synchronized(models) {
        models.getOrPut(w) {
            val file = "ocr_int8_w$w.tflite"
            val fd = context.assets.openFd(file)
            val buf = FileInputStream(fd.fileDescriptor).channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
            // Đọc thông số đầu vào / đầu ra bằng Interpreter (API mới không cho xem scale / zero point)
            val interp = Interpreter(buf, Interpreter.Options().setNumThreads(4).setUseXNNPACK(true))
            val inT = interp.getInputTensor(0)
            val outShape = interp.getOutputTensor(0).shape()
            val inInt8 = inT.dataType() == DataType.INT8
            val scale = inT.quantizationParams().scale
            val zero = inT.quantizationParams().zeroPoint
            if (compiled) {
                interp.close()
                val opts = CompiledModel.Options(Accelerator.CPU)
                opts.cpuOptions = CompiledModel.CpuOptions(4, null, null)
                val cm = CompiledModel.create(context.assets, file, opts, env)
                Loaded(null, cm, cm.createInputBuffers(), cm.createOutputBuffers(), inInt8, scale, zero, outShape[1], outShape[2])
            } else {
                Loaded(interp, null, null, null, inInt8, scale, zero, outShape[1], outShape[2])
            }
        }
    }

    // input: ảnh dòng đã chuẩn hoá [-1, 1], NHWC (48, w, 3), đã đệm tới đúng cỡ w.
    // Trả về ký tự điểm cao nhất ở từng lát (w / 8 lát).
    fun runIdx(input: FloatArray, w: Int): IntArray {
        val m = model(w)
        check(m.inInt8) { "model OCR phải có đầu vào int8" }
        // Đổi số thực -> int8 theo scale / zero point; làm tròn kiểu "chẵn gần nhất" như numpy trên Mac
        val q = ByteArray(input.size) { round(input[it] / m.scale + m.zero).coerceIn(-128f, 127f).toInt().toByte() }
        val out: ByteArray
        val t0 = System.nanoTime()
        if (m.cm != null) {
            m.inBufs!![0].writeInt8(q)
            m.cm.run(m.inBufs, m.outBufs!!)
            out = m.outBufs[0].readInt8()
            lastModelMs = (System.nanoTime() - t0) / 1e6
        } else {
            val inBuf = ByteBuffer.allocateDirect(q.size).order(ByteOrder.nativeOrder()).put(q)
            inBuf.rewind()
            val outBuf = ByteBuffer.allocateDirect(m.steps * m.classes).order(ByteOrder.nativeOrder())
            m.interp!!.run(inBuf, outBuf)
            lastModelMs = (System.nanoTime() - t0) / 1e6
            outBuf.rewind()
            out = ByteArray(m.steps * m.classes).also { outBuf.get(it) }
        }
        // Lấy chỉ số lớn nhất từng lát. Đầu ra int8: so thẳng số nguyên (đổi về số thực không làm đổi thứ tự)
        return IntArray(m.steps) { s ->
            var best = 0
            for (c in 1 until m.classes) if (out[s * m.classes + c] > out[s * m.classes + best]) best = c
            best
        }
    }

    // Giải mã CTC: bỏ ký tự trống (0), gộp ký tự giống nhau đứng liền nhau
    fun decode(idx: IntArray): String {
        val sb = StringBuilder()
        var prev = -1
        for (k in idx) {
            if (k != prev && k != 0) sb.append(chars[k])
            prev = k
        }
        return sb.toString()
    }

    fun close() = synchronized(models) {
        models.values.forEach { m -> m.interp?.close(); m.inBufs?.forEach { it.close() }; m.outBufs?.forEach { it.close() }; m.cm?.close() }
        models.clear()
        env.close()
    }
}
