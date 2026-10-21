package com.truongduke.docscanner

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.media.ExifInterface
import android.net.Uri
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.BackHandler
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Box
import java.io.File
import androidx.compose.ui.Alignment
import androidx.compose.foundation.selection.selectable
import androidx.compose.material3.RadioButton
import kotlinx.coroutines.launch
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.unit.sp
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.truongduke.docscanner.ui.theme.DocScannerTheme
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import mlsd.predLines
import mlsd.predSquares

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            DocScannerTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    App(Modifier.padding(innerPadding))
                }
            }
        }
    }
}

// Các màn hình của app (mốc 4): camera -> chỉnh góc -> kết quả; "Công cụ" là màn hình thử nghiệm của mốc 3, 5
enum class Screen { CAMERA, EDIT, RESULT, TOOLS }

@Composable
fun App(modifier: Modifier = Modifier) {
    val context = LocalContext.current
    var screen by remember { mutableStateOf(Screen.CAMERA) }
    // Camera dùng model INT8 đã sửa (nhanh nhất mà vẫn chính xác, xem mốc 5 trong CLAUDE.md)
    val model = remember { MlsdModel(context, ModelChoice.INT8_FIXED.file) }
    var photo by remember { mutableStateOf<Bitmap?>(null) }      // ảnh vừa chụp
    var quad by remember { mutableStateOf<FloatArray?>(null) }   // 4 góc tờ giấy trên ảnh đó
    var scan by remember { mutableStateOf<Bitmap?>(null) }       // ảnh đã nắn thẳng
    var captureInfo by remember { mutableStateOf("") }           // thời gian các bước lúc chụp

    // Nút quay lại của điện thoại: lùi 1 màn hình
    BackHandler(enabled = screen != Screen.CAMERA) {
        screen = if (screen == Screen.RESULT) Screen.EDIT else Screen.CAMERA
    }
    Box(modifier) {
        when (screen) {
            Screen.CAMERA -> CameraScreen(
                model,
                onCaptured = { b, q, info -> photo = b; quad = q; captureInfo = info; screen = Screen.EDIT },
                onTools = { screen = Screen.TOOLS },
            )
            Screen.EDIT -> EditScreen(photo!!, quad, captureInfo, onRetake = { screen = Screen.CAMERA }, onDone = { q ->
                quad = q
                scan = warpQuad(photo!!, q)
                screen = Screen.RESULT
            })
            Screen.RESULT -> ResultScreen(scan!!, onBack = { screen = Screen.EDIT }, onNew = { screen = Screen.CAMERA })
            Screen.TOOLS -> ScannerScreen()
        }
    }
}

@Composable
fun ScannerScreen(modifier: Modifier = Modifier) {
    val context = LocalContext.current
    // Ảnh đang hiển thị. null = chưa chọn ảnh nào
    var bitmap by remember { mutableStateOf<Bitmap?>(null) }
    // Model app dùng, và các model đã nạp (mỗi model nạp 1 lần, lúc dùng lần đầu)
    val choice = ModelChoice.INT8_FIXED
    val loaded = remember { mutableMapOf<ModelChoice, MlsdModel>() }
    fun modelFor(c: ModelChoice): MlsdModel = synchronized(loaded) {
        loaded.getOrPut(c) { MlsdModel(context, c.file).also { Log.d("MLSD", "${c.label}\n${it.describe()}") } }
    }
    // Dòng chữ báo kết quả model
    var info by remember { mutableStateOf("") }
    // Ảnh đã vẽ kết quả lên (null = chưa có, hiện ảnh gốc)
    var annotated by remember { mutableStateOf<Bitmap?>(null) }
    // Kết quả nút "Kiểm tra golden" (null = đang ở chế độ xem ảnh)
    var goldenReport by remember { mutableStateOf<String?>(null) }
    val scope = rememberCoroutineScope()

    // Mỗi khi đổi ảnh: chạy model ở luồng nền (không làm đơ giao diện), xong thì cập nhật info
    LaunchedEffect(bitmap, choice) {
        val bmp = bitmap ?: return@LaunchedEffect
        info = "Đang chạy model..."
        annotated = null
        val (text, drawn) = withContext(Dispatchers.Default) {
            // 1. Model
            val out = modelFor(choice).run(bmp)
            // 2. Post-processing: cùng code với golden test (postprocess/src)
            val t0 = System.nanoTime()
            val lines = predLines(out.pts, out.scores, out.vmap, bmp.height, bmp.width)
            val t1 = System.nanoTime()
            val sq = predSquares(out.pts, out.scores, out.vmap, bmp.height, bmp.width)
            val t2 = System.nanoTime()

            val best = sq.squares.firstOrNull()?.toList()?.chunked(2)?.joinToString(" ") { "(${it[0]}, ${it[1]})" } ?: "không có"
            val text = "Model: ${choice.label}\n" +
                "Tiền xử lý: ${out.prepMs} ms | Model: ${out.timeMs} ms\n" +
                "predLines: ${lines.size} đoạn, ${(t1 - t0) / 1_000_000} ms\n" +
                "predSquares: ${sq.segments.size} đoạn gộp, ${sq.interPoints.size} góc, " +
                "${sq.squares.size} tứ giác, ${(t2 - t1) / 1_000_000} ms\n" +
                "Tứ giác tốt nhất: $best"
            // 3. Vẽ kết quả lên bản sao của ảnh
            Pair(text, drawResult(bmp, lines, sq.squares.firstOrNull()))
        }
        info = text
        annotated = drawn
        Log.d("MLSD", info)
    }

    // Mở trình chọn ảnh của hệ thống; chọn xong thì đọc ảnh vào bitmap
    val pickImage = rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        if (uri != null) {
            goldenReport = null
            bitmap = loadBitmap(context, uri)
        }
    }

    Column(modifier = modifier.fillMaxSize().padding(16.dp).verticalScroll(rememberScrollState())) {
        Row {
            Button(onClick = {
                pickImage.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
            }) {
                Text("Chọn ảnh")
            }
            Spacer(Modifier.width(8.dp))
            // So kết quả app với bản Python trên các ảnh mẫu (xem GoldenCheck.kt)
            Button(onClick = {
                goldenReport = "Đang kiểm tra..."
                scope.launch {
                    // Golden sinh từ model NAVER nên luôn kiểm tra bằng model NAVER
                    val report = withContext(Dispatchers.Default) { runGoldenCheck(context, modelFor(ModelChoice.NAVER_FP32)) }
                    goldenReport = report
                    Log.d("MLSD", report)
                }
            }) {
                Text("Kiểm tra golden")
            }
        }
        // Đo tốc độ phần model trên 1 ảnh (ảnh đang chọn, không có thì dùng frame_1 của golden)
        Button(onClick = {
            goldenReport = "Đang đo tốc độ 6 tổ hợp model x CPU/GPU (mất vài phút)..."
            scope.launch {
                goldenReport = withContext(Dispatchers.Default) { runBenchmark(context, bitmap) }
                Log.d("MLSD", goldenReport ?: "")
            }
        }) {
            Text("Đo tốc độ")
        }
        // Model đọc chữ INT8 trên máy có ra cùng kết quả như Mac không, đọc đúng bao nhiêu, mỗi dòng bao lâu (OcrGoldenCheck.kt)
        Button(onClick = {
            goldenReport = "Đang kiểm tra OCR (nạp 4 model, đọc các dòng mẫu)..."
            scope.launch {
                goldenReport = withContext(Dispatchers.Default) {
                    val rec = OcrRecognizer(context)
                    try { runOcrGoldenCheck(context, rec) } finally { rec.close() }
                }
                Log.d("OCR", goldenReport ?: "")
            }
        }) {
            Text("Kiểm tra OCR")
        }
        // Cả chuỗi đọc trang (tìm dòng + đọc) so với bản Python trên 3 trang mẫu (OcrGoldenCheck.kt)
        Button(onClick = {
            goldenReport = "Đang kiểm tra OCR trang (3 trang, mỗi trang chạy 2 lần)..."
            scope.launch {
                goldenReport = withContext(Dispatchers.Default) {
                    val page = OcrPage(context)
                    try { runOcrPageGoldenCheck(context, page) } finally { page.close() }
                }
                Log.d("OCR", goldenReport ?: "")
            }
        }) {
            Text("Kiểm tra OCR trang")
        }
        Spacer(Modifier.height(16.dp))

        val bmp = bitmap
        val report = goldenReport
        if (report != null) {
            Text(report, fontFamily = FontFamily.Monospace, fontSize = 11.sp)
        } else if (bmp == null) {
            Text("Chưa chọn ảnh")
        } else {
            Text("Kích thước: ${bmp.width} x ${bmp.height}")
            Text(info)
            Text("Cam: đoạn thẳng (predLines) | Xanh: tứ giác tốt nhất (predSquares)")
            Spacer(Modifier.height(8.dp))
            Image(
                bitmap = (annotated ?: bmp).asImageBitmap(),
                contentDescription = "Ảnh đã chọn",
                // Rộng hết màn hình, cao theo đúng tỉ lệ ảnh (ảnh nhỏ cũng được phóng to cho dễ nhìn)
                modifier = Modifier.fillMaxWidth().aspectRatio(bmp.width.toFloat() / bmp.height),
                contentScale = ContentScale.Fit,
            )
        }
    }
}

// Đo tốc độ riêng phần model (không tính tiền xử lý / post-processing) của từng model trên CPU và GPU.
// Cả hàm chạy liền trên 1 luồng (model GPU phải tạo và dùng trên cùng 1 luồng).
// Mỗi tổ hợp: thời gian tạo model (GPU phải "dịch" model lúc này), 2 lần làm nóng, 5 lần đo.
// Kiểm tra GPU ra đúng: so các điểm center (score > 0.1) với chính model đó chạy CPU.
fun runBenchmark(context: Context, bitmap: Bitmap?): String {
    val golden = File(goldenDir(context), "frame_1_rgb.bin")
    val (rgb, w, h, src) = when {
        bitmap != null -> listOf(bitmapToRgb(bitmap), bitmap.width, bitmap.height, "ảnh đang chọn")
        golden.exists() -> listOf(golden.readBytes(), 1920, 1440, "frame_1 (golden)")
        else -> listOf(ByteArray(512 * 512 * 3) { 128.toByte() }, 512, 512, "ảnh xám")
    }
    rgb as ByteArray; w as Int; h as Int
    val sb = StringBuilder("Tốc độ phần model trên $src ($w x $h), CPU 4 thread (Interpreter), GPU FP16 (CompiledModel):\n\n")
    for (c in listOf(ModelChoice.NAVER_FP32, ModelChoice.NAVER_FP16, ModelChoice.INT8_FIXED)) {
        var cpuPts: Set<Int>? = null
        for (gpu in listOf(false, true)) {
            val name = "%-12s %s".format(c.label, if (gpu) "GPU" else "CPU")
            try {
                val t0 = System.nanoTime()
                val m = MlsdModel(context, c.file, useGpu = gpu)
                val initMs = (System.nanoTime() - t0) / 1_000_000
                val t = m.benchmark(rgb, w, h, warmup = 2, runs = 5).sorted()
                // Các điểm center được giữ (theo chỉ số ô y * 256 + x), để so CPU với GPU
                val out = m.runRgb(rgb, w, h)
                val pts = (0 until 200).filter { out.scores[it] > 0.1f }.map { out.pts[it * 2] * 256 + out.pts[it * 2 + 1] }.toSet()
                m.close()
                val same = cpuPts?.let { cp -> " | điểm trùng CPU ${(pts intersect cp).size}/${cp.size}" } ?: ""
                if (!gpu) cpuPts = pts
                sb.append("$name: ${t[t.size / 2]} ms (nhanh nhất ${t.first()}) | tạo model ${initMs} ms$same\n")
            } catch (e: Throwable) {
                sb.append("$name: LỖI ${e.javaClass.simpleName}: ${e.message?.take(120)}\n")
            }
        }
    }
    return sb.toString()
}

// Đọc ảnh từ uri. Ảnh quá to (điện thoại 50MP) thì thu nhỏ để cạnh dài nhất <= 2048,
// tránh hết bộ nhớ. Model chỉ cần 512x512 nên không mất gì.
fun loadBitmap(context: Context, uri: Uri, maxSide: Int = 2048): Bitmap? {
    // Lần 1: chỉ đọc kích thước, chưa đọc điểm ảnh
    val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
    context.contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, bounds) }

    // Tìm hệ số thu nhỏ (luỹ thừa của 2)
    var sample = 1
    while (maxOf(bounds.outWidth, bounds.outHeight) / sample > maxSide) sample *= 2

    // Lần 2: đọc ảnh thật với hệ số đó
    val opts = BitmapFactory.Options().apply { inSampleSize = sample }
    val bmp = context.contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, opts) }
        ?: return null

    // Lần 3: xoay ảnh cho đúng chiều theo thẻ EXIF.
    // Camera (nhất là iPhone) hay lưu điểm ảnh nằm ngang kèm thẻ "cần xoay 90°"; BitmapFactory bỏ qua thẻ này.
    // cv2.imread bên Python thì có xoay, nên xoay ở đây để giống bản Python.
    val orientation = context.contentResolver.openInputStream(uri)?.use {
        ExifInterface(it).getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)
    } ?: ExifInterface.ORIENTATION_NORMAL
    val m = Matrix()
    when (orientation) {
        ExifInterface.ORIENTATION_ROTATE_90 -> m.postRotate(90f)
        ExifInterface.ORIENTATION_ROTATE_180 -> m.postRotate(180f)
        ExifInterface.ORIENTATION_ROTATE_270 -> m.postRotate(270f)
        ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> m.postScale(-1f, 1f)
        ExifInterface.ORIENTATION_FLIP_VERTICAL -> m.postScale(1f, -1f)
        ExifInterface.ORIENTATION_TRANSPOSE -> { m.postRotate(90f); m.postScale(-1f, 1f) }
        ExifInterface.ORIENTATION_TRANSVERSE -> { m.postRotate(270f); m.postScale(-1f, 1f) }
        else -> return bmp   // ảnh đã đúng chiều
    }
    return Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, m, true)
}
