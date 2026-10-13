package com.truongduke.docscanner

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.util.Size
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.min

// Khung xanh đang hiển thị trên màn hình camera: tứ giác theo pixel của ảnh phân tích (w x h)
private class LiveQuad(val quad: FloatArray?, val w: Int, val h: Int, val ms: Long)

/**
 * Màn hình camera: hình camera trực tiếp, khung xanh bám theo tờ giấy, nút chụp.
 * - Preview: hình hiển thị (CameraX tự vẽ).
 * - ImageAnalysis: ảnh nhỏ 640x480 đưa vào model liên tục. Máy chậm (Bee 3 ~1.5 s / lần) thì
 *   CameraX tự bỏ các khung hình cũ (KEEP_ONLY_LATEST), chỉ phân tích khung mới nhất.
 * - ImageCapture: chụp ảnh nét (~2048x1536) khi bấm nút. Không tìm khung lại trên ảnh to (Bee 3 mất 2-3 s):
 *   dùng luôn khung đang hiện (cùng góc nhìn, cùng tỉ lệ 4:3), chỉ đổi toạ độ theo kích thước ảnh.
 * Cả 3 dùng tỉ lệ 4:3, preview hiển thị kiểu FIT_CENTER, nên đổi toạ độ ảnh -> màn hình chỉ cần 1 phép co giãn.
 */
@Composable
fun CameraScreen(
    model: MlsdModel,
    onCaptured: (Bitmap, FloatArray?, String) -> Unit,   // ảnh chụp, 4 góc, dòng báo thời gian
    onTools: () -> Unit,
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    var hasPermission by remember {
        mutableStateOf(ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED)
    }
    val askPermission = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { hasPermission = it }
    LaunchedEffect(Unit) { if (!hasPermission) askPermission.launch(Manifest.permission.CAMERA) }

    var live by remember { mutableStateOf(LiveQuad(null, 1, 1, 0)) }
    var busy by remember { mutableStateOf(false) }          // đang chụp + tìm khung trên ảnh nét
    var error by remember { mutableStateOf<String?>(null) }
    val analysisExecutor = remember { Executors.newSingleThreadExecutor() }
    val captureExecutor = remember { Executors.newSingleThreadExecutor() }   // chụp không phải xếp hàng sau phân tích
    val paused = remember { AtomicBoolean(false) }                           // tạm dừng phân tích trong lúc chụp
    val previewView = remember { PreviewView(context).apply { scaleType = PreviewView.ScaleType.FIT_CENTER } }
    val capture = remember {
        ImageCapture.Builder()
            .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
            .setResolutionSelector(selector43(Size(2048, 1536), ResolutionStrategy.FALLBACK_RULE_CLOSEST_LOWER_THEN_HIGHER))
            .build()
    }

    // Gắn camera vào vòng đời màn hình: mở khi màn hình hiện, đóng khi rời đi
    DisposableEffect(hasPermission) {
        if (!hasPermission) return@DisposableEffect onDispose { }
        val providerFuture = ProcessCameraProvider.getInstance(context)
        providerFuture.addListener({
            val provider = providerFuture.get()
            val preview = Preview.Builder().setResolutionSelector(selector43(null)).build()
                .also { it.setSurfaceProvider(previewView.surfaceProvider) }
            val analysis = ImageAnalysis.Builder()
                .setResolutionSelector(selector43(Size(640, 480)))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
            analysis.setAnalyzer(analysisExecutor) { proxy ->
                if (paused.get()) { proxy.close(); return@setAnalyzer }
                val bmp = proxy.toBitmap().rotated(proxy.imageInfo.rotationDegrees)
                proxy.close()
                val d = detectQuad(model, bmp)
                live = LiveQuad(d.quad, bmp.width, bmp.height, d.ms)
            }
            try {
                provider.unbindAll()
                provider.bindToLifecycle(lifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis, capture)
            } catch (e: Exception) {
                error = "Không mở được camera: ${e.message}"
            }
        }, ContextCompat.getMainExecutor(context))
        onDispose { providerFuture.get().unbindAll() }
    }
    DisposableEffect(Unit) { onDispose { analysisExecutor.shutdown(); captureExecutor.shutdown() } }

    Column(Modifier.fillMaxSize().background(Color.Black)) {
        Box(Modifier.weight(1f).fillMaxWidth()) {
            if (hasPermission) {
                AndroidView({ previewView }, Modifier.fillMaxSize())
                // Vẽ khung xanh: ảnh phân tích (w x h) hiển thị kiểu FIT_CENTER trong khung preview
                Canvas(Modifier.fillMaxSize()) {
                    val q = live.quad ?: return@Canvas
                    val s = min(size.width / live.w, size.height / live.h)
                    val ox = (size.width - live.w * s) / 2
                    val oy = (size.height - live.h * s) / 2
                    fun p(i: Int) = Offset(ox + q[i * 2] * s, oy + q[i * 2 + 1] * s)
                    val path = Path().apply { moveTo(p(0).x, p(0).y); for (i in 1..3) lineTo(p(i).x, p(i).y); close() }
                    drawPath(path, Color(0x5500C850))
                    drawPath(path, Color(0xFF00C850), style = Stroke(width = 4.dp.toPx()))
                }
            } else {
                Text("App cần quyền camera để chụp tài liệu", Modifier.align(Alignment.Center), color = Color.White)
            }
            Text(
                error ?: if (busy) "Đang chụp và tìm khung..." else
                    "Tìm khung: ${live.ms} ms" + if (live.quad == null && live.ms > 0) " (chưa thấy tờ giấy)" else "",
                Modifier.align(Alignment.TopCenter).padding(8.dp).background(Color(0x88000000)).padding(6.dp),
                color = Color.White,
            )
        }
        Row(
            Modifier.fillMaxWidth().padding(16.dp),
            horizontalArrangement = Arrangement.SpaceEvenly,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Button(onClick = onTools) { Text("Công cụ") }
            Button(enabled = hasPermission && !busy, onClick = {
                busy = true
                paused.set(true)
                val tPress = System.nanoTime()
                val liveAtPress = live
                capture.takePicture(captureExecutor, object : ImageCapture.OnImageCapturedCallback() {
                    override fun onCaptureSuccess(image: ImageProxy) {
                        val tShot = System.nanoTime()
                        val bmp = image.toBitmap().rotated(image.imageInfo.rotationDegrees)
                        image.close()
                        val tDecode = System.nanoTime()
                        // Khung: dùng khung đang hiện, đổi toạ độ từ ảnh phân tích sang ảnh chụp.
                        // Camera chưa thấy tờ giấy thì tìm lại trên bản thu nhỏ của ảnh chụp.
                        val quad = liveAtPress.quad?.let { q ->
                            FloatArray(8) { if (it % 2 == 0) q[it] * bmp.width / liveAtPress.w else q[it] * bmp.height / liveAtPress.h }
                        } ?: run {
                            val f = 640f / maxOf(bmp.width, bmp.height)
                            val small = Bitmap.createScaledBitmap(bmp, (bmp.width * f).toInt(), (bmp.height * f).toInt(), true)
                            detectQuad(model, small).quad?.let { q -> FloatArray(8) { q[it] / f } }
                        }
                        val tQuad = System.nanoTime()
                        fun ms(a: Long, b: Long) = (b - a) / 1_000_000
                        val timing = "Chụp ${ms(tPress, tShot)} ms | đọc ảnh ${bmp.width}x${bmp.height} ${ms(tShot, tDecode)} ms | " +
                            "khung ${ms(tDecode, tQuad)} ms" + if (liveAtPress.quad != null) " (lấy từ camera)" else " (tìm lại)"
                        ContextCompat.getMainExecutor(context).execute {
                            busy = false
                            paused.set(false)
                            onCaptured(bmp, quad, timing)
                        }
                    }

                    override fun onError(e: ImageCaptureException) {
                        busy = false
                        paused.set(false)
                        error = "Chụp lỗi: ${e.message}"
                    }
                })
            }) { Text("Chụp") }
        }
    }
}

// Chọn độ phân giải tỉ lệ 4:3, gần với kích thước mong muốn (null = để CameraX tự chọn)
private fun selector43(size: Size?, rule: Int = ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER): ResolutionSelector {
    val b = ResolutionSelector.Builder().setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
    if (size != null) b.setResolutionStrategy(ResolutionStrategy(size, rule))
    return b.build()
}
