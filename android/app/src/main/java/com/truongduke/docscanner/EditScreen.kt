package com.truongduke.docscanner

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Build
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.unit.sp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import androidx.core.content.ContextCompat
import kotlin.math.min

/**
 * Màn hình chỉnh góc: ảnh vừa chụp + tứ giác app tìm được, người dùng kéo 4 góc nếu sai.
 * Góc lưu theo pixel của ảnh gốc; lúc vẽ / kéo thì đổi qua lại với toạ độ trên màn hình (nhân / chia scale).
 */
@Composable
fun EditScreen(bitmap: Bitmap, initialQuad: FloatArray?, info: String, onRetake: () -> Unit, onDone: (FloatArray) -> Unit) {
    val q0 = initialQuad ?: defaultQuad(bitmap.width, bitmap.height)
    val corners = remember(bitmap) { mutableStateListOf(*Array(4) { Offset(q0[it * 2], q0[it * 2 + 1]) }) }
    var dragging by remember { mutableStateOf(-1) }

    Column(Modifier.fillMaxSize().background(Color.Black)) {
        Text(
            (if (initialQuad == null) "Không tìm thấy tờ giấy, hãy kéo 4 góc cho đúng" else "Kéo các góc nếu khung chưa đúng") +
                if (info.isNotEmpty()) "\n$info" else "",
            Modifier.fillMaxWidth().padding(12.dp),
            color = Color.White,
        )
        BoxWithConstraints(Modifier.weight(1f).fillMaxWidth(), contentAlignment = Alignment.Center) {
            // Ảnh co vừa khung (giữ tỉ lệ); scale = số pixel màn hình cho 1 pixel ảnh
            val scale = min(constraints.maxWidth.toFloat() / bitmap.width, constraints.maxHeight.toFloat() / bitmap.height)
            Box(Modifier.aspectRatio(bitmap.width.toFloat() / bitmap.height)) {
                Image(bitmap.asImageBitmap(), "Ảnh vừa chụp", Modifier.fillMaxSize(), contentScale = ContentScale.FillBounds)
                Canvas(
                    Modifier.fillMaxSize().pointerInput(bitmap) {
                        val grab = 40.dp.toPx()   // chạm trong bán kính này quanh góc thì kéo được góc đó
                        detectDragGestures(
                            onDragStart = { pos ->
                                val k = (0 until 4).minBy { (corners[it] * scale - pos).getDistance() }
                                dragging = if ((corners[k] * scale - pos).getDistance() < grab) k else -1
                            },
                            onDragEnd = { dragging = -1 },
                            onDrag = { change, amount ->
                                if (dragging >= 0) {
                                    change.consume()
                                    val p = corners[dragging] + amount / scale
                                    corners[dragging] = Offset(p.x.coerceIn(0f, bitmap.width.toFloat()), p.y.coerceIn(0f, bitmap.height.toFloat()))
                                }
                            },
                        )
                    }
                ) {
                    val pts = corners.map { it * scale }
                    val path = Path().apply { moveTo(pts[0].x, pts[0].y); for (i in 1..3) lineTo(pts[i].x, pts[i].y); close() }
                    drawPath(path, Color(0x4400C850))
                    drawPath(path, Color(0xFF00C850), style = Stroke(width = 3.dp.toPx()))
                    pts.forEachIndexed { i, p ->
                        drawCircle(if (i == dragging) Color.Yellow else Color(0xFF0078FF), radius = 14.dp.toPx(), center = p)
                        drawCircle(Color.White, radius = 14.dp.toPx(), center = p, style = Stroke(width = 2.dp.toPx()))
                    }
                }
            }
        }
        Row(Modifier.fillMaxWidth().padding(16.dp), horizontalArrangement = Arrangement.SpaceEvenly) {
            Button(onClick = onRetake) { Text("Chụp lại") }
            Button(onClick = { onDone(FloatArray(8) { if (it % 2 == 0) corners[it / 2].x else corners[it / 2].y }) }) { Text("Nắn thẳng") }
        }
    }
}

/** Màn hình kết quả: ảnh tài liệu đã nắn thẳng, nút lưu vào thư viện. */
@Composable
fun ResultScreen(scan: Bitmap, onBack: () -> Unit, onNew: () -> Unit) {
    val context = LocalContext.current
    var message by remember { mutableStateOf<String?>(null) }
    // Đọc chữ (OcrPage.kt): nạp model lúc bấm lần đầu, đóng khi rời màn hình
    val ocr = remember { lazy { OcrPage(context) } }
    DisposableEffect(Unit) { onDispose { if (ocr.isInitialized()) ocr.value.close() } }
    var ocrResult by remember { mutableStateOf<OcrPage.Result?>(null) }
    var reading by remember { mutableStateOf(false) }
    val scope = rememberCoroutineScope()
    val clipboard = LocalClipboardManager.current

    val res = ocrResult
    if (res != null) {
        // Màn hình chữ đọc được: bôi chọn được, có nút sao chép
        Column(Modifier.fillMaxSize().background(Color.White).padding(16.dp)) {
            Text("${res.lines.size} dòng | tìm dòng ${res.detMs} ms, xử lý ${res.postMs} ms, đọc chữ ${res.recMs} ms",
                fontSize = 12.sp, color = Color.Gray)
            SelectionContainer(Modifier.weight(1f).fillMaxWidth().verticalScroll(rememberScrollState()).padding(vertical = 8.dp)) {
                Text(res.text.ifEmpty { "(không tìm thấy chữ)" }, fontSize = 15.sp)
            }
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceEvenly) {
                Button(onClick = { clipboard.setText(AnnotatedString(res.text)); message = "Đã sao chép ${res.text.length} ký tự" }) { Text("Sao chép") }
                Button(onClick = { ocrResult = null }) { Text("Quay lại") }
            }
        }
        return
    }

    fun save() {
        message = try {
            "Đã lưu: ${saveToGallery(context, scan)}"
        } catch (e: Exception) {
            "Lưu lỗi: ${e.message}"
        }
    }
    // Android 9 trở xuống cần quyền ghi bộ nhớ (Android 10+ thì không)
    val askWrite = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { ok ->
        if (ok) save() else message = "Cần quyền lưu ảnh"
    }

    Column(Modifier.fillMaxSize().background(Color.Black)) {
        Box(Modifier.weight(1f).fillMaxWidth().padding(12.dp), contentAlignment = Alignment.Center) {
            Image(scan.asImageBitmap(), "Ảnh đã nắn thẳng", Modifier.fillMaxSize(), contentScale = ContentScale.Fit)
        }
        Text(message ?: "Kích thước: ${scan.width} x ${scan.height}", Modifier.fillMaxWidth().padding(horizontal = 16.dp), color = Color.White)
        Row(Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 8.dp), horizontalArrangement = Arrangement.SpaceEvenly) {
            Button(onClick = onBack) { Text("Chỉnh lại") }
            Button(onClick = {
                val needPermission = Build.VERSION.SDK_INT <= Build.VERSION_CODES.P &&
                    ContextCompat.checkSelfPermission(context, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED
                if (needPermission) askWrite.launch(Manifest.permission.WRITE_EXTERNAL_STORAGE) else save()
            }) { Text("Lưu") }
        }
        // 4 nút không vừa 1 hàng trên màn hình 720 px (Bee 3) -> 2 hàng
        Row(Modifier.fillMaxWidth().padding(start = 16.dp, end = 16.dp, bottom = 16.dp), horizontalArrangement = Arrangement.SpaceEvenly) {
            Button(enabled = !reading, onClick = {
                reading = true
                message = "Đang đọc chữ..."
                scope.launch {
                    try {
                        val r = withContext(Dispatchers.Default) { ocr.value.read(scan) }
                        ocrResult = r
                        message = null
                    } catch (e: Exception) {
                        message = "Đọc chữ lỗi: ${e.message}"
                    } finally {
                        reading = false
                    }
                }
            }) { Text("Đọc chữ") }
            Button(onClick = onNew) { Text("Chụp tiếp") }
        }
    }
}
