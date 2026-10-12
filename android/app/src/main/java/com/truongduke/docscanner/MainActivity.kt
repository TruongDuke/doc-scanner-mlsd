package com.truongduke.docscanner

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
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
                    ScannerScreen(Modifier.padding(innerPadding))
                }
            }
        }
    }
}

@Composable
fun ScannerScreen(modifier: Modifier = Modifier) {
    val context = LocalContext.current
    // Ảnh đang hiển thị. null = chưa chọn ảnh nào
    var bitmap by remember { mutableStateOf<Bitmap?>(null) }
    // Model chỉ nạp 1 lần khi mở màn hình
    val model = remember { MlsdModel(context).also { Log.d("MLSD", it.describe()) } }
    // Dòng chữ báo kết quả model
    var info by remember { mutableStateOf("") }

    // Mỗi khi đổi ảnh: chạy model ở luồng nền (không làm đơ giao diện), xong thì cập nhật info
    LaunchedEffect(bitmap) {
        val bmp = bitmap ?: return@LaunchedEffect
        info = "Đang chạy model..."
        info = withContext(Dispatchers.Default) {
            // 1. Model
            val out = model.run(bmp)
            // 2. Post-processing: cùng code với golden test (postprocess/src)
            val t0 = System.nanoTime()
            val lines = predLines(out.pts, out.scores, out.vmap, bmp.height, bmp.width)
            val t1 = System.nanoTime()
            val sq = predSquares(out.pts, out.scores, out.vmap, bmp.height, bmp.width)
            val t2 = System.nanoTime()

            val best = sq.squares.firstOrNull()?.toList()?.chunked(2)?.joinToString(" ") { "(${it[0]}, ${it[1]})" } ?: "không có"
            "Model: ${out.timeMs} ms\n" +
                "predLines: ${lines.size} đoạn, ${(t1 - t0) / 1_000_000} ms\n" +
                "predSquares: ${sq.segments.size} đoạn gộp, ${sq.interPoints.size} góc, " +
                "${sq.squares.size} tứ giác, ${(t2 - t1) / 1_000_000} ms\n" +
                "Tứ giác tốt nhất: $best"
        }
        Log.d("MLSD", info)
    }

    // Mở trình chọn ảnh của hệ thống; chọn xong thì đọc ảnh vào bitmap
    val pickImage = rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        if (uri != null) bitmap = loadBitmap(context, uri)
    }

    Column(modifier = modifier.fillMaxSize().padding(16.dp)) {
        Button(onClick = {
            pickImage.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
        }) {
            Text("Chọn ảnh")
        }

        Spacer(Modifier.height(16.dp))

        val bmp = bitmap
        if (bmp == null) {
            Text("Chưa chọn ảnh")
        } else {
            Text("Kích thước: ${bmp.width} x ${bmp.height}")
            Text(info)
            Spacer(Modifier.height(8.dp))
            Image(
                bitmap = bmp.asImageBitmap(),
                contentDescription = "Ảnh đã chọn",
                modifier = Modifier.fillMaxWidth(),
                contentScale = ContentScale.Fit,
            )
        }
    }
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
    return context.contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, opts) }
}
