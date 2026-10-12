import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.atan2
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

// Toàn bộ phần tính toán dùng Double vì bên Python (numpy) mọi thứ sau bước lọc đều là float64.
// Dùng Float sẽ lệch làm tròn, và vài bước làm tròn xuống số nguyên (hough, toạ độ góc) sẽ ra khác.

// Một góc tìm được: toạ độ nguyên (x, y) trên lưới 256, và 2 đoạn i < j tạo ra nó
data class Corner(val x: Int, val y: Int, val i: Int, val j: Int)

data class SquaresResult(
    val segments: List<DoubleArray>,   // đoạn đã gộp, (x1, y1, x2, y2) trên ảnh gốc
    val squares: List<IntArray>,       // mỗi tứ giác 8 số: x0, y0, x1, y1, x2, y2, x3, y3 trên ảnh gốc, điểm cao nhất trước
    val scores: DoubleArray,           // điểm của từng tứ giác, cùng thứ tự với squares
    val interPoints: List<IntArray>,   // các góc hợp lệ (x, y) trên ảnh gốc
)

/**
 * Viết lại pred_squares của repo M-LSD chính thức (mlsd-official/utils.py, dòng 50-161).
 * Đầu vào giống predLines. Thứ tự góc của tứ giác:
 *   0 -- 1
 *   |    |
 *   3 -- 2
 */
fun predSquares(
    pts: IntArray,
    ptsScore: FloatArray,
    vmap: FloatArray,
    imgH: Int,
    imgW: Int,
    scoreThr: Double = 0.06,
    outsideRatio: Double = 0.28,
    insideRatio: Double = 0.45,
    wOverlap: Double = 0.0,
    wDegree: Double = 1.95,
    wLength: Double = 0.0,
    wArea: Double = 1.86,
    wCenter: Double = 0.14,
): SquaresResult {
    // ===== 0. Lọc đoạn thẳng, giữ toạ độ trên lưới 256 (khác predLines: không đổi về ảnh gốc) =====
    val segs = mutableListOf<DoubleArray>()
    for (k in 0 until 200) {
        val y = pts[k * 2]
        val x = pts[k * 2 + 1]
        val base = (y * 256 + x) * 4
        val dxS = vmap[base]; val dyS = vmap[base + 1]; val dxE = vmap[base + 2]; val dyE = vmap[base + 3]
        // dist_map bên Python tính bằng float32, nên ở đây cũng dùng Float
        val dist = sqrt((dxS - dxE) * (dxS - dxE) + (dyS - dyE) * (dyS - dyE))
        // so sánh score (float32) với 0.06 (float64) như numpy: đổi score sang Double trước
        if (ptsScore[k].toDouble() > scoreThr && dist > 20.0f) {
            segs.add(doubleArrayOf(x + dxS.toDouble(), y + dyS.toDouble(), x + dxE.toDouble(), y + dyE.toDouble()))
        }
    }
    // Python crash ở đây khi không có đoạn nào, Kotlin trả kết quả rỗng
    if (segs.isEmpty()) return SquaresResult(emptyList(), emptyList(), DoubleArray(0), emptyList())

    // ===== 1. Gộp các đoạn gần trùng nhau bằng không gian Hough =====
    // Mỗi đoạn đổi thành (d, theta): d = khoảng cách từ gốc (0, 0) tới đường thẳng, theta = góc nghiêng
    val nSeg = segs.size
    val theta = DoubleArray(nSeg)
    val dCell = IntArray(nSeg)       // d làm tròn xuống, bước 1
    val tCell = IntArray(nSeg)       // theta làm tròn xuống, bước 2 độ
    for (k in 0 until nSeg) {
        val (x1, y1, x2, y2) = segs[k]
        val dx = x1 - x2
        val dy = y1 - y2
        val a = dy
        val b = -dx
        val c = a * x1 + b * y1
        val d = abs(c) / sqrt(a * a + b * b + 1e-10)
        var t = atan2(dx, dy) * 180 / PI
        if (t < 0.0) t += 180
        theta[k] = t
        dCell[k] = floor(d).toInt()
        tCell[k] = floor(t / 2).toInt()
    }

    // Lưới tích luỹ: mỗi ô đếm số đoạn rơi vào, và nhớ chỉ số đoạn đầu tiên (giống np.unique return_index)
    val accH = 512 + 1
    val accW = 360 / 2 + 1
    val acc = FloatArray(accH * accW)
    val idxMap = IntArray(accH * accW) { -1 }
    val cellMembers = HashMap<Int, MutableList<Int>>()   // ô -> mọi đoạn trong ô (thay cho vòng tìm hough bằng nhau bên Python)
    for (k in 0 until nSeg) {
        // Python sẽ crash (IndexError) nếu d > 512; ở đây bỏ qua đoạn đó
        if (dCell[k] >= accH || tCell[k] >= accW) continue
        val cell = dCell[k] * accW + tCell[k]
        acc[cell] += 1f
        if (idxMap[cell] == -1) idxMap[cell] = k
        cellMembers.getOrPut(cell) { mutableListOf() }.add(k)
    }

    // Giữ lại các ô là cực đại trong cửa sổ 5x5 (thay cho MaxPool2D padding='same' của TF)
    val peaks = mutableListOf<Int>()
    for (yy in 0 until accH) for (xx in 0 until accW) {
        val v = acc[yy * accW + xx]
        if (v == 0f) continue
        var m = v
        for (oy in -2..2) for (ox in -2..2) {
            val ny = yy + oy; val nx = xx + ox
            if (ny in 0 until accH && nx in 0 until accW) m = max(m, acc[ny * accW + nx])
        }
        if (v == m) peaks.add(yy * accW + xx)
    }
    // tf.math.top_k(k=200): giá trị giảm dần, bằng nhau thì ô có chỉ số nhỏ đứng trước
    peaks.sortWith(compareByDescending<Int> { acc[it] }.thenBy { it })
    val topPeaks = peaks.take(200)

    // numpy cho chỉ số âm quay vòng (-1 là phần tử cuối). Giữ đúng hành vi đó cho khớp kết quả;
    // chỉ số vượt quá cuối mảng thì Python crash, ở đây coi như ô trống.
    fun wrapCell(yy: Int, xx: Int): Int? {
        val ny = if (yy < 0) yy + accH else yy
        val nx = if (xx < 0) xx + accW else xx
        if (ny !in 0 until accH || nx !in 0 until accW) return null
        return ny * accW + nx
    }

    val merged = mutableListOf<DoubleArray>()
    for (cell in topPeaks) {
        val py = cell / accW
        val px = cell % accW
        // Gom mọi đầu mút của các đoạn trong 5x5 ô quanh đỉnh
        var xMin = Double.POSITIVE_INFINITY; var yMin = Double.POSITIVE_INFINITY
        var xMax = Double.NEGATIVE_INFINITY; var yMax = Double.NEGATIVE_INFINITY
        for (oy in -2..2) for (ox in -2..2) {
            val nc = wrapCell(py + oy, px + ox) ?: continue
            for (k in cellMembers[nc] ?: continue) {
                val s = segs[k]
                xMin = min(xMin, min(s[0], s[2])); xMax = max(xMax, max(s[0], s[2]))
                yMin = min(yMin, min(s[1], s[3])); yMax = max(yMax, max(s[1], s[3]))
            }
        }
        // Đoạn gộp là đường chéo của khung bao; hướng chéo theo góc của đoạn đại diện
        if (theta[idxMap[cell]] >= 90) merged.add(doubleArrayOf(xMin, yMax, xMax, yMin))
        else merged.add(doubleArrayOf(xMin, yMin, xMax, yMax))
    }

    // ===== 2. Giao điểm của mọi cặp đoạn gộp =====
    val n = merged.size
    val sx = DoubleArray(n) { merged[it][0] }; val sy = DoubleArray(n) { merged[it][1] }
    val ex = DoubleArray(n) { merged[it][2] }; val ey = DoubleArray(n) { merged[it][3] }
    val cx = DoubleArray(n) { (sx[it] + ex[it]) / 2.0 }
    val cy = DoubleArray(n) { (sy[it] + ey[it]) / 2.0 }
    val distSeg = DoubleArray(n) {
        val dx = sx[it] - ex[it]; val dy = sy[it] - ey[it]
        sqrt(dx * dx + dy * dy)
    }
    // đường thẳng ax + by = c
    val la = DoubleArray(n) { sy[it] - ey[it] }
    val lb = DoubleArray(n) { -(sx[it] - ex[it]) }
    val lc = DoubleArray(n) { la[it] * sx[it] + lb[it] * sy[it] }

    val interX = Array(n) { IntArray(n) }
    val interY = Array(n) { IntArray(n) }
    for (i in 0 until n) for (j in 0 until n) {
        // hai đoạn song song làm det = 0; + 1e-10 tránh chia 0 nhưng kết quả rất lớn.
        // toInt() bão hoà và NaN thành 0, giống numpy astype('int32') trên máy M1
        val det = la[i] * lb[j] - la[j] * lb[i]
        interY[i][j] = ((la[i] * lc[j] - la[j] * lc[i]) / (det + 1e-10)).toInt()
        interX[i][j] = ((lc[i] * lb[j] - lc[j] * lb[i]) / (det + 1e-10)).toInt()
    }

    fun dist(px: Int, py: Int, qx: Double, qy: Double): Double {
        val dx = px - qx; val dy = py - qy
        return sqrt(dx * dx + dy * dy)
    }
    // Khoảng cách từ giao điểm (i, j) tới 2 đầu đoạn i (seg1) và 2 đầu đoạn j (seg2), xếp [gần, xa]
    fun distToSeg1(i: Int, j: Int): Pair<Double, Double> {
        val a = dist(interX[i][j], interY[i][j], sx[i], sy[i]); val b = dist(interX[i][j], interY[i][j], ex[i], ey[i])
        return Pair(min(a, b), max(a, b))
    }
    fun distToSeg2(i: Int, j: Int): Pair<Double, Double> {
        val a = dist(interX[i][j], interY[i][j], sx[j], sy[j]); val b = dist(interX[i][j], interY[i][j], ex[j], ey[j])
        return Pair(min(a, b), max(a, b))
    }
    // Góc (độ, 0..360) nhìn từ giao điểm tới trung điểm một đoạn
    fun degTo(i: Int, j: Int, seg: Int): Double {
        var d = atan2(cy[seg] - interY[i][j], cx[seg] - interX[i][j]) * 180 / PI
        if (d < 0.0) d += 360
        return d
    }

    // ===== 3. Phân loại góc: chỉ giữ giao điểm gần đầu mút cả 2 đoạn và góc kẹp 60..120 độ =====
    val cornerDict = List(4) { mutableListOf<Corner>() }
    val interPoints = mutableListOf<IntArray>()
    for (i in 0 until n) for (j in i + 1 until n) {
        val x = interX[i][j]; val y = interY[i][j]
        val dA = degTo(i, j, i); val dB = degTo(i, j, j)
        val deg1 = min(dA, dB); val deg2 = max(dA, dB)
        var degDiff = abs(dA - dB)
        if (degDiff > 180) degDiff = 360 - degDiff

        val checkDegree = degDiff > 60 && degDiff < 120
        val (s1min, s1max) = distToSeg1(i, j)
        val (s2min, s2max) = distToSeg2(i, j)
        // giao điểm nằm ngoài đoạn thì phải sát đầu mút (outsideRatio), nằm trong đoạn thì nới hơn (insideRatio)
        val checkDistance =
            ((s1max >= distSeg[i] && s1min <= distSeg[i] * outsideRatio) ||
                (s1max <= distSeg[i] && s1min <= distSeg[i] * insideRatio)) &&
                ((s2max >= distSeg[j] && s2min <= distSeg[j] * outsideRatio) ||
                    (s2max <= distSeg[j] && s2min <= distSeg[j] * insideRatio))

        if (checkDegree && checkDistance) {
            val cornerInfo = when {
                (deg1 >= 0 && deg1 <= 45 && deg2 >= 45 && deg2 <= 120) ||
                    (deg2 >= 315 && deg1 >= 45 && deg1 <= 120) -> 0
                deg1 >= 45 && deg1 <= 125 && deg2 >= 125 && deg2 <= 225 -> 1
                deg1 >= 125 && deg1 <= 225 && deg2 >= 225 && deg2 <= 315 -> 2
                (deg1 >= 0 && deg1 <= 45 && deg2 >= 225 && deg2 <= 315) ||
                    (deg2 >= 315 && deg1 >= 225 && deg1 <= 315) -> 3
                else -> -1   // Python gọi là góc 4 (đỏ), không dùng
            }
            if (cornerInfo == -1) continue
            cornerDict[cornerInfo].add(Corner(x, y, i, j))
            interPoints.add(intArrayOf(x, y))
        }
    }

    // ===== 4. Ghép tứ giác: góc 0 -> 1 -> 2 -> 3 -> 0, mỗi cặp góc liền nhau phải chung 1 đoạn =====
    fun lines(c: Corner) = intArrayOf(c.i, c.j)
    // Python: vòng for ... break, trả về đoạn đầu tiên của a có mặt trong b (hoặc null)
    fun firstShared(a: Corner, b: Corner): Int? = lines(a).firstOrNull { it in lines(b) }

    val squareList = mutableListOf<IntArray>()     // 4 góc, toạ độ lưới 256
    val connectList = mutableListOf<IntArray>()    // đoạn nối 01, 12, 23, 30
    val segmentsList = mutableListOf<Array<Corner>>()
    for (c0 in cornerDict[0]) for (c1 in cornerDict[1]) {
        val l01 = firstShared(c0, c1) ?: continue
        for (c2 in cornerDict[2]) {
            val l12 = firstShared(c1, c2) ?: continue
            for (c3 in cornerDict[3]) {
                val l23 = firstShared(c2, c3) ?: continue
                // Python không break ở vòng cuối, nên mỗi đoạn chung đều thêm 1 tứ giác
                for (l30 in lines(c3)) {
                    if (l30 !in lines(c0)) continue
                    squareList.add(intArrayOf(c0.x, c0.y, c1.x, c1.y, c2.x, c2.y, c3.x, c3.y))
                    connectList.add(intArrayOf(l01, l12, l23, l30))
                    segmentsList.add(arrayOf(c0, c1, c2, c3))
                }
            }
        }
    }

    // ===== 5. Chấm điểm từng tứ giác =====
    val m = squareList.size
    var scores = DoubleArray(m)
    var squares: List<IntArray> = squareList
    if (m > 0) {   // Python: np.max của mảng rỗng ném lỗi, bị try/except nuốt
        val mapSize = 512 / 2.0
        val overlapScores = DoubleArray(m)
        val degreeScores = DoubleArray(m)
        val lengthScores = DoubleArray(m)
        val areaScores = DoubleArray(m)
        val centerScores = DoubleArray(m)

        // segInfo = 2 đoạn (i, j) tạo ra góc; xét giao điểm nằm ngoài hay trong đoạn connectIdx
        fun checkOutsideInside(segInfo: Corner, connectIdx: Int): Triple<Double, Int, Int> {
            val (minD, maxD) = if (connectIdx == segInfo.i) distToSeg1(segInfo.i, segInfo.j) else distToSeg2(segInfo.i, segInfo.j)
            return if (maxD > distSeg[connectIdx]) Triple(minD, 0, 1)   // outside: cover_param 0, peri_param 1
            else Triple(minD, -1, -1)                                   // inside
        }

        for (s in 0 until m) {
            val sq = squareList[s]
            val connects = connectList[s]
            val corners = segmentsList[s]

            // Góc tại mỗi đỉnh, từ 2 vector tới đỉnh trước và đỉnh sau (np.roll)
            val degree = DoubleArray(4) { k ->
                val prev = (k + 3) % 4; val next = (k + 1) % 4
                val v1x = (sq[prev * 2] - sq[k * 2]).toDouble(); val v1y = (sq[prev * 2 + 1] - sq[k * 2 + 1]).toDouble()
                val v2x = (sq[next * 2] - sq[k * 2]).toDouble(); val v2y = (sq[next * 2 + 1] - sq[k * 2 + 1]).toDouble()
                val n1 = sqrt(v1x * v1x + v1y * v1y) + 1e-10
                val n2 = sqrt(v2x * v2x + v2y * v2y) + 1e-10
                acos((v1x / n1) * (v2x / n2) + (v1y / n1) * (v2y / n2)) * 180 / PI
            }

            // Overlap: các cạnh tứ giác được đoạn thẳng thật phủ bao nhiêu
            var cover = 0.0
            var perimeter = 0.0
            val squareLength = DoubleArray(4)
            for (startIdx in 0 until 4) {
                val endIdx = (startIdx + 1) % 4
                val connectIdx = connects[startIdx]
                val (startMin, startCover, startPeri) = checkOutsideInside(corners[startIdx], connectIdx)
                val (endMin, endCover, endPeri) = checkOutsideInside(corners[endIdx], connectIdx)
                cover += distSeg[connectIdx] + startCover * startMin + endCover * endMin
                perimeter += distSeg[connectIdx] + startPeri * startMin + endPeri * endMin
                squareLength[startIdx] = distSeg[connectIdx] + startPeri * startMin + endPeri * endMin
            }
            overlapScores[s] = cover / perimeter

            // Degree: góc đối diện càng bằng nhau càng tốt
            var degRatio1 = degree[0] / degree[2]
            if (degRatio1 > 1.0) degRatio1 = 1 / degRatio1
            var degRatio2 = degree[1] / degree[3]
            if (degRatio2 > 1.0) degRatio2 = 1 / degRatio2
            degreeScores[s] = (degRatio1 + degRatio2) / 2

            // Length: cạnh đối diện càng bằng nhau càng tốt
            val (len0, len1, len2, len3) = squareLength
            val lenRatio1 = if (len2 > len0) len0 / len2 else len2 / len0
            val lenRatio2 = if (len3 > len1) len1 / len3 else len3 / len1
            lengthScores[s] = (lenRatio1 + lenRatio2) / 2

            // Area: diện tích (công thức shoelace) chia diện tích lưới
            var cross = 0L
            for (k in 0 until 4) {
                val nk = (k + 1) % 4
                cross += sq[k * 2].toLong() * sq[nk * 2 + 1] - sq[k * 2 + 1].toLong() * sq[nk * 2]
            }
            areaScores[s] = 0.5 * abs(cross.toDouble()) / (mapSize * mapSize)

            // Center: tâm tứ giác càng xa tâm ảnh càng bị trừ điểm
            val mx = (sq[0] + sq[2] + sq[4] + sq[6]) / 4.0
            val my = (sq[1] + sq[3] + sq[5] + sq[7]) / 4.0
            val ddx = 128.0 - mx; val ddy = 128.0 - my
            centerScores[s] = sqrt(ddx * ddx + ddy * ddy) / (mapSize / sqrt(2.0))
        }

        // np.max: có NaN thì kết quả là NaN
        val maxOverlap = if (overlapScores.any { it.isNaN() }) Double.NaN else overlapScores.max()
        for (s in 0 until m) overlapScores[s] /= maxOverlap

        val scoreArray = DoubleArray(m) { s ->
            wOverlap * overlapScores[s] + wDegree * degreeScores[s] + wArea * areaScores[s] -
                wCenter * centerScores[s] + wLength * lengthScores[s]
        }
        // np.argsort(...)[::-1]: tăng dần (NaN xếp cuối) rồi đảo ngược
        val ascending = (0 until m).sortedWith { p, q ->
            val a = scoreArray[p]; val b = scoreArray[q]
            when {
                a.isNaN() && b.isNaN() -> 0
                a.isNaN() -> 1
                b.isNaN() -> -1
                else -> a.compareTo(b)
            }
        }
        val order = ascending.reversed()
        scores = DoubleArray(m) { scoreArray[order[it]] }
        squares = order.map { squareList[it] }
    }

    // ===== 6. Đổi về ảnh gốc: nhân 2 (256 -> 512), chia 512, nhân kích thước ảnh =====
    // squares và interPoints bên Python là mảng int32, gán số thực vào bị cắt phần thập phân -> toInt()
    val outSegments = merged.map {
        doubleArrayOf(it[0] * 2 / 512 * imgW, it[1] * 2 / 512 * imgH, it[2] * 2 / 512 * imgW, it[3] * 2 / 512 * imgH)
    }
    val outSquares = squares.map { sq ->
        IntArray(8) { k -> if (k % 2 == 0) ((sq[k] * 2) / 512.0 * imgW).toInt() else ((sq[k] * 2) / 512.0 * imgH).toInt() }
    }
    val outInter = interPoints.map { intArrayOf(((it[0] * 2) / 512.0 * imgW).toInt(), ((it[1] * 2) / 512.0 * imgH).toInt()) }
    return SquaresResult(outSegments, outSquares, scores, outInter)
}

// ===================== Phần dưới: đọc dữ liệu golden và so sánh =====================

fun readBuf(path: String): ByteBuffer = ByteBuffer.wrap(File(path).readBytes()).order(ByteOrder.LITTLE_ENDIAN)
fun readF32(path: String): FloatArray { val b = readBuf(path); return FloatArray(b.remaining() / 4) { b.float } }
fun readF64(path: String): DoubleArray { val b = readBuf(path); return DoubleArray(b.remaining() / 8) { b.double } }
fun readI32(path: String): IntArray { val b = readBuf(path); return IntArray(b.remaining() / 4) { b.int } }

fun main() {
    val meta = File("golden/meta.json").readText()
    val entries = Regex("\"name\": \"([^\"]+)\",\\s*\"h\": (\\d+),\\s*\"w\": (\\d+)").findAll(meta)
    var passed = 0
    var total = 0
    for (mt in entries) {
        val (name, h, w) = mt.destructured
        val g = "golden/$name"
        val r = predSquares(readI32("${g}_pts.bin"), readF32("${g}_scores.bin"), readF32("${g}_vmap.bin"), h.toInt(), w.toInt())

        val expSeg = readF64("${g}_sq_segments.bin").toList().chunked(4)
        val expSq = readI32("${g}_squares.bin").toList().chunked(8)
        val expScore = readF64("${g}_sq_scores.bin")
        val expInter = readI32("${g}_inter.bin").toList().chunked(2)

        // Đoạn gộp: lệch < 0.01 px. Góc và tứ giác: toạ độ nguyên phải khớp hẳn. Điểm: lệch < 1e-6
        val segOk = r.segments.size == expSeg.size &&
            r.segments.zip(expSeg).all { (a, b) -> a.indices.all { abs(a[it] - b[it]) < 0.01 } }
        val interOk = r.interPoints.size == expInter.size &&
            r.interPoints.zip(expInter).all { (a, b) -> a.toList() == b }
        // np.argsort không ổn định với mảng lớn: các tứ giác bằng điểm nhau có thể bị đảo chỗ.
        // Vì vậy so theo từng nhóm cùng điểm (như một tập), không so thứ tự bên trong nhóm.
        val sqOk = r.squares.size == expSq.size && expScore.size == expSq.size &&
            expScore.indices.groupBy { expScore[it] }.values.all { idx ->
                idx.map { r.squares[it].toList() }.sortedBy { it.toString() } == idx.map { expSq[it] }.sortedBy { it.toString() }
            }
        val scoreOk = r.scores.size == expScore.size &&
            r.scores.indices.all { abs(r.scores[it] - expScore[it]) < 1e-6 || (r.scores[it].isNaN() && expScore[it].isNaN()) }

        val ok = segOk && interOk && sqOk && scoreOk
        total++
        if (ok) passed++
        fun mark(b: Boolean) = if (b) "ok" else "SAI"
        println("%-28s %s  doan gop %d/%d %s | goc %d/%d %s | tu giac %d/%d %s | diem %s".format(
            name, if (ok) "PASS" else "FAIL",
            r.segments.size, expSeg.size, mark(segOk),
            r.interPoints.size, expInter.size, mark(interOk),
            r.squares.size, expSq.size, mark(sqOk), mark(scoreOk)))
        if (ok && r.squares.isNotEmpty()) println("    tu giac tot nhat: ${r.squares[0].toList()}")
    }
    println("\nKet qua: $passed/$total PASS")
}
