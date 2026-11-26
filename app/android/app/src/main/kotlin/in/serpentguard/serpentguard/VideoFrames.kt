package `in`.serpentguard.serpentguard

import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.os.Build
import android.os.Handler
import android.os.Looper
import io.flutter.plugin.common.BinaryMessenger
import io.flutter.plugin.common.MethodChannel
import java.nio.ByteBuffer
import java.util.concurrent.Executors

/**
 * Reads frames from a video file for the app's video check, off the UI thread.
 *
 * "open" returns the clip's duration, frame count and size; "frames" decodes a run of
 * consecutive frames (one sequential decode, much faster than seeking to each frame),
 * keeps every [step]-th, scales it to [maxSide] on its long side and returns RGBA bytes.
 */
class VideoFrames(messenger: BinaryMessenger) {
    companion object {
        private const val FRAME_BUDGET_BYTES = 48L * 1024 * 1024
    }

    private val worker = Executors.newSingleThreadExecutor()
    private val main = Handler(Looper.getMainLooper())
    private val open = mutableMapOf<Int, MediaMetadataRetriever>()
    /** Per open clip: rotation in degrees and the encoded (unrotated) width and height. */
    private val geometry = mutableMapOf<Int, Triple<Int, Int, Int>>()
    private var nextId = 1

    init {
        MethodChannel(messenger, "serpentguard/video_frames").setMethodCallHandler { call, result ->
            if (worker.isShutdown) {
                result.error("video_error", "closing", null)
                return@setMethodCallHandler
            }
            worker.execute {
                try {
                    val value: Any? = when (call.method) {
                        "open" -> openClip(call.argument<String>("path")!!)
                        "frames" -> frames(
                            call.argument<Int>("id")!!, call.argument<Int>("start")!!,
                            call.argument<Int>("count")!!, call.argument<Int>("step")!!,
                            call.argument<Int>("maxSide")!!,
                        )
                        "frameAt" -> frameAt(
                            call.argument<Int>("id")!!, call.argument<Number>("timeMs")!!.toLong(),
                            call.argument<Int>("maxSide")!!,
                        )
                        "close" -> {
                            val id = call.argument<Int>("id")!!
                            geometry.remove(id)
                            open.remove(id)?.release()
                            null
                        }
                        "decodeImage" -> decodeImage(call.argument<ByteArray>("bytes")!!)
                        "sdkInt" -> Build.VERSION.SDK_INT
                        else -> { main.post { result.notImplemented() }; return@execute }
                    }
                    main.post { result.success(value) }
                } catch (e: Throwable) {
                    // Throwable, so an OutOfMemoryError is reported instead of killing the app.
                    main.post { result.error("video_error", e.message ?: e.toString(), null) }
                }
            }
        }
    }

    /** Releases every open clip and the worker thread (the Flutter engine is going away). */
    fun dispose() {
        worker.execute {
            open.values.forEach { runCatching { it.release() } }
            open.clear()
            geometry.clear()
        }
        worker.shutdown()
    }

    private fun openClip(path: String): Map<String, Any?> {
        val r = MediaMetadataRetriever()
        try {
            r.setDataSource(path)
        } catch (e: Exception) {
            r.release()
            throw e
        }
        fun meta(key: Int) = r.extractMetadata(key)
        val id = nextId++
        open[id] = r
        geometry[id] = Triple(
            meta(MediaMetadataRetriever.METADATA_KEY_VIDEO_ROTATION)?.toIntOrNull() ?: 0,
            meta(MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH)?.toIntOrNull() ?: 0,
            meta(MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT)?.toIntOrNull() ?: 0,
        )
        val count = if (Build.VERSION.SDK_INT >= 28)
            meta(MediaMetadataRetriever.METADATA_KEY_VIDEO_FRAME_COUNT)?.toIntOrNull() else null
        return mapOf(
            "id" to id,
            "durationMs" to (meta(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLongOrNull() ?: 0L),
            "frameCount" to count,
            "width" to meta(MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH)?.toIntOrNull(),
            "height" to meta(MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT)?.toIntOrNull(),
            "rotation" to meta(MediaMetadataRetriever.METADATA_KEY_VIDEO_ROTATION)?.toIntOrNull(),
        )
    }

    private fun frames(id: Int, start: Int, count: Int, step: Int, maxSide: Int): List<Map<String, Any>> {
        val r = open[id] ?: throw IllegalStateException("clip $id is not open")
        if (Build.VERSION.SDK_INT < 28) throw UnsupportedOperationException("needs Android 9")
        val (_, w, h) = geometry[id] ?: Triple(0, 1920, 1080)
        // Decoded frames are 2 bytes a pixel (RGB_565). Keep each batch under
        // the budget so a 4K clip cannot run the phone out of memory.
        val bytesPerFrame = maxOf(1L, w.toLong() * h * 2)
        val params = MediaMetadataRetriever.BitmapParams().apply { preferredConfig = Bitmap.Config.RGB_565 }
        val out = mutableListOf<Map<String, Any>>()
        fun keep(index: Int, bmp: Bitmap) {
            val turned = upright(id, bmp)
            out.add(mapOf("index" to index) + pixels(turned, maxSide))
            if (turned !== bmp) turned.recycle()
        }
        val end = start + count
        val perBatch = (FRAME_BUDGET_BYTES / bytesPerFrame).toInt()
        if (perBatch >= step) {
            // One sequential decode per batch (fast), keeping every step-th frame.
            val run = maxOf(step, perBatch / step * step)
            var s = start
            while (s < end) {
                val n = minOf(run, end - s)
                val bitmaps = r.getFramesAtIndex(s, n, params)
                bitmaps.forEachIndexed { i, bmp ->
                    if ((s + i - start) % step == 0) keep(s + i, bmp)
                    bmp.recycle()
                }
                s += n
            }
        } else {
            // Very large frames: one kept frame at a time.
            var i = start
            while (i < end) {
                r.getFrameAtIndex(i, params)?.let { keep(i, it); it.recycle() }
                i += step
            }
        }
        return out
    }

    private fun frameAt(id: Int, timeMs: Long, maxSide: Int): Map<String, Any>? {
        val r = open[id] ?: throw IllegalStateException("clip $id is not open")
        val bmp = r.getFrameAtTime(timeMs * 1000, MediaMetadataRetriever.OPTION_CLOSEST) ?: return null
        val turned = upright(id, bmp)
        return pixels(turned, maxSide).also { if (turned !== bmp) turned.recycle(); bmp.recycle() }
    }

    /**
     * Decodes a photo with Android's BitmapFactory (libjpeg-turbo, accurate DCT), upright per its
     * EXIF orientation. Returns RGBA bytes.
     */
    private fun decodeImage(bytes: ByteArray): Map<String, Any> {
        val options = android.graphics.BitmapFactory.Options().apply {
            inPreferredConfig = Bitmap.Config.ARGB_8888
            inScaled = false
            @Suppress("DEPRECATION")
            inPreferQualityOverSpeed = true
        }
        var bmp = android.graphics.BitmapFactory.decodeByteArray(bytes, 0, bytes.size, options)
            ?: throw IllegalArgumentException("not an image")
        val orientation = try {
            androidx.exifinterface.media.ExifInterface(java.io.ByteArrayInputStream(bytes)).getAttributeInt(
                androidx.exifinterface.media.ExifInterface.TAG_ORIENTATION,
                androidx.exifinterface.media.ExifInterface.ORIENTATION_NORMAL,
            )
        } catch (e: Exception) { androidx.exifinterface.media.ExifInterface.ORIENTATION_NORMAL }
        val m = android.graphics.Matrix()
        when (orientation) {
            androidx.exifinterface.media.ExifInterface.ORIENTATION_ROTATE_90 -> m.postRotate(90f)
            androidx.exifinterface.media.ExifInterface.ORIENTATION_ROTATE_180 -> m.postRotate(180f)
            androidx.exifinterface.media.ExifInterface.ORIENTATION_ROTATE_270 -> m.postRotate(270f)
            androidx.exifinterface.media.ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> m.postScale(-1f, 1f)
            androidx.exifinterface.media.ExifInterface.ORIENTATION_FLIP_VERTICAL -> m.postScale(1f, -1f)
            androidx.exifinterface.media.ExifInterface.ORIENTATION_TRANSPOSE -> { m.postRotate(90f); m.postScale(-1f, 1f) }
            androidx.exifinterface.media.ExifInterface.ORIENTATION_TRANSVERSE -> { m.postRotate(270f); m.postScale(-1f, 1f) }
        }
        if (!m.isIdentity) {
            val turned = Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, m, false)
            if (turned !== bmp) bmp.recycle()
            bmp = turned
        }
        val buffer = ByteBuffer.allocate(bmp.byteCount)
        bmp.copyPixelsToBuffer(buffer)
        return mapOf("width" to bmp.width, "height" to bmp.height, "rgba" to buffer.array()).also { bmp.recycle() }
    }

    /**
     * Phone videos filmed upright are stored sideways with a rotation flag. Some Android
     * versions return frames already turned, others not; if the frame still has the encoded
     * (unrotated) shape, turn it so the model sees the snake upright.
     */
    private fun upright(id: Int, bmp: Bitmap): Bitmap {
        val (rotation, w, h) = geometry[id] ?: return bmp
        if (rotation % 360 == 0 || w == 0 || h == 0) return bmp
        val stillEncoded = bmp.width == w && bmp.height == h && (rotation == 180 || w != h)
        if (!stillEncoded) return bmp
        val m = android.graphics.Matrix().apply { postRotate(rotation.toFloat()) }
        return Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, m, true)
    }

    /** RGBA bytes of [src] scaled (bilinear) so its long side is at most [maxSide]. */
    private fun pixels(src: Bitmap, maxSide: Int): Map<String, Any> {
        val long = maxOf(src.width, src.height)
        var bmp = src
        if (long > maxSide) {
            val scale = maxSide.toDouble() / long
            bmp = Bitmap.createScaledBitmap(
                src, maxOf(1, Math.round(src.width * scale).toInt()),
                maxOf(1, Math.round(src.height * scale).toInt()), true,
            )
        }
        val argb = if (bmp.config == Bitmap.Config.ARGB_8888) bmp else bmp.copy(Bitmap.Config.ARGB_8888, false)
        val buffer = ByteBuffer.allocate(argb.byteCount)
        argb.copyPixelsToBuffer(buffer) // ARGB_8888 is stored R, G, B, A in memory
        val result = mapOf("width" to argb.width, "height" to argb.height, "rgba" to buffer.array())
        if (argb !== bmp) argb.recycle()
        if (bmp !== src) bmp.recycle()
        return result
    }
}
