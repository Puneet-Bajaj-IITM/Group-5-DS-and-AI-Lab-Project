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
    private val worker = Executors.newSingleThreadExecutor()
    private val main = Handler(Looper.getMainLooper())
    private val open = mutableMapOf<Int, MediaMetadataRetriever>()
    private var nextId = 1

    init {
        MethodChannel(messenger, "serpentguard/video_frames").setMethodCallHandler { call, result ->
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
                        "close" -> { open.remove(call.argument<Int>("id")!!)?.release(); null }
                        "decodeImage" -> decodeImage(call.argument<ByteArray>("bytes")!!)
                        else -> { main.post { result.notImplemented() }; return@execute }
                    }
                    main.post { result.success(value) }
                } catch (e: Exception) {
                    main.post { result.error("video_error", e.message ?: e.toString(), null) }
                }
            }
        }
    }

    private fun openClip(path: String): Map<String, Any?> {
        val r = MediaMetadataRetriever()
        r.setDataSource(path)
        fun meta(key: Int) = r.extractMetadata(key)
        val id = nextId++
        open[id] = r
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
        val bitmaps = r.getFramesAtIndex(start, count)
        val out = mutableListOf<Map<String, Any>>()
        bitmaps.forEachIndexed { i, bmp ->
            if (i % step == 0) out.add(mapOf("index" to start + i) + pixels(bmp, maxSide))
            bmp.recycle()
        }
        return out
    }

    private fun frameAt(id: Int, timeMs: Long, maxSide: Int): Map<String, Any>? {
        val r = open[id] ?: throw IllegalStateException("clip $id is not open")
        val bmp = r.getFrameAtTime(timeMs * 1000, MediaMetadataRetriever.OPTION_CLOSEST) ?: return null
        return pixels(bmp, maxSide).also { bmp.recycle() }
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
