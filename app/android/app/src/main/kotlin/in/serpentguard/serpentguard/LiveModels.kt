package `in`.serpentguard.serpentguard

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import io.flutter.plugin.common.BinaryMessenger
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import io.flutter.plugin.common.StandardMethodCodec
import java.nio.FloatBuffer

/**
 * The live camera's two model lanes, each on its own background thread, so the
 * detector keeps up with the camera while the segmenter and classifier work on
 * an earlier frame.
 *
 * flutter_onnxruntime runs every call on one serial queue, so a detector call
 * there waits behind a 200 ms classifier call. These sessions use the same
 * ONNX Runtime build (the plugin's onnxruntime-android) and the same options as
 * the plugin (CPU, memory arena on, inter-op 1); only the thread counts differ.
 *
 * Lane "detector": open(path, threads), run(input) -> output0.
 * Lane "verdict": open(segmenter, classifier, snakeCheck?, threads),
 * check(input) -> {name: values} (the M5 snake check, when opened),
 * segment(input) -> probability, classify(input) -> {name: values}.
 * Inputs are (1, 3, size, size) float32; outputs are float32 arrays.
 */
class LiveModels(messenger: BinaryMessenger) {
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    @Volatile private var detector: OrtSession? = null
    @Volatile private var segmenter: OrtSession? = null
    @Volatile private var classifier: OrtSession? = null
    @Volatile private var snakeCheck: OrtSession? = null

    // A session is closed only while no call is running it: each lane's calls
    // and its close hold that lane's lock (dispose runs on the main thread,
    // the calls on the lanes' background queues).
    private val detectorLock = Any()
    private val verdictLock = Any()

    // The Dart session that opened each lane. A close from another (older)
    // session is ignored, so it cannot close a newer screen's models.
    private var detectorOwner: Int? = null
    private var verdictOwner: Int? = null
    private val detectorChannel = MethodChannel(
        messenger, "serpentguard/live_detector", StandardMethodCodec.INSTANCE, messenger.makeBackgroundTaskQueue()
    )
    private val verdictChannel = MethodChannel(
        messenger, "serpentguard/live_verdict", StandardMethodCodec.INSTANCE, messenger.makeBackgroundTaskQueue()
    )

    init {
        detectorChannel.setMethodCallHandler { call, result ->
            guard(result) { synchronized(detectorLock) { onDetector(call, result) } }
        }
        verdictChannel.setMethodCallHandler { call, result ->
            guard(result) { synchronized(verdictLock) { onVerdict(call, result) } }
        }
    }

    private fun guard(result: MethodChannel.Result, body: () -> Unit) {
        try {
            body()
        } catch (e: Throwable) {
            result.error("live_models", e.message ?: e.javaClass.simpleName, null)
        }
    }

    /** A session with the plugin's options; the options object is released after use. */
    private fun session(path: String, threads: Int): OrtSession =
        OrtSession.SessionOptions().use { options ->
            options.setIntraOpNumThreads(threads)
            options.setInterOpNumThreads(1)
            options.setCPUArenaAllocator(true)
            env.createSession(path, options)
        }

    /** True when [call] may close a lane opened by [owner] (no owner given: always). */
    private fun mayClose(call: MethodCall, owner: Int?): Boolean {
        val from = call.argument<Int>("owner") ?: return true
        return owner == null || from == owner
    }

    private fun onDetector(call: MethodCall, result: MethodChannel.Result) {
        when (call.method) {
            "cores" -> result.success(Runtime.getRuntime().availableProcessors())
            "open" -> {
                // Cleared first, so a failed open never leaves a closed session reachable.
                detector?.close()
                detector = null
                detector = session(call.argument<String>("path")!!, call.argument<Int>("threads")!!)
                detectorOwner = call.argument<Int>("owner")
                result.success(null)
            }
            "run" -> {
                val session = detector ?: throw IllegalStateException("detector not open")
                result.success(run(session, call)["output0"])
            }
            "close" -> {
                if (mayClose(call, detectorOwner)) {
                    detector?.close()
                    detector = null
                    detectorOwner = null
                }
                result.success(null)
            }
            else -> result.notImplemented()
        }
    }

    private fun onVerdict(call: MethodCall, result: MethodChannel.Result) {
        when (call.method) {
            "open" -> {
                val threads = call.argument<Int>("threads")!!
                closeVerdict()
                segmenter = session(call.argument<String>("segmenter")!!, threads)
                classifier = session(call.argument<String>("classifier")!!, threads)
                snakeCheck = call.argument<String>("snakeCheck")?.let { session(it, threads) }
                verdictOwner = call.argument<Int>("owner")
                result.success(null)
            }
            "check" -> {
                val session = snakeCheck ?: throw IllegalStateException("snake check not open")
                result.success(run(session, call))
            }
            "segment" -> {
                val session = segmenter ?: throw IllegalStateException("segmenter not open")
                result.success(run(session, call)["probability"])
            }
            "classify" -> {
                val session = classifier ?: throw IllegalStateException("classifier not open")
                result.success(run(session, call))
            }
            "close" -> {
                if (mayClose(call, verdictOwner)) {
                    closeVerdict()
                    verdictOwner = null
                }
                result.success(null)
            }
            else -> result.notImplemented()
        }
    }

    private fun closeVerdict() {
        segmenter?.close()
        classifier?.close()
        snakeCheck?.close()
        segmenter = null
        classifier = null
        snakeCheck = null
    }

    /** One (1, 3, size, size) float32 input named "images"; every output as a float array. */
    private fun run(session: OrtSession, call: MethodCall): Map<String, FloatArray> {
        val input = call.argument<FloatArray>("input")!!
        val size = call.argument<Int>("size")!!.toLong()
        OnnxTensor.createTensor(env, FloatBuffer.wrap(input), longArrayOf(1, 3, size, size)).use { tensor ->
            session.run(mapOf("images" to tensor)).use { outputs ->
                val out = LinkedHashMap<String, FloatArray>()
                for ((name, value) in outputs) {
                    val buffer = (value as OnnxTensor).floatBuffer
                    out[name] = FloatArray(buffer.remaining()).also { buffer.get(it) }
                }
                return out
            }
        }
    }

    fun dispose() {
        detectorChannel.setMethodCallHandler(null)
        verdictChannel.setMethodCallHandler(null)
        // Waits for a call still running on a lane (at most one model run).
        synchronized(detectorLock) {
            detector?.close()
            detector = null
        }
        synchronized(verdictLock) { closeVerdict() }
    }
}
