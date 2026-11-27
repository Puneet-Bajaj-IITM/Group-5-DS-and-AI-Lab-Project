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
 * Lane "verdict": open(segmenter, classifier, threads), segment(input) ->
 * probability, classify(input) -> {name: values}.
 * Inputs are (1, 3, size, size) float32; outputs are float32 arrays.
 */
class LiveModels(messenger: BinaryMessenger) {
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    @Volatile private var detector: OrtSession? = null
    @Volatile private var segmenter: OrtSession? = null
    @Volatile private var classifier: OrtSession? = null
    private val detectorChannel = MethodChannel(
        messenger, "serpentguard/live_detector", StandardMethodCodec.INSTANCE, messenger.makeBackgroundTaskQueue()
    )
    private val verdictChannel = MethodChannel(
        messenger, "serpentguard/live_verdict", StandardMethodCodec.INSTANCE, messenger.makeBackgroundTaskQueue()
    )

    init {
        detectorChannel.setMethodCallHandler { call, result -> guard(result) { onDetector(call, result) } }
        verdictChannel.setMethodCallHandler { call, result -> guard(result) { onVerdict(call, result) } }
    }

    private fun guard(result: MethodChannel.Result, body: () -> Unit) {
        try {
            body()
        } catch (e: Throwable) {
            result.error("live_models", e.message ?: e.javaClass.simpleName, null)
        }
    }

    private fun options(threads: Int) = OrtSession.SessionOptions().apply {
        setIntraOpNumThreads(threads)
        setInterOpNumThreads(1)
        setCPUArenaAllocator(true)
    }

    private fun onDetector(call: MethodCall, result: MethodChannel.Result) {
        when (call.method) {
            "cores" -> result.success(Runtime.getRuntime().availableProcessors())
            "open" -> {
                detector?.close()
                detector = env.createSession(call.argument<String>("path")!!, options(call.argument<Int>("threads")!!))
                result.success(null)
            }
            "run" -> {
                val session = detector ?: throw IllegalStateException("detector not open")
                result.success(run(session, call)["output0"])
            }
            "close" -> {
                detector?.close()
                detector = null
                result.success(null)
            }
            else -> result.notImplemented()
        }
    }

    private fun onVerdict(call: MethodCall, result: MethodChannel.Result) {
        when (call.method) {
            "open" -> {
                val threads = call.argument<Int>("threads")!!
                segmenter?.close()
                classifier?.close()
                segmenter = env.createSession(call.argument<String>("segmenter")!!, options(threads))
                classifier = env.createSession(call.argument<String>("classifier")!!, options(threads))
                result.success(null)
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
                segmenter?.close()
                classifier?.close()
                segmenter = null
                classifier = null
                result.success(null)
            }
            else -> result.notImplemented()
        }
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
        detector?.close()
        segmenter?.close()
        classifier?.close()
        detector = null
        segmenter = null
        classifier = null
    }
}
