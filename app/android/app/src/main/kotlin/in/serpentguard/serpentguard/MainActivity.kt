package `in`.serpentguard.serpentguard

import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import android.speech.tts.TextToSpeech
import android.speech.tts.Voice

class MainActivity : FlutterActivity() {
    private var videoFrames: VideoFrames? = null

    private var speech: TextToSpeech? = null
    private var speechStatus: Int? = null
    private val pending = mutableListOf<() -> Unit>()

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        videoFrames = VideoFrames(flutterEngine.dartExecutor.binaryMessenger)
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, "serpentguard/offline_speech")
            .setMethodCallHandler { call, result ->
                when (call.method) {
                    "voices" -> withSpeech(result) {
                        result.success(offlineVoices().map { it.locale.toLanguageTag() }.distinct().sorted())
                    }
                    "speak" -> withSpeech(result) {
                        val language = call.argument<String>("language")
                        val text = call.argument<String>("text")
                        if (language !in setOf("en", "hi", "ta", "te") || text.isNullOrBlank() ||
                            text.length > TextToSpeech.getMaxSpeechInputLength()) {
                            result.error("invalid_card", "Invalid speech language or card length.", null)
                        } else {
                            val voice = offlineVoices().filter { it.locale.language == language }
                                .sortedWith(compareBy<Voice> { it.locale.country != "IN" }.thenBy { it.name }).firstOrNull()
                            if (voice == null) {
                                result.error("voice_missing", "No installed offline voice for this language.", null)
                            } else if (speech!!.setVoice(voice) != TextToSpeech.SUCCESS ||
                                speech!!.speak(text, TextToSpeech.QUEUE_FLUSH, null, "serpentguard-card") != TextToSpeech.SUCCESS) {
                                result.error("speech_failed", "Could not start offline speech.", null)
                            } else result.success(true)
                        }
                    }
                    "stop" -> { speech?.stop(); result.success(null) }
                    else -> result.notImplemented()
                }
            }
    }

    private fun offlineVoices(): List<Voice> = speech?.voices.orEmpty().filter {
        !it.isNetworkConnectionRequired && !it.features.orEmpty().contains(TextToSpeech.Engine.KEY_FEATURE_NOT_INSTALLED)
    }

    private fun withSpeech(result: MethodChannel.Result, action: () -> Unit) {
        val run = {
            if (speechStatus == TextToSpeech.SUCCESS) action()
            else result.error("speech_unavailable", "Offline speech engine unavailable.", null)
        }
        if (speechStatus != null) { run(); return }
        pending.add(run)
        if (speech == null) {
            speech = TextToSpeech(this) { status -> runOnUiThread {
                speechStatus = status
                val ready = pending.toList()
                pending.clear()
                ready.forEach { it() }
            } }
        }
    }

    override fun cleanUpFlutterEngine(flutterEngine: FlutterEngine) {
        videoFrames?.dispose()
        videoFrames = null
        super.cleanUpFlutterEngine(flutterEngine)
    }

    override fun onDestroy() {
        speechStatus = TextToSpeech.ERROR
        val abandoned = pending.toList()
        pending.clear()
        abandoned.forEach { it() }
        speech?.stop()
        speech?.shutdown()
        speech = null
        super.onDestroy()
    }
}
