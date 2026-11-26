# Release builds shrink and rename Java classes (R8). ONNX Runtime's native
# library looks up its Java classes and methods by name through JNI, so a
# renamed class crashes the app the first time a model runs:
#   JNI DETECTED ERROR IN APPLICATION: java_class == null in call to GetMethodID
#   from ai.onnxruntime.OrtSession.run
# Keep the whole ai.onnxruntime package unchanged.
-keep class ai.onnxruntime.** { *; }
-keepclassmembers class ai.onnxruntime.** { *; }
-keepnames class ai.onnxruntime.** { *; }
