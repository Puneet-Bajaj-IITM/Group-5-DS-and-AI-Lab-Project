import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/services.dart' show AssetManifest, MethodChannel, rootBundle;
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'package:path_provider/path_provider.dart';

import 'bundle.dart';
import 'decision.dart';
import 'image_ops.dart';

/// Where the three fp32 models live in the app bundle. They are not in git
/// (54 MB); copy them in from the model bundle before building (see README).
const modelAssetDir = 'assets/models';
const modelFiles = ['classifier.onnx', 'detector.onnx', 'segmenter.onnx'];

/// What the gate saw on one image.
class GateResult {
  const GateResult({required this.snake, required this.boxConf, required this.maskArea});
  final bool snake;
  final double? boxConf;
  final double maskArea;
}

/// One image through the whole path, with the intermediate values the
/// golden-photo check compares (`golden_photos.json` "steps").
class PipelineResult {
  const PipelineResult({
    required this.width,
    required this.height,
    required this.exposureFixed,
    required this.gate,
    required this.answer,
    this.cropWindow,
    this.parts,
    this.classifierInputMean,
    this.classifierInputStd,
    this.timingsMs = const {},
  });

  final int width, height;
  final bool exposureFixed;
  final GateResult gate;

  /// (top, bottom, left, right), or null for the whole photo / no snake.
  final (int, int, int, int)? cropWindow;
  final ModelAnswer answer;

  /// The probabilities and embedding the decision read; null when no snake.
  /// A video pools these over frames.
  final DecisionParts? parts;
  final double? classifierInputMean, classifierInputStd;
  final Map<String, int> timingsMs;

  int get totalMs => timingsMs.values.fold(0, (a, b) => a + b);
}

/// The detector, segmenter and classifier with the settings from
/// `pipeline.json` and `tables.json`: a port of `reference_pipeline.py`'s
/// `Pipeline` class.
class SnakePipeline {
  SnakePipeline._(this.settings, this.tables, this._detector, this._segmenter, this._classifier)
      : rules = DecisionRules(settings.thresholds, tables);

  final PipelineSettings settings;
  final ModelTables tables;
  final DecisionRules rules;
  final OrtSession _detector, _segmenter, _classifier;

  /// True when the three model files are in the app bundle. Reads only the
  /// asset manifest, so it is quick.
  static Future<bool> modelsBundled() async {
    try {
      final manifest = await AssetManifest.loadFromAssetBundle(rootBundle);
      final assets = manifest.listAssets().toSet();
      return modelFiles.every((f) => assets.contains('$modelAssetDir/$f'));
    } catch (_) {
      return false;
    }
  }

  /// The settings and tables (committed under assets/model/).
  static Future<(PipelineSettings, ModelTables)> loadConfig() async => (
        PipelineSettings.parse(await rootBundle.loadString('assets/model/pipeline.json')),
        ModelTables.parse(await rootBundle.loadString('assets/model/tables.json')),
      );

  /// Opens the three ONNX sessions. Throws if a model file is missing.
  static Future<SnakePipeline> load(PipelineSettings settings, ModelTables tables) async {
    final ort = OnnxRuntime();
    final options = OrtSessionOptions(intraOpNumThreads: 2, interOpNumThreads: 1);
    Future<OrtSession> open(String name) async =>
        ort.createSession(await _extract(name), options: options);
    final sessions = await Future.wait([for (final f in modelFiles) open(f)]);
    return SnakePipeline._(settings, tables, sessions[1], sessions[2], sessions[0]);
  }

  /// ONNX Runtime needs a file path. Copies the asset out once, and again
  /// only when its size changes (a new model build).
  static Future<String> _extract(String name) async {
    final data = await rootBundle.load('$modelAssetDir/$name');
    final dir = Directory('${(await getApplicationSupportDirectory()).path}/models');
    await dir.create(recursive: true);
    final file = File('${dir.path}/$name');
    if (!await file.exists() || await file.length() != data.lengthInBytes) {
      final tmp = File('${file.path}.part');
      await tmp.writeAsBytes(data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes), flush: true);
      await tmp.rename(file.path);
    }
    return file.path;
  }

  /// Decodes a photo upright (EXIF rotation applied) as 8-bit RGB, with the
  /// same pixels PIL gives the reference. On Android that is BitmapFactory:
  /// Flutter's own decoder there gave different pixels on 3 of the 5 golden
  /// photos, enough to move the detector score by 0.2. Elsewhere (iOS, the
  /// test host) Flutter's decoder matches PIL.
  static Future<RgbImage> decode(Uint8List bytes) async {
    if (Platform.isAndroid) {
      final n = (await const MethodChannel('serpentguard/video_frames')
          .invokeMapMethod<String, dynamic>('decodeImage', {'bytes': bytes}))!;
      return RgbImage.fromRgba(n['width'] as int, n['height'] as int, n['rgba'] as Uint8List);
    }
    final codec = await ui.instantiateImageCodec(bytes);
    final frame = await codec.getNextFrame();
    final image = frame.image;
    final rgba = (await image.toByteData(format: ui.ImageByteFormat.rawRgba))!;
    final out = RgbImage.fromRgba(image.width, image.height, rgba.buffer.asUint8List());
    image.dispose();
    codec.dispose();
    return out;
  }

  Future<List<double>> _run(OrtSession session, Float32List input, String output) async {
    final value = await OrtValue.fromList(input, [1, 3, settings.inputSize, settings.inputSize]);
    try {
      final outputs = await session.run({'images': value});
      try {
        final list = await outputs[output]!.asFlattenedList();
        return [for (final v in list) (v as num).toDouble()];
      } finally {
        for (final o in outputs.values) {
          await o.dispose();
        }
      }
    } finally {
      await value.dispose();
    }
  }

  Future<Map<String, List<double>>> _runAll(OrtSession session, Float32List input) async {
    final value = await OrtValue.fromList(input, [1, 3, settings.inputSize, settings.inputSize]);
    try {
      final outputs = await session.run({'images': value});
      try {
        return {
          for (final e in outputs.entries)
            e.key: [for (final v in await e.value.asFlattenedList()) (v as num).toDouble()]
        };
      } finally {
        for (final o in outputs.values) {
          await o.dispose();
        }
      }
    } finally {
      await value.dispose();
    }
  }

  /// One photo (as decoded, before the exposure fix) to the app's answer.
  Future<PipelineResult> run(RgbImage photo) async {
    final s = settings;
    final size = s.inputSize;
    final times = <String, int>{};
    final clock = Stopwatch()..start();
    int lap() {
      final ms = clock.elapsedMilliseconds;
      clock.reset();
      return ms;
    }

    final rgb = fixExposure(photo, s.exposureBelow, s.exposureAbove, s.claheClip, s.claheTiles);
    final fixed = !identical(rgb, photo);
    times['exposure'] = lap();

    // Gate: detector box score and U-Net mask area.
    final detected = await _run(_detector, detectorInput(rgb, size, s.detectorPad), 'output0');
    final anchors = detected.length ~/ 5; // rows cx, cy, w, h and the snake score
    final conf = boxConf(detected, anchors, s.snakeClass, s.boxFloor);
    times['detector'] = lap();

    final segIn = normalise(letterbox(rgb, size, s.padRgb), s.mean, s.std, s.scale);
    final probability = Float32List.fromList(await _run(_segmenter, segIn, 'probability'));
    final mask = unletterboxMask(probability, rgb.height, rgb.width, size, s.segmenterThreshold);
    var maskPixels = 0;
    for (final v in mask) {
      maskPixels += v;
    }
    final area = maskPixels / mask.length;
    times['segmenter'] = lap();

    final snake = conf != null && conf >= s.gateBoxConf && area > 0 && area >= s.gateMaskArea;
    final gate = GateResult(snake: snake, boxConf: conf, maskArea: area);
    if (!snake) {
      return PipelineResult(
        width: rgb.width,
        height: rgb.height,
        exposureFixed: fixed,
        gate: gate,
        answer: ModelAnswer.noSnake,
        timingsMs: times,
      );
    }

    final cleaned = cleanMask(mask, rgb.width, rgb.height,
        connectivity: s.connectivity,
        minLargestShare: s.minLargestShare,
        keepShareOfLargest: s.keepShareOfLargest,
        keepShareOfPhoto: s.keepShareOfPhoto);
    final window = cropWindow(cleaned, rgb.height, rgb.width, s.cropPadding);
    final crop = window == null ? rgb : rgb.crop(window.$1, window.$2, window.$3, window.$4);
    times['crop'] = lap();

    final clsIn = normalise(letterbox(crop, size, s.padRgb), s.mean, s.std, s.scale);
    var sum = 0.0, sq = 0.0;
    for (final v in clsIn) {
      sum += v;
      sq += v * v;
    }
    final mean = sum / clsIn.length;
    final std = (sq / clsIn.length - mean * mean).clamp(0, double.infinity).toDouble();
    final out = await _runAll(_classifier, clsIn);
    times['classifier'] = lap();

    final outputs = ModelOutputs(
      embedding: out['embedding']!,
      speciesLogits: out['species_logits']!,
      whoLogits: out['who_logits']!,
      genusLogits: out['genus_logits']!,
      familyLogits: out['family_logits']!,
    );
    final parts = DecisionParts.fromOutputs(outputs, s.thresholds);
    final answer = rules.verdict(parts);
    times['decision'] = lap();
    return PipelineResult(
      width: rgb.width,
      height: rgb.height,
      exposureFixed: fixed,
      gate: gate,
      cropWindow: window,
      answer: answer,
      parts: parts,
      classifierInputMean: mean,
      classifierInputStd: math.sqrt(std),
      timingsMs: times,
    );
  }

  Future<void> close() async {
    await _detector.close();
    await _segmenter.close();
    await _classifier.close();
  }
}
