import 'dart:async';
import 'dart:io';
import 'dart:isolate';
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
  const GateResult({required this.snake, required this.boxConf, required this.maskArea, this.box});
  final bool snake;
  final double? boxConf;
  final double maskArea;

  /// The detector's most confident box in image pixels (drawn on screen).
  final PixelBox? box;
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
  SnakePipeline._(this.settings, this.tables, this._detector, this._segmenter, this._classifier, this._worker)
      : rules = DecisionRules(settings.thresholds, tables);

  final _PixelWorker _worker;

  /// Up to this many pixels (a 640 x 480 frame) the pixel work runs on the
  /// calling isolate; above it, on the background worker.
  static const inlinePixelLimit = 640 * 480;

  RgbImage? _inlineRgb;
  _Prepared _prepareHere(RgbImage photo) {
    final s = settings;
    final rgb = fixExposure(photo, s.exposureBelow, s.exposureAbove, s.claheClip, s.claheTiles);
    _inlineRgb = rgb;
    return _Prepared(-1, rgb.width, rgb.height, !identical(rgb, photo), detectorInput(rgb, s.inputSize, s.detectorPad),
        normalise(letterbox(rgb, s.inputSize, s.padRgb), s.mean, s.std, s.scale));
  }

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
    final opened = <OrtSession>[];
    try {
      for (final f in modelFiles) {
        opened.add(await open(f));
      }
    } catch (_) {
      for (final s in opened) {
        await s.close();
      }
      rethrow;
    }
    final worker = await _PixelWorker.start(settings);
    return SnakePipeline._(settings, tables, opened[1], opened[2], opened[0], worker);
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
  ///
  /// The pixel work runs in background isolates so the screen keeps
  /// drawing; only the three ONNX calls stay on this isolate (the plugin
  /// needs it).
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

    // Big photos go to the worker so the screen keeps drawing. Video and
    // live frames (640 px) are cheap and run here: on a 4-core phone the
    // worker isolate slowed the models themselves about 2x (measured
    // 180 ms vs 370 ms a frame), which costs more than the ~20 ms it saves.
    final inline = photo.width * photo.height <= inlinePixelLimit;
    final pre = inline ? _prepareHere(photo) : await _worker.prepare(photo);
    times['exposure'] = lap();

    // Gate: detector box score and U-Net mask area.
    final detected = await _run(_detector, pre.detectorInput, 'output0');
    final anchors = detected.length ~/ 5; // rows cx, cy, w, h and the snake score
    final conf = boxConf(detected, anchors, s.snakeClass, s.boxFloor);
    final box = conf == null ? null : bestBox(detected, anchors, s.snakeClass, pre.width, pre.height, size);
    times['detector'] = lap();

    final probability = Float32List.fromList(await _run(_segmenter, pre.segmenterInput, 'probability'));
    times['segmenter'] = lap();
    final post = inline ? _afterGate(_inlineRgb!, probability, conf, s) : await _worker.afterGate(pre.job, probability, conf);
    times['crop'] = lap();

    final gate = GateResult(snake: post.snake, boxConf: conf, maskArea: post.area, box: box);
    if (!post.snake) {
      return PipelineResult(
        width: pre.width,
        height: pre.height,
        exposureFixed: pre.fixed,
        gate: gate,
        answer: ModelAnswer.noSnake,
        timingsMs: times,
      );
    }

    final out = await _runAll(_classifier, post.classifierInput!);
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
      width: pre.width,
      height: pre.height,
      exposureFixed: pre.fixed,
      gate: gate,
      cropWindow: post.window,
      answer: answer,
      parts: parts,
      classifierInputMean: post.mean,
      classifierInputStd: post.std,
      timingsMs: times,
    );
  }

  Future<void> close() async {
    _worker.close();
    await _detector.close();
    await _segmenter.close();
    await _classifier.close();
  }
}

// ------------------------------------------------- isolate steps (pure Dart)

/// What the worker sends back after the first step. The photo itself stays
/// in the worker (under [job]) for the second step.
class _Prepared {
  const _Prepared(this.job, this.width, this.height, this.fixed, this.detectorInput, this.segmenterInput);
  final int job, width, height;
  final bool fixed;
  final Float32List detectorInput, segmenterInput;
}

class _AfterGate {
  const _AfterGate({required this.snake, required this.area, this.window, this.classifierInput, this.mean, this.std});
  final bool snake;
  final double area;
  final (int, int, int, int)? window;
  final Float32List? classifierInput;
  final double? mean, std;
}

/// One long-lived background isolate for the pixel work. Spawned once with
/// the settings, so each photo costs only copying its pixels across, not
/// starting an isolate (which, with `Isolate.run` closures, also copied the
/// pipeline's tables every time and made each video frame ~3x slower).
class _PixelWorker {
  _PixelWorker._(this._isolate, this._commands, ReceivePort replies) {
    replies.listen((msg) {
      final (int id, Object? value, String? error) = msg as (int, Object?, String?);
      final c = _pending.remove(id);
      if (c == null) return;
      error == null ? c.complete(value) : c.completeError(StateError(error));
    });
    _replies = replies;
  }

  final Isolate _isolate;
  final SendPort _commands;
  late final ReceivePort _replies;
  final _pending = <int, Completer<Object?>>{};
  int _next = 0;

  static Future<_PixelWorker> start(PipelineSettings settings) async {
    final replies = ReceivePort();
    final ready = Completer<SendPort>();
    final first = replies.first.then((m) => ready.complete(m as SendPort));
    final isolate = await Isolate.spawn(_workerMain, (replies.sendPort, settings.raw));
    await first;
    // `first` closed the port's subscription; a new port carries replies.
    final answers = ReceivePort();
    final commands = await ready.future;
    commands.send(answers.sendPort);
    return _PixelWorker._(isolate, commands, answers);
  }

  Future<T> _ask<T>(String op, List<Object?> args) {
    final id = _next++;
    final c = Completer<Object?>();
    _pending[id] = c;
    _commands.send((id, op, args));
    return c.future.then((v) => v as T);
  }

  Future<_Prepared> prepare(RgbImage photo) => _ask('prepare', [photo]);

  Future<_AfterGate> afterGate(int job, Float32List probability, double? conf) =>
      _ask('after', [job, probability, conf]);

  void close() {
    _replies.close();
    _isolate.kill(priority: Isolate.immediate);
  }
}

/// The worker isolate: keeps each photo between its two steps.
Future<void> _workerMain((SendPort, Map<String, dynamic>) init) async {
  final (boot, raw) = init;
  final s = PipelineSettings(raw);
  final inbox = ReceivePort();
  boot.send(inbox.sendPort);
  late SendPort out;
  final photos = <int, RgbImage>{};
  var jobs = 0;
  await for (final msg in inbox) {
    if (msg is SendPort) {
      out = msg;
      continue;
    }
    final (int id, String op, List<Object?> args) = msg as (int, String, List<Object?>);
    try {
      if (op == 'prepare') {
        final photo = args[0] as RgbImage;
        final rgb = fixExposure(photo, s.exposureBelow, s.exposureAbove, s.claheClip, s.claheTiles);
        final job = jobs++;
        photos[job] = rgb;
        // A job whose second step never comes (an error between steps)
        // must not keep its photo forever.
        photos.removeWhere((k, _) => k < job - 8);
        out.send((
          id,
          _Prepared(
            job,
            rgb.width,
            rgb.height,
            !identical(rgb, photo),
            detectorInput(rgb, s.inputSize, s.detectorPad),
            normalise(letterbox(rgb, s.inputSize, s.padRgb), s.mean, s.std, s.scale),
          ),
          null,
        ));
      } else {
        final rgb = photos.remove(args[0] as int);
        if (rgb == null) throw StateError('photo for job ${args[0]} is gone');
        out.send((id, _afterGate(rgb, args[1] as Float32List, args[2] as double?, s), null));
      }
    } catch (e) {
      out.send((id, null, '$e'));
    }
  }
}

/// Mask and gate; for a snake, the mask cleanup, the crop and the classifier input.
_AfterGate _afterGate(RgbImage rgb, Float32List probability, double? conf, PipelineSettings s) {
  final size = s.inputSize;
  final mask = unletterboxMask(probability, rgb.height, rgb.width, size, s.segmenterThreshold);
  var maskPixels = 0;
  for (final v in mask) {
    maskPixels += v;
  }
  final area = maskPixels / mask.length;
  final snake = conf != null && conf >= s.gateBoxConf && area > 0 && area >= s.gateMaskArea;
  if (!snake) return _AfterGate(snake: false, area: area);
  final cleaned = cleanMask(mask, rgb.width, rgb.height,
      connectivity: s.connectivity,
      minLargestShare: s.minLargestShare,
      keepShareOfLargest: s.keepShareOfLargest,
      keepShareOfPhoto: s.keepShareOfPhoto);
  final window = cropWindow(cleaned, rgb.height, rgb.width, s.cropPadding);
  final crop = window == null ? rgb : rgb.crop(window.$1, window.$2, window.$3, window.$4);
  final clsIn = normalise(letterbox(crop, size, s.padRgb), s.mean, s.std, s.scale);
  var sum = 0.0, sq = 0.0;
  for (final v in clsIn) {
    sum += v;
    sq += v * v;
  }
  final mean = sum / clsIn.length;
  final variance = (sq / clsIn.length - mean * mean).clamp(0, double.infinity).toDouble();
  return _AfterGate(
    snake: true,
    area: area,
    window: window,
    classifierInput: clsIn,
    mean: mean,
    std: math.sqrt(variance),
  );
}
