import 'dart:async';
import 'dart:io';
import 'dart:isolate';
import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:crypto/crypto.dart' as crypto;
import 'package:flutter/foundation.dart' show visibleForTesting;
import 'package:flutter/services.dart' show AssetManifest, MethodChannel, rootBundle;
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'package:path_provider/path_provider.dart';

import 'bundle.dart';
import 'decision.dart';
import 'flip_prior.dart';
import 'image_ops.dart';

/// Where the models live in the app bundle. They are not in git (65 MB);
/// `tool/m5/build_assets.py` copies them in (see assets/models/README.md).
const modelAssetDir = 'assets/models';
const classifierFile = 'classifier.onnx', detectorFile = 'detector.onnx', segmenterFile = 'segmenter.onnx';
const snakeCheckFile = 'snake_check_fp16.onnx';
const modelFiles = [classifierFile, detectorFile, segmenterFile, snakeCheckFile];
const _inputName = 'images';
const _detectorOutput = 'output0', _segmenterOutput = 'probability';
const _detectorRows = 5; // cx, cy, w, h and the snake score

/// What the no-snake decision saw on one image. With the M5 snake check,
/// [snake] is the check's call and [snakeProbability] its P(snake); the
/// detector and U-Net then only find the crop and the box, and are not run
/// at all when the check says no snake ([boxConf] null, [maskArea] 0).
/// A bundled model file whose contents are not the ones `pipeline.json`
/// names: an old or wrong model, which must not run with these settings.
class ModelMismatch implements Exception {
  const ModelMismatch(this.file, {required this.expected, required this.actual});
  final String file, expected, actual;

  @override
  String toString() => '$file is not the model pipeline.json names (SHA-256 $actual, expected $expected). '
      'Copy the right files into assets/models (see its README).';
}

class GateResult {
  const GateResult(
      {required this.snake, required this.boxConf, required this.maskArea, this.snakeProbability, this.box});
  final bool snake;
  final double? boxConf;
  final double maskArea;
  final double? snakeProbability;

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
    this.classifierOutputs,
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

  /// The classifier answer the rules read (mirror-averaged, prior added).
  final ModelOutputs? classifierOutputs;
  final double? classifierInputMean, classifierInputStd;
  final Map<String, int> timingsMs;

  int get totalMs => timingsMs.values.fold(0, (a, b) => a + b);
}
/// The snake check, detector, segmenter and classifier with the settings
/// from `pipeline.json` and `tables.json`: a port of `tool/m5/reference_m5.py`
/// (the M4 `reference_pipeline.py` plus the frozen M5 changes).
class SnakePipeline {
  SnakePipeline._(this.settings, this.tables, this._detector, this._segmenter, this._classifier, this._snakeCheck, this._worker)
      : rules = DecisionRules(settings.thresholds, tables),
        _classifierSettings = settings.classifier,
        _check = settings.snakeCheck {
    if (settings.usesSnakeCheck && _check == null) {
      throw StateError('pipeline.json: the gate rule is ${settings.gateRule} but there is no snake_check block');
    }
  }

  final _PixelWorker _worker;

  /// Up to this many pixels (a 640 x 480 frame) the pixel work runs on the
  /// calling isolate; above it, on the background worker.
  static const inlinePixelLimit = 640 * 480;

  RgbImage? _inlineRgb;
  _Prepared _prepareHere(RgbImage photo) {
    final rgb = fixExposure(photo, settings.exposureBelow, settings.exposureAbove, settings.claheClip, settings.claheTiles);
    _inlineRgb = rgb;
    return _prepared(-1, rgb, photo, settings);
  }


  final PipelineSettings settings;
  final ModelTables tables;
  final DecisionRules rules;
  final OrtSession _detector, _segmenter, _classifier, _snakeCheck;
  final ClassifierSettings _classifierSettings;
  final SnakeCheckSettings? _check;

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

  /// Opens the four ONNX sessions and the pixel worker. Throws if a model file is missing.
  static Future<SnakePipeline> load(PipelineSettings settings, ModelTables tables) async {
    final check = settings.snakeCheck;
    if (check != null && check.file != snakeCheckFile) {
      throw StateError('pipeline.json names ${check.file}, the app bundles $snakeCheckFile');
    }
    final ort = OnnxRuntime();
    final options = OrtSessionOptions(intraOpNumThreads: 2, interOpNumThreads: 1);
    Future<OrtSession> open(String name) async =>
        ort.createSession(
            await extractModel(name,
                sha256: settings.modelSha256[name], refused: settings.refusedModelSha256[name] ?? const {}),
            options: options);
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
    return SnakePipeline._(settings, tables, opened[1], opened[2], opened[0], opened[3], worker);
  }


  /// ONNX Runtime needs a file path. Copies the asset out, and again whenever
  /// its contents change. The copy's SHA-256 is kept beside it: two model
  /// builds can have the same size (A3 and M5_KD's classifier.onnx do), so
  /// a size check would keep the old model after an update.
  ///
  /// With [sha256] (from `pipeline.json` `model_sha256`), a bundled file with
  /// other contents is refused, so the wrong model never runs with these
  /// settings. The hash runs off the UI isolate.
  static Future<String> extractModel(String name, {String? sha256, Map<String, String> refused = const {}}) async {
    final data = await rootBundle.load('$modelAssetDir/$name');
    final dir = Directory('${(await getApplicationSupportDirectory()).path}/models');
    await dir.create(recursive: true);
    return copyModel(data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes), File('${dir.path}/$name'),
        sha256: sha256, refused: refused);
  }

  /// [extractModel]'s check and copy, for a model's [bytes] and its place on
  /// disk. Returns the file's path.
  @visibleForTesting
  static Future<String> copyModel(Uint8List bytes, File file,
      {String? sha256, Map<String, String> refused = const {}}) async {
    final name = file.uri.pathSegments.last;
    final digest = await Isolate.run(() => crypto.sha256.convert(bytes).toString());
    if (sha256 != null && digest != sha256) throw ModelMismatch(name, expected: sha256, actual: digest);
    final why = refused[digest];
    if (why != null) throw ModelMismatch(name, expected: 'not $digest ($why)', actual: digest);
    final stamp = File('${file.path}.sha256');
    final copied = await file.exists() && await stamp.exists() ? (await stamp.readAsString()).trim() : null;
    if (copied != digest) {
      if (await stamp.exists()) await stamp.delete();
      final tmp = File('${file.path}.part');
      await tmp.writeAsBytes(bytes, flush: true);
      await tmp.rename(file.path);
      await stamp.writeAsString(digest, flush: true);
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

  Future<List<double>> _run(OrtSession session, Float32List input, String output, [int? size]) async {
    final side = size ?? settings.inputSize;
    final value = await OrtValue.fromList(input, [1, 3, side, side]);
    try {
      final outputs = await session.run({_inputName: value});
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
      final outputs = await session.run({_inputName: value});
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
  /// The pixel work runs in a background isolate for big photos so the
  /// screen keeps drawing; the ONNX calls stay on this isolate (the plugin
  /// needs it).
  Future<PipelineResult> run(RgbImage photo) async {
    final s = settings;
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

    double? snakeProbability;
    if (s.usesSnakeCheck) {
      final check = _check!;
      final probs = await _run(_snakeCheck, pre.snakeCheckInput!, check.output, check.inputSize);
      snakeProbability = probs[check.snakeIndex];
      times['snake_check'] = lap();
      if (!check.isSnake(probs)) {
        if (!inline) _worker.drop(pre.job);
        return PipelineResult(
          width: pre.width,
          height: pre.height,
          exposureFixed: pre.fixed,
          gate: GateResult(snake: false, boxConf: null, maskArea: 0, snakeProbability: snakeProbability),
          answer: ModelAnswer.noSnake,
          timingsMs: times,
        );
      }
    }

    // Detector box score and U-Net mask: the crop and the box on screen, and
    // the M4 gate when there is no snake check.
    final detected = await _run(_detector, pre.detectorInput, _detectorOutput);
    final anchors = detected.length ~/ _detectorRows;
    final conf = boxConf(detected, anchors, s.snakeClass, s.boxFloor);
    final box = conf == null ? null : bestBox(detected, anchors, s.snakeClass, pre.width, pre.height, s.inputSize);
    times['detector'] = lap();

    final probability = Float32List.fromList(await _run(_segmenter, pre.segmenterInput, _segmenterOutput));
    times['segmenter'] = lap();
    final post = inline ? afterGate(_inlineRgb!, probability, conf, s) : await _worker.afterGate(pre.job, probability, conf);
    times['crop'] = lap();

    final gate = GateResult(
        snake: post.snake, boxConf: conf, maskArea: post.area, snakeProbability: snakeProbability, box: box);
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

    final plain = _outputs(await _runAll(_classifier, post.classifierInput!));
    final mirrored = post.mirroredInput == null ? null : _outputs(await _runAll(_classifier, post.mirroredInput!));
    final outputs = classifierAnswer(plain, mirrored, _classifierSettings);
    times['classifier'] = lap();

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
      classifierOutputs: outputs,
      classifierInputMean: post.mean,
      classifierInputStd: post.std,
      timingsMs: times,
    );
  }

  static ModelOutputs _outputs(Map<String, List<double>> out) => ModelOutputs(
        embedding: out['embedding']!,
        speciesLogits: out['species_logits']!,
        whoLogits: out['who_logits']!,
        genusLogits: out['genus_logits']!,
        familyLogits: out['family_logits']!,
      );

  Future<void> close() async {
    _worker.close();
    await _snakeCheck.close();
    await _detector.close();
    await _segmenter.close();
    await _classifier.close();
  }
}

// ------------------------------------------------- isolate steps (pure Dart)

/// What the first step sends back. The photo itself stays in the worker
/// (under [job]) for the second step.
class _Prepared {
  const _Prepared(this.job, this.width, this.height, this.fixed, this.detectorInput, this.segmenterInput,
      this.snakeCheckInput);
  final int job, width, height;
  final bool fixed;
  final Float32List detectorInput, segmenterInput;

  /// The snake check's input; null when the gate is the M4 detector + U-Net.
  final Float32List? snakeCheckInput;
}

/// The model inputs for one exposure-fixed photo.
_Prepared _prepared(int job, RgbImage rgb, RgbImage photo, PipelineSettings s) {
  final check = s.usesSnakeCheck ? s.snakeCheck : null;
  return _Prepared(
    job,
    rgb.width,
    rgb.height,
    !identical(rgb, photo),
    detectorInput(rgb, s.inputSize, s.detectorPad),
    normalise(letterbox(rgb, s.inputSize, s.padRgb), s.mean, s.std, s.scale),
    check == null ? null : normalise(letterbox(rgb, check.inputSize, s.padRgb), s.mean, s.std, s.scale),
  );
}

class AfterGate {
  const AfterGate(
      {required this.snake, required this.area, this.window, this.classifierInput, this.mirroredInput, this.mean, this.std});
  final bool snake;
  final double area;
  final (int, int, int, int)? window;
  final Float32List? classifierInput, mirroredInput;
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

  Future<AfterGate> afterGate(int job, Float32List probability, double? conf) =>
      _ask('after', [job, probability, conf]);

  /// Forgets a job whose second step will not come (the snake check said no snake).
  void drop(int job) => _commands.send((-1, 'drop', [job]));

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
        out.send((id, _prepared(job, rgb, photo, s), null));
      } else if (op == 'drop') {
        photos.remove(args[0] as int);
      } else {
        final rgb = photos.remove(args[0] as int);
        if (rgb == null) throw StateError('photo for job ${args[0]} is gone');
        out.send((id, afterGate(rgb, args[1] as Float32List, args[2] as double?, s), null));
      }
    } catch (e) {
      if (id >= 0) out.send((id, null, '$e'));
    }
  }
}

/// Mask and gate; for a snake, the mask cleanup, the crop and the classifier
/// inputs (the crop and, with the mirror pass, its mirror image). With the
/// snake check the photo already passed, so the mask only finds the crop.
AfterGate afterGate(RgbImage rgb, Float32List probability, double? conf, PipelineSettings s) {
  final size = s.inputSize;
  final mask = unletterboxMask(probability, rgb.height, rgb.width, size, s.segmenterThreshold);
  var maskPixels = 0;
  for (final v in mask) {
    maskPixels += v;
  }
  final area = maskPixels / mask.length;
  final snake = s.usesSnakeCheck || (conf != null && conf >= s.gateBoxConf && area > 0 && area >= s.gateMaskArea);
  if (!snake) return AfterGate(snake: false, area: area);
  final cleaned = cleanMask(mask, rgb.width, rgb.height,
      connectivity: s.connectivity,
      minLargestShare: s.minLargestShare,
      keepShareOfLargest: s.keepShareOfLargest,
      keepShareOfPhoto: s.keepShareOfPhoto);
  final window = cropWindow(cleaned, rgb.height, rgb.width, s.cropPadding);
  final crop = window == null ? rgb : rgb.crop(window.$1, window.$2, window.$3, window.$4);
  final clsIn = normalise(letterbox(crop, size, s.padRgb), s.mean, s.std, s.scale);
  final mirrored =
      s.classifier.flipTta ? normalise(letterbox(crop.mirrored(), size, s.padRgb), s.mean, s.std, s.scale) : null;
  var sum = 0.0, sq = 0.0;
  for (final v in clsIn) {
    sum += v;
    sq += v * v;
  }
  final mean = sum / clsIn.length;
  final variance = (sq / clsIn.length - mean * mean).clamp(0, double.infinity).toDouble();
  return AfterGate(
    snake: true,
    area: area,
    window: window,
    classifierInput: clsIn,
    mirroredInput: mirrored,
    mean: mean,
    std: math.sqrt(variance),
  );
}
