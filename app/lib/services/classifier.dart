import 'dart:isolate';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:crypto/crypto.dart';

import '../inference/decision.dart';
import '../inference/image_ops.dart';
import '../inference/pipeline.dart';
import '../inference/video_lock.dart';
import '../models/species.dart';
import '../models/verdict.dart';

/// Turns a photo into species probabilities and a verdict.
///
/// [OnnxSnakeClassifier] is the real model; [DemoClassifier] stands in only
/// when the model files are not in the build.
abstract class SnakeClassifier {
  bool get isDemo;
  Future<Identification> identify(Uint8List imageBytes);
}

/// Optional adapter for continuous camera analysis. Implement this together
/// with photo identify() when the verified mobile models are connected.
/// The adapter owns preprocessing, rotation and the shipped temporal rules.
abstract class LiveSnakeClassifier implements SnakeClassifier {
  Future<void> resetLiveSession();

  /// Null means temporal warm-up/no stable display yet. A non-snake decision
  /// must set snakeFound=false, rather than invent species probabilities.
  Future<Identification?> identifyFrame(LiveCameraFrame frame);
}

class LiveCameraPlane {
  const LiveCameraPlane(this.bytes, this.bytesPerRow, this.bytesPerPixel);
  final Uint8List bytes;
  final int bytesPerRow;
  final int? bytesPerPixel;
}

class LiveCameraFrame {
  const LiveCameraFrame({
    required this.width,
    required this.height,
    required this.format,
    required this.planes,
    required this.offsetMs,
    required this.sensorOrientation,
    required this.deviceOrientation,
    required this.frontFacing,
  });
  final int width, height, offsetMs, sensorOrientation;
  final String format, deviceOrientation;
  final bool frontFacing;
  final List<LiveCameraPlane> planes;
}

/// Simulated results, used only when the model files are not in the build.
/// The same photo always gives the same answer (seeded by the photo's hash).
/// It does not look at the image content, and the screen says so. The
/// verdict still comes from the real A3 decision rules.
class DemoClassifier implements SnakeClassifier {
  DemoClassifier(this.catalog, this.rules) : _rule = DecisionRule(catalog: catalog, rules: rules);

  final SpeciesCatalog catalog;
  final DecisionRules rules;
  final DecisionRule _rule;

  @override
  bool get isDemo => true;

  @override
  Future<Identification> identify(Uint8List imageBytes) async {
    final seed = sha256
        .convert(imageBytes)
        .bytes
        .take(4)
        .fold<int>(0, (a, b) => (a << 8) | b);
    await Future<void>.delayed(
      const Duration(milliseconds: 600),
    ); // feels like inference
    return _rule.decide(simulatedParts(seed), isDemo: true);
  }

  /// Also used by the "try a species" picker on the result screen.
  Identification forSpecies(int speciesId, {double confidence = 0.7}) {
    final n = catalog.all.length;
    final rest = (1 - confidence) / (n - 1);
    final probs = List<double>.filled(n, rest);
    probs[speciesId] = confidence;
    return _rule.decide(_parts(probs), isDemo: true);
  }

  DecisionParts simulatedParts(int seed) {
    final rnd = math.Random(seed);
    final n = catalog.all.length;
    // A peaked softmax over random logits, with the peak on 1 to 3 species.
    final logits = List<double>.generate(n, (_) => rnd.nextDouble());
    final peaks = 1 + rnd.nextInt(3);
    for (var i = 0; i < peaks; i++) {
      logits[rnd.nextInt(n)] += 4 + rnd.nextDouble() * 3;
    }
    return _parts(softmax(logits));
  }

  /// Heads consistent with the species probabilities: WHO and genus heads
  /// are the species mass per category, the family head points at a
  /// non-venomous family, and the embedding is the top species' prototype.
  DecisionParts _parts(List<double> probs) {
    final t = rules.tables;
    final who = Float64List(3), genus = Float64List(t.numGenera);
    for (var k = 0; k < probs.length; k++) {
      who[t.who[k]] += probs[k];
      genus[t.genusOfClass[k]] += probs[k];
    }
    final family = Float64List(t.venomousFamily.length);
    family[t.venomousFamily.indexOf(false)] = 1;
    final top = argmax(probs);
    return DecisionParts(
      probs: Float64List.fromList(probs),
      who: who,
      genus: genus,
      family: family,
      embedding: t.unitPrototypes[top],
    );
  }
}

/// The shipped model: A3 behind the M4 front end (detector + U-Net gate,
/// mask crop), fp32 ONNX, with the reference pipeline's decision rules.
/// Photos go through [identify]; the live camera and video files go through
/// the M4 video lock.
class OnnxSnakeClassifier implements LiveSnakeClassifier {
  /// [pipeline] may still be opening (copying the models out on first
  /// launch takes a few seconds); every call waits for it.
  OnnxSnakeClassifier(this.catalog, DecisionRules rules, this.pipeline)
      : _rule = DecisionRule(catalog: catalog, rules: rules),
        _lock = VideoLock(rules: rules),
        _rules = rules;

  final SpeciesCatalog catalog;
  final Future<SnakePipeline> pipeline;
  final DecisionRule _rule;
  final DecisionRules _rules;
  final VideoLock _lock;

  /// Phone video frames are scaled to this long side, as in M4's clips.
  static const frameMaxSide = 640;

  @override
  bool get isDemo => false;

  /// The opened models, or a clear error if they could not be opened (for
  /// example, no storage left to copy them out on first launch).
  Future<SnakePipeline> get _ready async {
    try {
      return await pipeline;
    } catch (e) {
      throw ModelUnavailable('$e');
    }
  }

  @override
  Future<Identification> identify(Uint8List imageBytes) async {
    final model = await _ready;
    return _identify(model, await SnakePipeline.decode(imageBytes));
  }

  Future<Identification> identifyImage(RgbImage image) async => _identify(await _ready, image);

  Future<Identification> _identify(SnakePipeline model, RgbImage image) async {
    final r = await model.run(image);
    return _rule.fromAnswer(r.answer, details: r);
  }

  /// One frame through the pipeline, without the video lock.
  Future<PipelineResult> runFrame(RgbImage frame) async {
    final model = await _ready;
    final sized = math.max(frame.width, frame.height) > frameMaxSide
        ? await Isolate.run(() => _phoneSize(frame))
        : frame;
    return model.run(sized);
  }

  static RgbImage _phoneSize(RgbImage img) {
    final long = math.max(img.width, img.height);
    if (long <= frameMaxSide) return img;
    final scale = frameMaxSide / long;
    return resizeArea(img, math.max(1, (img.width * scale).round()), math.max(1, (img.height * scale).round()));
  }

  /// The last live frame checked (its box is drawn even while the answer is
  /// still settling).
  PipelineResult? lastFrame;

  /// Bumped on every reset, so a frame still being checked when the live
  /// screen closes cannot land in the next session's lock.
  int _session = 0;

  @override
  Future<void> resetLiveSession() async {
    _session++;
    lastFrame = null;
    _lock.reset();
  }

  @override
  Future<Identification?> identifyFrame(LiveCameraFrame frame) async {
    final session = _session;
    // On this thread: an isolate per frame slowed the models (see SnakePipeline.run).
    final rgb = cameraFrameToRgb(frame, maxSide: frameMaxSide);
    if (rgb == null) throw UnsupportedError('camera format ${frame.format}');
    final r = await runFrame(rgb);
    if (session != _session) return null;
    lastFrame = r;
    final step = _lock.add(r.parts, frame.offsetMs / 1000);
    if (!r.gate.snake && step.pooled == null) {
      return _rule.fromAnswer(ModelAnswer.noSnake, details: r);
    }
    final shown = step.shown;
    return shown == null ? null : _rule.fromAnswer(shown, details: r);
  }

  /// A new video lock for analysing a clip.
  VideoLock newLock() => VideoLock(rules: _rules);

  Identification fromAnswer(ModelAnswer a, {PipelineResult? details}) =>
      _rule.fromAnswer(a, details: details);
}

/// The models could not be opened, so nothing can be identified.
class ModelUnavailable implements Exception {
  ModelUnavailable(this.detail);
  final String detail;

  @override
  String toString() => 'The identification model could not start on this phone. '
      'Keep away from the snake and use SOS if anyone was bitten. ($detail)';
}

/// Converts a camera frame (YUV 4:2:0 in three planes, NV21, or BGRA) to
/// upright RGB, using full-range BT.601 as Android cameras deliver. Large
/// frames are averaged down in whole blocks so the long side is about
/// [maxSide] (the size the model is given anyway), which keeps this fast.
RgbImage? cameraFrameToRgb(LiveCameraFrame f, {int maxSide = 640}) {
  final step = math.max(1, (math.max(f.width, f.height) / maxSide).floor());
  final w = f.width ~/ step, h = f.height ~/ step;
  if (w == 0 || h == 0) return null;
  final out = Uint8List(w * h * 3);
  int clamp(double v) => v < 0 ? 0 : (v > 255 ? 255 : v.round());
  void put(int i, double y, int u, int v) {
    final cu = u - 128, cv = v - 128;
    out[i * 3] = clamp(y + 1.402 * cv);
    out[i * 3 + 1] = clamp(y - 0.344136 * cu - 0.714136 * cv);
    out[i * 3 + 2] = clamp(y + 1.772 * cu);
  }

  final area = step * step;
  if (f.format == 'yuv420' && f.planes.length >= 3) {
    final yp = f.planes[0], up = f.planes[1], vp = f.planes[2];
    final uvPixel = up.bytesPerPixel ?? 1;
    for (var y = 0; y < h; y++) {
      for (var x = 0; x < w; x++) {
        final sy = y * step, sx = x * step;
        var ySum = 0;
        for (var j = 0; j < step; j++) {
          final row = (sy + j) * yp.bytesPerRow + sx;
          for (var i = 0; i < step; i++) {
            ySum += yp.bytes[row + i];
          }
        }
        final cy = sy + step ~/ 2, cx = sx + step ~/ 2;
        final uv = (cy >> 1) * up.bytesPerRow + (cx >> 1) * uvPixel;
        put(y * w + x, ySum / area, up.bytes[uv], vp.bytes[uv]);
      }
    }
  } else if (f.format == 'nv21' && f.planes.isNotEmpty) {
    final p = f.planes[0];
    final stride = p.bytesPerRow;
    final uvStart = stride * f.height;
    for (var y = 0; y < h; y++) {
      for (var x = 0; x < w; x++) {
        final sy = y * step, sx = x * step;
        var ySum = 0;
        for (var j = 0; j < step; j++) {
          for (var i = 0; i < step; i++) {
            ySum += p.bytes[(sy + j) * stride + sx + i];
          }
        }
        final cy = sy + step ~/ 2, cx = sx + step ~/ 2;
        final uv = uvStart + (cy >> 1) * stride + (cx & ~1);
        put(y * w + x, ySum / area, p.bytes[uv + 1], p.bytes[uv]);
      }
    }
  } else if (f.format == 'bgra8888' && f.planes.isNotEmpty) {
    final p = f.planes[0];
    for (var y = 0; y < h; y++) {
      for (var x = 0; x < w; x++) {
        final s = y * step * p.bytesPerRow + x * step * 4, d = (y * w + x) * 3;
        out[d] = p.bytes[s + 2];
        out[d + 1] = p.bytes[s + 1];
        out[d + 2] = p.bytes[s];
      }
    }
  } else {
    return null;
  }
  var img = RgbImage(w, h, out);
  // Turn the sensor image upright for the way the phone is held.
  final device = switch (f.deviceOrientation) {
    'landscapeLeft' => 90,
    'portraitDown' => 180,
    'landscapeRight' => 270,
    _ => 0,
  };
  final rotation = f.frontFacing
      ? (f.sensorOrientation + device) % 360
      : (f.sensorOrientation - device + 360) % 360;
  img = img.rotate(rotation);
  return f.frontFacing ? img.flipHorizontal() : img;
}
