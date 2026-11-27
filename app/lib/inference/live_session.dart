import 'dart:async';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/foundation.dart' show visibleForTesting;
import 'package:flutter/services.dart';

import '../models/verdict.dart';
import '../services/classifier.dart';
import 'bundle.dart';
import 'decision.dart';
import 'detections.dart';
import 'image_ops.dart';
import 'pipeline.dart';
import 'video_lock.dart';

/// The live camera with two lanes, so the boxes keep up with the camera while
/// the verdict is worked out on an earlier frame:
///
/// * **Box lane**, as many frames as it can take (newest first, older ones
///   dropped): camera frame -> RGB -> exposure fix -> detector. Every snake
///   box is tracked and drawn.
/// * **Verdict lane**, whenever it is free, on the newest frame the box lane
///   finished: segmenter -> gate -> mask crop -> classifier -> DR0 -> the
///   video lock. The same steps and settings as a photo ([SnakePipeline.run]),
///   on the same pixels, with the box lane's detector output reused.
///
/// A frame whose detector score is under the gate's box cutoff is "no snake"
/// whatever the mask says (the gate needs both), so the verdict lane skips it:
/// the answer is the same, only sooner.
///
/// Lock steps are applied in frame order. A "no snake" frame from the box lane
/// that is newer than the frame in the verdict lane waits for it, so the lock
/// sees frames in the order the camera took them, as before.
class LiveSession {
  /// [present] turns an answer into what the screen shows
  /// ([OnnxSnakeClassifier.fromAnswer]).
  @visibleForTesting
  LiveSession.withParts({
    required this._settings,
    required this._rules,
    required this._present,
    this._detector = _detectorChannel,
    this._verdict = _verdictChannel,
    this.laneThreads = (1, 2),
  }) : _lock = VideoLock(rules: _rules);

  final PipelineSettings _settings;
  final DecisionRules _rules;
  final Identification Function(ModelAnswer, PipelineResult) _present;
  final MethodChannel _detector, _verdict;
  final VideoLock _lock;
  final BoxTracker _tracker = BoxTracker();

  /// Intra-op threads of the box lane and the verdict lane.
  final (int, int) laneThreads;

  /// Extra boxes (not the one the verdict is about) are drawn from this
  /// detector score, Ultralytics' default confidence.
  static const displayScore = 0.25;

  static const _detectorChannel = MethodChannel('serpentguard/live_detector');
  static const _verdictChannel = MethodChannel('serpentguard/live_verdict');

  /// The two-lane session, or null where the native lanes do not exist (not
  /// Android) or could not open; the live screen then uses
  /// [OnnxSnakeClassifier.identifyFrame].
  static Future<LiveSession?> open(OnnxSnakeClassifier classifier) async {
    if (!Platform.isAndroid) return null;
    try {
      final pipeline = await classifier.pipeline;
      final cores = (await _detectorChannel.invokeMethod<int>('cores')) ?? 4;
      final boxThreads = cores >= 8 ? 2 : 1;
      final verdictThreads = (cores - boxThreads - 1).clamp(2, 4);
      await _detectorChannel.invokeMethod('open', {
        'path': await SnakePipeline.extractModel('detector.onnx'),
        'threads': boxThreads,
      });
      await _verdictChannel.invokeMethod('open', {
        'segmenter': await SnakePipeline.extractModel('segmenter.onnx'),
        'classifier': await SnakePipeline.extractModel('classifier.onnx'),
        'threads': verdictThreads,
      });
      return LiveSession.withParts(
        settings: pipeline.settings,
        rules: pipeline.rules,
        present: (a, details) => classifier.fromAnswer(a, details: details),
        laneThreads: (boxThreads, verdictThreads),
      );
    } catch (_) {
      return null;
    }
  }

  final _views = StreamController<LiveView>.broadcast();

  /// A new view after every box-lane frame and every verdict.
  Stream<LiveView> get views => _views.stream;

  LiveView? _view;
  LiveView? get view => _view;

  // Scheduling.
  LiveCameraFrame? _waiting;
  bool _boxBusy = false, _verdictBusy = false, _closed = false;
  int _session = 0;
  double? _verdictFrameSeconds;
  final List<double> _heldNoSnake = [];

  // What the screen shows.
  Identification? _shown;
  bool _hasShown = false;
  PipelineResult? _lastVerdictFrame;
  int _width = 0, _height = 0;

  // Diagnostics.
  final LaneTimer boxTimer = LaneTimer(), verdictTimer = LaneTimer();
  int _skippedByGate = 0;

  PipelineSettings get _s => _settings;

  /// Starts a new encounter: clears the lock, the tracks and the answer.
  void reset() {
    _session++;
    _waiting = null;
    _heldNoSnake.clear();
    _verdictFrameSeconds = null;
    _lock.reset();
    _tracker.reset();
    _shown = null;
    _hasShown = false;
    _lastVerdictFrame = null;
    _skippedByGate = 0;
    boxTimer.clear();
    verdictTimer.clear();
    _view = null;
  }

  /// A camera frame. Returns at once; the frame replaces any frame still
  /// waiting for the box lane.
  void offer(LiveCameraFrame frame) {
    if (_closed) return;
    if (_boxBusy) {
      _waiting = frame;
      return;
    }
    unawaited(_boxLane(frame));
  }

  Future<void> _boxLane(LiveCameraFrame first) async {
    _boxBusy = true;
    LiveCameraFrame? frame = first;
    try {
      while (frame != null && !_closed) {
        await _boxFrame(frame);
        frame = _waiting;
        _waiting = null;
      }
    } finally {
      _boxBusy = false;
    }
  }

  Future<void> _boxFrame(LiveCameraFrame frame) async {
    final session = _session;
    final clock = Stopwatch()..start();
    final seconds = frame.offsetMs / 1000;
    try {
      // The same frame the old path gave the model: upright RGB, long side
      // at most 640 (OnnxSnakeClassifier.runFrame).
      var rgb = cameraFrameToRgb(frame, maxSide: OnnxSnakeClassifier.frameMaxSide);
      if (rgb == null) throw UnsupportedError('camera format ${frame.format}');
      if (math.max(rgb.width, rgb.height) > OnnxSnakeClassifier.frameMaxSide) {
        rgb = OnnxSnakeClassifier.phoneSize(rgb);
      }
      final s = _s;
      final fixed = fixExposure(rgb, s.exposureBelow, s.exposureAbove, s.claheClip, s.claheTiles);
      final input = detectorInput(fixed, s.inputSize, s.detectorPad);
      final output = (await _detector.invokeMethod<Float32List>('run', {'input': input, 'size': s.inputSize}))!;
      if (session != _session || _closed) return;

      final anchors = output.length ~/ 5; // rows cx, cy, w, h and the snake score
      final conf = boxConf(output, anchors, s.snakeClass, s.boxFloor);
      final best = conf == null ? null : bestBox(output, anchors, s.snakeClass, fixed.width, fixed.height, s.inputSize);
      final found = detectAll(output, anchors, s.snakeClass, fixed.width, fixed.height, s.inputSize,
          minScore: displayScore);
      if (best != null && conf! >= s.gateBoxConf && (found.isEmpty || iou(found.first.box, best) < 0.5)) {
        found.insert(0, Detection(best, conf));
      }
      _tracker.update(found, seconds);
      _width = fixed.width;
      _height = fixed.height;
      boxTimer.add(clock.elapsedMilliseconds, seconds);

      final gateCanPass = conf != null && conf >= s.gateBoxConf;
      if (!gateCanPass) {
        _skippedByGate++;
        _noSnake(seconds, fixed, conf, rgb);
      } else if (!_verdictBusy) {
        unawaited(_verdictLane(fixed, !identical(fixed, rgb), conf, best, seconds, session));
      }
      _publish();
    } catch (e) {
      if (session == _session && !_closed) _views.addError(e);
    }
  }

  /// A frame the gate turns down on the detector score alone.
  void _noSnake(double seconds, RgbImage fixed, double? conf, RgbImage raw) {
    final verdictAt = _verdictFrameSeconds;
    if (verdictAt != null && seconds > verdictAt) {
      _heldNoSnake.add(seconds); // applied after the verdict lane's frame
      return;
    }
    _applyNoSnake(PipelineResult(
      width: fixed.width,
      height: fixed.height,
      exposureFixed: !identical(fixed, raw),
      gate: GateResult(snake: false, boxConf: conf, maskArea: 0),
      answer: ModelAnswer.noSnake,
    ), seconds);
  }

  void _applyNoSnake(PipelineResult details, double seconds) {
    final step = _lock.add(null, seconds);
    _show(step, details, snake: false);
  }

  Future<void> _verdictLane(
      RgbImage rgb, bool exposureFixed, double? conf, PixelBox? box, double seconds, int session) async {
    _verdictBusy = true;
    _verdictFrameSeconds = seconds;
    final clock = Stopwatch()..start();
    final times = <String, int>{};
    int lap() {
      final ms = clock.elapsedMilliseconds;
      clock.reset();
      return ms;
    }

    try {
      final s = _s;
      final size = s.inputSize;
      final segIn = normalise(letterbox(rgb, size, s.padRgb), s.mean, s.std, s.scale);
      final probability =
          (await _verdict.invokeMethod<Float32List>('segment', {'input': segIn, 'size': size}))!;
      times['segmenter'] = lap();
      if (session != _session || _closed) return;
      final post = afterGate(rgb, probability, conf, s);
      times['crop'] = lap();
      final gate = GateResult(snake: post.snake, boxConf: conf, maskArea: post.area, box: box);
      PipelineResult result;
      if (!post.snake) {
        result = PipelineResult(
            width: rgb.width, height: rgb.height, exposureFixed: exposureFixed, gate: gate,
            answer: ModelAnswer.noSnake, timingsMs: times);
      } else {
        final out = (await _verdict.invokeMapMethod<String, Float32List>(
            'classify', {'input': post.classifierInput!, 'size': size}))!;
        times['classifier'] = lap();
        if (session != _session || _closed) return;
        final outputs = ModelOutputs(
          embedding: out['embedding']!,
          speciesLogits: out['species_logits']!,
          whoLogits: out['who_logits']!,
          genusLogits: out['genus_logits']!,
          familyLogits: out['family_logits']!,
        );
        final parts = DecisionParts.fromOutputs(outputs, s.thresholds);
        result = PipelineResult(
          width: rgb.width,
          height: rgb.height,
          exposureFixed: exposureFixed,
          gate: gate,
          cropWindow: post.window,
          answer: _rules.verdict(parts),
          parts: parts,
          classifierInputMean: post.mean,
          classifierInputStd: post.std,
          timingsMs: times,
        );
      }
      _lastVerdictFrame = result;
      final step = _lock.add(result.parts, seconds);
      _show(step, result, snake: result.gate.snake);
      verdictTimer.add(times.values.fold(0, (a, b) => a + b), seconds);
    } catch (e) {
      if (session == _session && !_closed) _views.addError(e);
    } finally {
      // Always free the lane, also when a reset ended this frame's session
      // (the new session cannot have started a verdict meanwhile).
      _verdictBusy = false;
      if (session == _session) {
        _verdictFrameSeconds = null;
        final held = List.of(_heldNoSnake);
        _heldNoSnake.clear();
        for (final t in held) {
          _applyNoSnake(
              PipelineResult(
                  width: _width, height: _height, exposureFixed: false,
                  gate: const GateResult(snake: false, boxConf: null, maskArea: 0), answer: ModelAnswer.noSnake),
              t);
        }
        _publish();
      }
    }
  }

  /// What [OnnxSnakeClassifier.identifyFrame] returns for this lock step.
  void _show(LockStep step, PipelineResult details, {required bool snake}) {
    if (!snake && step.pooled == null) {
      _shown = _present(ModelAnswer.noSnake, details);
    } else {
      final shown = step.shown;
      _shown = shown == null ? null : _present(shown, details);
    }
    _hasShown = true;
  }

  void _publish() {
    final tracks = _tracker.tracks;
    final verdictBox = _lastVerdictFrame?.gate.snake == true ? _lastVerdictFrame!.gate.box : null;
    Track? main;
    if (verdictBox != null && tracks.isNotEmpty) {
      var bestIou = 0.1;
      for (final t in tracks) {
        final o = iou(t.box, verdictBox);
        if (o > bestIou) {
          bestIou = o;
          main = t;
        }
      }
    }
    final shown = [
      for (final t in tracks)
        if (t.score >= displayScore || identical(t, main)) t,
    ];
    final view = LiveView(
      width: _width,
      height: _height,
      tracks: shown,
      mainTrack: main?.id,
      identification: _shown,
      answered: _hasShown,
      snakeInView: _lastVerdictFrame?.gate.snake == true && main != null,
      boxFps: boxTimer.fps,
      verdictFps: verdictTimer.fps,
      boxMs: boxTimer.meanMs,
      verdictMs: verdictTimer.meanMs,
      skippedByGate: _skippedByGate,
      threads: laneThreads,
    );
    _view = view;
    if (!_views.isClosed) _views.add(view);
  }

  Future<void> close() async {
    _closed = true;
    _session++;
    await _views.close();
    try {
      await _detector.invokeMethod('close');
      await _verdict.invokeMethod('close');
    } catch (_) {}
  }
}

/// What the live screen draws after one lane step.
class LiveView {
  const LiveView({
    required this.width,
    required this.height,
    required this.tracks,
    required this.mainTrack,
    required this.identification,
    required this.answered,
    required this.snakeInView,
    required this.boxFps,
    required this.verdictFps,
    required this.boxMs,
    required this.verdictMs,
    required this.skippedByGate,
    required this.threads,
  });

  /// The size of the frames the boxes are in (upright, long side <= 640).
  final int width, height;

  /// The boxes to draw, most confident first.
  final List<Track> tracks;

  /// The track the verdict is about, if the last verdict frame had a snake.
  final int? mainTrack;

  /// What [OnnxSnakeClassifier.identifyFrame] would show: null while
  /// checking (once [answered]); a no-snake answer when nothing is in view.
  final Identification? identification;
  final bool answered;

  /// The verdict lane's last frame passed the gate and its box is tracked.
  final bool snakeInView;

  final double boxFps, verdictFps, boxMs, verdictMs;
  final int skippedByGate;
  final (int, int) threads;
}

/// Rate and mean time of one lane over its last few seconds.
class LaneTimer {
  final List<(double, int)> _recent = [];

  void add(int ms, double seconds) {
    _recent.add((seconds, ms));
    _recent.removeWhere((e) => seconds - e.$1 > 3);
  }

  void clear() => _recent.clear();

  double get fps {
    if (_recent.length < 2) return 0;
    final span = _recent.last.$1 - _recent.first.$1;
    return span <= 0 ? 0 : (_recent.length - 1) / span;
  }

  double get meanMs => _recent.isEmpty ? 0 : _recent.map((e) => e.$2).reduce((a, b) => a + b) / _recent.length;
}
