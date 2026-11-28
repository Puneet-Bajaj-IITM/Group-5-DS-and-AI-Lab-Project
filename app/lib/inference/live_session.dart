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
import 'flip_prior.dart';
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
///   finished: the M5 snake check (when `pipeline.json` uses it) ->
///   segmenter -> mask crop -> classifier (with the mirror pass and the prior
///   when the settings ask) -> DR0 -> the video lock. The same steps and
///   settings as a photo ([SnakePipeline.run]), on the same pixels, with the
///   box lane's detector output reused.
///
/// With the M4 gate (detector box and U-Net mask, no snake check), a frame
/// whose detector score is under the gate's box cutoff is "no snake" whatever
/// the mask says (the gate needs both), so the verdict lane skips it: the
/// answer is the same, only sooner. With the snake check the detector does not
/// decide, so every frame the verdict lane takes goes through the check.
///
/// The lock sees frames in camera order, at the rate the verdict lane can
/// take them: a frame the M4 gate turns down on its box score reaches the lock
/// only while the verdict lane is free, as a frame for the lane would.
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
    this.owner = 0,
  }) : _lock = VideoLock(rules: _rules);

  final PipelineSettings _settings;
  final DecisionRules _rules;
  final Identification Function(ModelAnswer, PipelineResult) _present;
  final MethodChannel _detector, _verdict;
  final VideoLock _lock;
  final BoxTracker _tracker = BoxTracker();

  /// Intra-op threads of the box lane and the verdict lane.
  final (int, int) laneThreads;

  /// Which open of the app-wide native lanes this session owns. A close from
  /// an older session (a screen that closed while a newer one opened) is
  /// ignored on the native side, so it cannot close the newer session's models.
  final int owner;
  static int _lastOwner = 0;

  static const _detectorChannel = MethodChannel('serpentguard/live_detector');
  static const _verdictChannel = MethodChannel('serpentguard/live_verdict');

  /// The two-lane session, or null where the native lanes do not exist (not
  /// Android) or could not open; the live screen then uses
  /// [OnnxSnakeClassifier.identifyFrame].
  static Future<LiveSession?> open(OnnxSnakeClassifier classifier) async {
    if (!Platform.isAndroid) return null;
    final owner = ++_lastOwner;
    try {
      final pipeline = await classifier.pipeline;
      // The same files, checked the same way, as the photo path.
      final s = pipeline.settings;
      Future<String> model(String f) =>
          SnakePipeline.extractModel(f, sha256: s.modelSha256[f], refused: s.refusedModelSha256[f] ?? const {});
      final cores = (await _detectorChannel.invokeMethod<int>('cores')) ?? 4;
      final boxThreads = cores >= 8 ? 2 : 1;
      final verdictThreads = (cores - boxThreads - 1).clamp(2, 4);
      await _detectorChannel.invokeMethod('open', {
        'owner': owner,
        'path': await model(detectorFile),
        'threads': boxThreads,
      });
      await _verdictChannel.invokeMethod('open', {
        'owner': owner,
        'segmenter': await model(segmenterFile),
        'classifier': await model(classifierFile),
        if (pipeline.settings.usesSnakeCheck) 'snakeCheck': await model(snakeCheckFile),
        'threads': verdictThreads,
      });
      return LiveSession.withParts(
        settings: pipeline.settings,
        rules: pipeline.rules,
        present: (a, details) => classifier.fromAnswer(a, details: details),
        laneThreads: (boxThreads, verdictThreads),
        owner: owner,
      );
    } catch (_) {
      // A lane that opened stays loaded otherwise; the screen falls back to
      // the photo path's models.
      for (final channel in [_detectorChannel, _verdictChannel]) {
        try {
          await channel.invokeMethod('close', {'owner': owner});
        } catch (_) {}
      }
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

  // The last failure of each lane, cleared by its next good frame.
  Object? _boxError, _verdictError;

  // What the screen shows.
  Identification? _shown;
  bool _hasShown = false;
  PipelineResult? _lastVerdictFrame;
  int _width = 0, _height = 0;

  // With several snakes in view, each tracked snake is named on its own: its
  // own box crop, snake check and lock, the verdict lane taking turns.
  final Map<int, VideoLock> _trackLocks = {};
  final Map<int, Identification> _trackNames = {};
  final Map<int, double> _trackNamedAt = {};

  // Diagnostics.
  final LaneTimer boxTimer = LaneTimer(), verdictTimer = LaneTimer();
  int _skippedByGate = 0;

  PipelineSettings get _s => _settings;

  /// Starts a new encounter: clears the lock, the tracks and the answer.
  void reset() {
    _session++;
    _waiting = null;
    _boxError = _verdictError = null;
    _lock.reset();
    _tracker.reset();
    _shown = null;
    _hasShown = false;
    _lastVerdictFrame = null;
    _clearTracks();
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
      // Parked until the box lane is free: keep its own copy of the pixels,
      // in case the camera reuses the buffers meanwhile.
      _waiting = _copied(frame);
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
      final found = snakeBoxes(output, anchors, s.snakeClass, fixed.width, fixed.height, s.inputSize);
      // The box the verdict is about is the most confident one, as on a
      // photo, so it is tracked even under the display score (it is drawn
      // only when the verdict lane names it). With the M4 gate only boxes
      // that can pass the gate qualify.
      final boxDecides = !s.usesSnakeCheck;
      if (best != null &&
          best.width > 0 &&
          best.height > 0 &&
          (!boxDecides || conf! >= s.gateBoxConf) &&
          (found.isEmpty || iou(found.first.box, best) < nmsIou)) {
        found.insert(0, Detection(best, conf!));
      }
      _tracker.update(found, seconds);
      _width = fixed.width;
      _height = fixed.height;
      boxTimer.add(clock.elapsedMilliseconds, seconds);

      final gateCanPass = !boxDecides || (conf != null && conf >= s.gateBoxConf);
      if (!gateCanPass) {
        // The M4 gate answers on the box score alone. Only frames that come
        // while the verdict lane is free reach the lock, as with the frames
        // the lane takes, so snake and no-snake frames are sampled at the same
        // rate and arrive in camera order.
        if (!_verdictBusy) {
          _skippedByGate++;
          _noSnake(seconds, fixed, conf, rgb);
        }
      } else if (!_verdictBusy) {
        final several = _namedTracks();
        if (several.length >= 2) {
          unawaited(_trackLane(fixed, !identical(fixed, rgb), several, seconds, session));
        } else {
          if (_trackNames.isNotEmpty) _clearTracks(); // back to one snake: the frozen path
          unawaited(_verdictLane(fixed, !identical(fixed, rgb), conf, best, seconds, session));
        }
      }
      _boxError = null;
      _publish();
    } catch (e) {
      if (session == _session && !_closed) {
        _boxError = e;
        _failed();
      }
    }
  }

  /// A lane failed: report it and stop showing the last verdict, which may no
  /// longer be about what the camera sees.
  /// The view carries the error ([LiveView.error]) until that lane works
  /// again, so a view published right after cannot hide it.
  void _failed() {
    _tracker.reset(); // no frozen boxes while a lane fails
    _shown = null;
    _hasShown = false;
    _lastVerdictFrame = null;
    _publish();
  }

  /// A frame the gate turns down on the detector score alone.
  void _noSnake(double seconds, RgbImage fixed, double? conf, RgbImage raw) {
    _verdictError = null; // answered without the verdict lane
    final details = PipelineResult(
      width: fixed.width,
      height: fixed.height,
      exposureFixed: !identical(fixed, raw),
      gate: GateResult(snake: false, boxConf: conf, maskArea: 0),
      answer: ModelAnswer.noSnake,
    );
    _lastVerdictFrame = details; // no snake now: no main box, snakeInView false
    final step = _lock.add(null, seconds);
    _show(step, details, snake: false);
  }

  Future<void> _verdictLane(
      RgbImage rgb, bool exposureFixed, double? conf, PixelBox? box, double seconds, int session) async {
    _verdictBusy = true;
    final clock = Stopwatch()..start();
    final times = <String, int>{};
    int lap() {
      final ms = clock.elapsedMilliseconds;
      clock.reset();
      return ms;
    }

    try {
      final result = await _verdictFrame(rgb, exposureFixed, conf, box, session, times, lap);
      if (result == null) return; // a reset or close ended this frame
      _lastVerdictFrame = result;
      final step = _lock.add(result.parts, seconds);
      _show(step, result, snake: result.gate.snake);
      verdictTimer.add(times.values.fold(0, (a, b) => a + b), seconds);
      _verdictError = null;
    } catch (e) {
      if (session == _session && !_closed) {
        _verdictError = e;
        _failed();
      }
    } finally {
      // Always free the lane, also when a reset ended this frame's session
      // (the new session cannot have started a verdict meanwhile).
      _verdictBusy = false;
      if (session == _session) _publish();
    }
  }

  /// One frame through [SnakePipeline.run]'s steps after the detector, on the
  /// native verdict lane; null when a reset or close ended it.
  Future<PipelineResult?> _verdictFrame(RgbImage rgb, bool exposureFixed, double? conf, PixelBox? box, int session,
      Map<String, int> times, int Function() lap) async {
    final s = _s;
    final size = s.inputSize;
    bool stale() => session != _session || _closed;
    PipelineResult noSnake(GateResult gate) => PipelineResult(
        width: rgb.width, height: rgb.height, exposureFixed: exposureFixed, gate: gate,
        answer: ModelAnswer.noSnake, timingsMs: times);

    double? snakeProbability;
    if (s.usesSnakeCheck) {
      final check = s.snakeCheck!;
      final checkIn = normalise(letterbox(rgb, check.inputSize, s.padRgb), s.mean, s.std, s.scale);
      final out = (await _verdict.invokeMapMethod<String, Float32List>(
          'check', {'input': checkIn, 'size': check.inputSize}))!;
      times['snake_check'] = lap();
      if (stale()) return null;
      final probs = out[check.output]!;
      snakeProbability = probs[check.snakeIndex];
      if (!check.isSnake(probs)) {
        return noSnake(GateResult(snake: false, boxConf: null, maskArea: 0, snakeProbability: snakeProbability));
      }
    }

    final segIn = normalise(letterbox(rgb, size, s.padRgb), s.mean, s.std, s.scale);
    final probability = (await _verdict.invokeMethod<Float32List>('segment', {'input': segIn, 'size': size}))!;
    times['segmenter'] = lap();
    if (stale()) return null;
    final post = afterGate(rgb, probability, conf, s);
    times['crop'] = lap();
    final gate = GateResult(
        snake: post.snake, boxConf: conf, maskArea: post.area, snakeProbability: snakeProbability, box: box);
    if (!post.snake) return noSnake(gate);

    Future<ModelOutputs> classify(Float32List input) async {
      final out = (await _verdict.invokeMapMethod<String, Float32List>('classify', {'input': input, 'size': size}))!;
      return ModelOutputs(
        embedding: out['embedding']!,
        speciesLogits: out['species_logits']!,
        whoLogits: out['who_logits']!,
        genusLogits: out['genus_logits']!,
        familyLogits: out['family_logits']!,
      );
    }

    final plain = await classify(post.classifierInput!);
    if (stale()) return null;
    final mirrored = post.mirroredInput == null ? null : await classify(post.mirroredInput!);
    if (stale()) return null;
    final outputs = classifierAnswer(plain, mirrored, s.classifier);
    times['classifier'] = lap();
    final parts = DecisionParts.fromOutputs(outputs, s.thresholds);
    return PipelineResult(
      width: rgb.width,
      height: rgb.height,
      exposureFixed: exposureFixed,
      gate: gate,
      cropWindow: post.window,
      answer: _rules.verdict(parts),
      parts: parts,
      classifierOutputs: outputs,
      classifierInputMean: post.mean,
      classifierInputStd: post.std,
      timingsMs: times,
    );
  }

  /// The tracked snakes confident enough to name separately.
  List<Track> _namedTracks() => [for (final t in _tracker.tracks) if (t.score >= snakeMinScore) t];

  void _clearTracks() {
    _trackLocks.clear();
    _trackNames.clear();
    _trackNamedAt.clear();
  }

  /// One turn of the verdict lane with several snakes in view: the snake named
  /// longest ago, through the snake check, the classifier and its own lock.
  Future<void> _trackLane(RgbImage rgb, bool exposureFixed, List<Track> tracks, double seconds, int session) async {
    _verdictBusy = true;
    final clock = Stopwatch()..start();
    final s = _s;
    final size = s.inputSize;
    bool stale() => session != _session || _closed;
    try {
      final track = tracks.reduce((a, b) =>
          (_trackNamedAt[a.id] ?? double.negativeInfinity) <= (_trackNamedAt[b.id] ?? double.negativeInfinity) ? a : b);
      _trackNamedAt[track.id] = seconds;
      final window = boxWindow(track.box, rgb.width, rgb.height, s.cropPadding);
      if (window == null) return;
      final crop = cropInputs(rgb, window, s, snakeCheck: s.usesSnakeCheck);
      final lock = _trackLocks.putIfAbsent(track.id, () => VideoLock(rules: _rules));
      final details = PipelineResult(
          width: rgb.width,
          height: rgb.height,
          exposureFixed: exposureFixed,
          gate: GateResult(snake: true, boxConf: track.score, maskArea: 0, box: track.box),
          answer: ModelAnswer.noSnake);
      if (crop.snakeCheckInput != null) {
        final check = s.snakeCheck!;
        final out = (await _verdict.invokeMapMethod<String, Float32List>(
            'check', {'input': crop.snakeCheckInput, 'size': check.inputSize}))!;
        if (stale()) return;
        if (!check.isSnake(out[check.output]!)) {
          lock.add(null, seconds);
          _trackNames.remove(track.id);
          return;
        }
      }
      Future<ModelOutputs> classify(Float32List input) async {
        final out = (await _verdict.invokeMapMethod<String, Float32List>('classify', {'input': input, 'size': size}))!;
        return ModelOutputs(
          embedding: out['embedding']!,
          speciesLogits: out['species_logits']!,
          whoLogits: out['who_logits']!,
          genusLogits: out['genus_logits']!,
          familyLogits: out['family_logits']!,
        );
      }

      final plain = await classify(crop.classifierInput);
      if (stale()) return;
      final mirrored = crop.mirroredInput == null ? null : await classify(crop.mirroredInput!);
      if (stale()) return;
      final parts = DecisionParts.fromOutputs(classifierAnswer(plain, mirrored, s.classifier), s.thresholds);
      final shown = lock.add(parts, seconds).shown;
      if (shown == null) {
        _trackNames.remove(track.id);
      } else {
        _trackNames[track.id] = _present(shown, details);
      }
      // The card shows the most dangerous named snake.
      final named = [for (final t in tracks) if (_trackNames[t.id] != null) _trackNames[t.id]!];
      _shown = named.isEmpty ? null : named.reduce((a, b) => b.verdict.severity > a.verdict.severity ? b : a);
      _hasShown = true;
      _lastVerdictFrame = details;
      verdictTimer.add(clock.elapsedMilliseconds, seconds);
      _verdictError = null;
    } catch (e) {
      if (session == _session && !_closed) {
        _verdictError = e;
        _failed();
      }
    } finally {
      _verdictBusy = false;
      if (session == _session) _publish();
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
    final live = {for (final t in tracks) t.id};
    for (final m in [_trackLocks, _trackNames, _trackNamedAt]) {
      m.removeWhere((id, _) => !live.contains(id));
    }
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
        if (t.score >= snakeMinScore || identical(t, main)) t,
    ];
    final view = LiveView(
      width: _width,
      height: _height,
      tracks: shown,
      mainTrack: main?.id,
      names: Map.unmodifiable(_trackNames),
      identification: _shown,
      answered: _hasShown,
      // The snake check (or the M4 gate) said snake on the last verdict
      // frame, whether or not its box is still tracked.
      snakeInView: _lastVerdictFrame?.gate.snake == true,
      boxFps: boxTimer.fps,
      verdictFps: verdictTimer.fps,
      boxMs: boxTimer.meanMs,
      verdictMs: verdictTimer.meanMs,
      skippedByGate: _skippedByGate,
      threads: laneThreads,
      error: _boxError ?? _verdictError,
    );
    _view = view;
    if (!_views.isClosed) _views.add(view);
  }

  Future<void> close() async {
    _closed = true;
    _session++;
    await _views.close();
    for (final channel in [_detector, _verdict]) {
      try {
        await channel.invokeMethod('close', {'owner': owner}); // each lane on its own
      } catch (_) {}
    }
  }
}

/// [frame] with its own copy of every plane's bytes.
LiveCameraFrame _copied(LiveCameraFrame frame) => LiveCameraFrame(
      width: frame.width,
      height: frame.height,
      format: frame.format,
      planes: [
        for (final p in frame.planes) LiveCameraPlane(Uint8List.fromList(p.bytes), p.bytesPerRow, p.bytesPerPixel)
      ],
      offsetMs: frame.offsetMs,
      sensorOrientation: frame.sensorOrientation,
      deviceOrientation: frame.deviceOrientation,
      frontFacing: frame.frontFacing,
    );

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
    this.names = const {},
    this.error,
  });

  /// The size of the frames the boxes are in (upright, long side <= 640).
  final int width, height;

  /// The boxes to draw, most confident first.
  final List<Track> tracks;

  /// The track the verdict is about, if the last verdict frame had a snake.
  final int? mainTrack;

  /// With several snakes in view: each named track's own answer, by track id.
  /// A tracked snake missing here is still being identified.
  final Map<int, Identification> names;

  /// What [OnnxSnakeClassifier.identifyFrame] would show: null while
  /// checking (once [answered]); a no-snake answer when nothing is in view.
  final Identification? identification;
  final bool answered;

  /// The verdict lane's last frame passed the gate and its box is tracked.
  final bool snakeInView;

  final double boxFps, verdictFps, boxMs, verdictMs;
  final int skippedByGate;
  final (int, int) threads;

  /// A lane's last failure while it keeps failing; null once both lanes work.
  final Object? error;
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
