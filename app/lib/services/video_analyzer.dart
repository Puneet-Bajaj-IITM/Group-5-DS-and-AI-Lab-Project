import 'dart:math' as math;

import 'package:flutter/services.dart';

import '../inference/image_ops.dart';
import '../inference/pipeline.dart';
import '../inference/snake_tracks.dart';
import '../inference/video_lock.dart';
import '../models/verdict.dart';
import 'classifier.dart';

/// One processed frame of a clip.
/// One tracked snake on a frame: its box and its own settled answer (null
/// while it is still being identified).
typedef SnakeLabel = ({PixelBox box, Identification? id});

class VideoStep {
  const VideoStep(
      {required this.seconds, required this.frame, required this.lock, required this.shown, this.snakes = const []});

  /// Every tracked snake on this frame, each named by its own lock.
  final List<SnakeLabel> snakes;

  /// Where the frame sits in the clip.
  final double seconds;

  /// The frame on its own (gate, top species, timings).
  final PipelineResult frame;

  /// The video lock after this frame.
  final LockStep lock;

  /// What the screen shows from this moment: null means "Checking...".
  final Identification? shown;
}

class VideoAnalysis {
  VideoAnalysis(this.steps, this.durationSeconds, this.analysedSeconds);

  final List<VideoStep> steps;
  final double durationSeconds, analysedSeconds;

  /// The step showing at [seconds] of playback.
  VideoStep? at(double seconds) {
    VideoStep? last;
    for (final s in steps) {
      if (s.seconds > seconds + 1e-6) break;
      last = s;
    }
    return last;
  }

  Identification? get finalShown => steps.isEmpty ? null : steps.last.shown;

  /// When the first verdict appeared, in seconds of the clip.
  double? get firstAnswerSeconds => steps.where((s) => s.shown != null).firstOrNull?.seconds;

  /// How often the shown verdict or species changed after the first answer.
  int get changes {
    String? prev;
    var n = -1;
    for (final s in steps) {
      final key = s.shown == null ? null : '${s.shown!.verdict}/${s.shown!.topSpecies?.id}';
      if (key != null && key != prev) {
        n++;
        prev = key;
      }
    }
    return math.max(0, n);
  }

  int get snakeFrames => steps.where((s) => s.frame.gate.snake).length;

  double get meanFrameMs =>
      steps.isEmpty ? 0 : steps.map((s) => s.frame.totalMs).reduce((a, b) => a + b) / steps.length;
}

class VideoProgress {
  const VideoProgress(this.done, this.total, {this.analysis, this.latest, this.frame});
  final int done, total;
  final VideoAnalysis? analysis;

  /// The step just checked, and the frame it was checked on, so the screen
  /// can show the analysis as it happens.
  final VideoStep? latest;

  /// The frame as RGBA bytes (as decoded on Android), ready to show.
  final VideoFrameImage? frame;
}

class VideoFrameImage {
  const VideoFrameImage(this.width, this.height, this.rgba);
  final int width, height;
  final Uint8List rgba;
}

/// Runs a video file through the model at up to [targetFps] frames a
/// second (the M4 live-video target), with the M4 video lock.
class VideoAnalyzer {
  VideoAnalyzer(this.classifier);

  final OnnxSnakeClassifier classifier;
  static const _channel = MethodChannel('serpentguard/video_frames');
  static const targetFps = 10;

  Stream<VideoProgress> analyse(String path, {double maxSeconds = 20}) async* {
    final info = (await _channel.invokeMapMethod<String, dynamic>('open', {'path': path}))!;
    final id = info['id'] as int;
    try {
      final duration = (info['durationMs'] as num).toDouble() / 1000;
      final frameCount = info['frameCount'] as int?;
      final seconds = math.min(duration, maxSeconds);
      final lock = classifier.newLock();
      final tracks = classifier.newTracks();
      final steps = <VideoStep>[];

      Future<VideoStep> process(RgbImage frame, double t) async {
        final r = await classifier.runFrame(frame, nameOthers: true);
        final step = lock.add(r.parts, t);
        final shown = !r.gate.snake && step.pooled == null
            ? null
            : (step.shown == null ? null : classifier.fromAnswer(step.shown!, details: r));
        final named = tracks.update([
          if (r.gate.snake && r.gate.box != null) FrameSnake(r.gate.box!, r.parts),
          for (final o in r.others) FrameSnake(o.box, o.parts),
        ], t);
        return VideoStep(seconds: t, frame: r, lock: step, shown: shown, snakes: [
          for (final n in named)
            (box: n.box, id: n.answer == null ? null : classifier.fromAnswer(n.answer!, details: r)),
        ]);
      }

      if (frameCount != null && frameCount > 0 && duration > 0) {
        final fps = frameCount / duration;
        final step = math.max(1, (fps / targetFps).round());
        final last = math.min(frameCount, (seconds * fps).floor());
        final total = (last / step).ceil();
        // Decode in short runs so a few full-size frames are in memory at a time.
        final run = step * 3;
        for (var start = 0; start < last; start += run) {
          final frames = await _channel.invokeListMethod<Map<dynamic, dynamic>>('frames', {
            'id': id,
            'start': start,
            'count': math.min(run, last - start),
            'step': step,
            'maxSide': OnnxSnakeClassifier.frameMaxSide,
          });
          for (final f in frames!) {
            final t = (f['index'] as int) / fps;
            final rgb = _rgb(f);
            steps.add(await process(rgb, t));
            yield VideoProgress(steps.length, total, latest: steps.last, frame: _image(f));
          }
        }
      } else {
        final total = (seconds * targetFps).floor();
        for (var i = 0; i < total; i++) {
          final t = i / targetFps;
          final f = await _channel.invokeMapMethod<dynamic, dynamic>('frameAt', {
            'id': id,
            'timeMs': (t * 1000).round(),
            'maxSide': OnnxSnakeClassifier.frameMaxSide,
          });
          if (f == null) continue;
          final rgb = _rgb(f);
          steps.add(await process(rgb, t));
          yield VideoProgress(steps.length, total, latest: steps.last, frame: _image(f));
        }
      }
      yield VideoProgress(steps.length, steps.length, analysis: VideoAnalysis(steps, duration, seconds));
    } finally {
      await _channel.invokeMethod('close', {'id': id});
    }
  }

  static VideoFrameImage _image(Map<dynamic, dynamic> f) =>
      VideoFrameImage(f['width'] as int, f['height'] as int, f['rgba'] as Uint8List);

  static RgbImage _rgb(Map<dynamic, dynamic> f) =>
      RgbImage.fromRgba(f['width'] as int, f['height'] as int, f['rgba'] as Uint8List);
}
