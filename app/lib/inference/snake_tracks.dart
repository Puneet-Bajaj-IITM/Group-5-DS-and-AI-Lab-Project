import 'decision.dart';
import 'detections.dart';
import 'image_ops.dart';
import 'video_lock.dart';

/// One snake on one frame: its box and, when it was named on that frame, the
/// decision parts of its own crop.
class FrameSnake {
  const FrameSnake(this.box, this.parts);
  final PixelBox box;
  final DecisionParts? parts;
}

/// One tracked snake as the screen draws it: its box, and the answer its own
/// lock has settled on (null while it is still being identified).
class NamedTrack {
  const NamedTrack(this.id, this.box, this.answer);
  final int id;
  final PixelBox box;
  final ModelAnswer? answer;
}

/// Every snake across the frames of a clip, each with its own video lock, so
/// two snakes keep two names and a new snake starts fresh instead of
/// inheriting the last one's answer.
class SnakeTracks {
  SnakeTracks(this._rules, {LockSettings settings = const LockSettings(), BoxTracker? tracker})
      : _settings = settings,
        _tracker = tracker ?? BoxTracker(keepSeconds: settings.holdSeconds);

  final DecisionRules _rules;
  final LockSettings _settings;
  final BoxTracker _tracker;
  final Map<int, VideoLock> _locks = {};
  final Map<int, ModelAnswer> _answers = {};

  /// Least IoU between a frame's box and a track's for the box to feed that track.
  static const matchIou = 0.3;

  /// One frame at [seconds]: feeds each snake's parts to its own track's lock
  /// and returns every live track with its settled answer.
  List<NamedTrack> update(List<FrameSnake> snakes, double seconds) {
    final tracks = _tracker.update([for (final s in snakes) Detection(s.box, 1)], seconds);
    final fed = <int>{};
    for (final s in snakes) {
      Track? best;
      var bestIou = matchIou;
      for (final t in tracks) {
        final o = iou(t.box, s.box);
        if (o >= bestIou && !fed.contains(t.id)) {
          best = t;
          bestIou = o;
        }
      }
      if (best == null || s.parts == null) continue;
      fed.add(best.id);
      final lock = _locks.putIfAbsent(best.id, () => VideoLock(rules: _rules, settings: _settings));
      final shown = lock.add(s.parts, seconds).shown;
      if (shown == null) {
        _answers.remove(best.id);
      } else {
        _answers[best.id] = shown;
      }
    }
    final live = {for (final t in tracks) t.id};
    _locks.removeWhere((id, _) => !live.contains(id));
    _answers.removeWhere((id, _) => !live.contains(id));
    return [for (final t in tracks) NamedTrack(t.id, t.box, _answers[t.id])];
  }

  void reset() {
    _tracker.reset();
    _locks.clear();
    _answers.clear();
  }
}
