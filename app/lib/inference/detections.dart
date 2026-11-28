import 'dart:math' as math;

import 'image_ops.dart';

/// Every snake the detector sees in one frame, and a tracker that keeps the
/// boxes steady between frames. Display only: the verdict never reads these
/// (it uses the gate's `box_conf`, the U-Net mask and the classifier).

/// One detector box in image pixels, with its score.
class Detection {
  const Detection(this.box, this.score);
  final PixelBox box;
  final double score;
}

/// A box is drawn as a snake from this detector score (Ultralytics' default confidence).
const snakeMinScore = 0.25;

/// Non-maximum suppression drops a box overlapping a better one this much (IoU).
const nmsIou = 0.5;

/// A box this much inside a better one (intersection over the smaller box's
/// area) is the same snake, e.g. a head box inside the body box.
const containLimit = 0.6;

/// At most this many snakes are boxed in one photo or frame.
const maxSnakes = 3;

double _intersection(PixelBox a, PixelBox b) {
  final w = math.min(a.right, b.right) - math.max(a.left, b.left);
  final h = math.min(a.bottom, b.bottom) - math.max(a.top, b.top);
  return w <= 0 || h <= 0 ? 0 : w * h;
}

double iou(PixelBox a, PixelBox b) {
  final inter = _intersection(a, b);
  return inter == 0 ? 0 : inter / (a.width * a.height + b.width * b.height - inter);
}

/// The share of the smaller box that lies inside the other.
double containment(PixelBox a, PixelBox b) {
  final inter = _intersection(a, b);
  return inter == 0 ? 0 : inter / math.min(a.width * a.height, b.width * b.height);
}

/// True when two boxes are taken for the same snake.
bool sameSnake(PixelBox a, PixelBox b) => iou(a, b) >= nmsIou || containment(a, b) >= containLimit;

/// The boxes scoring at least [minScore], mapped from the detector's
/// letterboxed [size] x [size] input to the [width] x [height] image (the
/// same mapping as [bestBox]), after non-maximum suppression at [iouLimit].
/// Highest score first. The output holds, per anchor, rows cx, cy, w, h and
/// the score.
List<Detection> detectAll(List<double> output, int anchors, int snakeClass, int width, int height, int size,
    {required double minScore, double iouLimit = nmsIou, int maxBoxes = 8}) {
  final row = 4 + snakeClass;
  final ratio = math.min(size / height, size / width);
  final nw = math.max(pyRound(width * ratio), 1), nh = math.max(pyRound(height * ratio), 1);
  final left = pyRound((size - nw) / 2 - 0.1), top = pyRound((size - nh) / 2 - 0.1);
  double clampX(double v) => v.clamp(0, width.toDouble()).toDouble();
  double clampY(double v) => v.clamp(0, height.toDouble()).toDouble();

  final candidates = <Detection>[];
  for (var i = 0; i < anchors; i++) {
    final score = output[row * anchors + i];
    if (score < minScore) continue;
    final cx = output[i], cy = output[anchors + i];
    final w = output[2 * anchors + i], h = output[3 * anchors + i];
    final box = PixelBox(
      clampX((cx - w / 2 - left) / ratio),
      clampY((cy - h / 2 - top) / ratio),
      clampX((cx + w / 2 - left) / ratio),
      clampY((cy + h / 2 - top) / ratio),
    );
    // A box clamped to nothing (an anchor in the letterbox padding) has no
    // overlap with anything, so NMS would never drop it: skip it here. Only
    // the display uses these boxes; the verdict's box is bestBox's.
    if (box.width <= 0 || box.height <= 0) continue;
    candidates.add(Detection(box, score));
  }
  candidates.sort((a, b) => b.score.compareTo(a.score));
  final kept = <Detection>[];
  for (final c in candidates) {
    if (kept.every((k) => iou(k.box, c.box) < iouLimit)) {
      kept.add(c);
      if (kept.length == maxBoxes) break;
    }
  }
  return kept;
}

/// The snakes in one frame: [detectAll]'s boxes from [snakeMinScore], with a
/// box mostly inside a better one dropped (the same snake), at most
/// [maxSnakes]. Highest score first.
List<Detection> snakeBoxes(List<double> output, int anchors, int snakeClass, int width, int height, int size) {
  final kept = <Detection>[];
  for (final d in detectAll(output, anchors, snakeClass, width, height, size, minScore: snakeMinScore)) {
    if (kept.every((k) => containment(k.box, d.box) < containLimit)) {
      kept.add(d);
      if (kept.length == maxSnakes) break;
    }
  }
  return kept;
}

/// A box followed across frames.
class Track {
  Track(this.id, this.box, this.score, this.lastSeen) : firstSeen = lastSeen;
  final int id;
  PixelBox box;
  double score;
  double lastSeen;
  final double firstSeen;
  int hits = 1;
}

/// Greedy IoU tracking with smoothed boxes, so a box glides with the snake
/// instead of jumping, and a missed frame or two does not make it blink.
class BoxTracker {
  BoxTracker({this.matchIou = 0.3, this.smoothing = 0.6, this.keepSeconds = 0.35});

  /// Least IoU for a detection to continue a track.
  final double matchIou;

  /// Weight of the new detection in the smoothed box (1 = no smoothing).
  final double smoothing;

  /// How long a track stays drawn without a matching detection.
  final double keepSeconds;

  final List<Track> _tracks = [];
  int _next = 0;

  List<Track> get tracks => List.unmodifiable(_tracks);

  void reset() {
    _tracks.clear();
    _next = 0;
  }

  /// One frame's detections at [seconds]; returns the live tracks, most
  /// confident first.
  List<Track> update(List<Detection> detections, double seconds) {
    final free = List<Track>.of(_tracks);
    for (final d in detections) {
      Track? best;
      var bestIou = matchIou;
      for (final t in free) {
        final o = iou(t.box, d.box);
        if (o >= bestIou) {
          best = t;
          bestIou = o;
        }
      }
      if (best == null) {
        _tracks.add(Track(_next++, d.box, d.score, seconds));
      } else {
        free.remove(best);
        best.box = _blend(best.box, d.box);
        best.score = d.score;
        best.lastSeen = seconds;
        best.hits++;
      }
    }
    _tracks.removeWhere((t) => seconds - t.lastSeen > keepSeconds);
    _tracks.sort((a, b) => b.score.compareTo(a.score));
    return tracks;
  }

  PixelBox _blend(PixelBox old, PixelBox now) {
    double mix(double a, double b) => a + smoothing * (b - a);
    return PixelBox(mix(old.left, now.left), mix(old.top, now.top), mix(old.right, now.right), mix(old.bottom, now.bottom));
  }
}
