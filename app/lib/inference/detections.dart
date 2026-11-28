import 'dart:math' as math;
import 'dart:typed_data';

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

/// [boxes] without repeats of the same snake ([sameSnake]), at most
/// [maxSnakes], highest score first.
List<Detection> distinctSnakes(Iterable<Detection> boxes) {
  final kept = <Detection>[];
  for (final d in boxes.toList()..sort((a, b) => b.score.compareTo(a.score))) {
    if (kept.every((k) => !sameSnake(k.box, d.box))) {
      kept.add(d);
      if (kept.length == maxSnakes) break;
    }
  }
  return kept;
}

/// The snakes in one frame: [detectAll]'s boxes from [snakeMinScore], with a
/// box mostly inside a better one dropped (the same snake), at most
/// [maxSnakes]. Highest score first.
List<Detection> snakeBoxes(List<double> output, int anchors, int snakeClass, int width, int height, int size) =>
    distinctSnakes(detectAll(output, anchors, snakeClass, width, height, size, minScore: snakeMinScore));

// Two snakes side by side often come back from the 320 px detector as one
// low box spanning both (0.24-0.30), while each half of the frame on its own
// scores each snake 0.73-0.85. So when the top box is unsure, the detector
// takes a second look at two overlapping halves along the long side. On
// validation (m5-results/multi_snake/split_pass.py) both snakes of a pair were
// found in 90 of 300 pairs instead of 42, with second boxes on one-snake
// photos unchanged (1 in 1,000) and none on 600 non-snakes.

/// A second look is taken when the best box ([snakeBoxes]) scores under this...
const splitBelow = 0.6;

/// ...and the detector's top score is at least this (an empty scene costs one run).
const splitAbove = 0.1;

/// A box from a half counts from this score.
const splitTileScore = 0.6;

/// Each half covers this share of the long side, so the two overlap.
const splitTileShare = 0.6;

/// A half's box this close to the cut (share of the long side) is a cut-off
/// piece of a snake that crosses it, not a snake of its own.
const splitSeamGap = 0.02;

/// One half of a frame for the second look, in frame pixels.
typedef SplitTile = ({int left, int top, int width, int height});

/// The detector's highest score among boxes inside the frame, when at least
/// [splitAbove] (an anchor in the letterbox padding does not count).
double? splitTop(List<double> output, int anchors, int snakeClass, int width, int height, int size) {
  final top = detectAll(output, anchors, snakeClass, width, height, size, minScore: splitAbove, maxBoxes: 1);
  return top.isEmpty ? null : top.first.score;
}

/// True when [found] (from [snakeBoxes]) is unsure enough for a second look;
/// [top] is [splitTop].
bool needsSplit(List<Detection> found, double? top) =>
    top != null && top >= splitAbove && (found.isEmpty || found.first.score < splitBelow);

/// The two overlapping halves of a [width] x [height] frame along its long side.
List<SplitTile> splitTiles(int width, int height) {
  if (width >= height) {
    final cut = pyRound(width * splitTileShare);
    return [(left: 0, top: 0, width: cut, height: height), (left: width - cut, top: 0, width: cut, height: height)];
  }
  final cut = pyRound(height * splitTileShare);
  return [(left: 0, top: 0, width: width, height: cut), (left: 0, top: height - cut, width: width, height: cut)];
}

/// The detector inputs of [rgb]'s two halves ([splitTiles]).
List<Float32List> splitInputs(RgbImage rgb, int size, int padValue) => [
      for (final t in splitTiles(rgb.width, rgb.height))
        detectorInput(rgb.crop(t.top, t.top + t.height, t.left, t.left + t.width), size, padValue)
    ];

/// The snakes the halves' detector [outputs] find in a [width] x [height]
/// frame, or null when they do not find two (the first look stands).
List<Detection>? splitSnakes(List<List<double>> outputs, int snakeClass, int width, int height, int size) {
  final tiles = splitTiles(width, height);
  final wide = width >= height;
  final seam = splitSeamGap * math.max(width, height);
  final found = <Detection>[];
  for (var i = 0; i < tiles.length; i++) {
    final t = tiles[i];
    final out = outputs[i];
    for (final d in detectAll(out, out.length ~/ 5, snakeClass, t.width, t.height, size, minScore: splitTileScore)) {
      final b = d.box;
      final gap = wide ? (i == 0 ? t.width - b.right : b.left) : (i == 0 ? t.height - b.bottom : b.top);
      if (gap < seam) continue;
      found.add(Detection(PixelBox(b.left + t.left, b.top + t.top, b.right + t.left, b.bottom + t.top), d.score));
    }
  }
  final kept = distinctSnakes(found);
  return kept.length >= 2 ? kept : null;
}

/// The index of the box in [boxes] that overlaps [target] most (IoU); the
/// first on a tie.
int closestBox(List<PixelBox> boxes, PixelBox target) {
  var best = 0;
  for (var i = 1; i < boxes.length; i++) {
    if (iou(boxes[i], target) > iou(boxes[best], target)) best = i;
  }
  return best;
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
