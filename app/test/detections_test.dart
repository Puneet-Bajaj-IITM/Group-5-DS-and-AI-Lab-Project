import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:serpentguard/inference/detections.dart';
import 'package:serpentguard/inference/image_ops.dart';

const anchors = 2100, size = 320;

/// A detector output (rows cx, cy, w, h, score) with the given boxes in
/// letterboxed input pixels; every other anchor scores 0.
Float32List output(List<(double cx, double cy, double w, double h, double score)> boxes) {
  final out = Float32List(5 * anchors);
  for (var i = 0; i < boxes.length; i++) {
    final (cx, cy, w, h, score) = boxes[i];
    out[i] = cx;
    out[anchors + i] = cy;
    out[2 * anchors + i] = w;
    out[3 * anchors + i] = h;
    out[4 * anchors + i] = score;
  }
  return out;
}

void main() {
  group('detectAll', () {
    test('its top box is bestBox exactly (unless bestBox is empty), on random outputs and frame shapes', () {
      final rnd = math.Random(7);
      var compared = 0;
      for (final (w, h) in [(640, 360), (360, 640), (640, 480), (500, 500), (333, 640)]) {
        for (var trial = 0; trial < 20; trial++) {
          final out = Float32List(5 * anchors);
          for (var i = 0; i < anchors; i++) {
            out[i] = rnd.nextDouble() * size;
            out[anchors + i] = rnd.nextDouble() * size;
            out[2 * anchors + i] = 5 + rnd.nextDouble() * 150;
            out[3 * anchors + i] = 5 + rnd.nextDouble() * 150;
            out[4 * anchors + i] = rnd.nextDouble();
          }
          final best = bestBox(out, anchors, 0, w, h, size)!;
          final found = detectAll(out, anchors, 0, w, h, size, minScore: 0);
          expect(found.every((d) => d.box.width > 0 && d.box.height > 0), isTrue, reason: 'no empty boxes');
          // An anchor in the letterbox padding clamps to nothing: bestBox keeps
          // it (the gate's box), detectAll (display only) skips it.
          if (best.width <= 0 || best.height <= 0) continue;
          final top = found.first.box;
          expect([top.left, top.top, top.right, top.bottom], [best.left, best.top, best.right, best.bottom]);
          compared++;
        }
      }
      expect(compared, greaterThan(50), reason: 'most random tops are real boxes');
    });

    test('keeps separate snakes, drops overlapping duplicates and low scores', () {
      final out = output([
        (100, 160, 80, 40, 0.9), // snake 1
        (104, 162, 80, 40, 0.8), // duplicate of snake 1
        (250, 160, 60, 60, 0.6), // snake 2
        (250, 80, 30, 30, 0.1), // under the cutoff
      ]);
      final found = detectAll(out, anchors, 0, 640, 640, size, minScore: 0.25);
      expect(found.map((d) => d.score), [closeTo(0.9, 1e-6), closeTo(0.6, 1e-6)]);
    });

    test('maps letterboxed pixels back to the frame', () {
      // 640 x 360 frame: scale 0.5, placed 320 x 180 with 70 px bars on top and bottom.
      final out = output([(160, 160, 100, 50, 0.9)]);
      final b = detectAll(out, anchors, 0, 640, 360, size, minScore: 0.25).single.box;
      expect([b.left, b.top, b.right, b.bottom], [220, 130, 420, 230]);
    });
  });

  group('snakeBoxes', () {
    // A 640 x 640 frame: the 320 px input at half scale, no letterbox bars.
    List<Detection> boxes(List<(double, double, double, double, double)> b) =>
        snakeBoxes(output(b), anchors, 0, 640, 640, size);

    test('a head box inside the body box is one snake', () {
      final body = (160.0, 160.0, 200.0, 100.0, 0.9), head = (90.0, 150.0, 40.0, 30.0, 0.7);
      expect(detectAll(output([body, head]), anchors, 0, 640, 640, size, minScore: snakeMinScore), hasLength(2),
          reason: 'NMS alone keeps both (IoU under 0.5)');
      expect(boxes([body, head]).map((d) => d.score), [closeTo(0.9, 1e-6)]);
    });

    test('two separate snakes are two boxes, best first', () {
      final found = boxes([(80, 80, 60, 60, 0.8), (240, 240, 60, 60, 0.9)]);
      expect(found.map((d) => d.score), [closeTo(0.9, 1e-6), closeTo(0.8, 1e-6)]);
    });

    test('at most maxSnakes boxes', () {
      final found = boxes([
        (40, 40, 40, 40, 0.9),
        (160, 40, 40, 40, 0.8),
        (280, 40, 40, 40, 0.7),
        (40, 280, 40, 40, 0.6),
      ]);
      expect(found, hasLength(maxSnakes));
      expect(found.last.score, closeTo(0.7, 1e-6));
    });

    test('scores under snakeMinScore are no snake', () {
      expect(boxes([(80, 80, 60, 60, 0.2), (240, 240, 60, 60, 0.1)]), isEmpty);
    });

    test('sameSnake: overlap or containment either way', () {
      const big = PixelBox(0, 0, 100, 100), inside = PixelBox(10, 10, 40, 40), apart = PixelBox(200, 0, 260, 60);
      expect(sameSnake(big, inside), isTrue);
      expect(sameSnake(inside, big), isTrue);
      expect(sameSnake(big, big), isTrue);
      expect(sameSnake(big, apart), isFalse);
      expect(containment(big, const PixelBox(80, 0, 180, 100)), closeTo(0.2, 1e-9));
    });
  });

  group('BoxTracker', () {
    PixelBox box(double x) => PixelBox(x, 100, x + 100, 200);

    test('follows a moving snake with one id and smooths the box', () {
      final t = BoxTracker(smoothing: 0.5);
      final first = t.update([Detection(box(0), 0.9)], 0.0).single;
      final next = t.update([Detection(box(20), 0.9)], 0.05).single;
      expect(next.id, first.id);
      expect(next.box.left, 10); // halfway to the new box
      expect(next.hits, 2);
    });

    test('keeps a track through a short miss and drops it after keepSeconds', () {
      final t = BoxTracker(keepSeconds: 0.35);
      t.update([Detection(box(0), 0.9)], 0.0);
      expect(t.update([], 0.2), hasLength(1));
      expect(t.update([], 0.5), isEmpty);
    });

    test('two snakes get two tracks, most confident first', () {
      final t = BoxTracker();
      final tracks = t.update([Detection(box(0), 0.5), Detection(box(300), 0.9)], 0.0);
      expect(tracks.map((x) => x.score), [0.9, 0.5]);
      expect(tracks.map((x) => x.id).toSet(), hasLength(2));
    });
  });

  group('second look on two halves', () {
    // A 2000 x 700 frame: halves x 0-1200 and x 800-2000, each 1200 x 700.
    const width = 2000, height = 700, offset = 800;
    final ratio = math.min(size / height, size / 1200);
    final top = pyRound((size - pyRound(height * ratio)) / 2 - 0.1);

    /// A half's detector output with boxes given in that half's pixels.
    Float32List half(List<(double left, double top, double right, double bottom, double score)> boxes) => output([
          for (final (l, t, r, b, s) in boxes)
            ((l + r) / 2 * ratio, (t + b) / 2 * ratio + top, (r - l) * ratio, (b - t) * ratio, s)
        ]);

    test('the halves are the long side\'s first and last 60 percent', () {
      expect(splitTiles(2000, 700), [
        (left: 0, top: 0, width: 1200, height: 700),
        (left: 800, top: 0, width: 1200, height: 700),
      ]);
      expect(splitTiles(700, 2000).last, (left: 0, top: 800, width: 700, height: 1200));
    });

    test('only an unsure top box gets a second look', () {
      const box = PixelBox(0, 0, 10, 10);
      expect(needsSplit([const Detection(box, 0.3)], 0.3), isTrue);
      expect(needsSplit([const Detection(box, 0.8)], 0.8), isFalse, reason: 'a sure box stands');
      expect(needsSplit(const [], 0.24), isTrue, reason: 'a box spanning two snakes can score under 0.25');
      expect(needsSplit(const [], 0.05), isFalse, reason: 'an empty scene costs one detector run');
      expect(needsSplit(const [], null), isFalse);
    });

    test('a snake in each half: two snakes in frame pixels, best first', () {
      final found = splitSnakes([
        half([(100, 100, 500, 600, 0.8)]),
        half([(500, 150, 1100, 450, 0.85)]),
      ], 0, width, height, size)!;
      expect(found, hasLength(2));
      expect(found.first.score, closeTo(0.85, 1e-6));
      expect(found.first.box.left, closeTo(500 + offset, 1));
      expect(found.last.box.right, closeTo(500, 1));
    });

    test('one snake across the cut is not two snakes', () {
      expect(
          splitSnakes([
            // A long snake: each half sees a piece cut off at its inner edge,
            // too little overlapping the other piece to merge as one box.
            half([(300, 100, 1198, 600, 0.8)]),
            half([(2, 100, 900, 600, 0.8)]),
          ], 0, width, height, size),
          isNull);
    });

    test('one snake in the overlap, seen by both halves, is one snake', () {
      expect(
          splitSnakes([
            half([(900, 100, 1100, 600, 0.8)]),
            half([(100, 100, 300, 600, 0.75)]),
          ], 0, width, height, size),
          isNull);
    });

    test('on a photo the main box is the split box the mask crop overlaps most', () {
      const rat = PixelBox(1456, 179, 2231, 444), cobra = PixelBox(450, 110, 1027, 700);
      const maskCrop = PixelBox(420, 90, 1060, 700);
      expect(closestBox([rat, cobra], maskCrop), 1);
      expect(closestBox([cobra, rat], maskCrop), 0);
    });

    test('boxes under splitTileScore do not count', () {
      expect(
          splitSnakes([
            half([(100, 100, 500, 600, splitTileScore - 0.05)]),
            half([(500, 150, 1100, 450, 0.9)]),
          ], 0, width, height, size),
          isNull);
    });
  });
}
