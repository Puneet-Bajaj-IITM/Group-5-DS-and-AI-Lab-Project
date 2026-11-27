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
    test('its top box is bestBox exactly, on random outputs and frame shapes', () {
      final rnd = math.Random(7);
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
          final top = detectAll(out, anchors, 0, w, h, size, minScore: 0).first.box;
          expect([top.left, top.top, top.right, top.bottom], [best.left, best.top, best.right, best.bottom]);
        }
      }
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
}
