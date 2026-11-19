import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:serpentguard/inference/image_ops.dart';

/// Pixel parity of the Dart image steps with cv2.
///
/// The cv2 side is written by `tool/make_cv_fixtures.py` (see the app
/// README). Set SG_CV_FIXTURES to its output folder; without it the parity
/// part is skipped and only the self-contained checks run.
void main() {
  test('ellipse closing fills a one-pixel gap and keeps the border', () {
    const w = 12, h = 7;
    final m = Uint8List(w * h);
    for (var y = 1; y < 6; y++) {
      for (var x = 0; x < w; x++) {
        if (x != 6) m[y * w + x] = 1;
      }
    }
    final cleaned = cleanMask(m, w, h,
        connectivity: 8, minLargestShare: 0.015, keepShareOfLargest: 0.12, keepShareOfPhoto: 0.008)!;
    expect(cleaned[3 * w + 6], 1, reason: 'gap closed');
    expect(cleaned[3 * w + 0], 1, reason: 'no erosion from the image border');
  });

  test('crop window pads by int((x2 - x1) * 0.06) and clips', () {
    const w = 100, h = 50;
    final m = Uint8List(w * h);
    for (var y = 10; y <= 40; y++) {
      for (var x = 20; x <= 70; x++) {
        m[y * w + x] = 1;
      }
    }
    expect(cropWindow(m, h, w, 0.06), (10 - 1, 40 + 1 + 1, 20 - 3, 70 + 3 + 1));
  });

  test('placed size uses Python rounding', () {
    expect(placedSize(500, 490, 320), (314, 320));
    expect(pyRound(2.5), 2);
    expect(pyRound(3.5), 4);
  });

  final dir = Platform.environment['SG_CV_FIXTURES'];
  final skip = dir == null ? 'set SG_CV_FIXTURES to run the cv2 parity checks' : null;

  group('parity with cv2', () {
    late Map<String, dynamic> index;
    setUpAll(() {
      if (dir != null) index = jsonDecode(File('$dir/index.json').readAsStringSync()) as Map<String, dynamic>;
    });

    Uint8List read(String stem, String kind) => File('$dir/$stem.$kind').readAsBytesSync();

    /// (largest difference, share of values that differ)
    (int, double) compare(Uint8List a, Uint8List b) {
      expect(a.length, b.length);
      var maxD = 0, n = 0;
      for (var i = 0; i < a.length; i++) {
        final d = (a[i] - b[i]).abs();
        if (d > 0) n++;
        maxD = math.max(maxD, d);
      }
      return (maxD, n / a.length);
    }

    void check(String name, Uint8List got, Uint8List want, {int maxDiff = 1, double maxShare = 0.02}) {
      final (d, share) = compare(got, want);
      // ignore: avoid_print
      print('$name: max diff $d, ${(share * 100).toStringAsFixed(3)}% of values differ');
      expect(d, lessThanOrEqualTo(maxDiff), reason: name);
      expect(share, lessThanOrEqualTo(maxShare), reason: name);
    }

    test('letterbox, detector input, resizes, mask, cleanup, exposure', () {
      for (final MapEntry(key: stem, value: info) in index.entries) {
        final w = info['w'] as int, h = info['h'] as int;
        final rgb = RgbImage(w, h, read(stem, 'rgb'));
        expect(rgb.meanValue, closeTo(info['mean'] as double, 1e-9));

        check('$stem letterbox (INTER_AREA)', letterbox(rgb, 320, const [124, 116, 104]).data,
            read(stem, 'letterbox'));

        final det = detectorInput(rgb, 320, 114);
        final det8 = Uint8List(320 * 320 * 3);
        for (var i = 0; i < 320 * 320; i++) {
          for (var c = 0; c < 3; c++) {
            det8[i * 3 + c] = (det[c * 320 * 320 + i] * 255).round();
          }
        }
        check('$stem detector input (INTER_LINEAR)', det8, read(stem, 'detector'));
        check('$stem upscale 1.5x (INTER_LINEAR)', resizeLinear(rgb, w * 3 ~/ 2, h * 3 ~/ 2).data,
            read(stem, 'up'));
        check('$stem half (INTER_AREA)', resizeArea(rgb, w ~/ 2, h ~/ 2).data, read(stem, 'half'));

        final prob = Float32List.view(read(stem, 'prob').buffer);
        final mask = unletterboxMask(prob, h, w, 320, 0.5);
        check('$stem unletterbox mask', mask, read(stem, 'mask'), maxShare: 0.001);
        final clean = cleanMask(read(stem, 'mask'), w, h,
            connectivity: 8, minLargestShare: 0.015, keepShareOfLargest: 0.12, keepShareOfPhoto: 0.008);
        expect(clean == null, info['clean_none']);
        if (clean != null) check('$stem clean mask', clean, read(stem, 'clean'), maxDiff: 1, maxShare: 0);
        final win = info['window'] as List?;
        final got = cropWindow(clean, h, w, 0.06);
        expect(got == null ? null : [got.$1, got.$2, got.$3, got.$4], win);

        final dark = RgbImage(w, h, read(stem, 'dark'));
        final lab = read(stem, 'lab');
        final (l, a, b) = rgbToLab(dark);
        final planes = Uint8List(lab.length);
        for (var i = 0; i < w * h; i++) {
          planes[i * 3] = l[i];
          planes[i * 3 + 1] = a[i];
          planes[i * 3 + 2] = b[i];
        }
        check('$stem RGB -> LAB', planes, lab, maxShare: 0.0001);
        final cvL = Uint8List.fromList([for (var i = 0; i < w * h; i++) lab[i * 3]]);
        check('$stem CLAHE', clahe(cvL, w, h, 2.0, (8, 8)), read(stem, 'clahe'), maxDiff: 0, maxShare: 0);
        check('$stem exposure fix (LAB + CLAHE)', fixExposure(dark, 60, 200, 2.0, (8, 8)).data,
            read(stem, 'fixed'), maxShare: 0.0001);
      }
    }, skip: skip);
  });
}
