import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:ui' as ui;

import 'package:crypto/crypto.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:serpentguard/inference/bundle.dart';
import 'package:serpentguard/inference/image_ops.dart';

/// The snake check's input on the golden photos, built by the Dart code on
/// the test host, against the Python reference's mean and std
/// (`golden_photos.json` steps.snake_check). No model runs here; the whole
/// path is the integration test. The photos are not in git: point
/// GOLDEN_PHOTO_DIR at a folder of `<photo_id>.jpg`, else this skips.
const _inputTolerance = 1e-4;

void main() {
  final dir = Platform.environment['GOLDEN_PHOTO_DIR'];
  final settings = PipelineSettings.parse(File('assets/model/pipeline.json').readAsStringSync());
  final golden = jsonDecode(File('test/golden/golden_photos.json').readAsStringSync()) as Map<String, dynamic>;

  testWidgets('snake-check inputs match Python on every golden photo', (tester) async {
    final check = settings.snakeCheck!;
    var worst = 0.0;
    for (final photo in (golden['photos'] as List).cast<Map<String, dynamic>>()) {
      final bytes = File('$dir/${photo['photo_id']}.jpg').readAsBytesSync();
      expect(sha1.convert(bytes).toString(), photo['sha1']);
      final rgb = await tester.runAsync(() async {
        final codec = await ui.instantiateImageCodec(bytes);
        final image = (await codec.getNextFrame()).image;
        final rgba = (await image.toByteData(format: ui.ImageByteFormat.rawRgba))!;
        return RgbImage.fromRgba(image.width, image.height, rgba.buffer.asUint8List());
      });
      final fixed = fixExposure(rgb!, settings.exposureBelow, settings.exposureAbove, settings.claheClip, settings.claheTiles);
      final input = normalise(letterbox(fixed, check.inputSize, settings.padRgb), settings.mean, settings.std, settings.scale);
      var sum = 0.0, sq = 0.0;
      for (final v in input) {
        sum += v;
        sq += v * v;
      }
      final mean = sum / input.length;
      final std = (sq / input.length - mean * mean).clamp(0, double.infinity).toDouble();
      final want = (photo['steps'] as Map)['snake_check'] as Map;
      final d = [(mean - (want['input_mean'] as num)).abs(), (math.sqrt(std) - (want['input_std'] as num)).abs()];
      for (final v in d) {
        if (v > worst) worst = v;
      }
      expect(d.every((v) => v <= _inputTolerance), isTrue, reason: '${photo['photo_id']}: $d');
    }
    // ignore: avoid_print
    print('largest snake-check input mean/std difference $worst');
  }, skip: dir == null);
}
