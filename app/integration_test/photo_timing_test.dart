import 'dart:convert';
import 'dart:io';

import 'package:crypto/crypto.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:serpentguard/inference/bundle.dart';
import 'package:serpentguard/inference/pipeline.dart';

/// The M5 report's phone numbers (section 7.3), on the device it runs on:
/// time per photo through the whole photo path with and without the mirror
/// pass, each stage's time, and the process's peak memory (VmHWM).
///
/// It prints the SHA-256 of every model it ran, so a number can always be
/// traced to its files: only figures from the frozen M5 files (pipeline.json
/// `model_sha256`, plus the M5_KD classifier) may go in the report. Speed and
/// memory depend on the architecture, not the weights, but say which files
/// were timed.
///
/// Run: `flutter test integration_test/photo_timing_test.dart -d DEVICE`
/// (use a release-like device build for report numbers: `--profile` is not
/// supported by integration tests, so note that the figures are from a debug
/// test build; the models run in native ONNX Runtime either way).
void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  const photosPerRun = 10;
  const warmUp = 2;

  int? peakKb() {
    try {
      final line = File('/proc/self/status').readAsLinesSync().firstWhere((l) => l.startsWith('VmHWM:'));
      return int.parse(RegExp(r'(\d+)').firstMatch(line)!.group(1)!);
    } catch (_) {
      return null; // not Linux/Android
    }
  }

  testWidgets('time per photo with and without the mirror pass, and peak memory', (tester) async {
    final (settings, tables) = await SnakePipeline.loadConfig();
    final report = StringBuffer();

    report.writeln('models:');
    for (final f in modelFiles) {
      final data = await rootBundle.load('$modelAssetDir/$f');
      final digest = sha256.convert(data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes)).toString();
      final pinned = settings.modelSha256[f];
      final note = pinned == null ? 'not pinned' : (pinned == digest ? 'matches pipeline.json' : 'DIFFERS from pipeline.json');
      report.writeln('  $f ${data.lengthInBytes} bytes sha256 $digest ($note)');
    }

    // The same settings with the mirror pass off.
    final raw = jsonDecode(jsonEncode(settings.raw)) as Map<String, dynamic>;
    (raw['classifier'] as Map<String, dynamic>)['flip_tta'] = false;
    final noMirror = PipelineSettings(raw);

    final photos = [
      for (final asset in ['assets/images/splash_poster.jpg', 'assets/images/app_logo.png'])
        (asset, await SnakePipeline.decode(Uint8List.sublistView(await rootBundle.load(asset)))),
    ];
    final beforeKb = peakKb();

    for (final (label, s) in [('with mirror pass', settings), ('without mirror pass', noMirror)]) {
      final pipeline = await SnakePipeline.load(s, tables);
      try {
        for (final (asset, rgb) in photos) {
          for (var i = 0; i < warmUp; i++) {
            await pipeline.run(rgb);
          }
          final totals = <int>[];
          final stages = <String, List<int>>{};
          var snake = false;
          for (var i = 0; i < photosPerRun; i++) {
            final clock = Stopwatch()..start();
            final r = await pipeline.run(rgb);
            totals.add(clock.elapsedMilliseconds);
            snake = r.gate.snake;
            for (final e in r.timingsMs.entries) {
              (stages[e.key] ??= []).add(e.value);
            }
          }
          int median(List<int> v) => (List.of(v)..sort())[v.length ~/ 2];
          report.writeln('$label, $asset (${rgb.width}x${rgb.height}, snake: $snake): '
              'median ${median(totals)} ms, min ${totals.reduce((a, b) => a < b ? a : b)} ms over $photosPerRun; '
              'stages ${{for (final e in stages.entries) e.key: median(e.value)}}');
        }
      } finally {
        await pipeline.close();
      }
    }
    final afterKb = peakKb();
    report.writeln('peak memory (VmHWM): before models ${beforeKb ?? '?'} kB, after both runs ${afterKb ?? '?'} kB');
    // One line per print: Android's log cuts long messages.
    for (final line in report.toString().trimRight().split(String.fromCharCode(10))) {
      // ignore: avoid_print
      print(line);
    }
    expect(afterKb == null || afterKb > 0, isTrue);
  });
}
