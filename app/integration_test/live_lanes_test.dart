import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:serpentguard/inference/image_ops.dart';
import 'package:serpentguard/inference/pipeline.dart';

/// On a phone or emulator, with the model files in assets/models:
///
/// 1. Parity: the live camera's native lanes (LiveModels.kt) give the same
///    outputs as flutter_onnxruntime, which photo mode uses, for the same
///    input tensors. The preprocessing is shared Dart code, so equal model
///    outputs mean equal answers.
/// 2. Timing: each model alone, and the box lane while the verdict lane runs
///    (they share the phone's cores).
///
/// Run: `flutter test integration_test/live_lanes_test.dart -d DEVICE`
void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();
  const detector = MethodChannel('serpentguard/live_detector');
  const verdict = MethodChannel('serpentguard/live_verdict');

  testWidgets('native lanes match flutter_onnxruntime, and how fast they are', (tester) async {
    final (settings, _) = await SnakePipeline.loadConfig();
    final size = settings.inputSize;
    final paths = {for (final f in modelFiles) f: await SnakePipeline.extractModel(f)};

    // Inputs: the two bundled images through the real preprocessing, and noise.
    final inputs = <String, (Float32List, Float32List)>{};
    for (final asset in ['assets/images/splash_poster.jpg', 'assets/images/app_logo.png']) {
      final rgb = await SnakePipeline.decode((await rootBundle.load(asset)).buffer.asUint8List());
      final fixed = fixExposure(rgb, settings.exposureBelow, settings.exposureAbove, settings.claheClip, settings.claheTiles);
      inputs[asset] = (
        detectorInput(fixed, size, settings.detectorPad),
        normalise(letterbox(fixed, size, settings.padRgb), settings.mean, settings.std, settings.scale),
      );
    }
    final rnd = math.Random(1);
    final noise = Float32List.fromList(List.generate(3 * size * size, (_) => rnd.nextDouble() * 2 - 1));
    inputs['noise'] = (noise, noise);

    final cores = (await detector.invokeMethod<int>('cores'))!;
    final boxThreads = cores >= 8 ? 2 : 1;
    final verdictThreads = (cores - boxThreads - 1).clamp(2, 4);
    await detector.invokeMethod('open', {'path': paths['detector.onnx'], 'threads': boxThreads});
    await verdict.invokeMethod('open', {
      'segmenter': paths['segmenter.onnx'],
      'classifier': paths['classifier.onnx'],
      'threads': verdictThreads,
    });

    // The photo path's sessions, with its options.
    final ort = OnnxRuntime();
    final options = OrtSessionOptions(intraOpNumThreads: 2, interOpNumThreads: 1);
    final plugin = {for (final f in modelFiles) f: await ort.createSession(paths[f]!, options: options)};

    Future<Map<String, List<double>>> viaPlugin(String model, Float32List input) async {
      final value = await OrtValue.fromList(input, [1, 3, size, size]);
      final out = await plugin[model]!.run({'images': value});
      final result = {
        for (final e in out.entries) e.key: [for (final v in await e.value.asFlattenedList()) (v as num).toDouble()]
      };
      for (final o in out.values) {
        await o.dispose();
      }
      await value.dispose();
      return result;
    }

    double gap(List<double> a, List<double> b) {
      expect(a.length, b.length);
      var worst = 0.0;
      for (var i = 0; i < a.length; i++) {
        worst = math.max(worst, (a[i] - b[i]).abs());
      }
      return worst;
    }

    final report = StringBuffer('cores $cores, threads box $boxThreads + verdict $verdictThreads\n');
    var worstGap = 0.0;
    for (final MapEntry(key: name, value: (det, img)) in inputs.entries) {
      final d = gap((await detector.invokeMethod<Float32List>('run', {'input': det, 'size': size}))!,
          (await viaPlugin('detector.onnx', det))['output0']!);
      final s = gap((await verdict.invokeMethod<Float32List>('segment', {'input': img, 'size': size}))!,
          (await viaPlugin('segmenter.onnx', img))['probability']!);
      final native = (await verdict.invokeMapMethod<String, Float32List>('classify', {'input': img, 'size': size}))!;
      final viaP = await viaPlugin('classifier.onnx', img);
      var c = 0.0;
      for (final k in viaP.keys) {
        c = math.max(c, gap(native[k]!, viaP[k]!));
      }
      report.writeln('parity $name: detector $d, segmenter $s, classifier $c (max abs difference)');
      worstGap = math.max(worstGap, math.max(d, math.max(s, c)));
    }

    Future<double> time(Future<void> Function() run, {int n = 10}) async {
      await run();
      final clock = Stopwatch()..start();
      for (var i = 0; i < n; i++) {
        await run();
      }
      return clock.elapsedMilliseconds / n;
    }

    final (det, img) = inputs['assets/images/splash_poster.jpg']!;
    Future<void> box() => detector.invokeMethod('run', {'input': det, 'size': size});
    Future<void> verdictStep() async {
      await verdict.invokeMethod('segment', {'input': img, 'size': size});
      await verdict.invokeMethod('classify', {'input': img, 'size': size});
    }

    final boxAlone = await time(box);
    final verdictAlone = await time(verdictStep);
    final pluginSerial = await time(() async {
      await viaPlugin('detector.onnx', det);
      await viaPlugin('segmenter.onnx', img);
      await viaPlugin('classifier.onnx', img);
    });
    // Both lanes at once: the verdict lane runs continuously while we time the box lane.
    var running = true, verdicts = 0;
    final verdictClock = Stopwatch()..start();
    final background = () async {
      while (running) {
        await verdictStep();
        verdicts++;
      }
    }();
    final boxBusy = await time(box, n: 30);
    running = false;
    await background;
    final verdictBusy = verdictClock.elapsedMilliseconds / math.max(1, verdicts);
    report
      ..writeln('box lane alone ${boxAlone.toStringAsFixed(1)} ms; verdict lane alone ${verdictAlone.toStringAsFixed(1)} ms')
      ..writeln('both at once: box ${boxBusy.toStringAsFixed(1)} ms, verdict ${verdictBusy.toStringAsFixed(1)} ms')
      ..writeln('old serial path (plugin, 3 models) ${pluginSerial.toStringAsFixed(1)} ms');
    // ignore: avoid_print
    print(report);

    for (final s in plugin.values) {
      await s.close();
    }
    await detector.invokeMethod('close');
    await verdict.invokeMethod('close');
    // ONNX Runtime's CPU kernels may split work differently with other thread
    // counts; anything beyond float rounding would show up here.
    expect(worstGap, lessThan(1e-4));
  });
}
