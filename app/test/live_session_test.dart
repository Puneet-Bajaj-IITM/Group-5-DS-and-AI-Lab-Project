import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:serpentguard/inference/bundle.dart';
import 'package:serpentguard/inference/decision.dart';
import 'package:serpentguard/inference/flip_prior.dart';
import 'package:serpentguard/inference/live_session.dart';
import 'package:serpentguard/models/verdict.dart';
import 'package:serpentguard/services/classifier.dart';

import 'helpers.dart';

const anchors = 2100;
const outputKeys = ['embedding', 'species_logits', 'who_logits', 'genus_logits', 'family_logits'];

/// The live session's two lanes against mocked native models: the M5 snake
/// check deciding in the verdict lane, the mirror pass and prior, the verdict
/// matching the direct path, the M4 gate's box shortcut, frame dropping and
/// stale results after a reset.
void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  final messenger = TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger;
  final json = File('assets/model/pipeline.json').readAsStringSync();
  final settings = PipelineSettings.parse(json);
  // The same bundle with the M4 gate (detector box and U-Net mask) instead of
  // the snake check, for the box-cutoff shortcut.
  final m4Json = jsonDecode(json) as Map<String, dynamic>;
  (m4Json['gate'] as Map<String, dynamic>)['rule'] = 'both';
  final m4Settings = PipelineSettings.parse(jsonEncode(m4Json));
  final rules = loadRules();
  final rule = DecisionRule(catalog: loadCatalog(), rules: rules);
  final golden = jsonDecode(File('test/golden/golden_decisions.json').readAsStringSync()) as Map<String, dynamic>;
  final rows = (golden['rows'] as List).cast<Map<String, dynamic>>();
  final dangerousRow = rows.firstWhere((r) => (r['expected'] as Map)['verdict'] == 'dangerous');
  final check = settings.snakeCheck!;
  final classifyPerFrame = settings.classifier.flipTta ? 2 : 1;

  const detector = MethodChannel('test/live_detector');
  const verdict = MethodChannel('test/live_verdict');

  late double score; // the detector's snake score for the next frames
  late double boxCy, boxH; // the box's centre y and height (letterboxed 320 px)
  late double snakeProbability; // the snake check's answer for the next frames
  // With the snake check the live verdict is the lean path: box crop, one classify.
  const leanClassifyPerFrame = 1;
  late int detectorCalls, checkCalls, segmentCalls, classifyCalls;
  Completer<void>? holdVerdict;
  // Two snakes side by side (left one more confident) instead of one box.
  late bool twoBoxes;
  // The snake check says snake only on a close crop of the bright left half
  // of [splitFrame], as on the checkerboard: far under the cutoff on the whole
  // frame, near or over it on a tight crop.
  late bool snakeOnlyInCrop;
  // Two snakes side by side as the 320 px detector sees them: one unsure box
  // spanning both on the whole frame, one sure snake in each half.
  late bool sideBySide;

  /// The golden row's outputs as the native lane returns them (float32).
  Map<String, Float32List> classifierOutputs(Map<String, dynamic> row) => {
        for (final k in outputKeys)
          k: Float32List.fromList(((row['outputs'] as Map)[k] as List).map((v) => (v as num).toDouble()).toList()),
      };

  ModelOutputs asOutputs(Map<String, Float32List> o) => ModelOutputs(
        embedding: o['embedding']!,
        speciesLogits: o['species_logits']!,
        whoLogits: o['who_logits']!,
        genusLogits: o['genus_logits']!,
        familyLogits: o['family_logits']!,
      );

  setUp(() {
    score = 0.9;
    boxCy = 160;
    boxH = 80;
    snakeProbability = 0.9;
    detectorCalls = checkCalls = segmentCalls = classifyCalls = 0;
    holdVerdict = null;
    twoBoxes = snakeOnlyInCrop = sideBySide = false;
    messenger.setMockMethodCallHandler(detector, (call) async {
      detectorCalls++;
      final out = Float32List(5 * anchors);
      if (sideBySide) {
        // Calls come in threes: the whole frame, then its two halves.
        final (cx, w, s) = [(160.0, 300.0, 0.3), (100.0, 60.0, 0.8), (220.0, 60.0, 0.8)][(detectorCalls - 1) % 3];
        out[0] = cx;
        out[anchors] = 160;
        out[2 * anchors] = w;
        out[3 * anchors] = 200;
        out[4 * anchors] = s;
        return out;
      }
      if (twoBoxes) {
        for (final (i, cx, s) in [(0, 80.0, 0.9), (1, 240.0, 0.8)]) {
          out[i] = cx;
          out[anchors + i] = 160;
          out[2 * anchors + i] = 140;
          out[3 * anchors + i] = 200;
          out[4 * anchors + i] = s;
        }
        return out;
      }
      out[0] = 160;
      out[anchors] = boxCy;
      out[2 * anchors] = 120;
      out[3 * anchors] = boxH;
      out[4 * anchors] = score;
      return out;
    });
    messenger.setMockMethodCallHandler(verdict, (call) async {
      switch (call.method) {
        case 'check':
          checkCalls++;
          await holdVerdict?.future;
          expect((call.arguments as Map)['size'], check.inputSize);
          final probs = Float32List(2);
          var p = snakeProbability;
          if (snakeOnlyInCrop) {
            final input = (call.arguments as Map)['input'] as Float32List;
            final mean = input.fold<double>(0, (a, v) => a + v) / input.length;
            p = mean > 1 ? check.cutoff + 0.05 : check.cutoff / 5; // bright crop vs whole frame
          }
          probs[check.snakeIndex] = p;
          probs[1 - check.snakeIndex] = 1 - snakeProbability;
          return {check.output: probs};
        case 'segment':
          segmentCalls++;
          await holdVerdict?.future; // the first verdict-lane call with the M4 gate
          return Float32List(320 * 320)..fillRange(0, 320 * 320, 1.0); // the whole frame is snake
        default:
          classifyCalls++;
          return classifierOutputs(dangerousRow);
      }
    });
  });

  LiveSession session([PipelineSettings? s]) => LiveSession.withParts(
        settings: s ?? settings,
        rules: rules,
        present: (a, details) => rule.fromAnswer(a, details: details),
        detector: detector,
        verdict: verdict,
      );

  LiveCameraFrame frame(int ms) => LiveCameraFrame(
        width: 64,
        height: 48,
        format: 'bgra8888',
        planes: [LiveCameraPlane(Uint8List(64 * 48 * 4)..fillRange(0, 64 * 48 * 4, 120), 64 * 4, 4)],
        offsetMs: ms,
        sensorOrientation: 0,
        deviceOrientation: 'portraitUp',
        frontFacing: false,
      );

  /// A frame whose left half is bright and right half dark.
  LiveCameraFrame splitFrame(int ms) {
    final bytes = Uint8List(64 * 48 * 4);
    for (var y = 0; y < 48; y++) {
      for (var x = 0; x < 64; x++) {
        bytes.fillRange((y * 64 + x) * 4, (y * 64 + x) * 4 + 4, x < 32 ? 250 : 10);
      }
    }
    return LiveCameraFrame(
      width: 64,
      height: 48,
      format: 'bgra8888',
      planes: [LiveCameraPlane(bytes, 64 * 4, 4)],
      offsetMs: ms,
      sensorOrientation: 0,
      deviceOrientation: 'portraitUp',
      frontFacing: false,
    );
  }

  test('with several snakes in view, no box is named when the whole frame fails the snake check', () async {
    twoBoxes = snakeOnlyInCrop = true;
    final live = session();
    for (var i = 0; i < 4; i++) {
      live.offer(splitFrame(i * 100));
      await pumpEventQueue();
    }
    expect(live.view!.tracks, hasLength(2), reason: 'both boxes are tracked');
    expect(checkCalls, greaterThan(0));
    expect(classifyCalls, 0, reason: 'a close crop alone never names a snake, as on a photo');
    expect(live.view!.names, isEmpty);
    expect(live.view!.identification!.snakeFound, isFalse);
  });

  test('two snakes the detector sees as one unsure box are boxed one by one after a look at each half', () async {
    sideBySide = true;
    final live = session();
    live.offer(frame(0));
    await pumpEventQueue();
    expect(detectorCalls, 3, reason: 'the frame, then its two halves');
    expect(live.view!.tracks, hasLength(2));
    final lefts = [for (final t in live.view!.tracks) t.box.left]..sort();
    expect(lefts.first, lessThan(32));
    expect(lefts.last, greaterThan(32));
  });

  test('with several snakes in view and the whole frame a snake, each box is named', () async {
    twoBoxes = true;
    final live = session();
    for (var i = 0; i < 4; i++) {
      live.offer(splitFrame(i * 100));
      await pumpEventQueue();
    }
    expect(classifyCalls, greaterThan(0));
    expect(live.view!.names, isNotEmpty);
    expect(live.view!.identification!.verdict, Verdict.dangerous);
  });

  test('a snake check that passes only on single frames never shows a verdict (the striped-screen false alarm)', () async {
    final live = session();
    final shown = <Verdict?>[];
    for (var i = 0; i < 12; i++) {
      snakeProbability = i % 3 == 0 ? 0.9 : check.cutoff / 2; // one passing frame in three
      live.offer(frame(i * 100));
      await pumpEventQueue();
      shown.add(live.view!.identification?.snakeFound == true ? live.view!.identification!.verdict : null);
    }
    expect(classifyCalls, greaterThan(0), reason: 'the passing frames were classified');
    expect(shown.where((v) => v != null), isEmpty);
  });

  test('the shipped bundle uses the snake check and the mirror pass', () {
    expect(settings.usesSnakeCheck, isTrue);
    expect(settings.classifier.flipTta, isTrue);
    expect(m4Settings.usesSnakeCheck, isFalse);
  });

  test('with the snake check, a low detector score still reaches the verdict lane', () async {
    score = settings.gateBoxConf / 2;
    final live = session();
    live.offer(frame(0));
    await pumpEventQueue();
    expect(checkCalls, 1, reason: 'the snake check decides, not the detector');
    expect(segmentCalls, 0, reason: 'lean: the detector box crop, no U-Net');
    expect(classifyCalls, leanClassifyPerFrame);
    live.offer(frame(100)); // a sighting starts on the second snake frame in a row
    await pumpEventQueue();
    expect(live.view!.skippedByGate, 0);
    expect(live.view!.identification!.verdict, Verdict.dangerous);
  });

  test('a frame the snake check turns down shows no snake and skips the crop and classifier', () async {
    snakeProbability = check.cutoff / 2;
    final live = session();
    live.offer(frame(0));
    await pumpEventQueue();
    expect(checkCalls, 1);
    expect(segmentCalls, 0);
    expect(classifyCalls, 0);
    expect(live.view!.answered, isTrue);
    expect(live.view!.identification!.snakeFound, isFalse);
    expect(live.view!.snakeInView, isFalse);
  });

  test('the verdict lane gives the lean path\'s answer: box crop and prior, no mirror pass', () async {
    final live = session();
    live.offer(frame(0));
    await pumpEventQueue();
    expect(segmentCalls, 0);
    expect(classifyCalls, leanClassifyPerFrame, reason: 'the box crop only, no mirror image');
    final out = asOutputs(classifierOutputs(dangerousRow));
    final direct = rules.verdictFromOutputs(classifierAnswer(out, null, settings.classifier.withoutMirror));
    expect(direct.label, dangerousLabel);
    live.offer(frame(100)); // a sighting starts on the second snake frame in a row
    await pumpEventQueue();
    // "Dangerous" shows while the lock is still checking, as in identifyFrame.
    final shown = live.view!.identification!;
    expect(shown.verdict, Verdict.dangerous);
    expect(shown.reason, rule.fromAnswer(direct).reason);
    expect(live.view!.snakeInView, isTrue);
    expect(live.view!.mainTrack, isNotNull);
  });

  test('with the M4 gate, a frame under the box cutoff never reaches the verdict lane', () async {
    score = m4Settings.gateBoxConf / 2;
    final live = session(m4Settings);
    live.offer(frame(0));
    await pumpEventQueue();
    expect(detectorCalls, 1);
    expect(checkCalls, 0);
    expect(segmentCalls, 0);
    expect(classifyCalls, 0);
    expect(live.view!.answered, isTrue);
    expect(live.view!.identification!.snakeFound, isFalse);
    expect(live.view!.skippedByGate, 1);
  });

  test('with the M4 gate there is no snake check call', () async {
    final live = session(m4Settings);
    live.offer(frame(0));
    await pumpEventQueue();
    expect(checkCalls, 0);
    expect(segmentCalls, 1);
    expect(classifyCalls, classifyPerFrame);
  });

  test('while the verdict lane works, newer frames feed only the boxes', () async {
    holdVerdict = Completer<void>();
    final live = session();
    for (var i = 0; i < 6; i++) {
      live.offer(frame(i * 33));
      await pumpEventQueue();
    }
    expect(checkCalls, 1, reason: 'one verdict at a time');
    expect(detectorCalls, 6, reason: 'the box lane kept up');
    holdVerdict!.complete();
    await pumpEventQueue();
    expect(classifyCalls, leanClassifyPerFrame);
    live.offer(frame(300));
    await pumpEventQueue();
    expect(checkCalls, 2, reason: 'the lane is free again');
  });

  test('a verdict still running at a reset is dropped, and the lane works again', () async {
    holdVerdict = Completer<void>();
    final live = session();
    live.offer(frame(0));
    await pumpEventQueue();
    live.reset();
    holdVerdict!.complete();
    await pumpEventQueue();
    expect(classifyCalls, 0, reason: 'the old frame stops after the reset');
    holdVerdict = null;
    live.offer(frame(100));
    await pumpEventQueue();
    expect(checkCalls, 2);
    expect(classifyCalls, leanClassifyPerFrame);
  });

  test('a failing lane keeps its error on every view and shows no stale verdict, until it works again', () async {
    final live = session();
    for (final ms in [0, 50]) {
      live.offer(frame(ms));
      await pumpEventQueue();
    }
    expect(live.view!.identification, isNotNull);
    expect(live.view!.error, isNull);
    final working = messenger;
    working.setMockMethodCallHandler(verdict, (call) async => throw PlatformException(code: 'live_models'));
    live.offer(frame(100));
    await pumpEventQueue();
    expect(live.view!.error, isNotNull, reason: 'the box lane publishes after the failure; the error stays');
    expect(live.view!.identification, isNull, reason: 'no stale verdict while analysis fails');
    expect(live.view!.answered, isFalse);
    live.offer(frame(200)); // the box lane works, the verdict lane still fails
    await pumpEventQueue();
    expect(live.view!.error, isNotNull);
    working.setMockMethodCallHandler(verdict, (call) async {
      switch (call.method) {
        case 'check':
          final probs = Float32List(2)..[check.snakeIndex] = 0.9;
          return {check.output: probs};
        case 'segment':
          return Float32List(320 * 320)..fillRange(0, 320 * 320, 1.0);
        default:
          return classifierOutputs(dangerousRow);
      }
    });
    live.offer(frame(300));
    await pumpEventQueue();
    expect(live.view!.error, isNull, reason: 'a good verdict clears it');
    expect(live.view!.identification, isNotNull);
  });

  test('close names its owner, so the native lanes ignore a close from an older session', () async {
    final closes = <Object?>[];
    messenger.setMockMethodCallHandler(detector, (call) async {
      if (call.method == 'close') closes.add(call.arguments);
      return null;
    });
    final live = LiveSession.withParts(
      settings: settings,
      rules: rules,
      present: (a, details) => rule.fromAnswer(a, details: details),
      detector: detector,
      verdict: verdict,
      owner: 7,
    );
    await live.close();
    expect(closes, [
      {'owner': 7}
    ]);
  });

  test('with the M4 gate, a frame turned down on the box score clears the main box', () async {
    final live = session(m4Settings);
    for (final ms in [0, 50]) {
      live.offer(frame(ms));
      await pumpEventQueue();
    }
    expect(live.view!.snakeInView, isTrue);
    score = m4Settings.gateBoxConf / 2; // the earlier track is still held, but no longer the snake
    live.offer(frame(100));
    await pumpEventQueue();
    // The box is no longer called the snake...
    expect(live.view!.snakeInView, isFalse);
    expect(live.view!.mainTrack, isNull);
    // ...while the lock still holds its answer for holdSeconds, as on a clip.
    expect(live.view!.identification!.verdict, Verdict.dangerous);
  });

  test('a best box clamped to nothing (in the letterbox padding) makes no track', () async {
    // A 64x48 frame sits between y 40 and 280 of the 320 px letterbox.
    boxCy = 10;
    boxH = 8;
    final live = session();
    live.offer(frame(0));
    await pumpEventQueue();
    live.offer(frame(33));
    await pumpEventQueue();
    expect(live.view!.tracks, isEmpty);
  });

  test('with the snake check, a confirmed snake with no detector box still reads "snake in view"', () async {
    score = settings.boxFloor / 2; // no box at all
    final live = session();
    live.offer(frame(0));
    await pumpEventQueue();
    expect(live.view!.mainTrack, isNull);
    expect(live.view!.snakeInView, isTrue);
  });

  test('after a failure, every view carries the error until the lane works', () async {
    final live = session();
    final views = <LiveView>[];
    live.views.listen(views.add);
    live.offer(frame(0));
    await pumpEventQueue();
    messenger.setMockMethodCallHandler(verdict, (call) async => throw PlatformException(code: 'live_models'));
    final failedFrom = views.length;
    for (var i = 1; i <= 3; i++) {
      live.offer(frame(i * 100));
      await pumpEventQueue();
    }
    final after = views.sublist(failedFrom);
    // The first view of the failing frame comes from the box lane, before the
    // verdict lane fails; every view after the failure carries it.
    final firstError = after.indexWhere((v) => v.error != null);
    expect(firstError, isNonNegative);
    expect(after.skip(firstError).every((v) => v.error != null), isTrue);
  });

  test('with the M4 gate, turned-down frames reach the lock only while the verdict lane is free', () async {
    holdVerdict = Completer<void>();
    final live = session(m4Settings);
    live.offer(frame(0)); // a snake frame: the verdict lane starts and waits
    await pumpEventQueue();
    score = m4Settings.gateBoxConf / 2;
    for (var i = 1; i <= 5; i++) {
      live.offer(frame(i * 33)); // turned down while the lane is busy: not counted
      await pumpEventQueue();
    }
    expect(live.view!.skippedByGate, 0);
    holdVerdict!.complete();
    await pumpEventQueue();
    live.offer(frame(300)); // the lane is free again: this one is counted
    await pumpEventQueue();
    expect(live.view!.skippedByGate, 1);
  });

  test('a reset clears a lane error', () async {
    final live = session();
    messenger.setMockMethodCallHandler(verdict, (call) async => throw PlatformException(code: 'live_models'));
    live.offer(frame(0));
    await pumpEventQueue();
    expect(live.view!.error, isNotNull);
    live.reset();
    final views = <LiveView>[];
    live.views.listen(views.add);
    live.offer(frame(100));
    await pumpEventQueue();
    // The box lane's view comes before the verdict lane fails again: no error
    // from before the reset is carried into it.
    expect(views.first.error, isNull);
  });

  test('with the M4 gate, a frame the gate answers alone clears a verdict-lane error', () async {
    final live = session(m4Settings);
    messenger.setMockMethodCallHandler(verdict, (call) async => throw PlatformException(code: 'live_models'));
    live.offer(frame(0));
    await pumpEventQueue();
    expect(live.view!.error, isNotNull);
    score = m4Settings.gateBoxConf / 2; // the snake is gone: the gate answers
    live.offer(frame(100));
    await pumpEventQueue();
    expect(live.view!.error, isNull);
  });

  test('boxes are tracked across frames', () async {
    final live = session();
    live.offer(frame(0));
    await pumpEventQueue();
    final id = live.view!.tracks.single.id;
    live.offer(frame(33));
    await pumpEventQueue();
    expect(live.view!.tracks.single.id, id);
  });
}
