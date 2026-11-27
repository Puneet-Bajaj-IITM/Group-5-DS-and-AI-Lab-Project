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
  late double snakeProbability; // the snake check's answer for the next frames
  late int detectorCalls, checkCalls, segmentCalls, classifyCalls;
  Completer<void>? holdSegment;

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
    snakeProbability = 0.9;
    detectorCalls = checkCalls = segmentCalls = classifyCalls = 0;
    holdSegment = null;
    messenger.setMockMethodCallHandler(detector, (call) async {
      detectorCalls++;
      final out = Float32List(5 * anchors);
      out[0] = 160;
      out[anchors] = 160;
      out[2 * anchors] = 120;
      out[3 * anchors] = 80;
      out[4 * anchors] = score;
      return out;
    });
    messenger.setMockMethodCallHandler(verdict, (call) async {
      switch (call.method) {
        case 'check':
          checkCalls++;
          expect((call.arguments as Map)['size'], check.inputSize);
          final probs = Float32List(2);
          probs[check.snakeIndex] = snakeProbability;
          probs[1 - check.snakeIndex] = 1 - snakeProbability;
          return {check.output: probs};
        case 'segment':
          segmentCalls++;
          await holdSegment?.future;
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
    expect(segmentCalls, 1);
    expect(classifyCalls, classifyPerFrame);
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

  test('the verdict lane gives the direct path\'s answer: mirror pass and prior included', () async {
    final live = session();
    live.offer(frame(0));
    await pumpEventQueue();
    expect(segmentCalls, 1);
    expect(classifyCalls, classifyPerFrame, reason: 'the crop and its mirror image');
    final out = asOutputs(classifierOutputs(dangerousRow));
    final direct = rules.verdictFromOutputs(
        classifierAnswer(out, settings.classifier.flipTta ? out : null, settings.classifier));
    expect(direct.label, dangerousLabel);
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
    holdSegment = Completer<void>();
    final live = session();
    for (var i = 0; i < 6; i++) {
      live.offer(frame(i * 33));
      await pumpEventQueue();
    }
    expect(segmentCalls, 1, reason: 'one verdict at a time');
    expect(detectorCalls, 6, reason: 'the box lane kept up');
    holdSegment!.complete();
    await pumpEventQueue();
    expect(classifyCalls, classifyPerFrame);
    live.offer(frame(300));
    await pumpEventQueue();
    expect(segmentCalls, 2, reason: 'the lane is free again');
  });

  test('a verdict still running at a reset is dropped, and the lane works again', () async {
    holdSegment = Completer<void>();
    final live = session();
    live.offer(frame(0));
    await pumpEventQueue();
    live.reset();
    holdSegment!.complete();
    await pumpEventQueue();
    expect(classifyCalls, 0, reason: 'the old frame stops after the reset');
    holdSegment = null;
    live.offer(frame(100));
    await pumpEventQueue();
    expect(segmentCalls, 2);
    expect(classifyCalls, classifyPerFrame);
  });

  test('a failing lane stops showing the last verdict and reports the error', () async {
    final live = session();
    final errors = <Object>[];
    live.views.listen((_) {}, onError: errors.add);
    live.offer(frame(0));
    await pumpEventQueue();
    expect(live.view!.identification, isNotNull);
    messenger.setMockMethodCallHandler(verdict, (call) async => throw PlatformException(code: 'live_models'));
    live.offer(frame(100));
    await pumpEventQueue();
    expect(errors, isNotEmpty);
    expect(live.view!.identification, isNull, reason: 'no stale verdict while analysis fails');
    expect(live.view!.answered, isFalse);
  });

  test('with the M4 gate, a frame turned down on the box score clears the main box', () async {
    final live = session(m4Settings);
    live.offer(frame(0));
    await pumpEventQueue();
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
