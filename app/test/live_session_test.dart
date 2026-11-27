import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:serpentguard/inference/bundle.dart';
import 'package:serpentguard/inference/decision.dart';
import 'package:serpentguard/inference/live_session.dart';
import 'package:serpentguard/models/verdict.dart';
import 'package:serpentguard/services/classifier.dart';

import 'helpers.dart';

const anchors = 2100;
const outputKeys = ['embedding', 'species_logits', 'who_logits', 'genus_logits', 'family_logits'];

/// The live session's two lanes against mocked native models: the box lane's
/// gate shortcut, the verdict matching the direct path, frame dropping and
/// stale results after a reset.
void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  final messenger = TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger;
  final settings = PipelineSettings.parse(File('assets/model/pipeline.json').readAsStringSync());
  final rules = loadRules();
  final rule = DecisionRule(catalog: loadCatalog(), rules: rules);
  final golden = jsonDecode(File('test/golden/golden_decisions.json').readAsStringSync()) as Map<String, dynamic>;
  final rows = (golden['rows'] as List).cast<Map<String, dynamic>>();
  final dangerousRow = rows.firstWhere((r) => (r['expected'] as Map)['verdict'] == 'dangerous');

  const detector = MethodChannel('test/live_detector');
  const verdict = MethodChannel('test/live_verdict');

  late double score; // the detector's snake score for the next frames
  late int detectorCalls, segmentCalls, classifyCalls;
  Completer<void>? holdSegment;

  /// The golden row's outputs as the native lane returns them (float32).
  Map<String, Float32List> classifierOutputs(Map<String, dynamic> row) => {
        for (final k in outputKeys)
          k: Float32List.fromList(((row['outputs'] as Map)[k] as List).map((v) => (v as num).toDouble()).toList()),
      };

  setUp(() {
    score = 0.9;
    detectorCalls = segmentCalls = classifyCalls = 0;
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
      if (call.method == 'segment') {
        segmentCalls++;
        await holdSegment?.future;
        return Float32List(320 * 320)..fillRange(0, 320 * 320, 1.0); // the whole frame is snake
      }
      classifyCalls++;
      return classifierOutputs(dangerousRow);
    });
  });

  LiveSession session() => LiveSession.withParts(
        settings: settings,
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

  test('a frame under the gate box cutoff never reaches the verdict lane, and shows no snake', () async {
    score = settings.gateBoxConf / 2;
    final live = session();
    live.offer(frame(0));
    await pumpEventQueue();
    expect(detectorCalls, 1);
    expect(segmentCalls, 0);
    expect(classifyCalls, 0);
    expect(live.view!.answered, isTrue);
    expect(live.view!.identification!.snakeFound, isFalse);
    expect(live.view!.skippedByGate, 1);
  });

  test('the verdict lane gives the direct path\'s answer for the same outputs', () async {
    final live = session();
    live.offer(frame(0));
    await pumpEventQueue();
    expect(segmentCalls, 1);
    expect(classifyCalls, 1);
    final direct = rules.verdictFromOutputs(ModelOutputs(
      embedding: classifierOutputs(dangerousRow)['embedding']!,
      speciesLogits: classifierOutputs(dangerousRow)['species_logits']!,
      whoLogits: classifierOutputs(dangerousRow)['who_logits']!,
      genusLogits: classifierOutputs(dangerousRow)['genus_logits']!,
      familyLogits: classifierOutputs(dangerousRow)['family_logits']!,
    ));
    expect(direct.label, dangerousLabel);
    // "Dangerous" shows while the lock is still checking, as in identifyFrame.
    final shown = live.view!.identification!;
    expect(shown.verdict, Verdict.dangerous);
    expect(shown.reason, rule.fromAnswer(direct).reason);
    expect(live.view!.snakeInView, isTrue);
    expect(live.view!.mainTrack, isNotNull);
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
    expect(classifyCalls, 1);
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
    expect(classifyCalls, 1);
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
