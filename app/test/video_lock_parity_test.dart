import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:serpentguard/inference/bundle.dart';
import 'package:serpentguard/inference/decision.dart';
import 'package:serpentguard/inference/video_lock.dart';

/// VideoLock against `video.pooled_lock` + `video.lock_verdicts` on real
/// per-frame outputs of dev clips (test/golden/video_lock_parity.json, made
/// by results/m5/g5/code/make_parity.py): every step must match.
///
/// The fixture's frames are A3 outputs, made with A3's thresholds and
/// prototypes; those are kept in `video_lock_parity_a3.json`, so this tests
/// the lock itself, not the M5 settings the app now ships.
void main() {
  final a3 = jsonDecode(File('test/golden/video_lock_parity_a3.json').readAsStringSync()) as Map<String, dynamic>;
  final tablesJson = jsonDecode(File('assets/model/tables.json').readAsStringSync()) as Map<String, dynamic>;
  final tables = ModelTables.parse(jsonEncode({...tablesJson, 'prototypes': a3['prototypes']}));
  final thresholds = DecisionThresholds.fromJson(a3['thresholds'] as Map<String, dynamic>);
  final rules = DecisionRules(thresholds, tables);
  final golden = jsonDecode(File('test/golden/video_lock_parity.json').readAsStringSync()) as Map<String, dynamic>;

  test('the fixture used the A3 thresholds it is paired with', () {
    expect(golden['theta_conf'], thresholds.thetaConf);
  });

  Float64List vec(dynamic v) => Float64List.fromList([for (final x in v as List) (x as num).toDouble()]);

  for (final seq in (golden['sequences'] as List).cast<Map<String, dynamic>>()) {
    test('parity: ${seq['name']}', () {
      final s = seq['settings'] as Map<String, dynamic>;
      final lock = VideoLock(
        rules: rules,
        settings: LockSettings(
          windowSeconds: (s['window_seconds'] as num).toDouble(),
          margin: (s['margin'] as num).toDouble(),
          switchSeconds: (s['switch_seconds'] as num).toDouble(),
          gateShare: (s['gate_share'] as num).toDouble(),
          minFrames: (s['min_frames'] as num).toInt(),
          holdSeconds: (s['hold_seconds'] as num).toDouble(),
        ),
      );
      final times = (seq['times'] as List).map((t) => (t as num).toDouble()).toList();
      final frames = seq['frames'] as List;
      final expected = (seq['expected'] as List).cast<Map<String, dynamic>>();
      expect(frames.length, expected.length);
      final problems = <String>[];
      for (var i = 0; i < frames.length; i++) {
        final f = frames[i] as Map<String, dynamic>?;
        final parts = f == null
            ? null
            : DecisionParts(
                probs: vec(f['probs']),
                who: vec(f['who']),
                genus: vec(f['genus']),
                family: vec(f['family']),
                embedding: vec(f['embedding']),
              );
        final step = lock.add(parts, times[i]);
        final want = expected[i];
        final species = step.shown == null || step.shown!.top3.isEmpty ? null : step.shown!.top3.first.classId;
        final got = {
          'checking': step.checking,
          'locked': step.lockedClass,
          'event': step.event,
          'label': step.shown?.label,
          'species': species,
        };
        for (final k in got.keys) {
          if (got[k] != want[k]) problems.add('step $i t=${times[i]} $k: dart ${got[k]} python ${want[k]}');
        }
      }
      expect(problems.take(10).toList(), isEmpty, reason: '${problems.length} mismatches');
    });
  }
}
