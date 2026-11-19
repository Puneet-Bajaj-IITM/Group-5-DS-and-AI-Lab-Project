import 'dart:convert';
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:serpentguard/inference/bundle.dart';
import 'package:serpentguard/inference/decision.dart';

/// The model bundle's decision fixtures: 200 saved model outputs and the
/// verdict each must give, covering every verdict/reason/fallback combination.
void main() {
  final settings = PipelineSettings.parse(File('assets/model/pipeline.json').readAsStringSync());
  final tables = ModelTables.parse(File('assets/model/tables.json').readAsStringSync());
  final golden = jsonDecode(File('test/golden/golden_decisions.json').readAsStringSync())
      as Map<String, dynamic>;
  final rules = DecisionRules(settings.thresholds, tables);

  test('thresholds in pipeline.json match the fixtures', () {
    final t = golden['thresholds'] as Map<String, dynamic>;
    expect(settings.thresholds.thetaConf, t['theta_conf']);
    expect(settings.thresholds.temperature, t['temperature']);
    expect(settings.thresholds.tauUnknown, t['tau_unknown']);
  });

  test('roundHalfEven and percent match Python', () {
    expect([0.5, 1.5, 2.5, 2.4999, 2.5001, -0.5].map(roundHalfEven), [0, 2, 2, 2, 3, 0]);
    expect(pyPercent(0.115), '12%');
    expect(pyPercent(0.125), '12%');
  });

  final rows = (golden['rows'] as List).cast<Map<String, dynamic>>();
  test('all ${rows.length} golden decisions match', () {
    var failures = <String>[];
    for (final row in rows) {
      final got = rules.verdictFromOutputs(ModelOutputs.fromJson(row['outputs'] as Map<String, dynamic>));
      final want = row['expected'] as Map<String, dynamic>;
      final problems = <String>[];
      if (got.label != want['verdict']) problems.add('verdict ${got.label} != ${want['verdict']}');
      if (got.message != want['message']) problems.add('message "${got.message}" != "${want['message']}"');
      if (!const ListEquality().equals(got.reasons, (want['reason'] as List).cast<String>())) {
        problems.add('reasons ${got.reasons} != ${want['reason']}');
      }
      final top = (want['top3'] as List).cast<Map<String, dynamic>>();
      for (var i = 0; i < top.length; i++) {
        if (tables.species[got.top3[i].classId] != top[i]['species']) {
          problems.add('top3[$i] ${tables.species[got.top3[i].classId]} != ${top[i]['species']}');
        }
        if ((got.top3[i].probability - (top[i]['probability'] as num)).abs() > 1e-6) {
          problems.add('top3[$i] probability ${got.top3[i].probability} != ${top[i]['probability']}');
        }
      }
      final fb = want['genus_fallback'] as Map<String, dynamic>?;
      if ((fb == null) != (got.genusFallback == null) ||
          (fb != null &&
              (fb['genus'] != got.genusFallback!.genus ||
                  fb['who_listed'] != got.genusFallback!.whoListed ||
                  ((fb['probability'] as num) - got.genusFallback!.probability).abs() > 1e-6))) {
        problems.add('genus_fallback ${got.genusFallback?.genus} != $fb');
      }
      if (want['margin'] != null && ((want['margin'] as num) - got.margin!).abs() > 1e-6) {
        problems.add('margin ${got.margin} != ${want['margin']}');
      }
      if (problems.isNotEmpty) failures.add('row ${row['row']}: ${problems.join('; ')}');
    }
    expect(failures, isEmpty, reason: failures.take(10).join('\n'));
  });

  test('a control with theta_conf moved fails some rows, so the check can fail', () {
    final t = settings.thresholds;
    final moved = DecisionRules(
      DecisionThresholds(
        temperature: t.temperature,
        temperatureWho: t.temperatureWho,
        thetaD: t.thetaD,
        thetaWho: t.thetaWho,
        tauUnknown: t.tauUnknown,
        thetaConf: t.thetaConf + 0.05,
        thetaV: t.thetaV,
        thetaGenus: t.thetaGenus,
      ),
      tables,
    );
    final differ = rows.where((row) {
      final got = moved.verdictFromOutputs(ModelOutputs.fromJson(row['outputs'] as Map<String, dynamic>));
      final want = row['expected'] as Map<String, dynamic>;
      return got.label != want['verdict'] || got.message != want['message'];
    }).length;
    expect(differ, greaterThan(0));
  });
}

class ListEquality {
  const ListEquality();
  bool equals(List<String> a, List<String> b) =>
      a.length == b.length && List.generate(a.length, (i) => a[i] == b[i]).every((x) => x);
}
