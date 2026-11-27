import 'dart:convert';
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:serpentguard/inference/bundle.dart';
import 'package:serpentguard/inference/decision.dart';
import 'package:serpentguard/inference/flip_prior.dart';

/// The M5 decision fixtures (tool/m5/make_golden_m5.py): 200 saved M5_KD
/// validation outputs, plain and mirrored, and the verdict each must give once
/// mirror-averaged and prior-corrected, with the app's thresholds and with the
/// frozen ones; every verdict/reason/fallback combination is covered.
void main() {
  final settings = PipelineSettings.parse(File('assets/model/pipeline.json').readAsStringSync());
  final tables = ModelTables.parse(File('assets/model/tables.json').readAsStringSync());
  final golden = jsonDecode(File('test/golden/golden_decisions.json').readAsStringSync())
      as Map<String, dynamic>;
  final rules = DecisionRules(settings.thresholds, tables);
  final frozenRules = DecisionRules(settings.frozenThresholds!, tables);

  ModelOutputs answer(Map<String, dynamic> row) => classifierAnswer(
        ModelOutputs.fromJson(row['outputs'] as Map<String, dynamic>),
        ModelOutputs.fromJson(row['outputs_flip'] as Map<String, dynamic>),
        settings.classifier,
      );

  void sameThresholds(DecisionThresholds got, Map<String, dynamic> want) {
    final t = DecisionThresholds.fromJson(want);
    expect([got.temperature, got.temperatureWho, got.thetaD, got.thetaWho, got.tauUnknown, got.thetaConf, got.thetaV, got.thetaGenus],
        [t.temperature, t.temperatureWho, t.thetaD, t.thetaWho, t.tauUnknown, t.thetaConf, t.thetaV, t.thetaGenus]);
  }

  test('thresholds and prior in pipeline.json match the fixtures', () {
    sameThresholds(settings.thresholds, golden['thresholds'] as Map<String, dynamic>);
    sameThresholds(settings.frozenThresholds!, golden['frozen_thresholds'] as Map<String, dynamic>);
    expect(settings.frozenThresholds!.thetaWho, double.infinity);
    expect(settings.classifier.flipTta, isTrue);
    expect(settings.classifier.priorTau, golden['prior_tau']);
    expect(settings.classifier.logCounts.length, tables.numClasses);
  });

  test('roundHalfEven and percent match Python', () {
    expect([0.5, 1.5, 2.5, 2.4999, 2.5001, -0.5].map(roundHalfEven), [0, 2, 2, 2, 3, 0]);
    expect(pyPercent(0.115), '12%');
    expect(pyPercent(0.125), '12%');
  });

  final rows = (golden['rows'] as List).cast<Map<String, dynamic>>();
  List<String> mismatches(DecisionRules rules, String key, ModelOutputs Function(Map<String, dynamic>) outputs) {
    var failures = <String>[];
    for (final row in rows) {
      final got = rules.verdictFromOutputs(outputs(row));
      final want = row[key] as Map<String, dynamic>;
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
    return failures;
  }

  test('all ${rows.length} golden decisions match with the app thresholds', () {
    final failures = mismatches(rules, 'expected', answer);
    expect(failures, isEmpty, reason: failures.take(10).join('\n'));
  });

  test('all ${rows.length} golden decisions match with the frozen thresholds', () {
    final failures = mismatches(frozenRules, 'expected_frozen', answer);
    expect(failures, isEmpty, reason: failures.take(10).join('\n'));
    expect(golden['rows_where_frozen_differs'], greaterThan(0));
  });

  test('controls: without the mirror pass or without the prior, some rows fail', () {
    ModelOutputs plainOnly(Map<String, dynamic> row) =>
        classifierAnswer(ModelOutputs.fromJson(row['outputs'] as Map<String, dynamic>), null,
            ClassifierSettings(flipTta: false, priorTau: settings.classifier.priorTau, logCounts: settings.classifier.logCounts));
    ModelOutputs noPrior(Map<String, dynamic> row) => classifierAnswer(
        ModelOutputs.fromJson(row['outputs'] as Map<String, dynamic>),
        ModelOutputs.fromJson(row['outputs_flip'] as Map<String, dynamic>),
        const ClassifierSettings(flipTta: true, priorTau: 0, logCounts: []));
    expect(mismatches(rules, 'expected', plainOnly), isNotEmpty);
    expect(mismatches(rules, 'expected', noPrior), isNotEmpty);
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
      final got = moved.verdictFromOutputs(answer(row));
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
