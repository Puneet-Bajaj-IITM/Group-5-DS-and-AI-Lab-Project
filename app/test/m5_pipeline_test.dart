import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:serpentguard/inference/bundle.dart';
import 'package:serpentguard/inference/decision.dart';
import 'package:serpentguard/inference/flip_prior.dart';
import 'package:serpentguard/inference/image_ops.dart';
import 'package:serpentguard/inference/pipeline.dart';

/// The M5 pieces that run without a model: the settings, the snake-check
/// rule, the mirror image and the mirror + prior combination.
void main() {
  final settings = PipelineSettings.parse(File('assets/model/pipeline.json').readAsStringSync());
  final photos = jsonDecode(File('test/golden/golden_photos.json').readAsStringSync()) as Map<String, dynamic>;

  test('pipeline.json holds the frozen snake check and classifier answer', () {
    expect(settings.gateRule, snakeCheckRule);
    final check = settings.snakeCheck!;
    expect(check.file, snakeCheckFile);
    expect(modelFiles, contains(check.file));
    expect(check.cutoff, 0.20484977503483762);
    expect(check.inputSize, 256);
    expect(settings.classifier.priorTau, 0.5);
    expect(settings.thresholds.thetaWho, double.infinity);
  });

  test('the snake check calls a snake at the cut-off and above', () {
    final check = settings.snakeCheck!;
    expect(check.isSnake([check.cutoff, 0, 0]), isTrue);
    expect(check.isSnake([check.cutoff - 1e-12, 0, 0]), isFalse);
  });

  test('golden photos: snake-check calls agree with their own probabilities', () {
    final check = settings.snakeCheck!;
    final cases = (photos['photos'] as List).cast<Map<String, dynamic>>();
    expect(cases.length, 15);
    for (final p in cases) {
      final probs = ((p['steps'] as Map)['snake_check']['probs'] as List).map(jsonDouble).toList();
      final answer = p['answer'] as Map<String, dynamic>;
      expect(check.isSnake(probs), (answer['snake_check'] as Map)['snake'], reason: '${p['photo_id']}');
      expect(answer['top3'] as List, check.isSnake(probs) ? isNotEmpty : isEmpty);
    }
  });

  test('jsonDouble reads infinity written as a string', () {
    expect(jsonDouble('Infinity'), double.infinity);
    expect(jsonDouble(2), 2.0);
  });

  test('mirrored reverses each row and keeps the pixels', () {
    final img = RgbImage(3, 2, Uint8List.fromList([for (var i = 0; i < 18; i++) i]));
    final m = img.mirrored();
    expect(m.data.sublist(0, 9), [6, 7, 8, 3, 4, 5, 0, 1, 2]);
    expect(m.data.sublist(9), [15, 16, 17, 12, 13, 14, 9, 10, 11]);
    expect(m.mirrored().data, img.data);
  });

  ModelOutputs outputs(double v) => ModelOutputs(
        embedding: [v, 0],
        speciesLogits: [v, 2 * v],
        whoLogits: [v],
        genusLogits: [v],
        familyLogits: [v],
      );

  test('classifierAnswer averages every head, re-normalises, adds the prior', () {
    const c = ClassifierSettings(flipTta: true, priorTau: 0.5, logCounts: [2, 4]);
    final mirrored = ModelOutputs(embedding: [0, 3], speciesLogits: [3, 4], whoLogits: [3], genusLogits: [5], familyLogits: [7]);
    final a = classifierAnswer(outputs(1), mirrored, c);
    expect(a.speciesLogits, [2 + 1, 3 + 2]);
    expect(a.whoLogits, [2]);
    expect(a.genusLogits, [3]);
    expect(a.familyLogits, [4]);
    expect(a.embedding[0], closeTo(1 / math.sqrt(10), 1e-12));
    expect(a.embedding[1], closeTo(3 / math.sqrt(10), 1e-12));
  });

  test('classifierAnswer refuses a missing mirror pass or a wrong prior length', () {
    const c = ClassifierSettings(flipTta: true, priorTau: 0.5, logCounts: [2, 4]);
    expect(() => classifierAnswer(outputs(1), null, c), throwsArgumentError);
    const short = ClassifierSettings(flipTta: false, priorTau: 0.5, logCounts: [2]);
    expect(() => classifierAnswer(outputs(1), null, short), throwsArgumentError);
    expect(classifierAnswer(outputs(1), null, ClassifierSettings.none).speciesLogits, [1, 2]);
  });
}
