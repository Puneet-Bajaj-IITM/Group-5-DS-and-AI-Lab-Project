import 'dart:math' as math;

import 'bundle.dart';
import 'decision.dart';

/// The frozen M5 classifier answer (results/m5/test/FREEZE.md, section 3):
/// every head averaged over the crop and its mirror image, the embedding
/// re-normalised, then the training prior `tau * log(n_c)` added to the
/// species logits. A port of `reference_frozen.Pipeline.classify`.
ModelOutputs classifierAnswer(ModelOutputs plain, ModelOutputs? mirrored, ClassifierSettings c) {
  if (c.flipTta && mirrored == null) {
    throw ArgumentError('the settings ask for a mirrored pass');
  }
  final both = mirrored != null;
  List<double> mean(List<double> a, List<double>? b) =>
      both ? [for (var i = 0; i < a.length; i++) (a[i] + b![i]) / 2] : a;

  var embedding = plain.embedding;
  if (both) {
    final sum = [for (var i = 0; i < embedding.length; i++) embedding[i] + mirrored.embedding[i]];
    var norm = 0.0;
    for (final v in sum) {
      norm += v * v;
    }
    norm = math.sqrt(norm);
    embedding = [for (final v in sum) v / norm];
  }
  var species = mean(plain.speciesLogits, mirrored?.speciesLogits);
  if (c.priorTau != 0) {
    if (c.logCounts.length != species.length) {
      throw ArgumentError('log_counts has ${c.logCounts.length} classes, the model ${species.length}');
    }
    species = [for (var i = 0; i < species.length; i++) species[i] + c.priorTau * c.logCounts[i]];
  }
  return ModelOutputs(
    embedding: embedding,
    speciesLogits: species,
    whoLogits: mean(plain.whoLogits, mirrored?.whoLogits),
    genusLogits: mean(plain.genusLogits, mirrored?.genusLogits),
    familyLogits: mean(plain.familyLogits, mirrored?.familyLogits),
  );
}
