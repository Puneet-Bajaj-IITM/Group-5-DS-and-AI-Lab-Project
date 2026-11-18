import 'dart:math' as math;
import 'dart:typed_data';

import 'package:crypto/crypto.dart';

import '../models/species.dart';
import '../models/verdict.dart';

/// Turns a photo into species probabilities and a verdict.
///
/// The trained pipeline (YOLO26n crop, then SnakeID in fp16 ONNX) will
/// implement this interface; until then [DemoClassifier] stands in so the
/// rest of the app can be built and tested.
abstract class SnakeClassifier {
  bool get isDemo;
  Future<Identification> identify(Uint8List imageBytes);
}

/// Simulated results. The same photo always gives the same answer (seeded by
/// the photo's hash), and results lean towards common lookalikes so every
/// verdict shows up while testing. It does not look at the image content.
class DemoClassifier implements SnakeClassifier {
  DemoClassifier(this.catalog) : _rule = DecisionRule(catalog: catalog);

  final SpeciesCatalog catalog;
  final DecisionRule _rule;

  @override
  bool get isDemo => true;

  @override
  Future<Identification> identify(Uint8List imageBytes) async {
    final seed = sha256.convert(imageBytes).bytes.take(4).fold<int>(0, (a, b) => (a << 8) | b);
    await Future<void>.delayed(const Duration(milliseconds: 600)); // feels like inference
    return _rule.decide(simulatedProbs(seed), isDemo: true);
  }

  /// Also used by the "try a species" picker on the result screen.
  Identification forSpecies(int speciesId, {double confidence = 0.7}) {
    final n = catalog.all.length;
    final rest = (1 - confidence) / (n - 1);
    final probs = List<double>.filled(n, rest);
    probs[speciesId] = confidence;
    return _rule.decide(probs, isDemo: true);
  }

  List<double> simulatedProbs(int seed) {
    final rnd = math.Random(seed);
    final n = catalog.all.length;
    // A peaked softmax over random logits, with the peak on 1 to 3 species.
    final logits = List<double>.generate(n, (_) => rnd.nextDouble());
    final peaks = 1 + rnd.nextInt(3);
    for (var i = 0; i < peaks; i++) {
      logits[rnd.nextInt(n)] += 4 + rnd.nextDouble() * 3;
    }
    final maxLogit = logits.reduce(math.max);
    final exps = logits.map((l) => math.exp(l - maxLogit)).toList();
    final total = exps.reduce((a, b) => a + b);
    return exps.map((e) => e / total).toList(growable: false);
  }
}
