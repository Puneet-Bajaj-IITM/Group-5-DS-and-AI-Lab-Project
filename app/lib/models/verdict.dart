import 'species.dart';

/// The three things the screen can say, as in the Milestone 3 decision layer.
enum Verdict {
  dangerous('Dangerous', 'Keep away. If anyone is bitten, go to hospital now.'),
  notListed('Not on the WHO list', 'Still keep away. Do not try to handle it.'),
  unknown("We don't know", 'Treat it as dangerous and keep away.');

  const Verdict(this.title, this.advice);
  final String title;
  final String advice;

  /// Dangerous and unknown both take the full emergency path.
  bool get escalate => this != Verdict.notListed;

  /// Stable wire code for Bluetooth alerts. Never reorder.
  int get code => index;
  static Verdict fromCode(int code) =>
      code >= 0 && code < values.length ? values[code] : Verdict.unknown;
}

class SpeciesScore {
  const SpeciesScore(this.species, this.probability);
  final Species species;
  final double probability;
}

/// What the classifier produced for one photo, plus the verdict.
class Identification {
  const Identification({
    required this.verdict,
    required this.reason,
    required this.top,
    required this.isDemo,
    this.snakeFound = true,
  });

  final Verdict verdict;
  final String reason;

  /// Top species, most likely first. Empty when no snake was found.
  final List<SpeciesScore> top;

  /// True while the trained model is not connected and results are simulated.
  final bool isDemo;
  final bool snakeFound;

  Species? get topSpecies => top.isEmpty ? null : top.first.species;
}

/// Turns species probabilities into a verdict. This mirrors the shape of the
/// M3 rule (`src/serpentguard/decide.py`); the thresholds here are
/// placeholders until the shipped thresholds file is wired in.
class DecisionRule {
  const DecisionRule({
    required this.catalog,
    this.dangerThreshold = 0.30,
    this.unsureThreshold = 0.35,
  });

  final SpeciesCatalog catalog;

  /// Total probability on dangerous species that makes a photo "dangerous".
  final double dangerThreshold;

  /// Below this top-species probability the model is "unsure".
  final double unsureThreshold;

  Identification decide(List<double> probs, {bool isDemo = false}) {
    assert(probs.length == catalog.all.length);
    final ranked = List.generate(probs.length, (i) => i)
      ..sort((a, b) => probs[b].compareTo(probs[a]));
    final top = ranked
        .take(3)
        .map((i) => SpeciesScore(catalog.all[i], probs[i]))
        .toList(growable: false);
    final best = top.first;

    var dangerMass = 0.0;
    for (final s in catalog.all) {
      if (s.dangerous) dangerMass += probs[s.id];
    }

    if (best.species.dangerous) {
      return Identification(
        verdict: Verdict.dangerous,
        reason: best.species.who == WhoCategory.notListed
            ? 'Venomous family, not on WHO\'s priority list'
            : best.species.who.label,
        top: top,
        isDemo: isDemo,
      );
    }
    if (dangerMass >= dangerThreshold) {
      return Identification(
        verdict: Verdict.dangerous,
        reason: 'Could be a dangerous species '
            '(${(dangerMass * 100).round()}% on dangerous species)',
        top: top,
        isDemo: isDemo,
      );
    }
    if (best.probability < unsureThreshold) {
      final genusRisk = catalog.dangerousGenera.contains(best.species.genus);
      return Identification(
        verdict: Verdict.unknown,
        reason: genusRisk
            ? 'Unclear, and its genus has dangerous species'
            : 'Unclear between species',
        top: top,
        isDemo: isDemo,
      );
    }
    return Identification(
      verdict: Verdict.notListed,
      reason: 'Most likely ${best.species.displayName}',
      top: top,
      isDemo: isDemo,
    );
  }
}
