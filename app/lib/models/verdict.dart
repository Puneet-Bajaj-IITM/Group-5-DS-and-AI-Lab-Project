import '../inference/decision.dart';
import '../inference/pipeline.dart';
import 'species.dart';

export '../inference/decision.dart' show GenusFallback;

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
    this.message,
    this.reasons = const [],
    this.genusFallback,
    this.details,
  });

  final Verdict verdict;

  /// The short reason the banner shows.
  final String reason;

  /// Top species, most likely first. Empty when no snake was found.
  final List<SpeciesScore> top;

  /// True while the trained model is not connected and results are simulated.
  final bool isDemo;
  final bool snakeFound;

  /// The reference pipeline's full message, e.g. "dangerous: WHO Category 1".
  final String? message;

  /// The reasons that fired, in the reference's order (e.g. "who_head").
  final List<String> reasons;

  /// When the verdict is "we don't know": the likely genus, if the genus
  /// head is sure enough, e.g. "Lycodon (80%)".
  final GenusFallback? genusFallback;

  /// The pipeline's intermediate values (gate, crop, timings), when a real
  /// model produced this.
  final PipelineResult? details;

  Species? get topSpecies => top.isEmpty ? null : top.first.species;
}

/// The decision rules: DR0 with A3's calibrated thresholds, ported from the
/// model bundle's `reference_pipeline.py` (see `lib/inference/decision.dart`,
/// tested against `golden_decisions.json`). This maps their answer onto
/// what the screens show.
class DecisionRule {
  const DecisionRule({required this.catalog, required this.rules});

  final SpeciesCatalog catalog;
  final DecisionRules rules;

  static const _dangerPrefix = 'dangerous: ';
  static const _unknownPrefix = "we don't know: ";

  Identification decide(DecisionParts parts, {bool isDemo = false}) =>
      fromAnswer(rules.verdict(parts), isDemo: isDemo);

  Identification fromAnswer(ModelAnswer a, {bool isDemo = false, PipelineResult? details}) {
    final noSnake = a.reasons.contains('no_snake');
    final verdict = switch (a.label) {
      dangerousLabel => Verdict.dangerous,
      notListedLabel => Verdict.notListed,
      _ => Verdict.unknown,
    };
    final top = [
      for (final t in a.top3)
        if (catalog.byId(t.classId) != null) SpeciesScore(catalog.byId(t.classId)!, t.probability)
    ];
    final String reason;
    if (a.message.startsWith(_dangerPrefix)) {
      reason = _sentence(a.message.substring(_dangerPrefix.length));
    } else if (a.message.startsWith(_unknownPrefix)) {
      reason = _sentence(a.message.substring(_unknownPrefix.length));
    } else {
      reason = top.isEmpty ? 'Not on the WHO list' : 'Most likely ${top.first.species.displayName}';
    }
    return Identification(
      verdict: verdict,
      reason: reason,
      top: top,
      isDemo: isDemo,
      snakeFound: !noSnake,
      message: a.message,
      reasons: a.reasons,
      genusFallback: a.genusFallback,
      details: details,
    );
  }

  static String _sentence(String s) => s.isEmpty ? s : s[0].toUpperCase() + s.substring(1);
}
