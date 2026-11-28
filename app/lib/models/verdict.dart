import '../inference/decision.dart';
import '../inference/image_ops.dart' show PixelBox;
import '../inference/pipeline.dart';
import 'species.dart';

export '../inference/decision.dart' show GenusFallback;

/// The three things the screen can say, as in the Milestone 3 decision layer.
/// "Unknown" is shown as a warning with the same urgency as "Dangerous"
/// (post-M5-test change): the model could not rule a dangerous snake out.
enum Verdict {
  dangerous('Dangerous', 'Keep away. If anyone is bitten, go to hospital now.'),
  notListed('Not on the WHO list', 'Still keep away. Do not try to handle it.'),
  unknown('Could be dangerous', 'Keep away and treat it as venomous. If anyone is bitten, go to hospital now.');

  const Verdict(this.title, this.advice);
  final String title;
  final String advice;

  /// Dangerous and unknown both take the full emergency path.
  bool get escalate => this != Verdict.notListed;

  /// Dangerous and unknown both get the urgent warning style.
  bool get warns => escalate;

  /// Higher is more severe: dangerous, then "could be dangerous", then not listed.
  int get severity => switch (this) {
        Verdict.dangerous => 2,
        Verdict.unknown => 1,
        Verdict.notListed => 0,
      };

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

/// Another snake in the photo: its box and its own answer.
class IdentifiedSnake {
  const IdentifiedSnake(this.box, this.identification);
  final PixelBox box;
  final Identification identification;
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
    this.others = const [],
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

  /// The other snakes in the photo, most confident first.
  final List<IdentifiedSnake> others;

  Species? get topSpecies => top.isEmpty ? null : top.first.species;

  /// The answer with the most severe verdict among every snake (this one on a tie).
  Identification get mostSevere => others
      .map((o) => o.identification)
      .fold(this, (worst, o) => o.verdict.severity > worst.verdict.severity ? o : worst);
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
      others: [
        for (final o in details?.others ?? const <OtherSnake>[])
          if (o.answer != null) IdentifiedSnake(o.box, fromAnswer(o.answer!, isDemo: isDemo)),
      ],
    );
  }

  static String _sentence(String s) => s.isEmpty ? s : s[0].toUpperCase() + s.substring(1);
}
