import 'dart:math' as math;
import 'dart:typed_data';

import 'bundle.dart';

/// A port of the decision rules (DR0 with A3's thresholds) in the model
/// bundle's `reference_pipeline.py`: `decide`, `danger_note` and `verdict`.
/// Pure Dart, so it is unit-tested against `golden_decisions.json` with no
/// model or photo. All arithmetic is in double, as in the reference.

const dangerousLabel = 'dangerous';
const unknownLabel = 'unknown';
const notListedLabel = 'not_listed';
const notListedMessage = 'not on the WHO list, still keep away';
const noSnakeMessage = "we don't know: no snake found";
const _whoCat2 = 1, _whoCat1 = 2; // tables.json "who" codes; 0 is not listed
const _underOnePercent = 0.01;
const _noGenus = -1;

/// The cosine used when there is no embedding: the distance check is off.
const perfectCosine = 1.0;

/// Python's round(): halves go to the even neighbour (2.5 -> 2).
int roundHalfEven(double x) {
  final f = x.floorToDouble();
  final diff = x - f;
  if (diff > 0.5) return f.toInt() + 1;
  if (diff < 0.5) return f.toInt();
  return f.toInt().isEven ? f.toInt() : f.toInt() + 1;
}

/// Python's f"{x:.0%}".
String pyPercent(double x) => '${roundHalfEven(x * 100)}%';

Float64List softmax(List<double> values, [double temperature = 1]) {
  final scaled = Float64List.fromList([for (final v in values) v / temperature]);
  var maxV = double.negativeInfinity;
  for (final v in scaled) {
    if (v > maxV) maxV = v;
  }
  var total = 0.0;
  for (var i = 0; i < scaled.length; i++) {
    scaled[i] = math.exp(scaled[i] - maxV);
    total += scaled[i];
  }
  for (var i = 0; i < scaled.length; i++) {
    scaled[i] /= total;
  }
  return scaled;
}

/// First index of the largest value (numpy's argmax).
int argmax(List<double> v) {
  var best = 0;
  for (var i = 1; i < v.length; i++) {
    if (v[i] > v[best]) best = i;
  }
  return best;
}

/// The raw outputs of `classifier.onnx` for one image.
class ModelOutputs {
  const ModelOutputs({
    required this.embedding,
    required this.speciesLogits,
    required this.whoLogits,
    required this.genusLogits,
    required this.familyLogits,
  });

  factory ModelOutputs.fromJson(Map<String, dynamic> j) {
    List<double> d(String k) => (j[k] as List).map((v) => (v as num).toDouble()).toList();
    return ModelOutputs(
      embedding: d('embedding'),
      speciesLogits: d('species_logits'),
      whoLogits: d('who_logits'),
      genusLogits: d('genus_logits'),
      familyLogits: d('family_logits'),
    );
  }

  final List<double> embedding, speciesLogits, whoLogits, genusLogits, familyLogits;
}

/// Everything the rules read: probabilities from every head, and the
/// embedding. A single photo's parts come from its logits; a video's are the
/// mean over the pooled frames (see video_lock.dart).
class DecisionParts {
  const DecisionParts({
    required this.probs,
    required this.who,
    required this.genus,
    required this.family,
    this.embedding,
  });

  factory DecisionParts.fromOutputs(ModelOutputs o, DecisionThresholds t) => DecisionParts(
        probs: softmax(o.speciesLogits, t.temperature),
        who: softmax(o.whoLogits, t.temperatureWho),
        genus: softmax(o.genusLogits),
        family: softmax(o.familyLogits),
        embedding: Float64List.fromList(o.embedding),
      );

  final Float64List probs, who, genus, family;
  final Float64List? embedding;
}

class GenusFallback {
  const GenusFallback(this.genus, this.probability, this.whoListed);
  final String genus;
  final double probability;
  final bool whoListed;

  @override
  String toString() => '$genus (${(probability * 100).round()}%)';
}

class ScoredClass {
  const ScoredClass(this.classId, this.probability);
  final int classId;
  final double probability;
}

/// What the screen shows for one photo (the reference's `verdict()` record).
class ModelAnswer {
  const ModelAnswer({
    required this.label,
    required this.reasons,
    required this.message,
    required this.top3,
    this.genusFallback,
    this.margin,
    this.bestCosine,
  });

  /// [dangerousLabel], [unknownLabel] or [notListedLabel].
  final String label;
  final List<String> reasons;
  final String message;
  final List<ScoredClass> top3;
  final GenusFallback? genusFallback;
  final double? margin;
  final double? bestCosine;

  static const noSnake = ModelAnswer(
    label: unknownLabel,
    reasons: ['no_snake'],
    message: noSnakeMessage,
    top3: [],
  );
}

class _Found {
  _Found({
    required this.probs,
    required this.whoHead,
    required this.genusHead,
    required this.listedMass,
    required this.genus,
    required this.fired,
    required this.shown,
    required this.margin,
    required this.cosine,
  });
  final Float64List probs, whoHead, genusHead;
  final double listedMass, margin, cosine;
  final int genus;
  final Map<String, bool> fired;
  final List<int> shown;
}

class DecisionRules {
  DecisionRules(this.thresholds, this.tables);

  final DecisionThresholds thresholds;
  final ModelTables tables;

  /// Cosine to the nearest species prototype; an all-zero prototype never wins.
  double bestCosine(List<double>? embedding) {
    if (embedding == null) return perfectCosine;
    var norm = 0.0;
    for (final v in embedding) {
      norm += v * v;
    }
    norm = math.sqrt(norm);
    var best = double.negativeInfinity;
    for (final proto in tables.unitPrototypes) {
      if (proto == null) continue;
      var dot = 0.0;
      for (var c = 0; c < proto.length; c++) {
        dot += proto[c] * (embedding[c] / norm);
      }
      if (dot > best) best = dot;
    }
    return best;
  }

  (int, double) _guardedGenus(Float64List probs, Float64List genusHead, int top1) {
    final genusMass = Float64List(tables.numGenera);
    for (var k = 0; k < probs.length; k++) {
      genusMass[tables.genusOfClass[k]] += probs[k];
    }
    final candidates = [tables.genusOfClass[top1], argmax(genusHead)];
    var best = _noGenus;
    var bestProb = double.negativeInfinity;
    for (final g in candidates) {
      final prob = math.max(genusMass[g], genusHead[g]);
      if (tables.guardedGenus[g] && prob > bestProb) {
        best = g;
        bestProb = prob;
      }
    }
    return (best, bestProb);
  }

  (String, List<String>, _Found) _decide(DecisionParts p) {
    final t = thresholds;
    final probs = p.probs;
    final cosine = bestCosine(p.embedding);
    final top1 = argmax(probs);
    final ranked = [...probs]..sort((a, b) => b.compareTo(a));

    var listedMass = 0.0;
    for (var k = 0; k < probs.length; k++) {
      if (tables.who[k] != 0) listedMass += probs[k];
    }
    var topDangerous = false;
    for (var k = 0; k < probs.length; k++) {
      if (probs[k] == ranked[0] && tables.dangerous[k]) topDangerous = true;
    }
    var venomousMass = 0.0;
    for (var f = 0; f < p.family.length; f++) {
      if (tables.venomousFamily[f]) venomousMass += p.family[f];
    }
    final fired = {
      'who_listed_mass': listedMass >= t.thetaD,
      'who_head': 1 - p.who[0] >= t.thetaWho,
      'top_species_dangerous': topDangerous,
      'venomous_family_mass': venomousMass >= t.thetaV,
    };
    final evidence = fired.values.any((v) => v);
    final unclear = ranked[0] < t.thetaConf;
    final unfamiliar = cosine < t.tauUnknown;
    final (genus, genusProb) = _guardedGenus(probs, p.genus, top1);
    final guarded = !evidence && !(unclear || unfamiliar) && genus != _noGenus;
    final byGenus = guarded && genusProb >= t.thetaGenus;
    final genusUnclear = guarded && !byGenus;
    final unknown = (!evidence && (unclear || unfamiliar)) || genusUnclear;
    final label = evidence || byGenus
        ? dangerousLabel
        : unknown
            ? unknownLabel
            : notListedLabel;
    final reasons = {
      for (final e in fired.entries) e.key: evidence && e.value,
      'dangerous_genus': byGenus,
      'no_snake': false,
      'unclear_between_species': (!evidence && unclear) || genusUnclear,
      'unfamiliar': !evidence && unfamiliar,
      'poor_image': false,
    };
    // Stable sort, highest first: ties keep the lower class id first.
    final order = List.generate(probs.length, (i) => i)
      ..sort((a, b) {
        final c = probs[b].compareTo(probs[a]);
        return c != 0 ? c : a.compareTo(b);
      });
    final found = _Found(
      probs: probs,
      whoHead: p.who,
      genusHead: p.genus,
      listedMass: listedMass,
      genus: genus,
      fired: fired,
      shown: order.take(3).toList(),
      margin: ranked[0] - ranked[1],
      cosine: cosine,
    );
    return (label, [for (final n in tables.reasons) if (reasons[n] == true) n], found);
  }

  String _dangerNote(_Found f) {
    final probs = f.probs;
    if (f.fired['top_species_dangerous']!) {
      final top = probs.reduce(math.max);
      for (var k = 0; k < probs.length; k++) {
        if (probs[k] == top && tables.dangerous[k]) return tables.note[k];
      }
    }
    if (f.fired['who_listed_mass']!) {
      final named = [for (final k in f.shown) if (tables.who[k] != 0) k];
      if (named.isNotEmpty) {
        final category = tables.who[named.first] == _whoCat1 ? 1 : 2;
        return 'could be a WHO-listed species: ${tables.species[named.first]} (Category $category)';
      }
      final share = f.listedMass < _underOnePercent ? 'under 1%' : pyPercent(f.listedMass);
      return 'could be a WHO-listed species: $share combined probability';
    }
    if (f.fired['who_head']!) {
      final category = f.whoHead[_whoCat1] >= f.whoHead[_whoCat2] ? 1 : 2;
      return 'the photo looks like a WHO Category $category snake';
    }
    if (f.fired['venomous_family_mass']!) return tables.venomousFamilyNote;
    final name = tables.genusNames[f.genus];
    if (tables.listedGenus[f.genus]) return 'genus $name has species on the WHO list';
    return 'genus $name has species named as medically important in the WHO guideline';
  }

  /// The record the app shows: verdict, reasons, message, top 3, genus fallback.
  ModelAnswer verdict(DecisionParts parts) {
    final (label, reasons, found) = _decide(parts);
    var message = notListedMessage;
    GenusFallback? fallback;
    if (label == dangerousLabel) {
      message = 'dangerous: ${_dangerNote(found)}';
    } else if (label == unknownLabel) {
      message = "we don't know: ${reasons.map((r) => tables.reasonText[r]).join(', ')}";
      final head = found.genusHead;
      final g = argmax(head);
      if (head[g] >= thresholds.thetaGenus) {
        fallback = GenusFallback(tables.genusNames[g], head[g], tables.listedGenus[g]);
      }
    }
    return ModelAnswer(
      label: label,
      reasons: reasons,
      message: message,
      top3: [for (final k in found.shown) ScoredClass(k, found.probs[k])],
      genusFallback: fallback,
      margin: found.margin,
      bestCosine: found.cosine,
    );
  }

  ModelAnswer verdictFromOutputs(ModelOutputs o) =>
      verdict(DecisionParts.fromOutputs(o, thresholds));
}
