import 'dart:convert';
import 'dart:math' as math;
import 'dart:typed_data';

/// Every size, gate and threshold the pipeline uses, read from the model
/// bundle's `pipeline.json`. Nothing here is typed into Dart by hand.
class PipelineSettings {
  PipelineSettings(this.raw);

  factory PipelineSettings.parse(String json) =>
      PipelineSettings(jsonDecode(json) as Map<String, dynamic>);

  final Map<String, dynamic> raw;

  Map<String, dynamic> _m(String key) => raw[key] as Map<String, dynamic>;
  static double _d(Object? v) => (v as num).toDouble();
  static List<double> _ds(Object? v) => (v as List).map(_d).toList(growable: false);

  int get inputSize => raw['input_size'] as int;

  // exposure fix
  double get exposureBelow => _d(_m('exposure_fix')['when_mean_below']);
  double get exposureAbove => _d(_m('exposure_fix')['when_mean_above']);
  double get claheClip => _d(_m('exposure_fix')['clahe_clip_limit']);
  (int, int) get claheTiles {
    final t = _m('exposure_fix')['clahe_tile_grid'] as List;
    return (t[0] as int, t[1] as int);
  }

  // normalisation and letterbox
  List<double> get mean => _ds(_m('normalise')['mean']);
  List<double> get std => _ds(_m('normalise')['std']);
  double get scale => _d(_m('normalise')['scale']);
  List<int> get padRgb => (_m('letterbox')['pad_rgb'] as List).cast<int>();

  // detector
  int get detectorPad => _m('detector')['pad_value'] as int;
  int get snakeClass => _m('detector')['snake_class'] as int;
  double get boxFloor => _d(raw['front_end_box_floor']);

  // segmenter, gate, cleanup, crop
  double get segmenterThreshold => _d(_m('segmenter')['threshold']);
  String get gateRule => _m('gate')['rule'] as String;
  double get gateBoxConf => _d((_m('gate')['thresholds'] as Map)['box_conf']);
  double get gateMaskArea => _d((_m('gate')['thresholds'] as Map)['mask_area']);
  int get connectivity => _m('mask_cleanup')['connectivity'] as int;
  double get minLargestShare => _d(_m('mask_cleanup')['min_largest_share']);
  double get keepShareOfLargest => _d(_m('mask_cleanup')['keep_part_share_of_largest']);
  double get keepShareOfPhoto => _d(_m('mask_cleanup')['keep_part_share_of_photo']);
  double get cropPadding => _d(_m('crop')['padding']);

  DecisionThresholds get thresholds =>
      DecisionThresholds.fromJson((_m('decision')['thresholds'] as Map).cast<String, dynamic>());
}

/// A3's calibrated decision thresholds (DR0).
class DecisionThresholds {
  const DecisionThresholds({
    required this.temperature,
    required this.temperatureWho,
    required this.thetaD,
    required this.thetaWho,
    required this.tauUnknown,
    required this.thetaConf,
    required this.thetaV,
    required this.thetaGenus,
  });

  factory DecisionThresholds.fromJson(Map<String, dynamic> j) {
    double d(String k) => (j[k] as num).toDouble();
    return DecisionThresholds(
      temperature: d('temperature'),
      temperatureWho: d('temperature_who'),
      thetaD: d('theta_d'),
      thetaWho: d('theta_who'),
      tauUnknown: d('tau_unknown'),
      thetaConf: d('theta_conf'),
      thetaV: d('theta_v'),
      thetaGenus: d('theta_genus'),
    );
  }

  final double temperature, temperatureWho, thetaD, thetaWho, tauUnknown, thetaConf, thetaV, thetaGenus;
}

/// Per-species, per-genus and per-family tables from `tables.json`, in
/// class-id order (the same order as `assets/species.json`).
class ModelTables {
  ModelTables._({
    required this.species,
    required this.whoCategory,
    required this.genusNames,
    required this.genusOfClass,
    required this.who,
    required this.dangerous,
    required this.note,
    required this.guardedGenus,
    required this.listedGenus,
    required this.venomousFamily,
    required this.reasons,
    required this.reasonText,
    required this.venomousFamilyNote,
    required this.prototypes,
    required this.prototypeDim,
  });

  factory ModelTables.parse(String json) {
    final j = jsonDecode(json) as Map<String, dynamic>;
    List<String> strs(String k) => (j[k] as List).cast<String>();
    List<bool> bools(String k) => (j[k] as List).map((v) => v == true || v == 1).toList(growable: false);
    List<int> ints(String k) => (j[k] as List).map((v) => (v as num).toInt()).toList(growable: false);

    final protoRows = j['prototypes'] as List;
    final dim = (protoRows.first as List).length;
    final protos = Float64List(protoRows.length * dim);
    for (var r = 0; r < protoRows.length; r++) {
      final row = protoRows[r] as List;
      for (var c = 0; c < dim; c++) {
        protos[r * dim + c] = (row[c] as num).toDouble();
      }
    }
    return ModelTables._(
      species: strs('species'),
      whoCategory: (j['who_category'] as List).map((v) => '$v').toList(growable: false),
      genusNames: strs('genus_names'),
      genusOfClass: ints('genus_of_class'),
      who: ints('who'),
      dangerous: bools('dangerous'),
      note: strs('note'),
      guardedGenus: bools('guarded_genus'),
      listedGenus: bools('listed_genus'),
      venomousFamily: bools('venomous_family'),
      reasons: strs('reasons'),
      reasonText: (j['reason_text'] as Map).cast<String, String>(),
      venomousFamilyNote: j['venomous_family_note'] as String,
      prototypes: protos,
      prototypeDim: dim,
    );
  }

  final List<String> species;

  /// "1", "2" or "not_listed".
  final List<String> whoCategory;
  final List<String> genusNames;
  final List<int> genusOfClass;

  /// 0 not listed, 1 WHO Category 2, 2 WHO Category 1.
  final List<int> who;
  final List<bool> dangerous;

  /// What the dangerous screen says when this species is on top.
  final List<String> note;
  final List<bool> guardedGenus;
  final List<bool> listedGenus;
  final List<bool> venomousFamily;

  /// Reason names in the order the screen lists them.
  final List<String> reasons;
  final Map<String, String> reasonText;
  final String venomousFamilyNote;

  /// numClasses x prototypeDim, row-major.
  final Float64List prototypes;
  final int prototypeDim;

  int get numClasses => species.length;
  int get numGenera => genusNames.length;

  /// Prototype rows normalised to length 1; all-zero rows stay null and never win.
  late final List<Float64List?> unitPrototypes = List.generate(numClasses, (r) {
    var norm = 0.0;
    for (var c = 0; c < prototypeDim; c++) {
      final v = prototypes[r * prototypeDim + c];
      norm += v * v;
    }
    if (norm <= 0) return null;
    norm = math.sqrt(norm);
    return Float64List.fromList(
        List.generate(prototypeDim, (c) => prototypes[r * prototypeDim + c] / norm));
  });
}
