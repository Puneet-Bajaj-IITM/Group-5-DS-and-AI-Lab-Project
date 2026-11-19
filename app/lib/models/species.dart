import 'dart:convert';

import 'package:flutter/services.dart' show rootBundle;

/// One of the 184 classes the model knows, from `assets/species.json`
/// (built by `tool/build_species_asset.py` from the repo's species cards).
class Species {
  const Species({
    required this.id,
    required this.name,
    required this.commonName,
    required this.genus,
    required this.family,
    required this.who,
    required this.dangerous,
    this.regionalNames = const [],
  });

  factory Species.fromJson(Map<String, dynamic> json) => Species(
        id: json['id'] as int,
        name: json['species'] as String,
        commonName: json['common'] as String?,
        genus: json['genus'] as String,
        family: json['family'] as String,
        who: WhoCategory.parse(json['who'] as String),
        dangerous: json['dangerous'] as bool,
        regionalNames: [
          for (final r in (json['regional'] as List? ?? const []))
            RegionalName.fromJson(r as Map<String, dynamic>),
        ],
      );

  final int id;
  final String name;
  final String? commonName;
  final String genus;
  final String family;
  final WhoCategory who;

  /// On our danger table: WHO-listed, a viper or elapid, or a Rhabdophis keelback.
  final bool dangerous;

  /// Names in Indian languages (draft, see tool/regional_names.csv).
  final List<RegionalName> regionalNames;

  String get displayName => commonName ?? name;

  /// "Spectacled (Indian) cobra (Naja naja)" or just the scientific name.
  String get fullName => commonName == null ? name : '$commonName ($name)';
}

enum WhoCategory {
  category1('WHO Category 1'),
  category2('WHO Category 2'),
  notListed('Not on the WHO list');

  const WhoCategory(this.label);
  final String label;

  static WhoCategory parse(String raw) => switch (raw) {
        '1' => WhoCategory.category1,
        '2' => WhoCategory.category2,
        _ => WhoCategory.notListed,
      };
}

/// The class list, loaded once from the bundled asset.
class SpeciesCatalog {
  SpeciesCatalog(this.all);

  final List<Species> all;

  static Future<SpeciesCatalog> load() async {
    final raw = await rootBundle.loadString('assets/species.json');
    final list = (jsonDecode(raw) as List)
        .map((e) => Species.fromJson(e as Map<String, dynamic>))
        .toList(growable: false);
    return SpeciesCatalog(list);
  }

  Species? byId(int id) => id >= 0 && id < all.length ? all[id] : null;

  /// Genera that hold at least one dangerous species: the app never says
  /// "not listed" for these when it is unsure (the M3 decision rule).
  late final Set<String> dangerousGenera =
      all.where((s) => s.dangerous).map((s) => s.genus).toSet();
}

/// A species' name in one Indian language, in its script and in Latin letters.
class RegionalName {
  const RegionalName({required this.lang, required this.name, required this.latin});

  factory RegionalName.fromJson(Map<String, dynamic> j) =>
      RegionalName(lang: j['lang'] as String, name: j['name'] as String, latin: j['latin'] as String);

  /// ISO 639-1 code: hi, mr, bn, ta, te, kn, ml.
  final String lang;
  final String name, latin;

  static const languages = {
    'hi': 'Hindi',
    'mr': 'Marathi',
    'bn': 'Bengali',
    'ta': 'Tamil',
    'te': 'Telugu',
    'kn': 'Kannada',
    'ml': 'Malayalam',
  };

  String get language => languages[lang] ?? lang;

  @override
  String toString() => '$name ($latin, $language)';
}
