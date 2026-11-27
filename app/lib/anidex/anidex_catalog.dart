import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter/services.dart';

/// The AniDex's animals, breeds, label vectors and card text, read from
/// assets/anidex/ (built by tool/anidex/build_anidex.py). Separate from
/// the snake pipeline: nothing here feeds a snake verdict.
class AniDexCatalog {
  AniDexCatalog._(this.raw, this.species, this.labels, this.labelOwner, this.breedLabels);

  static const dir = 'assets/anidex';

  static Future<AniDexCatalog> load() async {
    final raw = jsonDecode(await rootBundle.loadString('$dir/anidex.json')) as Map<String, dynamic>;
    Future<Float32List> matrix(String file, int rows, int cols) async {
      final bytes = await rootBundle.load('$dir/$file');
      final m = bytes.buffer.asFloat32List(bytes.offsetInBytes, bytes.lengthInBytes ~/ 4);
      if (m.length != rows * cols) throw StateError('$file has ${m.length} values, expected $rows x $cols');
      return m;
    }

    final labels = await matrix(raw['labels'] as String, raw['label_count'] as int, raw['dim'] as int);
    final breedLabels =
        await matrix(raw['breed_labels'] as String, raw['breed_label_count'] as int, raw['breed_dim'] as int);
    final species = [for (final s in raw['species'] as List) AniDexSpecies.fromJson(s as Map<String, dynamic>)];
    final owner = Int32List.fromList((raw['label_owner'] as List).cast<int>());
    return AniDexCatalog._(raw, species, labels, owner, breedLabels);
  }

  final Map<String, dynamic> raw;
  final List<AniDexSpecies> species;

  /// BioCLIP 2's species labels: row-major [labelCount x dim], each row
  /// L2-normalised.
  final Float32List labels;

  /// For each label, the index in [species] it counts towards.
  final Int32List labelOwner;

  /// The general CLIP's breed labels: row-major [breedLabelCount x breedDim],
  /// grouped by species (see [AniDexSpecies.breedVectorsStart]).
  final Float32List breedLabels;

  EncoderFile get encoder => EncoderFile(
      raw['encoder'] as String, raw['encoder_sha256'] as String, raw['encoder_url'] as String?);
  EncoderFile get breedEncoder => EncoderFile(
      raw['breed_encoder'] as String, raw['breed_encoder_sha256'] as String, raw['breed_encoder_url'] as String?);
  int get dim => raw['dim'] as int;
  int get labelCount => raw['label_count'] as int;
  int get breedDim => raw['breed_dim'] as int;
  int get inputSize => raw['input_size'] as int;
  List<double> get mean => [for (final v in raw['mean'] as List) (v as num).toDouble()];
  List<double> get std => [for (final v in raw['std'] as List) (v as num).toDouble()];
  double get logitScale => (raw['logit_scale'] as num).toDouble();
  double get lowConfidence => (raw['low_confidence'] as num).toDouble();
  double get breedMin => (raw['breed_min'] as num).toDouble();
  List<String> get groups => (raw['groups'] as List).cast<String>();
  int get breedCount => species.fold(0, (n, s) => n + s.breeds.length);

  AniDexSpecies? byScientific(String name) {
    for (final s in species) {
      if (s.scientific == name) return s;
    }
    return null;
  }
}

/// One ONNX encoder: its asset file name, SHA-256, and where to download it
/// when it isn't bundled (null: bundled only).
class EncoderFile {
  const EncoderFile(this.file, this.sha256, this.url);
  final String file, sha256;
  final String? url;
}

class WikiCard {
  const WikiCard({required this.extract, required this.image, required this.url});

  factory WikiCard.fromJson(Map<String, dynamic>? j) => WikiCard(
        extract: (j?['extract'] as String?) ?? '',
        image: (j?['image'] as String?) ?? '',
        url: (j?['url'] as String?) ?? '',
      );

  final String extract, image, url;
}

class AniDexBreed {
  const AniDexBreed(this.name, this.wiki);
  final String name;
  final WikiCard wiki;
}

class AniDexSpecies {
  AniDexSpecies({
    required this.no,
    required this.scientific,
    required this.name,
    required this.group,
    required this.wiki,
    required this.gbifKey,
    required this.taxonomy,
    required this.iucn,
    required this.subspecies,
    required this.breedNoun,
    required this.breedLabelsStart,
    required this.breedVectorsStart,
    required this.breeds,
  });

  factory AniDexSpecies.fromJson(Map<String, dynamic> j) => AniDexSpecies(
        no: j['no'] as int,
        scientific: j['scientific'] as String,
        name: j['name'] as String,
        group: j['group'] as String,
        wiki: WikiCard.fromJson(j['wiki'] as Map<String, dynamic>?),
        gbifKey: j['gbif_key'] as int?,
        taxonomy: (j['taxonomy'] as Map).cast<String, String>(),
        iucn: j['iucn'] as String?,
        subspecies: ((j['subspecies'] as List?) ?? const []).cast<String>(),
        breedNoun: j['breed_noun'] as String?,
        breedLabelsStart: j['breed_labels_start'] as int?,
        breedVectorsStart: j['breed_vectors_start'] as int?,
        breeds: [
          for (final b in (j['breeds'] as List?) ?? const [])
            AniDexBreed(b['name'] as String, WikiCard.fromJson(b['wiki'] as Map<String, dynamic>?)),
        ],
      );

  final int no;
  final String scientific, name, group;
  final WikiCard wiki;
  final int? gbifKey;
  final Map<String, String> taxonomy;

  /// IUCN Red List code (LC, NT, VU, EN, CR...), wild animals only.
  final String? iucn;
  final List<String> subspecies;
  final String? breedNoun;
  /// Where this species' breed labels start among BioCLIP's labels (they
  /// count towards the species score).
  final int? breedLabelsStart;

  /// Where its breeds start in the general CLIP's breed labels (the breed answer).
  final int? breedVectorsStart;
  final List<AniDexBreed> breeds;

  bool get hasBreeds => breeds.isNotEmpty;

  static const _snakeFamilies = {'Elapidae', 'Viperidae', 'Colubridae', 'Pythonidae', 'Boidae', 'Typhlopidae'};
  bool get isSnake => _snakeFamilies.contains(taxonomy['family']);

  AniDexBreed? breed(String name) {
    for (final b in breeds) {
      if (b.name == name) return b;
    }
    return null;
  }
}
