import 'dart:convert';
import 'dart:io';

import 'package:serpentguard/inference/bundle.dart';
import 'package:serpentguard/inference/decision.dart';
import 'package:serpentguard/models/species.dart';

/// The real class list, read from the bundled asset file.
SpeciesCatalog loadCatalog() {
  final list = (jsonDecode(File('assets/species.json').readAsStringSync()) as List)
      .map((e) => Species.fromJson(e as Map<String, dynamic>))
      .toList();
  return SpeciesCatalog(list);
}

Species byName(SpeciesCatalog c, String name) => c.all.firstWhere((s) => s.name == name);

/// A3's decision rules, from the committed settings and tables.
DecisionRules loadRules() {
  final settings = PipelineSettings.parse(File('assets/model/pipeline.json').readAsStringSync());
  final tables = ModelTables.parse(File('assets/model/tables.json').readAsStringSync());
  return DecisionRules(settings.thresholds, tables);
}
