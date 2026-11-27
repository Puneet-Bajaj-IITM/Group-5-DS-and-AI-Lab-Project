import 'dart:convert';

import 'package:flutter_test/flutter_test.dart';
import 'package:serpentguard/inference/image_ops.dart' show PixelBox;
import 'package:serpentguard/models/records.dart';
import 'package:serpentguard/models/verdict.dart';

import 'helpers.dart';

/// A saved sighting must reopen with the result the user saw: verdict,
/// reason, species and percentages, genus fallback and the box on the photo.
void main() {
  final catalog = loadCatalog();
  final cobra = byName(catalog, 'Naja naja');
  final ratSnake = byName(catalog, 'Ptyas mucosa');

  test('a saved result survives the round trip through storage', () {
    final saved = SavedResult(
      reason: 'WHO Category 1 species is likely',
      top: [(cobra.id, 0.82), (ratSnake.id, 0.11)],
      isDemo: false,
      snakeFound: true,
      message: 'dangerous: WHO Category 1',
      reasons: const ['who_head'],
      genusFallback: const GenusFallback('Naja', 0.9, true),
      box: const PixelBox(10, 20, 300, 400),
      imageWidth: 640,
      imageHeight: 480,
    );
    final s = Sighting(time: DateTime(2026, 11, 27, 18, 5), verdict: Verdict.dangerous, speciesId: cobra.id,
        photoPath: '/x.jpg', result: saved);
    final back = Sighting.fromJson(jsonDecode(jsonEncode(s.toJson())) as Map<String, dynamic>);
    final id = back.toIdentification(catalog);
    expect(id.verdict, Verdict.dangerous);
    expect(id.reason, 'WHO Category 1 species is likely');
    expect(id.top.map((t) => t.species.name), ['Naja naja', 'Ptyas mucosa']);
    expect(id.top.first.probability, closeTo(0.82, 1e-9));
    expect(id.message, 'dangerous: WHO Category 1');
    expect(id.reasons, ['who_head']);
    expect(id.genusFallback.toString(), 'Naja (90%)');
    expect(id.genusFallback!.whoListed, isTrue);
    expect(back.result!.box!.right, 300);
    expect(back.result!.imageWidth, 640);
  });

  test('a sighting saved before full results still opens, without percentages', () {
    final old = Sighting.fromJson({'time': '2026-11-01T10:00:00.000', 'verdict': Verdict.unknown.code, 'species': cobra.id});
    expect(old.result, isNull);
    final id = old.toIdentification(catalog);
    expect(id.verdict, Verdict.unknown);
    expect(id.reason, Verdict.unknown.advice);
    expect(id.top.single.species.name, 'Naja naja');
    expect(id.top.single.probability.isNaN, isTrue);
  });

  test('a damaged saved result falls back to the old view instead of failing', () {
    final s = Sighting.fromJson({
      'time': '2026-11-01T10:00:00.000',
      'verdict': Verdict.notListed.code,
      'species': ratSnake.id,
      'result': {'top': 'not a list'},
    });
    expect(s.result, isNull);
    expect(s.toIdentification(catalog).top.single.species.name, 'Ptyas mucosa');
  });
}
