import 'dart:convert';
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:serpentguard/anidex/anidex_catalog.dart';
import 'package:serpentguard/anidex/anidex_model.dart';

/// Checks the AniDex's Dart preprocessing and scoring against the Python
/// reference (fixtures from tool/anidex/make_fixtures.py). The ONNX encoder
/// itself needs a device, so the fixtures carry its embeddings.
void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  final fixtures = jsonDecode(File('test/anidex/fixtures.json').readAsStringSync()) as Map<String, dynamic>;

  test('the catalog loads: 154 species, labels match their owners', () async {
    final c = await AniDexCatalog.load();
    expect(c.species.length, 154);
    expect(c.labelOwner.length, c.labelCount);
    expect(c.labels.length, c.labelCount * c.dim);
    expect(c.breedLabels.length, c.breedCount * c.breedDim);
    expect(c.breedCount, greaterThan(c.species.length));
    expect(c.species.first.name, 'Dog');
    expect(c.byScientific('Bos taurus')!.breeds.any((b) => b.name == 'Gir'), isTrue);
    for (final s in c.species.where((s) => s.hasBreeds)) {
      for (var i = 0; i < s.breeds.length; i++) {
        expect(c.labelOwner[s.breedLabelsStart! + i], c.species.indexOf(s), reason: '${s.name} breed $i');
      }
    }
  });

  test('preprocessing matches BioCLIP\'s (stretch to 224, CLIP mean and std)', () async {
    final c = await AniDexCatalog.load();
    final p = fixtures['preprocess'] as Map<String, dynamic>;
    final bytes = File('test/anidex/${p['file']}').readAsBytesSync();
    final got = preprocess(bytes, c.inputSize, c.mean, c.std);
    final want = (p['values'] as List).cast<num>();
    final stride = p['stride'] as int;
    var worst = 0.0, total = 0.0;
    for (var i = 0; i < want.length; i++) {
      final d = (got[i * stride] - want[i]).abs();
      worst = d > worst ? d : worst;
      total += d;
    }
    // The same antialiased bilinear filter as torchvision, so only float
    // rounding differs. (package:image's own resize was 0.19 off on photos,
    // enough to turn a Siamese cat into a dog.)
    expect(total / want.length, lessThan(0.002));
    expect(worst, lessThan(0.01));
  });

  test('scoring gives the Python answers on real photo embeddings', () async {
    final c = await AniDexCatalog.load();
    for (final f in (fixtures['scores'] as List).cast<Map<String, dynamic>>()) {
      List<double> vec(String k) => (f[k] as List).cast<num>().map((v) => v.toDouble()).toList();
      final scan = score(c, vec('embedding'), breedEmbedding: vec('breed_embedding'));
      expect(scan.species.scientific, f['species'], reason: f['photo'] as String);
      expect(scan.score, closeTo((f['score'] as num).toDouble(), 1e-3), reason: f['photo'] as String);
      expect(scan.breed?.$1.name, f['breed'], reason: f['photo'] as String);
    }
  });
}
