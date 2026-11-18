import 'package:flutter_test/flutter_test.dart';
import 'package:serpentguard/models/records.dart';
import 'package:serpentguard/models/species.dart';
import 'package:serpentguard/models/verdict.dart';
import 'package:serpentguard/services/classifier.dart';
import 'package:serpentguard/services/hospital_service.dart';
import 'package:serpentguard/services/sos_service.dart';
import 'package:serpentguard/util/geo.dart';

import 'helpers.dart';

void main() {
  final catalog = loadCatalog();
  final rule = DecisionRule(catalog: catalog);

  List<double> probs(Map<String, double> named) {
    final n = catalog.all.length;
    final rest = 1 - named.values.fold(0.0, (a, b) => a + b);
    final p = List<double>.filled(n, rest / (n - named.length));
    named.forEach((name, v) => p[byName(catalog, name).id] = v);
    return p;
  }

  test('catalog matches the M3 danger table', () {
    expect(catalog.all.length, 184);
    expect(catalog.all.where((s) => s.dangerous).length, 50);
    expect(byName(catalog, 'Bungarus caeruleus').who, WhoCategory.category1);
  });

  test('a dangerous top species is dangerous', () {
    final id = rule.decide(probs({'Daboia russelii': 0.8}));
    expect(id.verdict, Verdict.dangerous);
    expect(id.reason, 'WHO Category 1');
    expect(id.top.first.species.name, 'Daboia russelii');
  });

  test('enough mass on dangerous species is dangerous even if the top is harmless', () {
    final id = rule.decide(probs({'Boiga siamensis': 0.45, 'Naja naja': 0.2, 'Daboia russelii': 0.15}));
    expect(id.verdict, Verdict.dangerous);
  });

  test('a confident harmless species is not listed, an unsure one is unknown', () {
    expect(rule.decide(probs({'Ptyas mucosa': 0.85})).verdict, Verdict.notListed);
    expect(rule.decide(probs({'Ptyas mucosa': 0.2, 'Coelognathus helena': 0.18})).verdict,
        Verdict.unknown);
  });

  test('demo classifier is deterministic and every verdict is reachable', () {
    final demo = DemoClassifier(catalog);
    expect(demo.simulatedProbs(5), demo.simulatedProbs(5));
    final seen = <Verdict>{
      for (var seed = 0; seed < 400; seed++) rule.decide(demo.simulatedProbs(seed)).verdict
    };
    expect(seen, Verdict.values.toSet());
  });

  test('geo helpers', () {
    const a = GeoPoint(12.9716, 77.5946);
    const b = GeoPoint(13.0827, 80.2707); // Chennai
    expect(distanceMeters(a, b) / 1000, closeTo(290, 5));
    expect(compassPoint(bearingDegrees(a, b)), 'E');
    expect(formatDistance(340), '340 m');
    expect(formatDistance(2345), '2.3 km');
  });

  test('SOS message carries the bite, species and a maps link', () {
    final msg = SosService.message(
      bitten: true,
      location: const GeoPoint(12.97161, 77.59463),
      accuracy: 8.4,
      species: byName(catalog, 'Bungarus caeruleus'),
      verdict: Verdict.dangerous,
    );
    expect(msg, contains('bitten'));
    expect(msg, contains('Common krait (Bungarus caeruleus)'));
    expect(msg, contains('https://maps.google.com/?q=12.97161,77.59463'));
    expect(msg, contains('within 8 m'));
    expect(SosService.message(bitten: false), contains('Location not available'));
  });

  test('Overpass parsing keeps named hospitals, merges duplicates, sorts by distance', () {
    final parsed = HospitalService.parseOverpass({
      'elements': [
        {'type': 'node', 'lat': 12.98, 'lon': 77.60, 'tags': {'name': 'Near Hospital', 'emergency': 'yes', 'phone': '+91 80 1234 5678;+91 80 9999'}},
        {'type': 'way', 'center': {'lat': 12.9801, 'lon': 77.6001}, 'tags': {'name': 'Near Hospital'}},
        {'type': 'node', 'lat': 13.10, 'lon': 77.70, 'tags': {'name': 'Far Hospital'}},
        {'type': 'node', 'lat': 12.975, 'lon': 77.595, 'tags': {'name': 'Clinic No ER', 'emergency': 'no'}},
        {'type': 'node', 'lat': 12.99, 'lon': 77.61, 'tags': {'amenity': 'hospital'}},
        {'type': 'node', 'lat': 12.972, 'lon': 77.5947, 'tags': {'name': 'Eye Hospital', 'healthcare:speciality': 'ophthalmology'}},
      ],
    });
    expect(parsed.map((h) => h.name), ['Near Hospital', 'Far Hospital', 'Clinic No ER', 'Eye Hospital']);
    expect(parsed.first.phone, '+91 80 1234 5678');
    final sorted = HospitalService.sortedByDistance(parsed, const GeoPoint(12.9716, 77.5946));
    expect(sorted.map((h) => h.name), ['Near Hospital', 'Far Hospital', 'Eye Hospital', 'Clinic No ER'],
        reason: 'specialty-only and no-ER hospitals go after the rest, nearest first among them');
    expect(Hospital.fromJson(sorted.first.toJson()).emergency, isTrue);
  });

  test('Nominatim fallback keeps only hospitals', () {
    final parsed = HospitalService.parseNominatim([
      {'category': 'amenity', 'type': 'hospital', 'lat': '12.98', 'lon': '77.60', 'name': 'Sagar Hospital', 'extratags': {'phone': '+91 80 4288 8888', 'emergency': 'yes'}},
      {'category': 'amenity', 'type': 'clinic', 'lat': '12.97', 'lon': '77.59', 'name': 'Small Clinic'},
      {'category': 'amenity', 'type': 'hospital', 'lat': '12.99', 'lon': '77.61', 'name': 'Govt Hospital', 'extratags': null},
    ]);
    expect(parsed.map((h) => h.name), ['Sagar Hospital', 'Govt Hospital']);
    expect(parsed.first.phone, '+91 80 4288 8888');
    expect(parsed.first.emergency, isTrue);
  });
}
