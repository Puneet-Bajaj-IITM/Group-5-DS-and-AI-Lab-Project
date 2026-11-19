import 'dart:convert';

import 'package:http/http.dart' as http;

import '../models/records.dart';
import '../util/geo.dart';
import 'store.dart';
import 'hospital_index.dart';

/// Finds hospitals near the user from OpenStreetMap (via the Overpass API)
/// while online, and keeps the last list on the phone so it still works
/// offline. OSM data is © OpenStreetMap contributors, ODbL.
///
/// OSM does not record anti-snake venom (ASV) stock, so the app always says
/// "call ahead to confirm ASV".
class HospitalService {
  HospitalService(
    this._store, {
    http.Client? client,
    Future<HospitalIndex> Function()? openIndex,
  }) : _client = client ?? http.Client(),
       _openIndex = openIndex ?? HospitalIndex.open;

  final Store _store;
  final http.Client _client;
  final Future<HospitalIndex> Function() _openIndex;
  Future<HospitalIndex>? _index;

  Future<({List<Hospital> hospitals, DateTime? oldest})> savedNear(
    GeoPoint centre,
  ) async {
    final index = await (_index ??= _openIndex());
    final legacy = cached();
    if (legacy.hospitals.isNotEmpty && legacy.fetched != null) {
      await index.save(legacy.hospitals, legacy.fetched!);
    }
    return index.near(centre);
  }

  static const _cacheKey = 'hospitals.cache';

  /// Public Overpass servers, tried in turn. They are shared and often busy,
  /// which is why the offline cache (and, later, a bundled per-state
  /// hospital file) matters more than this download.
  static const endpoints = [
    'https://overpass-api.de/api/interpreter',
    'https://overpass.kumi.systems/api/interpreter',
    'https://maps.mail.ru/osm/tools/overpass/api/interpreter',
  ];
  static const userAgent = 'SerpentGuard/0.1 (student research app)';

  /// The cached list, its centre and when it was fetched.
  ({List<Hospital> hospitals, GeoPoint? centre, DateTime? fetched}) cached() {
    final m = _store.readMap(_cacheKey);
    if (m == null) return (hospitals: const [], centre: null, fetched: null);
    return (
      hospitals: (m['items'] as List)
          .map((e) => Hospital.fromJson(e as Map<String, dynamic>))
          .toList(),
      centre: m['centre'] == null
          ? null
          : GeoPoint.fromJson(m['centre'] as Map<String, dynamic>),
      fetched: m['fetched'] == null
          ? null
          : DateTime.tryParse(m['fetched'] as String),
    );
  }

  /// Downloads hospitals around [centre] and replaces the cache: within
  /// 10 km first, widening to 30 km when fewer than 5 are found.
  /// Throws when every server fails, so the caller can fall back to [cached].
  Future<List<Hospital>> refresh(GeoPoint centre) async {
    var hospitals = await _fetch(centre, 10);
    if (hospitals.length < 5) {
      try {
        final wider = await _fetch(centre, 30);
        final combined = {
          for (final h in hospitals)
            '${h.name}@${h.location.lat},${h.location.lon}': h,
        };
        for (final h in wider) {
          combined['${h.name}@${h.location.lat},${h.location.lon}'] = h;
        }
        hospitals = combined.values.toList();
      } catch (_) {
        if (hospitals.isEmpty) rethrow;
      }
    }
    final fetched = DateTime.now();
    await _store.writeMap(_cacheKey, {
      'items': hospitals.map((h) => h.toJson()).toList(),
      'centre': centre.toJson(),
      'fetched': fetched.toIso8601String(),
    });
    await (await (_index ??= _openIndex())).save(hospitals, fetched);
    return hospitals;
  }

  Future<List<Hospital>> _fetch(GeoPoint centre, int radiusKm) async {
    final query =
        '[out:json][timeout:40];'
        'nwr["amenity"="hospital"](around:${radiusKm * 1000},${centre.lat},${centre.lon});'
        'out center tags 200;';
    Object? lastError;
    for (final url in endpoints) {
      try {
        final res = await _client
            .post(
              Uri.parse(url),
              headers: {'User-Agent': userAgent},
              body: {'data': query},
            )
            .timeout(const Duration(seconds: 15));
        if (res.statusCode == 200) {
          return parseOverpass(jsonDecode(res.body) as Map<String, dynamic>);
        }
        lastError = http.ClientException('$url returned ${res.statusCode}');
      } catch (e) {
        lastError = e;
      }
    }
    // Fallback: one Nominatim search for hospitals inside a box around the
    // user. Less complete than Overpass (at most 50 results), but fast.
    try {
      final d =
          radiusKm / 111.0; // degrees of latitude per km, close enough here
      final uri = Uri.https('nominatim.openstreetmap.org', '/search', {
        'q': 'hospital',
        'format': 'jsonv2',
        'extratags': '1',
        'bounded': '1',
        'limit': '50',
        'viewbox':
            '${centre.lon - d},${centre.lat + d},${centre.lon + d},${centre.lat - d}',
      });
      final res = await _client
          .get(uri, headers: {'User-Agent': userAgent})
          .timeout(const Duration(seconds: 20));
      if (res.statusCode == 200) {
        return parseNominatim(jsonDecode(res.body) as List);
      }
      lastError = http.ClientException('Nominatim returned ${res.statusCode}');
    } catch (e) {
      lastError = e;
    }
    throw lastError;
  }

  static List<Hospital> parseNominatim(List<dynamic> json) => parseOverpass({
    'elements': [
      for (final e in json.cast<Map<String, dynamic>>())
        if (e['category'] == 'amenity' && e['type'] == 'hospital')
          {
            'lat': double.tryParse('${e['lat']}'),
            'lon': double.tryParse('${e['lon']}'),
            'tags': {
              'name': e['name'],
              ...?(e['extratags'] as Map?)?.cast<String, dynamic>(),
            },
          },
    ],
  });

  static List<Hospital> parseOverpass(Map<String, dynamic> json) {
    final seen = <String>{};
    final out = <Hospital>[];
    for (final e in (json['elements'] as List? ?? const [])) {
      final el = e as Map<String, dynamic>;
      final tags = (el['tags'] as Map?)?.cast<String, dynamic>() ?? const {};
      final lat = (el['lat'] ?? (el['center'] as Map?)?['lat']) as num?;
      final lon = (el['lon'] ?? (el['center'] as Map?)?['lon']) as num?;
      if (lat == null || lon == null) continue;
      if (!lat.isFinite || !lon.isFinite || lat.abs() > 90 || lon.abs() > 180) {
        continue;
      }
      final name = (tags['name:en'] ?? tags['name']) as String?;
      if (name == null || name.trim().isEmpty) continue;
      // The same hospital is often mapped as a node and a building outline.
      final key =
          '${name.toLowerCase()}@${lat.toStringAsFixed(3)},${lon.toStringAsFixed(3)}';
      if (!seen.add(key)) continue;
      final phone =
          (tags['phone'] ?? tags['contact:phone'] ?? tags['emergency:phone'])
              as String?;
      final address = [
        tags['addr:street'],
        tags['addr:city'] ?? tags['addr:district'],
      ].whereType<String>().join(', ');
      out.add(
        Hospital(
          name: name.trim(),
          location: GeoPoint(lat.toDouble(), lon.toDouble()),
          phone: phone?.split(';').first.trim(),
          emergency: switch (tags['emergency']) {
            'yes' => true,
            'no' => false,
            _ => null,
          },
          address: address.isEmpty ? null : address,
          speciality: tags['healthcare:speciality'] as String?,
        ),
      );
    }
    return out;
  }

  /// Nearest first, but hospitals that say they have no emergency department
  /// or that list only a specialty (eye, dental...) go after the rest.
  static List<Hospital> sortedByDistance(
    List<Hospital> hospitals,
    GeoPoint from,
  ) {
    int rank(Hospital h) => (h.emergency == false || h.isSpecialtyOnly) ? 1 : 0;
    final list = [...hospitals];
    list.sort((a, b) {
      final ra = rank(a), rb = rank(b);
      if (ra != rb) return ra - rb;
      return distanceMeters(
        from,
        a.location,
      ).compareTo(distanceMeters(from, b.location));
    });
    return list;
  }
}
