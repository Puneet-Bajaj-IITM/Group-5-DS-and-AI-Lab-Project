import 'dart:async';
import 'dart:convert';

import 'package:geocoding/geocoding.dart' as geo;
import 'package:http/http.dart' as http;

import '../util/geo.dart';
import 'store.dart';

/// Turns a position into a street address a person can read out on a call.
///
/// Tries Android's own geocoder first, then OpenStreetMap's reverse lookup
/// (Nominatim). Answers are kept on the phone per ~100 m square, so the last
/// address still shows without signal. When nothing is known the caller shows
/// coordinates instead.
class AddressService {
  AddressService(this._store, {http.Client? client, Future<String?> Function(GeoPoint)? platform})
      : _client = client ?? http.Client(),
        _platform = platform ?? _platformLookup;

  final Store _store;
  final http.Client _client;
  final Future<String?> Function(GeoPoint) _platform;

  static const _cacheKey = 'address.cache';
  static const _maxCached = 60;
  static const userAgent = 'SerpentGuard/0.1 (student research app)';

  /// Cache key: the position rounded to 0.001 degrees (about 110 m).
  static String keyFor(GeoPoint p) => '${p.lat.toStringAsFixed(3)},${p.lon.toStringAsFixed(3)}';

  String? cached(GeoPoint p) => _store.readMap(_cacheKey)?[keyFor(p)] as String?;

  Future<String?> lookup(GeoPoint p) async {
    final hit = cached(p);
    if (hit != null) return hit;
    String? found;
    try {
      found = await _platform(p).timeout(const Duration(seconds: 6));
    } catch (_) {}
    if (found == null || found.isEmpty) {
      try {
        found = await _nominatim(p);
      } catch (_) {}
    }
    if (found != null && found.isNotEmpty) await _remember(p, found);
    return found;
  }

  Future<void> _remember(GeoPoint p, String address) async {
    final map = Map<String, dynamic>.from(_store.readMap(_cacheKey) ?? const {});
    map.remove(keyFor(p));
    map[keyFor(p)] = address;
    while (map.length > _maxCached) {
      map.remove(map.keys.first);
    }
    await _store.writeMap(_cacheKey, map);
  }

  static Future<String?> _platformLookup(GeoPoint p) async {
    final g = geo.Geocoding();
    if (!await g.isPresent()) return null;
    final marks = await g.placemarkFromCoordinates(p.lat, p.lon);
    if (marks.isEmpty) return null;
    final m = marks.first;
    return formatParts([
      [m.subThoroughfare, m.thoroughfare].whereType<String>().where((s) => s.isNotEmpty).join(' '),
      m.subLocality,
      m.locality,
      m.postalCode,
    ]);
  }

  Future<String?> _nominatim(GeoPoint p) async {
    final uri = Uri.https('nominatim.openstreetmap.org', '/reverse', {
      'format': 'jsonv2',
      'lat': '${p.lat}',
      'lon': '${p.lon}',
      'zoom': '18',
      'addressdetails': '1',
    });
    final res = await _client.get(uri, headers: {'User-Agent': userAgent}).timeout(const Duration(seconds: 8));
    if (res.statusCode != 200) return null;
    return fromNominatim(jsonDecode(res.body) as Map<String, dynamic>);
  }

  /// "12, MG Road, Shanthala Nagar, Bengaluru 560001" from Nominatim's parts.
  static String? fromNominatim(Map<String, dynamic> json) {
    final a = (json['address'] as Map?)?.cast<String, dynamic>();
    if (a == null) return json['display_name'] as String?;
    String? pick(List<String> keys) => keys.map((k) => a[k] as String?).firstWhere((v) => v != null && v.isNotEmpty, orElse: () => null);
    return formatParts([
      [a['house_number'], pick(['road', 'pedestrian', 'footway', 'path'])].whereType<String>().join(' '),
      pick(['neighbourhood', 'suburb', 'quarter', 'hamlet']),
      pick(['city', 'town', 'village', 'county']),
      a['postcode'] as String?,
    ]) ?? json['display_name'] as String?;
  }

  /// Joins the non-empty parts once each, in order.
  static String? formatParts(List<String?> parts) {
    final seen = <String>{};
    final out = [
      for (final p in parts)
        if (p != null && p.trim().isNotEmpty && seen.add(p.trim().toLowerCase())) p.trim()
    ];
    return out.isEmpty ? null : out.join(', ');
  }
}
