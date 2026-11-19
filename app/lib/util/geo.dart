import 'dart:math' as math;

/// A plain latitude/longitude pair in degrees.
class GeoPoint {
  const GeoPoint(this.lat, this.lon);
  final double lat;
  final double lon;

  Map<String, dynamic> toJson() => {'lat': lat, 'lon': lon};
  factory GeoPoint.fromJson(Map<String, dynamic> j) =>
      GeoPoint((j['lat'] as num).toDouble(), (j['lon'] as num).toDouble());

  String get mapsUrl =>
      'https://maps.google.com/?q=${lat.toStringAsFixed(5)},${lon.toStringAsFixed(5)}';

  @override
  String toString() => '${lat.toStringAsFixed(5)}, ${lon.toStringAsFixed(5)}';
}

const _earthRadiusM = 6371000.0;
double _rad(double deg) => deg * math.pi / 180;

/// Great-circle distance in metres.
double distanceMeters(GeoPoint a, GeoPoint b) {
  final dLat = _rad(b.lat - a.lat);
  final dLon = _rad(b.lon - a.lon);
  final h = math.pow(math.sin(dLat / 2), 2) +
      math.cos(_rad(a.lat)) * math.cos(_rad(b.lat)) * math.pow(math.sin(dLon / 2), 2);
  return 2 * _earthRadiusM * math.asin(math.min(1, math.sqrt(h)));
}

/// Initial bearing from [a] to [b], 0 to 360 degrees clockwise from north.
double bearingDegrees(GeoPoint a, GeoPoint b) {
  final y = math.sin(_rad(b.lon - a.lon)) * math.cos(_rad(b.lat));
  final x = math.cos(_rad(a.lat)) * math.sin(_rad(b.lat)) -
      math.sin(_rad(a.lat)) * math.cos(_rad(b.lat)) * math.cos(_rad(b.lon - a.lon));
  return (math.atan2(y, x) * 180 / math.pi + 360) % 360;
}

String compassPoint(double bearing) {
  const points = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'];
  return points[((bearing + 22.5) % 360 ~/ 45)];
}

String formatDistance(double meters) {
  if (meters < 1000) return '${(meters / 10).round() * 10} m';
  if (meters < 10000) return '${(meters / 1000).toStringAsFixed(1)} km';
  return '${(meters / 1000).round()} km';
}

String formatAge(Duration age) {
  if (age.inMinutes < 1) return 'just now';
  if (age.inMinutes < 60) return '${age.inMinutes} min ago';
  if (age.inHours < 24) return '${age.inHours} h ago';
  return '${age.inDays} d ago';
}
