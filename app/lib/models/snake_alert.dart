import 'dart:math' as math;
import 'dart:typed_data';

import '../util/geo.dart';
import 'verdict.dart';

/// A short-lived "snake seen here" warning passed between nearby phones over
/// Bluetooth Low Energy. It carries no name, phone number or device id: only a
/// random alert id, the species, the verdict, a location rounded to about
/// 100 m and the time.
class SnakeAlert {
  const SnakeAlert({
    required this.id,
    required this.speciesId,
    required this.verdict,
    required this.location,
    required this.time,
    this.hops = 0,
  });

  /// Random 32-bit id, used to drop duplicates and relays of our own alerts.
  final int id;

  /// Class id 0..183, or [unknownSpecies].
  final int speciesId;
  final Verdict verdict;
  final GeoPoint location;
  final DateTime time;

  /// How many phones have passed it on. 0 = from the phone that saw the snake.
  final int hops;

  static const unknownSpecies = 255;

  /// Alerts older than this are dropped and never relayed.
  static const lifetime = Duration(minutes: 60);

  /// An alert is passed on at most this many times.
  static const maxHops = 2;

  bool isExpired(DateTime now) => now.difference(time) > lifetime;

  SnakeAlert relayed() => SnakeAlert(
        id: id,
        speciesId: speciesId,
        verdict: verdict,
        location: location,
        time: time,
        hops: hops + 1,
      );

  static int newId([math.Random? random]) =>
      (random ?? math.Random.secure()).nextInt(1 << 32);

  Map<String, dynamic> toJson() => {
        'id': id,
        'species': speciesId,
        'verdict': verdict.code,
        'loc': location.toJson(),
        'time': time.toIso8601String(),
        'hops': hops,
      };
  factory SnakeAlert.fromJson(Map<String, dynamic> j) => SnakeAlert(
        id: j['id'] as int,
        speciesId: j['species'] as int,
        verdict: Verdict.fromCode(j['verdict'] as int),
        location: GeoPoint.fromJson(j['loc'] as Map<String, dynamic>),
        time: DateTime.parse(j['time'] as String),
        hops: j['hops'] as int,
      );
}

/// The 20-byte payload inside the advertisement's manufacturer data.
///
/// | bytes | field                                         |
/// |-------|-----------------------------------------------|
/// | 0-1   | magic "SG"                                    |
/// | 2     | format version (1)                            |
/// | 3-6   | alert id, uint32 big-endian                   |
/// | 7     | species class id, 255 = unknown               |
/// | 8     | verdict code                                  |
/// | 9     | hop count                                     |
/// | 10-12 | latitude × 1000, int24 (about 110 m)          |
/// | 13-15 | longitude × 1000, int24                       |
/// | 16-19 | minutes since the Unix epoch, uint32          |
///
/// A legacy advertisement leaves 24 bytes for manufacturer data, so this fits.
class AlertCodec {
  /// 0xFFFF is the Bluetooth SIG id reserved for testing; a released app
  /// needs the team's own company id.
  static const manufacturerId = 0xFFFF;
  static const magic = [0x53, 0x47];
  static const version = 1;
  static const length = 20;

  static Uint8List encode(SnakeAlert a) {
    final b = ByteData(length);
    b.setUint8(0, magic[0]);
    b.setUint8(1, magic[1]);
    b.setUint8(2, version);
    b.setUint32(3, a.id);
    b.setUint8(7, a.speciesId.clamp(0, 255));
    b.setUint8(8, a.verdict.code);
    b.setUint8(9, a.hops.clamp(0, 255));
    _setInt24(b, 10, (a.location.lat * 1000).round());
    _setInt24(b, 13, (a.location.lon * 1000).round());
    b.setUint32(16, a.time.toUtc().millisecondsSinceEpoch ~/ 60000);
    return b.buffer.asUint8List();
  }

  /// Returns null for anything that is not a well-formed SerpentGuard alert.
  static SnakeAlert? decode(List<int> bytes) {
    if (bytes.length < length) return null;
    if (bytes[0] != magic[0] || bytes[1] != magic[1] || bytes[2] != version) {
      return null;
    }
    final b = ByteData.sublistView(Uint8List.fromList(bytes));
    final verdictCode = b.getUint8(8);
    if (verdictCode >= Verdict.values.length) return null;
    final lat = _getInt24(b, 10) / 1000;
    final lon = _getInt24(b, 13) / 1000;
    if (lat.abs() > 90 || lon.abs() > 180) return null;
    return SnakeAlert(
      id: b.getUint32(3),
      speciesId: b.getUint8(7),
      verdict: Verdict.fromCode(verdictCode),
      hops: b.getUint8(9),
      location: GeoPoint(lat, lon),
      time: DateTime.fromMillisecondsSinceEpoch(b.getUint32(16) * 60000, isUtc: true)
          .toLocal(),
    );
  }

  static void _setInt24(ByteData b, int offset, int value) {
    final v = value & 0xFFFFFF;
    b.setUint8(offset, (v >> 16) & 0xFF);
    b.setUint8(offset + 1, (v >> 8) & 0xFF);
    b.setUint8(offset + 2, v & 0xFF);
  }

  static int _getInt24(ByteData b, int offset) {
    final v = (b.getUint8(offset) << 16) | (b.getUint8(offset + 1) << 8) | b.getUint8(offset + 2);
    return v & 0x800000 != 0 ? v - 0x1000000 : v;
  }
}
