import 'dart:convert';
import 'dart:math' as math;

import 'package:sqflite/sqflite.dart';

import '../models/records.dart';
import '../util/geo.dart';

/// A persistent, location-indexed store of downloaded hospitals. It cannot establish
/// current ASV stock, and it does not imply coverage outside saved regions.
class HospitalIndex {
  HospitalIndex(this.db);
  final Database db;

  /// Version 1 used an SQLite R-tree, but the system SQLite on many Android
  /// phones is built without the R-tree module ("no such module: rtree"), so
  /// the database never opened and saved hospitals were lost. Version 2 uses a
  /// plain table with a (lat, lon) index, which every SQLite has; a few
  /// thousand hospitals are still found in well under a millisecond.
  static const _schemaVersion = 2;

  static Future<void> _createV2(DatabaseExecutor db) async {
    await db.execute(
      'CREATE TABLE IF NOT EXISTS hospitals (id INTEGER PRIMARY KEY, cache_key TEXT UNIQUE NOT NULL, '
      'payload TEXT NOT NULL, lat REAL NOT NULL, lon REAL NOT NULL, saved INTEGER NOT NULL)',
    );
    await db.execute('CREATE INDEX IF NOT EXISTS hospitals_lat_lon ON hospitals(lat, lon)');
  }

  static Future<HospitalIndex> open({String? path}) async {
    final db = await openDatabase(
      path ?? '${await getDatabasesPath()}/hospitals-v1.sqlite3',
      version: _schemaVersion,
      onCreate: (db, _) => _createV2(db),
      onUpgrade: (db, from, _) async {
        if (from < 2) {
          // Phones whose SQLite had the R-tree module: drop its table and
          // triggers, keep the saved hospitals.
          for (final t in ['hospital_insert', 'hospital_update', 'hospital_delete']) {
            await db.execute('DROP TRIGGER IF EXISTS $t');
          }
          try {
            await db.execute('DROP TABLE IF EXISTS hospital_bounds');
          } on DatabaseException {
            // Without the R-tree module the virtual table cannot even be
            // dropped; it is unused from here on, so leave it.
          }
          await _createV2(db);
        }
      },
    );
    return HospitalIndex(db);
  }

  Future<void> save(List<Hospital> hospitals, DateTime fetched) async {
    await db.transaction((tx) async {
      for (final h in hospitals) {
        if (!h.location.lat.isFinite ||
            !h.location.lon.isFinite ||
            h.location.lat.abs() > 90 ||
            h.location.lon.abs() > 180) {
          continue;
        }
        final key =
            '${h.name.trim().toLowerCase()}@${h.location.lat.toStringAsFixed(4)},${h.location.lon.toStringAsFixed(4)}';
        final row = {
          'cache_key': key,
          'payload': jsonEncode(h.toJson()),
          'lat': h.location.lat,
          'lon': h.location.lon,
          'saved': fetched.toUtc().millisecondsSinceEpoch,
        };
        final existing = await tx.query(
          'hospitals',
          columns: ['id'],
          where: 'cache_key=?',
          whereArgs: [key],
        );
        if (existing.isEmpty) {
          await tx.insert('hospitals', row);
        } else {
          await tx.update(
            'hospitals',
            row,
            where: 'id=?',
            whereArgs: [existing.single['id']],
          );
        }
      }
    });
  }

  Future<({List<Hospital> hospitals, DateTime? oldest})> near(
    GeoPoint centre, {
    double radiusKm = 50,
  }) async {
    if (!centre.lat.isFinite ||
        !centre.lon.isFinite ||
        centre.lat.abs() > 90 ||
        centre.lon.abs() > 180 ||
        !radiusKm.isFinite ||
        radiusKm <= 0 ||
        radiusKm > 500) {
      throw const FormatException('Invalid hospital search region.');
    }
    final latDelta = radiusKm / 110.0;
    final lonDelta = math.min(
      180.0,
      radiusKm /
          (110 * math.max(0.001, math.cos(centre.lat * math.pi / 180).abs())),
    );
    final low = centre.lon - lonDelta, high = centre.lon + lonDelta;
    final ranges = low < -180
        ? [(low + 360, 180.0), (-180.0, high)]
        : high > 180
        ? [(low, 180.0), (-180.0, high - 360)]
        : [(low, high)];
    final matches = <int, Map<String, Object?>>{};
    for (final range in ranges) {
      final rows = await db.rawQuery(
        'SELECT * FROM hospitals WHERE lat BETWEEN ? AND ? AND lon BETWEEN ? AND ?',
        [centre.lat - latDelta, centre.lat + latDelta, range.$1, range.$2],
      );
      for (final row in rows) {
        matches[row['id'] as int] = row;
      }
    }
    final out = <Hospital>[];
    DateTime? oldest;
    for (final row in matches.values) {
      final h = Hospital.fromJson(
        jsonDecode(row['payload'] as String) as Map<String, dynamic>,
      );
      if (distanceMeters(centre, h.location) > radiusKm * 1000) continue;
      out.add(h);
      final saved = DateTime.fromMillisecondsSinceEpoch(
        row['saved'] as int,
        isUtc: true,
      );
      if (oldest == null || saved.isBefore(oldest)) oldest = saved;
    }
    return (hospitals: out, oldest: oldest);
  }
}
