import 'dart:convert';
import 'dart:math' as math;

import 'package:sqflite/sqflite.dart';

import '../models/records.dart';
import '../util/geo.dart';

/// A persistent spatial index of downloaded hospitals. It cannot establish
/// current ASV stock, and it does not imply coverage outside saved regions.
class HospitalIndex {
  HospitalIndex(this.db);
  final Database db;

  static Future<HospitalIndex> open({String? path}) async {
    final db = await openDatabase(
      path ?? '${await getDatabasesPath()}/hospitals-v1.sqlite3',
      version: 1,
      onCreate: (db, _) async {
        await db.execute(
          'CREATE TABLE hospitals (id INTEGER PRIMARY KEY, cache_key TEXT UNIQUE NOT NULL, '
          'payload TEXT NOT NULL, lat REAL NOT NULL, lon REAL NOT NULL, saved INTEGER NOT NULL)',
        );
        await db.execute(
          'CREATE VIRTUAL TABLE hospital_bounds USING rtree(id,min_lat,max_lat,min_lon,max_lon)',
        );
        await db.execute(
          'CREATE TRIGGER hospital_insert AFTER INSERT ON hospitals BEGIN '
          'INSERT INTO hospital_bounds VALUES(new.id,new.lat,new.lat,new.lon,new.lon); END',
        );
        await db.execute(
          'CREATE TRIGGER hospital_update AFTER UPDATE ON hospitals BEGIN '
          'UPDATE hospital_bounds SET min_lat=new.lat,max_lat=new.lat,min_lon=new.lon,max_lon=new.lon WHERE id=new.id; END',
        );
        await db.execute(
          'CREATE TRIGGER hospital_delete AFTER DELETE ON hospitals BEGIN '
          'DELETE FROM hospital_bounds WHERE id=old.id; END',
        );
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
        'SELECT h.* FROM hospitals h JOIN hospital_bounds r ON h.id=r.id '
        'WHERE r.max_lat>=? AND r.min_lat<=? AND r.max_lon>=? AND r.min_lon<=?',
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
