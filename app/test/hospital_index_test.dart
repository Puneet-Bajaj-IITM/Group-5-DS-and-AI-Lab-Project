import 'dart:io';
import 'dart:convert';

import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:serpentguard/models/records.dart';
import 'package:serpentguard/services/hospital_index.dart';
import 'package:serpentguard/services/hospital_service.dart';
import 'package:serpentguard/services/store.dart';
import 'package:serpentguard/util/geo.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:sqflite_common_ffi/sqflite_ffi.dart';

void main() {
  setUpAll(() {
    sqfliteFfiInit();
    databaseFactory = databaseFactoryFfi;
  });
  test(
    'the index keeps saved regions across restart and excludes distant hospitals',
    () async {
      final folder = await Directory.systemTemp.createTemp('sg-hospital-test-');
      final path = '${folder.path}/hospitals.sqlite3';
      var index = await HospitalIndex.open(path: path);
      try {
        final date = DateTime.utc(2026, 11, 18);
        await index.save([
          const Hospital(
            name: 'Test Bengaluru facility',
            location: GeoPoint(12.98, 77.60),
          ),
          const Hospital(
            name: 'Test Chennai facility',
            location: GeoPoint(13.08, 80.27),
          ),
        ], date);
        await index.db.close();
        index = await HospitalIndex.open(path: path);
        final near = await index.near(const GeoPoint(12.97, 77.59));
        expect(near.hospitals.map((h) => h.name), ['Test Bengaluru facility']);
        expect(near.oldest, date);
        await index.save([
          const Hospital(
            name: 'Test Bengaluru facility',
            location: GeoPoint(12.98, 77.60),
            phone: 'test',
          ),
        ], date);
        expect(
          (await index.near(const GeoPoint(12.97, 77.59)))
              .hospitals
              .single
              .phone,
          'test',
        );
        expect(
          (await index.db.rawQuery('SELECT count(*) AS n FROM hospitals'))
              .single['n'],
          2,
          reason: 'saving the same hospital again updates it, not duplicates it',
        );
        await index.db.delete('hospitals');
        expect((await index.near(const GeoPoint(12.97, 77.59))).hospitals, isEmpty);
      } finally {
        await index.db.close();
        await folder.delete(recursive: true);
      }
    },
  );

  test(
    'failed wider search preserves the successfully downloaded nearby hospital',
    () async {
      SharedPreferences.setMockInitialValues({});
      final folder = await Directory.systemTemp.createTemp(
        'sg-hospital-service-',
      );
      final index = await HospitalIndex.open(
        path: '${folder.path}/hospitals.sqlite3',
      );
      var calls = 0;
      final client = MockClient((req) async {
        calls++;
        if (calls == 1) {
          return http.Response(
            jsonEncode({
              'elements': [
                {
                  'lat': 12.98,
                  'lon': 77.60,
                  'tags': {'name': 'Test near facility'},
                },
              ],
            }),
            200,
          );
        }
        throw const SocketException('offline');
      });
      try {
        final service = HospitalService(
          await Store.open(),
          client: client,
          openIndex: () async => index,
        );
        final hospitals = await service.refresh(const GeoPoint(12.97, 77.59));
        expect(hospitals.single.name, 'Test near facility');
        expect(
          (await service.savedNear(const GeoPoint(12.97, 77.59)))
              .hospitals
              .single
              .name,
          'Test near facility',
        );
      } finally {
        client.close();
        await index.db.close();
        await folder.delete(recursive: true);
      }
    },
  );

  test('a version 1 database with the old R-tree opens as version 2 and keeps its hospitals', () async {
    final folder = await Directory.systemTemp.createTemp('sg-hospital-upgrade-');
    final path = '${folder.path}/hospitals.sqlite3';
    final v1 = await openDatabase(path, version: 1, onCreate: (db, _) async {
      await db.execute(
        'CREATE TABLE hospitals (id INTEGER PRIMARY KEY, cache_key TEXT UNIQUE NOT NULL, '
        'payload TEXT NOT NULL, lat REAL NOT NULL, lon REAL NOT NULL, saved INTEGER NOT NULL)',
      );
      await db.execute('CREATE VIRTUAL TABLE hospital_bounds USING rtree(id,min_lat,max_lat,min_lon,max_lon)');
      await db.execute(
        'CREATE TRIGGER hospital_insert AFTER INSERT ON hospitals BEGIN '
        'INSERT INTO hospital_bounds VALUES(new.id,new.lat,new.lat,new.lon,new.lon); END',
      );
    });
    const h = Hospital(name: 'Old saved facility', location: GeoPoint(12.98, 77.60));
    await v1.insert('hospitals', {
      'cache_key': 'old', 'payload': jsonEncode(h.toJson()), 'lat': 12.98, 'lon': 77.60, 'saved': 0,
    });
    await v1.close();

    final index = await HospitalIndex.open(path: path);
    try {
      expect(await index.db.getVersion(), 2);
      final near = await index.near(const GeoPoint(12.97, 77.59));
      expect(near.hospitals.map((x) => x.name), ['Old saved facility']);
      final triggers = await index.db.rawQuery("SELECT name FROM sqlite_master WHERE type='trigger'");
      expect(triggers, isEmpty, reason: 'old R-tree triggers are dropped');
      await index.save([const Hospital(name: 'New facility', location: GeoPoint(12.99, 77.61))], DateTime.utc(2026, 11, 21));
      expect((await index.near(const GeoPoint(12.97, 77.59))).hospitals.length, 2);
    } finally {
      await index.db.close();
      await folder.delete(recursive: true);
    }
  });
}
