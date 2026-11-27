import 'dart:io';
import 'dart:isolate';

import 'package:flutter/foundation.dart';
import 'package:image/image.dart' as img;
import 'package:path_provider/path_provider.dart';

import '../services/store.dart';
import 'anidex_catalog.dart';
import 'anidex_model.dart';

/// One caught species: how often it was scanned, the breeds found, and the
/// user's first photo of it (a small copy, on this phone only).
class CaughtEntry {
  CaughtEntry({required this.count, required this.breeds, required this.photo, required this.firstSeen, this.lastBreed});

  factory CaughtEntry.fromJson(Map<String, dynamic> j) => CaughtEntry(
        count: j['count'] as int,
        breeds: (j['breeds'] as Map).cast<String, int>(),
        photo: j['photo'] as String?,
        firstSeen: DateTime.parse(j['first_seen'] as String),
        lastBreed: j['last_breed'] as String?,
      );

  int count;
  final Map<String, int> breeds;
  String? photo;
  final DateTime firstSeen;
  String? lastBreed;

  Map<String, dynamic> toJson() => {
        'count': count,
        'breeds': breeds,
        'photo': photo,
        'first_seen': firstSeen.toIso8601String(),
        'last_breed': lastBreed,
      };
}

enum CatchKind { newSpecies, newBreed, seenAgain }

/// The user's AniDex, saved on the phone.
class AniDexCollection extends ChangeNotifier {
  AniDexCollection(this._store) {
    final saved = _store.readMap(_key) ?? const {};
    for (final e in saved.entries) {
      try {
        caught[e.key] = CaughtEntry.fromJson(e.value as Map<String, dynamic>);
      } catch (_) {}
    }
  }

  static const _key = 'anidex.caught';
  final Store _store;

  /// Scientific name to entry.
  final Map<String, CaughtEntry> caught = {};

  int get breedCount => caught.values.fold(0, (n, e) => n + e.breeds.length);
  int get scanCount => caught.values.fold(0, (n, e) => n + e.count);

  /// Records a scan and returns whether it was a new species, a new breed
  /// or a repeat.
  Future<CatchKind> record(AniDexScan scan, Uint8List photo) async {
    final s = scan.species;
    final breed = scan.breed?.$1.name;
    var entry = caught[s.scientific];
    final CatchKind kind;
    if (entry == null) {
      kind = CatchKind.newSpecies;
      entry = CaughtEntry(count: 0, breeds: {}, photo: await _savePhoto(s, photo), firstSeen: DateTime.now());
      caught[s.scientific] = entry;
    } else {
      kind = breed != null && !entry.breeds.containsKey(breed) ? CatchKind.newBreed : CatchKind.seenAgain;
      if (entry.photo == null || !File(entry.photo!).existsSync()) entry.photo = await _savePhoto(s, photo);
    }
    entry.count++;
    entry.lastBreed = breed ?? entry.lastBreed;
    if (breed != null) entry.breeds[breed] = (entry.breeds[breed] ?? 0) + 1;
    await _save();
    notifyListeners();
    return kind;
  }

  Future<void> clearAll() async {
    for (final e in caught.values) {
      if (e.photo != null) {
        try {
          await File(e.photo!).delete();
        } catch (_) {}
      }
    }
    caught.clear();
    await _save();
    notifyListeners();
  }

  Future<void> _save() => _store.writeMap(_key, {for (final e in caught.entries) e.key: e.value.toJson()});

  /// A 480 px JPEG of the photo for the grid and the card.
  static Future<String?> _savePhoto(AniDexSpecies s, Uint8List photo) async {
    try {
      final dir = Directory('${(await getApplicationDocumentsDirectory()).path}/anidex');
      await dir.create(recursive: true);
      // A new name each time, so Flutter's image cache never shows an old photo.
      final path = '${dir.path}/${s.no.toString().padLeft(3, '0')}-${DateTime.now().millisecondsSinceEpoch}.jpg';
      final jpeg = await Isolate.run(() {
        final decoded = img.decodeImage(photo);
        if (decoded == null) return null;
        final upright = img.bakeOrientation(decoded);
        final small = upright.width >= upright.height
            ? img.copyResize(upright, height: 480)
            : img.copyResize(upright, width: 480);
        return Uint8List.fromList(img.encodeJpg(small, quality: 85));
      });
      if (jpeg == null) return null;
      await File(path).writeAsBytes(jpeg, flush: true);
      return path;
    } catch (_) {
      return null;
    }
  }
}
