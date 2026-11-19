import 'dart:convert';
import 'dart:math';
import 'dart:typed_data';

import 'package:crypto/crypto.dart';
import 'package:image/image.dart' as img;
import 'package:sqflite/sqflite.dart';

import 'voting_api.dart';

String votingId() {
  final random = Random.secure();
  final b = List<int>.generate(16, (_) => random.nextInt(256));
  b[6] = (b[6] & 15) | 64;
  b[8] = (b[8] & 63) | 128;
  final h = b.map((v) => v.toRadixString(16).padLeft(2, '0')).join();
  return '${h.substring(0, 8)}-${h.substring(8, 12)}-${h.substring(12, 16)}-'
      '${h.substring(16, 20)}-${h.substring(20)}';
}

/// Rotate and resize pixels, then write a fresh JPEG with no source metadata.
Uint8List votingJpeg(Uint8List bytes) {
  var decoded = img.decodeImage(bytes);
  if (decoded == null) {
    throw const FormatException('This image cannot be decoded.');
  }
  decoded = img.bakeOrientation(decoded);
  if (max(decoded.width, decoded.height) > 1600) {
    decoded = img.copyResize(
      decoded,
      width: decoded.width >= decoded.height ? 1600 : null,
      height: decoded.height > decoded.width ? 1600 : null,
    );
  }
  // Copy pixels into a clean image: EXIF/GPS and embedded previews are omitted.
  final clean = img.Image(
    width: decoded.width,
    height: decoded.height,
    numChannels: 3,
  );
  img.compositeImage(clean, decoded);
  for (final quality in [90, 80, 65, 50]) {
    final jpeg = Uint8List.fromList(img.encodeJpg(clean, quality: quality));
    if (jpeg.length <= 1024 * 1024) return jpeg;
  }
  throw const FormatException('Image is too large. Choose a smaller crop.');
}

class VotingQueue {
  static const draftAccount = 'local-unsubmitted-drafts';
  VotingQueue(this.db);
  final Database db;
  bool _syncing = false;

  static Future<VotingQueue> open({String? path}) async {
    final db = await openDatabase(
      path ?? '${await getDatabasesPath()}/community-v1.sqlite3',
      version: 3,
      onUpgrade: (db, oldVersion, _) async {
        if (oldVersion < 2) await _createConsensus(db);
        if (oldVersion < 3) await _createSyncState(db);
      },
      onCreate: (db, _) async {
        await db.execute(
          'CREATE TABLE jobs (id TEXT PRIMARY KEY, account TEXT NOT NULL, '
          'kind TEXT NOT NULL, sighting TEXT NOT NULL, payload TEXT NOT NULL, '
          "state TEXT NOT NULL DEFAULT 'pending', error TEXT, created INTEGER NOT NULL)",
        );
        await db.execute(
          'CREATE TABLE media (job TEXT NOT NULL, slot INTEGER NOT NULL, '
          'jpeg BLOB NOT NULL, PRIMARY KEY(job,slot))',
        );
        await db.execute(
          'CREATE TABLE feed (account TEXT NOT NULL, id TEXT NOT NULL, '
          'payload TEXT NOT NULL, PRIMARY KEY(account,id))',
        );
        await db.execute(
          'CREATE TABLE feed_images (account TEXT NOT NULL, id TEXT NOT NULL, '
          'slot INTEGER NOT NULL, jpeg BLOB NOT NULL, PRIMARY KEY(account,id,slot))',
        );
        await _createConsensus(db);
        await _createSyncState(db);
      },
    );
    return VotingQueue(db);
  }

  static Future<void> _createConsensus(Database db) => db.execute(
    'CREATE TABLE consensus (account TEXT NOT NULL, id TEXT NOT NULL, '
    'payload TEXT NOT NULL, PRIMARY KEY(account,id))',
  );

  static Future<void> _createSyncState(Database db) async {
    await db.execute(
      'CREATE TABLE sync_leases (account TEXT PRIMARY KEY, owner TEXT NOT NULL, expires INTEGER NOT NULL)',
    );
    await db.execute(
      'CREATE TABLE feed_pages (account TEXT PRIMARY KEY, cursor INTEGER, fetched INTEGER NOT NULL)',
    );
  }

  Future<bool> _lease(String account, String owner) => db.transaction((
    tx,
  ) async {
    final now = DateTime.now().millisecondsSinceEpoch;
    final rows = await tx.query(
      'sync_leases',
      where: 'account=?',
      whereArgs: [account],
    );
    if (rows.isNotEmpty && (rows.single['expires'] as int) > now) return false;
    await tx.insert('sync_leases', {
      'account': account,
      'owner': owner,
      'expires': now + 120000,
    }, conflictAlgorithm: ConflictAlgorithm.replace);
    return true;
  });

  Future<bool> _renew(String account, String owner) async =>
      await db.update(
        'sync_leases',
        {'expires': DateTime.now().millisecondsSinceEpoch + 120000},
        where: 'account=? AND owner=?',
        whereArgs: [account, owner],
      ) ==
      1;

  Future<bool> hasPending(String account) async => (await db.query(
    'jobs',
    columns: ['id'],
    where: "account=? AND state='pending'",
    whereArgs: [account],
    limit: 1,
  )).isNotEmpty;

  Future<int?> nextCursor(String account) async {
    final rows = await db.query(
      'feed_pages',
      where: 'account=?',
      whereArgs: [account],
    );
    return rows.isEmpty ? null : rows.single['cursor'] as int?;
  }

  Future<void> _saveConsensus(
    String account,
    String id,
    Map<String, dynamic> value,
  ) => db.insert('consensus', {
    'account': account,
    'id': id,
    'payload': jsonEncode(value),
  }, conflictAlgorithm: ConflictAlgorithm.replace);

  Future<({Map<String, dynamic>? value, bool cached})> consensus(
    VotingApi api,
    String id,
  ) async {
    try {
      final result = await api.json('GET', '/sightings/$id/consensus');
      await _saveConsensus(api.account, id, result);
      return (value: result, cached: false);
    } catch (_) {
      final rows = await db.query(
        'consensus',
        where: 'account=? AND id=?',
        whereArgs: [api.account, id],
      );
      return (
        value: rows.isEmpty
            ? null
            : jsonDecode(rows.single['payload'] as String)
                  as Map<String, dynamic>,
        cached: true,
      );
    }
  }

  Future<List<Map<String, Object?>>> jobs(String account) => db.query(
    'jobs',
    where: 'account = ?',
    whereArgs: [account],
    orderBy: 'created DESC',
  );

  Future<void> claimDrafts(String account) => db.update('jobs', {'account': account},
      where: "account=? AND kind='sighting' AND state='pending'", whereArgs: [draftAccount]);

  Future<String> sighting(
    String account,
    List<Uint8List> jpegs, {
    required bool trainingConsent,
    required DateTime capturedAt,
    List<int>? offsets,
    String? coarseLocation,
  }) async {
    if (jpegs.isEmpty ||
        jpegs.length > 3 ||
        (offsets == null && jpegs.length != 1) ||
        (offsets != null &&
            (offsets.length != jpegs.length ||
                offsets.toSet().length != offsets.length ||
                offsets.any((v) => v < 0)))) {
      throw const FormatException(
        'Use one photo or 1–3 distinct video stills.',
      );
    }
    if (jpegs.any((b) => b.isEmpty || b.length > 1024 * 1024)) {
      throw const FormatException('Each JPEG must be at most 1 MiB.');
    }
    final hashes = jpegs.map((b) => sha256.convert(b).toString()).toList();
    if (hashes.toSet().length != hashes.length) {
      throw const FormatException('Choose distinct views of the same snake.');
    }
    final id = votingId();
    final payload = <String, dynamic>{
      'id': id,
      'trigger': 'unknown',
      'source': offsets == null ? 'photo' : 'live_video',
      'snake_found': true,
      'captured_at': capturedAt.toUtc().toIso8601String(),
      'training_consent': trainingConsent,
      if (coarseLocation != null && coarseLocation.isNotEmpty)
        'coarse_location': coarseLocation,
      'evidence': [
        for (var i = 0; i < jpegs.length; i++)
          {'sha256': hashes[i], if (offsets != null) 'offset_ms': offsets[i]},
      ],
    };
    // Media and immutable request are committed together before any network IO.
    await db.transaction((tx) async {
      await tx.insert('jobs', {
        'id': id,
        'account': account,
        'kind': 'sighting',
        'sighting': id,
        'payload': jsonEncode(payload),
        'created': DateTime.now().millisecondsSinceEpoch,
      });
      for (var i = 0; i < jpegs.length; i++) {
        await tx.insert('media', {'job': id, 'slot': i, 'jpeg': jpegs[i]});
      }
    });
    return id;
  }

  Future<void> vote(
    String account,
    String sighting,
    String rank,
    String label, {
    String? previous,
  }) async {
    final id = votingId();
    await db.transaction((tx) async {
      final existing = await tx.query(
        'jobs',
        where: "account=? AND sighting=? AND kind='vote' AND state!='done'",
        whereArgs: [account, sighting],
      );
      if (existing.isNotEmpty) {
        throw StateError('Sync or reconcile the queued vote first.');
      }
      await tx.insert('jobs', {
        'id': id,
        'account': account,
        'kind': 'vote',
        'sighting': sighting,
        'payload': jsonEncode({
          'id': id,
          'rank': rank,
          'label': label,
          'previous_vote_id': ?previous,
        }),
        'created': DateTime.now().millisecondsSinceEpoch,
      });
    });
  }

  Future<bool> sync(
    VotingApi api, {
    Future<bool> Function()? allowed,
    int maxJobs = 20,
  }) async {
    if (_syncing) return false;
    _syncing = true;
    final owner = votingId();
    try {
      if (!await _lease(api.account, owner)) return false;
      final pending = await db.query(
        'jobs',
        where: "account=? AND state='pending'",
        whereArgs: [api.account],
        orderBy: 'created',
        limit: maxJobs,
      );
      for (final job in pending) {
        if (allowed != null && !await allowed()) return true;
        if (!await _renew(api.account, owner)) return false;
        final id = job['id'] as String;
        final sighting = job['sighting'] as String;
        final payload =
            jsonDecode(job['payload'] as String) as Map<String, dynamic>;
        try {
          if (job['kind'] == 'sighting') {
            await api.json('POST', '/sightings', payload);
            final media = await db.query(
              'media',
              where: 'job=?',
              whereArgs: [id],
              orderBy: 'slot',
            );
            if (media.length != (payload['evidence'] as List).length) {
              throw StateError('Saved evidence is incomplete.');
            }
            Map<String, dynamic>? last;
            for (final frame in media) {
              if (allowed != null && !await allowed()) return true;
              if (!await _renew(api.account, owner)) return false;
              last = await api.upload(
                sighting,
                frame['slot'] as int,
                frame['jpeg'] as Uint8List,
              );
            }
            if (last?['ready'] != true) {
              throw StateError('Server has not confirmed all evidence.');
            }
          } else {
            final result = await api.json(
              'POST',
              '/sightings/$sighting/votes',
              payload,
            );
            await _saveConsensus(api.account, sighting, result);
            // Keep the cached predecessor current even if the sighting has
            // already left the unresolved feed after this vote.
            final rows = await db.query(
              'feed',
              where: 'account=? AND id=?',
              whereArgs: [api.account, sighting],
            );
            if (rows.isNotEmpty) {
              final item = jsonDecode(
                rows.single['payload'] as String,
              ) as Map<String, dynamic>;
              item['your_vote'] = {
                'id': id,
                'rank': payload['rank'],
                'label': payload['label'],
              };
              await db.update(
                'feed',
                {'payload': jsonEncode(item)},
                where: 'account=? AND id=?',
                whereArgs: [api.account, sighting],
              );
            }
          }
          await db.transaction((tx) async {
            await tx.update(
              'jobs',
              {'state': 'done', 'error': null},
              where: 'id=?',
              whereArgs: [id],
            );
            await tx.delete('media', where: 'job=?', whereArgs: [id]);
          });
        } catch (error) {
          final permanent = error is VotingApiError && !error.retryable;
          final message = error is VotingApiError
              ? error.toString()
              : 'Connection interrupted; saved for retry.';
          await db.update(
            'jobs',
            {'state': permanent ? 'blocked' : 'pending', 'error': message},
            where: 'id=?',
            whereArgs: [id],
          );
          // An unavailable server or invalid session will affect every job.
          if (!permanent || error.status == 401) break;
        }
      }
      return !await hasPending(api.account);
    } finally {
      await db.delete(
        'sync_leases',
        where: 'account=? AND owner=?',
        whereArgs: [api.account, owner],
      );
      _syncing = false;
    }
  }

  Future<List<Map<String, dynamic>>> feed(String account) async =>
      (await db.query('feed', where: 'account=?', whereArgs: [account]))
          .map(
            (r) => jsonDecode(r['payload'] as String) as Map<String, dynamic>,
          )
          .toList();

  /// Cache one page and its stills. At most 100 encounters are kept on-device.
  /// A failed image
  /// fetch leaves it unavailable, never substituted with a different sighting.
  Future<void> refresh(VotingApi api, {bool more = false}) async {
    final cursor = more ? await nextCursor(api.account) : 0;
    if (cursor == null) return;
    final page = await api.json(
      'GET',
      '/sightings/review-feed?limit=20&cursor=$cursor',
    );
    final items = (page['items'] as List).cast<Map<String, dynamic>>();
    await db.transaction((tx) async {
      if (!more) {
        await tx.delete('feed', where: 'account=?', whereArgs: [api.account]);
        await tx.delete(
          'feed_images',
          where: 'account=?',
          whereArgs: [api.account],
        );
      }
      for (final item in items) {
        await tx.insert('feed', {
          'account': api.account,
          'id': item['id'],
          'payload': jsonEncode(item),
        }, conflictAlgorithm: ConflictAlgorithm.replace);
      }
      await tx.insert('feed_pages', {
        'account': api.account,
        'cursor': page['next_cursor'],
        'fetched': DateTime.now().millisecondsSinceEpoch,
      }, conflictAlgorithm: ConflictAlgorithm.replace);
      final all = await tx.query(
        'feed',
        columns: ['id'],
        where: 'account=?',
        whereArgs: [api.account],
        orderBy: 'rowid DESC',
      );
      for (final old in all.skip(100)) {
        await tx.delete(
          'feed',
          where: 'account=? AND id=?',
          whereArgs: [api.account, old['id']],
        );
        await tx.delete(
          'feed_images',
          where: 'account=? AND id=?',
          whereArgs: [api.account, old['id']],
        );
      }
    });
    for (final item in items) {
      final evidence = item['evidence'] as List? ?? [];
      for (var slot = 0; slot < min(evidence.length, 3); slot++) {
        try {
          final jpeg = await api.evidence(item['id'] as String, slot);
          await db.insert('feed_images', {
            'account': api.account,
            'id': item['id'],
            'slot': slot,
            'jpeg': jpeg,
          }, conflictAlgorithm: ConflictAlgorithm.replace);
        } catch (_) {
          // The screen explains unavailable evidence and disables voting.
        }
      }
    }
  }

  Future<List<Uint8List>> images(String account, String id) async =>
      (await db.query(
        'feed_images',
        where: 'account=? AND id=?',
        whereArgs: [account, id],
        orderBy: 'slot',
      )).map((row) => row['jpeg'] as Uint8List).toList();

  /// Explicit user confirmation is required after retrieving /my-vote.
  Future<void> reconcile(
    String jobId,
    String account,
    String? currentVote,
  ) async {
    await db.transaction((tx) async {
      final rows = await tx.query(
        'jobs',
        where: "id=? AND account=? AND kind='vote' AND state='blocked'",
        whereArgs: [jobId, account],
      );
      if (rows.isEmpty) {
        throw StateError('This conflict is no longer available.');
      }
      final row = rows.single;
      final payload =
          jsonDecode(row['payload'] as String) as Map<String, dynamic>;
      payload['id'] = votingId();
      payload.remove('previous_vote_id');
      if (currentVote != null) payload['previous_vote_id'] = currentVote;
      await tx.delete('jobs', where: 'id=?', whereArgs: [jobId]);
      await tx.insert('jobs', {
        ...row,
        'id': payload['id'],
        'payload': jsonEncode(payload),
        'state': 'pending',
        'error': null,
      });
    });
  }

  Future<void> retry(String account) => db.update(
    'jobs',
    {'state': 'pending', 'error': null},
    where: "account=? AND state='blocked' AND kind='sighting'",
    whereArgs: [account],
  );
}
