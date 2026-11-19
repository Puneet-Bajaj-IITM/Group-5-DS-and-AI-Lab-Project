import 'dart:convert';
import 'dart:async';
import 'dart:io';
import 'dart:typed_data';

import 'package:crypto/crypto.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:image/image.dart' as img;
import 'package:serpentguard/services/voting_api.dart';
import 'package:serpentguard/services/voting_queue.dart';
import 'package:sqflite_common_ffi/sqflite_ffi.dart';

void main() {
  late Directory folder;
  late VotingQueue queue;
  late Uint8List jpeg;

  setUpAll(() {
    sqfliteFfiInit();
    databaseFactory = databaseFactoryFfi;
  });
  setUp(() async {
    folder = await Directory.systemTemp.createTemp('sg-voting-test-');
    queue = await VotingQueue.open(path: '${folder.path}/queue.sqlite3');
    jpeg = votingJpeg(
      Uint8List.fromList(img.encodePng(img.Image(width: 32, height: 24))),
    );
  });
  tearDown(() async {
    await queue.db.close();
    await folder.delete(recursive: true);
  });

  test('rejects unsafe origins and never follows bearer redirects', () async {
    for (final bad in [
      'http://example.org',
      'https://user:secret@example.org',
      'https://example.org/path',
      'https://example.org?token=secret',
      'file:///tmp',
    ]) {
      expect(() => VotingApi(bad, 'token'), throwsFormatException);
    }
    final api = VotingApi(
      'https://review.example.org',
      'token',
      client: MockClient((req) async {
        expect(req.followRedirects, isFalse);
        return http.Response(
          '',
          302,
          headers: {'location': 'https://other.example.org'},
        );
      }),
    );
    await expectLater(
      api.json('GET', '/sightings/review-feed'),
      throwsA(isA<VotingApiError>()),
    );
    api.close();
  });

  test('evidence survives restart; interrupted upload retries exact bytes and UUID', () async {
    var postCount = 0;
    var uploadCount = 0;
    String? savedPost;
    final api = VotingApi(
      'https://review.example.org',
      'token',
      client: MockClient((req) async {
        if (req.method == 'POST') {
          postCount++;
          savedPost ??= req.body;
          expect(req.body, savedPost);
          return http.Response('{}', postCount == 1 ? 201 : 200);
        }
        uploadCount++;
        expect(req.bodyBytes, jpeg);
        if (uploadCount == 1) throw const SocketException('offline');
        return http.Response('{"ready":true}', 200);
      }),
    );
    final id = await queue.sighting(
      api.account,
      [jpeg],
      trainingConsent: false,
      capturedAt: DateTime.utc(2026),
    );
    await queue.db.close();
    queue = await VotingQueue.open(path: '${folder.path}/queue.sqlite3');
    await queue.sync(api);
    expect((await queue.jobs(api.account)).single['state'], 'pending');
    await queue.sync(api);
    expect((await queue.jobs(api.account)).single['state'], 'done');
    expect(postCount, 2);
    expect(jsonDecode(savedPost!)['id'], id);
    expect(
      jsonDecode(savedPost!)['evidence'][0]['sha256'],
      sha256.convert(jpeg).toString(),
    );
    expect(await queue.db.query('media'), isEmpty);
    api.close();
  });

  test('account switch cannot send another account queue', () async {
    final api = VotingApi(
      'https://review.example.org',
      'other-token',
      client: MockClient((_) async {
        fail('No request should be sent for a different account');
      }),
    );
    await queue.vote('different-account', votingId(), 'genus', 'Naja');
    await queue.sync(api);
    expect((await queue.jobs('different-account')).single['state'], 'pending');
    api.close();
  });

  test(
    'vote conflicts block retries until explicit reconciliation with new ID',
    () async {
      var calls = 0;
      Map<String, dynamic>? first;
      Map<String, dynamic>? second;
      final api = VotingApi(
        'https://review.example.org',
        'token',
        client: MockClient((req) async {
          calls++;
          if (calls == 1) {
            first = jsonDecode(req.body);
            return http.Response('{}', 409);
          }
          second = jsonDecode(req.body);
          return http.Response('{}', 201);
        }),
      );
      final sighting = votingId();
      await queue.vote(api.account, sighting, 'genus', 'Naja');
      await queue.sync(api);
      await queue.sync(api);
      expect(calls, 1);
      final job = (await queue.jobs(api.account)).single;
      expect(job['state'], 'blocked');
      final previous = votingId();
      await queue.reconcile(job['id'] as String, api.account, previous);
      await queue.sync(api);
      expect(second!['id'], isNot(first!['id']));
      expect(second!['previous_vote_id'], previous);
      expect((await queue.jobs(api.account)).single['state'], 'done');
      api.close();
    },
  );

  test(
    'one unresolved ballot per encounter; live views are bounded and distinct',
    () async {
      final sid = votingId();
      await queue.vote('account', sid, 'genus', 'Naja');
      await expectLater(
        queue.vote('account', sid, 'species', 'Naja naja'),
        throwsStateError,
      );
      await expectLater(
        queue.sighting(
          'account',
          [jpeg, jpeg],
          trainingConsent: false,
          capturedAt: DateTime.now(),
          offsets: [0, 100],
        ),
        throwsFormatException,
      );
      await expectLater(
        queue.sighting(
          'account',
          [jpeg],
          trainingConsent: false,
          capturedAt: DateTime.now(),
          offsets: [-1],
        ),
        throwsFormatException,
      );
      final large = votingJpeg(
        Uint8List.fromList(img.encodePng(img.Image(width: 2400, height: 100))),
      );
      expect(img.decodeJpg(large)!.width, 1600);
      expect(large.length, lessThanOrEqualTo(1024 * 1024));
    },
  );

  test('foreground and background connections cannot drain the same job concurrently', () async {
    final other = await VotingQueue.open(path: '${folder.path}/queue.sqlite3');
    final entered = Completer<void>();
    final release = Completer<void>();
    var calls = 0;
    final api = VotingApi(
      'https://review.example.org',
      'token',
      client: MockClient((_) async {
        calls++;
        entered.complete();
        await release.future;
        return http.Response('{}', 201);
      }),
    );
    try {
      await queue.vote(api.account, votingId(), 'genus', 'Naja');
      final first = queue.sync(api);
      await entered.future;
      expect(await other.sync(api), isFalse);
      expect(calls, 1);
      release.complete();
      expect(await first, isTrue);
      expect((await other.jobs(api.account)).single['state'], 'done');
    } finally {
      if (!release.isCompleted) release.complete();
      api.close();
      await other.db.close();
    }
  });

  test('unbound drafts persist and never upload before explicit account attachment', () async {
    var calls = 0;
    final api = VotingApi('https://review.example.org', 'token', client: MockClient((req) async {
      calls++;
      return http.Response(req.method == 'PUT' ? '{"ready":true}' : '{}', 201);
    }));
    final id = await queue.sighting(VotingQueue.draftAccount, [jpeg], trainingConsent: false,
        capturedAt: DateTime.utc(2026, 11, 19));
    final original = (await queue.jobs(VotingQueue.draftAccount)).single['payload'];
    await queue.db.close();
    queue = await VotingQueue.open(path: '${folder.path}/queue.sqlite3');
    await queue.sync(api);
    expect(calls, 0);
    await queue.claimDrafts(api.account);
    final claimed = (await queue.jobs(api.account)).single;
    expect(claimed['id'], id);
    expect(claimed['payload'], original);
    await queue.sync(api);
    expect(calls, 2);
    expect((await queue.jobs(api.account)).single['state'], 'done');
    api.close();
  });

  test(
    'feed pagination appends offline data and a failed refresh preserves it',
    () async {
      final ids = [votingId(), votingId()];
      var offline = false;
      final api = VotingApi(
        'https://review.example.org',
        'token',
        client: MockClient((req) async {
          if (offline) throw const SocketException('offline');
          final cursor = req.url.queryParameters['cursor'];
          final slot = cursor == '0' ? 0 : 1;
          return http.Response(
            jsonEncode({
              'items': [
                {'id': ids[slot], 'evidence': []},
              ],
              'next_cursor': slot == 0 ? 1 : null,
            }),
            200,
          );
        }),
      );
      await queue.refresh(api);
      expect(await queue.nextCursor(api.account), 1);
      await queue.refresh(api, more: true);
      expect(
        (await queue.feed(api.account)).map((i) => i['id']).toSet(),
        ids.toSet(),
      );
      expect(await queue.nextCursor(api.account), isNull);
      offline = true;
      await expectLater(queue.refresh(api), throwsA(isA<SocketException>()));
      expect((await queue.feed(api.account)).length, 2);
      api.close();
    },
  );
}
