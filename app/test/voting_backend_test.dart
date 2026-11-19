import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;
import 'package:serpentguard/services/voting_api.dart';
import 'package:serpentguard/services/voting_queue.dart';
import 'package:sqflite_common_ffi/sqflite_ffi.dart';

/// Uses the real SQLAlchemy WSGI API and a disposable database. Enable with
/// SERPENTGUARD_PYTHON=/absolute/path/to/.venv/bin/python flutter test ...
void main() {
  final python = Platform.environment['SERPENTGUARD_PYTHON'];
  test(
    'real backend accepts offline evidence, blocks self-votes and resolves expert genus vote',
    () async {
      sqfliteFfiInit();
      databaseFactory = databaseFactoryFfi;
      final dir = await Directory.systemTemp.createTemp(
        'sg-backend-integration-',
      );
      Process? server;
      VotingQueue? queue;
      VotingApi? reporter;
      VotingApi? reviewer;
      try {
        server = await Process.start(
          python!,
          [
            '-u',
            '-c',
            '''
import json,sys
from pathlib import Path
from wsgiref.simple_server import make_server, WSGIRequestHandler
from serpentguard.voting_api import make_app,provision_user
class Quiet(WSGIRequestHandler):
    def log_message(self,*args): pass
db=Path(sys.argv[1])
reporter=provision_user(db,'citizen')
reviewer=provision_user(db,'expert')
server=make_server('127.0.0.1',0,make_app(db),handler_class=Quiet)
print(json.dumps({'port':server.server_port,'reporter':reporter['token'],'reviewer':reviewer['token']}),flush=True)
server.serve_forever()
''',
            '${dir.path}/server.sqlite3',
          ],
          environment: {'PYTHONPATH': '${Directory.current.parent.path}/src'},
        );
        final errors = server.stderr.drain<void>();
        final ready = jsonDecode(
          await server.stdout
              .transform(utf8.decoder)
              .transform(const LineSplitter())
              .first
              .timeout(const Duration(seconds: 20)),
        ) as Map<String, dynamic>;
        final origin = 'http://127.0.0.1:${ready['port']}';
        reporter = VotingApi(origin, ready['reporter'] as String);
        reviewer = VotingApi(origin, ready['reviewer'] as String);
        queue = await VotingQueue.open(path: '${dir.path}/phone.sqlite3');
        final frames = [
          for (var i = 0; i < 3; i++)
            votingJpeg(
              Uint8List.fromList(
                img.encodePng(img.Image(width: 40 + i, height: 30)),
              ),
            ),
        ];
        final id = await queue.sighting(
          reporter.account,
          frames,
          trainingConsent: false,
          capturedAt: DateTime.now(),
          offsets: [0, 500, 1000],
        );
        await queue.sync(reporter);
        expect((await queue.jobs(reporter.account)).single['state'], 'done');
        await queue.refresh(reviewer);
        final feed = await queue.feed(reviewer.account);
        expect(feed.single['id'], id);
        expect(feed.single['source'], 'live_video');
        expect(feed.single.containsKey('top_predictions'), isFalse);
        expect((await queue.images(reviewer.account, id)).length, 3);

        await queue.vote(reporter.account, id, 'genus', 'Naja');
        await queue.sync(reporter);
        expect(
          (await queue.jobs(reporter.account))
              .where((j) => j['kind'] == 'vote')
              .single['state'],
          'blocked',
        );
        await queue.vote(reviewer.account, id, 'genus', 'Naja');
        await queue.sync(reviewer);
        expect((await queue.jobs(reviewer.account)).single['state'], 'done');
        final consensus = await reviewer.json(
          'GET',
          '/sightings/$id/consensus',
        );
        expect(consensus['rank'], 'genus');
        expect(consensus['label'], 'Naja');
        expect(consensus['posterior_confidence'], greaterThanOrEqualTo(0.85));
        server.kill();
        await server.exitCode;
        server = null;
        await errors;
        // Evidence remains usable after the server stops (airplane-mode path).
        expect((await queue.images(reviewer.account, id)).length, 3);
        final saved = await queue.consensus(reviewer, id);
        expect(saved.cached, isTrue);
        expect(saved.value?['label'], 'Naja');
      } finally {
        reporter?.close();
        reviewer?.close();
        if (server != null) {
          server.kill();
          await server.exitCode;
        }
        await queue?.db.close();
        await dir.delete(recursive: true);
      }
    },
    skip: python == null
        ? 'Set SERPENTGUARD_PYTHON to enable the real API integration test.'
        : false,
  );
}
