import 'dart:convert';

import 'package:crypto/crypto.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:http/http.dart' as http;
import 'package:serpentguard/inference/pipeline.dart';

/// The whole path on a phone or emulator, against the model bundle's
/// `golden_photos.json`: same gate, top species and verdict, scores within
/// the file's tolerance (0.01).
///
/// The photos are credited CC images and stay out of git: the test downloads
/// them from the URLs in the JSON and checks each SHA-1. Run it with
///   python3 tool/golden_defines.py > build/golden_defines.json
///   flutter test integration_test/golden_photos_test.dart \
///     --dart-define-from-file=build/golden_defines.json
void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('golden photos give the reference answers on the device', (tester) async {
    const raw = String.fromEnvironment('GOLDEN_PHOTOS');
    expect(raw, isNotEmpty, reason: 'pass --dart-define-from-file=build/golden_defines.json');
    final spec = jsonDecode(raw) as Map<String, dynamic>;
    final tolerance = (spec['tolerance'] as num).toDouble();
    expect(await SnakePipeline.modelsBundled(), isTrue, reason: 'copy the models into assets/models/');
    final (settings, tables) = await SnakePipeline.loadConfig();
    final pipeline = await SnakePipeline.load(settings, tables);

    final report = <String>[];
    var failures = 0;
    for (final photo in (spec['photos'] as List).cast<Map<String, dynamic>>()) {
      final id = photo['photo_id'];
      final bytes = (await http.get(Uri.parse(photo['url'] as String),
              headers: {'User-Agent': 'SerpentGuard-test/0.1 (student research app)'}))
          .bodyBytes;
      expect(sha1.convert(bytes).toString(), photo['sha1'], reason: '$id.jpg is not the golden photo');
      final r = await pipeline.run(await SnakePipeline.decode(bytes));
      final want = photo['answer'] as Map<String, dynamic>;
      final steps = photo['steps'] as Map<String, dynamic>;
      final wantTop = (want['top3'] as List).cast<Map<String, dynamic>>();
      final problems = <String>[];
      final gate = want['gate'] as Map<String, dynamic>;
      if (r.gate.snake != gate['snake']) problems.add('gate ${r.gate.snake} != ${gate['snake']}');
      if (r.answer.label != want['verdict']) problems.add('verdict ${r.answer.label} != ${want['verdict']}');
      for (var i = 0; i < wantTop.length; i++) {
        final got = r.answer.top3[i];
        if (tables.species[got.classId] != wantTop[i]['species']) {
          problems.add('top3[$i] ${tables.species[got.classId]} != ${wantTop[i]['species']}');
        }
        if ((got.probability - (wantTop[i]['probability'] as num)).abs() > tolerance) {
          problems.add('top3[$i] p ${got.probability} != ${wantTop[i]['probability']}');
        }
      }
      if (gate['box_conf'] != null && ((gate['box_conf'] as num) - (r.gate.boxConf ?? 0)).abs() > tolerance) {
        problems.add('box_conf ${r.gate.boxConf} != ${gate['box_conf']}');
      }
      if (((gate['mask_area'] as num) - r.gate.maskArea).abs() > tolerance) {
        problems.add('mask_area ${r.gate.maskArea} != ${gate['mask_area']}');
      }
      final win = steps['crop_window_top_bottom_left_right'] as List?;
      final gotWin = r.cropWindow == null ? null : [r.cropWindow!.$1, r.cropWindow!.$2, r.cropWindow!.$3, r.cropWindow!.$4];
      if (r.gate.snake && jsonEncode(gotWin) != jsonEncode(win)) problems.add('crop $gotWin != $win');
      final line = '$id: ${r.answer.label} | ${r.answer.message} | top ${r.answer.top3.isEmpty ? '-' : '${tables.species[r.answer.top3.first.classId]} ${r.answer.top3.first.probability.toStringAsFixed(4)}'}'
          ' | box ${r.gate.boxConf?.toStringAsFixed(4)} (golden ${(gate['box_conf'] as num?)?.toStringAsFixed(4)})'
          ' | mask ${r.gate.maskArea.toStringAsFixed(4)} | exposure fixed ${r.exposureFixed} | ${r.totalMs} ms ${r.timingsMs}'
          '${problems.isEmpty ? ' | OK' : ' | MISMATCH ${problems.join('; ')}'}';
      report.add(line);
      if (problems.isNotEmpty) failures++;
    }
    // ignore: avoid_print
    print('GOLDEN PHOTOS\n${report.join('\n')}');
    await pipeline.close();
    expect(failures, 0, reason: report.join('\n'));
  });
}
