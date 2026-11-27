import 'dart:convert';

import 'package:crypto/crypto.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:http/http.dart' as http;
import 'package:serpentguard/inference/decision.dart';
import 'package:serpentguard/inference/pipeline.dart';

/// The whole M5 path on a phone or emulator, against `golden_photos.json`
/// (tool/m5/make_golden_m5.py): the same snake-check call, verdict (with the
/// app's and the frozen thresholds), message and top 3 as the Python
/// reference; P(snake) and the top-3 probabilities within the file's
/// tolerance (1e-3); the detector score and mask area within 0.01.
///
/// The photos are credited CC images and stay out of git: the test downloads
/// them from the URLs in the JSON and checks each SHA-1. Run it with
///   python3 tool/golden_defines.py > build/golden_defines.json
///   flutter test integration_test/golden_photos_test.dart \
///     --dart-define-from-file=build/golden_defines.json
const _frontEndTolerance = 0.01;

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('golden photos give the reference answers on the device', (tester) async {
    const raw = String.fromEnvironment('GOLDEN_PHOTOS');
    expect(raw, isNotEmpty, reason: 'pass --dart-define-from-file=build/golden_defines.json');
    final spec = jsonDecode(raw) as Map<String, dynamic>;
    final tolerance = (spec['tolerance'] as num).toDouble();
    expect(await SnakePipeline.modelsBundled(), isTrue, reason: 'run tool/m5/build_assets.py');
    final (settings, tables) = await SnakePipeline.loadConfig();
    final pipeline = await SnakePipeline.load(settings, tables);
    final frozen = DecisionRules(settings.frozenThresholds!, tables);

    final report = <String>[];
    var failures = 0;
    var worstSnakeP = 0.0, worstTop = 0.0;
    for (final photo in (spec['photos'] as List).cast<Map<String, dynamic>>()) {
      final id = photo['photo_id'];
      final bytes = (await http.get(Uri.parse(photo['url'] as String),
              headers: {'User-Agent': 'SerpentGuard-test/0.1 (student research app)'}))
          .bodyBytes;
      expect(sha1.convert(bytes).toString(), photo['sha1'], reason: '$id.jpg is not the golden photo');
      final r = await pipeline.run(await SnakePipeline.decode(bytes));
      final want = photo['answer'] as Map<String, dynamic>;
      final steps = photo['steps'] as Map<String, dynamic>;
      final problems = <String>[];

      final check = want['snake_check'] as Map<String, dynamic>;
      final snakeP = (check['snake_probability'] as num).toDouble();
      final gotP = r.gate.snakeProbability ?? double.nan;
      final dp = (gotP - snakeP).abs();
      worstSnakeP = dp > worstSnakeP ? dp : worstSnakeP;
      if (r.gate.snake != check['snake']) problems.add('snake check ${r.gate.snake} != ${check['snake']}');
      if (!(dp <= tolerance)) problems.add('P(snake) $gotP != $snakeP');
      if (r.answer.label != want['verdict']) problems.add('verdict ${r.answer.label} != ${want['verdict']}');
      if (r.answer.message != want['message']) problems.add('message "${r.answer.message}" != "${want['message']}"');

      final wantTop = (want['top3'] as List).cast<Map<String, dynamic>>();
      if (r.answer.top3.length != wantTop.length) problems.add('top3 length ${r.answer.top3.length} != ${wantTop.length}');
      for (var i = 0; i < wantTop.length && i < r.answer.top3.length; i++) {
        final got = r.answer.top3[i];
        if (tables.species[got.classId] != wantTop[i]['species']) {
          problems.add('top3[$i] ${tables.species[got.classId]} != ${wantTop[i]['species']}');
        }
        final d = (got.probability - (wantTop[i]['probability'] as num)).abs();
        worstTop = d > worstTop ? d : worstTop;
        if (d > tolerance) problems.add('top3[$i] p ${got.probability} != ${wantTop[i]['probability']}');
      }

      final front = want['front_end'] as Map<String, dynamic>?;
      if (front != null) {
        final box = front['box_conf'] as num?;
        if (box != null && (box - (r.gate.boxConf ?? 0)).abs() > _frontEndTolerance) {
          problems.add('box_conf ${r.gate.boxConf} != $box');
        }
        if (((front['mask_area'] as num) - r.gate.maskArea).abs() > _frontEndTolerance) {
          problems.add('mask_area ${r.gate.maskArea} != ${front['mask_area']}');
        }
        final win = steps['crop_window_top_bottom_left_right'] as List?;
        final gotWin = r.cropWindow == null ? null : [r.cropWindow!.$1, r.cropWindow!.$2, r.cropWindow!.$3, r.cropWindow!.$4];
        if (jsonEncode(gotWin) != jsonEncode(win)) problems.add('crop $gotWin != $win');
        final frozenLabel = frozen.verdictFromOutputs(r.classifierOutputs!).label;
        if (frozenLabel != want['frozen_verdict']) problems.add('frozen verdict $frozenLabel != ${want['frozen_verdict']}');
      }

      final top = r.answer.top3.isEmpty
          ? '-'
          : '${tables.species[r.answer.top3.first.classId]} ${r.answer.top3.first.probability.toStringAsFixed(4)}';
      report.add('$id (${photo['set']}): P(snake) ${gotP.toStringAsFixed(4)} (golden ${snakeP.toStringAsFixed(4)})'
          ' | ${r.answer.label} | top $top | box ${r.gate.boxConf?.toStringAsFixed(4)}'
          ' | mask ${r.gate.maskArea.toStringAsFixed(4)} | ${r.totalMs} ms ${r.timingsMs}'
          '${problems.isEmpty ? ' | OK' : ' | MISMATCH ${problems.join('; ')}'}');
      if (problems.isNotEmpty) failures++;
    }
    report.add('largest |P(snake) diff| $worstSnakeP, largest |top-3 p diff| $worstTop');
    // ignore: avoid_print
    print('GOLDEN PHOTOS\n${report.join('\n')}');
    await pipeline.close();
    expect(failures, 0, reason: report.join('\n'));
  });
}
