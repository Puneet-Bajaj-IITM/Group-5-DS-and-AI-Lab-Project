import 'dart:convert';
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:material_ui/material_ui.dart';
import 'package:provider/provider.dart';
import 'package:serpentguard/inference/decision.dart';
import 'package:serpentguard/inference/image_ops.dart';
import 'package:serpentguard/inference/pipeline.dart';
import 'package:serpentguard/models/verdict.dart';
import 'package:serpentguard/screens/result_screen.dart';
import 'package:serpentguard/services/classifier.dart';
import 'package:serpentguard/services/store.dart';
import 'package:serpentguard/state/app_state.dart';
import 'package:serpentguard/widgets/common.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'helpers.dart';

/// A 1 x 1 PNG, so the result screen draws its photo and boxes.
const _pixelPng = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==';

/// Several snakes on the photo screen, and no species named while unsure.
void main() {
  final catalog = loadCatalog();
  final rules = loadRules();
  final rule = DecisionRule(catalog: catalog, rules: rules);
  final krait = byName(catalog, 'Bungarus caeruleus'), rat = byName(catalog, 'Ptyas mucosa');
  final cobra = byName(catalog, 'Naja naja');
  const mainBox = PixelBox(10, 10, 60, 60), otherBox = PixelBox(70, 20, 95, 70);

  final dangerous = ModelAnswer(
    label: dangerousLabel,
    reasons: const ['who_head'],
    message: 'dangerous: WHO Category 1',
    top3: [ScoredClass(krait.id, 0.8), ScoredClass(cobra.id, 0.1), ScoredClass(rat.id, 0.05)],
  );
  final notListed = ModelAnswer(label: notListedLabel, reasons: const [], message: 'not listed', top3: [ScoredClass(rat.id, 0.9)]);
  final unknown = ModelAnswer(
    label: unknownLabel,
    reasons: const ['unclear_between_species'],
    message: "we don't know: unclear between species",
    top3: [ScoredClass(rat.id, 0.3), ScoredClass(krait.id, 0.2)],
  );

  Identification photo(ModelAnswer main, {List<ModelAnswer> others = const []}) => rule.fromAnswer(main,
      details: PipelineResult(
        width: 100,
        height: 80,
        exposureFixed: false,
        gate: const GateResult(snake: true, boxConf: 0.9, maskArea: 0.2, box: mainBox),
        answer: main,
        others: [for (final o in others) OtherSnake(otherBox, answer: o)],
      ));

  late String photoPath;
  setUpAll(() {
    final dir = Directory.systemTemp.createTempSync('multi_snake');
    photoPath = '${dir.path}/photo.png';
    File(photoPath).writeAsBytesSync(base64Decode(_pixelPng));
  });

  Future<void> pumpResult(WidgetTester tester, Identification id) async {
    SharedPreferences.setMockInitialValues({});
    // Tall enough for the whole result list to be built.
    tester.view.physicalSize = const Size(800, 4000);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.reset);
    final state = AppState(catalog: catalog, classifier: DemoClassifier(catalog, rules), store: await Store.open());
    await tester.pumpWidget(ChangeNotifierProvider.value(
      value: state,
      child: MaterialApp(home: ResultScreen(identification: id, photoPath: photoPath)),
    ));
    await tester.pumpAndSettle();
  }

  List<String?> boxLabels(WidgetTester tester) =>
      tester.widget<SnakeBoxesOverlay>(find.byType(SnakeBoxesOverlay)).boxes.map((b) => b.label).toList();

  testWidgets('two snakes: both boxed and listed, the card shows the most severe verdict', (tester) async {
    await pumpResult(tester, photo(notListed, others: [dangerous]));
    expect(find.text('2 snakes in this photo'), findsOneWidget);
    expect(find.text('${rat.displayName} · 90%'), findsOneWidget);
    expect(find.text('${krait.displayName} · 80%'), findsOneWidget);
    expect(boxLabels(tester), ['${rat.displayName} · 90%', '${krait.displayName} · 80%']);
    final banner = tester.widget<VerdictBanner>(find.byType(VerdictBanner));
    expect(banner.verdict, Verdict.dangerous);
    expect(find.text('Someone was bitten: get help'), findsOneWidget);
  });

  testWidgets('one snake: no snake count', (tester) async {
    await pumpResult(tester, photo(dangerous));
    expect(find.textContaining('snakes in this photo'), findsNothing);
    expect(boxLabels(tester), ['${krait.displayName} · 80%']);
  });

  testWidgets('could be dangerous: "Snake · not sure", no species and no percentages', (tester) async {
    await pumpResult(tester, photo(unknown));
    expect(boxLabels(tester), ['Snake · not sure']);
    expect(find.textContaining('%'), findsNothing);
    expect(find.text(rat.displayName), findsNothing);
    expect(find.text(krait.displayName), findsNothing);
    expect(find.text('Other possibilities'), findsNothing);
  });

  testWidgets('an unsure other snake is not named either', (tester) async {
    await pumpResult(tester, photo(dangerous, others: [unknown]));
    expect(boxLabels(tester), ['${krait.displayName} · 80%', 'Snake · not sure']);
    expect(find.text('Snake · not sure'), findsOneWidget);
  });

  testWidgets('the top species first, the other two under a collapsed "Other possibilities"', (tester) async {
    await pumpResult(tester, photo(dangerous));
    expect(find.text(krait.displayName), findsOneWidget);
    expect(find.text('80%'), findsOneWidget);
    expect(find.text(cobra.displayName), findsNothing);
    await tester.tap(find.text('Other possibilities'));
    await tester.pumpAndSettle();
    expect(find.text(cobra.displayName), findsOneWidget);
    expect(find.text('10%'), findsOneWidget);
  });
}
