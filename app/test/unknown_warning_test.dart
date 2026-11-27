import 'package:material_ui/material_ui.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:provider/provider.dart';
import 'package:serpentguard/inference/decision.dart';
import 'package:serpentguard/models/verdict.dart';
import 'package:serpentguard/screens/result_screen.dart';
import 'package:serpentguard/services/classifier.dart';
import 'package:serpentguard/services/store.dart';
import 'package:serpentguard/state/app_state.dart';
import 'package:serpentguard/widgets/common.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'helpers.dart';

/// Post-M5-test change: "unknown" with a snake found is an urgent warning
/// ("Could be dangerous"), styled and escalated like "Dangerous"; a photo with
/// no snake stays a neutral "No snake found", never an all-clear.
void main() {
  final catalog = loadCatalog();
  final rules = loadRules();
  final rule = DecisionRule(catalog: catalog, rules: rules);
  final rat = byName(catalog, 'Ptyas mucosa');

  Identification unknownWithSnake() => rule.fromAnswer(ModelAnswer(
        label: unknownLabel,
        reasons: const ['unclear_between_species'],
        message: "we don't know: unclear between species",
        top3: [ScoredClass(rat.id, 0.3)],
      ));

  Future<void> pumpResult(WidgetTester tester, Identification id) async {
    SharedPreferences.setMockInitialValues({});
    final state = AppState(catalog: catalog, classifier: DemoClassifier(catalog, rules), store: await Store.open());
    await tester.pumpWidget(ChangeNotifierProvider.value(
      value: state,
      child: MaterialApp(home: ResultScreen(identification: id)),
    ));
    await tester.pumpAndSettle();
  }

  Color bannerColour(WidgetTester tester) {
    final box = tester.widget<Container>(
        find.descendant(of: find.byType(VerdictBanner), matching: find.byType(Container)).first);
    return (box.decoration as BoxDecoration).color!;
  }

  test('unknown warns and escalates like dangerous', () {
    expect(Verdict.unknown.title, 'Could be dangerous');
    expect(Verdict.unknown.advice, contains('treat it as venomous'));
    expect(Verdict.unknown.escalate, isTrue);
    expect(Verdict.unknown.warns, isTrue);
    expect(Verdict.notListed.warns, isFalse);
    expect(VerdictStyle.of(Verdict.unknown).background, VerdictStyle.of(Verdict.dangerous).background);
  });

  testWidgets('an unknown result shows the keep-away warning with first aid and SOS', (tester) async {
    await pumpResult(tester, unknownWithSnake());
    expect(find.text('Could be dangerous'), findsOneWidget);
    expect(find.text('Unclear between species'), findsOneWidget);
    expect(find.textContaining('treat it as venomous'), findsOneWidget);
    expect(find.text('Someone was bitten: get help'), findsOneWidget);
    expect(find.text("We don't know"), findsNothing);
    expect(bannerColour(tester), VerdictStyle.of(Verdict.dangerous).background);
  });

  testWidgets('no snake found is neutral, not a warning and not an all-clear', (tester) async {
    await pumpResult(tester, rule.fromAnswer(ModelAnswer.noSnake));
    expect(find.text(noSnakeTitle), findsOneWidget);
    expect(find.textContaining('does not mean the area is safe'), findsOneWidget);
    expect(find.text('Could be dangerous'), findsNothing);
    expect(bannerColour(tester), VerdictStyle.noSnake.background);
  });
}
