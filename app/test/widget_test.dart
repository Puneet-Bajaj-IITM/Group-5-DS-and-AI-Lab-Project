import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:provider/provider.dart';
import 'package:serpentguard/screens/result_screen.dart';
import 'package:serpentguard/services/classifier.dart';
import 'package:serpentguard/services/store.dart';
import 'package:serpentguard/state/app_state.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'helpers.dart';

void main() {
  testWidgets('result screen shows the verdict, the demo warning and the SOS path',
      (tester) async {
    SharedPreferences.setMockInitialValues({});
    final catalog = loadCatalog();
    final demo = DemoClassifier(catalog);
    final state = AppState(catalog: catalog, classifier: demo, store: await Store.open());
    final krait = byName(catalog, 'Bungarus caeruleus');

    await tester.pumpWidget(ChangeNotifierProvider.value(
      value: state,
      child: MaterialApp(home: ResultScreen(identification: demo.forSpecies(krait.id))),
    ));
    await tester.pumpAndSettle();

    expect(find.text('Dangerous'), findsOneWidget);
    expect(find.textContaining('Demo result'), findsOneWidget);
    expect(find.text('Someone was bitten: get help'), findsOneWidget);
    expect(find.text('Common krait'), findsOneWidget);
    expect(state.sightings.single.speciesId, krait.id);
  });
}
