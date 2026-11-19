import 'dart:async';

import 'package:material_ui/material_ui.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:provider/provider.dart';
import 'package:serpentguard/screens/result_screen.dart';
import 'package:serpentguard/screens/splash_screen.dart';
import 'package:serpentguard/screens/sos_screen.dart';
import 'package:serpentguard/services/classifier.dart';
import 'package:serpentguard/services/store.dart';
import 'package:serpentguard/state/app_state.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'helpers.dart';

void main() {
  testWidgets('SOS omits simulated species claims', (tester) async {
    SharedPreferences.setMockInitialValues({});
    final catalog = loadCatalog();
    final demo = DemoClassifier(catalog);
    final state = AppState(
      catalog: catalog,
      classifier: demo,
      store: await Store.open(),
    );
    await tester.pumpWidget(
      ChangeNotifierProvider.value(
        value: state,
        child: MaterialApp(
          home: SosScreen(
            bitten: true,
            identification: demo.forSpecies(
              byName(catalog, 'Bungarus caeruleus').id,
            ),
          ),
        ),
      ),
    );
    await tester.pumpAndSettle();
    expect(find.text('Go to a hospital now'), findsOneWidget);
    expect(find.textContaining('Tell the doctor: possibly'), findsNothing);
  });
  testWidgets(
    'result screen shows the verdict, the demo warning and the SOS path',
    (tester) async {
      SharedPreferences.setMockInitialValues({});
      final catalog = loadCatalog();
      final demo = DemoClassifier(catalog);
      final state = AppState(
        catalog: catalog,
        classifier: demo,
        store: await Store.open(),
      );
      final krait = byName(catalog, 'Bungarus caeruleus');

      await tester.pumpWidget(
        ChangeNotifierProvider.value(
          value: state,
          child: MaterialApp(
            home: ResultScreen(identification: demo.forSpecies(krait.id)),
          ),
        ),
      );
      await tester.pumpAndSettle();

      expect(find.text('Dangerous'), findsOneWidget);
      expect(find.textContaining('Demo result'), findsOneWidget);
      expect(find.text('Someone was bitten: get help'), findsOneWidget);
      expect(find.text('Common krait'), findsOneWidget);
      expect(state.sightings.single.speciesId, krait.id);
    },
  );

  testWidgets('splash screen shows branding, skip button, and SOS bypass', (
    tester,
  ) async {
    SharedPreferences.setMockInitialValues({});
    final catalog = loadCatalog();
    final state = AppState(
      catalog: catalog,
      classifier: DemoClassifier(catalog),
      store: await Store.open(),
    );

    await tester.pumpWidget(
      ChangeNotifierProvider.value(
        value: state,
        child: const MaterialApp(
          home: SplashScreen(minDuration: Duration(seconds: 5)),
        ),
      ),
    );
    await tester.pump(const Duration(milliseconds: 50));

    expect(find.text('SERPENT GUARD', findRichText: true), findsOneWidget);
    expect(find.text('WATCH  ·  PROTECT  ·  DEFEND'), findsOneWidget);
    expect(find.text('Skip'), findsOneWidget);
    expect(find.text('Emergency / Bitten? Open SOS'), findsOneWidget);

    await tester.tap(find.text('Skip'));
    await tester.pumpAndSettle();

    expect(find.text('Seen a snake?'), findsOneWidget);
  });

  testWidgets(
    'splash screen auto-transitions after minDuration and initFuture',
    (tester) async {
      SharedPreferences.setMockInitialValues({});
      final catalog = loadCatalog();
      final state = AppState(
        catalog: catalog,
        classifier: DemoClassifier(catalog),
        store: await Store.open(),
      );
      final initCompleter = Completer<void>();

      await tester.pumpWidget(
        ChangeNotifierProvider.value(
          value: state,
          child: MaterialApp(
            home: SplashScreen(
              initFuture: initCompleter.future,
              minDuration: const Duration(milliseconds: 300),
            ),
          ),
        ),
      );

      await tester.pump(const Duration(milliseconds: 350));
      expect(find.text('Skip'), findsOneWidget);

      initCompleter.complete();
      await tester.pumpAndSettle();

      expect(find.text('Seen a snake?'), findsOneWidget);
    },
  );

  testWidgets(
    'splash screen SOS button jumps straight to snakebite emergency',
    (tester) async {
      SharedPreferences.setMockInitialValues({});
      final catalog = loadCatalog();
      final state = AppState(
        catalog: catalog,
        classifier: DemoClassifier(catalog),
        store: await Store.open(),
      );

      await tester.pumpWidget(
        ChangeNotifierProvider.value(
          value: state,
          child: const MaterialApp(
            home: SplashScreen(minDuration: Duration(seconds: 5)),
          ),
        ),
      );
      await tester.pump(const Duration(milliseconds: 50));

      await tester.tap(find.text('Emergency / Bitten? Open SOS'));
      await tester.pumpAndSettle();

      expect(find.text('Snakebite emergency'), findsOneWidget);
    },
  );
}
