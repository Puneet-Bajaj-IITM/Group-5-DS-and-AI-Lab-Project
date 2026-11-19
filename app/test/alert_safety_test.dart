import 'package:flutter_test/flutter_test.dart';
import 'package:serpentguard/services/classifier.dart';
import 'package:serpentguard/services/store.dart';
import 'package:serpentguard/state/app_state.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'helpers.dart';

void main() {
  test('demo classifications never enter the real broadcast queue', () async {
    SharedPreferences.setMockInitialValues({});
    final catalog = loadCatalog();
    final demo = DemoClassifier(catalog);
    final state = AppState(
      catalog: catalog,
      classifier: demo,
      store: await Store.open(),
    );
    expect(
      await state.sendAlert(demo.forSpecies(0)),
      contains('Demo results cannot'),
    );
    expect(state.alertBook.sent, isEmpty);
  });

  test(
    'simulated incoming alert exercises UI without becoming relayable',
    () async {
      SharedPreferences.setMockInitialValues({});
      final catalog = loadCatalog();
      final state = AppState(
        catalog: catalog,
        classifier: DemoClassifier(catalog),
        store: await Store.open(),
      );
      state.simulateIncomingAlert();
      expect(state.incoming, isNotNull);
      expect(state.alertBook.outgoing(DateTime.now(), relay: true), isEmpty);
    },
  );
}
