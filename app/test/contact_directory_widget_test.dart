import 'package:material_ui/material_ui.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:provider/provider.dart';
import 'package:serpentguard/models/records.dart';
import 'package:serpentguard/screens/contacts_screen.dart';
import 'package:serpentguard/screens/sos_screen.dart';
import 'package:serpentguard/services/classifier.dart';
import 'package:serpentguard/services/store.dart';
import 'package:serpentguard/state/app_state.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'helpers.dart';

void main() {
  testWidgets('animal hospitals are excluded from human snakebite help', (tester) async {
    SharedPreferences.setMockInitialValues({});
    final catalog = loadCatalog();
    final state = AppState(catalog: catalog, classifier: DemoClassifier(catalog, loadRules()),
        store: await Store.open());
    addTearDown(state.dispose);
    await state.addContact(const Contact(name: 'Animal-only clinic', phone: '+914425381366',
        kind: ContactKind.veterinarian));
    await state.addContact(const Contact(name: 'Human medical contact', phone: '108',
        kind: ContactKind.doctor));
    await tester.pumpWidget(ChangeNotifierProvider.value(value: state,
        child: const MaterialApp(home: SosScreen(bitten: true))));
    await tester.pumpAndSettle();
    await tester.scrollUntilVisible(find.text('Human medical contact'), 300);
    await tester.pumpAndSettle();
    expect(find.text('Animal-only clinic'), findsNothing);
    expect(find.text('Human medical contact'), findsOneWidget);
  });
  testWidgets('offline directory searches and saves without adding SOS recipients', (tester) async {
    SharedPreferences.setMockInitialValues({});
    tester.view.physicalSize = const Size(360, 800);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);
    final catalog = loadCatalog();
    final state = AppState(catalog: catalog, classifier: DemoClassifier(catalog, loadRules()),
        store: await Store.open());
    addTearDown(state.dispose);
    await tester.pumpWidget(ChangeNotifierProvider.value(value: state,
        child: const MaterialApp(home: ContactsScreen())));
    await tester.pumpAndSettle();
    await tester.enterText(find.byType(TextField), 'Delhi');
    await tester.pumpAndSettle();
    expect(find.text('Wildlife SOS Delhi helpline'), findsOneWidget);
    expect(find.text('Madras Veterinary College Teaching Hospital'), findsNothing);
    FocusManager.instance.primaryFocus?.unfocus();
    tester.testTextInput.hide();
    await tester.pumpAndSettle();
    await tester.ensureVisible(find.text('Save contact'));
    await tester.pumpAndSettle();
    await tester.tap(find.text('Save contact').hitTestable());
    await tester.pumpAndSettle();
    expect(state.contacts.single.phone, '+919871963535');
    expect(state.emergencyContacts, isEmpty);
    expect(find.text('Saved'), findsOneWidget);
    expect(tester.takeException(), isNull);
    expect(state.contacts.single.kind.name, 'rescuer');
  });
}
