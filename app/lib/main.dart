import 'package:material_ui/material_ui.dart';
import 'package:provider/provider.dart';

import 'app.dart';
import 'models/species.dart';
import 'services/classifier.dart';
import 'services/store.dart';
import 'services/voting_background.dart';
import 'services/voting_session.dart';
import 'state/app_state.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final (catalog, store) = await (SpeciesCatalog.load(), Store.open()).wait;
  final state = AppState(
    catalog: catalog,
    // Swap for the ONNX classifier once the trained model is bundled.
    classifier: DemoClassifier(catalog),
    store: store,
  );
  final initFuture = Future<void>.delayed(
    const Duration(milliseconds: 450),
    state.init,
  );
  runApp(
    ChangeNotifierProvider.value(
      value: state,
      child: SerpentGuardApp(initFuture: initFuture),
    ),
  );
  // A failed scheduler or locked credential store must not block emergency UI.
  try {
    await VotingBackground.initialize();
    final session = await VotingSessionStore().load();
    if (session != null) await VotingBackground.schedule(session);
  } catch (_) {
    // Foreground sync remains available in Community.
  }
  await initFuture;
}
