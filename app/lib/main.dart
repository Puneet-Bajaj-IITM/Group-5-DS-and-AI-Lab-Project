import 'package:material_ui/material_ui.dart';
import 'package:provider/provider.dart';

import 'app.dart';
import 'models/species.dart';
import 'services/classifier.dart';
import 'services/store.dart';
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
  runApp(ChangeNotifierProvider.value(
    value: state,
    child: SerpentGuardApp(initFuture: initFuture),
  ));
  await initFuture;
}
