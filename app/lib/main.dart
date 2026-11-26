import 'package:material_ui/material_ui.dart';
import 'package:provider/provider.dart';

import 'app.dart';
import 'inference/bundle.dart';
import 'inference/decision.dart';
import 'inference/pipeline.dart';
import 'models/species.dart';
import 'services/classifier.dart';
import 'services/store.dart';
import 'services/voting_background.dart';
import 'services/voting_session.dart';
import 'state/app_state.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final (catalog, store, config, bundled) = await (
    SpeciesCatalog.load(),
    Store.open(),
    SnakePipeline.loadConfig(),
    SnakePipeline.modelsBundled(),
  ).wait;
  final (settings, tables) = config;
  final rules = DecisionRules(settings.thresholds, tables);
  final state = AppState(
    catalog: catalog,
    // The real model when its files are in the build (assets/models/, see
    // README); otherwise a clearly marked demo.
    classifier: bundled
        ? OnnxSnakeClassifier(catalog, rules, _openModels(settings, tables))
        : DemoClassifier(catalog, rules),
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

/// Opens the models in the background. A failure is kept for the first
/// identification to report, instead of surfacing as an uncaught error.
Future<SnakePipeline> _openModels(PipelineSettings settings, ModelTables tables) {
  final f = SnakePipeline.load(settings, tables);
  f.ignore();
  return f;
}
