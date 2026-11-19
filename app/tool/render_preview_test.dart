// Run explicitly: flutter test tool/render_preview_test.dart
// Renders the actual Flutter widgets; it is not a device or camera test.
// ignore_for_file: invalid_use_of_visible_for_testing_member
import 'dart:io';
import 'dart:ui' as ui;

import 'package:flutter/rendering.dart';
import 'package:flutter/services.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:material_ui/material_ui.dart';
import 'package:provider/provider.dart';
import 'package:serpentguard/app.dart';
import 'package:serpentguard/screens/community_screen.dart';
import 'package:serpentguard/screens/result_screen.dart';
import 'package:serpentguard/services/classifier.dart';
import 'package:serpentguard/services/store.dart';
import 'package:serpentguard/services/voting_queue.dart';
import 'package:serpentguard/state/app_state.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:sqflite_common_ffi/sqflite_ffi.dart';

import '../test/helpers.dart';

void main() {
  testWidgets('render current app screens', (tester) async {
    tester.view.physicalSize = const Size(430, 932);
    tester.view.devicePixelRatio = 1;
    SharedPreferences.setMockInitialValues({});
    FlutterSecureStorage.setMockInitialValues({});
    sqfliteFfiInit();
    databaseFactory = databaseFactoryFfi;
    final root = Platform.environment['SERPENTGUARD_FLUTTER_ROOT']!;
    final fonts = Directory('$root/bin/cache/artifacts/material_fonts');
    final roboto = FontLoader('Roboto');
    for (final file in fonts.listSync().whereType<File>().where(
      (f) => f.path.endsWith('.ttf') && f.path.contains('Roboto-'),
    )) {
      roboto.addFont(
        Future.value(ByteData.sublistView(file.readAsBytesSync())),
      );
    }
    await roboto.load();
    // Unspecified platform font styles default to Ahem in flutter_test.
    // Use Android's Roboto for these preview renders as well.
    final platformFont = FontLoader('Ahem');
    for (final name in [
      'Roboto-Regular.ttf',
      'Roboto-Bold.ttf',
      'Roboto-Black.ttf',
    ]) {
      platformFont.addFont(
        Future.value(
          ByteData.sublistView(File('${fonts.path}/$name').readAsBytesSync()),
        ),
      );
    }
    await platformFont.load();
    final icon = FontLoader('MaterialIcons')
      ..addFont(
        Future.value(
          ByteData.sublistView(
            File(
              '$root/bin/cache/artifacts/material_fonts/MaterialIcons-Regular.otf',
            ).readAsBytesSync(),
          ),
        ),
      );
    await icon.load();
    final catalog = loadCatalog();
    final demo = DemoClassifier(catalog, loadRules());
    final state = AppState(
      catalog: catalog,
      classifier: demo,
      store: await Store.open(),
    );
    final out = Directory('build/demo-preview')..createSync(recursive: true);
    final queue = (await tester.runAsync(
      () => VotingQueue.open(path: '${out.absolute.path}/preview.sqlite3'),
    ))!;
    try {
      Future<void> capture(String name, Widget child) async {
        final key = GlobalKey();
        await tester.pumpWidget(
          RepaintBoundary(
            key: key,
            child: ChangeNotifierProvider.value(value: state, child: child),
          ),
        );
        await tester.pumpAndSettle();
        final boundary =
            key.currentContext!.findRenderObject() as RenderRepaintBoundary;
        await tester.runAsync(() async {
          final image = await boundary.toImage(pixelRatio: 2);
          final bytes = await image.toByteData(format: ui.ImageByteFormat.png);
          File('${out.path}/$name.png')
              .writeAsBytesSync(bytes!.buffer.asUint8List());
          image.dispose();
        });
      }

      await capture('identify', const SerpentGuardApp(showSplash: false));
      await capture(
        'community',
        MaterialApp(
          debugShowCheckedModeBanner: false,
          theme: SerpentGuardApp.theme(Brightness.light),
          home: CommunityScreen(openQueue: () async => queue),
        ),
      );
      await capture(
        'result-demo',
        MaterialApp(
          debugShowCheckedModeBanner: false,
          theme: SerpentGuardApp.theme(Brightness.light),
          home: ResultScreen(
            identification: demo.forSpecies(
              byName(catalog, 'Bungarus caeruleus').id,
            ),
          ),
        ),
      );
      await tester.pumpWidget(const SizedBox.shrink());
    } finally {
      await tester.runAsync(queue.db.close);
      tester.view.resetPhysicalSize();
      tester.view.resetDevicePixelRatio();
    }
  });
}
