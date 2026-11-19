import 'dart:io';
import 'dart:ui';

import 'package:flutter/widgets.dart';
import 'package:workmanager/workmanager.dart';

import 'voting_queue.dart';
import 'voting_session.dart';

@pragma('vm:entry-point')
void votingDispatcher() {
  Workmanager().executeTask((task, input) async {
    WidgetsFlutterBinding.ensureInitialized();
    DartPluginRegistrant.ensureInitialized();
    VotingQueue? queue;
    try {
      final store = VotingSessionStore();
      final session = await store.load();
      if (session == null || session.account != input?['account']) return true;
      final api = session.createApi();
      try {
        queue = await VotingQueue.open();
        return await queue.sync(
          api,
          allowed: () async => (await store.load())?.account == session.account,
        );
      } finally {
        api.close();
      }
    } catch (_) {
      // No exception strings or request details are logged with credentials.
      return false;
    } finally {
      await queue?.db.close();
    }
  });
}

class VotingBackground {
  static const _tag = 'community-sync';
  static bool _initialized = false;

  static Future<void> initialize() async {
    if (!Platform.isAndroid || _initialized) return;
    await Workmanager().initialize(votingDispatcher);
    _initialized = true;
  }

  static Future<void> schedule(VotingSession session) async {
    if (!Platform.isAndroid) return;
    await initialize();
    final input = {'account': session.account};
    await Workmanager().registerOneOffTask(
      'community-now-${session.account}',
      _tag,
      tag: _tag,
      inputData: input,
      existingWorkPolicy: ExistingWorkPolicy.append,
      constraints: Constraints(networkType: NetworkType.connected),
      backoffPolicy: BackoffPolicy.exponential,
      backoffPolicyDelay: const Duration(seconds: 30),
    );
    await Workmanager().registerPeriodicTask(
      'community-periodic-${session.account}',
      _tag,
      tag: _tag,
      inputData: input,
      frequency: const Duration(minutes: 15),
      existingWorkPolicy: ExistingPeriodicWorkPolicy.keep,
      constraints: Constraints(networkType: NetworkType.connected),
      backoffPolicy: BackoffPolicy.exponential,
    );
  }

  static Future<void> cancel() async {
    if (!Platform.isAndroid) return;
    await initialize();
    await Workmanager().cancelByTag(_tag);
  }
}
