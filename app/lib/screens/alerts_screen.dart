import 'package:material_ui/material_ui.dart';
import 'package:provider/provider.dart';

import '../models/snake_alert.dart';
import '../state/app_state.dart';
import '../util/geo.dart';
import '../widgets/common.dart';

class AlertsScreen extends StatelessWidget {
  const AlertsScreen({super.key});

  /// Rough reading of signal strength. Bluetooth distance is very unreliable.
  static String _signal(int rssi) => rssi >= -65
      ? 'strong signal'
      : rssi >= -85
          ? 'medium signal'
          : 'weak signal';

  @override
  Widget build(BuildContext context) {
    final s = context.watch<AppState>();
    final text = Theme.of(context).textTheme;
    final scheme = Theme.of(context).colorScheme;
    final now = DateTime.now();
    final active = s.alertBook.active(now);
    final mine = s.alertBook.sent.where((a) => !a.isExpired(now)).toList();

    return Scaffold(
      appBar: AppBar(title: const Text('Nearby alerts')),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.fromLTRB(16, 4, 16, 96),
          children: [
            Card(
              child: Column(children: [
                SwitchListTile(
                  value: s.alertsOn,
                  onChanged: s.setAlertsOn,
                  secondary: const Icon(Icons.bluetooth_searching_rounded),
                  title: const Text('Send and receive nearby alerts',
                      style: TextStyle(fontWeight: FontWeight.w700)),
                  subtitle: const Text(
                      'Uses Bluetooth, no internet. Works while the app is open.'),
                ),
                const Divider(height: 1, indent: 16, endIndent: 16),
                SwitchListTile(
                  value: s.relayOn,
                  onChanged: s.alertsOn ? s.setRelayOn : null,
                  secondary: const Icon(Icons.share_rounded),
                  title: const Text('Pass alerts on'),
                  subtitle: Text('Re-send alerts you receive, up to '
                      '${SnakeAlert.maxHops} times, so they reach further.'),
                ),
              ]),
            ),
            if (s.alertsError != null)
              Padding(
                padding: const EdgeInsets.only(top: 10),
                child: Text(s.alertsError!, style: TextStyle(color: scheme.error)),
              ),
            if (s.alertsOn && !s.radio.canBroadcast)
              Padding(
                padding: const EdgeInsets.only(top: 10),
                child: Text('This phone can receive alerts but not send them (iPhone).',
                    style: text.bodySmall),
              ),
            SectionTitle('Snakes reported near you (${active.length})'),
            if (active.isEmpty)
              Text(
                s.alertsOn
                    ? 'Listening. Alerts from the last hour appear here.'
                    : 'Turn on nearby alerts to hear about snakes other people report close by.',
                style: text.bodyMedium?.copyWith(color: scheme.onSurfaceVariant),
              ),
            ...active.map((r) {
              final a = r.alert;
              final sp = a.speciesId == SnakeAlert.unknownSpecies ? null : s.catalog.byId(a.speciesId);
              final st = VerdictStyle.of(a.verdict);
              final where = s.position == null
                  ? 'distance unknown'
                  : '${formatDistance(distanceMeters(s.position!, a.location))} '
                      '${compassPoint(bearingDegrees(s.position!, a.location))}';
              return Card(
                margin: const EdgeInsets.only(bottom: 8),
                child: ListTile(
                  leading: CircleAvatar(
                    backgroundColor: st.background,
                    child: Icon(st.icon, color: st.foreground),
                  ),
                  title: Text('${sp?.displayName ?? 'Unidentified snake'} · ${a.verdict.title}',
                      style: const TextStyle(fontWeight: FontWeight.w700)),
                  subtitle: Text(
                    '$where · ${formatAge(now.difference(a.time))} · ${_signal(r.rssi)}'
                    '${a.hops > 0 ? ' · passed on ${a.hops}×' : ''}\n'
                    'Unverified report from another phone.',
                  ),
                  isThreeLine: true,
                ),
              );
            }),
            if (mine.isNotEmpty) ...[
              const SectionTitle('Alerts you are sending'),
              ...mine.map((a) {
                final sp = s.catalog.byId(a.speciesId);
                final left = SnakeAlert.lifetime - now.difference(a.time);
                return Card(
                  margin: const EdgeInsets.only(bottom: 8),
                  child: ListTile(
                    leading: const Icon(Icons.campaign_rounded),
                    title: Text(sp?.displayName ?? 'Unidentified snake'),
                    subtitle: Text('${a.verdict.title} · ${left.inMinutes} min left'),
                  ),
                );
              }),
            ],
            const SectionTitle('Testing'),
            OutlinedButton.icon(
              onPressed: s.simulateIncomingAlert,
              icon: const Icon(Icons.bug_report_outlined),
              label: const Text('Simulate an alert from another phone'),
            ),
            const SizedBox(height: 8),
            Text(
              'Each alert carries only a random id, the species, the verdict, a location '
              'rounded to about 100 m and the time. No name, number or device id.',
              style: text.bodySmall?.copyWith(color: scheme.onSurfaceVariant),
            ),
          ],
        ),
      ),
    );
  }
}
