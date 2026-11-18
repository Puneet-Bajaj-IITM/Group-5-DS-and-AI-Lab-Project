import '../models/snake_alert.dart';
import '../util/geo.dart';

class ReceivedAlert {
  ReceivedAlert(this.alert, this.firstSeen, this.rssi);

  SnakeAlert alert;
  final DateTime firstSeen;

  /// Strongest signal seen, in dBm. Around -60 is close, below -90 is far.
  int rssi;
}

/// Keeps track of alerts sent and received: drops duplicates, expired and
/// far-away alerts, our own alerts coming back, and limits how often this
/// phone can send. Pure logic with no radio, so it is unit tested.
class AlertBook {
  final List<SnakeAlert> sent = [];
  final Map<int, ReceivedAlert> received = {};

  /// Alerts further than this from the user are ignored.
  static const maxDistanceMeters = 5000.0;

  /// At most this many own alerts per [sendWindow], and [minGap] apart.
  static const maxSendsPerWindow = 3;
  static const sendWindow = Duration(minutes: 60);
  static const minGap = Duration(minutes: 2);

  Set<int> get _ownIds => sent.map((a) => a.id).toSet();

  /// Records an incoming alert. Returns true only for a new, relevant one.
  bool accept(SnakeAlert alert, int rssi, DateTime now, {GeoPoint? me}) {
    if (_ownIds.contains(alert.id)) return false;
    if (alert.isExpired(now)) return false;
    if (alert.time.isAfter(now.add(const Duration(minutes: 5)))) return false; // bad clock
    if (me != null && distanceMeters(me, alert.location) > maxDistanceMeters) return false;
    final existing = received[alert.id];
    if (existing != null) {
      if (rssi > existing.rssi) existing.rssi = rssi;
      if (alert.hops < existing.alert.hops) existing.alert = alert;
      return false;
    }
    received[alert.id] = ReceivedAlert(alert, now, rssi);
    return true;
  }

  /// Null if this phone may send a new alert now, else when it may.
  DateTime? nextSendAllowed(DateTime now) {
    final recent = sent.where((a) => now.difference(a.time) < sendWindow).toList()
      ..sort((a, b) => a.time.compareTo(b.time));
    if (recent.isNotEmpty && now.difference(recent.last.time) < minGap) {
      return recent.last.time.add(minGap);
    }
    if (recent.length >= maxSendsPerWindow) {
      return recent.first.time.add(sendWindow);
    }
    return null;
  }

  void recordSent(SnakeAlert alert) => sent.add(alert);

  /// What this phone should advertise now: its own live alerts, then
  /// received ones that may still be passed on (if [relay] is on).
  List<SnakeAlert> outgoing(DateTime now, {required bool relay}) => [
        ...sent.where((a) => !a.isExpired(now)),
        if (relay)
          ...received.values
              .map((r) => r.alert)
              .where((a) => !a.isExpired(now) && a.hops < SnakeAlert.maxHops)
              .map((a) => a.relayed()),
      ];

  /// Live received alerts, newest first.
  List<ReceivedAlert> active(DateTime now) => received.values
      .where((r) => !r.alert.isExpired(now))
      .toList()
    ..sort((a, b) => b.alert.time.compareTo(a.alert.time));

  void prune(DateTime now) {
    received.removeWhere((_, r) => r.alert.isExpired(now));
    sent.removeWhere((a) => now.difference(a.time) > sendWindow * 2);
  }
}
