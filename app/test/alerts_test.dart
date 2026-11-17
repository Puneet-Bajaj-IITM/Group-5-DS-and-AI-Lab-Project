import 'dart:math';

import 'package:flutter_test/flutter_test.dart';
import 'package:serpentguard/models/snake_alert.dart';
import 'package:serpentguard/models/verdict.dart';
import 'package:serpentguard/services/alert_book.dart';
import 'package:serpentguard/util/geo.dart';

SnakeAlert alert({
  int id = 42,
  int species = 7,
  Verdict verdict = Verdict.dangerous,
  GeoPoint location = const GeoPoint(12.9716, 77.5946),
  DateTime? time,
  int hops = 0,
}) =>
    SnakeAlert(
      id: id,
      speciesId: species,
      verdict: verdict,
      location: location,
      time: time ?? DateTime(2026, 11, 18, 10, 30),
      hops: hops,
    );

void main() {
  group('AlertCodec', () {
    test('round-trips every field, location to 0.001 degrees', () {
      final a = alert(id: 0xDEADBEEF, species: 183, verdict: Verdict.unknown, hops: 2);
      final bytes = AlertCodec.encode(a);
      expect(bytes.length, AlertCodec.length);
      expect(bytes.length, lessThanOrEqualTo(24), reason: 'must fit a legacy advertisement');
      final b = AlertCodec.decode(bytes)!;
      expect(b.id, 0xDEADBEEF);
      expect(b.speciesId, 183);
      expect(b.verdict, Verdict.unknown);
      expect(b.hops, 2);
      expect(b.location.lat, closeTo(12.972, 1e-9));
      expect(b.location.lon, closeTo(77.595, 1e-9));
      expect(b.time, a.time);
    });

    test('handles southern and western coordinates', () {
      final b = AlertCodec.decode(
          AlertCodec.encode(alert(location: const GeoPoint(-33.8688, -151.2093))))!;
      expect(b.location.lat, closeTo(-33.869, 1e-9));
      expect(b.location.lon, closeTo(-151.209, 1e-9));
    });

    test('rejects other advertisements', () {
      final good = AlertCodec.encode(alert());
      expect(AlertCodec.decode(good.sublist(0, 10)), isNull);
      expect(AlertCodec.decode([0x00, ...good.skip(1)]), isNull);
      final badVerdict = [...good]..[8] = 9;
      expect(AlertCodec.decode(badVerdict), isNull);
    });

    test('random ids use the full 32 bits', () {
      final ids = List.generate(200, (_) => SnakeAlert.newId(Random(1)));
      expect(ids.every((i) => i >= 0 && i < (1 << 32)), isTrue);
    });
  });

  group('AlertBook', () {
    final now = DateTime(2026, 11, 18, 11, 0);
    const me = GeoPoint(12.9716, 77.5946);

    test('accepts a new alert once and keeps the strongest signal', () {
      final book = AlertBook();
      final a = alert(time: now.subtract(const Duration(minutes: 5)));
      expect(book.accept(a, -90, now, me: me), isTrue);
      expect(book.accept(a, -60, now, me: me), isFalse);
      expect(book.received[a.id]!.rssi, -60);
    });

    test('drops expired, future, far-away and own alerts', () {
      final book = AlertBook();
      expect(book.accept(alert(time: now.subtract(const Duration(minutes: 61))), -70, now), isFalse);
      expect(book.accept(alert(time: now.add(const Duration(minutes: 30))), -70, now), isFalse);
      expect(
          book.accept(alert(location: const GeoPoint(13.2, 77.6), time: now), -70, now, me: me),
          isFalse);
      final mine = alert(id: 7, time: now);
      book.recordSent(mine);
      expect(book.accept(mine.relayed(), -70, now), isFalse);
    });

    test('relays only fresh alerts under the hop limit, with one more hop', () {
      final book = AlertBook();
      book.accept(alert(id: 1, time: now, hops: 0), -70, now);
      book.accept(alert(id: 2, time: now, hops: SnakeAlert.maxHops), -70, now);
      final out = book.outgoing(now, relay: true);
      expect(out.map((a) => a.id), [1]);
      expect(out.single.hops, 1);
      expect(book.outgoing(now, relay: false), isEmpty);
    });

    test('limits sending to 3 an hour, 2 minutes apart', () {
      final book = AlertBook();
      expect(book.nextSendAllowed(now), isNull);
      book.recordSent(alert(id: 1, time: now));
      expect(book.nextSendAllowed(now.add(const Duration(minutes: 1))), now.add(AlertBook.minGap));
      book.recordSent(alert(id: 2, time: now.add(const Duration(minutes: 3))));
      book.recordSent(alert(id: 3, time: now.add(const Duration(minutes: 6))));
      expect(book.nextSendAllowed(now.add(const Duration(minutes: 10))),
          now.add(AlertBook.sendWindow));
      expect(book.nextSendAllowed(now.add(const Duration(minutes: 61))), isNull);
    });
  });
}
