import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:flutter_ble_peripheral/flutter_ble_peripheral.dart';
import 'package:flutter_blue_plus/flutter_blue_plus.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:serpentguard/models/snake_alert.dart';
import 'package:serpentguard/models/verdict.dart';
import 'package:serpentguard/services/ble_alert_radio.dart';
import 'package:serpentguard/util/geo.dart';

/// RF probe only: ST magic is rejected by production SG receivers. No sighting,
/// location permission result, inference verdict or app alert queue is fabricated.
void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();
  const role = String.fromEnvironment('BLE_PROBE_ROLE', defaultValue: 'self');
  const session = int.fromEnvironment('BLE_PROBE_SESSION', defaultValue: 0);
  testWidgets('BLE hardware probe ($role)', (tester) async {
    await tester.runAsync(() async {
      expect(Platform.isAndroid, isTrue, reason: 'Use a physical Android phone');
      expect(['self', 'sender', 'receiver'], contains(role));
      expect(session, inInclusiveRange(1, 0xffffffff), reason: 'Set a fresh shared session ID');
      final radio = BleAlertRadio();
      final peripheral = FlutterBlePeripheral();
      StreamSubscription<List<ScanResult>>? subscription;
      var packets = 0;
      int? firstRssi;
      final watch = Stopwatch();
      final bytes = AlertCodec.encode(SnakeAlert(id: session,
          speciesId: SnakeAlert.unknownSpecies, verdict: Verdict.unknown,
          location: const GeoPoint(0, 0), time: DateTime.now()));
      bytes[1] = 0x54; // ST, not production SG.
      expect(AlertCodec.decode(bytes), isNull);
      try {
        expect(await radio.ensurePermissions(), isNull);
        if (role != 'sender') {
          subscription = FlutterBluePlus.onScanResults.listen((results) {
            for (final result in results) {
              final data = result.advertisementData.manufacturerData[AlertCodec.manufacturerId];
              if (data == null || data.length != AlertCodec.length) continue;
              if (List.generate(7, (i) => data[i] == bytes[i]).every((v) => v)) {
                packets++;
                firstRssi ??= result.rssi;
              }
            }
          });
          await FlutterBluePlus.startScan(
              withMsd: [MsdFilter(AlertCodec.manufacturerId, data: [0x53, 0x54])],
              continuousUpdates: true, androidScanMode: AndroidScanMode.balanced);
          expect(FlutterBluePlus.isScanningNow, isTrue);
        }
        if (role != 'receiver') {
          await peripheral.start(advertiseData: AdvertiseDataCore(
              manufacturerId: AlertCodec.manufacturerId, manufacturerData: bytes),
              androidSettings: const AndroidAdvertiseSettings(advertiseSettings:
                  AdvertiseSettings(advertiseMode: AdvertiseMode.advertiseModeBalanced,
                      txPowerLevel: AdvertiseTxPower.advertiseTxPowerHigh, connectable: false)));
          expect(await peripheral.isAdvertising, isTrue);
        }
        watch.start();
        await Future<void>.delayed(const Duration(seconds: 30));
        watch.stop();
        // ignore: avoid_print
        print('BLE_PROBE_RECEIPT ${jsonEncode({'role': role, 'session': session,
          'utc': DateTime.now().toUtc().toIso8601String(), 'seconds': watch.elapsedMilliseconds / 1000,
          'matching_scan_reports': packets, 'first_rssi': firstRssi,
          'scope': role == 'self' ? 'local scan/advertise startup only; no delivery claim' : 'isolated ST radio probe'})}');
        if (role == 'receiver') {
          expect(packets, greaterThan(0), reason: 'No matching packet received from the other phone');
        }
      } finally {
        await subscription?.cancel();
        if (FlutterBluePlus.isScanningNow) await FlutterBluePlus.stopScan();
        if (await peripheral.isAdvertising) await peripheral.stop();
        await radio.dispose();
      }
    });
  }, timeout: const Timeout(Duration(minutes: 3)));
}
