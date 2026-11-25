import 'dart:async';
import 'dart:io' show Platform;

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:flutter_ble_peripheral/flutter_ble_peripheral.dart';
import 'package:flutter_blue_plus/flutter_blue_plus.dart';
import 'package:permission_handler/permission_handler.dart';

import '../models/snake_alert.dart';

/// Sends and receives [SnakeAlert]s as Bluetooth Low Energy advertisements.
///
/// No pairing and no connection: each phone just shouts a 20-byte packet and
/// listens for others. Range is roughly 10 to 100 m depending on walls and
/// phones. Works only while the app is open (no background service yet).
class BleAlertRadio {
  final _peripheral = FlutterBlePeripheral();
  StreamSubscription<List<ScanResult>>? _scanSub;
  Timer? _rotateTimer;
  Timer? _rescanTimer;
  int _rotateIndex = 0;
  bool _listening = false;

  /// Each queued alert is advertised for this long before the next one.
  static const rotateEvery = Duration(seconds: 12);

  /// Android throttles scans that run for more than 30 minutes.
  static const rescanEvery = Duration(minutes: 20);

  /// iOS cannot put manufacturer data in an advertisement, so iPhones can
  /// receive alerts but not send them.
  bool get canBroadcast => Platform.isAndroid;

  Future<bool> get isSupported async {
    try {
      return await FlutterBluePlus.isSupported;
    } catch (_) {
      return false;
    }
  }

  /// Asks for the Bluetooth (and, on old Android, location) permissions.
  /// Returns null when everything is granted, otherwise a short reason.
  Future<String?> ensurePermissions() async {
    if (!await isSupported) return 'This phone has no Bluetooth Low Energy.';
    final needed = <Permission>[
      if (Platform.isAndroid) ...[
        Permission.bluetoothScan,
        Permission.bluetoothAdvertise,
        Permission.bluetoothConnect,
        Permission.locationWhenInUse, // scanning needs it on Android 11 and older
      ],
      if (Platform.isIOS) Permission.bluetooth,
    ];
    final results = await needed.request();
    final denied = results.entries
        .where((e) => !(e.value.isGranted || e.value.isLimited))
        .map((e) => e.key)
        .toList();
    // Android 12+ scans without location (the manifest says the scan is
    // never used for location). Android 11 and older only deliver scan
    // results with location allowed and switched on.
    if (Platform.isAndroid && await _sdkInt() <= 30) {
      if (denied.contains(Permission.locationWhenInUse)) {
        return 'On this Android version, nearby alerts need the location permission.';
      }
      if (!await Permission.locationWhenInUse.serviceStatus.isEnabled) {
        return 'Turn on location (GPS) to receive nearby alerts on this phone.';
      }
    }
    if (denied.isEmpty ||
        (denied.length == 1 && denied.single == Permission.locationWhenInUse)) {
      if (await FlutterBluePlus.adapterState.first != BluetoothAdapterState.on) {
        if (Platform.isAndroid) {
          try {
            await FlutterBluePlus.turnOn();
          } catch (_) {
            return 'Turn on Bluetooth to send and receive alerts.';
          }
        } else {
          return 'Turn on Bluetooth to send and receive alerts.';
        }
      }
      return null;
    }
    return 'Bluetooth permission was not given.';
  }

  static int? _sdk;
  static Future<int> _sdkInt() async {
    try {
      return _sdk ??= await const MethodChannel('serpentguard/video_frames').invokeMethod<int>('sdkInt') ?? 99;
    } catch (_) {
      return 99;
    }
  }

  /// Starts scanning; [onAlert] gets every well-formed alert with its signal strength.
  Future<void> startListening(void Function(SnakeAlert alert, int rssi) onAlert) async {
    await stopListening();
    _listening = true;
    _scanSub = FlutterBluePlus.onScanResults.listen((results) {
      for (final r in results) {
        final bytes = r.advertisementData.manufacturerData[AlertCodec.manufacturerId];
        if (bytes == null) continue;
        final alert = AlertCodec.decode(bytes);
        if (alert != null) onAlert(alert, r.rssi);
      }
    }, onError: (Object e) => debugPrint('BLE scan error: $e'));
    await _startScan();
    _rescanTimer = Timer.periodic(rescanEvery, (_) async {
      if (!_listening) return;
      try {
        await FlutterBluePlus.stopScan();
        await _startScan();
      } catch (e) {
        // Bluetooth was switched off; the next tick tries again.
        debugPrint('BLE rescan failed: $e');
      }
    });
  }

  Future<void> _startScan() => FlutterBluePlus.startScan(
        withMsd: [MsdFilter(AlertCodec.manufacturerId, data: AlertCodec.magic)],
        continuousUpdates: true,
        androidScanMode: AndroidScanMode.balanced,
      );

  Future<void> stopListening() async {
    _listening = false;
    _rescanTimer?.cancel();
    await _scanSub?.cancel();
    _scanSub = null;
    if (FlutterBluePlus.isScanningNow) await FlutterBluePlus.stopScan();
  }

  /// Advertises the alerts [outgoing] returns, one at a time in turn.
  /// [outgoing] is read again on every turn, so new or expired alerts are
  /// picked up without restarting.
  Future<void> startBroadcasting(List<SnakeAlert> Function() outgoing) async {
    if (!canBroadcast) return;
    _rotateTimer?.cancel();
    Future<void> tick() async {
      final queue = outgoing();
      if (queue.isEmpty) {
        if (await _peripheral.isAdvertising) await _peripheral.stop();
        return;
      }
      final alert = queue[_rotateIndex++ % queue.length];
      try {
        if (await _peripheral.isAdvertising) await _peripheral.stop();
        await _peripheral.start(
          advertiseData: AdvertiseDataCore(
            manufacturerId: AlertCodec.manufacturerId,
            manufacturerData: AlertCodec.encode(alert),
          ),
          androidSettings: const AndroidAdvertiseSettings(
            advertiseSettings: AdvertiseSettings(
              advertiseMode: AdvertiseMode.advertiseModeBalanced,
              txPowerLevel: AdvertiseTxPower.advertiseTxPowerHigh,
              connectable: false,
            ),
          ),
        );
      } catch (e) {
        debugPrint('BLE advertise error: $e');
      }
    }

    await tick();
    _rotateTimer = Timer.periodic(rotateEvery, (_) => tick());
  }

  Future<void> stopBroadcasting() async {
    _rotateTimer?.cancel();
    _rotateTimer = null;
    try {
      if (await _peripheral.isAdvertising) await _peripheral.stop();
    } catch (_) {}
  }

  Future<void> dispose() async {
    await stopBroadcasting();
    await stopListening();
  }
}
