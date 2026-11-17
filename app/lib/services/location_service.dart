import 'dart:async';

import 'package:geolocator/geolocator.dart';

import '../util/geo.dart';

enum LocationStatus { unknown, serviceOff, denied, deniedForever, granted }

/// Foreground location only: the app asks while it is open and never tracks
/// in the background.
class LocationService {
  StreamSubscription<Position>? _sub;

  Future<LocationStatus> ensurePermission({bool request = true}) async {
    if (!await Geolocator.isLocationServiceEnabled()) return LocationStatus.serviceOff;
    var permission = await Geolocator.checkPermission();
    if (permission == LocationPermission.denied && request) {
      permission = await Geolocator.requestPermission();
    }
    return switch (permission) {
      LocationPermission.always || LocationPermission.whileInUse => LocationStatus.granted,
      LocationPermission.deniedForever => LocationStatus.deniedForever,
      _ => LocationStatus.denied,
    };
  }

  /// A fresh fix, or the last known one if a fresh fix takes too long.
  Future<({GeoPoint point, double accuracy})?> current() async {
    try {
      final p = await Geolocator.getCurrentPosition(
        locationSettings: const LocationSettings(
          accuracy: LocationAccuracy.high,
          timeLimit: Duration(seconds: 12),
        ),
      );
      return (point: GeoPoint(p.latitude, p.longitude), accuracy: p.accuracy);
    } catch (_) {
      final last = await Geolocator.getLastKnownPosition();
      if (last == null) return null;
      return (point: GeoPoint(last.latitude, last.longitude), accuracy: last.accuracy);
    }
  }

  void watch(void Function(GeoPoint point, double accuracy) onFix) {
    _sub?.cancel();
    _sub = Geolocator.getPositionStream(
      locationSettings: const LocationSettings(
        accuracy: LocationAccuracy.high,
        distanceFilter: 25,
      ),
    ).listen(
      (p) => onFix(GeoPoint(p.latitude, p.longitude), p.accuracy),
      onError: (_) {},
    );
  }

  Future<void> openSettings() => Geolocator.openLocationSettings();
  Future<void> openAppSettings() => Geolocator.openAppSettings();

  void dispose() => _sub?.cancel();
}
