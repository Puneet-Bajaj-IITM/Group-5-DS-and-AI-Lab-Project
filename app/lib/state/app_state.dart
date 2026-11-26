import 'dart:async';

import 'package:flutter/foundation.dart';

import '../models/records.dart';
import '../models/snake_alert.dart';
import '../models/species.dart';
import '../models/verdict.dart';
import '../services/address_service.dart';
import '../services/alert_book.dart';
import '../services/ble_alert_radio.dart';
import '../services/classifier.dart';
import '../services/hospital_service.dart';
import '../services/location_service.dart';
import '../services/sos_service.dart';
import '../services/store.dart';
import '../util/geo.dart';

/// Everything the screens share. One instance for the life of the app.
class AppState extends ChangeNotifier {
  AppState({
    required this.catalog,
    required this.classifier,
    required Store store,
    LocationService? location,
    HospitalService? hospitals,
    BleAlertRadio? radio,
    SosService? sos,
    AddressService? addresses,
  }) : _store = store,
       _addresses = addresses ?? AddressService(store),
       _location = location ?? LocationService(),
       _hospitals = hospitals ?? HospitalService(store),
       radio = radio ?? BleAlertRadio(),
       sos = sos ?? SosService();

  final SpeciesCatalog catalog;
  final SnakeClassifier classifier;
  final Store _store;
  final AddressService _addresses;
  final LocationService _location;
  final HospitalService _hospitals;
  final BleAlertRadio radio;
  final SosService sos;
  final AlertBook alertBook = AlertBook();

  Timer? _ticker;

  // ---- location ----
  LocationStatus locationStatus = LocationStatus.unknown;
  GeoPoint? position;
  double? accuracy;

  /// A readable street address for [position], when one is known.
  String? address;
  GeoPoint? _addressFor;
  bool _addressBusy = false;

  /// What to show for "where am I": the address, or coordinates if none yet.
  String? get placeLabel => address ?? position?.toString();

  // ---- saved data ----
  List<Contact> contacts = [];
  List<Sighting> sightings = [];

  // ---- hospitals ----
  List<Hospital> hospitals = [];
  DateTime? hospitalsFetched;
  bool hospitalsLoading = false;
  String? hospitalsError;

  // ---- nearby alerts ----
  bool alertsOn = false;
  bool relayOn = true;
  String? alertsError;

  /// Set when a new alert arrives; the shell shows it once and clears it.
  ReceivedAlert? incoming;

  static const _kContacts = 'contacts';
  static const _kSightings = 'sightings';
  static const _kAlertsOn = 'alerts.on';
  static const _kRelayOn = 'alerts.relay';

  Future<void> init() async {
    contacts = _store.readList(_kContacts, Contact.fromJson);
    sightings = _store.readList(_kSightings, Sighting.fromJson);
    relayOn = _store.getBool(_kRelayOn, fallback: true);
    final cache = _hospitals.cached();
    hospitals = cache.hospitals;
    hospitalsFetched = cache.fetched;
    notifyListeners();

    await refreshLocation();
    if (_store.getBool(_kAlertsOn)) await setAlertsOn(true);

    // Ages ("4 min ago") and expiry need a regular redraw.
    _ticker = Timer.periodic(const Duration(seconds: 30), (_) {
      alertBook.prune(DateTime.now());
      notifyListeners();
    });
  }

  // ---------------------------------------------------------------- location

  Future<void> refreshLocation({bool request = true}) async {
    locationStatus = await _location.ensurePermission(request: request);
    notifyListeners();
    if (locationStatus != LocationStatus.granted) return;
    final fix = await _location.current();
    if (fix != null) {
      _setFix(fix.point, fix.accuracy);
      try {
        final saved = await _hospitals.savedNear(fix.point);
        hospitals = saved.hospitals;
        hospitalsFetched = saved.oldest;
        hospitalsError = hospitals.isEmpty
            ? 'No hospitals saved within 50 km of this location.'
            : null;
        notifyListeners();
      } catch (_) {
        hospitals = hospitals
            .where((h) => distanceMeters(fix.point, h.location) <= 50000)
            .toList();
      }
    }
    _location.watch(_setFix);
    if (hospitals.isEmpty && position != null) unawaited(refreshHospitals());
  }

  void _setFix(GeoPoint p, double acc) {
    final first = position == null;
    position = p;
    accuracy = acc;
    // Never show an address for a place we have left: drop it after moving
    // more than about 100 m, use a saved one for the new spot if there is
    // one, and look it up (again) until a lookup succeeds.
    if (_addressFor == null || distanceMeters(_addressFor!, p) > 100) {
      address = _addresses.cached(p);
      unawaited(_lookupAddress(p));
    }
    // A first fix that arrives late (after the app opened without one)
    // still needs the hospitals saved for this area.
    if (first) unawaited(_loadSavedHospitals(p));
    notifyListeners();
  }

  Future<void> _lookupAddress(GeoPoint p) async {
    if (_addressBusy) return;
    _addressBusy = true;
    try {
      final found = await _addresses.lookup(p);
      // Only a successful lookup counts; otherwise the next fix tries again.
      if (found != null && position != null && distanceMeters(position!, p) <= 100) {
        _addressFor = p;
        address = found;
        notifyListeners();
      }
    } finally {
      _addressBusy = false;
    }
  }

  Future<void> _loadSavedHospitals(GeoPoint p) async {
    try {
      final saved = await _hospitals.savedNear(p);
      if (saved.hospitals.isNotEmpty) {
        hospitals = saved.hospitals;
        hospitalsFetched = saved.oldest;
        hospitalsError = null;
        notifyListeners();
      } else if (hospitals.isEmpty) {
        unawaited(refreshHospitals());
      }
    } catch (_) {}
  }

  Future<void> openLocationSettings() =>
      locationStatus == LocationStatus.serviceOff
      ? _location.openSettings()
      : _location.openAppSettings();

  // ---------------------------------------------------------------- contacts

  List<Contact> get emergencyContacts =>
      contacts.where((c) => c.kind == ContactKind.emergency).toList();

  Future<void> addContact(Contact c) async {
    contacts = [...contacts, c];
    await _store.writeList(_kContacts, contacts, (Contact c) => c.toJson());
    notifyListeners();
  }

  Future<void> removeContact(Contact c) async {
    contacts = contacts.where((x) => x != c).toList();
    await _store.writeList(_kContacts, contacts, (Contact c) => c.toJson());
    notifyListeners();
  }

  // ---------------------------------------------------------------- sightings

  Future<void> saveSighting(Identification id, {String? photoPath}) async {
    // A photo with no snake in it is not a sighting (and must not be pinned
    // on the map as one).
    if (!id.snakeFound) return;
    sightings = [
      Sighting(
        time: DateTime.now(),
        verdict: id.verdict,
        speciesId: id.topSpecies?.id,
        location: position,
        photoPath: photoPath,
      ),
      ...sightings,
    ].take(200).toList();
    await _store.writeList(_kSightings, sightings, (Sighting s) => s.toJson());
    notifyListeners();
  }

  Future<void> clearSightings() async {
    sightings = [];
    await _store.writeList(_kSightings, sightings, (Sighting s) => s.toJson());
    notifyListeners();
  }

  // ---------------------------------------------------------------- hospitals

  /// Hospitals within 50 km, nearest first. Without a location we cannot say
  /// which saved hospitals are near, so the list stays empty.
  List<Hospital> get hospitalsNearby => position == null
      ? const []
      : HospitalService.sortedByDistance(
          hospitals
              .where((h) => distanceMeters(position!, h.location) <= 50000)
              .toList(),
          position!,
        );

  Future<void> refreshHospitals() async {
    if (hospitalsLoading) return;
    final centre = position;
    if (centre == null) {
      hospitalsError = 'Location is needed to find hospitals.';
      notifyListeners();
      return;
    }
    hospitalsLoading = true;
    hospitalsError = null;
    notifyListeners();
    try {
      hospitals = await _hospitals.refresh(centre);
      hospitalsFetched = DateTime.now();
    } catch (_) {
      hospitalsError = hospitals.isEmpty
          ? 'Could not download hospitals. Check the internet connection.'
          : 'Offline: showing the list saved earlier.';
    } finally {
      hospitalsLoading = false;
      notifyListeners();
    }
  }

  // ---------------------------------------------------------------- alerts

  Future<void> setAlertsOn(bool on) async {
    alertsError = null;
    if (!on) {
      alertsOn = false;
      await radio.dispose();
      await _store.setBool(_kAlertsOn, false);
      notifyListeners();
      return;
    }
    final problem = await radio.ensurePermissions();
    if (problem != null) {
      alertsOn = false;
      alertsError = problem;
      notifyListeners();
      return;
    }
    alertsOn = true;
    await _store.setBool(_kAlertsOn, true);
    notifyListeners();
    await radio.startListening(_onRadioAlert);
    await radio.startBroadcasting(
      () => alertBook.outgoing(DateTime.now(), relay: relayOn),
    );
  }

  Future<void> setRelayOn(bool on) async {
    relayOn = on;
    await _store.setBool(_kRelayOn, on);
    notifyListeners();
  }

  void _onRadioAlert(SnakeAlert alert, int rssi) {
    // Phones repeat each advertisement many times a second; redraw only
    // when a new alert arrives.
    if (alertBook.accept(alert, rssi, DateTime.now(), me: position)) {
      incoming = alertBook.received[alert.id];
      notifyListeners();
    }
  }

  void clearIncoming() => incoming = null;

  /// Sends a warning to phones nearby. Returns a message for the user.
  Future<String> sendAlert(Identification id) async {
    if (id.isDemo) return 'Demo results cannot be broadcast to nearby phones.';
    if (!id.snakeFound) return 'A snake sighting is required to send an alert.';
    if (!alertsOn) return 'Turn on nearby alerts first (Alerts tab).';
    if (!radio.canBroadcast) {
      return 'Sending alerts works on Android phones only.';
    }
    final where = position;
    if (where == null) return 'Location is needed to send an alert.';
    final now = DateTime.now();
    final wait = alertBook.nextSendAllowed(now);
    if (wait != null) {
      return 'Alert limit reached. You can send another in ${wait.difference(now).inMinutes + 1} min.';
    }
    alertBook.recordSent(
      SnakeAlert(
        id: SnakeAlert.newId(),
        speciesId: id.topSpecies?.id ?? SnakeAlert.unknownSpecies,
        verdict: id.verdict,
        location: where,
        time: now,
      ),
    );
    notifyListeners();
    return 'Alert queued for Bluetooth broadcast for ${SnakeAlert.lifetime.inMinutes} minutes. '
        'Delivery to nearby phones is not confirmed.';
  }

  /// For testing on one phone: pretend an alert arrived from someone else.
  void simulateIncomingAlert() {
    final here = position ?? const GeoPoint(12.9716, 77.5946);
    final krait = catalog.all.firstWhere((s) => s.name == 'Bungarus caeruleus');
    final alert = SnakeAlert(
      id: SnakeAlert.newId(),
      speciesId: krait.id,
      verdict: Verdict.dangerous,
      location: GeoPoint(here.lat + 0.0021, here.lon + 0.0015),
      time: DateTime.now().subtract(const Duration(minutes: 3)),
      hops: 1,
    );
    // Exercise the banner only. Test packets must never enter the relay book.
    incoming = ReceivedAlert(alert, DateTime.now(), -78);
    notifyListeners();
  }

  @override
  void dispose() {
    _ticker?.cancel();
    _location.dispose();
    radio.dispose();
    super.dispose();
  }
}
