import 'dart:io' show Platform;

import 'package:url_launcher/url_launcher.dart';

import '../models/records.dart';
import '../models/species.dart';
import '../models/verdict.dart';
import '../util/geo.dart';

/// National numbers that work across India. State forest and rescue
/// helplines differ by district, so users add those as contacts.
class EmergencyNumber {
  const EmergencyNumber(this.number, this.label, this.detail);
  final String number;
  final String label;
  final String detail;
}

const nationalNumbers = [
  EmergencyNumber('112', 'Emergency', 'National emergency response (police, fire, ambulance)'),
  EmergencyNumber('108', 'Ambulance', 'Emergency ambulance in most states'),
  EmergencyNumber('102', 'Ambulance (102)', 'Ambulance service in many states'),
];

/// Opens the dialler, the SMS app and maps. The app never places a call or
/// sends a message on its own: the user always presses the final button.
class SosService {
  Future<bool> call(String number) =>
      launchUrl(Uri(scheme: 'tel', path: _clean(number)));

  /// Opens the SMS app with every recipient and the message filled in.
  /// Works with mobile signal and no data; nothing works with no signal at all.
  Future<bool> textContacts(List<Contact> contacts, String message) {
    final separator = Platform.isIOS ? ',' : ';';
    final uri = Uri(
      scheme: 'sms',
      path: contacts.map((c) => _clean(c.phone)).join(separator),
      query: 'body=${Uri.encodeComponent(message)}',
    );
    return launchUrl(uri);
  }

  Future<bool> directions(GeoPoint to) => launchUrl(
        Uri.parse('https://www.google.com/maps/dir/?api=1'
            '&destination=${to.lat},${to.lon}&travelmode=driving'),
        mode: LaunchMode.externalApplication,
      );

  static String _clean(String phone) => phone.replaceAll(RegExp(r'[^0-9+]'), '');

  /// The SOS text. Short enough to send as one or two SMS.
  static String message({
    required bool bitten,
    GeoPoint? location,
    double? accuracy,
    Species? species,
    Verdict? verdict,
  }) {
    final b = StringBuffer(bitten
        ? 'EMERGENCY: I have been bitten by a snake and need help.'
        : 'Snake alert: I need help with a snake near me.');
    if (species != null || verdict != null) {
      b.write(' Snake: ');
      b.write(species?.fullName ?? 'not identified');
      if (verdict != null) b.write(' (${verdict.title})');
      b.write('.');
    }
    if (location != null) {
      b.write(' My location: ${location.mapsUrl}');
      if (accuracy != null) b.write(' (within ${accuracy.round()} m)');
      b.write('.');
    } else {
      b.write(' Location not available, please call me.');
    }
    b.write(' Sent from SerpentGuard.');
    return b.toString();
  }
}
