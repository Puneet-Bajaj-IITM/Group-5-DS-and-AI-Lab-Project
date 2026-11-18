import '../util/geo.dart';
import 'verdict.dart';

enum ContactKind {
  emergency('Emergency contact'),
  rescuer('Snake rescuer'),
  forest('Forest / wildlife office'),
  doctor('Doctor / clinic');

  const ContactKind(this.label);
  final String label;
}

/// A phone number the user saved. Emergency contacts get the SOS text message.
class Contact {
  const Contact({required this.name, required this.phone, required this.kind});

  final String name;
  final String phone;
  final ContactKind kind;

  Map<String, dynamic> toJson() => {'name': name, 'phone': phone, 'kind': kind.name};
  factory Contact.fromJson(Map<String, dynamic> j) => Contact(
        name: j['name'] as String,
        phone: j['phone'] as String,
        kind: ContactKind.values.asNameMap()[j['kind']] ?? ContactKind.emergency,
      );
}

/// A snake the user identified, kept on the phone only.
class Sighting {
  const Sighting({
    required this.time,
    required this.verdict,
    this.speciesId,
    this.location,
    this.photoPath,
  });

  final DateTime time;
  final Verdict verdict;
  final int? speciesId;
  final GeoPoint? location;
  final String? photoPath;

  Map<String, dynamic> toJson() => {
        'time': time.toIso8601String(),
        'verdict': verdict.code,
        'species': speciesId,
        'loc': location?.toJson(),
        'photo': photoPath,
      };
  factory Sighting.fromJson(Map<String, dynamic> j) => Sighting(
        time: DateTime.parse(j['time'] as String),
        verdict: Verdict.fromCode(j['verdict'] as int),
        speciesId: j['species'] as int?,
        location: j['loc'] == null ? null : GeoPoint.fromJson(j['loc'] as Map<String, dynamic>),
        photoPath: j['photo'] as String?,
      );
}

/// A hospital from OpenStreetMap, cached on the phone for offline use.
class Hospital {
  const Hospital({
    required this.name,
    required this.location,
    this.phone,
    this.emergency,
    this.address,
    this.speciality,
  });

  final String name;
  final GeoPoint location;
  final String? phone;

  /// OSM `emergency=yes/no`, when mapped.
  final bool? emergency;
  final String? address;

  /// OSM `healthcare:speciality`, e.g. "ophthalmology", when mapped.
  final String? speciality;

  /// A specialty-only hospital (eye, dental, maternity...) is a poor first
  /// stop for a snakebite, so it is listed after general ones.
  bool get isSpecialtyOnly {
    final s = speciality?.toLowerCase();
    if (s == null || s.isEmpty) return false;
    return !RegExp(r'general|emergency|internal|toxicolog|intensive|trauma').hasMatch(s);
  }

  Map<String, dynamic> toJson() => {
        'name': name,
        'loc': location.toJson(),
        'phone': phone,
        'emergency': emergency,
        'address': address,
        'speciality': speciality,
      };
  factory Hospital.fromJson(Map<String, dynamic> j) => Hospital(
        name: j['name'] as String,
        location: GeoPoint.fromJson(j['loc'] as Map<String, dynamic>),
        phone: j['phone'] as String?,
        emergency: j['emergency'] as bool?,
        address: j['address'] as String?,
        speciality: j['speciality'] as String?,
      );
}
