import '../inference/image_ops.dart' show PixelBox;
import '../util/geo.dart';
import 'species.dart';
import 'verdict.dart';

enum ContactKind {
  emergency('Emergency contact'),
  rescuer('Snake rescuer'),
  forest('Forest / wildlife office'),
  doctor('Doctor / clinic'),
  veterinarian('Veterinarian / animal hospital');

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
    this.result,
  });

  final DateTime time;
  final Verdict verdict;
  final int? speciesId;
  final GeoPoint? location;
  final String? photoPath;

  /// The full result, so the sighting can be reopened in detail. Null for
  /// sightings saved before the app kept it.
  final SavedResult? result;

  Map<String, dynamic> toJson() => {
        'time': time.toIso8601String(),
        'verdict': verdict.code,
        'species': speciesId,
        'loc': location?.toJson(),
        'photo': photoPath,
        if (result != null) 'result': result!.toJson(),
      };
  factory Sighting.fromJson(Map<String, dynamic> j) => Sighting(
        time: DateTime.parse(j['time'] as String),
        verdict: Verdict.fromCode(j['verdict'] as int),
        speciesId: j['species'] as int?,
        location: j['loc'] == null ? null : GeoPoint.fromJson(j['loc'] as Map<String, dynamic>),
        photoPath: j['photo'] as String?,
        result: _savedResult(j['result']),
      );

  static SavedResult? _savedResult(Object? j) {
    if (j is! Map<String, dynamic>) return null;
    try {
      return SavedResult.fromJson(j);
    } catch (_) {
      return null; // an unreadable record still opens, with less detail
    }
  }

  /// The result as the result screen shows it. An old sighting without a
  /// saved result gets its verdict and top species, with no percentage
  /// (probability NaN).
  Identification toIdentification(SpeciesCatalog catalog) {
    final r = result;
    if (r == null) {
      final sp = speciesId == null ? null : catalog.byId(speciesId!);
      return Identification(
        verdict: verdict,
        reason: verdict.advice,
        top: [if (sp != null) SpeciesScore(sp, double.nan)],
        isDemo: false,
      );
    }
    return Identification(
      verdict: verdict,
      reason: r.reason,
      top: [
        for (final (id, p) in r.top)
          if (catalog.byId(id) case final sp?) SpeciesScore(sp, p),
      ],
      isDemo: r.isDemo,
      snakeFound: r.snakeFound,
      message: r.message,
      reasons: r.reasons,
      genusFallback: r.genusFallback,
    );
  }
}

/// What a result screen needs to show a sighting again later.
class SavedResult {
  const SavedResult({
    required this.reason,
    required this.top,
    required this.isDemo,
    required this.snakeFound,
    this.message,
    this.reasons = const [],
    this.genusFallback,
    this.box,
    this.imageWidth,
    this.imageHeight,
  });

  factory SavedResult.of(Identification id) {
    final d = id.details;
    return SavedResult(
      reason: id.reason,
      top: [for (final t in id.top) (t.species.id, t.probability)],
      isDemo: id.isDemo,
      snakeFound: id.snakeFound,
      message: id.message,
      reasons: id.reasons,
      genusFallback: id.genusFallback,
      box: d?.gate.box,
      imageWidth: d?.width,
      imageHeight: d?.height,
    );
  }

  factory SavedResult.fromJson(Map<String, dynamic> j) {
    final g = j['genus'] as Map<String, dynamic>?;
    final b = (j['box'] as List?)?.map((v) => (v as num).toDouble()).toList();
    return SavedResult(
      reason: j['reason'] as String,
      top: [
        for (final t in j['top'] as List) ((t as List)[0] as int, (t[1] as num).toDouble()),
      ],
      isDemo: j['demo'] as bool? ?? false,
      snakeFound: j['snake'] as bool? ?? true,
      message: j['message'] as String?,
      reasons: ((j['reasons'] as List?) ?? const []).cast<String>(),
      genusFallback: g == null
          ? null
          : GenusFallback(g['genus'] as String, (g['p'] as num).toDouble(), g['who'] as bool),
      box: b == null ? null : PixelBox(b[0], b[1], b[2], b[3]),
      imageWidth: j['w'] as int?,
      imageHeight: j['h'] as int?,
    );
  }

  final String reason;

  /// (species id, probability), most likely first.
  final List<(int, double)> top;
  final bool isDemo, snakeFound;
  final String? message;
  final List<String> reasons;
  final GenusFallback? genusFallback;

  /// The detector's box on the photo, and the photo's size, for the overlay.
  final PixelBox? box;
  final int? imageWidth, imageHeight;

  Map<String, dynamic> toJson() => {
        'reason': reason,
        'top': [for (final (id, p) in top) [id, p]],
        'demo': isDemo,
        'snake': snakeFound,
        'message': message,
        'reasons': reasons,
        if (genusFallback != null)
          'genus': {'genus': genusFallback!.genus, 'p': genusFallback!.probability, 'who': genusFallback!.whoListed},
        if (box != null) 'box': [box!.left, box!.top, box!.right, box!.bottom],
        'w': imageWidth,
        'h': imageHeight,
      };
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
    if (s != null && s.isNotEmpty) {
      return !RegExp(r'general|emergency|internal|toxicolog|intensive|trauma').hasMatch(s);
    }
    // OpenStreetMap often has no specialty tag, but the name says it:
    // "Mahaveer Eye Hospital", "Smile Dental Care", "Supra Diagnostics".
    return _specialtyName.hasMatch(name.toLowerCase());
  }

  static final _specialtyName = RegExp(
    r'\b(eye|eyes|netralaya|nethralaya|nethra|netra|optical|dental|dentist|dent|teeth|ent|skin|derma\w*|cosmetic|'
    r'maternity|fertility|ivf|diagnostic\w*|scan|scans|imaging|lab|labs|physio\w*|ayurved\w*|homeopath\w*|'
    r'veterinary|vet|pet|cancer|onco\w*|psychiatr\w*|rehab\w*|de-addiction)\b',
  );

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
