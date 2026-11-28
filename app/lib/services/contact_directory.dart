import 'dart:convert';

import 'package:flutter/services.dart' show AssetBundle, rootBundle;

import '../models/records.dart';
import '../util/geo.dart';
import 'store.dart';

/// Public organization contacts, checked against the linked publisher's page.
/// Website review does not establish telephone availability or rescue coverage.
class DirectoryContact {
  DirectoryContact.fromJson(Map<String, dynamic> j)
      : id = j['id'] as String,
        name = j['name'] as String,
        phone = j['phone'] as String,
        kind = ContactKind.values.byName(j['kind'] as String),
        locality = j['locality'] as String,
        state = j['state'] as String,
        source = Uri.parse(j['source_url'] as String),
        checkedAt = DateTime.parse(j['source_checked_at'] as String),
        notes = j['notes'] as String,
        location = j['location'] == null
            ? null
            : GeoPoint.fromJson(j['location'] as Map<String, dynamic>) {
    if (id.isEmpty || name.isEmpty || locality.isEmpty || state.isEmpty ||
        !RegExp(r'^\+91[0-9]{10}$').hasMatch(phone) ||
        source.scheme != 'https' || source.host.isEmpty ||
        j['verification_method'] != 'official_website_not_called' ||
        (location != null &&
            (!location!.lat.isFinite || !location!.lon.isFinite ||
             location!.lat.abs() > 90 || location!.lon.abs() > 180 ||
             (j['coordinate_source_url'] as String? ?? '').isEmpty))) {
      throw const FormatException('Invalid directory contact');
    }
  }

  final String id, name, phone, locality, state, notes;
  final ContactKind kind;
  final Uri source;
  final DateTime checkedAt;
  final GeoPoint? location;

  bool needsRecheck(DateTime now) => now.difference(checkedAt).inDays > 180;
  Contact get savedContact => Contact(name: name, phone: phone, kind: kind);
}

class ContactMatch {
  const ContactMatch(this.contact, this.distanceM);
  final DirectoryContact contact;
  final double? distanceM;
}

class ContactDirectory {
  ContactDirectory.fromJson(Map<String, dynamic> j)
      : revision = j['revision'] as int,
        contacts = (j['contacts'] as List)
            .map((r) => DirectoryContact.fromJson(r as Map<String, dynamic>))
            .toList(growable: false) {
    if (j['schema_version'] != 1 || revision < 1 || contacts.isEmpty ||
        contacts.map((c) => c.id).toSet().length != contacts.length) {
      throw const FormatException('Invalid contact directory');
    }
  }

  final int revision;
  final List<DirectoryContact> contacts;

  /// Every query word must occur in the name, locality or state. Unmapped
  /// regional helplines remain searchable, but never receive city-centre points.
  List<ContactMatch> search({String query = '', ContactKind? kind,
      GeoPoint? origin, double? radiusKm}) {
    if (radiusKm != null && (origin == null || !radiusKm.isFinite || radiusKm <= 0)) {
      throw ArgumentError('A positive radius requires a location');
    }
    final words = query.toLowerCase().split(RegExp(r'\s+')).where((w) => w.isNotEmpty);
    final result = <ContactMatch>[];
    for (final c in contacts) {
      final searchable = '${c.name} ${c.locality} ${c.state}'.toLowerCase();
      if (kind != null && c.kind != kind) continue;
      if (!words.every(searchable.contains)) continue;
      final distance = origin != null && c.location != null
          ? distanceMeters(origin, c.location!) : null;
      if (radiusKm != null && (distance == null || distance > radiusKm * 1000)) continue;
      result.add(ContactMatch(c, distance));
    }
    result.sort((a, b) {
      if (a.distanceM != null && b.distanceM != null) {
        final order = a.distanceM!.compareTo(b.distanceM!);
        if (order != 0) return order;
      } else if (a.distanceM != null) {
        return -1;
      } else if (b.distanceM != null) {
        return 1;
      }
      return a.contact.name.compareTo(b.contact.name);
    });
    return result;
  }

  /// Only this app's reviewed bundle is accepted. Replacing an arbitrary cache
  /// with a higher revision cannot inject contact numbers into this directory.
  static Future<ContactDirectory> load(Store store, {AssetBundle? bundle}) async {
    const key = 'reviewed_contact_directory_v1';
    final published = jsonDecode(await (bundle ?? rootBundle)
        .loadString('assets/contact_directory.json')) as Map<String, dynamic>;
    final directory = ContactDirectory.fromJson(published);
    final cached = store.readMap(key);
    if (cached != null && jsonEncode(cached) == jsonEncode(published)) {
      return ContactDirectory.fromJson(cached);
    }
    // The bundled directory already works without network or cache writes.
    try {
      await store.writeMap(key, published);
    } catch (_) {
      // A storage failure must not hide the bundled rescue numbers.
    }
    return directory;
  }
}
