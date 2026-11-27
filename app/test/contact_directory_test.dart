import 'dart:convert';
import 'dart:io';

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:serpentguard/models/records.dart';
import 'package:serpentguard/services/contact_directory.dart';
import 'package:serpentguard/services/store.dart';
import 'package:serpentguard/util/geo.dart';

class DirectoryBundle extends CachingAssetBundle {
  DirectoryBundle(this.content);
  final String content;
  @override
  Future<ByteData> load(String key) async =>
      ByteData.sublistView(Uint8List.fromList(utf8.encode(content)));
}

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  final raw = File('assets/contact_directory.json').readAsStringSync();
  final directory = ContactDirectory.fromJson(jsonDecode(raw) as Map<String, dynamic>);

  test('offline locality and kind search; unmapped helplines keep no distance', () {
    expect(directory.search(query: 'chennai', kind: ContactKind.veterinarian), hasLength(2));
    expect(directory.search(query: 'delhi').single.distanceM, isNull);
    expect(directory.search(query: 'unknown town'), isEmpty);
    expect(directory.search(query: 'Maharashtra Pune').single.contact.id, 'resq-pune');
  });

  test('nearest mapped centre first; radius never invents helpline points', () {
    const pune = GeoPoint(18.4949, 73.7652);
    expect(directory.search(origin: pune).first.contact.id, 'resq-pune');
    final nearby = directory.search(origin: pune, radiusKm: 25);
    expect(nearby, hasLength(1));
    expect(nearby.single.distanceM, lessThan(10));
    expect(directory.search(origin: const GeoPoint(28.6, 77.2), radiusKm: 25), isEmpty);
    expect(() => directory.search(radiusKm: 25), throwsArgumentError);
    expect(directory.contacts.every((c) => c.needsRecheck(DateTime(2027, 6, 1))), isTrue);
  });

  test('bundle repairs corrupt or tampered cache and survives reopen offline', () async {
    SharedPreferences.setMockInitialValues({'reviewed_contact_directory_v1': '{broken'});
    final store = await Store.open();
    final bundle = DirectoryBundle(raw);
    expect((await ContactDirectory.load(store, bundle: bundle)).contacts, hasLength(10));
    final tampered = jsonDecode(raw) as Map<String, dynamic>;
    (tampered['contacts'] as List).first['phone'] = '+919999999999';
    await store.writeMap('reviewed_contact_directory_v1', tampered);
    expect((await ContactDirectory.load(await Store.open(), bundle: bundle)).contacts.first.phone,
        '+919871963535');
  });

  test('saved veterinary contacts retain their separate type', () {
    expect(Contact.fromJson(Contact(name: 'Animal hospital', phone: '+914425381366',
        kind: ContactKind.veterinarian).toJson()).kind, ContactKind.veterinarian);
  });
}
