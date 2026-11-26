import 'dart:convert';

import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:serpentguard/services/address_service.dart';
import 'package:serpentguard/services/sos_service.dart';
import 'package:serpentguard/services/store.dart';
import 'package:serpentguard/util/geo.dart';
import 'package:shared_preferences/shared_preferences.dart';

void main() {
  const here = GeoPoint(12.97161, 77.59463);

  test('uses the phone geocoder first and caches per ~100 m', () async {
    SharedPreferences.setMockInitialValues({});
    var calls = 0;
    final s = AddressService(
      await Store.open(),
      client: MockClient((_) async => throw StateError('should not reach the network')),
      platform: (_) async {
        calls++;
        return '12, MG Road, Bengaluru 560001';
      },
    );
    expect(await s.lookup(here), '12, MG Road, Bengaluru 560001');
    expect(await s.lookup(const GeoPoint(12.97170, 77.59471)), '12, MG Road, Bengaluru 560001');
    expect(calls, 1, reason: 'the second spot is in the same ~100 m square');
    expect(s.cached(here), isNotNull);
  });

  test('falls back to OpenStreetMap when the phone has no geocoder', () async {
    SharedPreferences.setMockInitialValues({});
    final s = AddressService(
      await Store.open(),
      platform: (_) async => null,
      client: MockClient((req) async {
        expect(req.url.path, '/reverse');
        return http.Response(
          jsonEncode({
            'display_name': 'long name',
            'address': {
              'house_number': '4',
              'road': 'Kasturba Road',
              'suburb': 'Shanthala Nagar',
              'city': 'Bengaluru',
              'postcode': '560001',
            },
          }),
          200,
        );
      }),
    );
    expect(await s.lookup(here), '4 Kasturba Road, Shanthala Nagar, Bengaluru, 560001');
  });

  test('gives null, not an error, with no network and nothing cached', () async {
    SharedPreferences.setMockInitialValues({});
    final s = AddressService(
      await Store.open(),
      platform: (_) async => throw StateError('no service'),
      client: MockClient((_) async => throw http.ClientException('offline')),
    );
    expect(await s.lookup(here), isNull);
  });

  test('formatParts drops empty and repeated parts', () {
    expect(AddressService.formatParts(['MG Road', '', null, 'Bengaluru', 'bengaluru']), 'MG Road, Bengaluru');
    expect(AddressService.formatParts([null, ' ']), isNull);
  });

  test('the SOS text puts the address before the maps link', () {
    final msg = SosService.message(bitten: true, location: here, address: '4 Kasturba Road, Bengaluru');
    expect(msg, contains('I am at: 4 Kasturba Road, Bengaluru, https://maps.google.com/?q=12.97161,77.59463'));
  });
}
