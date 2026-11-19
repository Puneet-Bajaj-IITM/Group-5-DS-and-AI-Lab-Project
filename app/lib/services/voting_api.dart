import 'dart:convert';
import 'dart:typed_data';

import 'package:crypto/crypto.dart';
import 'package:http/http.dart' as http;

class VotingApiError implements Exception {
  VotingApiError(this.status);
  final int status;
  bool get retryable => status == 408 || status == 429 || status >= 500;
  @override
  String toString() => switch (status) {
    401 => 'Session rejected. Check your reviewer token.',
    403 => 'This account cannot perform that action (including self-voting).',
    409 => 'The server has a conflicting version. Review before replacing.',
    _ => 'Community server returned HTTP $status.',
  };
}

/// Tokens are held in memory only, never in the SQLite outbox or preferences.
class VotingApi {
  VotingApi(String server, String token, {http.Client? client})
    : base = validateServer(server),
      _token = token.trim(),
      _client = client ?? http.Client() {
    if (_token.isEmpty || _token.contains(RegExp(r'\s'))) {
      throw const FormatException('Enter the provisioned reviewer token.');
    }
  }

  final Uri base;
  final String _token;
  final http.Client _client;

  // Bind queued jobs to the exact server and credential. Switching accounts
  // cannot upload another account's photos or votes. Token rotation needs a
  // deliberate migration, rather than silently changing attribution.
  String get account =>
      sha256.convert(utf8.encode('$base\n$_token')).toString();

  static Uri validateServer(String value) {
    final uri = Uri.tryParse(value.trim());
    if (uri == null ||
        !uri.hasAuthority ||
        uri.userInfo.isNotEmpty ||
        uri.hasQuery ||
        uri.hasFragment ||
        (uri.path != '' && uri.path != '/')) {
      throw const FormatException(
        'Use a server origin, for example https://review.example.org.',
      );
    }
    final local = [
      'localhost',
      '127.0.0.1',
      '10.0.2.2',
      '::1',
    ].contains(uri.host);
    if (uri.scheme != 'https' && !(uri.scheme == 'http' && local)) {
      throw const FormatException(
        'HTTPS is required; HTTP is allowed only for local development.',
      );
    }
    return uri.replace(path: '/');
  }

  Future<http.Response> _request(
    String method,
    String path, {
    Map<String, dynamic>? json,
    Uint8List? jpeg,
  }) async {
    // Never send a bearer token to an absolute URL supplied by a feed item.
    if (!path.startsWith('/') ||
        path.startsWith('//') ||
        path.contains('://')) {
      throw const FormatException('Invalid community API path.');
    }
    final req = http.Request(method, base.resolve(path))
      ..followRedirects = false
      ..headers['Authorization'] = 'Bearer $_token';
    if (json != null) {
      req.headers['Content-Type'] = 'application/json';
      req.body = jsonEncode(json);
    }
    if (jpeg != null) {
      req.headers['Content-Type'] = 'image/jpeg';
      req.bodyBytes = jpeg;
    }
    final response = await (() async => http.Response.fromStream(
      await _client.send(req),
    ))().timeout(const Duration(seconds: 30));
    if (response.statusCode < 200 || response.statusCode >= 300) {
      throw VotingApiError(response.statusCode);
    }
    return response;
  }

  Future<Map<String, dynamic>> json(
    String method,
    String path, [
    Map<String, dynamic>? data,
  ]) async =>
      jsonDecode((await _request(method, path, json: data)).body)
          as Map<String, dynamic>;

  Future<Map<String, dynamic>> upload(
    String id,
    int slot,
    Uint8List jpeg,
  ) async => jsonDecode(
    (await _request('PUT', '/sightings/$id/evidence/$slot', jpeg: jpeg)).body,
  ) as Map<String, dynamic>;

  Future<Uint8List> evidence(String id, int slot) async =>
      (await _request('GET', '/sightings/$id/evidence/$slot')).bodyBytes;

  void close() => _client.close();
}
