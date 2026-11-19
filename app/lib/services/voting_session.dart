import 'dart:convert';

import 'package:flutter_secure_storage/flutter_secure_storage.dart';

import 'voting_api.dart';

class VotingSession {
  VotingSession(this.server, this.token) {
    final api = createApi();
    account = api.account;
    api.close();
  }
  final String server;
  final String token;
  late final String account;
  VotingApi createApi() => VotingApi(server, token);
}

/// Device-protected credential storage. Tokens never enter SQLite or work input.
class VotingSessionStore {
  VotingSessionStore({FlutterSecureStorage? storage})
    : _storage = storage ?? const FlutterSecureStorage();
  final FlutterSecureStorage _storage;
  static const _key = 'community.session.v1';

  Future<VotingSession?> load() async {
    final raw = await _storage.read(key: _key);
    if (raw == null) return null;
    final value = jsonDecode(raw) as Map<String, dynamic>;
    return VotingSession(value['server'] as String, value['token'] as String);
  }

  Future<void> save(VotingSession session) => _storage.write(
    key: _key,
    value: jsonEncode({'server': session.server, 'token': session.token}),
  );

  Future<void> clear() => _storage.delete(key: _key);
}
