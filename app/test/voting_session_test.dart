import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:serpentguard/services/voting_session.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  test(
    'secure account restores queue identity and disconnect removes credentials',
    () async {
      FlutterSecureStorage.setMockInitialValues({});
      final session = VotingSession('https://review.example.org', 'test-token');
      await VotingSessionStore().save(session);
      final restored = await VotingSessionStore().load();
      expect(restored?.account, session.account);
      final api = restored!.createApi();
      expect(api.account, session.account);
      api.close();
      await VotingSessionStore().clear();
      expect(await VotingSessionStore().load(), isNull);
    },
  );
}
