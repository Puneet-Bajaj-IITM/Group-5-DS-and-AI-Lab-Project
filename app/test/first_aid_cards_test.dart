import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:serpentguard/services/first_aid_cards.dart';

Map<String, dynamic> sample() => {
  'schema_version': 1,
  'cards': {
    for (final lang in ['en', 'hi', 'ta', 'te'])
      lang: {
        for (final id in firstAidIds)
          id: {
            'text': 'Test fixture $lang',
            'source_url': 'https://example.org/review-fixture',
            'review_id': 'test-only',
          },
      },
  },
};

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  test('card fallback preserves the actual text language for speech', () async {
    final cards = FirstAidCards.parse(sample());
    expect(cards.resolve(firstAidIds.first, 'hi-IN').language, 'hi');
    expect(cards.resolve(firstAidIds.first, 'fr-FR').language, 'en');
    MethodCall? spoken;
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(
          const MethodChannel('serpentguard/offline_speech'),
          (call) async {
            spoken = call;
            return null;
          },
        );
    await OfflineSpeech().speak(cards.resolve(firstAidIds.first, 'fr-FR'));
    expect(spoken?.arguments['language'], 'en');
    expect(spoken?.arguments['text'], 'Test fixture en');
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(
          const MethodChannel('serpentguard/offline_speech'),
          null,
        );
  });
  test('incomplete medical provenance and HTTP sources are rejected', () {
    final pack = sample();
    pack['cards']['hi'][firstAidIds.first]['review_id'] = '';
    expect(() => FirstAidCards.parse(pack), throwsFormatException);
    final other = sample();
    other['cards']['te'][firstAidIds.first]['source_url'] =
        'http://example.org';
    expect(() => FirstAidCards.parse(other), throwsFormatException);
  });
}
