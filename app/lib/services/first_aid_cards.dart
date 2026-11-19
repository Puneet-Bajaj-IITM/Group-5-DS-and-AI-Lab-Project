import 'dart:convert';
import 'dart:io';

import 'package:flutter/services.dart';

const firstAidIds = [
  'who-110-01',
  'who-119-04',
  'who-118-04',
  'who-119-03',
  'who-120-02',
  'who-120-05',
];

class FirstAidCards {
  FirstAidCards.parse(Map<String, dynamic> pack) {
    if (pack['schema_version'] != 1 ||
        pack.length != 2 ||
        pack['cards'] is! Map) {
      throw const FormatException('Invalid first-aid card pack.');
    }
    final cards = (pack['cards'] as Map).cast<String, dynamic>();
    if (cards.length != 4) {
      throw const FormatException('Four reviewed languages are required.');
    }
    for (final language in ['en', 'hi', 'ta', 'te']) {
      final set = cards[language];
      if (set is! Map || set.length != firstAidIds.length) {
        throw const FormatException('Incomplete first-aid language.');
      }
      for (final id in firstAidIds) {
        final item = set[id];
        if (item is! Map ||
            item.length != 3 ||
            ['text', 'source_url', 'review_id'].any(
              (key) =>
                  item[key] is! String || (item[key] as String).trim().isEmpty,
            )) {
          throw const FormatException(
            'A card needs reviewed text and source provenance.',
          );
        }
        final url = Uri.tryParse(item['source_url'] as String);
        if (url == null ||
            url.scheme != 'https' ||
            url.host.isEmpty ||
            (item['text'] as String).length > 4000) {
          throw const FormatException('Invalid source or speech card length.');
        }
      }
    }
    _cards = cards;
  }
  late final Map<String, dynamic> _cards;
  ({String text, String language}) resolve(String id, String locale) {
    final language = locale.replaceAll('_', '-').split('-').first.toLowerCase();
    final selected = _cards.containsKey(language) ? language : 'en';
    return (text: _cards[selected][id]['text'] as String, language: selected);
  }

  static Future<FirstAidCards?> load() async {
    final manifest = await AssetManifest.loadFromAssetBundle(rootBundle);
    if (!manifest.listAssets().contains('assets/first_aid.json')) return null;
    return FirstAidCards.parse(
      jsonDecode(await rootBundle.loadString('assets/first_aid.json'))
          as Map<String, dynamic>,
    );
  }
}

class OfflineSpeech {
  static const _channel = MethodChannel('serpentguard/offline_speech');
  Future<List<String>> voices() async {
    if (!Platform.isAndroid) return [];
    return (await _channel.invokeListMethod<String>('voices')) ?? [];
  }

  Future<void> speak(({String text, String language}) card) async {
    await _channel.invokeMethod<void>('speak', {
      'text': card.text,
      'language': card.language,
    });
  }

  Future<void> stop() =>
      Platform.isAndroid ? _channel.invokeMethod<void>('stop') : Future.value();
}
