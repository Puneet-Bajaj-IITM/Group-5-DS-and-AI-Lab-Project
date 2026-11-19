import 'dart:async';

import 'package:material_ui/material_ui.dart';

import '../services/first_aid_cards.dart';

class FirstAidScreen extends StatefulWidget {
  const FirstAidScreen({super.key});
  @override
  State<FirstAidScreen> createState() => _FirstAidScreenState();
}

class _FirstAidScreenState extends State<FirstAidScreen> {
  FirstAidCards? _cards;
  final _speech = OfflineSpeech();
  String _language = 'en';
  List<String> _voices = [];
  String _status = 'Loading saved cards…';
  @override
  void initState() {
    super.initState();
    unawaited(_load());
  }

  Future<void> _load() async {
    try {
      final cards = await FirstAidCards.load();
      List<String> voices = [];
      try {
        voices = await _speech.voices();
      } catch (_) {
        /* Text still works. */
      }
      if (mounted) {
        setState(() {
          _cards = cards;
          _voices = voices;
          _status = cards == null
              ? 'Reviewed regional first-aid cards are unavailable in this build. Use Emergency help to call for assistance.'
              : 'These saved cards work offline. Audio requires an installed offline voice.';
        });
      }
    } catch (_) {
      if (mounted) {
        setState(
          () => _status =
              'The saved card pack could not be verified. Use Emergency help.',
        );
      }
    }
  }

  @override
  Widget build(BuildContext context) => Scaffold(
    appBar: AppBar(title: const Text('First-aid cards')),
    body: ListView(
      padding: const EdgeInsets.all(16),
      children: [
        Text(_status),
        DropdownButton<String>(
          value: _language,
          isExpanded: true,
          items: const [
            DropdownMenuItem(value: 'en', child: Text('English')),
            DropdownMenuItem(value: 'hi', child: Text('हिन्दी')),
            DropdownMenuItem(value: 'ta', child: Text('தமிழ்')),
            DropdownMenuItem(value: 'te', child: Text('తెలుగు')),
          ],
          onChanged: (v) {
            unawaited(_speech.stop());
            setState(() => _language = v!);
          },
        ),
        if (_cards != null)
          for (final id in firstAidIds)
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(_cards!.resolve(id, _language).text),
                    TextButton.icon(
                      icon: const Icon(Icons.volume_up),
                      label: const Text('Read aloud offline'),
                      onPressed:
                          _voices.any((v) => v.split('-').first == _language)
                          ? () async {
                              try {
                                await _speech.speak(
                                  _cards!.resolve(id, _language),
                                );
                              } catch (_) {
                                if (mounted) {
                                  setState(
                                    () => _status = 'Offline speech unavailable. The saved text remains readable.',
                                  );
                                }
                              }
                            }
                          : null,
                    ),
                  ],
                ),
              ),
            ),
        if (!_voices.any((v) => v.split('-').first == _language))
          const Text(
            'No installed offline voice for this language. Text remains available when a reviewed pack is bundled.',
          ),
      ],
    ),
  );
  @override
  void dispose() {
    unawaited(_speech.stop().catchError((_) {}));
    super.dispose();
  }
}
