import 'package:material_ui/material_ui.dart';
import 'package:provider/provider.dart';
import 'package:url_launcher/url_launcher.dart';

import '../models/records.dart';
import '../services/contact_directory.dart';
import '../services/store.dart';
import '../state/app_state.dart';
import '../util/geo.dart';

class ContactDirectoryPanel extends StatefulWidget {
  const ContactDirectoryPanel({super.key});

  @override
  State<ContactDirectoryPanel> createState() => _ContactDirectoryPanelState();
}

class _ContactDirectoryPanelState extends State<ContactDirectoryPanel> {
  late final Future<ContactDirectory> _directory;
  final _query = TextEditingController();
  ContactKind? _kind;
  double? _radiusKm;
  String? _actionError;
  bool _locating = false;

  @override
  void initState() {
    super.initState();
    _directory = Store.open().then(ContactDirectory.load);
  }

  @override
  void dispose() {
    _query.dispose();
    super.dispose();
  }

  Future<void> _useLocation(AppState state) async {
    setState(() { _locating = true; _actionError = null; });
    try {
      await state.refreshLocation();
      if (mounted && state.position == null) {
        setState(() => _actionError = 'Location unavailable. Search by city or state instead.');
      }
    } catch (_) {
      if (mounted) setState(() => _actionError = 'Location unavailable. Search by city or state instead.');
    } finally {
      if (mounted) setState(() => _locating = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final state = context.watch<AppState>();
    return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      Text('Find rescuers and vets', style: Theme.of(context).textTheme.titleLarge),
      const SizedBox(height: 8),
      const Text('Offline pilot directory. Numbers were checked on official websites, '
          'not by phone. Call to confirm availability. Vets treat animals; for a person '
          'with a suspected snakebite use Emergency help.'),
      const SizedBox(height: 12),
      TextField(
        controller: _query,
        decoration: const InputDecoration(labelText: 'Name, city or state',
            prefixIcon: Icon(Icons.search), border: OutlineInputBorder()),
        onChanged: (_) => setState(() {}),
      ),
      Wrap(spacing: 8, children: [
        for (final entry in <ContactKind?, String>{null: 'All',
          ContactKind.rescuer: 'Rescuers', ContactKind.veterinarian: 'Vets'}.entries)
          ChoiceChip(label: Text(entry.value), selected: _kind == entry.key,
              onSelected: (_) => setState(() => _kind = entry.key)),
      ]),
      Wrap(spacing: 12, crossAxisAlignment: WrapCrossAlignment.center, children: [
        TextButton.icon(onPressed: _locating ? null : () => _useLocation(state),
            icon: const Icon(Icons.my_location),
            label: Text(_locating ? 'Finding location…' : 'Use my location')),
        if (state.position != null)
          DropdownButton<double>(
            hint: const Text('All distances'),
            value: _radiusKm,
            items: const [
              DropdownMenuItem(value: null, child: Text('All distances')),
              DropdownMenuItem(value: 25, child: Text('Mapped within 25 km')),
              DropdownMenuItem(value: 100, child: Text('Mapped within 100 km')),
            ],
            onChanged: (value) => setState(() => _radiusKm = value),
          ),
      ]),
      const Text('Distance uses mapped centres only and is straight-line, not travel distance. '
          'Regional helplines without coordinates remain available under All distances.'),
      if (_actionError != null) Text(_actionError!),
      FutureBuilder<ContactDirectory>(future: _directory, builder: (context, snapshot) {
        if (snapshot.hasError) return const Text('Directory unavailable. Your saved contacts are below.');
        if (!snapshot.hasData) {
          return const LinearProgressIndicator();
        }
        final matches = snapshot.data!.search(query: _query.text, kind: _kind,
            origin: state.position, radiusKm: state.position == null ? null : _radiusKm);
        if (matches.isEmpty) {
          return const Padding(padding: EdgeInsets.symmetric(vertical: 12),
              child: Text('No matches in this pilot directory. Try another locality or All distances.'));
        }
        return Column(children: [for (final match in matches) _card(state, match)]);
      }),
      const Divider(height: 32),
    ]);
  }

  Widget _card(AppState state, ContactMatch match) {
    final c = match.contact;
    final saved = state.contacts.any((item) =>
        item.phone.replaceAll(RegExp(r'\D'), '') == c.phone.replaceAll(RegExp(r'\D'), ''));
    return Card(child: Padding(padding: const EdgeInsets.all(12), child: Column(
      crossAxisAlignment: CrossAxisAlignment.start, children: [
        Text(c.name, style: const TextStyle(fontWeight: FontWeight.bold)),
        Text('${c.locality}, ${c.state} · ${c.kind.label}'),
        Text('${c.phone}${match.distanceM == null ? '' : ' · ${formatDistance(match.distanceM!)} straight-line'}'),
        Text('Website checked ${c.checkedAt.toIso8601String().substring(0, 10)}'
            '${c.needsRecheck(DateTime.now()) ? ' · recheck overdue' : ''}'),
        Text(c.notes),
        Wrap(spacing: 8, children: [
          TextButton.icon(onPressed: () async {
            try {
              if (!await state.sos.call(c.phone) && mounted) {
                setState(() => _actionError = 'Could not open dialler. Dial ${c.phone} manually.');
              }
            } catch (_) {
              if (mounted) setState(() => _actionError = 'Could not open dialler. Dial ${c.phone} manually.');
            }
          }, icon: const Icon(Icons.call), label: const Text('Call')),
          TextButton(onPressed: saved ? null : () async {
            try {
              await state.addContact(c.savedContact);
            } catch (_) {
              if (mounted) setState(() => _actionError = 'Could not save contact. The directory remains available.');
            }
          }, child: Text(saved ? 'Saved' : 'Save contact')),
          TextButton(onPressed: () async {
            try {
              if (!await launchUrl(c.source, mode: LaunchMode.externalApplication) && mounted) {
                setState(() => _actionError = 'Source needs a browser and internet: ${c.source}');
              }
            } catch (_) {
              if (mounted) setState(() => _actionError = 'Source needs a browser and internet: ${c.source}');
            }
          }, child: const Text('Official source')),
        ]),
      ],
    )));
  }
}
