import 'dart:io';

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../app.dart';
import '../models/records.dart';
import '../models/species.dart';
import '../models/verdict.dart';
import '../services/classifier.dart';
import '../state/app_state.dart';
import '../widgets/common.dart';

class ResultScreen extends StatefulWidget {
  const ResultScreen({super.key, required this.identification, this.photoPath});
  final Identification identification;
  final String? photoPath;

  @override
  State<ResultScreen> createState() => _ResultScreenState();
}

class _ResultScreenState extends State<ResultScreen> {
  late Identification _id = widget.identification;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      context.read<AppState>().saveSighting(_id, photoPath: widget.photoPath);
    });
  }

  Future<void> _sendAlert() async {
    final state = context.read<AppState>();
    final messenger = ScaffoldMessenger.of(context);
    final ok = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        icon: const Icon(Icons.campaign_rounded),
        title: const Text('Warn people nearby?'),
        content: Text(
          'Phones with SerpentGuard within about 100 m will see '
          '"${_id.topSpecies?.displayName ?? 'unidentified snake'} (${_id.verdict.title})" '
          'and its location, rounded to about 100 m, for the next hour.\n\n'
          'Your name and number are not shared.',
        ),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx, false), child: const Text('Cancel')),
          FilledButton(onPressed: () => Navigator.pop(ctx, true), child: const Text('Send alert')),
        ],
      ),
    );
    if (ok != true) return;
    final msg = await state.sendAlert(_id);
    messenger.showSnackBar(SnackBar(content: Text(msg)));
  }

  Future<void> _pickDemoSpecies() async {
    final state = context.read<AppState>();
    final picked = await showModalBottomSheet<Species>(
      context: context,
      isScrollControlled: true,
      showDragHandle: true,
      builder: (_) => _SpeciesPicker(catalog: state.catalog),
    );
    if (picked == null || state.classifier is! DemoClassifier) return;
    setState(() => _id = (state.classifier as DemoClassifier).forSpecies(picked.id));
  }

  @override
  Widget build(BuildContext context) {
    final s = context.watch<AppState>();
    final text = Theme.of(context).textTheme;
    final scheme = Theme.of(context).colorScheme;
    final rescuers = s.contacts
        .where((c) => c.kind == ContactKind.rescuer || c.kind == ContactKind.forest)
        .toList();

    return Scaffold(
      appBar: AppBar(title: const Text('Result')),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.fromLTRB(16, 0, 16, 32),
          children: [
            if (widget.photoPath != null && File(widget.photoPath!).existsSync())
              ClipRRect(
                borderRadius: BorderRadius.circular(20),
                child: AspectRatio(
                  aspectRatio: 4 / 3,
                  child: Image.file(File(widget.photoPath!), fit: BoxFit.cover, cacheWidth: 1200),
                ),
              ),
            const SizedBox(height: 14),
            if (_id.isDemo) ...[
              const DemoBadge(),
              const SizedBox(height: 12),
            ],
            VerdictBanner(verdict: _id.verdict, reason: _id.reason),
            const SizedBox(height: 16),
            if (_id.verdict.escalate) ...[
              EmergencyButton(
                label: 'Someone was bitten: get help',
                onPressed: () => openSos(context, bitten: true, identification: _id),
              ),
              const SizedBox(height: 10),
            ],
            if (_id.verdict == Verdict.notListed && rescuers.isNotEmpty)
              ...rescuers.map((c) => Padding(
                    padding: const EdgeInsets.only(bottom: 10),
                    child: SizedBox(
                      width: double.infinity,
                      height: 52,
                      child: FilledButton.tonalIcon(
                        onPressed: () => s.sos.call(c.phone),
                        icon: const Icon(Icons.call_rounded),
                        label: Text('Call ${c.name} (${c.kind.label.toLowerCase()})'),
                      ),
                    ),
                  )),
            if (_id.verdict == Verdict.notListed && rescuers.isEmpty)
              Padding(
                padding: const EdgeInsets.only(bottom: 10),
                child: Text(
                  'Tip: add a local snake rescuer or forest office in Contacts so you can '
                  'call them from here to move the snake safely.',
                  style: text.bodyMedium?.copyWith(color: scheme.onSurfaceVariant),
                ),
              ),
            SizedBox(
              width: double.infinity,
              height: 52,
              child: OutlinedButton.icon(
                onPressed: _sendAlert,
                icon: const Icon(Icons.campaign_outlined),
                label: const Text('Warn people nearby'),
              ),
            ),
            SectionTitle('Most likely species',
                trailing: _id.isDemo
                    ? TextButton.icon(
                        onPressed: _pickDemoSpecies,
                        icon: const Icon(Icons.swap_horiz_rounded, size: 18),
                        label: const Text('Try another (demo)'),
                      )
                    : null),
            ..._id.top.map((t) => _SpeciesRow(score: t)),
            const SectionTitle('First aid'),
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                  Text('If someone is bitten, go to the nearest hospital with anti-snake '
                      'venom (ASV) straight away and call 108 or 112.',
                      style: text.bodyLarge?.copyWith(fontWeight: FontWeight.w600)),
                  const SizedBox(height: 8),
                  Text(
                    'The WHO first-aid passages will appear here once the team has '
                    'permission to include the guideline text.',
                    style: text.bodySmall?.copyWith(color: scheme.onSurfaceVariant),
                  ),
                ]),
              ),
            ),
            const SizedBox(height: 16),
            Text(
              'SerpentGuard is a student research project. It can be wrong, it does not '
              'diagnose bites and "not listed" never means a snake is safe.',
              style: text.bodySmall?.copyWith(color: scheme.onSurfaceVariant),
            ),
          ],
        ),
      ),
    );
  }
}

class _SpeciesRow extends StatelessWidget {
  const _SpeciesRow({required this.score});
  final SpeciesScore score;

  @override
  Widget build(BuildContext context) {
    final sp = score.species;
    final scheme = Theme.of(context).colorScheme;
    final pct = (score.probability * 100).round();
    return Card(
      margin: const EdgeInsets.only(bottom: 8),
      child: Padding(
        padding: const EdgeInsets.fromLTRB(16, 12, 16, 14),
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Row(children: [
            Expanded(
              child: Text(sp.displayName,
                  style: const TextStyle(fontWeight: FontWeight.w700, fontSize: 16)),
            ),
            Text('$pct%', style: const TextStyle(fontWeight: FontWeight.w700)),
          ]),
          if (sp.commonName != null)
            Text(sp.name,
                style: TextStyle(fontStyle: FontStyle.italic, color: scheme.onSurfaceVariant)),
          const SizedBox(height: 8),
          ClipRRect(
            borderRadius: BorderRadius.circular(4),
            child: LinearProgressIndicator(
              value: score.probability.clamp(0, 1),
              minHeight: 6,
              color: sp.dangerous ? const Color(0xFFB3261E) : scheme.primary,
              backgroundColor: scheme.surfaceContainerHighest,
            ),
          ),
          const SizedBox(height: 8),
          Wrap(spacing: 6, runSpacing: 6, children: [
            _Tag(sp.who.label, danger: sp.who != WhoCategory.notListed),
            if (sp.dangerous && sp.who == WhoCategory.notListed)
              const _Tag('Venomous family', danger: true),
            _Tag(sp.family),
          ]),
        ]),
      ),
    );
  }
}

class _Tag extends StatelessWidget {
  const _Tag(this.text, {this.danger = false});
  final String text;
  final bool danger;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
      decoration: BoxDecoration(
        color: danger ? scheme.errorContainer : scheme.surfaceContainerHighest,
        borderRadius: BorderRadius.circular(6),
      ),
      child: Text(text,
          style: TextStyle(
              fontSize: 12,
              fontWeight: FontWeight.w600,
              color: danger ? scheme.onErrorContainer : scheme.onSurfaceVariant)),
    );
  }
}

class _SpeciesPicker extends StatefulWidget {
  const _SpeciesPicker({required this.catalog});
  final SpeciesCatalog catalog;

  @override
  State<_SpeciesPicker> createState() => _SpeciesPickerState();
}

class _SpeciesPickerState extends State<_SpeciesPicker> {
  String _q = '';

  @override
  Widget build(BuildContext context) {
    final q = _q.toLowerCase();
    final list = widget.catalog.all
        .where((s) =>
            q.isEmpty ||
            s.name.toLowerCase().contains(q) ||
            (s.commonName?.toLowerCase().contains(q) ?? false))
        .toList();
    return SizedBox(
      height: MediaQuery.sizeOf(context).height * 0.75,
      child: Column(children: [
        Padding(
          padding: const EdgeInsets.fromLTRB(16, 0, 16, 8),
          child: TextField(
            autofocus: true,
            decoration: const InputDecoration(
              prefixIcon: Icon(Icons.search),
              hintText: 'Search 184 species',
              border: OutlineInputBorder(),
            ),
            onChanged: (v) => setState(() => _q = v),
          ),
        ),
        Expanded(
          child: ListView.builder(
            itemCount: list.length,
            itemBuilder: (_, i) {
              final s = list[i];
              return ListTile(
                title: Text(s.displayName),
                subtitle: Text(s.commonName == null ? s.family : s.name),
                trailing: s.dangerous
                    ? const Icon(Icons.warning_amber_rounded, color: Color(0xFFB3261E))
                    : null,
                onTap: () => Navigator.pop(context, s),
              );
            },
          ),
        ),
      ]),
    );
  }
}
