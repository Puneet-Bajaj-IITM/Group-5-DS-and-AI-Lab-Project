import 'dart:io';
import 'dart:typed_data';

import 'package:material_ui/material_ui.dart';
import 'package:url_launcher/url_launcher.dart';

import '../ui/brand.dart';
import 'anidex_catalog.dart';
import 'anidex_collection.dart';
import 'anidex_model.dart';

const anidexGroupColours = {
  'Pet': Color(0xFFD96C25),
  'Farm': Color(0xFF8D6036),
  'Bird': Color(0xFF2F86C6),
  'Wild': Color(0xFF3F7A2B),
  'Reptile & amphibian': Color(0xFF1F8A7E),
  'Water': Color(0xFF2B55B0),
  'Bug': Color(0xFF7E8E14),
};

Color anidexColour(String group) => anidexGroupColours[group] ?? Brand.forest600;

const _iucn = {
  'LC': ('Least concern', Color(0xFFD4F0D4)),
  'NT': ('Near threatened', Color(0xFFE8F0B8)),
  'VU': ('Vulnerable', Color(0xFFFFE6A6)),
  'EN': ('Endangered', Color(0xFFFFC69A)),
  'CR': ('Critically endangered', Color(0xFFFF9D9D)),
  'EW': ('Extinct in the wild', Color(0xFFFF9D9D)),
  'DD': ('Data deficient', Color(0xFFE6E9EC)),
};

/// A AniDex card: either a fresh scan ([scan] and [photo] set) or a caught
/// entry opened from the grid ([entry] set).
class AniDexCardScreen extends StatelessWidget {
  const AniDexCardScreen({
    super.key,
    required this.species,
    this.scan,
    this.photo,
    this.kind,
    this.entry,
    required this.lowConfidence,
  });

  final AniDexSpecies species;
  final AniDexScan? scan;
  final Uint8List? photo;
  final CatchKind? kind;
  final CaughtEntry? entry;
  final double lowConfidence;

  @override
  Widget build(BuildContext context) {
    final text = Theme.of(context).textTheme;
    final scheme = Theme.of(context).colorScheme;
    final colour = anidexColour(species.group);
    final breed = scan != null
        ? scan!.breed?.$1
        : (entry?.lastBreed == null ? null : species.breed(entry!.lastBreed!));
    final wiki = breed != null && breed.wiki.extract.isNotEmpty ? breed.wiki : species.wiki;
    final title = breed?.name ?? species.name;
    final badge = switch (kind) {
      CatchKind.newSpecies => 'NEW!',
      CatchKind.newBreed => 'NEW BREED!',
      CatchKind.seenAgain => 'SEEN ${entry?.count ?? ''}×',
      null when scan != null => 'NOT CAUGHT',
      null => entry == null ? null : 'CAUGHT ${entry!.count}×',
    };

    Widget photoWidget;
    if (photo != null) {
      photoWidget = Image.memory(photo!, fit: BoxFit.cover);
    } else if (entry?.photo != null && File(entry!.photo!).existsSync()) {
      photoWidget = Image.file(File(entry!.photo!), fit: BoxFit.cover);
    } else {
      photoWidget = ColoredBox(color: colour.withValues(alpha: 0.15), child: Icon(Icons.pets, size: 64, color: colour));
    }

    return Scaffold(
      appBar: AppBar(
        backgroundColor: colour,
        foregroundColor: Colors.white,
        title: Text('#${species.no.toString().padLeft(3, '0')}  ${species.name.toUpperCase()}',
            style: const TextStyle(
                fontFamily: Brand.display, fontWeight: FontWeight.w800, letterSpacing: 1.2, color: Colors.white)),
      ),
      body: ListView(
        padding: const EdgeInsets.fromLTRB(16, 16, 16, 32),
        children: [
          Stack(children: [
            ClipRRect(
              borderRadius: BorderRadius.circular(20),
              child: AspectRatio(aspectRatio: 4 / 3, child: photoWidget),
            ),
            if (badge != null)
              Positioned(
                top: 12,
                right: 12,
                child: Container(
                  padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
                  decoration: BoxDecoration(
                    color: kind == CatchKind.seenAgain || kind == null ? Colors.black54 : Brand.venom,
                    borderRadius: BorderRadius.circular(10),
                  ),
                  child: Text(badge,
                      style: Brand.label(kind == CatchKind.seenAgain || kind == null ? Colors.white : Brand.ink)),
                ),
              ),
          ]),
          const SizedBox(height: 16),
          Text(title, style: text.headlineMedium),
          const SizedBox(height: 2),
          Text.rich(TextSpan(children: [
            if (breed != null) TextSpan(text: '${species.name} · '),
            TextSpan(text: species.scientific, style: const TextStyle(fontStyle: FontStyle.italic)),
          ]), style: text.bodyLarge?.copyWith(color: scheme.onSurfaceVariant)),
          const SizedBox(height: 10),
          Wrap(spacing: 8, runSpacing: 8, children: [
            _Chip(label: species.group.toUpperCase(), background: colour, foreground: Colors.white),
            if (_iucn[species.iucn] != null)
              _Chip(label: '${species.iucn} · ${_iucn[species.iucn]!.$1}', background: _iucn[species.iucn]!.$2, foreground: Brand.ink),
          ]),
          if (scan != null) ...[
            const SizedBox(height: 14),
            ClipRRect(
              borderRadius: BorderRadius.circular(8),
              child: LinearProgressIndicator(
                value: scan!.score, minHeight: 10, color: colour, backgroundColor: colour.withValues(alpha: 0.15)),
            ),
            const SizedBox(height: 4),
            Text('Match confidence ${(scan!.score * 100).round()}%', style: text.bodySmall),
            if (scan!.score < lowConfidence)
              _Note(
                icon: Icons.help_outline_rounded,
                text: 'Low confidence: this may not be an animal, or not one of the animals in the AniDex. '
                    'This is the closest match, and it was not added to your AniDex.',
                colour: Brand.caution,
                soft: Brand.cautionSoft,
              ),
          ],
          if (species.isSnake)
            const _Note(
              icon: Icons.warning_amber_rounded,
              text: 'This is a snake. The AniDex is not a safety check: use Identify for SerpentGuard\'s venom '
                  'check, and stay at least 2 metres away.',
              colour: Brand.danger,
              soft: Brand.dangerSoft,
            ),
          if (species.hasBreeds) _breedSection(context, colour),
          _Section(title: 'Taxonomy', colour: colour, child: Text.rich(TextSpan(children: [
            TextSpan(text: [for (final r in ['phylum', 'class', 'order', 'family', 'genus']) ?species.taxonomy[r]].join(' › ')),
            TextSpan(text: ' › ${species.scientific}', style: const TextStyle(fontStyle: FontStyle.italic)),
          ]))),
          _Section(
            title: 'Entry',
            colour: colour,
            child: Text(wiki.extract.isEmpty ? 'No description saved for this animal.' : wiki.extract),
          ),
          if (!species.hasBreeds && species.subspecies.isNotEmpty)
            _Section(
              title: 'Known subspecies',
              colour: colour,
              child: Wrap(spacing: 6, runSpacing: 6, children: [
                for (final s in species.subspecies.take(12))
                  _Chip(label: s, background: scheme.surfaceContainerHigh, foreground: scheme.onSurface, italic: true),
                if (species.subspecies.length > 12) Text('and ${species.subspecies.length - 12} more'),
              ]),
            ),
          if (scan != null && scan!.alternatives.isNotEmpty)
            _Section(
              title: 'Other guesses',
              colour: colour,
              child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                for (final (s, p) in scan!.alternatives)
                  Padding(
                    padding: const EdgeInsets.only(bottom: 2),
                    child: Text('${s.name} (${s.scientific})  ${(p * 100).round()}%'),
                  ),
              ]),
            ),
          if (wiki.image.isNotEmpty)
            _Section(
              title: 'Reference photo',
              colour: colour,
              child: ClipRRect(
                borderRadius: BorderRadius.circular(14),
                child: Image.network(
                  wiki.image,
                  height: 180,
                  width: double.infinity,
                  fit: BoxFit.cover,
                  errorBuilder: (_, _, _) => Text('Shown when the phone is online.', style: text.bodySmall),
                ),
              ),
            ),
          const SizedBox(height: 8),
          Wrap(spacing: 8, children: [
            if (wiki.url.isNotEmpty)
              OutlinedButton.icon(
                onPressed: () => launchUrl(Uri.parse(wiki.url), mode: LaunchMode.externalApplication),
                icon: const Icon(Icons.menu_book_rounded),
                label: const Text('Wikipedia'),
              ),
            if (species.gbifKey != null)
              OutlinedButton.icon(
                onPressed: () => launchUrl(Uri.parse('https://www.gbif.org/species/${species.gbifKey}'),
                    mode: LaunchMode.externalApplication),
                icon: const Icon(Icons.public_rounded),
                label: const Text('GBIF'),
              ),
          ]),
          const SizedBox(height: 8),
          Text('Descriptions from Wikipedia (CC BY-SA); taxonomy and status from GBIF.',
              style: text.bodySmall?.copyWith(color: scheme.onSurfaceVariant)),
        ],
      ),
    );
  }

  Widget _breedSection(BuildContext context, Color colour) {
    final scheme = Theme.of(context).colorScheme;
    final noun = species.breedNoun ?? 'animal';
    if (scan != null) {
      final lead = scan!.breed == null
          ? 'Mixed or unknown breed'
          : '${scan!.breed!.$1.name} (${(scan!.breed!.$2 * 100).round()}% among breeds)';
      return _Section(
        title: 'Breed',
        colour: colour,
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Text(lead, style: const TextStyle(fontWeight: FontWeight.w700)),
          const SizedBox(height: 6),
          Wrap(spacing: 6, runSpacing: 6, children: [
            for (final (b, p) in scan!.breeds)
              _Chip(label: '${b.name} ${(p * 100).round()}%', background: scheme.surfaceContainerHigh, foreground: scheme.onSurface),
          ]),
          const SizedBox(height: 6),
          Text('Checked against ${species.breeds.length} $noun breeds.',
              style: Theme.of(context).textTheme.bodySmall),
        ]),
      );
    }
    final found = entry?.breeds ?? const {};
    return _Section(
      title: 'Breeds found (${found.length} of ${species.breeds.length})',
      colour: colour,
      child: found.isEmpty
          ? const Text('None yet: breeds show up when the model is sure of one.')
          : Wrap(spacing: 6, runSpacing: 6, children: [
              for (final e in found.entries)
                _Chip(label: '${e.key} ×${e.value}', background: scheme.surfaceContainerHigh, foreground: scheme.onSurface),
            ]),
    );
  }
}

class _Section extends StatelessWidget {
  const _Section({required this.title, required this.colour, required this.child});
  final String title;
  final Color colour;
  final Widget child;

  @override
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.only(top: 18),
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Text(title.toUpperCase(), style: Brand.label(colour)),
          const SizedBox(height: 6),
          DefaultTextStyle.merge(style: Theme.of(context).textTheme.bodyMedium, child: child),
        ]),
      );
}

class _Chip extends StatelessWidget {
  const _Chip({required this.label, required this.background, required this.foreground, this.italic = false});
  final String label;
  final Color background, foreground;
  final bool italic;

  @override
  Widget build(BuildContext context) => Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
        decoration: BoxDecoration(color: background, borderRadius: BorderRadius.circular(999)),
        child: Text(label,
            style: TextStyle(
                color: foreground,
                fontWeight: FontWeight.w600,
                fontSize: 13,
                fontStyle: italic ? FontStyle.italic : FontStyle.normal)),
      );
}

class _Note extends StatelessWidget {
  const _Note({required this.icon, required this.text, required this.colour, required this.soft});
  final IconData icon;
  final String text;
  final Color colour, soft;

  @override
  Widget build(BuildContext context) => Container(
        margin: const EdgeInsets.only(top: 12),
        padding: const EdgeInsets.all(12),
        decoration: BoxDecoration(
          color: soft,
          borderRadius: BorderRadius.circular(12),
          border: Border(left: BorderSide(color: colour, width: 4)),
        ),
        child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Icon(icon, color: colour),
          const SizedBox(width: 10),
          Expanded(child: Text(text, style: const TextStyle(color: Brand.ink))),
        ]),
      );
}
