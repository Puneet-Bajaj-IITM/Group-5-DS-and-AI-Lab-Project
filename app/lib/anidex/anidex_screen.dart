import 'dart:io';

import 'package:image_picker/image_picker.dart';
import 'package:material_ui/material_ui.dart';

import '../services/store.dart';
import '../ui/brand.dart';
import 'anidex_card_screen.dart';
import 'anidex_catalog.dart';
import 'anidex_collection.dart';
import 'anidex_model.dart';

/// The AniDex tab: scan any common animal (pets, farm animals, birds and
/// more), see its card, and fill the grid. Runs BioCLIP 2 on the phone,
/// separately from the snake pipeline.
class AniDexScreen extends StatefulWidget {
  const AniDexScreen({super.key});

  @override
  State<AniDexScreen> createState() => _AniDexScreenState();
}

class _AniDexScreenState extends State<AniDexScreen> {
  AniDexCatalog? _catalog;
  AniDexCollection? _collection;
  Future<AniDexModel>? _model;
  bool? _modelAvailable;
  String? _loadError;
  String? _status;
  String _group = 'All';

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    try {
      final (catalog, store) = await (AniDexCatalog.load(), Store.open()).wait;
      final available = await AniDexModel.available(catalog);
      if (!mounted) return;
      setState(() {
        _catalog = catalog;
        _collection = AniDexCollection(store)..addListener(_redraw);
        _modelAvailable = available || (catalog.encoder.url != null && catalog.breedEncoder.url != null);
      });
    } catch (e) {
      if (mounted) setState(() => _loadError = '$e');
    }
  }

  void _redraw() {
    if (mounted) setState(() {});
  }

  @override
  void dispose() {
    _collection?.removeListener(_redraw);
    _model?.then((m) => m.close()).ignore();
    super.dispose();
  }

  /// Opens the model on first use (it is large), downloading it if needed.
  Future<AniDexModel> _openModel() => _model ??= AniDexModel.open(_catalog!, onDownload: (p) {
        if (mounted) setState(() => _status = 'Downloading the AniDex model… ${(p * 100).round()}%');
      }).catchError((Object e) {
        _model = null; // try again on the next scan
        throw e;
      });

  Future<void> _pick(ImageSource source) async {
    if (_status != null || _catalog == null) return;
    XFile? file;
    try {
      file = await ImagePicker().pickImage(source: source, maxWidth: 1600, imageQuality: 92);
    } catch (e) {
      _snack('Could not open the camera or gallery: $e');
      return;
    }
    if (file == null || !mounted) return;
    final bytes = await file.readAsBytes();
    setState(() => _status = _model == null ? 'Loading the AniDex model…' : 'Looking at your photo…');
    try {
      final model = await _openModel();
      if (mounted) setState(() => _status = 'Looking at your photo…');
      final scan = await model.scan(bytes);
      // A shaky match (often not an animal, or one not in the AniDex) is shown but not caught.
      final kind = scan.score >= _catalog!.lowConfidence ? await _collection!.record(scan, bytes) : null;
      if (!mounted) return;
      setState(() => _status = null);
      await Navigator.of(context).push(MaterialPageRoute(
        builder: (_) => AniDexCardScreen(
          species: scan.species,
          scan: scan,
          photo: bytes,
          kind: kind,
          entry: _collection!.caught[scan.species.scientific],
          lowConfidence: _catalog!.lowConfidence,
        ),
      ));
    } catch (e) {
      if (!mounted) return;
      setState(() => _status = null);
      _snack(e is AniDexModelMissing ? '$e' : 'Could not scan this photo: $e');
    }
  }

  void _open(AniDexSpecies s) {
    final entry = _collection!.caught[s.scientific];
    if (entry == null) return;
    Navigator.of(context).push(MaterialPageRoute(
      builder: (_) => AniDexCardScreen(species: s, entry: entry, lowConfidence: _catalog!.lowConfidence),
    ));
  }

  Future<void> _clearAll() async {
    final ok = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Clear your AniDex?'),
        content: const Text('This clears your AniDex and deletes its saved photos from this phone.'),
        actions: [
          TextButton(onPressed: () => Navigator.pop(context, false), child: const Text('Cancel')),
          TextButton(onPressed: () => Navigator.pop(context, true), child: const Text('Clear')),
        ],
      ),
    );
    if (ok == true) await _collection!.clearAll();
  }

  void _snack(String message) =>
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(message)));

  @override
  Widget build(BuildContext context) {
    final catalog = _catalog, collection = _collection;
    if (_loadError != null) {
      return Scaffold(body: Center(child: Padding(
        padding: const EdgeInsets.all(24),
        child: Text('The AniDex could not load: $_loadError', textAlign: TextAlign.center),
      )));
    }
    if (catalog == null || collection == null) {
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    }
    final text = Theme.of(context).textTheme;
    final scheme = Theme.of(context).colorScheme;
    final shown = [for (final s in catalog.species) if (_group == 'All' || s.group == _group) s];
    final caught = collection.caught.length, total = catalog.species.length;

    return Scaffold(
      body: Stack(children: [
        CustomScrollView(slivers: [
          SliverToBoxAdapter(
            child: Container(
              decoration: const BoxDecoration(
                gradient: Brand.night,
                borderRadius: BorderRadius.vertical(bottom: Radius.circular(30)),
              ),
              padding: const EdgeInsets.fromLTRB(20, 0, 20, 22),
              child: SafeArea(
                bottom: false,
                child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                  const SizedBox(height: 14),
                  Row(children: [
                    const Icon(Icons.pets, color: Brand.venom, size: 30),
                    const SizedBox(width: 10),
                    Text('ANIDEX', style: text.headlineSmall?.copyWith(color: Colors.white, letterSpacing: 2)),
                  ]),
                  const SizedBox(height: 14),
                  Text('Scan any animal', style: text.displaySmall?.copyWith(color: Colors.white)),
                  const SizedBox(height: 6),
                  Text(
                    'Pets, farm animals, birds and more: ${catalog.species.length} species and '
                    '${catalog.breedCount} breeds to catch. Runs on this phone.',
                    style: text.bodyMedium?.copyWith(color: Colors.white.withValues(alpha: 0.78)),
                  ),
                  const SizedBox(height: 16),
                  Text('$caught / $total caught · ${collection.breedCount} '
                      'breed${collection.breedCount == 1 ? '' : 's'} · ${collection.scanCount} '
                      'scan${collection.scanCount == 1 ? '' : 's'}',
                      style: Brand.label(Brand.venom)),
                  const SizedBox(height: 8),
                  ClipRRect(
                    borderRadius: BorderRadius.circular(8),
                    child: LinearProgressIndicator(
                      value: caught / total,
                      minHeight: 8,
                      color: Brand.venom,
                      backgroundColor: Colors.white.withValues(alpha: 0.15),
                    ),
                  ),
                ]),
              ),
            ),
          ),
          SliverPadding(
            padding: const EdgeInsets.fromLTRB(16, 16, 16, 0),
            sliver: SliverToBoxAdapter(
              child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
                if (_modelAvailable == false)
                  Container(
                    margin: const EdgeInsets.only(bottom: 12),
                    padding: const EdgeInsets.all(14),
                    decoration: BoxDecoration(
                      color: Brand.cautionSoft,
                      borderRadius: BorderRadius.circular(14),
                    ),
                    child: const Text(
                      'The AniDex model is not in this build. Build it with tool/anidex/build_anidex.py '
                      '(see assets/anidex/README.md).',
                      style: TextStyle(color: Brand.ink),
                    ),
                  ),
                Row(children: [
                  Expanded(child: _ScanTile(icon: Icons.photo_camera_rounded, label: 'Photo',
                      onTap: _status != null ? null : () => _pick(ImageSource.camera))),
                  const SizedBox(width: 10),
                  Expanded(child: _ScanTile(icon: Icons.photo_library_rounded, label: 'Gallery',
                      onTap: _status != null ? null : () => _pick(ImageSource.gallery))),
                ]),
                const SizedBox(height: 14),
                SizedBox(
                  height: 40,
                  child: ListView(scrollDirection: Axis.horizontal, children: [
                    for (final g in ['All', ...catalog.groups])
                      Padding(
                        padding: const EdgeInsets.only(right: 8),
                        child: ChoiceChip(
                          label: Text(g),
                          labelStyle: TextStyle(
                            fontWeight: FontWeight.w600,
                            color: _group == g ? scheme.onPrimaryContainer : scheme.onSurface,
                          ),
                          checkmarkColor: scheme.onPrimaryContainer,
                          selected: _group == g,
                          onSelected: (_) => setState(() => _group = g),
                        ),
                      ),
                  ]),
                ),
                const SizedBox(height: 12),
              ]),
            ),
          ),
          SliverPadding(
            padding: const EdgeInsets.symmetric(horizontal: 16),
            sliver: SliverGrid(
              gridDelegate: const SliverGridDelegateWithMaxCrossAxisExtent(
                maxCrossAxisExtent: 130,
                mainAxisSpacing: 10,
                crossAxisSpacing: 10,
                childAspectRatio: 0.78,
              ),
              delegate: SliverChildBuilderDelegate(
                (context, i) => _Cell(
                  species: shown[i],
                  entry: collection.caught[shown[i].scientific],
                  onTap: () => _open(shown[i]),
                ),
                childCount: shown.length,
              ),
            ),
          ),
          SliverToBoxAdapter(
            child: Padding(
              // Room at the bottom so the app's SOS button never covers the footer.
              padding: const EdgeInsets.fromLTRB(16, 16, 16, 96),
              child: Column(children: [
                if (caught > 0)
                  TextButton.icon(
                    onPressed: _clearAll,
                    icon: const Icon(Icons.delete_sweep_outlined),
                    label: const Text('Clear AniDex'),
                  ),
                Text(
                  'Animal ID by BioCLIP 2 (Imageomics, MIT licence). For fun and learning: it is not a '
                  'safety check for snakes.',
                  textAlign: TextAlign.center,
                  style: text.bodySmall?.copyWith(color: scheme.onSurfaceVariant),
                ),
              ]),
            ),
          ),
        ]),
        if (_status != null)
          Positioned.fill(
            child: ColoredBox(
              color: Colors.black54,
              child: Center(
                child: Card(
                  child: Padding(
                    padding: const EdgeInsets.all(24),
                    child: Column(mainAxisSize: MainAxisSize.min, children: [
                      const CircularProgressIndicator(),
                      const SizedBox(height: 16),
                      Text(_status!, textAlign: TextAlign.center),
                    ]),
                  ),
                ),
              ),
            ),
          ),
      ]),
    );
  }
}

class _ScanTile extends StatelessWidget {
  const _ScanTile({required this.icon, required this.label, required this.onTap});
  final IconData icon;
  final String label;
  final VoidCallback? onTap;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Material(
      color: scheme.surfaceContainerLow,
      borderRadius: BorderRadius.circular(20),
      child: InkWell(
        borderRadius: BorderRadius.circular(20),
        onTap: onTap,
        child: Container(
          height: 88,
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(20),
            border: Border.all(color: scheme.outlineVariant),
          ),
          child: Column(mainAxisAlignment: MainAxisAlignment.center, children: [
            Container(
              width: 40,
              height: 40,
              decoration: BoxDecoration(color: scheme.primaryContainer, borderRadius: BorderRadius.circular(12)),
              child: Icon(icon, color: scheme.onPrimaryContainer),
            ),
            const SizedBox(height: 6),
            Text(label, style: Theme.of(context).textTheme.titleSmall),
          ]),
        ),
      ),
    );
  }
}

class _Cell extends StatelessWidget {
  const _Cell({required this.species, required this.entry, required this.onTap});
  final AniDexSpecies species;
  final CaughtEntry? entry;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final text = Theme.of(context).textTheme;
    final number = '#${species.no.toString().padLeft(3, '0')}';
    final e = entry;
    if (e == null) {
      return Container(
        decoration: BoxDecoration(
          color: scheme.surfaceContainerHigh,
          borderRadius: BorderRadius.circular(16),
        ),
        child: Column(children: [
          Expanded(child: Center(child: Text('?', style: text.headlineLarge?.copyWith(color: scheme.outline)))),
          Text(number, style: Brand.label(scheme.outline)),
          Text('???', style: text.bodySmall?.copyWith(color: scheme.outline)),
          const SizedBox(height: 8),
        ]),
      );
    }
    final colour = anidexColour(species.group);
    final hasPhoto = e.photo != null && File(e.photo!).existsSync();
    return Material(
      color: scheme.surface,
      borderRadius: BorderRadius.circular(16),
      clipBehavior: Clip.antiAlias,
      child: InkWell(
        onTap: onTap,
        child: Container(
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(16),
            border: Border.all(color: colour, width: 3),
          ),
          child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
            Expanded(
              child: hasPhoto
                  ? Image.file(File(e.photo!), fit: BoxFit.cover, cacheWidth: 260)
                  : ColoredBox(color: colour.withValues(alpha: 0.15), child: Icon(Icons.pets, color: colour)),
            ),
            Padding(
              padding: const EdgeInsets.fromLTRB(8, 4, 8, 6),
              child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                Text(number, style: Brand.label(colour).copyWith(fontSize: 12)),
                Text(species.name, maxLines: 1, overflow: TextOverflow.ellipsis, style: text.labelLarge),
                Text(
                  species.hasBreeds ? '${e.breeds.length} breed${e.breeds.length == 1 ? '' : 's'} · ${e.count}×' : '${e.count}×',
                  style: text.bodySmall?.copyWith(color: scheme.onSurfaceVariant),
                ),
              ]),
            ),
          ]),
        ),
      ),
    );
  }
}
