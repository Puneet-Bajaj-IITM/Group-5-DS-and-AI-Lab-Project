import 'dart:io';

import 'package:material_ui/material_ui.dart';
import 'package:image_picker/image_picker.dart';
import 'package:path_provider/path_provider.dart';
import 'package:provider/provider.dart';

import '../app.dart';
import '../services/location_service.dart';
import '../state/app_state.dart';
import '../util/geo.dart';
import '../ui/brand.dart';
import '../widgets/common.dart';
import '../services/classifier.dart';
import 'result_screen.dart';
import 'video_check_screen.dart';
import 'live_review_screen.dart';
import 'community_screen.dart';

class IdentifyScreen extends StatefulWidget {
  const IdentifyScreen({super.key});

  @override
  State<IdentifyScreen> createState() => _IdentifyScreenState();
}

class _IdentifyScreenState extends State<IdentifyScreen> {
  bool _busy = false;

  Future<void> _liveCamera() async {
    if (_busy) return;
    setState(() => _busy = true);
    try {
      final evidence = await Navigator.of(context).push<LiveEvidence>(
        MaterialPageRoute(builder: (_) => const LiveReviewScreen()),
      );
      if (mounted && evidence != null) {
        await Navigator.of(context).push(
          MaterialPageRoute(
            builder: (_) => CommunityScreen(initialEvidence: evidence),
          ),
        );
      }
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  /// Shown over the screen while a photo is being checked.
  String? _status;

  @override
  void initState() {
    super.initState();
    // Android may kill the app while the camera app is open (common on
    // phones with little memory). Pick the photo up again on return.
    WidgetsBinding.instance.addPostFrameCallback((_) => _recoverLostPhoto());
  }

  Future<void> _recoverLostPhoto() async {
    try {
      final lost = await ImagePicker().retrieveLostData();
      final file = lost.file ?? (lost.files?.isNotEmpty == true ? lost.files!.first : null);
      if (file != null && mounted) await _identifyFile(file);
    } catch (_) {}
  }

  Future<void> _pick(ImageSource source) async {
    if (_busy) return;
    XFile? file;
    try {
      file = await ImagePicker().pickImage(source: source, maxWidth: 1600, imageQuality: 92);
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Could not open the camera or gallery: $e')));
      }
      return;
    }
    if (file != null && mounted) await _identifyFile(file);
  }

  Future<void> _identifyFile(XFile file) async {
    if (_busy) return;
    setState(() {
      _busy = true;
      _status = 'Checking the photo…';
    });
    final state = context.read<AppState>();
    final navigator = Navigator.of(context);
    final messenger = ScaffoldMessenger.of(context);
    try {
      final bytes = await file.readAsBytes();
      final saved = await _keepCopy(file);
      final result = await state.classifier.identify(bytes);
      if (!mounted) return;
      navigator.push(
        MaterialPageRoute(builder: (_) => ResultScreen(identification: result, photoPath: saved)),
      );
    } on ModelUnavailable catch (e) {
      if (mounted) messenger.showSnackBar(SnackBar(content: Text('$e'), duration: const Duration(seconds: 8)));
    } catch (e) {
      if (mounted) messenger.showSnackBar(SnackBar(content: Text('Could not read this photo. ($e)')));
    } finally {
      if (mounted) {
        setState(() {
          _busy = false;
          _status = null;
        });
      }
    }
  }

  Future<void> _pickVideo() async {
    if (_busy) return;
    final state = context.read<AppState>();
    final messenger = ScaffoldMessenger.of(context);
    final classifier = state.classifier;
    if (classifier is! OnnxSnakeClassifier) {
      messenger.showSnackBar(const SnackBar(
          content: Text('Video checks need the model files, which are not in this build.')));
      return;
    }
    final file = await ImagePicker().pickVideo(source: ImageSource.gallery);
    if (file == null || !mounted) return;
    await Navigator.of(context).push(MaterialPageRoute(
      builder: (_) => VideoCheckScreen(path: file.path, classifier: classifier),
    ));
  }

  /// The picker's file lives in a cache that Android may clear.
  Future<String?> _keepCopy(XFile file) async {
    try {
      final dir = Directory(
        '${(await getApplicationDocumentsDirectory()).path}/sightings',
      );
      await dir.create(recursive: true);
      final path = '${dir.path}/${DateTime.now().millisecondsSinceEpoch}.jpg';
      await File(file.path).copy(path);
      return path;
    } catch (_) {
      return null;
    }
  }

  @override
  Widget build(BuildContext context) {
    final s = context.watch<AppState>();
    final scheme = Theme.of(context).colorScheme;
    final text = Theme.of(context).textTheme;

    final locationLabel = switch (s.locationStatus) {
      LocationStatus.granted when s.address != null => s.address!,
      LocationStatus.granted when s.position != null => 'Finding your address…',
      LocationStatus.granted => 'Finding your location…',
      LocationStatus.serviceOff => 'Location is off · tap to turn on',
      LocationStatus.denied || LocationStatus.deniedForever => 'Location not allowed · tap to fix',
      LocationStatus.unknown => 'Checking location…',
    };
    final demo = s.classifier.isDemo;

    return Scaffold(
      body: Stack(children: [
        _home(context, s, text, scheme, locationLabel, demo),
        if (_status != null) _CheckingOverlay(message: _status!),
      ]),
    );
  }

  Widget _home(BuildContext context, AppState s, TextTheme text, ColorScheme scheme, String locationLabel, bool demo) {
    return Scaffold(
      body: CustomScrollView(
        slivers: [
          SliverToBoxAdapter(
            child: Container(
              decoration: const BoxDecoration(
                gradient: Brand.night,
                borderRadius: BorderRadius.vertical(bottom: Radius.circular(30)),
              ),
              padding: const EdgeInsets.fromLTRB(20, 0, 20, 26),
              child: SafeArea(
                bottom: false,
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const SizedBox(height: 14),
                    Row(
                      children: [
                        Image.asset('assets/images/app_logo.png', width: 34, height: 34, filterQuality: FilterQuality.medium),
                        const SizedBox(width: 10),
                        Text('SERPENTGUARD',
                            style: text.headlineSmall?.copyWith(color: Colors.white, letterSpacing: 2)),
                        const Spacer(),
                        Container(
                          padding: const EdgeInsets.symmetric(horizontal: 9, vertical: 4),
                          decoration: BoxDecoration(
                            border: Border.all(color: Brand.venom.withValues(alpha: 0.7)),
                            borderRadius: BorderRadius.circular(8),
                          ),
                          child: Row(mainAxisSize: MainAxisSize.min, children: [
                            const Icon(Icons.offline_bolt_rounded, size: 14, color: Brand.venom),
                            const SizedBox(width: 4),
                            Text(demo ? 'DEMO' : 'ON-DEVICE', style: Brand.label(Brand.venom).copyWith(fontSize: 12)),
                          ]),
                        ),
                      ],
                    ),
                    const SizedBox(height: 22),
                    Text('Seen a snake?', style: text.displaySmall?.copyWith(color: Colors.white)),
                    const SizedBox(height: 6),
                    Text(
                      'Stay at least 2 metres away. Point the camera from where you stand. Never go closer for a better picture.',
                      style: text.bodyMedium?.copyWith(color: Colors.white.withValues(alpha: 0.78)),
                    ),
                    const SizedBox(height: 16),
                    Wrap(spacing: 8, runSpacing: 8, children: [
                      ConstrainedBox(
                        constraints: BoxConstraints(maxWidth: MediaQuery.sizeOf(context).width - 40),
                        child: StatusPill(
                          onDark: true,
                          icon: Icons.place_rounded,
                          label: locationLabel,
                          ok: s.locationStatus == LocationStatus.granted,
                          onTap: s.locationStatus == LocationStatus.granted ? s.refreshLocation : s.openLocationSettings,
                        ),
                      ),
                      StatusPill(
                        onDark: true,
                        icon: Icons.cell_tower_rounded,
                        label: s.alertsOn ? 'Nearby alerts on' : 'Nearby alerts off',
                        ok: s.alertsOn,
                      ),
                    ]),
                  ],
                ),
              ),
            ),
          ),
          SliverToBoxAdapter(
            child: Transform.translate(
              offset: Offset.zero,
              child: Padding(
                padding: const EdgeInsets.fromLTRB(16, 16, 16, 24),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    _LiveTile(
                      busy: _busy && _status == null,
                      demo: demo,
                      onTap: _busy ? null : _liveCamera,
                    ),
                    const SizedBox(height: 10),
                    Row(children: [
                      Expanded(child: _ActionTile(icon: Icons.photo_camera_rounded, label: 'Photo', onTap: _busy ? null : () => _pick(ImageSource.camera))),
                      const SizedBox(width: 10),
                      Expanded(child: _ActionTile(icon: Icons.photo_library_rounded, label: 'Gallery', onTap: _busy ? null : () => _pick(ImageSource.gallery))),
                      const SizedBox(width: 10),
                      Expanded(child: _ActionTile(icon: Icons.movie_filter_rounded, label: 'Video', onTap: _busy ? null : _pickVideo)),
                    ]),
                    const SizedBox(height: 18),
                    EmergencyButton(
                      label: "I'VE BEEN BITTEN",
                      icon: Icons.emergency_rounded,
                      onPressed: () => openSos(context, bitten: true),
                    ),
                    const SizedBox(height: 10),
                    OutlinedButton.icon(
                      style: OutlinedButton.styleFrom(
                        foregroundColor: Brand.danger,
                        side: const BorderSide(color: Brand.danger, width: 1.5),
                      ),
                      onPressed: () => openSos(context),
                      icon: const Icon(Icons.local_hospital_rounded),
                      label: const Text('Emergency numbers & hospitals'),
                    ),
                    SectionTitle(
                      'Your sightings',
                      trailing: s.sightings.isEmpty
                          ? null
                          : TextButton(onPressed: s.clearSightings, child: const Text('Clear')),
                    ),
                    if (s.sightings.isEmpty)
                      Container(
                        padding: const EdgeInsets.all(18),
                        decoration: BoxDecoration(
                          color: scheme.surfaceContainerLow,
                          borderRadius: BorderRadius.circular(18),
                          border: Border.all(color: scheme.outlineVariant),
                        ),
                        child: Row(children: [
                          Icon(Icons.photo_album_outlined, color: scheme.onSurfaceVariant),
                          const SizedBox(width: 12),
                          Expanded(
                            child: Text('Snakes you identify are saved here, on this phone only.',
                                style: text.bodyMedium?.copyWith(color: scheme.onSurfaceVariant)),
                          ),
                        ]),
                      )
                    else
                      ...s.sightings.take(10).map((x) {
                        final sp = x.speciesId == null ? null : s.catalog.byId(x.speciesId!);
                        final style = VerdictStyle.of(x.verdict);
                        final hasPhoto = x.photoPath != null && File(x.photoPath!).existsSync();
                        return Padding(
                          padding: const EdgeInsets.only(bottom: 10),
                          child: Card(
                            child: Padding(
                              padding: const EdgeInsets.all(10),
                              child: Row(children: [
                                ClipRRect(
                                  borderRadius: BorderRadius.circular(12),
                                  child: SizedBox(
                                    width: 60,
                                    height: 60,
                                    child: hasPhoto
                                        ? Image.file(File(x.photoPath!), fit: BoxFit.cover, cacheWidth: 180)
                                        : ColoredBox(color: style.soft, child: Icon(style.icon, color: style.background)),
                                  ),
                                ),
                                const SizedBox(width: 12),
                                Expanded(
                                  child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                                    Text(sp?.displayName ?? 'Unidentified snake',
                                        style: text.titleMedium, maxLines: 1, overflow: TextOverflow.ellipsis),
                                    const SizedBox(height: 4),
                                    Row(children: [
                                      Container(
                                        padding: const EdgeInsets.symmetric(horizontal: 7, vertical: 2),
                                        decoration: BoxDecoration(color: style.soft, borderRadius: BorderRadius.circular(6)),
                                        child: Text(style.tag, style: Brand.label(style.background).copyWith(fontSize: 11.5)),
                                      ),
                                      const SizedBox(width: 8),
                                      Text(formatAge(DateTime.now().difference(x.time)),
                                          style: text.bodySmall?.copyWith(color: scheme.onSurfaceVariant)),
                                    ]),
                                  ]),
                                ),
                              ]),
                            ),
                          ),
                        );
                      }),
                  ],
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _LiveTile extends StatelessWidget {
  const _LiveTile({required this.busy, required this.demo, required this.onTap});
  final bool busy, demo;
  final VoidCallback? onTap;

  @override
  Widget build(BuildContext context) {
    final text = Theme.of(context).textTheme;
    return Material(
      color: Brand.forest600,
      borderRadius: BorderRadius.circular(24),
      elevation: 3,
      shadowColor: Brand.forest900.withValues(alpha: 0.4),
      child: InkWell(
        borderRadius: BorderRadius.circular(24),
        onTap: onTap,
        child: Padding(
          padding: const EdgeInsets.fromLTRB(20, 18, 16, 18),
          child: Row(children: [
            Container(
              width: 58,
              height: 58,
              decoration: BoxDecoration(color: Colors.white.withValues(alpha: 0.14), shape: BoxShape.circle),
              child: busy
                  ? const Padding(padding: EdgeInsets.all(16), child: CircularProgressIndicator(strokeWidth: 3, color: Colors.white))
                  : const Icon(Icons.center_focus_strong_rounded, color: Colors.white, size: 32),
            ),
            const SizedBox(width: 16),
            Expanded(
              child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                Text(busy ? 'Opening…' : 'Live camera', style: text.headlineSmall?.copyWith(color: Colors.white)),
                const SizedBox(height: 3),
                Text(
                  demo
                      ? 'Model files missing from this build: demo only.'
                      : 'Point at the snake. It settles on one answer once the frames agree.',
                  style: text.bodySmall?.copyWith(color: Colors.white.withValues(alpha: 0.82)),
                ),
              ]),
            ),
            const Icon(Icons.chevron_right_rounded, color: Colors.white, size: 30),
          ]),
        ),
      ),
    );
  }
}

class _ActionTile extends StatelessWidget {
  const _ActionTile({required this.icon, required this.label, required this.onTap});
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
          height: 96,
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(20),
            border: Border.all(color: scheme.outlineVariant),
          ),
          child: Column(mainAxisAlignment: MainAxisAlignment.center, children: [
            Container(
              width: 42,
              height: 42,
              decoration: BoxDecoration(color: scheme.primaryContainer, borderRadius: BorderRadius.circular(13)),
              child: Icon(icon, color: scheme.onPrimaryContainer),
            ),
            const SizedBox(height: 8),
            Text(label, style: Theme.of(context).textTheme.titleSmall),
          ]),
        ),
      ),
    );
  }
}


/// A calm full-screen card while the model looks at a photo.
class _CheckingOverlay extends StatelessWidget {
  const _CheckingOverlay({required this.message});
  final String message;

  @override
  Widget build(BuildContext context) {
    final text = Theme.of(context).textTheme;
    return Positioned.fill(
      child: ColoredBox(
        color: Brand.forest950.withValues(alpha: 0.72),
        child: Center(
          child: Container(
            margin: const EdgeInsets.all(32),
            padding: const EdgeInsets.fromLTRB(26, 26, 26, 22),
            decoration: BoxDecoration(
              color: Theme.of(context).colorScheme.surfaceContainerLow,
              borderRadius: BorderRadius.circular(24),
            ),
            child: Column(mainAxisSize: MainAxisSize.min, children: [
              const SizedBox(width: 44, height: 44, child: CircularProgressIndicator(strokeWidth: 4)),
              const SizedBox(height: 18),
              Text(message, style: text.titleLarge),
              const SizedBox(height: 6),
              Text('On this phone, no internet needed. Keep your distance meanwhile.',
                  textAlign: TextAlign.center, style: text.bodySmall),
            ]),
          ),
        ),
      ),
    );
  }
}
