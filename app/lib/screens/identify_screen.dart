import 'dart:io';

import 'package:material_ui/material_ui.dart';
import 'package:image_picker/image_picker.dart';
import 'package:path_provider/path_provider.dart';
import 'package:provider/provider.dart';

import '../app.dart';
import '../services/location_service.dart';
import '../state/app_state.dart';
import '../util/geo.dart';
import '../widgets/common.dart';
import 'result_screen.dart';
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

  Future<void> _pick(ImageSource source) async {
    if (_busy) return;
    setState(() => _busy = true);
    final state = context.read<AppState>();
    final navigator = Navigator.of(context);
    final messenger = ScaffoldMessenger.of(context);
    try {
      final file = await ImagePicker().pickImage(
        source: source,
        maxWidth: 1600,
        imageQuality: 90,
      );
      if (file == null) return;
      final bytes = await file.readAsBytes();
      final saved = await _keepCopy(file);
      final result = await state.classifier.identify(bytes);
      if (!mounted) return;
      navigator.push(
        MaterialPageRoute(
          builder: (_) =>
              ResultScreen(identification: result, photoPath: saved),
        ),
      );
    } catch (e) {
      if (mounted) {
        messenger.showSnackBar(
          SnackBar(content: Text('Could not open the photo: $e')),
        );
      }
    } finally {
      if (mounted) setState(() => _busy = false);
    }
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
      LocationStatus.granted when s.position != null =>
        'Location on${s.accuracy == null ? '' : ' · ±${s.accuracy!.round()} m'}',
      LocationStatus.granted => 'Finding location…',
      LocationStatus.serviceOff => 'Location is off',
      LocationStatus.denied ||
      LocationStatus.deniedForever => 'Location not allowed',
      LocationStatus.unknown => 'Location…',
    };

    return Scaffold(
      appBar: AppBar(
        title: Row(
          children: [
            Image.asset(
              'assets/images/app_logo.png',
              width: 30,
              height: 30,
              filterQuality: FilterQuality.medium,
            ),
            const SizedBox(width: 10),
            const Text('SerpentGuard'),
          ],
        ),
      ),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.fromLTRB(16, 4, 16, 24),
          children: [
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: [
                StatusPill(
                  icon: Icons.my_location_rounded,
                  label: locationLabel,
                  ok: s.locationStatus == LocationStatus.granted,
                  onTap: s.locationStatus == LocationStatus.granted
                      ? s.refreshLocation
                      : s.openLocationSettings,
                ),
                StatusPill(
                  icon: Icons.cell_tower_rounded,
                  label: s.alertsOn ? 'Nearby alerts on' : 'Nearby alerts off',
                  ok: s.alertsOn,
                ),
              ],
            ),
            const SizedBox(height: 16),
            Card(
              child: Padding(
                padding: const EdgeInsets.all(20),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Seen a snake?',
                      style: text.headlineSmall?.copyWith(
                        fontWeight: FontWeight.w800,
                      ),
                    ),
                    const SizedBox(height: 6),
                    Text(
                      'Stay at least 2 metres away and take a photo from where you are. '
                      'Never go closer for a better picture.',
                      style: text.bodyMedium?.copyWith(
                        color: scheme.onSurfaceVariant,
                      ),
                    ),
                    const SizedBox(height: 18),
                    if (_busy)
                      const Padding(
                        padding: EdgeInsets.symmetric(vertical: 14),
                        child: Row(
                          children: [
                            SizedBox(
                              width: 22,
                              height: 22,
                              child: CircularProgressIndicator(strokeWidth: 3),
                            ),
                            SizedBox(width: 14),
                            Text('Preparing capture…'),
                          ],
                        ),
                      )
                    else ...[
                      SizedBox(
                        width: double.infinity,
                        height: 56,
                        child: FilledButton.icon(
                          onPressed: _liveCamera,
                          icon: const Icon(Icons.videocam_rounded),
                          label: const Text(
                            'Live camera',
                            style: TextStyle(fontSize: 17),
                          ),
                        ),
                      ),
                      const Padding(
                        padding: EdgeInsets.symmetric(vertical: 8),
                        child: Text(
                          'Continuous preview. Live model identification is awaiting integration.',
                        ),
                      ),
                      SizedBox(
                        width: double.infinity,
                        height: 56,
                        child: OutlinedButton.icon(
                          onPressed: () => _pick(ImageSource.camera),
                          icon: const Icon(Icons.photo_camera_rounded),
                          label: const Text(
                            'Take a photo',
                            style: TextStyle(fontSize: 17),
                          ),
                        ),
                      ),
                      const SizedBox(height: 10),
                      SizedBox(
                        width: double.infinity,
                        height: 52,
                        child: OutlinedButton.icon(
                          onPressed: () => _pick(ImageSource.gallery),
                          icon: const Icon(Icons.photo_library_outlined),
                          label: const Text('Choose from gallery'),
                        ),
                      ),
                    ],
                  ],
                ),
              ),
            ),
            const SizedBox(height: 16),
            EmergencyButton(
              label: "I've been bitten",
              icon: Icons.emergency_rounded,
              onPressed: () => openSos(context, bitten: true),
            ),
            const SizedBox(height: 10),
            SizedBox(
              width: double.infinity,
              height: 52,
              child: OutlinedButton.icon(
                style: OutlinedButton.styleFrom(
                  foregroundColor: const Color(0xFFB3261E),
                  side: const BorderSide(color: Color(0xFFB3261E), width: 1.5),
                ),
                onPressed: () => openSos(context),
                icon: const Icon(Icons.local_hospital_outlined),
                label: const Text('Emergency numbers and hospitals'),
              ),
            ),
            SectionTitle(
              'Your sightings',
              trailing: s.sightings.isEmpty
                  ? null
                  : TextButton(
                      onPressed: s.clearSightings,
                      child: const Text('Clear'),
                    ),
            ),
            if (s.sightings.isEmpty)
              Text(
                'Snakes you identify are saved here, on this phone only.',
                style: text.bodyMedium?.copyWith(
                  color: scheme.onSurfaceVariant,
                ),
              )
            else
              ...s.sightings.take(10).map((x) {
                final sp = x.speciesId == null
                    ? null
                    : s.catalog.byId(x.speciesId!);
                final style = VerdictStyle.of(x.verdict);
                return Card(
                  margin: const EdgeInsets.only(bottom: 8),
                  child: ListTile(
                    leading:
                        x.photoPath != null && File(x.photoPath!).existsSync()
                        ? ClipRRect(
                            borderRadius: BorderRadius.circular(10),
                            child: Image.file(
                              File(x.photoPath!),
                              width: 48,
                              height: 48,
                              fit: BoxFit.cover,
                              cacheWidth: 144,
                            ),
                          )
                        : CircleAvatar(
                            backgroundColor: style.background,
                            child: Icon(style.icon, color: style.foreground),
                          ),
                    title: Text(sp?.displayName ?? 'Unidentified snake'),
                    subtitle: Text(
                      '${x.verdict.title} · ${formatAge(DateTime.now().difference(x.time))}',
                    ),
                  ),
                );
              }),
          ],
        ),
      ),
    );
  }
}
