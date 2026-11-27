import 'package:material_ui/material_ui.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:latlong2/latlong.dart';
import 'package:provider/provider.dart';
import 'package:url_launcher/url_launcher.dart';

import '../state/app_state.dart';
import '../util/geo.dart';
import '../widgets/common.dart';
import 'result_screen.dart';

/// Map tiles are cached as you browse: areas you have looked at stay
/// available offline for 30 days (up to 300 MB). OpenStreetMap's tile policy
/// forbids bulk pre-downloading from its servers, so "download my district"
/// needs our own tile source (planned).
final _tileCache = BuiltInMapCachingProvider.getOrCreateInstance(
  maxCacheSize: 300 * 1024 * 1024,
  overrideFreshAge: const Duration(days: 30),
);

class MapScreen extends StatefulWidget {
  const MapScreen({super.key});

  @override
  State<MapScreen> createState() => _MapScreenState();
}

class _MapScreenState extends State<MapScreen> {
  final _map = MapController();
  bool _showHospitals = true, _showSightings = true, _showAlerts = true;
  bool _centredOnce = false;

  static const _india = LatLng(22.5, 79.0);

  LatLng _ll(GeoPoint p) => LatLng(p.lat, p.lon);

  @override
  Widget build(BuildContext context) {
    final s = context.watch<AppState>();
    final scheme = Theme.of(context).colorScheme;
    final me = s.position;
    if (me != null && !_centredOnce) {
      _centredOnce = true;
      WidgetsBinding.instance.addPostFrameCallback((_) {
        if (mounted) _map.move(_ll(me), 14);
      });
    }

    final markers = <Marker>[
      if (_showHospitals)
        ...s.hospitals.map((h) => Marker(
              point: _ll(h.location),
              width: 36,
              height: 36,
              child: Tooltip(
                message: h.name,
                child: const _Pin(icon: Icons.local_hospital_rounded, color: Color(0xFFD32F2F)),
              ),
            )),
      if (_showSightings)
        ...s.sightings.where((x) => x.location != null).map((x) {
          final st = VerdictStyle.of(x.verdict);
          return Marker(
            point: _ll(x.location!),
            width: 32,
            height: 32,
            child: GestureDetector(
              onTap: () => Navigator.of(context).push(MaterialPageRoute(
                builder: (_) => ResultScreen.saved(x, s.catalog),
              )),
              child: _Pin(icon: Icons.pest_control_rodent_outlined, color: st.background),
            ),
          );
        }),
      if (_showAlerts)
        ...s.alertBook.active(DateTime.now()).map((r) => Marker(
              point: _ll(r.alert.location),
              width: 40,
              height: 40,
              child: _Pin(
                icon: Icons.campaign_rounded,
                color: r.alert.verdict.warns
                    ? const Color(0xFFB3261E)
                    : const Color(0xFFE8A317),
              ),
            )),
      if (me != null)
        Marker(
          point: _ll(me),
          width: 24,
          height: 24,
          child: Container(
            decoration: BoxDecoration(
              color: const Color(0xFF1A73E8),
              shape: BoxShape.circle,
              border: Border.all(color: Colors.white, width: 3),
              boxShadow: const [BoxShadow(blurRadius: 6, color: Colors.black26)],
            ),
          ),
        ),
    ];

    return Scaffold(
      appBar: AppBar(title: const Text('Map')),
      body: Stack(children: [
        FlutterMap(
          mapController: _map,
          options: MapOptions(
            initialCenter: me == null ? _india : _ll(me),
            initialZoom: me == null ? 4.5 : 14,
            minZoom: 3,
            maxZoom: 18,
          ),
          children: [
            TileLayer(
              urlTemplate: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
              userAgentPackageName: 'in.serpentguard.serpentguard',
              tileProvider: NetworkTileProvider(cachingProvider: _tileCache),
            ),
            if (me != null && s.accuracy != null)
              CircleLayer(circles: [
                CircleMarker(
                  point: _ll(me),
                  radius: s.accuracy!,
                  useRadiusInMeter: true,
                  color: const Color(0x221A73E8),
                  borderColor: const Color(0x661A73E8),
                  borderStrokeWidth: 1,
                ),
              ]),
            MarkerLayer(markers: markers),
            RichAttributionWidget(alignment: AttributionAlignment.bottomLeft, attributions: [
              TextSourceAttribution(
                'OpenStreetMap contributors',
                onTap: () => launchUrl(Uri.parse('https://openstreetmap.org/copyright')),
              ),
            ]),
          ],
        ),
        Positioned(
          left: 12,
          right: 12,
          top: 8,
          child: SingleChildScrollView(
            scrollDirection: Axis.horizontal,
            child: Row(children: [
              FilterChip(
                avatar: const Icon(Icons.local_hospital_rounded, size: 18),
                label: Text('Hospitals (${s.hospitals.length})'),
                selected: _showHospitals,
                onSelected: (v) => setState(() => _showHospitals = v),
              ),
              const SizedBox(width: 8),
              FilterChip(
                avatar: const Icon(Icons.history_rounded, size: 18),
                label: const Text('My sightings'),
                selected: _showSightings,
                onSelected: (v) => setState(() => _showSightings = v),
              ),
              const SizedBox(width: 8),
              FilterChip(
                avatar: const Icon(Icons.campaign_rounded, size: 18),
                label: const Text('Nearby alerts'),
                selected: _showAlerts,
                onSelected: (v) => setState(() => _showAlerts = v),
              ),
            ]),
          ),
        ),
        Positioned(
          left: 12,
          bottom: 72,
          child: FloatingActionButton.small(
            heroTag: 'recentre',
            backgroundColor: scheme.surface,
            tooltip: 'Centre on me',
            onPressed: () async {
              if (s.position == null) await s.refreshLocation();
              if (s.position != null) _map.move(_ll(s.position!), 15);
            },
            child: const Icon(Icons.my_location_rounded),
          ),
        ),
      ]),
    );
  }
}

class _Pin extends StatelessWidget {
  const _Pin({required this.icon, required this.color});
  final IconData icon;
  final Color color;

  @override
  Widget build(BuildContext context) => Container(
        decoration: BoxDecoration(
          color: color,
          shape: BoxShape.circle,
          border: Border.all(color: Colors.white, width: 2),
          boxShadow: const [BoxShadow(blurRadius: 4, color: Colors.black26)],
        ),
        child: Icon(icon, color: Colors.white, size: 18),
      );
}
