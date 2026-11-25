import 'package:material_ui/material_ui.dart';
import 'package:provider/provider.dart';

import 'models/verdict.dart';
import 'screens/alerts_screen.dart';
import 'screens/contacts_screen.dart';
import 'screens/community_screen.dart';
import 'screens/identify_screen.dart';
import 'screens/map_screen.dart';
import 'screens/sos_screen.dart';
import 'screens/splash_screen.dart';
import 'state/app_state.dart';
import 'util/geo.dart';

class SerpentGuardApp extends StatelessWidget {
  const SerpentGuardApp({super.key, this.initFuture, this.showSplash = true});

  final Future<void>? initFuture;
  final bool showSplash;

  static const seed = Color(0xFF2F5D45);

  static ThemeData theme(Brightness b) {
    final scheme = ColorScheme.fromSeed(seedColor: seed, brightness: b);
    return ThemeData(
      colorScheme: scheme,
      useMaterial3: true,
      cardTheme: CardThemeData(
        elevation: 0,
        color: scheme.surfaceContainerLow,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
        margin: EdgeInsets.zero,
      ),
      appBarTheme: AppBarTheme(
        backgroundColor: scheme.surface,
        centerTitle: false,
        titleTextStyle: TextStyle(
          color: scheme.onSurface,
          fontSize: 22,
          fontWeight: FontWeight.w800,
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'SerpentGuard',
      debugShowCheckedModeBanner: false,
      theme: theme(Brightness.light),
      darkTheme: theme(Brightness.dark),
      home: showSplash
          ? SplashScreen(initFuture: initFuture)
          : const HomeShell(),
    );
  }
}

class HomeShell extends StatefulWidget {
  const HomeShell({super.key});

  @override
  State<HomeShell> createState() => _HomeShellState();
}

class _HomeShellState extends State<HomeShell> {
  int _tab = 0;
  final Set<int> _visitedTabs = {0};
  AppState? _state;

  void _selectTab(int index) {
    setState(() {
      _tab = index;
      _visitedTabs.add(index);
    });
  }

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    final s = context.read<AppState>();
    if (s != _state) {
      _state?.removeListener(_onState);
      _state = s..addListener(_onState);
    }
  }

  @override
  void dispose() {
    _state?.removeListener(_onState);
    super.dispose();
  }

  /// Pops a banner the moment a new nearby alert arrives.
  void _onState() {
    final s = _state!;
    final r = s.incoming;
    if (r == null) return;
    s.clearIncoming();
    final species = s.catalog.byId(r.alert.speciesId);
    final where = s.position == null
        ? ''
        : ' · ${formatDistance(distanceMeters(s.position!, r.alert.location))} '
              '${compassPoint(bearingDegrees(s.position!, r.alert.location))}';
    final messenger = ScaffoldMessenger.of(context);
    messenger.hideCurrentMaterialBanner();
    messenger.showMaterialBanner(
      MaterialBanner(
        backgroundColor: r.alert.verdict.warns
            ? const Color(0xFFFFDAD6)
            : Theme.of(context).colorScheme.tertiaryContainer,
        leading: const Icon(Icons.campaign_rounded, color: Color(0xFF8C1D18)),
        content: Text(
          'Snake reported nearby: ${species?.displayName ?? 'unidentified snake'} '
          '(${r.alert.verdict.title})$where',
          style: const TextStyle(
            color: Color(0xFF410002),
            fontWeight: FontWeight.w600,
          ),
        ),
        actions: [
          TextButton(
            onPressed: () {
              messenger.hideCurrentMaterialBanner();
              _selectTab(2);
            },
            child: const Text('View'),
          ),
          TextButton(
            onPressed: messenger.hideCurrentMaterialBanner,
            child: const Text('Dismiss'),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final alerts = context.select<AppState, int>(
      (s) => s.alertBook.active(DateTime.now()).length,
    );
    final pages = [
      const IdentifyScreen(),
      _visitedTabs.contains(1) ? const MapScreen() : const SizedBox.shrink(),
      _visitedTabs.contains(2) ? const AlertsScreen() : const SizedBox.shrink(),
      _visitedTabs.contains(3)
          ? const ContactsScreen()
          : const SizedBox.shrink(),
      _visitedTabs.contains(4)
          ? const CommunityScreen()
          : const SizedBox.shrink(),
    ];
    return Scaffold(
      body: IndexedStack(index: _tab, children: pages),
      floatingActionButton: _tab == 0
          ? null
          : FloatingActionButton.extended(
              heroTag: 'sos',
              backgroundColor: const Color(0xFFB3261E),
              foregroundColor: Colors.white,
              onPressed: () => openSos(context),
              icon: const Icon(Icons.emergency_rounded),
              label: const Text(
                'SOS',
                style: TextStyle(fontWeight: FontWeight.w800),
              ),
            ),
      bottomNavigationBar: NavigationBar(
        selectedIndex: _tab,
        onDestinationSelected: _selectTab,
        destinations: [
          const NavigationDestination(
            icon: Icon(Icons.center_focus_strong_outlined),
            selectedIcon: Icon(Icons.center_focus_strong),
            label: 'Identify',
          ),
          const NavigationDestination(
            icon: Icon(Icons.map_outlined),
            selectedIcon: Icon(Icons.map),
            label: 'Map',
          ),
          NavigationDestination(
            icon: Badge(
              isLabelVisible: alerts > 0,
              label: Text('$alerts'),
              child: const Icon(Icons.cell_tower_outlined),
            ),
            selectedIcon: const Icon(Icons.cell_tower),
            label: 'Alerts',
          ),
          const NavigationDestination(
            icon: Icon(Icons.contacts_outlined),
            selectedIcon: Icon(Icons.contacts),
            label: 'Contacts',
          ),
          const NavigationDestination(
            icon: Icon(Icons.groups_outlined),
            selectedIcon: Icon(Icons.groups),
            label: 'Community',
          ),
        ],
      ),
    );
  }
}

void openSos(
  BuildContext context, {
  bool bitten = false,
  Identification? identification,
}) {
  Navigator.of(context).push(
    MaterialPageRoute(
      builder: (_) => SosScreen(bitten: bitten, identification: identification),
    ),
  );
}
