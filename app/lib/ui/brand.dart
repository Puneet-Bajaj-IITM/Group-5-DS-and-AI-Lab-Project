import 'package:material_ui/material_ui.dart';

/// SerpentGuard's look: a field guide crossed with safety signage.
/// Deep jungle green for the brand, warm paper for surfaces, a venom-yellow
/// accent used sparingly, and verdict colours that read at a glance in sun.
/// Barlow for text, Barlow Condensed for headings and labels.
class Brand {
  Brand._();

  // Greens
  static const forest950 = Color(0xFF07140E);
  static const forest900 = Color(0xFF0D1F17);
  static const forest800 = Color(0xFF143426);
  static const forest700 = Color(0xFF1B4332);
  static const forest600 = Color(0xFF2D6A4F);
  static const forest500 = Color(0xFF40916C);
  static const moss100 = Color(0xFFDDEBE2);
  static const moss50 = Color(0xFFEEF5F0);

  // Paper
  static const paper = Color(0xFFF6F3EA);
  static const paperDeep = Color(0xFFECE6D6);
  static const ink = Color(0xFF16211B);
  static const inkSoft = Color(0xFF55615A);
  static const line = Color(0xFFE0D9C6);

  // Accent and verdicts
  static const venom = Color(0xFFE9C46A);
  static const danger = Color(0xFFB91C1C);
  static const dangerDeep = Color(0xFF7F1010);
  static const dangerSoft = Color(0xFFFBE3E1);
  static const caution = Color(0xFFD98A0B);
  static const cautionSoft = Color(0xFFFCEFD7);
  static const calm = Color(0xFF2D6A4F);
  static const calmSoft = Color(0xFFE2F0E7);

  static const display = 'BarlowCondensed';
  static const body = 'Barlow';

  /// The brand gradient behind the home header and video viewer.
  static const night = LinearGradient(
    begin: Alignment.topLeft,
    end: Alignment.bottomRight,
    colors: [forest800, forest950],
  );

  static TextStyle label(Color color) => TextStyle(
        fontFamily: display,
        fontWeight: FontWeight.w700,
        fontSize: 13,
        letterSpacing: 1.4,
        color: color,
      );

  static ThemeData theme(Brightness brightness) {
    final dark = brightness == Brightness.dark;
    final base = ColorScheme.fromSeed(seedColor: forest600, brightness: brightness);
    final scheme = base.copyWith(
      primary: dark ? const Color(0xFF7FC8A0) : forest600,
      onPrimary: dark ? forest950 : Colors.white,
      primaryContainer: dark ? forest700 : moss100,
      onPrimaryContainer: dark ? moss50 : forest800,
      secondary: dark ? venom : forest800,
      onSecondary: dark ? forest950 : Colors.white,
      tertiary: venom,
      onTertiary: forest950,
      tertiaryContainer: dark ? const Color(0xFF4A3D14) : const Color(0xFFF8EDC9),
      onTertiaryContainer: dark ? const Color(0xFFF8EDC9) : const Color(0xFF4A3D14),
      error: danger,
      onError: Colors.white,
      errorContainer: dark ? dangerDeep : dangerSoft,
      onErrorContainer: dark ? dangerSoft : dangerDeep,
      surface: dark ? forest950 : paper,
      onSurface: dark ? const Color(0xFFE8EFEA) : ink,
      onSurfaceVariant: dark ? const Color(0xFFA9B7AF) : inkSoft,
      surfaceContainerLowest: dark ? forest950 : Colors.white,
      surfaceContainerLow: dark ? forest900 : Colors.white,
      surfaceContainer: dark ? forest900 : const Color(0xFFFBF9F3),
      surfaceContainerHigh: dark ? forest800 : paperDeep,
      surfaceContainerHighest: dark ? forest700 : const Color(0xFFE4DDCA),
      outline: dark ? const Color(0xFF4E6458) : const Color(0xFFBDB4A0),
      outlineVariant: dark ? forest800 : line,
    );

    TextStyle d(double size, {FontWeight w = FontWeight.w800, double ls = 0.2}) =>
        TextStyle(fontFamily: display, fontSize: size, fontWeight: w, letterSpacing: ls, height: 1.05);
    TextStyle b(double size, {FontWeight w = FontWeight.w400, double h = 1.35}) =>
        TextStyle(fontFamily: body, fontSize: size, fontWeight: w, height: h);

    final textTheme = TextTheme(
      displayLarge: d(52),
      displayMedium: d(44),
      displaySmall: d(36),
      headlineLarge: d(32),
      headlineMedium: d(28),
      headlineSmall: d(24),
      titleLarge: b(20, w: FontWeight.w700, h: 1.2),
      titleMedium: b(16.5, w: FontWeight.w700, h: 1.25),
      titleSmall: b(14.5, w: FontWeight.w600, h: 1.25),
      bodyLarge: b(16.5),
      bodyMedium: b(15),
      bodySmall: b(13),
      labelLarge: b(16, w: FontWeight.w700, h: 1.1),
      labelMedium: b(13.5, w: FontWeight.w600),
      labelSmall: b(12, w: FontWeight.w600),
    ).apply(bodyColor: scheme.onSurface, displayColor: scheme.onSurface);

    final rounded = RoundedRectangleBorder(borderRadius: BorderRadius.circular(16));
    return ThemeData(
      useMaterial3: true,
      brightness: brightness,
      colorScheme: scheme,
      fontFamily: body,
      textTheme: textTheme,
      scaffoldBackgroundColor: scheme.surface,
      visualDensity: VisualDensity.standard,
      appBarTheme: AppBarTheme(
        backgroundColor: scheme.surface,
        foregroundColor: scheme.onSurface,
        surfaceTintColor: Colors.transparent,
        elevation: 0,
        scrolledUnderElevation: 0,
        centerTitle: false,
        titleTextStyle: d(26).copyWith(color: scheme.onSurface),
      ),
      cardTheme: CardThemeData(
        elevation: 0,
        color: scheme.surfaceContainerLow,
        surfaceTintColor: Colors.transparent,
        margin: EdgeInsets.zero,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(20),
          side: BorderSide(color: scheme.outlineVariant),
        ),
      ),
      filledButtonTheme: FilledButtonThemeData(
        style: FilledButton.styleFrom(
          minimumSize: const Size(64, 54),
          shape: rounded,
          textStyle: b(16.5, w: FontWeight.w700),
        ),
      ),
      outlinedButtonTheme: OutlinedButtonThemeData(
        style: OutlinedButton.styleFrom(
          minimumSize: const Size(64, 52),
          shape: rounded,
          foregroundColor: scheme.onSurface,
          side: BorderSide(color: scheme.outline, width: 1.2),
          textStyle: b(16, w: FontWeight.w700),
        ),
      ),
      textButtonTheme: TextButtonThemeData(
        style: TextButton.styleFrom(textStyle: b(15, w: FontWeight.w700)),
      ),
      chipTheme: ChipThemeData(
        shape: const StadiumBorder(),
        side: BorderSide(color: scheme.outlineVariant),
        labelStyle: b(13.5, w: FontWeight.w600),
        backgroundColor: scheme.surfaceContainerLow,
        selectedColor: scheme.primaryContainer,
      ),
      navigationBarTheme: NavigationBarThemeData(
        backgroundColor: scheme.surfaceContainerLow,
        surfaceTintColor: Colors.transparent,
        indicatorColor: scheme.primaryContainer,
        height: 70,
        elevation: 0,
        labelTextStyle: WidgetStateProperty.resolveWith(
          (s) => b(12.5, w: s.contains(WidgetState.selected) ? FontWeight.w800 : FontWeight.w600),
        ),
      ),
      snackBarTheme: SnackBarThemeData(
        behavior: SnackBarBehavior.floating,
        backgroundColor: dark ? forest700 : forest900,
        contentTextStyle: b(15, w: FontWeight.w500).copyWith(color: Colors.white),
        shape: rounded,
      ),
      inputDecorationTheme: InputDecorationTheme(
        filled: true,
        fillColor: scheme.surfaceContainerLow,
        border: OutlineInputBorder(borderRadius: BorderRadius.circular(14), borderSide: BorderSide(color: scheme.outline)),
        enabledBorder: OutlineInputBorder(borderRadius: BorderRadius.circular(14), borderSide: BorderSide(color: scheme.outlineVariant)),
      ),
      listTileTheme: ListTileThemeData(
        titleTextStyle: b(16, w: FontWeight.w700).copyWith(color: scheme.onSurface),
        subtitleTextStyle: b(14).copyWith(color: scheme.onSurfaceVariant),
      ),
      dividerTheme: DividerThemeData(color: scheme.outlineVariant, space: 1),
      progressIndicatorTheme: ProgressIndicatorThemeData(color: scheme.primary, linearTrackColor: scheme.surfaceContainerHigh),
      floatingActionButtonTheme: const FloatingActionButtonThemeData(elevation: 2),
    );
  }
}
