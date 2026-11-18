import 'package:material_ui/material_ui.dart';

import '../models/verdict.dart';

/// Colours that mean the same thing on every screen.
class VerdictStyle {
  const VerdictStyle(this.background, this.foreground, this.icon);
  final Color background;
  final Color foreground;
  final IconData icon;

  static VerdictStyle of(Verdict v) => switch (v) {
        Verdict.dangerous =>
          const VerdictStyle(Color(0xFFB3261E), Colors.white, Icons.dangerous_rounded),
        Verdict.unknown =>
          const VerdictStyle(Color(0xFFE8A317), Color(0xFF231A00), Icons.help_rounded),
        Verdict.notListed =>
          const VerdictStyle(Color(0xFF3D6B4F), Colors.white, Icons.visibility_rounded),
      };
}

class VerdictBanner extends StatelessWidget {
  const VerdictBanner({super.key, required this.verdict, required this.reason});
  final Verdict verdict;
  final String reason;

  @override
  Widget build(BuildContext context) {
    final s = VerdictStyle.of(verdict);
    final text = Theme.of(context).textTheme;
    return Semantics(
      liveRegion: true,
      label: '${verdict.title}. $reason. ${verdict.advice}',
      child: Container(
        width: double.infinity,
        padding: const EdgeInsets.all(20),
        decoration: BoxDecoration(color: s.background, borderRadius: BorderRadius.circular(20)),
        child: ExcludeSemantics(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(children: [
                Icon(s.icon, color: s.foreground, size: 34),
                const SizedBox(width: 12),
                Expanded(
                  child: Text(verdict.title,
                      style: text.headlineSmall
                          ?.copyWith(color: s.foreground, fontWeight: FontWeight.w800)),
                ),
              ]),
              const SizedBox(height: 8),
              Text(reason, style: text.titleMedium?.copyWith(color: s.foreground)),
              const SizedBox(height: 6),
              Text(verdict.advice,
                  style: text.bodyLarge
                      ?.copyWith(color: s.foreground, fontWeight: FontWeight.w600)),
            ],
          ),
        ),
      ),
    );
  }
}

/// Shown on every result while the trained model is not connected.
class DemoBadge extends StatelessWidget {
  const DemoBadge({super.key});

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: scheme.tertiaryContainer,
        borderRadius: BorderRadius.circular(12),
      ),
      child: Row(children: [
        Icon(Icons.science_outlined, size: 18, color: scheme.onTertiaryContainer),
        const SizedBox(width: 8),
        Expanded(
          child: Text(
            'Demo result: the trained model is not connected yet, so this does not '
            'look at your photo. Never rely on it.',
            style: TextStyle(color: scheme.onTertiaryContainer, fontSize: 13),
          ),
        ),
      ]),
    );
  }
}

/// The red emergency button. Big, high contrast and always reachable.
class EmergencyButton extends StatelessWidget {
  const EmergencyButton({super.key, required this.label, required this.onPressed, this.icon});
  final String label;
  final VoidCallback onPressed;
  final IconData? icon;

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: double.infinity,
      height: 64,
      child: FilledButton.icon(
        style: FilledButton.styleFrom(
          backgroundColor: const Color(0xFFB3261E),
          foregroundColor: Colors.white,
          textStyle: const TextStyle(fontSize: 19, fontWeight: FontWeight.w800),
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(18)),
        ),
        onPressed: onPressed,
        icon: Icon(icon ?? Icons.sos_rounded, size: 28),
        label: Text(label),
      ),
    );
  }
}

class StatusPill extends StatelessWidget {
  const StatusPill({
    super.key,
    required this.icon,
    required this.label,
    required this.ok,
    this.onTap,
  });
  final IconData icon;
  final String label;
  final bool ok;
  final VoidCallback? onTap;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final bg = ok ? scheme.secondaryContainer : scheme.errorContainer;
    final fg = ok ? scheme.onSecondaryContainer : scheme.onErrorContainer;
    return Material(
      color: bg,
      borderRadius: BorderRadius.circular(40),
      child: InkWell(
        borderRadius: BorderRadius.circular(40),
        onTap: onTap,
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
          child: Row(mainAxisSize: MainAxisSize.min, children: [
            Icon(icon, size: 18, color: fg),
            const SizedBox(width: 6),
            Flexible(
              child: Text(label,
                  overflow: TextOverflow.ellipsis,
                  style: TextStyle(color: fg, fontWeight: FontWeight.w600, fontSize: 13)),
            ),
          ]),
        ),
      ),
    );
  }
}

class SectionTitle extends StatelessWidget {
  const SectionTitle(this.text, {super.key, this.trailing});
  final String text;
  final Widget? trailing;

  @override
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.fromLTRB(4, 20, 4, 8),
        child: Row(children: [
          Expanded(
            child: Text(text,
                style: Theme.of(context)
                    .textTheme
                    .titleMedium
                    ?.copyWith(fontWeight: FontWeight.w700)),
          ),
          ?trailing,
        ]),
      );
}
