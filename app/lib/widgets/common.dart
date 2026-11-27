import 'package:material_ui/material_ui.dart';

import '../inference/image_ops.dart' show PixelBox;
import '../models/verdict.dart';
import '../ui/brand.dart';

/// Colours that mean the same thing on every screen.
class VerdictStyle {
  const VerdictStyle(this.background, this.foreground, this.icon, this.soft, this.tag);
  final Color background;
  final Color foreground;
  final IconData icon;

  /// A pale tint of [background] for chips and timelines.
  final Color soft;

  /// A one-word label in signage style.
  final String tag;

  static VerdictStyle of(Verdict v) => switch (v) {
        Verdict.dangerous =>
          const VerdictStyle(Brand.danger, Colors.white, Icons.warning_amber_rounded, Brand.dangerSoft, 'DANGER'),
        Verdict.unknown =>
          const VerdictStyle(Brand.caution, Brand.forest950, Icons.help_outline_rounded, Brand.cautionSoft, 'UNSURE'),
        Verdict.notListed =>
          const VerdictStyle(Brand.calm, Colors.white, Icons.visibility_rounded, Brand.calmSoft, 'CAUTION'),
      };
}

/// The verdict card: a coloured sign with the answer, its reason and what to do.
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
      child: ExcludeSemantics(
        child: Container(
          width: double.infinity,
          decoration: BoxDecoration(
            color: s.background,
            borderRadius: BorderRadius.circular(22),
            boxShadow: [BoxShadow(color: s.background.withValues(alpha: 0.28), blurRadius: 18, offset: const Offset(0, 8))],
          ),
          child: Stack(children: [
            Positioned(
              right: -18,
              top: -18,
              child: Icon(s.icon, size: 120, color: s.foreground.withValues(alpha: 0.10)),
            ),
            Padding(
              padding: const EdgeInsets.fromLTRB(20, 18, 20, 20),
              child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                Row(children: [
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                    decoration: BoxDecoration(
                      color: s.foreground.withValues(alpha: 0.16),
                      borderRadius: BorderRadius.circular(6),
                    ),
                    child: Text(s.tag, style: Brand.label(s.foreground)),
                  ),
                ]),
                const SizedBox(height: 10),
                Row(children: [
                  Icon(s.icon, color: s.foreground, size: 34),
                  const SizedBox(width: 10),
                  Expanded(
                    child: Text(verdict.title, style: text.headlineMedium?.copyWith(color: s.foreground)),
                  ),
                ]),
                const SizedBox(height: 8),
                Text(reason, style: text.titleMedium?.copyWith(color: s.foreground)),
                const SizedBox(height: 10),
                Container(height: 1, color: s.foreground.withValues(alpha: 0.22)),
                const SizedBox(height: 10),
                Text(verdict.advice,
                    style: text.bodyLarge?.copyWith(color: s.foreground, fontWeight: FontWeight.w600)),
              ]),
            ),
          ]),
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
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
      decoration: BoxDecoration(
        color: scheme.tertiaryContainer,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: Brand.venom),
      ),
      child: Row(children: [
        Icon(Icons.science_outlined, size: 20, color: scheme.onTertiaryContainer),
        const SizedBox(width: 10),
        Expanded(
          child: Text(
            'Demo result: the model files are not in this build, so this does not '
            'look at your photo. Never rely on it.',
            style: TextStyle(color: scheme.onTertiaryContainer, fontSize: 13.5, fontWeight: FontWeight.w600),
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
      height: 62,
      child: FilledButton.icon(
        style: FilledButton.styleFrom(
          backgroundColor: Brand.danger,
          foregroundColor: Colors.white,
          textStyle: const TextStyle(fontFamily: Brand.display, fontSize: 22, fontWeight: FontWeight.w800, letterSpacing: 0.4),
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(18)),
          elevation: 2,
          shadowColor: Brand.danger.withValues(alpha: 0.4),
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
    this.onDark = false,
  });
  final IconData icon;
  final String label;
  final bool ok;
  final VoidCallback? onTap;

  /// Use the light-on-dark style (inside the home header).
  final bool onDark;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final Color bg, fg;
    if (onDark) {
      bg = Colors.white.withValues(alpha: ok ? 0.12 : 0.08);
      fg = ok ? Colors.white : Brand.venom;
    } else {
      bg = ok ? scheme.primaryContainer : scheme.errorContainer;
      fg = ok ? scheme.onPrimaryContainer : scheme.onErrorContainer;
    }
    return Material(
      color: bg,
      borderRadius: BorderRadius.circular(40),
      child: InkWell(
        borderRadius: BorderRadius.circular(40),
        onTap: onTap,
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
          child: Row(mainAxisSize: MainAxisSize.min, children: [
            Icon(icon, size: 17, color: fg),
            const SizedBox(width: 6),
            Flexible(
              child: Text(label,
                  overflow: TextOverflow.ellipsis,
                  maxLines: 1,
                  style: TextStyle(color: fg, fontWeight: FontWeight.w600, fontSize: 13.5)),
            ),
          ]),
        ),
      ),
    );
  }
}

/// A section heading in condensed capitals, like a field-guide divider.
class SectionTitle extends StatelessWidget {
  const SectionTitle(this.text, {super.key, this.trailing});
  final String text;
  final Widget? trailing;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Padding(
      padding: const EdgeInsets.fromLTRB(2, 22, 2, 10),
      child: Row(children: [
        Container(width: 4, height: 18, decoration: BoxDecoration(color: Brand.venom, borderRadius: BorderRadius.circular(2))),
        const SizedBox(width: 8),
        Expanded(
          child: Text(
            text.toUpperCase(),
            style: TextStyle(
              fontFamily: Brand.display,
              fontWeight: FontWeight.w800,
              fontSize: 17,
              letterSpacing: 1.1,
              color: scheme.onSurface,
            ),
          ),
        ),
        ?trailing,
      ]),
    );
  }
}

/// Draws the detector's box over an image shown with BoxFit.contain or
/// cover. [imageSize] is the image's own pixel size.
class SnakeBoxOverlay extends StatelessWidget {
  const SnakeBoxOverlay({
    super.key,
    required this.box,
    required this.imageSize,
    required this.color,
    this.label,
    this.fit = BoxFit.contain,
  });
  final PixelBox box;
  final Size imageSize;
  final Color color;
  final String? label;
  final BoxFit fit;

  @override
  Widget build(BuildContext context) => SnakeBoxesOverlay(
    boxes: [OverlayBox(box, color, label: label, main: true)],
    imageSize: imageSize,
    fit: fit,
  );
}

/// One box to draw. The [main] box gets the shade around it and the bold
/// frame; the others a thinner frame.
class OverlayBox {
  const OverlayBox(this.box, this.color, {this.label, this.main = false});
  final PixelBox box;
  final Color color;
  final String? label;
  final bool main;

  @override
  bool operator ==(Object other) =>
      other is OverlayBox &&
      other.color == color &&
      other.label == label &&
      other.main == main &&
      other.box.left == box.left &&
      other.box.top == box.top &&
      other.box.right == box.right &&
      other.box.bottom == box.bottom;

  @override
  int get hashCode => Object.hash(box.left, box.top, box.right, box.bottom, color, label, main);
}

/// Draws several detector boxes (the live camera draws every snake it sees).
class SnakeBoxesOverlay extends StatelessWidget {
  const SnakeBoxesOverlay({super.key, required this.boxes, required this.imageSize, this.fit = BoxFit.contain});
  final List<OverlayBox> boxes;
  final Size imageSize;
  final BoxFit fit;

  @override
  Widget build(BuildContext context) => IgnorePointer(
    child: CustomPaint(size: Size.infinite, painter: _BoxPainter(boxes, imageSize, fit)),
  );
}

class _BoxPainter extends CustomPainter {
  _BoxPainter(this.boxes, this.imageSize, this.fit);
  final List<OverlayBox> boxes;
  final Size imageSize;
  final BoxFit fit;

  @override
  void paint(Canvas canvas, Size size) {
    if (imageSize.isEmpty || boxes.isEmpty) return;
    final fitted = applyBoxFit(fit, imageSize, size);
    final dest = Alignment.center.inscribe(fitted.destination, Offset.zero & size);
    final sx = dest.width / imageSize.width, sy = dest.height / imageSize.height;
    Rect place(PixelBox b) => Rect.fromLTRB(
      dest.left + b.left * sx,
      dest.top + b.top * sy,
      dest.left + b.right * sx,
      dest.top + b.bottom * sy,
    ).intersect(Offset.zero & size);
    canvas.save();
    canvas.clipRect(Offset.zero & size);
    // Soft shade outside the main box.
    for (final b in boxes.where((b) => b.main)) {
      final r = place(b.box);
      if (r.isEmpty) continue;
      final shade = Path()
        ..fillType = PathFillType.evenOdd
        ..addRect(Offset.zero & size)
        ..addRRect(RRect.fromRectAndRadius(r, const Radius.circular(10)));
      canvas.drawPath(shade, Paint()..color = Colors.black.withValues(alpha: 0.22));
    }
    // The others first, so the main box and its label are on top.
    for (final b in [...boxes.where((b) => !b.main), ...boxes.where((b) => b.main)]) {
      final r = place(b.box);
      if (r.isEmpty) continue;
      _frame(canvas, r, b);
      if (b.label != null) _label(canvas, size, r, b);
    }
    canvas.restore();
  }

  void _frame(Canvas canvas, Rect r, OverlayBox b) {
    canvas.drawRRect(
      RRect.fromRectAndRadius(r, const Radius.circular(10)),
      Paint()
        ..color = b.color
        ..style = PaintingStyle.stroke
        ..strokeWidth = b.main ? 3 : 2,
    );
    final tick = (r.shortestSide * 0.18).clamp(10.0, 28.0);
    final p = Paint()
      ..color = Colors.white.withValues(alpha: b.main ? 1 : 0.7)
      ..style = PaintingStyle.stroke
      ..strokeWidth = b.main ? 4 : 2.5
      ..strokeCap = StrokeCap.round;
    for (final (o, dx, dy) in [
      (r.topLeft, 1.0, 1.0),
      (r.topRight, -1.0, 1.0),
      (r.bottomLeft, 1.0, -1.0),
      (r.bottomRight, -1.0, -1.0),
    ]) {
      canvas.drawLine(o, o + Offset(tick * dx, 0), p);
      canvas.drawLine(o, o + Offset(0, tick * dy), p);
    }
  }

  void _label(Canvas canvas, Size size, Rect r, OverlayBox b) {
    final tp = TextPainter(
      text: TextSpan(
        text: b.label,
        style: TextStyle(
          fontFamily: Brand.display,
          fontWeight: FontWeight.w800,
          fontSize: b.main ? 15 : 13,
          color: Colors.white,
          letterSpacing: 0.4,
        ),
      ),
      textDirection: TextDirection.ltr,
      maxLines: 1,
      ellipsis: '…',
    )..layout(maxWidth: size.width - 16);
    final pad = const EdgeInsets.symmetric(horizontal: 8, vertical: 4);
    var top = r.top - tp.height - pad.vertical - 4;
    if (top < 4) top = r.top + 6;
    final left = r.left.clamp(4.0, size.width - tp.width - pad.horizontal - 4);
    final bg = RRect.fromRectAndRadius(
      Rect.fromLTWH(left, top, tp.width + pad.horizontal, tp.height + pad.vertical),
      const Radius.circular(8),
    );
    canvas.drawRRect(bg, Paint()..color = b.color);
    tp.paint(canvas, Offset(left + pad.left, top + pad.top));
  }

  @override
  bool shouldRepaint(_BoxPainter old) => old.imageSize != imageSize || old.fit != fit || !_same(old.boxes, boxes);

  static bool _same(List<OverlayBox> a, List<OverlayBox> b) {
    if (a.length != b.length) return false;
    for (var i = 0; i < a.length; i++) {
      if (a[i] != b[i]) return false;
    }
    return true;
  }
}
