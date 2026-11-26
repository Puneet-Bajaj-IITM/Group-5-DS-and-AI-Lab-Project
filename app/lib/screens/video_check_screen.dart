import 'dart:async';
import 'dart:io';
import 'dart:ui' as ui;

import 'package:material_ui/material_ui.dart';
import 'package:video_player/video_player.dart';

import '../app.dart';
import '../models/verdict.dart';
import '../services/classifier.dart';
import '../services/video_analyzer.dart';
import '../ui/brand.dart';
import '../widgets/common.dart';

/// Checks a video with the real model. While it runs, each frame the model
/// looks at is shown with the snake's box and the current answer, so the
/// check is visible as it happens. When it finishes the video plays back
/// with the same boxes and answers at each moment.
class VideoCheckScreen extends StatefulWidget {
  const VideoCheckScreen({super.key, required this.path, required this.classifier});
  final String path;
  final OnnxSnakeClassifier classifier;

  @override
  State<VideoCheckScreen> createState() => _VideoCheckScreenState();
}

class _VideoCheckScreenState extends State<VideoCheckScreen> {
  VideoAnalysis? _analysis;
  VideoPlayerController? _player;
  bool _playerFailed = false;
  int _done = 0, _total = 0;
  String? _error;
  StreamSubscription<VideoProgress>? _sub;

  // The frame being checked right now.
  ui.Image? _frame;
  VideoStep? _latest;
  bool _converting = false;

  @override
  void initState() {
    super.initState();
    _sub = VideoAnalyzer(widget.classifier).analyse(widget.path).listen(
      (p) {
        if (!mounted) return;
        setState(() {
          _done = p.done;
          _total = p.total;
          if (p.latest != null) _latest = p.latest;
        });
        if (p.frame != null) _showFrame(p.frame!);
        if (p.analysis != null) unawaited(_ready(p.analysis!));
      },
      onError: (Object e) {
        if (mounted) setState(() => _error = 'Could not read this video. ($e)');
      },
    );
  }

  /// Shows the checked frame. Refreshed at most twice a second: drawing
  /// every frame competed with the model for the CPU on a 4-core phone
  /// emulator, and two updates a second already reads as live.
  DateTime _lastShown = DateTime.fromMillisecondsSinceEpoch(0);

  Future<void> _showFrame(VideoFrameImage f) async {
    final now = DateTime.now();
    if (_converting || now.difference(_lastShown) < const Duration(milliseconds: 500)) return;
    _converting = true;
    _lastShown = now;
    try {
      final done = Completer<ui.Image>();
      ui.decodeImageFromPixels(f.rgba, f.width, f.height, ui.PixelFormat.rgba8888, done.complete);
      final img = await done.future;
      if (!mounted || _analysis != null) {
        img.dispose();
        return;
      }
      final old = _frame;
      setState(() => _frame = img);
      old?.dispose();
    } finally {
      _converting = false;
    }
  }

  Future<void> _ready(VideoAnalysis a) async {
    if (!mounted) return;
    setState(() => _analysis = a);
    final player = VideoPlayerController.file(File(widget.path));
    try {
      await player.initialize();
      await player.setLooping(true);
      player.addListener(_onTick);
      if (!mounted) {
        await player.dispose();
        return;
      }
      setState(() => _player = player);
      await player.play();
    } catch (_) {
      // The answer is already known; show it even if this phone cannot play
      // the file back.
      await player.dispose();
      if (mounted) setState(() => _playerFailed = true);
    }
  }

  void _onTick() {
    if (mounted) setState(() {});
  }

  @override
  void dispose() {
    _sub?.cancel();
    _player?.removeListener(_onTick);
    _player?.dispose();
    _frame?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final a = _analysis;
    return Scaffold(
      appBar: AppBar(title: const Text('Video check')),
      body: SafeArea(
        child: _error != null
            ? Padding(padding: const EdgeInsets.all(24), child: Text(_error!))
            : ListView(
                padding: const EdgeInsets.fromLTRB(16, 4, 16, 32),
                children: [
                  _viewer(context, a),
                  const SizedBox(height: 10),
                  if (a == null) _progressLine(context) else _Timeline(analysis: a, player: _player),
                  if (a != null) ..._summary(context, a),
                ],
              ),
      ),
    );
  }

  /// The big dark viewer: live frames while checking, then playback.
  Widget _viewer(BuildContext context, VideoAnalysis? a) {
    final player = _player;
    final playing = a != null && player != null;
    VideoStep? step;
    double aspect;
    Widget media;
    if (playing) {
      step = a.at(player.value.position.inMilliseconds / 1000);
      aspect = player.value.aspectRatio;
      media = GestureDetector(
        onTap: () => player.value.isPlaying ? player.pause() : player.play(),
        child: VideoPlayer(player),
      );
    } else {
      step = a?.steps.lastOrNull ?? _latest;
      final f = _frame;
      aspect = f == null ? 16 / 9 : f.width / f.height;
      media = f == null
          ? const Center(child: CircularProgressIndicator(color: Colors.white))
          : RawImage(image: f, fit: BoxFit.contain);
    }
    final shown = step?.shown;
    final style = shown == null ? null : VerdictStyle.of(shown.verdict);
    final box = step?.frame.gate.box;
    final frameSize = step == null ? null : Size(step.frame.width.toDouble(), step.frame.height.toDouble());

    return ClipRRect(
      borderRadius: BorderRadius.circular(22),
      child: Container(
        decoration: const BoxDecoration(gradient: Brand.night),
        child: AspectRatio(
          aspectRatio: aspect.clamp(0.5, 2.2).toDouble(),
          child: Stack(fit: StackFit.expand, children: [
            media,
            if (box != null && frameSize != null && step!.frame.gate.snake)
              SnakeBoxOverlay(
                box: box,
                imageSize: frameSize,
                color: style?.background ?? Brand.venom,
                // The banner above already names the species.
                label: null,
              ),
            Positioned(left: 10, right: 10, top: 10, child: _Hud(step: step, analysing: a == null)),
            if (a == null)
              Positioned(
                right: 10,
                bottom: 10,
                child: _chip('ANALYSING · $_done/${_total == 0 ? '…' : _total}', Brand.venom, Brand.forest950),
              ),
            if (playing && !player.value.isPlaying)
              const Center(child: Icon(Icons.play_circle_fill_rounded, size: 64, color: Colors.white70)),
          ]),
        ),
      ),
    );
  }

  Widget _progressLine(BuildContext context) {
    final step = _latest;
    final scheme = Theme.of(context).colorScheme;
    return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      ClipRRect(
        borderRadius: BorderRadius.circular(4),
        child: LinearProgressIndicator(value: _total == 0 ? null : _done / _total, minHeight: 6),
      ),
      const SizedBox(height: 8),
      Text(
        step == null
            ? 'Opening the video…'
            : 'At ${step.seconds.toStringAsFixed(1)} s · ${step.frame.gate.snake ? 'snake in view' : 'no snake in this frame'}'
                ' · ${step.frame.totalMs} ms per frame',
        style: TextStyle(color: scheme.onSurfaceVariant, fontSize: 13),
      ),
      const SizedBox(height: 6),
      Text('Up to 10 frames a second, the first 20 seconds. Everything runs on this phone.',
          style: TextStyle(color: scheme.onSurfaceVariant, fontSize: 12.5)),
    ]);
  }

  List<Widget> _summary(BuildContext context, VideoAnalysis a) {
    final finalId = a.finalShown;
    return [
      if (_playerFailed)
        const Padding(
          padding: EdgeInsets.only(top: 8),
          child: Text('This phone cannot play this video back, but the check below is complete.'),
        ),
      const SectionTitle('Answer at the end of the clip'),
      if (finalId == null)
        const Text('Still checking at the end: the frames never agreed on one snake, or there was no snake. '
            'Keep away anyway.')
      else ...[
        VerdictBanner(verdict: finalId.verdict, reason: _reason(finalId)),
        const SizedBox(height: 12),
        if (finalId.verdict.escalate)
          EmergencyButton(
            label: 'SOMEONE WAS BITTEN',
            onPressed: () => openSos(context, bitten: true, identification: finalId),
          ),
      ],
      const SectionTitle('How the check went'),
      Card(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
          child: Column(children: [
            _Stat('Frames checked', '${a.steps.length} · ${a.analysedSeconds.toStringAsFixed(1)} of ${a.durationSeconds.toStringAsFixed(1)} s'),
            _Stat('Frames with a snake', '${a.snakeFrames}'),
            _Stat('First answer', a.firstAnswerSeconds == null ? 'none' : 'at ${a.firstAnswerSeconds!.toStringAsFixed(1)} s'),
            _Stat('Changes after the first answer', '${a.changes}'),
            _Stat('Time per frame on this phone', '${a.meanFrameMs.toStringAsFixed(0)} ms'),
          ]),
        ),
      ),
      const SizedBox(height: 12),
      Text(
        'The answer pools the last 5 frames with a snake and settles on one species once it is sure; '
        '"Dangerous" can show before that. SerpentGuard can be wrong. Keep away from any snake.',
        style: TextStyle(color: Theme.of(context).colorScheme.onSurfaceVariant, fontSize: 12.5),
      ),
    ];
  }

  static String _reason(Identification id) {
    final s = id.topSpecies;
    final fb = id.genusFallback;
    final extra = id.verdict == Verdict.unknown && fb != null
        ? ' · likely genus $fb'
        : (s != null && id.verdict != Verdict.notListed ? ' · ${s.displayName}' : '');
    return '${id.reason}$extra';
  }
}

Widget _chip(String text, Color bg, Color fg) => Container(
      padding: const EdgeInsets.symmetric(horizontal: 9, vertical: 4),
      decoration: BoxDecoration(color: bg, borderRadius: BorderRadius.circular(8)),
      child: Text(text, style: Brand.label(fg).copyWith(fontSize: 12)),
    );

/// The answer banner over the video, like a camera HUD.
class _Hud extends StatelessWidget {
  const _Hud({required this.step, required this.analysing});
  final VideoStep? step;
  final bool analysing;

  @override
  Widget build(BuildContext context) {
    final s = step;
    if (s == null) return const SizedBox.shrink();
    final shown = s.shown;
    final style = shown == null ? null : VerdictStyle.of(shown.verdict);
    final text = shown == null
        ? (s.frame.gate.snake ? 'Checking…' : 'Looking for a snake…')
        : '${shown.verdict.title} · ${shown.topSpecies?.displayName ?? shown.reason}';
    return Align(
      alignment: Alignment.topLeft,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 7),
        decoration: BoxDecoration(
          color: style?.background.withValues(alpha: 0.94) ?? Colors.black.withValues(alpha: 0.55),
          borderRadius: BorderRadius.circular(12),
        ),
        child: Row(mainAxisSize: MainAxisSize.min, children: [
          Icon(style?.icon ?? Icons.radar_rounded, color: style?.foreground ?? Colors.white, size: 18),
          const SizedBox(width: 6),
          Flexible(
            child: Text(text,
                maxLines: 2,
                style: TextStyle(color: style?.foreground ?? Colors.white, fontWeight: FontWeight.w700, fontSize: 14.5)),
          ),
          if (s.lock.lockedClass != null) ...[
            const SizedBox(width: 6),
            Icon(Icons.lock_rounded, size: 15, color: style?.foreground ?? Colors.white),
          ],
        ]),
      ),
    );
  }
}

class _Stat extends StatelessWidget {
  const _Stat(this.label, this.value);
  final String label, value;

  @override
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.symmetric(vertical: 6),
        child: Row(children: [
          Expanded(child: Text(label)),
          Text(value, style: const TextStyle(fontWeight: FontWeight.w700)),
        ]),
      );
}

/// A strip coloured by what the screen showed over the clip; tap to jump.
class _Timeline extends StatelessWidget {
  const _Timeline({required this.analysis, required this.player});
  final VideoAnalysis analysis;
  final VideoPlayerController? player;

  @override
  Widget build(BuildContext context) {
    final total = analysis.analysedSeconds <= 0 ? 1.0 : analysis.analysedSeconds;
    final scheme = Theme.of(context).colorScheme;
    final position = (player?.value.position.inMilliseconds ?? 0) / 1000;
    return LayoutBuilder(builder: (context, box) {
      final w = box.maxWidth;
      return GestureDetector(
        onTapDown: (d) => player?.seekTo(Duration(milliseconds: (d.localPosition.dx / w * total * 1000).round())),
        child: SizedBox(
          height: 22,
          child: Stack(children: [
            Positioned.fill(
              top: 5,
              bottom: 5,
              child: DecoratedBox(
                decoration: BoxDecoration(color: scheme.surfaceContainerHigh, borderRadius: BorderRadius.circular(6)),
              ),
            ),
            for (var i = 0; i < analysis.steps.length; i++)
              () {
                final s = analysis.steps[i];
                if (s.shown == null) return const SizedBox.shrink();
                final end = i + 1 < analysis.steps.length ? analysis.steps[i + 1].seconds : total;
                return Positioned(
                  left: s.seconds / total * w,
                  width: ((end - s.seconds) / total * w).clamp(1, w).toDouble(),
                  top: 5,
                  bottom: 5,
                  child: ColoredBox(color: VerdictStyle.of(s.shown!.verdict).background),
                );
              }(),
            Positioned(
              left: (position.clamp(0, total) / total * w - 1.5).clamp(0, w - 3).toDouble(),
              width: 3,
              top: 0,
              bottom: 0,
              child: DecoratedBox(
                decoration: BoxDecoration(color: scheme.onSurface, borderRadius: BorderRadius.circular(2)),
              ),
            ),
          ]),
        ),
      );
    });
  }
}
