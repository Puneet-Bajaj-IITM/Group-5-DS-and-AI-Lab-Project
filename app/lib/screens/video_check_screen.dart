import 'dart:async';
import 'dart:io';

import 'package:material_ui/material_ui.dart';
import 'package:video_player/video_player.dart';

import '../app.dart';
import '../models/verdict.dart';
import '../services/classifier.dart';
import '../services/video_analyzer.dart';
import '../widgets/common.dart';

/// Runs a video through the real model (up to 10 frames a second, with the
/// M4 video lock), then plays it back with the answer the screen would have
/// shown at each moment.
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
  int _done = 0, _total = 0;
  String? _error;
  StreamSubscription<VideoProgress>? _sub;

  @override
  void initState() {
    super.initState();
    _sub = VideoAnalyzer(widget.classifier).analyse(widget.path).listen(
      (p) {
        if (!mounted) return;
        setState(() {
          _done = p.done;
          _total = p.total;
        });
        if (p.analysis != null) _ready(p.analysis!);
      },
      onError: (Object e) {
        if (mounted) setState(() => _error = 'Could not read this video: $e');
      },
    );
  }

  Future<void> _ready(VideoAnalysis a) async {
    final player = VideoPlayerController.file(File(widget.path));
    await player.initialize();
    await player.setLooping(true);
    player.addListener(() {
      if (mounted) setState(() {});
    });
    if (!mounted) {
      await player.dispose();
      return;
    }
    setState(() {
      _analysis = a;
      _player = player;
    });
    await player.play();
  }

  @override
  void dispose() {
    _sub?.cancel();
    _player?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final a = _analysis, player = _player;
    return Scaffold(
      appBar: AppBar(title: const Text('Video check')),
      body: SafeArea(
        child: _error != null
            ? Padding(padding: const EdgeInsets.all(24), child: Text(_error!))
            : a == null || player == null
                ? _progress(context)
                : _results(context, a, player),
      ),
    );
  }

  Widget _progress(BuildContext context) => Padding(
        padding: const EdgeInsets.all(24),
        child: Column(mainAxisAlignment: MainAxisAlignment.center, children: [
          const Text('Checking the video frame by frame…', style: TextStyle(fontSize: 18)),
          const SizedBox(height: 16),
          LinearProgressIndicator(value: _total == 0 ? null : _done / _total),
          const SizedBox(height: 8),
          Text(_total == 0 ? 'Opening the video' : '$_done of $_total frames'),
          const SizedBox(height: 24),
          const Text('Up to 10 frames a second, the first 20 seconds. Everything runs on this phone.',
              textAlign: TextAlign.center),
        ]),
      );

  Widget _results(BuildContext context, VideoAnalysis a, VideoPlayerController player) {
    final now = player.value.position.inMilliseconds / 1000;
    final step = a.at(now);
    final shown = step?.shown;
    final scheme = Theme.of(context).colorScheme;
    final finalId = a.finalShown;
    return ListView(
      padding: const EdgeInsets.fromLTRB(16, 8, 16, 32),
      children: [
        ClipRRect(
          borderRadius: BorderRadius.circular(16),
          child: AspectRatio(
            aspectRatio: player.value.aspectRatio,
            child: Stack(fit: StackFit.expand, children: [
              GestureDetector(
                onTap: () => player.value.isPlaying ? player.pause() : player.play(),
                child: VideoPlayer(player),
              ),
              Positioned(left: 8, right: 8, top: 8, child: _overlay(step, shown)),
            ]),
          ),
        ),
        const SizedBox(height: 8),
        _Timeline(analysis: a, position: now, onSeek: (s) => player.seekTo(Duration(milliseconds: (s * 1000).round()))),
        const SizedBox(height: 4),
        Text(
          step == null
              ? ''
              : 'This frame: ${step.frame.gate.snake ? 'snake found' : 'no snake'}'
                  ' · box ${step.frame.gate.boxConf?.toStringAsFixed(2) ?? '-'}'
                  ' · mask ${(step.frame.gate.maskArea * 100).toStringAsFixed(1)}%'
                  ' · ${step.frame.totalMs} ms',
          style: TextStyle(color: scheme.onSurfaceVariant, fontSize: 12),
        ),
        const SectionTitle('Answer at the end of the clip'),
        if (finalId == null)
          const Text("Still checking at the end: the frames never agreed on one snake, or there was no snake. "
              "Keep away anyway.")
        else ...[
          VerdictBanner(verdict: finalId.verdict, reason: _reason(finalId)),
          const SizedBox(height: 12),
          if (finalId.verdict.escalate)
            EmergencyButton(
              label: 'Someone was bitten: get help',
              onPressed: () => openSos(context, bitten: true, identification: finalId),
            ),
        ],
        const SectionTitle('How the check went'),
        _Stat('Frames checked', '${a.steps.length} (${a.analysedSeconds.toStringAsFixed(1)} s of ${a.durationSeconds.toStringAsFixed(1)} s)'),
        _Stat('Frames with a snake', '${a.snakeFrames}'),
        _Stat('First answer', a.firstAnswerSeconds == null ? 'none' : 'at ${a.firstAnswerSeconds!.toStringAsFixed(1)} s'),
        _Stat('Changes after the first answer', '${a.changes}'),
        _Stat('Time per frame on this phone', '${a.meanFrameMs.toStringAsFixed(0)} ms'),
        const SizedBox(height: 12),
        Text(
          'The answer pools the last 5 frames with a snake and settles on one species once it is sure; '
          '"Dangerous" can show before that. SerpentGuard can be wrong. Keep away from any snake.',
          style: TextStyle(color: scheme.onSurfaceVariant, fontSize: 12),
        ),
      ],
    );
  }

  static String _reason(Identification id) {
    final s = id.topSpecies;
    final fb = id.genusFallback;
    final extra = id.verdict == Verdict.unknown && fb != null
        ? ' · likely genus $fb'
        : (s != null && id.verdict != Verdict.notListed ? ' · ${s.displayName}' : '');
    return '${id.reason}$extra';
  }

  Widget _overlay(VideoStep? step, Identification? shown) {
    if (step == null) return const SizedBox.shrink();
    final style = shown == null ? null : VerdictStyle.of(shown.verdict);
    final text = shown == null
        ? (step.frame.gate.snake ? 'Checking…' : 'Looking for a snake…')
        : '${shown.verdict.title} · ${shown.topSpecies?.displayName ?? shown.reason}';
    return Align(
      alignment: Alignment.topLeft,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
        decoration: BoxDecoration(
          color: style?.background.withValues(alpha: 0.92) ?? Colors.black54,
          borderRadius: BorderRadius.circular(10),
        ),
        child: Row(mainAxisSize: MainAxisSize.min, children: [
          Icon(style?.icon ?? Icons.hourglass_top_rounded, color: style?.foreground ?? Colors.white, size: 18),
          const SizedBox(width: 6),
          Flexible(
            child: Text(text,
                style: TextStyle(color: style?.foreground ?? Colors.white, fontWeight: FontWeight.w700)),
          ),
          if (step.lock.lockedClass != null) ...[
            const SizedBox(width: 6),
            Icon(Icons.lock_rounded, size: 14, color: style?.foreground ?? Colors.white),
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
        padding: const EdgeInsets.symmetric(vertical: 3),
        child: Row(children: [
          Expanded(child: Text(label)),
          Text(value, style: const TextStyle(fontWeight: FontWeight.w600)),
        ]),
      );
}

/// A strip coloured by what the screen showed over the clip.
class _Timeline extends StatelessWidget {
  const _Timeline({required this.analysis, required this.position, required this.onSeek});
  final VideoAnalysis analysis;
  final double position;
  final ValueChanged<double> onSeek;

  @override
  Widget build(BuildContext context) {
    final total = analysis.analysedSeconds <= 0 ? 1.0 : analysis.analysedSeconds;
    final scheme = Theme.of(context).colorScheme;
    return LayoutBuilder(builder: (context, box) {
      final w = box.maxWidth;
      return GestureDetector(
        onTapDown: (d) => onSeek(d.localPosition.dx / w * total),
        child: SizedBox(
          height: 18,
          child: Stack(children: [
            for (var i = 0; i < analysis.steps.length; i++)
              () {
                final s = analysis.steps[i];
                final end = i + 1 < analysis.steps.length ? analysis.steps[i + 1].seconds : total;
                final color = s.shown == null
                    ? scheme.surfaceContainerHighest
                    : VerdictStyle.of(s.shown!.verdict).background;
                return Positioned(
                  left: s.seconds / total * w,
                  width: ((end - s.seconds) / total * w).clamp(1, w),
                  top: 4,
                  bottom: 4,
                  child: ColoredBox(color: color),
                );
              }(),
            Positioned(
              left: (position.clamp(0, total) / total * w - 1).clamp(0, w - 2),
              width: 2,
              top: 0,
              bottom: 0,
              child: ColoredBox(color: scheme.onSurface),
            ),
          ]),
        ),
      );
    });
  }
}
