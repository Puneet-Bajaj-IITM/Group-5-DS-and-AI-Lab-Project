import 'dart:async';

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:material_ui/material_ui.dart';
import 'package:provider/provider.dart';

import '../services/voting_queue.dart';
import '../services/classifier.dart';
import '../widgets/common.dart';
import '../ui/brand.dart';
import '../inference/detections.dart' show Track;
import '../inference/live_session.dart';
import '../inference/pipeline.dart';
import '../models/verdict.dart';
import '../state/app_state.dart';

class LiveEvidence {
  LiveEvidence(this.jpegs, this.offsets, this.started);
  final List<Uint8List> jpegs;
  final List<int> offsets;
  final DateTime started;
}

/// Live preview with the model checking frames continuously (box drawn over
/// the preview, answer steadied by the video lock) and explicit still capture
/// for community review.
class LiveReviewScreen extends StatefulWidget {
  const LiveReviewScreen({super.key});
  @override
  State<LiveReviewScreen> createState() => _LiveReviewScreenState();
}

class _LiveReviewScreenState extends State<LiveReviewScreen>
    with WidgetsBindingObserver {
  CameraController? _camera;
  final _frames = <Uint8List>[];
  final _offsets = <int>[];
  final _started = DateTime.now();
  final _clock = Stopwatch()..start();
  bool _busy = false;
  bool _opening = false;
  int _generation = 0;
  String? _error;
  Identification? _identification;
  PipelineResult? _lastFrame;
  bool _analyzing = false;
  LiveSnakeClassifier? _classifier;

  /// The two-lane live session (boxes on every frame the detector can take,
  /// verdict on the newest frame); null falls back to identifyFrame.
  LiveSession? _live;
  StreamSubscription<LiveView>? _liveViews;
  LiveView? _view;
  bool _diagnostics = false;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    unawaited(_open());
  }

  Future<void> _open() async {
    if (_opening || !mounted) return;
    _opening = true;
    final generation = ++_generation;
    final classifier = context.read<AppState>().classifier;
    _classifier = classifier is LiveSnakeClassifier && !classifier.isDemo
        ? classifier
        : null;
    CameraController? camera;
    try {
      final cameras = await availableCameras();
      if (cameras.isEmpty) throw StateError('No camera available');
      final rear = cameras.where(
        (c) => c.lensDirection == CameraLensDirection.back,
      );
      camera = CameraController(
        rear.isEmpty ? cameras.first : rear.first,
        ResolutionPreset.high,
        enableAudio: false,
      );
      await camera.initialize();
      if (!mounted || generation != _generation) {
        await camera.dispose();
        return;
      }
      setState(() {
        _camera = camera;
        _error = null;
      });
      if (_classifier != null) {
        await _classifier!.resetLiveSession();
        final c = _classifier;
        if (_live == null && c is OnnxSnakeClassifier) {
          // Opening copies and checks four models, which can take seconds; if
          // the screen closed meanwhile, close the session it no longer owns.
          final opened = await LiveSession.open(c);
          // Closed or sent to the background meanwhile: let go of the models
          // (a resume opens them again, quickly: the files are checked once).
          if (!mounted || generation != _generation) {
            await opened?.close();
            return;
          }
          _live = opened;
          _liveViews = opened?.views.listen(
            (v) {
              // The view says whether a lane is failing; a good view clears it.
              if (mounted) {
                setState(() {
                  _view = v;
                  _error = v.error == null ? null : 'Live analysis unavailable. Keep away and use community review.';
                });
              }
            },
            onError: (Object _) {
              if (mounted) {
                setState(() => _error = 'Live analysis unavailable. Keep away and use community review.');
              }
            },
          );
        }
        _live?.reset();
        _view = null;
        if (!mounted || generation != _generation) return;
        await camera.startImageStream(
          (image) => _live != null
              ? _offer(image, camera!, generation)
              : _analyze(image, camera!, generation),
        );
      }
    } catch (_) {
      await camera?.dispose();
      // Never leave the disposed controller on screen or behind Capture.
      if (mounted && identical(_camera, camera)) setState(() => _camera = null);
      if (mounted && generation == _generation) {
        setState(
          () => _error = 'Camera unavailable. Check permission or use a photo.',
        );
      }
    } finally {
      _opening = false;
      if (mounted &&
          generation != _generation &&
          _camera == null &&
          WidgetsBinding.instance.lifecycleState == AppLifecycleState.resumed) {
        unawaited(_open());
      }
    }
  }

  void _offer(CameraImage image, CameraController camera, int generation) {
    final live = _live;
    if (live == null || _busy || !mounted || generation != _generation) return;
    live.offer(_frameOf(image, camera, copy: false));
  }

  LiveCameraFrame _frameOf(CameraImage image, CameraController camera, {required bool copy}) => LiveCameraFrame(
        width: image.width,
        height: image.height,
        format: image.format.group.name,
        offsetMs: _clock.elapsedMilliseconds,
        sensorOrientation: camera.description.sensorOrientation,
        deviceOrientation: camera.value.deviceOrientation.name,
        frontFacing: camera.description.lensDirection == CameraLensDirection.front,
        planes: image.planes
            .map((p) => LiveCameraPlane(copy ? Uint8List.fromList(p.bytes) : p.bytes, p.bytesPerRow, p.bytesPerPixel))
            .toList(),
      );

  Future<void> _analyze(
    CameraImage image,
    CameraController camera,
    int generation,
  ) async {
    final classifier = _classifier;
    if (classifier == null ||
        _analyzing ||
        _busy ||
        !mounted ||
        generation != _generation) {
      return;
    }
    _analyzing = true;
    try {
      final result = await classifier.identifyFrame(_frameOf(image, camera, copy: true));
      if (mounted && generation == _generation) {
        setState(() {
          _identification = result;
          final c = _classifier;
          _lastFrame = c is OnnxSnakeClassifier ? c.lastFrame : null;
          _error = null;
        });
      }
    } catch (_) {
      if (mounted && generation == _generation) {
        setState(() {
          _identification = null;
          _error =
              'Live analysis unavailable. Keep away and use community review.';
        });
      }
    } finally {
      _analyzing = false;
    }
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.inactive ||
        state == AppLifecycleState.paused) {
      _generation++;
      _identification = null;
      _live?.reset();
      _view = null;
      final camera = _camera;
      _camera = null;
      unawaited(camera?.dispose());
      if (mounted) setState(() {});
    } else if (state == AppLifecycleState.resumed) {
      unawaited(_open());
    }
  }

  Future<void> _capture() async {
    final camera = _camera;
    if (_busy ||
        camera == null ||
        !camera.value.isInitialized ||
        _frames.length >= 3) {
      return;
    }
    setState(() => _busy = true);
    try {
      final offset = _clock.elapsedMilliseconds;
      final picture = await camera.takePicture();
      final jpeg = await compute(votingJpeg, await picture.readAsBytes());
      if (mounted) {
        setState(() {
          _frames.add(jpeg);
          _offsets.add(offset);
          _error = null;
        });
      }
    } catch (_) {
      if (mounted) {
        setState(
          () => _error =
              'Could not capture this view. Your earlier stills are kept.',
        );
      }
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) => Scaffold(
    appBar: AppBar(title: const Text('Live camera')),
    body: SafeArea(
      child: Column(
        children: [
          const Padding(
            padding: EdgeInsets.all(12),
            child: Text(
              'Stay away. Capture 1–3 clear views of the same snake. '
              'Keep the preview on the same encounter.',
            ),
          ),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 12),
            child: _live != null
                ? GestureDetector(
                    onLongPress: () => setState(() => _diagnostics = !_diagnostics),
                    child: _LiveStatus(text: _liveText(_view), verdict: _liveVerdict(_view)),
                  )
                : _LiveStatus(
              text: _classifier == null
                  ? 'Live model not connected in this build. Preview and evidence capture are available.'
                  : _identification == null
                  ? (_lastFrame?.gate.snake == true ? snakeInViewIdentifying : 'Looking for a snake…')
                  : !_identification!.snakeFound
                  ? 'No snake detected. This does not mean the area is safe.'
                  : '${_identification!.verdict.title} · ${_identification!.topSpecies?.displayName ?? _identification!.reason}',
              verdict: _identification?.snakeFound == true ? _identification!.verdict : null,
            ),
          ),
          if (_live != null && _diagnostics && _view != null)
            Padding(
              padding: const EdgeInsets.fromLTRB(12, 6, 12, 0),
              child: Text(
                'Boxes ${_view!.boxFps.toStringAsFixed(1)}/s (${_view!.boxMs.round()} ms) · '
                'verdict ${_view!.verdictFps.toStringAsFixed(1)}/s (${_view!.verdictMs.round()} ms) · '
                'threads ${_view!.threads.$1}+${_view!.threads.$2} · skipped ${_view!.skippedByGate}',
                style: const TextStyle(fontSize: 12, fontFeatures: [FontFeature.tabularFigures()]),
              ),
            ),
          const SizedBox(height: 8),
          Expanded(
            child: _camera?.value.isInitialized == true
                ? Center(
                    child: AspectRatio(
                      aspectRatio: _camera!.value.aspectRatio,
                      child: CameraPreview(
                        _camera!,
                        child: _live != null
                            ? _liveBoxes(_view)
                            : _lastFrame?.gate.snake == true && _lastFrame!.gate.box != null
                            ? SnakeBoxOverlay(
                                box: _lastFrame!.gate.box!,
                                imageSize: Size(_lastFrame!.width.toDouble(), _lastFrame!.height.toDouble()),
                                fit: BoxFit.fill,
                                color: _identification?.snakeFound == true
                                    ? VerdictStyle.of(_identification!.verdict).background
                                    : Brand.venom,
                                label: _identification?.snakeFound == true
                                    ? _identification!.topSpecies?.displayName
                                    : identifyingText,
                              )
                            : null,
                      ),
                    ),
                  )
                : Center(child: Text(_error ?? 'Opening camera…')),
          ),
          if (_error != null) Text(_error!),
          SizedBox(
            height: 90,
            child: ListView(
              scrollDirection: Axis.horizontal,
              children: [
                for (var i = 0; i < _frames.length; i++)
                  Stack(
                    children: [
                      Padding(
                        padding: const EdgeInsets.all(4),
                        child: Image.memory(
                          _frames[i],
                          width: 100,
                          fit: BoxFit.contain,
                        ),
                      ),
                      IconButton(
                        onPressed: _busy
                            ? null
                            : () => setState(() {
                                _frames.removeAt(i);
                                _offsets.removeAt(i);
                              }),
                        tooltip: 'Remove view ${i + 1}',
                        icon: const Icon(Icons.close),
                      ),
                    ],
                  ),
              ],
            ),
          ),
          Padding(
            padding: const EdgeInsets.all(12),
            child: Wrap(
              spacing: 8,
              children: [
                OutlinedButton(
                  onPressed: _busy || _frames.length >= 3 || _camera == null
                      ? null
                      : _capture,
                  child: Text(
                    _busy
                        ? 'Saving still…'
                        : 'Capture view (${_frames.length}/3)',
                  ),
                ),
                FilledButton(
                  onPressed: _busy || _frames.isEmpty
                      ? null
                      : () => Navigator.pop(
                          context,
                          LiveEvidence(
                            List.of(_frames),
                            List.of(_offsets),
                            _started,
                          ),
                        ),
                  child: const Text('Review and share'),
                ),
              ],
            ),
          ),
        ],
      ),
    ),
  );

  /// The same wording identifyFrame's answers get below.
  String _liveText(LiveView? v) {
    final id = v?.identification;
    if (v == null || !v.answered || id == null) {
      return v?.snakeInView == true ? snakeInViewIdentifying : 'Looking for a snake…';
    }
    if (!id.snakeFound) return 'No snake detected. This does not mean the area is safe.';
    return '${id.verdict.title} · ${id.topSpecies?.displayName ?? id.reason}';
  }

  Verdict? _liveVerdict(LiveView? v) =>
      v?.identification?.snakeFound == true ? v!.identification!.verdict : null;

  /// Every tracked snake; the verdict's colour and name on the one it is about.
  Widget? _liveBoxes(LiveView? v) {
    if (v == null || v.tracks.isEmpty || v.width == 0) return null;
    final verdict = _liveVerdict(v);
    return SnakeBoxesOverlay(
      imageSize: Size(v.width.toDouble(), v.height.toDouble()),
      fit: BoxFit.fill,
      boxes: [
        if (v.names.isNotEmpty || v.tracks.length >= 2)
          for (final t in v.tracks) _namedBox(t, v.names[t.id])
        else
          for (final t in v.tracks)
            if (t.id == v.mainTrack)
            OverlayBox(
              t.box,
              verdict == null ? Brand.venom : VerdictStyle.of(verdict).background,
              label: verdict == null ? identifyingText : v.identification!.topSpecies?.displayName ?? verdict.title,
              main: true,
            )
          else
            OverlayBox(t.box, Brand.venom.withValues(alpha: 0.85), label: 'Snake?'),
      ],
    );
  }

  /// A snake named on its own (several in view): its own colour and name,
  /// "Identifying…" until its own lock settles.
  OverlayBox _namedBox(Track t, Identification? id) {
    final named = id != null && id.snakeFound;
    return OverlayBox(
      t.box,
      named ? VerdictStyle.of(id.verdict).background : Brand.venom,
      label: named ? id.topSpecies?.displayName ?? id.verdict.title : identifyingText,
      main: named,
    );
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _generation++;
    unawaited(_liveViews?.cancel());
    unawaited(_live?.close());
    _clock.stop();
    unawaited(_camera?.dispose());
    super.dispose();
  }
}


class _LiveStatus extends StatelessWidget {
  const _LiveStatus({required this.text, this.verdict});
  final String text;
  final Verdict? verdict;

  @override
  Widget build(BuildContext context) {
    final style = verdict == null ? null : VerdictStyle.of(verdict!);
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
      decoration: BoxDecoration(
        color: style?.background ?? Brand.forest800,
        borderRadius: BorderRadius.circular(16),
      ),
      child: Row(children: [
        Icon(style?.icon ?? Icons.radar_rounded, color: style?.foreground ?? Colors.white),
        const SizedBox(width: 10),
        Expanded(
          child: Text(text,
              style: TextStyle(color: style?.foreground ?? Colors.white, fontWeight: FontWeight.w700, fontSize: 15)),
        ),
      ]),
    );
  }
}
