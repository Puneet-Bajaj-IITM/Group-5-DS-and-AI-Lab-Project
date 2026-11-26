import 'dart:async';

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:material_ui/material_ui.dart';
import 'package:provider/provider.dart';

import '../services/voting_queue.dart';
import '../services/classifier.dart';
import '../widgets/common.dart';
import '../ui/brand.dart';
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
        await camera.startImageStream(
          (image) => _analyze(image, camera!, generation),
        );
      }
    } catch (_) {
      await camera?.dispose();
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
      final result = await classifier.identifyFrame(
        LiveCameraFrame(
          width: image.width,
          height: image.height,
          format: image.format.group.name,
          offsetMs: _clock.elapsedMilliseconds,
          sensorOrientation: camera.description.sensorOrientation,
          deviceOrientation: camera.value.deviceOrientation.name,
          frontFacing:
              camera.description.lensDirection == CameraLensDirection.front,
          planes: image.planes
              .map(
                (p) => LiveCameraPlane(
                  Uint8List.fromList(p.bytes),
                  p.bytesPerRow,
                  p.bytesPerPixel,
                ),
              )
              .toList(),
        ),
      );
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
            child: _LiveStatus(
              text: _classifier == null
                  ? 'Live model not connected in this build. Preview and evidence capture are available.'
                  : _identification == null
                  ? (_lastFrame?.gate.snake == true ? 'Snake in view · checking…' : 'Looking for a snake…')
                  : !_identification!.snakeFound
                  ? 'No snake detected. This does not mean the area is safe.'
                  : '${_identification!.verdict.title} · ${_identification!.topSpecies?.displayName ?? _identification!.reason}',
              verdict: _identification?.snakeFound == true ? _identification!.verdict : null,
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
                        child: _lastFrame?.gate.snake == true && _lastFrame!.gate.box != null
                            ? SnakeBoxOverlay(
                                box: _lastFrame!.gate.box!,
                                imageSize: Size(_lastFrame!.width.toDouble(), _lastFrame!.height.toDouble()),
                                fit: BoxFit.fill,
                                color: _identification?.snakeFound == true
                                    ? VerdictStyle.of(_identification!.verdict).background
                                    : Brand.venom,
                                label: _identification?.snakeFound == true
                                    ? _identification!.topSpecies?.displayName
                                    : 'Checking…',
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

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _generation++;
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
