import 'dart:async';
import 'dart:math' as math;

import 'package:flutter/services.dart';
import 'package:material_ui/material_ui.dart';
import 'package:video_player/video_player.dart';

import '../app.dart';
import 'sos_screen.dart';

/// Living 6-layer splash screen that bridges the native launch screen to [HomeShell].
///
/// Layers from back to front:
/// 1. Hardware-decoded muted MP4 loop (`assets/videos/serpent_intro.mp4`).
/// 2. Frame-0 poster image (`assets/images/splash_poster.jpg`) that fades out on play.
/// 3. Top and bottom obsidian-emerald vignette gradients.
/// 4. Native 60fps emerald spore particles and shield glow pulse.
/// 5. Crisp native Flutter typography and startup readiness indicator.
/// 6. Skip control and immediate Snakebite SOS bypass.
class SplashScreen extends StatefulWidget {
  const SplashScreen({
    super.key,
    this.initFuture,
    this.minDuration = const Duration(milliseconds: 3800),
    this.nextScreen = const HomeShell(),
  });

  final Future<void>? initFuture;
  final Duration minDuration;
  final Widget nextScreen;

  @override
  State<SplashScreen> createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen>
    with WidgetsBindingObserver, TickerProviderStateMixin {
  static const _bgColor = Color(0xFF07120F);
  static const _emerald = Color(0xFF35D99B);

  VideoPlayerController? _videoController;
  late final AnimationController _ambientController;
  late final AnimationController _progressController;

  Timer? _minTimer;
  bool _precached = false;
  bool _videoPlaying = false;
  bool _posterHidden = false;
  bool _minElapsed = false;
  bool _initDone = false;
  bool _navigated = false;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);

    _ambientController = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 6),
    )..repeat();

    _progressController = AnimationController(
      vsync: this,
      duration: widget.minDuration > Duration.zero
          ? widget.minDuration
          : const Duration(milliseconds: 300),
    )..forward();

    _initVideo();
    _armTransitions();
  }

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    if (!_precached) {
      _precached = true;
      precacheImage(const AssetImage('assets/images/splash_poster.jpg'), context);
    }
  }

  Future<void> _initVideo() async {
    try {
      final controller = VideoPlayerController.asset(
        'assets/videos/serpent_intro.mp4',
        videoPlayerOptions: VideoPlayerOptions(mixWithOthers: true),
      );
      _videoController = controller;
      await controller.initialize();
      await controller.setLooping(true);
      await controller.setVolume(0);

      if (!mounted || _navigated) return;
      controller.addListener(_onVideoTick);
      await controller.play();
    } catch (error) {
      // Fall back gracefully to the Frame-0 poster + native particle layer.
      debugPrint('Splash video fallback: $error');
    }
  }

  void _onVideoTick() {
    final controller = _videoController;
    if (controller == null || !mounted || _videoPlaying) return;
    final value = controller.value;
    if (value.isInitialized &&
        value.isPlaying &&
        value.position >= const Duration(milliseconds: 40)) {
      setState(() => _videoPlaying = true);
    }
  }

  void _armTransitions() {
    _minElapsed = widget.minDuration <= Duration.zero;
    _initDone = widget.initFuture == null;

    if (!_minElapsed) {
      _minTimer = Timer(widget.minDuration, () {
        _minElapsed = true;
        _tryAutoNavigate();
      });
    }

    if (!_initDone) {
      widget.initFuture!
          .timeout(const Duration(milliseconds: 4500), onTimeout: () {})
          .whenComplete(() {
        if (!mounted) return;
        _initDone = true;
        _tryAutoNavigate();
      });
    }

    if (_minElapsed && _initDone) {
      WidgetsBinding.instance.addPostFrameCallback((_) => _tryAutoNavigate());
    }
  }

  void _tryAutoNavigate() {
    if (_minElapsed && _initDone) {
      _navigateToNext();
    }
  }

  void _navigateToNext({bool openSosImmediately = false}) {
    if (_navigated || !mounted) return;
    _navigated = true;
    _minTimer?.cancel();
    _ambientController.stop();
    _progressController.stop();
    _videoController?.pause();

    final navigator = Navigator.of(context);
    navigator.pushReplacement(_buildRoute(widget.nextScreen));
    if (openSosImmediately) {
      navigator.push(
        MaterialPageRoute(builder: (_) => const SosScreen(bitten: true)),
      );
    }
  }

  PageRoute<void> _buildRoute(Widget target) {
    return PageRouteBuilder<void>(
      transitionDuration: const Duration(milliseconds: 450),
      pageBuilder: (_, _, _) => target,
      transitionsBuilder: (_, animation, _, child) {
        return FadeTransition(
          opacity: CurvedAnimation(
            parent: animation,
            curve: Curves.easeInOutCubic,
          ),
          child: child,
        );
      },
    );
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    final controller = _videoController;
    if (controller == null || !controller.value.isInitialized || _navigated) {
      return;
    }
    if (state == AppLifecycleState.paused || state == AppLifecycleState.inactive) {
      controller.pause();
    } else if (state == AppLifecycleState.resumed) {
      controller.play();
    }
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _minTimer?.cancel();
    _videoController?.removeListener(_onVideoTick);
    _videoController?.dispose();
    _ambientController.dispose();
    _progressController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AnnotatedRegion<SystemUiOverlayStyle>(
      value: SystemUiOverlayStyle.light,
      child: Scaffold(
        backgroundColor: _bgColor,
        body: Stack(
          fit: StackFit.expand,
          children: [
            _buildVideoLayer(),
            _buildPosterLayer(),
            _buildVignetteLayer(),
            _buildParticleLayer(),
            _buildTopSkipLayer(),
            _buildBottomBrandingAndSosLayer(),
          ],
        ),
      ),
    );
  }

  Widget _buildVideoLayer() {
    final controller = _videoController;
    if (controller == null || !controller.value.isInitialized) {
      return const SizedBox.shrink();
    }
    final size = controller.value.size;
    if (size.width <= 0 || size.height <= 0) {
      return const SizedBox.shrink();
    }
    return SizedBox.expand(
      child: FittedBox(
        fit: BoxFit.cover,
        clipBehavior: Clip.hardEdge,
        child: SizedBox(
          width: size.width,
          height: size.height,
          child: VideoPlayer(controller),
        ),
      ),
    );
  }

  Widget _buildPosterLayer() {
    if (_posterHidden) {
      return const SizedBox.shrink();
    }
    return IgnorePointer(
      child: AnimatedOpacity(
        opacity: _videoPlaying ? 0.0 : 1.0,
        duration: const Duration(milliseconds: 220),
        curve: Curves.easeOut,
        onEnd: () {
          if (_videoPlaying && mounted) {
            setState(() => _posterHidden = true);
          }
        },
        child: Image.asset(
          'assets/images/splash_poster.jpg',
          fit: BoxFit.cover,
          gaplessPlayback: true,
          width: double.infinity,
          height: double.infinity,
        ),
      ),
    );
  }

  Widget _buildVignetteLayer() {
    return IgnorePointer(
      child: DecoratedBox(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            stops: const [0.0, 0.16, 0.56, 0.82, 1.0],
            colors: [
              _bgColor.withValues(alpha: 0.72),
              Colors.transparent,
              Colors.transparent,
              _bgColor.withValues(alpha: 0.84),
              _bgColor,
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildParticleLayer() {
    return IgnorePointer(
      child: RepaintBoundary(
        child: CustomPaint(
          painter: _GuardianAtmospherePainter(_ambientController),
          size: Size.infinite,
        ),
      ),
    );
  }

  Widget _buildTopSkipLayer() {
    return SafeArea(
      child: Align(
        alignment: Alignment.topRight,
        child: Padding(
          padding: const EdgeInsets.only(top: 12, right: 16),
          child: Material(
            color: const Color(0x8807120F),
            shape: StadiumBorder(
              side: BorderSide(color: _emerald.withValues(alpha: 0.35)),
            ),
            child: InkWell(
              customBorder: const StadiumBorder(),
              onTap: () => _navigateToNext(),
              child: const Padding(
                padding: EdgeInsets.symmetric(horizontal: 14, vertical: 8),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Text(
                      'Skip',
                      style: TextStyle(
                        color: Color(0xFFE1EFE9),
                        fontSize: 13,
                        fontWeight: FontWeight.w600,
                        letterSpacing: 0.5,
                      ),
                    ),
                    SizedBox(width: 2),
                    Icon(
                      Icons.chevron_right_rounded,
                      size: 18,
                      color: _emerald,
                    ),
                  ],
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildBottomBrandingAndSosLayer() {
    return SafeArea(
      child: Align(
        alignment: Alignment.bottomCenter,
        child: Padding(
          padding: const EdgeInsets.fromLTRB(24, 0, 24, 20),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              RichText(
                textAlign: TextAlign.center,
                text: const TextSpan(
                  style: TextStyle(
                    fontSize: 27,
                    fontWeight: FontWeight.w800,
                    letterSpacing: 3.4,
                    shadows: [
                      Shadow(
                        color: Color(0xCC000000),
                        blurRadius: 14,
                        offset: Offset(0, 2),
                      ),
                    ],
                  ),
                  children: [
                    TextSpan(
                      text: 'SERPENT ',
                      style: TextStyle(color: Color(0xFFF1F7F4)),
                    ),
                    TextSpan(
                      text: 'GUARD',
                      style: TextStyle(color: _emerald),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 6),
              const Text(
                'WATCH  ·  PROTECT  ·  DEFEND',
                textAlign: TextAlign.center,
                style: TextStyle(
                  color: Color(0xCCBEE3D3),
                  fontSize: 11,
                  fontWeight: FontWeight.w600,
                  letterSpacing: 3.4,
                ),
              ),
              const SizedBox(height: 16),
              _ThreeDotPulse(animation: _ambientController),
              const SizedBox(height: 14),
              ClipRRect(
                borderRadius: BorderRadius.circular(4),
                child: SizedBox(
                  width: 148,
                  height: 3,
                  child: AnimatedBuilder(
                    animation: _progressController,
                    builder: (_, _) => LinearProgressIndicator(
                      value: _progressController.value,
                      backgroundColor: Colors.white.withValues(alpha: 0.14),
                      valueColor: const AlwaysStoppedAnimation<Color>(_emerald),
                    ),
                  ),
                ),
              ),
              const SizedBox(height: 18),
              OutlinedButton.icon(
                style: OutlinedButton.styleFrom(
                  backgroundColor: const Color(0x668C1D18),
                  foregroundColor: const Color(0xFFFFDAD6),
                  side: const BorderSide(color: Color(0xFFB3261E), width: 1.2),
                  padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 10),
                  shape: const StadiumBorder(),
                ),
                onPressed: () => _navigateToNext(openSosImmediately: true),
                icon: const Icon(Icons.emergency_rounded, size: 18, color: Color(0xFFFFB4AB)),
                label: const Text(
                  'Emergency / Bitten? Open SOS',
                  style: TextStyle(fontSize: 13, fontWeight: FontWeight.w700),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _ThreeDotPulse extends StatelessWidget {
  const _ThreeDotPulse({required this.animation});

  final Animation<double> animation;

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: animation,
      builder: (_, _) {
        return Row(
          mainAxisSize: MainAxisSize.min,
          children: List.generate(3, (index) {
            final phase = (animation.value * 3 - index * 0.25) % 1.0;
            final pulse = 0.45 + 0.55 * (0.5 + 0.5 * math.sin(phase * math.pi * 2));
            return Container(
              width: 7,
              height: 7,
              margin: const EdgeInsets.symmetric(horizontal: 5),
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: Color.lerp(
                  const Color(0xFF1A5B44),
                  const Color(0xFF35D99B),
                  pulse,
                ),
                boxShadow: [
                  BoxShadow(
                    color: const Color(0xFF35D99B).withValues(alpha: 0.45 * pulse),
                    blurRadius: 8 * pulse,
                  ),
                ],
              ),
            );
          }),
        );
      },
    );
  }
}

class _Spore {
  const _Spore({
    required this.x,
    required this.yOffset,
    required this.radius,
    required this.speed,
    required this.swayAmp,
    required this.phase,
    required this.alpha,
  });

  final double x;
  final double yOffset;
  final double radius;
  final double speed;
  final double swayAmp;
  final double phase;
  final double alpha;
}

class _GuardianAtmospherePainter extends CustomPainter {
  _GuardianAtmospherePainter(this.animation) : super(repaint: animation);

  final Animation<double> animation;

  static final List<_Spore> _spores = _buildSpores();

  static List<_Spore> _buildSpores() {
    final rng = math.Random(42);
    return List.generate(18, (i) {
      return _Spore(
        x: 0.08 + rng.nextDouble() * 0.84,
        yOffset: rng.nextDouble(),
        radius: 1.2 + rng.nextDouble() * 2.1,
        speed: 1.0 + (i % 3),
        swayAmp: 6.0 + rng.nextDouble() * 14.0,
        phase: rng.nextDouble() * math.pi * 2,
        alpha: 0.28 + rng.nextDouble() * 0.48,
      );
    });
  }

  @override
  void paint(Canvas canvas, Size size) {
    final t = animation.value;
    final pulse = 0.5 + 0.5 * math.sin(t * math.pi * 2);

    // Subtle breathing halo aligned with the central emblem on the monolith.
    final haloCenter = Offset(size.width * 0.5, size.height * 0.44);
    final haloRadius = size.shortestSide * (0.28 + 0.04 * pulse);
    final haloPaint = Paint()
      ..shader = RadialGradient(
        colors: [
          const Color(0xFF35D99B).withValues(alpha: 0.11 + 0.06 * pulse),
          const Color(0xFF35D99B).withValues(alpha: 0.0),
        ],
      ).createShader(Rect.fromCircle(center: haloCenter, radius: haloRadius));
    canvas.drawCircle(haloCenter, haloRadius, haloPaint);

    // Drifting emerald spores across the screen (pre-softened circles, zero blur).
    final dotPaint = Paint()..style = PaintingStyle.fill;

    for (final s in _spores) {
      final yProgress = (s.yOffset - t * s.speed) % 1.0;
      final normalizedY = yProgress < 0 ? yProgress + 1.0 : yProgress;
      final y = normalizedY * size.height;
      final x = s.x * size.width + math.sin(t * math.pi * 2 + s.phase) * s.swayAmp;
      final edgeFade = math.sin(normalizedY * math.pi).clamp(0.0, 1.0);
      final opacity = s.alpha * edgeFade;
      if (opacity <= 0.02) continue;

      final center = Offset(x, y);
      dotPaint.color = const Color(0xFF35D99B).withValues(alpha: opacity * 0.22);
      canvas.drawCircle(center, s.radius * 2.4, dotPaint);

      dotPaint.color = const Color(0xFFA7F3D0).withValues(alpha: opacity);
      canvas.drawCircle(center, s.radius, dotPaint);
    }
  }

  @override
  bool shouldRepaint(covariant _GuardianAtmospherePainter oldDelegate) => false;
}
