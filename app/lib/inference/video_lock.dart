import 'dart:math' as math;
import 'dart:typed_data';

import 'decision.dart';

/// The lock's settings, all in seconds of the clip so the lock behaves the
/// same however fast the phone runs the model. The defaults are the M5
/// retune (`results/m5/g5/chosen_settings.json`), except the switch time: 2 s since M6
/// (`results/m6/lock_switch/`, faster to move to a new snake, still at most 1
/// change on every dev clip); the names match
/// `lock_settings` in `src/serpentguard/video.py`.
class LockSettings {
  const LockSettings({
    this.windowSeconds = 0.5,
    this.margin = 0.05,
    this.switchSeconds = 2.0,
    this.gateShare = 0.0,
    this.minFrames = 1,
    this.holdSeconds = 2.0,
    this.snakeRun = 2,
  });

  /// Snake frames of the last [windowSeconds] are pooled, and never fewer
  /// than the last [minFrames] (a slow phone still pools several frames).
  final double windowSeconds;

  /// The pooled top species must lead the runner-up by this to lock.
  final double margin;

  /// Another species must clearly lead this long to replace the lock.
  final double switchSeconds;

  /// Share of the window's processed frames the gate must call a snake.
  final double gateShare;

  /// Snake frames the pool must hold to lock.
  final int minFrames;

  /// With no snake frame for this long the lock and the pool are dropped.
  final double holdSeconds;

  /// A sighting starts only after this many snake frames in a row, so one
  /// frame that scrapes past the snake check shows nothing. App only (M6):
  /// on a striped screen the live camera's snake check passed 13 of 852
  /// frames, each shown at once as "Dangerous"; two in a row is far rarer.
  final int snakeRun;
}

const _timeEpsilon = 1e-9;

/// The video rule (`pooled_lock` and `lock_verdicts` in
/// `src/serpentguard/video.py`), for a clip or the live camera.
///
/// Every output the verdict reads is averaged over the snake frames of the
/// last [LockSettings.windowSeconds]. The screen shows "Checking..." until
/// the sighting has lasted a window, the pooled top species holds [conf]
/// and leads the runner-up by the margin; then it locks on that species and
/// shows it, with the verdict of the pool at that moment. A lock changes
/// only when another species holds [conf] and leads the locked one by more
/// than the margin on every snake frame for the switch time. Safety first:
/// "dangerous" on the current pool turns the shown verdict dangerous until
/// the lock changes, and while checking "dangerous" shows with no species
/// named yet. With no snake for [LockSettings.holdSeconds] everything is
/// dropped, so panning to another snake starts afresh.
class VideoLock {
  VideoLock({required this.rules, this.settings = const LockSettings()}) : conf = rules.thresholds.thetaConf;

  final DecisionRules rules;
  final LockSettings settings;
  final double conf;

  final List<(double, DecisionParts?)> _seen = [];
  int? _locked, _rival;
  double? _since, _sighting, _lastSnake;
  int _run = 0; // snake frames in a row
  ModelAnswer? _held;
  LockStep? _last;

  int? get lockedClass => _locked;

  void reset() {
    _seen.clear();
    _locked = _rival = null;
    _since = _sighting = _lastSnake = null;
    _run = 0;
    _held = null;
    _last = null;
  }

  /// One processed frame at [seconds]; [parts] is null when the gate found
  /// no snake.
  LockStep add(DecisionParts? parts, double seconds) {
    final s = settings;
    String? event;
    final keep = math.max(s.windowSeconds, s.holdSeconds);
    _seen.removeWhere((e) => e.$1 <= seconds - keep + _timeEpsilon);
    if (parts == null) {
      _run = 0;
      _seen.add((seconds, null));
      if (_lastSnake != null && seconds - _lastSnake! >= s.holdSeconds - _timeEpsilon) {
        reset();
        return _last = const LockStep(checking: true, pooled: null, shown: null);
      }
      return _last = _last?.repeated ?? const LockStep(checking: true, pooled: null, shown: null);
    }
    _seen.add((seconds, parts));
    _run++;
    if (_sighting == null && _run < s.snakeRun) {
      return _last = const LockStep(checking: true, pooled: null, shown: null);
    }
    _sighting ??= seconds;
    _lastSnake = seconds;
    // The window's snake frames, and never fewer than the last minFrames.
    final snakeTimes = [for (final e in _seen) if (e.$2 != null) e.$1];
    final inWindow = snakeTimes.firstWhere((t) => t > seconds - s.windowSeconds + _timeEpsilon, orElse: () => seconds);
    final byCount = snakeTimes.length >= s.minFrames ? snakeTimes[snakeTimes.length - s.minFrames] : snakeTimes.first;
    final start = math.min(inWindow, byCount);
    final spanned = [for (final e in _seen) if (e.$1 >= start - _timeEpsilon) e];
    final snake = [for (final e in spanned) if (e.$2 != null) e.$2!];
    final pooled = _pool(snake);
    final probs = pooled.probs;
    final order = _rank(probs);
    final top = order[0];
    final lead = probs[top] - (probs.length > 1 ? probs[order[1]] : 0.0);
    if (_locked == null) {
      if (seconds - _sighting! >= s.windowSeconds - _timeEpsilon &&
          snake.length >= s.minFrames &&
          snake.length / spanned.length >= s.gateShare - _timeEpsilon &&
          probs[top] >= conf &&
          lead >= s.margin) {
        _locked = top;
        event = 'lock';
      }
    } else {
      final leads = top != _locked && probs[top] >= conf && probs[top] - probs[_locked!] > s.margin;
      if (!leads) {
        _rival = null;
        _since = null;
      } else {
        if (_rival != top) {
          _rival = top;
          _since = seconds;
        }
        if (seconds - _since! >= s.switchSeconds - _timeEpsilon) {
          _locked = _rival;
          _rival = null;
          _since = null;
          event = 'switch';
        }
      }
    }
    final answer = rules.verdict(pooled);
    if (_locked == null) {
      _held = null;
      final shown = answer.label == dangerousLabel ? _withSpecies(answer, null) : null;
      return _last = LockStep(checking: true, pooled: answer, shown: shown);
    }
    if (event != null || _held == null) {
      _held = answer;
    } else if (answer.label == dangerousLabel && _held!.label != dangerousLabel) {
      _held = answer;
    }
    return _last = LockStep(
      checking: false,
      pooled: answer,
      shown: _withSpecies(_held!, _locked, probs),
      lockedClass: _locked,
      event: event,
    );
  }

  /// [a] with the shown species: none while checking, else the locked one
  /// first in the top 3 (its pooled probability), then the next two.
  static ModelAnswer _withSpecies(ModelAnswer a, int? species, [Float64List? probs]) {
    final List<ScoredClass> top3;
    if (species == null) {
      top3 = const [];
    } else {
      final rest = _rank(probs!).where((k) => k != species).take(2);
      top3 = [ScoredClass(species, probs[species]), for (final k in rest) ScoredClass(k, probs[k])];
    }
    return ModelAnswer(
      label: a.label,
      reasons: a.reasons,
      message: a.message,
      top3: top3,
      genusFallback: a.genusFallback,
      margin: a.margin,
      bestCosine: a.bestCosine,
    );
  }

  /// Class ids by probability, highest first; ties keep the lower id first.
  static List<int> _rank(Float64List probs) => List.generate(probs.length, (i) => i)
    ..sort((a, b) {
      final c = probs[b].compareTo(probs[a]);
      return c != 0 ? c : a.compareTo(b);
    });

  /// The mean of each part over the window; the embedding re-normalised.
  static DecisionParts _pool(List<DecisionParts> frames) {
    Float64List mean(Float64List Function(DecisionParts) pick) {
      final first = pick(frames.first);
      final out = Float64List(first.length);
      for (final f in frames) {
        final v = pick(f);
        for (var i = 0; i < out.length; i++) {
          out[i] += v[i];
        }
      }
      for (var i = 0; i < out.length; i++) {
        out[i] /= frames.length;
      }
      return out;
    }

    Float64List? embedding;
    if (frames.every((f) => f.embedding != null)) {
      embedding = mean((f) => f.embedding!);
      var norm = 0.0;
      for (final v in embedding) {
        norm += v * v;
      }
      norm = math.max(math.sqrt(norm), 1e-12);
      for (var i = 0; i < embedding.length; i++) {
        embedding[i] /= norm;
      }
    }
    return DecisionParts(
      probs: mean((f) => f.probs),
      who: mean((f) => f.who),
      genus: mean((f) => f.genus),
      family: mean((f) => f.family),
      embedding: embedding,
    );
  }
}

class LockStep {
  const LockStep({
    required this.checking,
    required this.pooled,
    required this.shown,
    this.lockedClass,
    this.event,
  });

  /// True until the lock is set (and again after the hold expires).
  final bool checking;

  /// The verdict on the current pool (null before any snake frame).
  final ModelAnswer? pooled;

  /// What the screen shows; null means "Checking...". Once locked its top
  /// species is always [lockedClass]; while checking it is only ever a
  /// "dangerous" answer with an empty top 3 (no species named yet).
  final ModelAnswer? shown;
  final int? lockedClass;

  /// 'lock' or 'switch' on the frame the lock is set or changed.
  final String? event;

  /// The same screen on a frame with no snake: no new event.
  LockStep get repeated =>
      LockStep(checking: checking, pooled: pooled, shown: shown, lockedClass: lockedClass);
}
