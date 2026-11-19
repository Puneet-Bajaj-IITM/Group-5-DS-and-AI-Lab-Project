import 'dart:math' as math;
import 'dart:typed_data';

import 'decision.dart';

/// The Milestone 4 video rule (`pooled_lock` and `lock_verdicts` in
/// `src/serpentguard/video.py`), for a clip or the live camera.
///
/// Every output the verdict reads (species, WHO, genus and family
/// probabilities and the embedding) is averaged over the last [window]
/// frames with a snake. The screen shows "Checking..." until the pooled top
/// species holds [conf] and leads the runner-up by [margin], then locks on
/// it. A lock changes only when another species holds [conf] and leads the
/// locked one by more than [margin] on every frame for [switchSeconds]. A
/// frame with no snake repeats the previous step. "Dangerous" may show while
/// checking; "not listed" and "unknown" only once locked.
///
/// A3 ships window 5 and margin 0.05 (`results/m4/video/A3/margin_tuning.csv`),
/// with conf = theta_conf.
class VideoLock {
  VideoLock({required this.rules, this.window = 5, this.margin = 0.05, this.switchSeconds = 1.0})
      : conf = rules.thresholds.thetaConf;

  final DecisionRules rules;
  final int window;
  final double margin, conf, switchSeconds;

  final List<DecisionParts> _recent = [];
  int? _locked, _rival;
  double? _since;
  LockStep? _last;

  int? get lockedClass => _locked;

  void reset() {
    _recent.clear();
    _locked = _rival = null;
    _since = null;
    _last = null;
  }

  /// One frame at [seconds]; [parts] is null when the gate found no snake.
  LockStep add(DecisionParts? parts, double seconds) {
    if (parts == null) {
      return _last ??= const LockStep(checking: true, pooled: null, shown: null);
    }
    _recent.add(parts);
    if (_recent.length > window) _recent.removeAt(0);
    final pooled = _pool(_recent);
    final probs = pooled.probs;
    final order = List.generate(probs.length, (i) => i)..sort((a, b) => probs[b].compareTo(probs[a]));
    final top = order[0];
    final lead = probs[top] - (probs.length > 1 ? probs[order[1]] : 0.0);
    if (_locked == null) {
      if (_recent.length == window && probs[top] >= conf && lead >= margin) _locked = top;
    } else {
      final leads = top != _locked && probs[top] >= conf && probs[top] - probs[_locked!] > margin;
      if (!leads) {
        _rival = null;
        _since = null;
      } else {
        if (_rival != top) {
          _rival = top;
          _since = seconds;
        }
        if (seconds - _since! >= switchSeconds - 1e-9) {
          _locked = _rival;
          _rival = null;
          _since = null;
        }
      }
    }
    final answer = rules.verdict(pooled);
    final checking = _locked == null;
    final shown = (!checking || answer.label == dangerousLabel) ? answer : null;
    return _last = LockStep(checking: checking, pooled: answer, shown: shown, lockedClass: _locked);
  }

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
  const LockStep({required this.checking, required this.pooled, required this.shown, this.lockedClass});

  /// True until the lock is set.
  final bool checking;

  /// The verdict on the pooled outputs (null before any snake frame).
  final ModelAnswer? pooled;

  /// What the screen shows: the pooled verdict once locked, or "dangerous"
  /// while checking; null means "Checking...".
  final ModelAnswer? shown;
  final int? lockedClass;
}
