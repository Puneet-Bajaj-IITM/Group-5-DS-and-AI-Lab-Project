import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:serpentguard/inference/bundle.dart';
import 'package:serpentguard/inference/decision.dart';
import 'package:serpentguard/inference/video_lock.dart';

const blindSnake = 0; // Indotyphlops braminus, not WHO-listed
const vineSnake = 1; // Ahaetulla prasina, not WHO-listed
const cobra = 8; // Naja naja, WHO Category 1
const peak = 0.9;

void main() {
  final settings = PipelineSettings.parse(File('assets/model/pipeline.json').readAsStringSync());
  final tables = ModelTables.parse(File('assets/model/tables.json').readAsStringSync());
  final rules = DecisionRules(settings.thresholds, tables);

  /// Parts of a frame sure of [species]: heads consistent with the species
  /// probabilities, a non-venomous family, the species' prototype.
  DecisionParts sure(int species, {double p = peak}) {
    final n = tables.numClasses;
    final probs = Float64List(n)..fillRange(0, n, (1 - p) / (n - 1));
    probs[species] = p;
    final who = Float64List(3), genus = Float64List(tables.numGenera);
    for (var k = 0; k < n; k++) {
      who[tables.who[k]] += probs[k];
      genus[tables.genusOfClass[k]] += probs[k];
    }
    final family = Float64List(tables.venomousFamily.length);
    family[tables.venomousFamily.indexOf(false)] = 1;
    return DecisionParts(probs: probs, who: who, genus: genus, family: family, embedding: tables.unitPrototypes[species]);
  }

  /// Runs [frames] (null = no snake) at [fps]; returns the steps.
  List<LockStep> run(VideoLock lock, List<DecisionParts?> frames, double fps, {double start = 0}) =>
      [for (var i = 0; i < frames.length; i++) lock.add(frames[i], start + i / fps)];

  int? species(LockStep s) => s.shown == null || s.shown!.top3.isEmpty ? null : s.shown!.top3.first.classId;

  double? firstLock(List<LockStep> steps, double fps) {
    final i = steps.indexWhere((s) => !s.checking);
    return i < 0 ? null : i / fps;
  }

  test('checks until a full window, then locks and names the species', () {
    final lock = VideoLock(rules: rules);
    final steps = run(lock, List.filled(20, sure(vineSnake)), 10);
    expect(steps.first.checking, isTrue);
    expect(steps.first.shown, isNull);
    expect(firstLock(steps, 10), closeTo(const LockSettings().windowSeconds, 1e-9));
    expect(steps.last.lockedClass, vineSnake);
    expect(species(steps.last), vineSnake);
    expect(steps.firstWhere((s) => !s.checking).event, 'lock');
  });

  test('the window is in seconds: a slow phone locks at the same time', () {
    for (final fps in [10.0, 2.5, 1.25]) {
      final steps = run(VideoLock(rules: rules, settings: const LockSettings(windowSeconds: 1.0)),
          List.filled((6 * fps).round(), sure(vineSnake)), fps);
      expect(firstLock(steps, fps), closeTo(1.0, 0.8 + 1e-9), reason: '$fps fps');
      expect(firstLock(steps, fps)! >= 1.0 - 1e-9, isTrue, reason: '$fps fps');
    }
  });

  test('a slow phone still pools min frames: at 0.8 s a frame a 0.5 s window locks on the 2nd or 3rd frame', () {
    final two = run(VideoLock(rules: rules, settings: const LockSettings(windowSeconds: 0.5, minFrames: 2)),
        List.filled(6, sure(vineSnake)), 1 / 0.8);
    expect(firstLock(two, 1 / 0.8), closeTo(0.8, 1e-9));
    final three = run(VideoLock(rules: rules, settings: const LockSettings(windowSeconds: 0.5, minFrames: 3)),
        List.filled(6, sure(vineSnake)), 1 / 0.8);
    expect(firstLock(three, 1 / 0.8), closeTo(1.6, 1e-9));
  });

  test('the shown species is the locked one, even while the pool briefly prefers another', () {
    final lock = VideoLock(rules: rules);
    final frames = [
      ...List.filled(15, sure(vineSnake)),
      ...List.filled(5, sure(blindSnake, p: 0.95)), // 0.5 s: shorter than the switch time
      ...List.filled(15, sure(vineSnake)),
    ];
    final steps = run(lock, frames, 10);
    final named = steps.map(species).whereType<int>().toSet();
    expect(named, {vineSnake});
    expect(steps.any((s) => s.pooled?.top3.first.classId == blindSnake), isTrue,
        reason: 'the pool did prefer the other species');
  });

  test('a rival leading for the switch time replaces the lock once', () {
    final lock = VideoLock(rules: rules);
    final steps = run(lock, [...List.filled(20, sure(vineSnake)), ...List.filled(40, sure(blindSnake))], 10);
    final named = steps.map(species).whereType<int>().toList();
    var changes = 0;
    for (var i = 1; i < named.length; i++) {
      if (named[i] != named[i - 1]) changes++;
    }
    expect(changes, 1);
    expect(steps.last.lockedClass, blindSnake);
    expect(steps.where((s) => s.event == 'switch').length, 1);
  });

  test('panning away for the hold time drops the lock; the next snake starts afresh', () {
    final lock = VideoLock(rules: rules);
    final hold = const LockSettings().holdSeconds;
    final frames = [
      ...List.filled(20, sure(vineSnake)),
      ...List<DecisionParts?>.filled((hold * 10).round() + 2, null),
      ...List.filled(20, sure(blindSnake)),
    ];
    final steps = run(lock, frames, 10);
    final gapEnd = 20 + (hold * 10).round() + 2;
    expect(steps[20].lockedClass, vineSnake, reason: 'a short gap keeps the lock');
    expect(steps[gapEnd - 1].checking, isTrue);
    expect(steps[gapEnd - 1].shown, isNull);
    expect(steps[gapEnd].checking, isTrue, reason: 'a new sighting must fill its window first');
    expect(steps.last.lockedClass, blindSnake);
    expect(steps.last.event, isNull);
    expect(steps.where((s) => s.event == 'lock').length, 2);
  });

  test('a short gap with no snake keeps showing the lock', () {
    final lock = VideoLock(rules: rules);
    final steps = run(lock, [...List.filled(20, sure(vineSnake)), null, null, null, ...List.filled(5, sure(vineSnake))], 10);
    expect(steps.skip(15).every((s) => s.lockedClass == vineSnake), isTrue);
  });

  test('gate share: a snake on every other frame never locks at 0.8', () {
    final lock = VideoLock(rules: rules, settings: const LockSettings(gateShare: 0.8));
    final frames = [for (var i = 0; i < 60; i++) i.isEven ? sure(vineSnake) : null];
    expect(run(lock, frames, 10).every((s) => s.checking), isTrue);
    final open = run(VideoLock(rules: rules), frames, 10);
    expect(open.last.checking, isFalse, reason: 'with no gate share the same frames lock');
  });

  test('not sure enough never locks', () {
    final steps = run(VideoLock(rules: rules), List.filled(40, sure(vineSnake, p: 0.2)), 10);
    expect(steps.every((s) => s.checking && s.lockedClass == null), isTrue);
    expect(steps.every((s) => s.shown == null || s.shown!.top3.isEmpty), isTrue, reason: 'no species is named');
  });

  test('dangerous shows while checking, with no species named', () {
    final steps = run(VideoLock(rules: rules), List.filled(5, sure(cobra)), 10);
    final first = steps.first;
    expect(first.checking, isTrue);
    expect(first.shown?.label, dangerousLabel);
    expect(first.shown!.top3, isEmpty);
  });

  test('dangerous on the pool turns a locked safe answer dangerous, the species stays', () {
    final lock = VideoLock(rules: rules);
    final frames = [...List.filled(15, sure(vineSnake)), ...List.filled(5, sure(cobra, p: 0.95))];
    final steps = run(lock, frames, 10);
    expect(steps[14].shown!.label, isNot(dangerousLabel));
    final after = steps.skip(15).toList();
    expect(after.every((s) => s.lockedClass == vineSnake), isTrue);
    expect(after.last.shown!.label, dangerousLabel);
    expect(species(after.last), vineSnake);
  });

  test('reset clears the lock', () {
    final lock = VideoLock(rules: rules);
    run(lock, List.filled(20, sure(vineSnake)), 10);
    expect(lock.lockedClass, vineSnake);
    lock.reset();
    expect(lock.lockedClass, isNull);
    expect(lock.add(null, 3).checking, isTrue);
  });
}
