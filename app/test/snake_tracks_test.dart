import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:serpentguard/inference/bundle.dart';
import 'package:serpentguard/inference/decision.dart';
import 'package:serpentguard/inference/image_ops.dart';
import 'package:serpentguard/inference/pipeline.dart';
import 'package:serpentguard/inference/snake_tracks.dart';

const vineSnake = 1; // Ahaetulla prasina, not WHO-listed
const cobra = 8; // Naja naja, WHO Category 1
const ratSnake = 2; // Ptyas mucosa, not WHO-listed
const fps = 10.0;

void main() {
  final settings = PipelineSettings.parse(File('assets/model/pipeline.json').readAsStringSync());
  final tables = ModelTables.parse(File('assets/model/tables.json').readAsStringSync());
  final rules = DecisionRules(settings.thresholds, tables);

  DecisionParts sure(int species, {double p = 0.9}) {
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

  const left = PixelBox(10, 100, 200, 300), right = PixelBox(400, 100, 600, 300);
  int? species(NamedTrack t) => t.answer == null || t.answer!.top3.isEmpty ? null : t.answer!.top3.first.classId;

  test('two snakes in view keep two names', () {
    final tracks = SnakeTracks(rules);
    late List<NamedTrack> last;
    for (var i = 0; i < 30; i++) {
      last = tracks.update([FrameSnake(left, sure(cobra)), FrameSnake(right, sure(vineSnake))], i / fps);
    }
    expect(last, hasLength(2));
    final byX = {for (final t in last) t.box.left < 300: species(t)};
    expect(byX[true], cobra);
    expect(byX[false], vineSnake);
  });

  test('a snake that arrives later is identified on its own, not given the first one\'s name', () {
    final tracks = SnakeTracks(rules);
    for (var i = 0; i < 20; i++) {
      tracks.update([FrameSnake(left, sure(cobra))], i / fps);
    }
    final arrival = tracks.update([FrameSnake(left, sure(cobra)), FrameSnake(right, sure(ratSnake))], 20 / fps);
    final newcomer = arrival.firstWhere((t) => t.box.left > 300);
    expect(newcomer.answer, isNull, reason: 'still identifying on its first frame');
    late List<NamedTrack> later;
    for (var i = 21; i < 40; i++) {
      later = tracks.update([FrameSnake(left, sure(cobra)), FrameSnake(right, sure(ratSnake))], i / fps);
    }
    expect(species(later.firstWhere((t) => t.box.left > 300)), ratSnake);
    expect(species(later.firstWhere((t) => t.box.left < 300)), cobra);
  });

  test('a boxed but unnamed snake is tracked and stays unnamed', () {
    final tracks = SnakeTracks(rules);
    late List<NamedTrack> last;
    for (var i = 0; i < 20; i++) {
      last = tracks.update([const FrameSnake(left, null)], i / fps);
    }
    expect(last.single.answer, isNull);
  });

  test('sharpness ranks a sharp pattern above its blurred copy', () {
    final sharp = RgbImage(64, 64, Uint8List(64 * 64 * 3));
    for (var y = 0; y < 64; y++) {
      for (var x = 0; x < 64; x++) {
        final v = ((x ~/ 4 + y ~/ 4).isEven) ? 255 : 0;
        sharp.data.fillRange((y * 64 + x) * 3, (y * 64 + x) * 3 + 3, v);
      }
    }
    final flat = RgbImage.filled(64, 64, [128, 128, 128]);
    final blurred = resizeArea(resizeArea(sharp, 8, 8), 64, 64);
    expect(sharpness(sharp), greaterThan(sharpness(blurred)));
    expect(sharpness(flat), 0);
  });

  test('the lean crop is the detector box with no mirror image', () {
    final photo = RgbImage.filled(400, 300, [90, 120, 60]);
    const box = PixelBox(100, 50, 300, 250);
    final post = boxCrops(photo, box, settings);
    expect(post.snake, isTrue);
    expect(post.mirroredInput, isNull);
    expect(post.window, boxWindow(box, 400, 300, settings.cropPadding));
    expect(boxCrops(photo, null, settings).window, isNull, reason: 'no box: the whole photo');
  });
}
