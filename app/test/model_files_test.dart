import 'dart:io';
import 'dart:typed_data';

import 'package:crypto/crypto.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:serpentguard/inference/bundle.dart';
import 'package:serpentguard/inference/pipeline.dart';

/// The model files: the copy the app runs is the bundled one (content, not
/// size), a file pipeline.json pins is refused when it differs, and the
/// files in assets/models are the ones the M5 settings were made for.
void main() {
  final settings = PipelineSettings.parse(File('assets/model/pipeline.json').readAsStringSync());
  // The M4 bundle's classifier (A3). M5_KD is the same architecture and size.
  const a3Classifier = '355a975449e48b0b2dd7d633641132cf90c68d5d79a8974849b01b6c4651252e';
  late Directory dir;

  setUp(() => dir = Directory.systemTemp.createTempSync('sg_models_'));
  tearDown(() => dir.deleteSync(recursive: true));

  Uint8List bytes(int fill, [int n = 4096]) => Uint8List(n)..fillRange(0, n, fill);
  String hash(Uint8List b) => sha256.convert(b).toString();

  test('a new model of the same size replaces the old copy', () async {
    final file = File('${dir.path}/classifier.onnx');
    final old = bytes(1), update = bytes(2);
    await SnakePipeline.copyModel(old, file);
    expect(file.readAsBytesSync(), old);
    await SnakePipeline.copyModel(update, file);
    expect(file.readAsBytesSync(), update, reason: 'same size, other contents: copied again');
    expect(File('${file.path}.sha256').readAsStringSync(), hash(update));
  });

  test('an unchanged model is not copied again', () async {
    final file = File('${dir.path}/detector.onnx');
    await SnakePipeline.copyModel(bytes(3), file);
    final written = file.lastModifiedSync();
    await Future<void>.delayed(const Duration(milliseconds: 20));
    await SnakePipeline.copyModel(bytes(3), file);
    expect(file.lastModifiedSync(), written);
  });

  test('a copy without its hash stamp (an older app build) is replaced', () async {
    final file = File('${dir.path}/classifier.onnx')..writeAsBytesSync(bytes(1));
    await SnakePipeline.copyModel(bytes(2), file);
    expect(file.readAsBytesSync(), bytes(2));
  });

  test('a model other than the one pipeline.json names is refused, and nothing is copied', () async {
    final file = File('${dir.path}/snake_check_fp16.onnx');
    expect(
      () => SnakePipeline.copyModel(bytes(4), file, sha256: hash(bytes(5))),
      throwsA(isA<ModelMismatch>().having((e) => e.file, 'file', 'snake_check_fp16.onnx')),
    );
    await pumpEventQueue();
    expect(file.existsSync(), isFalse);
    expect(await SnakePipeline.copyModel(bytes(5), file, sha256: hash(bytes(5))), file.path);
  });

  test('pipeline.json pins the frozen M4 front end and the M5 snake check (FREEZE.md)', () {
    expect(settings.modelSha256, {
      'detector.onnx': '1d2da244575a39c23ebab6ee2f4f04b1c672794031a56fab9d49dc5176f4f9ba',
      'segmenter.onnx': 'da5d59baca34eda5993d837673449b99ccd178c98bbd6755a6b20bbea6f99269',
      'snake_check_fp16.onnx': 'fb5b5bda5e0dbdb1895448ae1c70e56e93042ef322f9e5595cfcd6e0b8ec9fc3',
      if (settings.modelSha256.containsKey(classifierFile)) classifierFile: settings.modelSha256[classifierFile],
    });
    expect(settings.modelSha256[classifierFile], isNot(a3Classifier));
  });

  test('the model files in assets/models match pipeline.json', () {
    final present = {
      for (final f in modelFiles)
        if (File('assets/models/$f').existsSync()) f: sha256.convert(File('assets/models/$f').readAsBytesSync()).toString(),
    };
    for (final MapEntry(key: f, value: digest) in present.entries) {
      final want = settings.modelSha256[f];
      if (want != null) expect(digest, want, reason: '$f is not the file pipeline.json names');
    }
    // With all four files the app runs for real, so the classifier must not
    // be A3 (until its own hash is pinned above).
    if (present.length == modelFiles.length) {
      expect(present[classifierFile], isNot(a3Classifier),
          reason: 'assets/models/classifier.onnx is A3; the M5 settings need M5_KD');
    }
  });
}
