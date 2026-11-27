import 'dart:io';
import 'dart:isolate';
import 'dart:math' as math;

import 'package:crypto/crypto.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'package:http/http.dart' as http;
import 'package:image/image.dart' as img;
import 'package:path_provider/path_provider.dart';

import 'anidex_catalog.dart';

/// One scored photo.
class AniDexScan {
  const AniDexScan({
    required this.species,
    required this.score,
    required this.alternatives,
    required this.breeds,
    required this.breed,
  });

  final AniDexSpecies species;
  final double score;

  /// The next two species and their scores.
  final List<(AniDexSpecies, double)> alternatives;

  /// The top three breeds of [species] (empty for animals without breeds),
  /// scores renormalised over that species' breeds.
  final List<(AniDexBreed, double)> breeds;

  /// The top breed, when it clears the catalog's breed bar.
  final (AniDexBreed, double)? breed;
}

/// BioCLIP 2's image encoder, run on the phone, scored against the
/// catalog's precomputed label vectors. Zero-shot: no training on these
/// animals, and the text encoder isn't needed on the phone.
class AniDexModel {
  AniDexModel._(this.catalog, this._session, this._breedSession);

  final AniDexCatalog catalog;

  /// BioCLIP 2 (species) and a general CLIP (breeds).
  final OrtSession _session, _breedSession;

  /// Opens both encoders. [onDownload] reports progress (0 to 1) when they
  /// have to be downloaded first.
  static Future<AniDexModel> open(AniDexCatalog catalog, {void Function(double)? onDownload}) async {
    final ort = OnnxRuntime();
    final options = OrtSessionOptions(intraOpNumThreads: 4, interOpNumThreads: 1);
    final species = await ort.createSession(
      await _encoderPath(catalog.encoder, (p) => onDownload?.call(p * 0.77)),
      options: options,
    );
    try {
      final breeds = await ort.createSession(
        await _encoderPath(catalog.breedEncoder, (p) => onDownload?.call(0.77 + p * 0.23)),
        options: options,
      );
      return AniDexModel._(catalog, species, breeds);
    } catch (_) {
      await species.close();
      rethrow;
    }
  }

  /// Whether both encoders are in the app bundle or already on the phone.
  static Future<bool> available(AniDexCatalog catalog) async {
    for (final e in [catalog.encoder, catalog.breedEncoder]) {
      if (!await _bundled(e) && !await (await _localFile(e)).exists()) return false;
    }
    return true;
  }

  static Future<File> _localFile(EncoderFile e) async {
    final dir = Directory('${(await getApplicationSupportDirectory()).path}/models');
    await dir.create(recursive: true);
    // The checksum in the name: a new build of the encoder gets a new file.
    final stem = e.file.replaceAll('.onnx', '');
    return File('${dir.path}/$stem-${e.sha256.substring(0, 12)}.onnx');
  }

  static Future<bool> _bundled(EncoderFile e) async {
    try {
      final manifest = await AssetManifest.loadFromAssetBundle(rootBundle);
      return manifest.listAssets().contains('${AniDexCatalog.dir}/${e.file}');
    } catch (_) {
      return false;
    }
  }

  static Future<String> _encoderPath(EncoderFile e, void Function(double)? onDownload) async {
    final file = await _localFile(e);
    if (await file.exists()) return file.path;
    final tmp = File('${file.path}.part');
    if (await _bundled(e)) {
      // ONNX Runtime needs a file path: copy the asset out once.
      final data = await rootBundle.load('${AniDexCatalog.dir}/${e.file}');
      await tmp.writeAsBytes(data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes), flush: true);
    } else if (e.url != null) {
      await _download(Uri.parse(e.url!), tmp, onDownload);
    } else {
      throw const AniDexModelMissing();
    }
    final partPath = tmp.path;
    final sha = await Isolate.run(() async => (await sha256.bind(File(partPath).openRead()).first).toString());
    if (sha != e.sha256) {
      await tmp.delete();
      throw StateError('The AniDex model file is damaged (checksum mismatch). Try again.');
    }
    await tmp.rename(file.path);
    return file.path;
  }

  static Future<void> _download(Uri url, File to, void Function(double)? onProgress) async {
    final client = http.Client();
    try {
      final response = await client.send(http.Request('GET', url));
      if (response.statusCode != 200) {
        throw HttpException('Download failed (HTTP ${response.statusCode})', uri: url);
      }
      final total = response.contentLength ?? 0;
      var received = 0;
      final sink = to.openWrite();
      try {
        await for (final chunk in response.stream) {
          sink.add(chunk);
          received += chunk.length;
          if (total > 0) onProgress?.call(received / total);
        }
      } finally {
        await sink.close();
      }
    } finally {
      client.close();
    }
  }

  /// Scores one photo (JPEG or PNG bytes): the species, then, for an animal
  /// with breeds, the breed.
  Future<AniDexScan> scan(Uint8List photo) async {
    final c = catalog;
    final size = c.inputSize, mean = c.mean, std = c.std;
    final clock = Stopwatch()..start();
    // Both encoders take the same stretched input.
    final input = await Isolate.run(() => preprocess(photo, size, mean, std));
    final prepared = clock.elapsedMilliseconds;
    final value = await OrtValue.fromList(input, [1, 3, size, size]);
    AniDexScan result;
    var breedMs = 0;
    try {
      final embedding = await _embed(_session, value);
      result = score(c, embedding);
      if (result.species.hasBreeds) {
        final t = clock.elapsedMilliseconds;
        result = score(c, embedding, breedEmbedding: await _embed(_breedSession, value));
        breedMs = clock.elapsedMilliseconds - t;
      }
    } finally {
      await value.dispose();
    }
    debugPrint('AniDex scan: preprocess ${prepared}ms, species+breed models '
        '${clock.elapsedMilliseconds - prepared - breedMs}+${breedMs}ms -> ${result.species.name} '
        '${(result.score * 100).round()}% ${result.breed?.$1.name ?? ''}');
    return result;
  }

  static Future<List<double>> _embed(OrtSession session, OrtValue input) async {
    final outputs = await session.run({'image': input});
    try {
      return [for (final v in await outputs['embedding']!.asFlattenedList()) (v as num).toDouble()];
    } finally {
      for (final o in outputs.values) {
        await o.dispose();
      }
    }
  }

  Future<void> close() async {
    await _session.close();
    await _breedSession.close();
  }
}

class AniDexModelMissing implements Exception {
  const AniDexModelMissing();
  @override
  String toString() => 'The AniDex model is not in this build.';
}

/// Decodes a photo upright and turns it into the encoder's input: stretched
/// to size x size (as BioCLIP's own preprocessing does), RGB in 0..1,
/// normalised with CLIP's mean and std, channels first.
Float32List preprocess(Uint8List bytes, int size, List<double> mean, List<double> std) {
  final decoded = img.decodeImage(bytes);
  if (decoded == null) throw const FormatException('Could not read this photo.');
  final upright = img.bakeOrientation(decoded);
  final w = upright.width, h = upright.height;
  final rgb = upright.getBytes(order: img.ChannelOrder.rgb);
  final plane = size * size;
  final out = Float32List(3 * plane);
  final across = _weights(w, size), down = _weights(h, size);
  // Separable resize in floats on 0..1 values, as torchvision does after
  // ToTensor: first along each row, then down each column.
  final rows = Float32List(h * size);
  for (var ch = 0; ch < 3; ch++) {
    for (var y = 0; y < h; y++) {
      final base = y * w;
      for (var x = 0; x < size; x++) {
        final (start, ws) = across[x];
        var v = 0.0;
        for (var k = 0; k < ws.length; k++) {
          v += ws[k] * rgb[(base + start + k) * 3 + ch];
        }
        rows[y * size + x] = v / 255.0;
      }
    }
    for (var y = 0; y < size; y++) {
      final (start, ws) = down[y];
      for (var x = 0; x < size; x++) {
        var v = 0.0;
        for (var k = 0; k < ws.length; k++) {
          v += ws[k] * rows[(start + k) * size + x];
        }
        out[ch * plane + y * size + x] = (v - mean[ch]) / std[ch];
      }
    }
  }
  return out;
}

/// Antialiased bilinear (triangle filter) weights for resizing [inSize] to
/// [outSize], the same as PIL's and torchvision's `Resize(antialias=True)`:
/// for each output pixel, the first input pixel and the normalised weights.
List<(int, Float64List)> _weights(int inSize, int outSize) {
  final scale = inSize / outSize;
  final filterScale = math.max(scale, 1.0);
  final support = filterScale; // the triangle filter's support is 1
  return [
    for (var o = 0; o < outSize; o++)
      () {
        final centre = (o + 0.5) * scale;
        final lo = math.max((centre - support + 0.5).floor(), 0);
        final hi = math.min((centre + support + 0.5).floor(), inSize);
        final ws = Float64List(hi - lo);
        var total = 0.0;
        for (var i = lo; i < hi; i++) {
          final t = ((i - centre + 0.5) / filterScale).abs();
          ws[i - lo] = t < 1 ? 1 - t : 0;
          total += ws[i - lo];
        }
        for (var k = 0; k < ws.length; k++) {
          ws[k] /= total;
        }
        return (lo, ws);
      }(),
  ];
}

/// softmax(scale * rows . e / |e|) over [count] rows starting at row [first]
/// of a row-major [rows] matrix with [dim] columns.
Float64List _softmaxScores(Float32List rows, int dim, int first, int count, List<double> e, double scale) {
  var norm = 0.0;
  for (final v in e) {
    norm += v * v;
  }
  norm = math.sqrt(norm);
  final out = Float64List(count);
  var top = double.negativeInfinity;
  for (var l = 0; l < count; l++) {
    var dot = 0.0;
    final row = (first + l) * dim;
    for (var k = 0; k < dim; k++) {
      dot += rows[row + k] * e[k];
    }
    out[l] = scale * dot / norm;
    top = math.max(top, out[l]);
  }
  var sum = 0.0;
  for (var l = 0; l < count; l++) {
    out[l] = math.exp(out[l] - top);
    sum += out[l];
  }
  for (var l = 0; l < count; l++) {
    out[l] /= sum;
  }
  return out;
}

/// Scores BioCLIP's [embedding] against every species label and adds each
/// species' label scores. With [breedEmbedding] (the general CLIP's, for a
/// species with breeds), also ranks that species' breeds.
AniDexScan score(AniDexCatalog c, List<double> embedding, {List<double>? breedEmbedding}) {
  final probs = _softmaxScores(c.labels, c.dim, 0, c.labelCount, embedding, c.logitScale);
  final speciesProbs = Float64List(c.species.length);
  for (var l = 0; l < c.labelCount; l++) {
    speciesProbs[c.labelOwner[l]] += probs[l];
  }
  final order = List<int>.generate(c.species.length, (i) => i)
    ..sort((a, b) => speciesProbs[b].compareTo(speciesProbs[a]));
  final winner = c.species[order.first];

  var breeds = <(AniDexBreed, double)>[];
  if (winner.hasBreeds && breedEmbedding != null) {
    final p = _softmaxScores(
        c.breedLabels, c.breedDim, winner.breedVectorsStart!, winner.breeds.length, breedEmbedding, c.logitScale);
    breeds = [for (var i = 0; i < p.length; i++) (winner.breeds[i], p[i])]..sort((a, b) => b.$2.compareTo(a.$2));
    breeds = breeds.take(3).toList();
  }
  return AniDexScan(
    species: winner,
    score: speciesProbs[order.first],
    alternatives: [for (final i in order.skip(1).take(2)) (c.species[i], speciesProbs[i])],
    breeds: breeds,
    breed: breeds.isNotEmpty && breeds.first.$2 >= c.breedMin ? breeds.first : null,
  );
}
