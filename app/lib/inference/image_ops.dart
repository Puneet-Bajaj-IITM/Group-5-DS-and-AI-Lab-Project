import 'dart:math' as math;
import 'dart:typed_data';

/// The image steps of `reference_pipeline.py`, in plain Dart.
///
/// The reference uses OpenCV. These functions follow OpenCV's algorithms
/// (resize INTER_LINEAR and INTER_AREA, CLAHE, 5x5 ellipse closing,
/// 8-connected components) so a photo gives the same pixels to within a
/// grey level or so. OpenCV itself differs that much between platforms
/// (IPP on x86, NEON on ARM), and `test/image_ops_test.dart` checks these
/// against cv2 output. Plain Dart keeps `flutter test` free of native builds.

/// An 8-bit RGB image, rows top to bottom, pixels interleaved R, G, B.
class RgbImage {
  RgbImage(this.width, this.height, this.data) : assert(data.length == width * height * 3);

  factory RgbImage.filled(int width, int height, List<int> rgb) {
    final d = Uint8List(width * height * 3);
    for (var i = 0; i < d.length; i += 3) {
      d[i] = rgb[0];
      d[i + 1] = rgb[1];
      d[i + 2] = rgb[2];
    }
    return RgbImage(width, height, d);
  }

  /// From RGBA bytes (Flutter's decoded image format).
  factory RgbImage.fromRgba(int width, int height, Uint8List rgba) {
    final d = Uint8List(width * height * 3);
    for (var p = 0, q = 0; q < d.length; p += 4, q += 3) {
      d[q] = rgba[p];
      d[q + 1] = rgba[p + 1];
      d[q + 2] = rgba[p + 2];
    }
    return RgbImage(width, height, d);
  }

  final int width, height;
  final Uint8List data;

  /// Mean of every pixel value, all channels (numpy's `rgb.mean()`).
  double get meanValue {
    var sum = 0;
    for (final v in data) {
      sum += v;
    }
    return sum / data.length;
  }

  /// Rows [top, bottom) and columns [left, right).
  RgbImage crop(int top, int bottom, int left, int right) {
    final w = right - left, h = bottom - top;
    final out = Uint8List(w * h * 3);
    for (var y = 0; y < h; y++) {
      final src = ((top + y) * width + left) * 3;
      out.setRange(y * w * 3, (y + 1) * w * 3, data, src);
    }
    return RgbImage(w, h, out);
  }

  /// The left-right mirror image (numpy's `rgb[:, ::-1]`).
  RgbImage mirrored() {
    final out = Uint8List(data.length);
    for (var y = 0; y < height; y++) {
      final row = y * width * 3;
      for (var x = 0; x < width; x++) {
        final src = row + x * 3, dst = row + (width - 1 - x) * 3;
        out[dst] = data[src];
        out[dst + 1] = data[src + 1];
        out[dst + 2] = data[src + 2];
      }
    }
    return RgbImage(width, height, out);
  }

  /// Rotates clockwise by 0, 90, 180 or 270 degrees.
  RgbImage rotate(int degrees) {
    final d = ((degrees % 360) + 360) % 360;
    if (d == 0) return this;
    final w2 = d == 180 ? width : height, h2 = d == 180 ? height : width;
    final out = Uint8List(data.length);
    for (var y = 0; y < height; y++) {
      for (var x = 0; x < width; x++) {
        final (nx, ny) = switch (d) {
          90 => (height - 1 - y, x),
          180 => (width - 1 - x, height - 1 - y),
          _ => (y, width - 1 - x),
        };
        final s = (y * width + x) * 3, t = (ny * w2 + nx) * 3;
        out[t] = data[s];
        out[t + 1] = data[s + 1];
        out[t + 2] = data[s + 2];
      }
    }
    return RgbImage(w2, h2, out);
  }

  /// Mirror left to right (front camera).
  RgbImage flipHorizontal() {
    final out = Uint8List(data.length);
    for (var y = 0; y < height; y++) {
      for (var x = 0; x < width; x++) {
        final s = (y * width + x) * 3, t = (y * width + (width - 1 - x)) * 3;
        out[t] = data[s];
        out[t + 1] = data[s + 1];
        out[t + 2] = data[s + 2];
      }
    }
    return RgbImage(width, height, out);
  }
}

extension on int {
  /// C integer division (rounds toward zero).
  int truncDiv(int d) => (this / d).truncate();
}

int _sat8(num v) => v < 0 ? 0 : (v > 255 ? 255 : v.toInt());

/// cvRound: round half to even (rint).
int cvRound(double v) {
  final f = v.floorToDouble();
  final d = v - f;
  if (d > 0.5) return f.toInt() + 1;
  if (d < 0.5) return f.toInt();
  return f.toInt().isEven ? f.toInt() : f.toInt() + 1;
}

// ---------------------------------------------------------------- resize

/// OpenCV's coordinate map for INTER_LINEAR: (source index, weight of the next pixel).
(Int32List, Float64List) _linearTaps(int src, int dst) {
  final scale = src / dst;
  final idx = Int32List(dst);
  final frac = Float64List(dst);
  for (var d = 0; d < dst; d++) {
    // OpenCV computes this in float32.
    var f = _f32((d + 0.5) * scale - 0.5);
    var s = f.floor();
    f = _f32(f - s);
    if (s < 0) {
      f = 0;
      s = 0;
    }
    if (s >= src - 1) {
      f = 0;
      s = src - 1;
    }
    idx[d] = s;
    frac[d] = f;
  }
  return (idx, frac);
}

final _f32buf = Float32List(1);
double _f32(double v) {
  _f32buf[0] = v;
  return _f32buf[0];
}

/// OpenCV's bit-exact linear taps: the source coordinate (d + 0.5) * src / dst - 0.5
/// in exact arithmetic, its fraction as an 8-bit weight rounded half up, and
/// clamped at both borders. Returns (index, weight of the next pixel, 0..256).
(Int32List, Int32List) _exactTaps(int src, int dst) {
  final idx = Int32List(dst), wt = Int32List(dst);
  final den = 2 * dst;
  for (var d = 0; d < dst; d++) {
    final num = (2 * d + 1) * src - dst; // coordinate = num / den
    var s = (num / den).floor();
    var w = ((num - s * den) * 512 + den) ~/ (2 * den); // round(frac * 256), half up
    if (w == 256) {
      s += 1;
      w = 0;
    }
    if (s < 0) {
      s = 0;
      w = 0;
    }
    if (s >= src - 1) {
      s = src - 1;
      w = 0;
    }
    idx[d] = s;
    wt[d] = w;
  }
  return (idx, wt);
}

/// cv2.resize(..., INTER_LINEAR) on uint8, as OpenCV's generic bit-exact
/// resize (INTER_LINEAR_EXACT): 8-bit weights, rounding half up. ARM builds of
/// cv2 (carotene) and x86 builds (IPP) round slightly differently; on the
/// golden photos this moves no answer and probabilities by under 0.001.
RgbImage resizeLinear(RgbImage img, int w, int h) {
  if (w == img.width && h == img.height) return img;
  const one = 256;
  final (xs, ax1) = _exactTaps(img.width, w);
  final (ys, ay1) = _exactTaps(img.height, h);
  final src = img.data, sw = img.width, sh = img.height;
  final rowCache = <int, Int32List>{};
  Int32List hrow(int sy) {
    sy = sy >= sh ? sh - 1 : sy;
    return rowCache.putIfAbsent(sy, () {
      final r = Int32List(w * 3);
      final base = sy * sw * 3;
      for (var d = 0; d < w; d++) {
        final s0 = base + xs[d] * 3;
        final s1 = xs[d] + 1 < sw ? s0 + 3 : s0;
        final a1 = ax1[d], a0 = one - a1;
        for (var c = 0; c < 3; c++) {
          r[d * 3 + c] = src[s0 + c] * a0 + src[s1 + c] * a1;
        }
      }
      return r;
    });
  }

  final out = Uint8List(w * h * 3);
  for (var d = 0; d < h; d++) {
    final b1 = ay1[d], b0 = one - b1;
    final r0 = hrow(ys[d]), r1 = hrow(ys[d] + 1);
    final o = d * w * 3;
    for (var i = 0; i < w * 3; i++) {
      out[o + i] = _sat8((r0[i] * b0 + r1[i] * b1 + (1 << 15)) >> 16);
    }
    if (rowCache.length > 4) rowCache.removeWhere((k, _) => k < ys[d] - 1);
  }
  return RgbImage(w, h, out);
}

/// cv2.resize(..., INTER_AREA) on uint8 when shrinking. Integer ratios use
/// OpenCV's block average; other ratios its area-weight tables in float.
RgbImage resizeArea(RgbImage img, int w, int h) {
  if (w == img.width && h == img.height) return img;
  final scaleX = img.width / w, scaleY = img.height / h;
  if (scaleX < 1 || scaleY < 1) return resizeLinear(img, w, h); // OpenCV does the same
  final ix = scaleX.round(), iy = scaleY.round();
  if ((scaleX - ix).abs() < 2.220446049250313e-16 && (scaleY - iy).abs() < 2.220446049250313e-16) {
    return _resizeAreaFast(img, w, h, ix, iy);
  }
  final xt = _areaTab(img.width, w, scaleX);
  final yt = _areaTab(img.height, h, scaleY);
  final src = img.data, sw = img.width;
  final out = Uint8List(w * h * 3);
  final buf = Float64List(w * 3), sum = Float64List(w * 3);
  var prevDy = -1;
  void flush(int dy) {
    for (var i = 0; i < w * 3; i++) {
      out[dy * w * 3 + i] = _sat8(cvRound(_f32(sum[i])));
      sum[i] = 0;
    }
  }

  for (final ty in yt) {
    if (prevDy != -1 && ty.d != prevDy) flush(prevDy);
    prevDy = ty.d;
    buf.fillRange(0, buf.length, 0);
    final base = ty.s * sw * 3;
    for (final tx in xt) {
      final s = base + tx.s * 3, dd = tx.d * 3;
      buf[dd] = _f32(buf[dd] + src[s] * tx.a);
      buf[dd + 1] = _f32(buf[dd + 1] + src[s + 1] * tx.a);
      buf[dd + 2] = _f32(buf[dd + 2] + src[s + 2] * tx.a);
    }
    for (var i = 0; i < w * 3; i++) {
      sum[i] = _f32(sum[i] + ty.a * buf[i]);
    }
  }
  if (prevDy != -1) flush(prevDy);
  return RgbImage(w, h, out);
}

class _Tap {
  const _Tap(this.d, this.s, this.a);
  final int d, s;
  final double a;
}

/// OpenCV's computeResizeAreaTab.
List<_Tap> _areaTab(int ssize, int dsize, double scale) {
  final tab = <_Tap>[];
  for (var dx = 0; dx < dsize; dx++) {
    final fsx1 = dx * scale;
    final fsx2 = fsx1 + scale;
    final cellWidth = math.min(scale, ssize - fsx1);
    var sx1 = fsx1.ceil();
    var sx2 = fsx2.floor();
    sx2 = math.min(sx2, ssize - 1);
    sx1 = math.min(sx1, sx2);
    if (sx1 - fsx1 > 1e-3) tab.add(_Tap(dx, sx1 - 1, _f32((sx1 - fsx1) / cellWidth)));
    for (var sx = sx1; sx < sx2; sx++) {
      tab.add(_Tap(dx, sx, _f32(1.0 / cellWidth)));
    }
    if (fsx2 - sx2 > 1e-3) {
      tab.add(_Tap(dx, sx2, _f32(math.min(math.min(fsx2 - sx2, 1.0), cellWidth) / cellWidth)));
    }
  }
  return tab;
}

RgbImage _resizeAreaFast(RgbImage img, int w, int h, int sx, int sy) {
  final out = Uint8List(w * h * 3);
  final src = img.data, sw = img.width;
  final area = sx * sy;
  for (var y = 0; y < h; y++) {
    for (var x = 0; x < w; x++) {
      for (var c = 0; c < 3; c++) {
        var s = 0;
        for (var j = 0; j < sy; j++) {
          for (var i = 0; i < sx; i++) {
            s += src[((y * sy + j) * sw + x * sx + i) * 3 + c];
          }
        }
        // 2x2 uses OpenCV's SIMD (s + 2) >> 2; other factors cvRound(s / area).
        out[(y * w + x) * 3 + c] = area == 4 ? (s + 2) >> 2 : _sat8(cvRound(_f32(s * _f32(1 / area))));
      }
    }
  }
  return RgbImage(w, h, out);
}

/// cv2.resize of a float32 single-channel map with INTER_LINEAR.
Float32List resizeLinearFloat(Float32List src, int sw, int sh, int w, int h) {
  final (xs, xf) = _linearTaps(sw, w);
  final scaleY = sh / h;
  final out = Float32List(w * h);
  final row0 = Float32List(w), row1 = Float32List(w);
  void hrow(int sy, Float32List r) {
    sy = sy < 0 ? 0 : (sy >= sh ? sh - 1 : sy);
    for (var d = 0; d < w; d++) {
      final s0 = sy * sw + xs[d];
      final s1 = xs[d] + 1 < sw ? s0 + 1 : s0;
      r[d] = src[s0] * (1 - xf[d]) + src[s1] * xf[d];
    }
  }

  for (var d = 0; d < h; d++) {
    final f = _f32((d + 0.5) * scaleY - 0.5);
    final s = f.floor();
    final b1 = _f32(f - s), b0 = 1 - b1;
    hrow(s, row0);
    hrow(s + 1, row1);
    for (var x = 0; x < w; x++) {
      out[d * w + x] = row0[x] * b0 + row1[x] * b1;
    }
  }
  return out;
}

// ---------------------------------------------------------------- letterbox

/// Python's round(): halves to the even neighbour.
int pyRound(double v) => cvRound(v);

/// Width and height of the photo inside the size x size letterbox.
(int, int) placedSize(int height, int width, int size) {
  final scale = size / math.max(height, width);
  return (math.max(1, pyRound(width * scale)), math.max(1, pyRound(height * scale)));
}

/// The U-Net's and classifier's input: resized to fit, centred on [pad].
RgbImage letterbox(RgbImage img, int size, List<int> pad) {
  final (nw, nh) = placedSize(img.height, img.width, size);
  final scale = size / math.max(img.height, img.width);
  final resized = scale < 1 ? resizeArea(img, nw, nh) : resizeLinear(img, nw, nh);
  final top = (size - nh) ~/ 2, left = (size - nw) ~/ 2;
  final canvas = RgbImage.filled(size, size, pad);
  for (var y = 0; y < nh; y++) {
    canvas.data.setRange(((top + y) * size + left) * 3, ((top + y) * size + left + nw) * 3,
        resized.data, y * nw * 3);
  }
  return canvas;
}

/// uint8 HWC RGB -> float32 NCHW, (x / scale - mean) / std.
Float32List normalise(RgbImage canvas, List<double> mean, List<double> std, double scale) {
  final n = canvas.width * canvas.height;
  final out = Float32List(3 * n);
  final s = _f32(scale);
  final m = [for (final v in mean) _f32(v)], sd = [for (final v in std) _f32(v)];
  for (var i = 0; i < n; i++) {
    for (var c = 0; c < 3; c++) {
      out[c * n + i] = _f32((_f32(canvas.data[i * 3 + c] / s) - m[c]) / sd[c]);
    }
  }
  return out;
}

/// YOLO's letterbox: INTER_LINEAR, centred with [padValue], RGB / 255, NCHW.
Float32List detectorInput(RgbImage img, int size, int padValue) {
  final ratio = math.min(size / img.height, size / img.width);
  final nw = math.max(pyRound(img.width * ratio), 1), nh = math.max(pyRound(img.height * ratio), 1);
  final resized = (img.width, img.height) != (nw, nh) ? resizeLinear(img, nw, nh) : img;
  final dw = (size - nw) / 2, dh = (size - nh) / 2;
  final top = pyRound(dh - 0.1), left = pyRound(dw - 0.1);
  final n = size * size;
  final out = Float32List(3 * n);
  final padF = _f32(padValue / 255.0);
  out.fillRange(0, out.length, padF);
  for (var y = 0; y < nh; y++) {
    for (var x = 0; x < nw; x++) {
      final p = (top + y) * size + left + x;
      final s = (y * nw + x) * 3;
      out[p] = _f32(resized.data[s] / 255.0);
      out[n + p] = _f32(resized.data[s + 1] / 255.0);
      out[2 * n + p] = _f32(resized.data[s + 2] / 255.0);
    }
  }
  return out;
}

/// The highest snake score, or null when none is above [floor] (strictly).
double? boxConf(List<double> output, int anchors, int snakeClass, double floor) {
  final row = 4 + snakeClass;
  var best = double.negativeInfinity;
  for (var i = row * anchors; i < (row + 1) * anchors; i++) {
    if (output[i] > best) best = output[i];
  }
  return best > floor ? best : null;
}

/// A box in image pixels.
class PixelBox {
  const PixelBox(this.left, this.top, this.right, this.bottom);
  final double left, top, right, bottom;
  double get width => right - left;
  double get height => bottom - top;
}

/// The detector's most confident box, mapped from its letterboxed
/// [size] x [size] input back to the [width] x [height] image. The output
/// holds, per anchor, rows cx, cy, w and h (input pixels) and the score.
PixelBox? bestBox(List<double> output, int anchors, int snakeClass, int width, int height, int size) {
  final row = 4 + snakeClass;
  var best = -1;
  var bestScore = double.negativeInfinity;
  for (var i = 0; i < anchors; i++) {
    final v = output[row * anchors + i];
    if (v > bestScore) {
      bestScore = v;
      best = i;
    }
  }
  if (best < 0) return null;
  final ratio = math.min(size / height, size / width);
  final nw = math.max(pyRound(width * ratio), 1), nh = math.max(pyRound(height * ratio), 1);
  final left = pyRound((size - nw) / 2 - 0.1), top = pyRound((size - nh) / 2 - 0.1);
  final cx = output[best], cy = output[anchors + best];
  final w = output[2 * anchors + best], h = output[3 * anchors + best];
  double clampX(double v) => v.clamp(0, width.toDouble()).toDouble();
  double clampY(double v) => v.clamp(0, height.toDouble()).toDouble();
  return PixelBox(
    clampX((cx - w / 2 - left) / ratio),
    clampY((cy - h / 2 - top) / ratio),
    clampX((cx + w / 2 - left) / ratio),
    clampY((cy + h / 2 - top) / ratio),
  );
}

/// The U-Net output for the placed window, resized to the photo and thresholded.
Uint8List unletterboxMask(Float32List probability, int height, int width, int size, double threshold) {
  final (nw, nh) = placedSize(height, width, size);
  final top = (size - nh) ~/ 2, left = (size - nw) ~/ 2;
  final window = Float32List(nw * nh);
  for (var y = 0; y < nh; y++) {
    window.setRange(y * nw, (y + 1) * nw, probability, (top + y) * size + left);
  }
  final full = resizeLinearFloat(window, nw, nh, width, height);
  final mask = Uint8List(width * height);
  for (var i = 0; i < mask.length; i++) {
    mask[i] = full[i] > threshold ? 1 : 0;
  }
  return mask;
}

// ---------------------------------------------------------------- mask cleanup

/// OpenCV's getStructuringElement(MORPH_ELLIPSE, (5, 5)).
const _ellipse5 = [
  [0, 0, 1, 0, 0],
  [1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1],
  [0, 0, 1, 0, 0],
];

Uint8List _morph(Uint8List m, int w, int h, {required bool dilate}) {
  final out = Uint8List(m.length);
  for (var y = 0; y < h; y++) {
    for (var x = 0; x < w; x++) {
      var v = dilate ? 0 : 1;
      outer:
      for (var ky = 0; ky < 5; ky++) {
        final yy = y + ky - 2;
        if (yy < 0 || yy >= h) continue; // default border: never changes the result
        for (var kx = 0; kx < 5; kx++) {
          if (_ellipse5[ky][kx] == 0) continue;
          final xx = x + kx - 2;
          if (xx < 0 || xx >= w) continue;
          final p = m[yy * w + xx];
          if (dilate && p != 0) {
            v = 1;
            break outer;
          }
          if (!dilate && p == 0) {
            v = 0;
            break outer;
          }
        }
      }
      out[y * w + x] = v;
    }
  }
  return out;
}

/// 8- or 4-connected components. Returns (labels, areas); label 0 is background.
(Int32List, List<int>) connectedComponents(Uint8List m, int w, int h, {int connectivity = 8}) {
  final labels = Int32List(m.length);
  final areas = <int>[0];
  final stack = <int>[];
  for (var i = 0; i < m.length; i++) {
    if (m[i] == 0 || labels[i] != 0) continue;
    final label = areas.length;
    var area = 0;
    labels[i] = label;
    stack.add(i);
    while (stack.isNotEmpty) {
      final p = stack.removeLast();
      area++;
      final py = p ~/ w, px = p % w;
      for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
          if (dx == 0 && dy == 0) continue;
          if (connectivity == 4 && dx != 0 && dy != 0) continue;
          final ny = py + dy, nx = px + dx;
          if (ny < 0 || ny >= h || nx < 0 || nx >= w) continue;
          final q = ny * w + nx;
          if (m[q] != 0 && labels[q] == 0) {
            labels[q] = label;
            stack.add(q);
          }
        }
      }
    }
    areas.add(area);
  }
  return (labels, areas);
}

/// The reference's clean_mask: closed, small parts dropped; null when the
/// largest part is under [minLargestShare] of the photo.
Uint8List? cleanMask(
  Uint8List mask,
  int w,
  int h, {
  required int connectivity,
  required double minLargestShare,
  required double keepShareOfLargest,
  required double keepShareOfPhoto,
}) {
  final closed = _morph(_morph(mask, w, h, dilate: true), w, h, dilate: false);
  final (labels, areas) = connectedComponents(closed, w, h, connectivity: connectivity);
  if (areas.length <= 1) return null;
  final largest = areas.skip(1).reduce(math.max);
  final total = w * h;
  if (largest < minLargestShare * total) return null;
  final floor = math.max(keepShareOfLargest * largest, keepShareOfPhoto * total);
  final keep = [for (final a in areas) a >= floor];
  keep[0] = false;
  final out = Uint8List(total);
  for (var i = 0; i < total; i++) {
    if (keep[labels[i]]) out[i] = 1;
  }
  return out;
}

/// (top, bottom, left, right) of the mask's box plus padding, or null for the whole photo.
(int, int, int, int)? cropWindow(Uint8List? mask, int height, int width, double padding) {
  if (mask == null) return null;
  var x1 = width, y1 = height, x2 = -1, y2 = -1;
  for (var y = 0; y < height; y++) {
    for (var x = 0; x < width; x++) {
      if (mask[y * width + x] == 0) continue;
      if (x < x1) x1 = x;
      if (x > x2) x2 = x;
      if (y < y1) y1 = y;
      if (y > y2) y2 = y;
    }
  }
  if (x2 < 0) return null;
  if (x2 - x1 < 1 || y2 - y1 < 1) return null;
  final padX = ((x2 - x1) * padding).toInt(), padY = ((y2 - y1) * padding).toInt();
  return (
    math.max(0, y1 - padY),
    math.min(height, y2 + padY + 1),
    math.max(0, x1 - padX),
    math.min(width, x2 + padX + 1),
  );
}

// ---------------------------------------------------------------- exposure fix

// OpenCV's 8-bit LAB conversions (color_lab.cpp, bit-exact integer paths).
// Checked over every one of the 16.7M colours against cv2 5.0: LAB -> RGB is
// exact; RGB -> LAB is exact on L and one level off on 0.003% of a and b.

const _labShift = 12, _gammaShift = 3, _labShift2 = _labShift + _gammaShift;
const _rgb2xyz = [0.412453, 0.357580, 0.180423, 0.212671, 0.715160, 0.072169, 0.019334, 0.119193, 0.950227];
const _xyz2rgb = [3.240479, -1.53715, -0.498535, -0.969256, 1.875991, 0.041556, 0.055648, -0.204043, 1.057311];
const _d65 = [0.950456, 1.0, 1.088754];

int _descale(int x, int n) => (x + (1 << (n - 1))) >> n;

final Int32List _gammaTab = Int32List.fromList([
  for (var i = 0; i < 256; i++)
    cvRound(255.0 * (1 << _gammaShift) *
        (i / 255.0 <= 0.04045 ? (i / 255.0) / 12.92 : math.pow((i / 255.0 + 0.055) / 1.055, 2.4)))
]);

final Int32List _cbrtTab = Int32List.fromList([
  for (var i = 0; i < 256 * 3 ~/ 2 * (1 << _gammaShift); i++)
    () {
      final x = _f32(i / (255.0 * (1 << _gammaShift)));
      final v = x < 0.008856 ? x * 7.787 + 0.13793103448275862 : math.pow(x, 1 / 3).toDouble();
      return cvRound((1 << _labShift2) * v);
    }()
]);

final List<int> _labCoeffs = [
  for (var r = 0; r < 3; r++)
    for (var c = 0; c < 3; c++) cvRound((1 << _labShift) * _rgb2xyz[r * 3 + c] / _d65[r])
];

/// cv2.cvtColor(rgb, COLOR_RGB2LAB) for uint8: (L, a, b) planes.
(Uint8List, Uint8List, Uint8List) rgbToLab(RgbImage img) {
  final n = img.width * img.height;
  final l = Uint8List(n), a = Uint8List(n), b = Uint8List(n);
  final c = _labCoeffs;
  const lScale = (116 * 255 + 50) ~/ 100;
  const lShift = -((16 * 255 * (1 << _labShift2) + 50) ~/ 100);
  for (var i = 0; i < n; i++) {
    final r = _gammaTab[img.data[i * 3]], g = _gammaTab[img.data[i * 3 + 1]], bl = _gammaTab[img.data[i * 3 + 2]];
    final fx = _cbrtTab[_descale(r * c[0] + g * c[1] + bl * c[2], _labShift)];
    final fy = _cbrtTab[_descale(r * c[3] + g * c[4] + bl * c[5], _labShift)];
    final fz = _cbrtTab[_descale(r * c[6] + g * c[7] + bl * c[8], _labShift)];
    l[i] = _sat8(_descale(lScale * fy + lShift, _labShift2));
    a[i] = _sat8(_descale(500 * (fx - fy) + 128 * (1 << _labShift2), _labShift2));
    b[i] = _sat8(_descale(200 * (fy - fz) + 128 * (1 << _labShift2), _labShift2));
  }
  return (l, a, b);
}

const _base = 1 << 14;
final Int32List _labToY = Int32List(256), _labToIfy = Int32List(256);
final Int32List _invGammaTab = Int32List(4096);
final List<int> _rgbCoeffs = [
  for (var r = 0; r < 3; r++)
    for (var c = 0; c < 3; c++) cvRound(4096 * _xyz2rgb[r * 3 + c] * _d65[c])
];
var _lab2rgbReady = false;
void _initLab2Rgb() {
  if (_lab2rgbReady) return;
  for (var i = 0; i < 256; i++) {
    if (i <= 20) {
      _labToY[i] = cvRound(i * _base * 20 * 9 / (17.0 * 29 * 29 * 29));
      _labToIfy[i] = cvRound(_base * (16.0 / 116 + i * 5.0 / (3 * 17 * 29)));
    } else {
      final fy = i * 100.0 * _base / (255 * 116) + 16.0 * _base / 116;
      _labToIfy[i] = cvRound(fy);
      _labToY[i] = cvRound(fy * fy * fy / (_base * _base));
    }
  }
  for (var i = 0; i < 4096; i++) {
    final x = i / 4096.0;
    final g = x <= 0.0031308 ? 12.92 * x : 1.055 * math.pow(x, 1 / 2.4) - 0.055;
    _invGammaTab[i] = cvRound(255.0 * g);
  }
  _lab2rgbReady = true;
}

int _abToXZ(int v) {
  if (v <= 3390) return (v * 108).truncDiv(841) - (_base * 16 ~/ 116 * 108 ~/ 841);
  return (v * v ~/ _base) * v ~/ _base;
}

/// cv2.cvtColor(lab, COLOR_LAB2RGB) for uint8.
RgbImage labToRgb(int width, int height, Uint8List l, Uint8List a, Uint8List b) {
  _initLab2Rgb();
  final n = width * height;
  final out = Uint8List(n * 3);
  final c = _rgbCoeffs;
  for (var i = 0; i < n; i++) {
    final y = _labToY[l[i]], ify = _labToIfy[l[i]];
    final adiv = ((5 * a[i] * 53687 + (1 << 7)) >> 13) - 128 * _base ~/ 500;
    final bdiv = ((b[i] * 41943 + (1 << 4)) >> 9) - 128 * _base ~/ 200 + 1;
    final x = _abToXZ(ify + adiv), z = _abToXZ(ify - bdiv);
    for (var k = 0; k < 3; k++) {
      var v = _descale(c[k * 3] * x + c[k * 3 + 1] * y + c[k * 3 + 2] * z, 14);
      v = v < 0 ? 0 : (v > 4095 ? 4095 : v);
      out[i * 3 + k] = _invGammaTab[v];
    }
  }
  return RgbImage(width, height, out);
}

/// The reference's fix_exposure: CLAHE on the LAB lightness when the mean of
/// all pixel values is outside [below, above].
RgbImage fixExposure(RgbImage img, double below, double above, double clip, (int, int) tiles) {
  final m = img.meanValue;
  if (below <= m && m <= above) return img;
  final (l, a, b) = rgbToLab(img);
  return labToRgb(img.width, img.height, clahe(l, img.width, img.height, clip, tiles), a, b);
}

/// OpenCV's CLAHE on an 8-bit single-channel image.
Uint8List clahe(Uint8List src, int w, int h, double clipLimit, (int, int) tiles) {
  final (tx, ty) = tiles;
  // Pad to a multiple of the tile grid with BORDER_REFLECT_101, as OpenCV does.
  // OpenCV pads both sides unless both divide evenly, so a dimension that
  // divides evenly still gains a whole extra tile's worth of rows or columns.
  final even = w % tx == 0 && h % ty == 0;
  final ew = even ? w : w + tx - w % tx;
  final eh = even ? h : h + ty - h % ty;
  int reflect(int i, int n) {
    if (n == 1) return 0;
    while (i < 0 || i >= n) {
      i = i < 0 ? -i : 2 * n - 2 - i;
    }
    return i;
  }

  final tw = ew ~/ tx, th = eh ~/ ty;
  final tileArea = tw * th;
  final lutScale = _f32(255 / tileArea);
  var clip = (clipLimit * tileArea / 256).toInt();
  clip = math.max(clip, 1);
  final luts = List.generate(tx * ty, (_) => Uint8List(256));
  for (var j = 0; j < ty; j++) {
    for (var i = 0; i < tx; i++) {
      final hist = Int32List(256);
      for (var y = j * th; y < (j + 1) * th; y++) {
        final sy = reflect(y, h);
        for (var x = i * tw; x < (i + 1) * tw; x++) {
          hist[src[sy * w + reflect(x, w)]]++;
        }
      }
      var clipped = 0;
      for (var k = 0; k < 256; k++) {
        if (hist[k] > clip) {
          clipped += hist[k] - clip;
          hist[k] = clip;
        }
      }
      final batch = clipped ~/ 256;
      var residual = clipped - batch * 256;
      for (var k = 0; k < 256; k++) {
        hist[k] += batch;
      }
      if (residual != 0) {
        final step = math.max(256 ~/ residual, 1);
        for (var k = 0; k < 256 && residual > 0; k += step, residual--) {
          hist[k]++;
        }
      }
      var sum = 0;
      final lut = luts[j * tx + i];
      for (var k = 0; k < 256; k++) {
        sum += hist[k];
        lut[k] = _sat8(cvRound(_f32(sum * lutScale)));
      }
    }
  }
  final out = Uint8List(w * h);
  // float32 throughout, as OpenCV's CLAHE_Interpolation_Body.
  final invTw = _f32(1 / tw), invTh = _f32(1 / th);
  final ind1 = Int32List(w), ind2 = Int32List(w);
  final xaP = Float64List(w), xa1P = Float64List(w);
  for (var x = 0; x < w; x++) {
    final txf = _f32(x * invTw - 0.5); // fused multiply-add, as compiled for arm64
    var tx1 = txf.floor();
    final tx2 = math.min(tx1 + 1, tx - 1);
    xaP[x] = _f32(txf - tx1);
    xa1P[x] = _f32(1 - xaP[x]);
    tx1 = math.max(tx1, 0);
    ind1[x] = tx1;
    ind2[x] = tx2;
  }
  for (var y = 0; y < h; y++) {
    final tyf = _f32(y * invTh - 0.5);
    var ty1 = tyf.floor();
    final ty2 = math.min(ty1 + 1, ty - 1);
    final ya = _f32(tyf - ty1), ya1 = _f32(1 - ya);
    ty1 = math.max(ty1, 0);
    for (var x = 0; x < w; x++) {
      final v = src[y * w + x];
      // fma(a, x0, b * x1) at each level, matching cv2 bit for bit
      final top = _f32(luts[ty1 * tx + ind1[x]][v] * xa1P[x] + _f32(luts[ty1 * tx + ind2[x]][v] * xaP[x]));
      final bot = _f32(luts[ty2 * tx + ind1[x]][v] * xa1P[x] + _f32(luts[ty2 * tx + ind2[x]][v] * xaP[x]));
      final res = _f32(top * ya1 + _f32(bot * ya));
      out[y * w + x] = _sat8(cvRound(res));
    }
  }
  return out;
}

/// How sharp [img] is: the mean squared Laplacian of its grey values on a
/// grid every [step] pixels. Higher is sharper; motion blur and missed focus
/// drive it down, which is what the live camera uses it for.
double sharpness(RgbImage img, {int step = 4}) {
  final w = img.width, h = img.height, d = img.data;
  double grey(int x, int y) {
    final i = (y * w + x) * 3;
    return 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
  }

  var sum = 0.0;
  var n = 0;
  for (var y = step; y < h - step; y += step) {
    for (var x = step; x < w - step; x += step) {
      final lap = grey(x - step, y) + grey(x + step, y) + grey(x, y - step) + grey(x, y + step) - 4 * grey(x, y);
      sum += lap * lap;
      n++;
    }
  }
  return n == 0 ? 0 : sum / n;
}
