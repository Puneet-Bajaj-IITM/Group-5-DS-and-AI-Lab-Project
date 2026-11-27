import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:serpentguard/inference/image_ops.dart';
import 'package:serpentguard/services/classifier.dart';

const _portrait = 'portraitUp';

LiveCameraFrame _frame(String format, int w, int h, List<LiveCameraPlane> planes,
        {int sensor = 0, String device = _portrait, bool front = false}) =>
    LiveCameraFrame(
      width: w,
      height: h,
      format: format,
      planes: planes,
      offsetMs: 0,
      sensorOrientation: sensor,
      deviceOrientation: device,
      frontFacing: front,
    );

List<int> _px(RgbImage img, int x, int y) => img.data.sublist((y * img.width + x) * 3, (y * img.width + x) * 3 + 3);

void main() {
  test('BGRA: channels swapped to RGB, row stride honoured', () {
    // 2 x 1 image, rows padded to 12 bytes
    final bytes = Uint8List.fromList([10, 20, 30, 255, 40, 50, 60, 255, 0, 0, 0, 0]);
    final img = cameraFrameToRgb(_frame('bgra8888', 2, 1, [LiveCameraPlane(bytes, 12, 4)]))!;
    expect(img.width, 2);
    expect(_px(img, 0, 0), [30, 20, 10]);
    expect(_px(img, 1, 0), [60, 50, 40]);
  });

  test('YUV 4:2:0, three planes: grey stays grey, pure V gives red', () {
    final y = LiveCameraPlane(Uint8List.fromList([128, 128, 128, 128]), 2, 1);
    final grey = cameraFrameToRgb(_frame('yuv420', 2, 2, [
      y,
      LiveCameraPlane(Uint8List.fromList([128]), 1, 1),
      LiveCameraPlane(Uint8List.fromList([128]), 1, 1),
    ]))!;
    expect(_px(grey, 1, 1), [128, 128, 128]);
    final red = cameraFrameToRgb(_frame('yuv420', 2, 2, [
      y,
      LiveCameraPlane(Uint8List.fromList([128]), 1, 1),
      LiveCameraPlane(Uint8List.fromList([228]), 1, 1),
    ]))!;
    final p = _px(red, 0, 0);
    expect(p[0], 255, reason: "128 + 1.402 * 100 clamps to 255");
    expect(p[1], (128 - 0.714136 * 100).round());
    expect(p[2], 128);
  });

  test('NV21: V before U in the interleaved plane', () {
    // 2 x 2 luma, then one V,U pair
    final bytes = Uint8List.fromList([100, 100, 100, 100, 228, 128]);
    final img = cameraFrameToRgb(_frame('nv21', 2, 2, [LiveCameraPlane(bytes, 2, 1)]))!;
    final p = _px(img, 0, 0);
    expect(p[0], (100 + 1.402 * 100).round());
    expect(p[2], 100);
  });

  test('a 90 degree sensor turns the frame upright; the front camera is mirrored', () {
    // 2 x 1 BGRA: left red, right blue
    final bytes = Uint8List.fromList([0, 0, 255, 255, 255, 0, 0, 255]);
    final plane = [LiveCameraPlane(bytes, 8, 4)];
    final back = cameraFrameToRgb(_frame('bgra8888', 2, 1, plane, sensor: 90))!;
    expect([back.width, back.height], [1, 2]);
    expect(_px(back, 0, 0), [255, 0, 0]);
    expect(_px(back, 0, 1), [0, 0, 255]);
    final flat = cameraFrameToRgb(_frame('bgra8888', 2, 1, plane, front: true))!;
    expect(_px(flat, 0, 0), [0, 0, 255], reason: 'mirrored');
  });

  test('an unknown format gives null', () {
    expect(cameraFrameToRgb(_frame('jpeg', 1, 1, [LiveCameraPlane(Uint8List(3), 3, 3)])), isNull);
  });
}
