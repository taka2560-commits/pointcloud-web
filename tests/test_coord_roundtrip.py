"""
取り込み座標系変換（測量X/Y入替・Y-up・Z反転）のロスレス往復テスト

ユーザー決定: 座標系変換して取り込んだ点群は、保存時に元の座標系へ逆変換して書き戻す。
本テストは
  (1) 全モードで transform → inverse が浮動小数点でも完全一致すること
  (2) 変換後も ScaledInteger の生整数・スケール定義と points が厳密に整合すること
  (3) FileSaveWorker 経由の保存で、軸ごとにスケール/オフセットが異なる ScaledInteger が
      ビット完全一致（生整数一致）で元の座標系に書き戻されること
  (4) 逆変換を通さず変換後のまま書いた場合（安全網）でも cartesianBounds が実データと一致すること
を検証する。
"""

import os
import tempfile
import unittest
import numpy as np
from pye57 import libe57

from core.e57_io import read_e57_scans, write_e57_scans
from core.processor import (
    _COORD_MODE_TABLE,
    inverse_transform_coordinate_system,
    restore_scan_original_coordinates,
    transform_coordinate_system,
    transform_scan_coordinate_system,
)


def _write_scaled_integer_e57(path: str, raw_ix, raw_iy, raw_iz, scales, offsets, with_bounds: bool = True):
    """軸ごとに異なる scale / offset を持つ ScaledInteger E57 を合成する"""
    n = len(raw_ix)
    imf = libe57.ImageFile(path, "w")
    imf.extensionsAdd("", libe57.E57_V1_0_URI)
    root = imf.root()
    data3d = libe57.VectorNode(imf, True)
    root.set("data3D", data3d)

    scan = libe57.StructureNode(imf)
    scan.set("guid", libe57.StringNode(imf, "{aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee}"))
    proto = libe57.StructureNode(imf)
    proto.set("cartesianX", libe57.ScaledIntegerNode(imf, 0, -100000000, 100000000, scales[0], offsets[0]))
    proto.set("cartesianY", libe57.ScaledIntegerNode(imf, 0, -100000000, 100000000, scales[1], offsets[1]))
    proto.set("cartesianZ", libe57.ScaledIntegerNode(imf, 0, -100000000, 100000000, scales[2], offsets[2]))
    if with_bounds:
        xs = raw_ix * scales[0] + offsets[0]
        ys = raw_iy * scales[1] + offsets[1]
        zs = raw_iz * scales[2] + offsets[2]
        bbox = libe57.StructureNode(imf)
        bbox.set("xMinimum", libe57.FloatNode(imf, float(xs.min()), libe57.E57_DOUBLE))
        bbox.set("xMaximum", libe57.FloatNode(imf, float(xs.max()), libe57.E57_DOUBLE))
        bbox.set("yMinimum", libe57.FloatNode(imf, float(ys.min()), libe57.E57_DOUBLE))
        bbox.set("yMaximum", libe57.FloatNode(imf, float(ys.max()), libe57.E57_DOUBLE))
        bbox.set("zMinimum", libe57.FloatNode(imf, float(zs.min()), libe57.E57_DOUBLE))
        bbox.set("zMaximum", libe57.FloatNode(imf, float(zs.max()), libe57.E57_DOUBLE))
        scan.set("cartesianBounds", bbox)

    codecs = libe57.VectorNode(imf, True)
    pts = libe57.CompressedVectorNode(imf, proto, codecs)
    scan.set("points", pts)
    data3d.append(scan)

    bufs = libe57.VectorSourceDestBuffer()
    bufs.append(libe57.SourceDestBuffer(imf, "cartesianX", raw_ix, n, True, False))
    bufs.append(libe57.SourceDestBuffer(imf, "cartesianY", raw_iy, n, True, False))
    bufs.append(libe57.SourceDestBuffer(imf, "cartesianZ", raw_iz, n, True, False))
    w = pts.writer(bufs)
    w.write(n)
    w.close()
    imf.close()


def _bounds_of(scan):
    b = scan.meta_snapshot["children"]["cartesianBounds"]["children"]
    return np.array(
        [[b["xMinimum"]["value"], b["yMinimum"]["value"], b["zMinimum"]["value"]],
         [b["xMaximum"]["value"], b["yMaximum"]["value"], b["zMaximum"]["value"]]],
        dtype=np.float64,
    )


class TestCoordinateTransformMath(unittest.TestCase):
    """(1) 全モードの可逆性"""

    def test_all_modes_are_exactly_invertible(self):
        rng = np.random.default_rng(42)
        pts = rng.uniform(-50000.0, 50000.0, (1000, 3))
        for mode in _COORD_MODE_TABLE:
            fwd = transform_coordinate_system(pts, mode)
            back = inverse_transform_coordinate_system(fwd, mode)
            np.testing.assert_array_equal(back, pts, err_msg=f"mode={mode} の逆変換が完全一致しません")

    def test_known_mode_semantics(self):
        p = np.array([[10.0, 20.0, 30.0]])
        np.testing.assert_array_equal(transform_coordinate_system(p, "survey_swap"), [[20.0, 10.0, 30.0]])
        np.testing.assert_array_equal(transform_coordinate_system(p, "survey_swap_inv"), [[20.0, -10.0, 30.0]])
        np.testing.assert_array_equal(transform_coordinate_system(p, "y_up_to_z_up"), [[10.0, -30.0, 20.0]])
        np.testing.assert_array_equal(transform_coordinate_system(p, "z_invert"), [[10.0, 20.0, -30.0]])
        self.assertIs(transform_coordinate_system(p, "cartesian"), p)


class TestScanTransformConsistency(unittest.TestCase):
    """(2)(3)(4) E57ScanData に対する変換・復元・保存"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="e57_coord_rt_")
        n = 400
        self.n = n
        rng = np.random.default_rng(2026)
        self.raw_ix = rng.integers(-5_000_000, 5_000_000, n, dtype=np.int64)
        self.raw_iy = rng.integers(-5_000_000, 5_000_000, n, dtype=np.int64)
        self.raw_iz = rng.integers(-500_000, 500_000, n, dtype=np.int64)
        # 軸ごとに異なるスケール・オフセット（公共座標系の大きな値を想定）
        self.scales = (0.001, 0.0025, 0.0001)
        self.offsets = (-35000.0, 12000.0, 55.5)
        self.in_path = os.path.join(self.temp_dir, "scaled_axes_in.e57")
        _write_scaled_integer_e57(self.in_path, self.raw_ix, self.raw_iy, self.raw_iz, self.scales, self.offsets)

    def tearDown(self):
        for root_dir, _, files in os.walk(self.temp_dir, topdown=False):
            for name in files:
                try:
                    os.remove(os.path.join(root_dir, name))
                except Exception:
                    pass
            try:
                os.rmdir(root_dir)
            except Exception:
                pass

    def _expected_points(self):
        return np.column_stack([
            self.raw_ix * self.scales[0] + self.offsets[0],
            self.raw_iy * self.scales[1] + self.offsets[1],
            self.raw_iz * self.scales[2] + self.offsets[2],
        ])

    def test_transform_keeps_scaled_integer_consistency(self):
        """変換後も points == raw_unscaled * scale + offset が軸ごとに厳密に成り立ち、変更済みと判定される"""
        for mode in ("survey_swap", "survey_swap_inv", "y_up_to_z_up", "z_invert"):
            scan = read_e57_scans(self.in_path)[0]
            self.assertFalse(scan.is_coordinates_modified())
            transform_scan_coordinate_system(scan, mode)
            self.assertEqual(scan.coord_mode, mode)
            self.assertTrue(scan.is_coordinates_modified(), f"mode={mode}: 変換後は変更済みと判定されるべき")
            expected = transform_coordinate_system(self._expected_points(), mode)
            np.testing.assert_allclose(scan.points, expected, rtol=0, atol=1e-9)
            for axis, key in enumerate(("cartesianX", "cartesianY", "cartesianZ")):
                sch = scan.field_schemas[key]
                recon = scan.raw_unscaled_fields[key].astype(np.float64) * sch.scale + sch.offset
                np.testing.assert_allclose(scan.points[:, axis], recon, rtol=0, atol=1e-9,
                                           err_msg=f"mode={mode} axis={key}: 生整数×スケールと points が不一致")
                np.testing.assert_array_equal(scan.raw_fields[key], scan.points[:, axis])

    def test_restore_returns_to_bit_exact_original(self):
        """restore_scan_original_coordinates で読み込み直後の状態（生整数・スキーマ・未変更判定）へ完全復帰する"""
        scan = read_e57_scans(self.in_path)[0]
        transform_scan_coordinate_system(scan, "survey_swap")
        mode = restore_scan_original_coordinates(scan)
        self.assertEqual(mode, "survey_swap")
        self.assertEqual(scan.coord_mode, "cartesian")
        self.assertFalse(scan.is_coordinates_modified(), "復元後は未変更と判定されるべき（生整数経路で書けるように）")
        np.testing.assert_array_equal(scan.raw_unscaled_fields["cartesianX"], self.raw_ix)
        np.testing.assert_array_equal(scan.raw_unscaled_fields["cartesianY"], self.raw_iy)
        np.testing.assert_array_equal(scan.raw_unscaled_fields["cartesianZ"], self.raw_iz)
        self.assertEqual(scan.field_schemas["cartesianX"].scale, self.scales[0])
        self.assertEqual(scan.field_schemas["cartesianY"].scale, self.scales[1])
        np.testing.assert_allclose(scan.points, self._expected_points(), rtol=0, atol=1e-9)

    def test_double_transform_does_not_stack(self):
        """同じスキャンに別モードを再適用しても二重変換にならない（元に戻してから適用）"""
        scan = read_e57_scans(self.in_path)[0]
        transform_scan_coordinate_system(scan, "survey_swap")
        transform_scan_coordinate_system(scan, "z_invert")
        expected = transform_coordinate_system(self._expected_points(), "z_invert")
        np.testing.assert_allclose(scan.points, expected, rtol=0, atol=1e-9)
        self.assertEqual(scan.coord_mode, "z_invert")

    def test_filter_by_indices_propagates_transform_state(self):
        scan = read_e57_scans(self.in_path)[0]
        transform_scan_coordinate_system(scan, "survey_swap")
        sub = scan.filter_by_indices(np.arange(0, self.n, 3))
        self.assertEqual(sub.coord_mode, "survey_swap")
        self.assertTrue(sub.is_coordinates_modified())

    def test_save_worker_writes_back_in_original_frame_bit_exact(self):
        """(3) FileSaveWorker: 測量X/Y入替で取り込み → 一部削除 → 保存 → 元の座標系・生整数が完全一致"""
        from PySide6.QtCore import QCoreApplication
        app = QCoreApplication.instance() or QCoreApplication([])
        from gui.worker_thread import FileSaveWorker

        scan = read_e57_scans(self.in_path)[0]
        transform_scan_coordinate_system(scan, "survey_swap")
        displayed_before = scan.points.copy()

        layer_mask = np.ones(self.n, dtype=bool)
        layer_mask[::5] = False  # 5点に1点をノイズとして除外
        keep_idx = np.where(layer_mask)[0]
        out_path = os.path.join(self.temp_dir, "saved.e57")
        results = []
        worker = FileSaveWorker(out_path, [scan], layer_mask)
        worker.finished_success.connect(lambda a, b, c: results.append((a, b, c)))
        worker.finished_error.connect(lambda msg: self.fail(f"保存エラー: {msg}"))
        worker.run()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], len(keep_idx))

        # 保存後、メモリ上のスキャンは表示用（変換後）の座標系へ戻っていること
        self.assertEqual(scan.coord_mode, "survey_swap")
        np.testing.assert_array_equal(scan.points, displayed_before)

        # 書き出されたファイルは元の座標系・元のスケール・生整数が完全一致していること
        reloaded = read_e57_scans(out_path)[0]
        self.assertEqual(reloaded.field_schemas["cartesianX"].node_type, "ScaledInteger")
        self.assertEqual(reloaded.field_schemas["cartesianX"].scale, self.scales[0])
        self.assertEqual(reloaded.field_schemas["cartesianY"].scale, self.scales[1])
        self.assertEqual(reloaded.field_schemas["cartesianX"].offset, self.offsets[0])
        np.testing.assert_array_equal(reloaded.raw_unscaled_fields["cartesianX"], self.raw_ix[keep_idx])
        np.testing.assert_array_equal(reloaded.raw_unscaled_fields["cartesianY"], self.raw_iy[keep_idx])
        np.testing.assert_array_equal(reloaded.raw_unscaled_fields["cartesianZ"], self.raw_iz[keep_idx])
        np.testing.assert_allclose(reloaded.points, self._expected_points()[keep_idx], rtol=0, atol=1e-9)
        # cartesianBounds は点数が減ったので実データで再計算されている
        b = _bounds_of(reloaded)
        np.testing.assert_allclose(b[0], reloaded.points.min(axis=0), rtol=0, atol=1e-9)
        np.testing.assert_allclose(b[1], reloaded.points.max(axis=0), rtol=0, atol=1e-9)

    def test_writing_transformed_scan_directly_keeps_bounds_and_values_exact(self):
        """(4) 安全網: 逆変換せず変換後のまま write_e57_scans しても、座標値は厳密一致し bounds も更新される"""
        scan = read_e57_scans(self.in_path)[0]
        transform_scan_coordinate_system(scan, "survey_swap")
        out_path = os.path.join(self.temp_dir, "saved_transformed.e57")
        write_e57_scans(out_path, [scan])
        reloaded = read_e57_scans(out_path)[0]
        expected = transform_coordinate_system(self._expected_points(), "survey_swap")
        np.testing.assert_allclose(reloaded.points, expected, rtol=0, atol=1e-9)
        # 軸入替に合わせてスケール定義も入れ替わっている（Xに旧Yのスケール）
        self.assertEqual(reloaded.field_schemas["cartesianX"].scale, self.scales[1])
        self.assertEqual(reloaded.field_schemas["cartesianY"].scale, self.scales[0])
        b = _bounds_of(reloaded)
        np.testing.assert_allclose(b[0], expected.min(axis=0), rtol=0, atol=1e-9)
        np.testing.assert_allclose(b[1], expected.max(axis=0), rtol=0, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
