"""
E57 の pose（回転・平行移動）の扱いのテスト

- 表示座標は world = R @ local + t で、scan.points（保存用の生座標）は変わらないこと
- pose が恒等なら座標をコピーしないこと
- 実際の読込経路（FileLoadWorker）で表示座標が pose 適用後になること。座標系変換は pose の後に適用されること
- 複数スキャンがそれぞれの pose で配置されること
- 保存すると生座標と pose がそのまま書き戻され、他のソフトで開いても同じ位置になること
- 距離フィルタはスキャン原点からの距離を保つこと
- 計測で拾った座標が pose 適用後の値で、ファイル系も軸入替を外した pose 適用後の値になること
"""

import os
import sys
import tempfile
import unittest

import numpy as np
from PySide6.QtCore import QPoint
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from core.e57_io import E57ScanData, read_e57_scans, write_e57_scans
from core.processor import (
    FilterConfig,
    PointCloudProcessor,
    scan_display_origin,
    scan_display_points,
    scan_pose_transform,
    transform_coordinate_system,
    transform_scan_coordinate_system,
)
from gui.gl_viewer import PointCloudViewer
from gui.measure_text import file_coords
from gui.worker_thread import FileLoadWorker, FileSaveWorker

# Z 軸まわり 90° + 傾きなし
ROT_90Z = np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)])
TRANS = np.array([1000.5, -148753.25, 12.75])


def _expected_world(local):
    # Z 軸まわり +90°: (x, y, z) → (-y, x, z)
    return np.column_stack([-local[:, 1], local[:, 0], local[:, 2]]) + TRANS


def _make_scan(index, local, rotation=ROT_90Z, translation=TRANS, name=None):
    fields = {"cartesianX": local[:, 0].copy(), "cartesianY": local[:, 1].copy(), "cartesianZ": local[:, 2].copy()}
    return E57ScanData(index, local.copy(), fields, rotation=rotation, translation=translation, name=name)


class TestPoseMath(unittest.TestCase):
    def setUp(self):
        self.local = np.random.default_rng(0).random((200, 3)) * 50.0

    def test_world_formula_and_raw_points_unchanged(self):
        scan = _make_scan(0, self.local)
        np.testing.assert_allclose(scan_display_points(scan), _expected_world(self.local), atol=1e-9)
        np.testing.assert_array_equal(scan.points, self.local)

    def test_tilted_pose_matches_quaternion_definition(self):
        from pyquaternion import Quaternion

        q = np.array([0.691207, 0.006596, 0.006611, 0.722596])
        scan = _make_scan(0, self.local, rotation=q)
        expected = self.local @ Quaternion(q).rotation_matrix.T + TRANS
        np.testing.assert_allclose(scan_display_points(scan), expected, atol=1e-9)

    def test_identity_pose_shares_array(self):
        for rot, tr in ((None, None), (np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3))):
            scan = _make_scan(0, self.local, rotation=rot, translation=tr)
            self.assertIsNone(scan_pose_transform(scan))
            self.assertIs(scan_display_points(scan), scan.points)

    def test_translation_only_pose_is_applied(self):
        scan = _make_scan(0, self.local, rotation=None)
        np.testing.assert_allclose(scan_display_points(scan), self.local + TRANS)

    def test_coord_mode_is_applied_after_pose(self):
        for mode in ("survey_swap", "survey_swap_inv", "y_up_to_z_up", "z_invert"):
            scan = _make_scan(0, self.local)
            transform_scan_coordinate_system(scan, mode)
            expected = transform_coordinate_system(_expected_world(self.local), mode)
            np.testing.assert_allclose(scan_display_points(scan), expected, atol=1e-9, err_msg=mode)
            np.testing.assert_allclose(scan_display_origin(scan), transform_coordinate_system(TRANS.reshape(1, 3), mode)[0], err_msg=mode)


class TestPoseDistanceFilter(unittest.TestCase):
    def test_distance_is_measured_from_scan_origin(self):
        local = np.array([[1.0, 0.0, 0.0], [20.0, 0.0, 0.0], [0.0, 30.0, 0.0], [0.0, 0.0, 200.0]])
        scan = _make_scan(0, local)
        world = scan_display_points(scan)
        cfg = FilterConfig(use_sor=False, use_distance_filter=True, min_distance=5.0, max_distance=100.0)
        keep, _ = PointCloudProcessor(config=cfg).apply_filters_to_points(world, distance_points=world - scan_display_origin(scan))
        self.assertEqual(sorted(np.asarray(keep).tolist()), [1, 2])
        # 基準を渡さないと全点が原点から遠く、別の結果になる（基準の受け渡しが意味を持つことの確認）
        keep_wrong, _ = PointCloudProcessor(config=cfg).apply_filters_to_points(world)
        self.assertNotEqual(sorted(np.asarray(keep_wrong).tolist()), [1, 2])


class TestPoseFilePipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "pose点群.e57")
        rng = np.random.default_rng(1)
        self.local_a = rng.random((3000, 3)) * 40.0
        self.local_b = rng.random((2000, 3)) * 40.0
        write_e57_scans(self.path, [_make_scan(0, self.local_a, name="A")])

    def tearDown(self):
        self.tmp.cleanup()

    def _load(self, path, coord_mode="cartesian"):
        holder, errors = [], []
        worker = FileLoadWorker(path, point_budget=2_000_000, coord_mode=coord_mode)
        worker.finished_success.connect(holder.append)
        worker.finished_error.connect(errors.append)
        worker.run()
        self.assertFalse(errors, errors)
        return holder[0]

    def test_load_gives_world_and_keeps_raw(self):
        pre = self._load(self.path)
        np.testing.assert_allclose(pre.points_raw, _expected_world(self.local_a), atol=1e-6)
        np.testing.assert_allclose(pre.scans[0].points, self.local_a, atol=1e-9)
        self.assertIsNot(pre.points_raw, pre.scans[0].points)

    def test_multi_scan_each_uses_its_own_pose(self):
        other = np.array([500.0, 600.0, 7.0])
        path = os.path.join(self.tmp.name, "複数.e57")
        write_e57_scans(path, [
            _make_scan(0, self.local_a, name="A"),
            _make_scan(1, self.local_b, rotation=None, translation=other, name="B"),
        ])
        pre = self._load(path)
        n = len(self.local_a)
        np.testing.assert_allclose(pre.points_raw[:n], _expected_world(self.local_a), atol=1e-6)
        np.testing.assert_allclose(pre.points_raw[n:], self.local_b + other, atol=1e-6)

    def test_save_writes_raw_and_pose_back(self):
        pre = self._load(self.path, coord_mode="survey_swap")
        mask = np.ones(len(pre.points_raw), dtype=bool)
        mask[::3] = False
        out = os.path.join(self.tmp.name, "出力_clean.e57")
        errors = []
        worker = FileSaveWorker(out, pre.scans, mask)
        worker.finished_error.connect(errors.append)
        worker.run()
        self.assertFalse(errors, errors)
        saved = read_e57_scans(out)[0]
        np.testing.assert_array_equal(saved.points, read_e57_scans(self.path)[0].points[mask])
        np.testing.assert_array_equal(saved.rotation, read_e57_scans(self.path)[0].rotation)
        np.testing.assert_array_equal(saved.translation, read_e57_scans(self.path)[0].translation)
        # 保存後の再読込は、保存前に見えていた点と同じ位置（軸入替は読み直しの設定で適用されるので cartesian で比較）
        np.testing.assert_allclose(self._load(out).points_raw, _expected_world(self.local_a)[mask], atol=1e-6)
        # 保存でメモリ上の表示用の座標系が壊れない
        self.assertEqual(pre.scans[0].coord_mode, "survey_swap")

    def test_measure_pick_returns_world_coordinates(self):
        pre = self._load(self.path, coord_mode="survey_swap")
        viewer = PointCloudViewer()
        viewer.resize(1000, 700)
        viewer.show()
        QTest.qWait(50)
        viewer.set_preloaded_data(pre, reset_camera=True)
        viewer.update()
        self.app.processEvents()
        self.assertEqual(viewer.coord_mode, "survey_swap")
        x, y, valid = viewer._project_to_screen(viewer.display_points[7].reshape(1, 3))
        self.assertTrue(valid[0])
        hit = viewer._pick_nearest_point(QPoint(int(round(float(x[0]))), int(round(float(y[0])))), max_px=12.0)
        self.assertIsNotNone(hit)
        world_all = _expected_world(self.local_a)
        fc = file_coords(hit[1], viewer.coord_mode)
        self.assertLess(float(np.min(np.linalg.norm(world_all - fc, axis=1))), 1e-6, "ファイル系が pose 適用後の座標と一致しません")
        viewer.close()


if __name__ == "__main__":
    unittest.main()
