"""
計測ツールのピックと読込連携のテスト（実際の PointCloudViewer を使う）

- 密な点群でも、クリックした点から画面 1〜2px 相当以内の点を拾うこと（間引きで別の点を拾わない）
- 奥の面の点ではなく手前の面の点を拾うこと
- 大きな座標でも points_raw の原本値を返すこと。高速プレビュー中も正しい座標を返すこと
- 許容距離の外では拾わないこと
- 別の点群を設定すると計測点が消えること
- 実際の読込データ（座標系変換あり）で coord_mode が伝わり、ファイル系の座標が元ファイルの点と一致すること
"""

import os
import sys
import tempfile
import unittest

import numpy as np
from PySide6.QtCore import QPoint
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from core.e57_io import read_e57_scans
from gui.gl_viewer import PointCloudViewer
from gui.measure_text import file_coords
from gui.worker_thread import FileLoadWorker

SAMPLE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "samples", "サンプル点群.e57")


class TestMeasurePick(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def setUp(self):
        self.v = PointCloudViewer()
        self.v.resize(1000, 700)
        self.v.show()
        QTest.qWait(50)

    def tearDown(self):
        self.v.close()

    def _load(self, pts):
        self.v.set_point_cloud(pts)
        self.v.is_orthographic = False
        self.v.update()
        self.app.processEvents()

    def _screen_pos(self, centered):
        x, y, valid = self.v._project_to_screen(np.asarray(centered, dtype=np.float32).reshape(1, 3))
        self.assertTrue(valid[0])
        return QPoint(int(round(float(x[0]))), int(round(float(y[0]))))

    def test_dense_cloud_picks_point_under_cursor(self):
        rng = np.random.default_rng(0)
        pts = rng.random((800_000, 3)) * [50.0, 50.0, 0.01]
        self._load(pts)
        v = self.v
        tested = 0
        for _ in range(40):
            c = v.display_points[rng.integers(len(v.display_points))]
            pos = self._screen_pos(c)
            if not (0 <= pos.x() < v.width() and 0 <= pos.y() < v.height()):
                continue
            hit = v._pick_nearest_point(pos, max_px=v.MEASURE_PICK_MAX_PX)
            self.assertIsNotNone(hit)
            depth = float(v._view_depth(np.asarray(c).reshape(1, 3))[0])
            self.assertLess(np.linalg.norm(hit[0] - c), v._world_per_pixel(depth) * 4.0)
            self.assertLessEqual(v.last_pick_px, 1.5)
            tested += 1
        self.assertGreater(tested, 10)

    def test_picks_front_surface_not_back(self):
        rng = np.random.default_rng(1)
        top = rng.random((400_000, 3)) * [10.0, 10.0, 0.0] + [0.0, 0.0, 5.0]  # 隙間から奥が見えない密度
        bottom = rng.random((400_000, 3)) * [10.0, 10.0, 0.0]
        self._load(np.vstack([top, bottom]))
        v = self.v
        v.pitch = 89.9
        v.yaw = 0.0
        v.update()
        self.app.processEvents()
        for _ in range(20):
            hit = v._pick_nearest_point(QPoint(int(rng.integers(450, 550)), int(rng.integers(300, 400))), max_px=12.0)
            self.assertIsNotNone(hit)
            self.assertAlmostEqual(float(hit[1][2]), 5.0, places=6, msg="奥の面の点を拾いました")

    def test_returns_original_coordinates(self):
        rng = np.random.default_rng(2)
        pts = rng.random((5_000, 3)) * 10.0 + [500000.1234567, 3900000.7654321, 100.5]
        self._load(pts)
        v = self.v
        hit = v._pick_nearest_point(self._screen_pos(v.display_points[10]), max_px=12.0)
        self.assertIsNotNone(hit)
        self.assertTrue(np.any(np.all(pts == hit[1], axis=1)), "原本の座標と完全一致しません")

    def test_fast_preview_still_returns_correct_coordinates(self):
        rng = np.random.default_rng(3)
        pts = rng.random((5_000, 3)) * 10.0
        self._load(pts)
        v = self.v
        v.clip_z_enabled = True
        v.clip_z_range = (0.0, 5.0)
        v._update_display_arrays_fast_preview()
        self.assertLess(len(v.display_points), len(pts))
        hit = v._pick_nearest_point(self._screen_pos(v.display_points[3]), max_px=12.0)
        self.assertIsNotNone(hit)
        self.assertLess(float(np.min(np.linalg.norm(pts - hit[1], axis=1))), 1e-5)

    def test_outside_tolerance_is_not_picked(self):
        self._load(np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]]))
        v = self.v
        pos = self._screen_pos(v.display_points[0])
        off = QPoint(pos.x() + 20, pos.y())
        self.assertIsNone(v._pick_nearest_point(off, max_px=v.MEASURE_PICK_MAX_PX))
        self.assertIsNotNone(v._pick_nearest_point(off))  # 回転中心の指定などの既定は 30px のまま

    def test_new_point_cloud_clears_measure_points(self):
        self._load(np.random.default_rng(4).random((100, 3)))
        v = self.v
        v.measure_mode = True
        c = v.display_points[0]
        v._on_measure_pick(c, c.astype(np.float64) + v.center_offset)
        self.assertEqual(len(v.measure_points), 1)
        v.set_point_cloud(np.random.default_rng(5).random((100, 3)) + 1000.0)
        self.assertEqual(v.measure_points, [])
        self.assertEqual(v.measure_coord_lines, [])
        self.assertEqual(v.measure_result_text, "")

    def test_snap_notice_in_status(self):
        self._load(np.random.default_rng(6).random((100, 3)))
        v = self.v
        msgs = []
        v.measure_updated.connect(msgs.append)
        c = v.display_points[0]
        v._on_measure_pick(c, c.astype(np.float64) + v.center_offset, snap_px=9.0)
        self.assertIn("9px", msgs[-1])
        v._on_measure_pick(c, c.astype(np.float64) + v.center_offset, snap_px=1.0)
        self.assertNotIn("px", msgs[-1])

    def _preloaded(self, coord_mode):
        holder = []
        worker = FileLoadWorker(SAMPLE, point_budget=2_000_000, coord_mode=coord_mode)
        worker.finished_success.connect(holder.append)
        worker.run()
        self.assertTrue(holder, "サンプルの読込に失敗しました")
        return holder[0]

    @unittest.skipUnless(os.path.exists(SAMPLE), "サンプル点群がありません")
    def test_real_load_passes_coord_mode_and_file_coordinates_match(self):
        v = self.v
        v.set_preloaded_data(self._preloaded("survey_swap"), reset_camera=True)
        self.assertEqual(v.coord_mode, "survey_swap")

        hit = v._pick_nearest_point(self._screen_pos(v.display_points[5]), max_px=12.0)
        self.assertIsNotNone(hit)
        v.measure_mode = True
        v._on_measure_pick(hit[0], hit[1], snap_px=v.last_pick_px)
        self.assertTrue(v.measure_coord_lines[1].strip().startswith("ファイル系"))

        original = np.vstack([s.points for s in read_e57_scans(SAMPLE)])
        fc = file_coords(hit[1], v.coord_mode)
        self.assertTrue(np.any(np.all(original == fc, axis=1)), "ファイル系の座標が元ファイルの点と一致しません")

        # 変換なしのデータを読み直すと、計測点は消えて coord_mode も戻る
        v.set_preloaded_data(self._preloaded("cartesian"), reset_camera=True)
        self.assertEqual(v.coord_mode, "cartesian")
        self.assertEqual(v.measure_points, [])


if __name__ == "__main__":
    unittest.main()
