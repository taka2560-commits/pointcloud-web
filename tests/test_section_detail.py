"""
断面ボックス内の全点表示のテスト
- 断面有効中はボックス内の点を表示予算で間引かず描く（上限 SECTION_DETAIL_MAX_POINTS）
- OFF のときは従来どおり表示予算で間引く
- 断面無効時は表示予算がそのまま効く
- スライダードラッグ中の高速プレビューは確定表示の部分集合、確定で全点に戻る
- MainWindow のチェックボックスと設定の保存・読込
"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.settings_sandbox import use_sandbox_settings  # noqa: E402

use_sandbox_settings()  # この PC の本番アプリの保存設定を、テストで書き換えない

import numpy as np
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from gui.gl_viewer import PointCloudViewer

N_POINTS = 300_000
BUDGET = 50_000
Z_RANGE = (2.0, 4.0)  # 10m 立方の 1/5 ≒ 60,000 点（予算 50,000 点を超える厚み）


def _make_viewer() -> PointCloudViewer:
    rng = np.random.default_rng(7)
    pts = rng.random((N_POINTS, 3)) * 10.0
    v = PointCloudViewer()
    v.set_point_budget(BUDGET)
    v.set_point_cloud(pts)
    return v


def _box_count(v: PointCloudViewer) -> int:
    z = v.points_raw[:, 2]
    return int(np.sum((z >= Z_RANGE[0]) & (z <= Z_RANGE[1])))


class TestSectionFullDetail(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def setUp(self):
        self.v = _make_viewer()

    def tearDown(self):
        self.v.set_point_cloud(None)
        self.v.deleteLater()
        QApplication.processEvents()

    def test_no_section_applies_budget(self):
        """断面無効時は表示予算どおり間引かれ、ボックス内点数は 0"""
        v = self.v
        self.assertFalse(v.is_section_active())
        self.assertLessEqual(len(v.display_points), BUDGET)
        self.assertEqual(v.section_box_count, 0)
        self.assertFalse(v.section_detail_capped)
        v.validate_invariants()

    def test_section_shows_all_points_in_box(self):
        """断面有効中はボックス内の点（予算超）がすべて表示される"""
        v = self.v
        v.set_section_z(True, *Z_RANGE)
        n_box = _box_count(v)
        self.assertGreater(n_box, BUDGET, "テスト前提: ボックス内点数が表示予算を超えていること")
        self.assertEqual(len(v.display_points), n_box)
        self.assertEqual(v.section_box_count, n_box)
        self.assertFalse(v.section_detail_capped)
        # 表示点はすべてボックス内
        z_disp = v.points_raw[v.display_orig_indices, 2]
        self.assertTrue(np.all((z_disp >= Z_RANGE[0]) & (z_disp <= Z_RANGE[1])))
        v.validate_invariants()

    def test_detail_off_applies_budget_inside_box(self):
        """OFF にすると従来どおりボックス内も表示予算で間引く"""
        v = self.v
        v.set_section_full_detail(False)
        v.set_section_z(True, *Z_RANGE)
        n_box = _box_count(v)
        step = math.ceil(n_box / BUDGET)
        self.assertEqual(len(v.display_points), len(range(0, n_box, step)))
        self.assertLessEqual(len(v.display_points), BUDGET)
        self.assertEqual(v.section_box_count, n_box)
        self.assertFalse(v.section_detail_capped)
        # 再び ON にすると断面有効中なので即時に全点へ
        v.set_section_full_detail(True)
        self.assertEqual(len(v.display_points), n_box)
        v.validate_invariants()

    def test_cap_limits_points_and_flags_it(self):
        """ボックス内が上限を超えるときは上限まで間引き、フラグが立つ"""
        v = self.v
        v.SECTION_DETAIL_MAX_POINTS = 30_000  # インスタンス属性で上限を下げる（表示予算より大きい値が上限になる）
        v.set_point_budget(20_000)
        v.set_section_z(True, *Z_RANGE)
        n_box = _box_count(v)
        step = math.ceil(n_box / 30_000)
        self.assertEqual(len(v.display_points), len(range(0, n_box, step)))
        self.assertLessEqual(len(v.display_points), 30_000)
        self.assertTrue(v.section_detail_capped)
        self.assertEqual(v.section_box_count, n_box)
        v.validate_invariants()

    def test_unlimited_budget_is_untouched(self):
        """表示予算「無制限」(0) のときは上限を掛けない"""
        v = self.v
        v.SECTION_DETAIL_MAX_POINTS = 30_000
        v.set_point_budget(0)
        v.set_section_z(True, *Z_RANGE)
        self.assertEqual(len(v.display_points), _box_count(v))
        self.assertFalse(v.section_detail_capped)

    def test_section_off_returns_to_budget(self):
        """断面を解除すると表示予算に戻る"""
        v = self.v
        v.set_section_z(True, *Z_RANGE)
        self.assertGreater(len(v.display_points), BUDGET)
        v.set_section_z(False, *Z_RANGE)
        self.assertLessEqual(len(v.display_points), BUDGET)
        self.assertEqual(v.section_box_count, 0)
        v.validate_invariants()

    def test_fast_preview_is_subset_then_commit_restores_full(self):
        """ドラッグ中の高速プレビューは確定表示の部分集合、確定で再び全点"""
        v = self.v
        v.set_section_z(True, *Z_RANGE)
        n_full = len(v.display_points)
        narrow = (Z_RANGE[0] + 0.5, Z_RANGE[1] - 0.5)
        v.set_section_z(True, *narrow, is_preview=True)
        n_prev = len(v.display_points)
        self.assertLess(n_prev, n_full)
        z_prev = v.display_points[:, 2] + float(v.center_offset[2])
        self.assertTrue(np.all((z_prev >= narrow[0] - 1e-6) & (z_prev <= narrow[1] + 1e-6)))
        # 確定（is_preview=False）で全点データから再構築
        v.set_section_z(True, *narrow, is_preview=False)
        z = v.points_raw[:, 2]
        n_narrow = int(np.sum((z >= narrow[0]) & (z <= narrow[1])))
        self.assertEqual(len(v.display_points), n_narrow)
        v.validate_invariants()

    def test_wall_section_also_full_detail(self):
        """2点斜め壁スライスでも同じ規則が効く"""
        v = self.v
        v.wall_align_angle_deg = 0.0
        v.wall_align_center_world = np.array([5.0, 5.0, 5.0], dtype=np.float64)
        v.set_section_wall(True, -1.0, 1.0)
        self.assertTrue(v.is_section_active())
        n_box = int(np.sum(np.abs(v.points_raw[:, 1] - 5.0) <= 1.0))
        self.assertGreater(n_box, BUDGET)
        self.assertEqual(len(v.display_points), n_box)
        v.validate_invariants()


class TestSectionFullDetailUI(unittest.TestCase):
    """MainWindow 側のチェックボックスと設定の保存・読込"""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def setUp(self):
        from gui.main_window import MainWindow

        self.win = MainWindow()
        self.win._reset_app_settings()

    def tearDown(self):
        self.win.close()
        self.win.deleteLater()
        QApplication.processEvents()

    def test_checkbox_drives_viewer(self):
        win = self.win
        self.assertTrue(win.chk_sec_full_detail.isChecked())
        self.assertTrue(win.viewer.section_full_detail)
        win.chk_sec_full_detail.setChecked(False)
        self.assertFalse(win.viewer.section_full_detail)
        win.chk_sec_full_detail.setChecked(True)
        self.assertTrue(win.viewer.section_full_detail)

    def test_setting_roundtrip(self):
        win = self.win
        s = QSettings("E57PointCloudTool", "Settings")
        before = s.value("sec_full_detail", None)
        try:
            win.chk_sec_full_detail.setChecked(False)
            win._save_app_settings(silent=True)
            self.assertFalse(QSettings("E57PointCloudTool", "Settings").value("sec_full_detail", True, type=bool))
            win.chk_sec_full_detail.setChecked(True)
            win._load_app_settings()
            self.assertFalse(win.chk_sec_full_detail.isChecked())
            self.assertFalse(win.viewer.section_full_detail)
        finally:
            if before is None:
                s.remove("sec_full_detail")
            else:
                s.setValue("sec_full_detail", before)


if __name__ == "__main__":
    unittest.main()
