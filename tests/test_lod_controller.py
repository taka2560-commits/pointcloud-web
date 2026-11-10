"""
視点連動の高精細表示（gui.lod_controller / PointCloudViewer.set_view_overlay / MainWindow のチェックボックス）のテスト
- 視点に入っている範囲が元ファイルの密度で重ね描きされ、読み込み済み（間引き）より密になること
- 予算を超えないこと、画面外の点は読まないこと、寄ると点が増えること
- 断面が有効なら箱の中だけになること、断面の「全点で読み直す」の重ね描きは上書きしないこと
- ノイズへ移した領域の点は重ね描きからも隠れること（読み込まれなかった仲間の点も）
- 無効化・点群入替で消えること、タイマー経由（視点が止まってから）でも更新されること
- MainWindow: LOD データが無ければ確認のうえ作成して有効化、次回の読込では自動で有効、設定の保存・読込
"""

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import numpy as np
from PySide6.QtCore import QCoreApplication, QElapsedTimer, QSettings, QThread
from PySide6.QtGui import QVector3D
from PySide6.QtWidgets import QApplication, QMessageBox

from core.lod_store import LodStore, prepare_lod_for_scans
from core.section_cache import remove_cache
from gui.gl_viewer import PointCloudViewer
from gui.lod_controller import LodController
from gui.worker_thread import FileLoadWorker
from test_section_reload import _Scene

BASE = 2_000
LEAF_MAX = 3_000


def _pump(ms: int):
    timer = QElapsedTimer()
    timer.start()
    while timer.elapsed() < ms:
        QCoreApplication.processEvents()
        QThread.msleep(5)


def _look_down_at(v: PointCloudViewer, world_xy, distance: float):
    """真上から world_xy を見下ろす正射影にする"""
    v.is_orthographic = True
    v.pitch = 89.9
    v.yaw = 0.0
    v.camera_distance = float(distance)
    v.camera_target = QVector3D(
        float(world_xy[0] - v.center_offset[0]), float(world_xy[1] - v.center_offset[1]), float(0.0 - v.center_offset[2])
    )


class TestLodController(_Scene):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        remove_cache(cls.path)
        holder = []
        worker = FileLoadWorker(cls.path, point_budget=50_000, max_load_points=cls.load_budget)
        worker.finished_success.connect(holder.append)
        worker.run()
        cls.pre = holder[0]
        assert all(s.is_subsampled for s in cls.pre.scans)
        prepare_lod_for_scans(cls.pre.scans, base_points=BASE)

    @classmethod
    def tearDownClass(cls):
        cls.pre = None
        remove_cache(cls.path)
        super().tearDownClass()

    def setUp(self):
        self.v = PointCloudViewer()
        self.v.resize(800, 600)
        self.v.set_shading_enabled(False)
        self.v.set_preloaded_data(self.pre, reset_camera=True)
        self.c = LodController(self.v)
        self.assertTrue(self.c.attach(self.pre.scans, leaf_max_points=LEAF_MAX))
        self.c.set_enabled(True)

    def tearDown(self):
        self.c.shutdown()
        self.v.set_point_cloud(None)
        self.v.deleteLater()
        QApplication.processEvents()

    def _overlay_world(self):
        return self.v.overlay_points.astype(np.float64) + self.v.center_offset

    def test_overlay_is_denser_than_loaded_and_inside_view(self):
        v, c = self.v, self.c
        _look_down_at(v, (2.0, 1.5), 6.0)  # スキャン A（床 4×3m）の中央。画面はおよそ 5.6m × 4.2m
        n = c.refresh_now()
        self.assertGreater(n, 0)
        self.assertEqual(v.overlay_kind, "lod")
        self.assertTrue(v.has_section_overlay)
        self.assertLessEqual(n, c.budget)
        w = self._overlay_world()
        # 画面に入る葉だけ: B（x≈100, y≈200）は読まない
        self.assertTrue(np.all(w[:, 0] < 50.0))
        # 床 (z=0, 0≤x≤4, 0≤y≤3) は元ファイルの全点（30,351 点）に近い密度。読み込み済み（3 点に 1 点）より密
        on_floor = (np.abs(w[:, 2]) < 0.005) & (w[:, 0] >= 0) & (w[:, 0] <= 4) & (w[:, 1] >= 0) & (w[:, 1] <= 3)
        raw = v.points_raw
        loaded_floor = (np.abs(raw[:, 2]) < 0.005) & (raw[:, 0] >= 0) & (raw[:, 0] <= 4) & (raw[:, 1] >= 0) & (raw[:, 1] <= 3)
        self.assertGreater(np.count_nonzero(on_floor), 2 * np.count_nonzero(loaded_floor))
        self.assertGreater(np.count_nonzero(on_floor), 25_000)
        self.assertEqual(len(v.points_raw), len(self.pre.points_raw), "読み込み済みの点は変わらない")
        self.assertEqual(v.overlay_colors.shape, (n, 3))
        v.validate_invariants()

    def test_budget_and_zoom(self):
        v, c = self.v, self.c
        _look_down_at(v, (2.0, 1.5), 6.0)
        c.set_budget(100_000)
        n_wide_full = c.refresh_now()
        c.budget = 5_000  # テスト用に小さな予算
        n_small = c.refresh_now()
        self.assertGreater(n_small, 0)
        self.assertLessEqual(n_small, 5_000)
        self.assertLess(n_small, n_wide_full)
        self.assertTrue(v.overlay_capped)
        # 同じ予算でも、寄ると画面内の密度が上がる（床 1m 四方あたりの点数）
        w = self._overlay_world()
        dens_wide = np.count_nonzero((np.abs(w[:, 0] - 2.0) < 0.5) & (np.abs(w[:, 1] - 1.5) < 0.5) & (np.abs(w[:, 2]) < 0.005))
        _look_down_at(v, (2.0, 1.5), 1.5)
        n_zoom = c.refresh_now()
        self.assertLessEqual(n_zoom, 5_000)
        w = self._overlay_world()
        dens_zoom = np.count_nonzero((np.abs(w[:, 0] - 2.0) < 0.5) & (np.abs(w[:, 1] - 1.5) < 0.5) & (np.abs(w[:, 2]) < 0.005))
        self.assertGreater(dens_zoom, dens_wide)
        v.validate_invariants()

    def test_section_box_limits_overlay_and_manual_overlay_is_kept(self):
        v, c = self.v, self.c
        _look_down_at(v, (2.0, 1.5), 6.0)
        v.set_section_x(True, 1.0, 2.0)
        n = c.refresh_now()
        self.assertGreater(n, 0)
        w = self._overlay_world()
        self.assertTrue(np.all((w[:, 0] >= 1.0 - 1e-4) & (w[:, 0] <= 2.0 + 1e-4)), "断面が有効なら箱の中だけ")
        v.validate_invariants()
        # 断面の「全点で読み直す」で作った重ね描き（種別 section）は上書きしない
        manual = np.column_stack([np.linspace(1.1, 1.9, 50), np.full(50, 1.0), np.zeros(50)])
        v.set_section_overlay(manual, None, None, None, n_in_box=50, capped=False)
        self.assertEqual(v.overlay_kind, "section")
        self.assertEqual(c.refresh_now(), 0)
        self.assertEqual(v.overlay_kind, "section")
        self.assertEqual(len(v.overlay_points), 50)
        # 箱が変わると重ね描きは消え、LOD が作り直す
        v.set_section_x(True, 1.0, 3.0)
        self.assertFalse(v.has_section_overlay)
        self.assertGreater(c.refresh_now(), 0)
        self.assertEqual(v.overlay_kind, "lod")
        # 断面を解除しても LOD は続く（断面なしの重ね描き）
        v.set_section_x(False, 1.0, 3.0)
        self.assertFalse(v.has_section_overlay)
        self.assertGreater(c.refresh_now(), 0)
        v.validate_invariants()

    def test_thin_section_is_refined_to_use_the_budget(self):
        """遠くから見た薄い断面: 葉の広さだけで決めると箱の中にわずかしか残らないので、目標密度を上げて読み直す"""
        from gui.lod_controller import compute_lod_overlay

        v, c = self.v, self.c
        _look_down_at(v, (2.0, 1.5), 60.0)  # 遠景（床 4×3m が画面の 1 割弱）
        v.set_section_z(True, -0.005, 0.005)  # 床だけの薄い断面
        inputs = c._job_inputs()
        plain = compute_lod_overlay(**inputs, refine=False)
        refined = compute_lod_overlay(**inputs, refine=True)
        self.assertGreater(len(refined["points"]), 2 * max(1, len(plain["points"])))
        self.assertLessEqual(len(refined["points"]), c.budget)
        z = refined["points"][:, 2].astype(np.float64) + v.center_offset[2]
        self.assertTrue(np.all(np.abs(z) <= 0.005 + 1e-4))
        # 予算が小さければ、読み直したあとも予算以内に収まる
        inputs["budget"] = 4_000
        small = compute_lod_overlay(**inputs, refine=True)
        self.assertLessEqual(len(small["points"]), 4_000)
        self.assertGreater(len(small["points"]), 0)
        # コントローラ経由でも同じ（箱の中だけ・不変条件）
        n = c.refresh_now()
        self.assertEqual(n, len(refined["points"]))
        v.validate_invariants()

    def test_noise_region_is_hidden_in_overlay(self):
        v, c = self.v, self.c
        _look_down_at(v, (-9.0, 1.0), 6.0)  # スキャン C（x -10〜-8 のまばらな塊）を見る
        n_before = c.refresh_now()
        w = self._overlay_world()
        in_c = w[:, 0] < -7.0
        self.assertGreater(np.count_nonzero(in_c), 1500, "C の点（2,000 点）がほぼ全て重ね描きされる")
        # C の読み込み済みの点（3 点に 1 点）をすべてノイズへ → 読み込まれなかった仲間の点も重ね描きから消える
        idx = np.flatnonzero(v.points_raw[:, 0] < -7.0)
        self.assertGreater(len(idx), 500)
        v.move_indices_to_noise(idx)
        n_after = c.refresh_now()
        w = self._overlay_world() if v.has_section_overlay else np.empty((0, 3))
        self.assertEqual(np.count_nonzero(w[:, 0] < -7.0), 0)
        self.assertLess(n_after, n_before)
        self.assertGreater(c.last_result_info.get("n_hidden", 0) if n_after else 1, 0)
        # ノイズを戻すと再び出る
        v.restore_indices_to_normal(idx)
        c.refresh_now()
        w = self._overlay_world()
        self.assertGreater(np.count_nonzero(w[:, 0] < -7.0), 1500)
        v.validate_invariants()

    def test_disable_and_cloud_replacement(self):
        v, c = self.v, self.c
        _look_down_at(v, (2.0, 1.5), 6.0)
        self.assertGreater(c.refresh_now(), 0)
        c.set_enabled(False)
        self.assertFalse(v.has_section_overlay)
        self.assertEqual(c.refresh_now(), 0)
        c.set_enabled(True)
        self.assertGreater(c.refresh_now(), 0)
        # 点群を入れ替えると重ね描きは消え、コントローラは次の監視で外れる
        v.set_point_cloud(np.random.default_rng(2).random((500, 3)))
        self.assertFalse(v.has_section_overlay)
        self.assertEqual(c.refresh_now(), 0)
        c._on_poll()
        self.assertIsNone(c.scene)
        v.validate_invariants()

    def test_timer_refreshes_after_view_settles(self):
        v, c = self.v, self.c
        got = []
        c.refreshed.connect(got.append)
        _look_down_at(v, (2.0, 1.5), 6.0)
        _pump(1500)
        self.assertTrue(got, "視点が止まってからタイマーで読み込まれる")
        self.assertEqual(v.overlay_kind, "lod")
        n_first = len(got)
        _pump(600)
        self.assertEqual(len(got), n_first, "視点が同じ間は読み直さない")
        _look_down_at(v, (2.0, 1.5), 2.0)
        _pump(1500)
        self.assertGreater(len(got), n_first, "視点が変わると読み直す")
        v.validate_invariants()


class TestMainWindowLod(_Scene):
    def setUp(self):
        from gui.main_window import MainWindow

        remove_cache(self.path)
        self._settings_backup = QSettings("E57PointCloudTool", "Settings").value("lod_enabled", None)
        self.win = MainWindow()
        self.win._reset_app_settings()
        self.win.show()
        QApplication.processEvents()
        self._orig_question = QMessageBox.question

    def tearDown(self):
        QMessageBox.question = self._orig_question
        self.win.close()
        self.win.deleteLater()
        QApplication.processEvents()
        s = QSettings("E57PointCloudTool", "Settings")
        if self._settings_backup is None:
            s.remove("lod_enabled")
        else:
            s.setValue("lod_enabled", self._settings_backup)
        remove_cache(self.path)

    def _load(self):
        win = self.win
        win.combo_load_budget.addItem("テスト用 2 万点", self.load_budget)
        win.combo_load_budget.setCurrentIndex(win.combo_load_budget.count() - 1)
        self.assertTrue(win._load_file_to_viewer(self.path) is not False)
        self.assertTrue(win._wait_for_background(60_000))
        QApplication.processEvents()

    def test_toggle_builds_then_enables_and_next_load_is_automatic(self):
        win = self.win
        self.assertFalse(win.chk_lod.isChecked())
        # 点群が無い間はチェックだけ覚える
        win.chk_lod.setChecked(True)
        self.assertFalse(win.lod.is_active)
        win.chk_lod.setChecked(False)

        self._load()
        # 断る → チェックが戻り、何も作られない
        QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.No)
        win.chk_lod.setChecked(True)
        self.assertFalse(win.chk_lod.isChecked())
        self.assertIsNone(LodStore.open(self.path))
        # 承諾 → バックグラウンドで作成 → 有効化
        asked = []

        def yes(*a, **k):
            asked.append(a[2] if len(a) > 2 else "")
            return QMessageBox.Yes

        QMessageBox.question = staticmethod(yes)
        win.chk_lod.setChecked(True)
        self.assertTrue(asked and "LOD" in asked[0])
        self.assertTrue(win._wait_for_background(120_000), "LOD データの作成が終わりませんでした")
        QApplication.processEvents()
        self.assertTrue(win.chk_lod.isChecked())
        self.assertTrue(win.lod.is_active)
        self.assertIsNotNone(LodStore.open(self.path))
        self.assertIn("LOD", win.log_text.toPlainText())
        self.assertEqual(win.lod.scene.n_points, self.n_total)
        _look_down_at(win.viewer, (2.0, 1.5), 6.0)
        self.assertGreater(win.lod.refresh_now(), 0)
        self.assertEqual(win.viewer.overlay_kind, "lod")
        win.viewer.validate_invariants()

        # OFF で消える。もう一度 ON は確認なしですぐ有効（作成済み）
        win.chk_lod.setChecked(False)
        self.assertFalse(win.viewer.has_section_overlay)
        self.assertFalse(win.lod.is_active)
        asked.clear()
        win.chk_lod.setChecked(True)
        self.assertFalse(asked)
        self.assertTrue(win.lod.is_active)

        # 設定に保存され、読み直しても ON のまま自動で有効になる
        win._save_app_settings(silent=True)
        self.assertTrue(QSettings("E57PointCloudTool", "Settings").value("lod_enabled", False, type=bool))
        self._load()
        self.assertTrue(win.chk_lod.isChecked())
        self.assertTrue(win.lod.is_active)
        self.assertGreater(win.lod.refresh_now(), 0)

    def test_load_without_store_unchecks(self):
        win = self.win
        win.chk_lod.setChecked(True)  # 点群なし: 覚えるだけ
        self.assertTrue(win.chk_lod.isChecked())
        self._load()                  # LOD データ未作成のファイル
        self.assertFalse(win.chk_lod.isChecked(), "LOD データの無いファイルでは OFF に戻す（確認ダイアログは出さない）")
        self.assertFalse(win.lod.is_active)


if __name__ == "__main__":
    unittest.main()
