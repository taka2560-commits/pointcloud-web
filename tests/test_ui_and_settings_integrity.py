"""
UIコンパクト化・設定パネル・ワークフロー切替・感度設定・断面折りたたみ・保存常設の
包括的エラー・バグ・整合性検証テストスイート
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import unittest
import numpy as np
from PySide6.QtCore import Qt, QSettings
from PySide6.QtWidgets import QApplication

# ヘッドレス実行用のQApplication
app = QApplication.instance() or QApplication(sys.argv)

from gui.main_window import MainWindow
from gui.gl_viewer import PointCloudViewer
from core.processor import E57ScanData


class TestUIAndSettingsIntegrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # QSettings をテスト用組織・アプリ名に設定
        cls.orig_org = "E57PointCloudTool"
        cls.orig_app = "Settings"

    def setUp(self):
        # 毎回クリーンな MainWindow インスタンスを生成
        self.win = MainWindow()
        # テスト実行の一貫性を保つため初期値にリセット
        self.win._reset_app_settings()
        self.win.show()

    def tearDown(self):
        self.win.close()
        self.win.deleteLater()
        QApplication.processEvents()

    # -------------------------------------------------------------
    # 1. ⚙ 設定パネルと Viewer の同期・不変条件テスト
    # -------------------------------------------------------------
    def test_settings_controls_exist_and_defaults(self):
        """設定パネルの各UI部品が存在し、初期値が設計通りか検証"""
        self.assertTrue(hasattr(self.win, "size_slider"))
        self.assertTrue(hasattr(self.win, "shape_combo"))
        self.assertTrue(hasattr(self.win, "budget_combo"))
        self.assertTrue(hasattr(self.win, "bg_combo"))
        self.assertTrue(hasattr(self.win, "grid_chk"))
        self.assertTrue(hasattr(self.win, "slider_rot_speed"))
        self.assertTrue(hasattr(self.win, "slider_pan_speed"))
        self.assertTrue(hasattr(self.win, "slider_zoom_speed"))
        self.assertTrue(hasattr(self.win, "slider_wasd_speed"))
        self.assertTrue(hasattr(self.win, "chk_invert_orbit"))
        self.assertTrue(hasattr(self.win, "settings_default_color_combo"))
        self.assertTrue(hasattr(self.win, "slider_fov"))
        self.assertTrue(hasattr(self.win, "chk_depth_fog"))
        self.assertTrue(hasattr(self.win, "combo_sec_step"))
        self.assertTrue(hasattr(self.win, "combo_undo_limit"))
        self.assertTrue(hasattr(self.win, "chk_auto_save_settings"))

        # 初期値の検証
        self.assertEqual(self.win.slider_rot_speed.value(), 10)
        self.assertEqual(self.win.slider_pan_speed.value(), 10)
        self.assertEqual(self.win.slider_zoom_speed.value(), 10)
        self.assertEqual(self.win.slider_wasd_speed.value(), 10)
        self.assertFalse(self.win.chk_invert_orbit.isChecked())
        self.assertEqual(self.win.slider_fov.value(), 45)
        self.assertFalse(self.win.chk_depth_fog.isChecked())
        self.assertEqual(self.win.combo_sec_step.currentIndex(), 2)  # 1cm
        self.assertEqual(self.win.combo_undo_limit.currentIndex(), 1) # 50回
        self.assertTrue(self.win.chk_auto_save_settings.isChecked())

        # ビューア側の初期値との整合
        self.assertAlmostEqual(self.win.viewer.rotate_speed, 0.5)
        self.assertAlmostEqual(self.win.viewer.pan_speed_factor, 1.0)
        self.assertAlmostEqual(self.win.viewer.zoom_speed_factor, 1.0)
        self.assertAlmostEqual(self.win.viewer.wasd_speed_factor, 1.0)
        self.assertFalse(self.win.viewer.invert_orbit_y)
        self.assertAlmostEqual(self.win.viewer.fov, 45.0)
        self.assertFalse(self.win.viewer.depth_fog_enabled)

    def test_settings_sensitivity_changes(self):
        """各種感度スライダーを動かした時、Viewer内部変数および表示ラベルが完全同期するか検証"""
        # 回転感度
        self.win.slider_rot_speed.setValue(15)  # 1.5x
        self.assertEqual(self.win.lbl_rot_speed.text(), "1.5x")
        self.assertAlmostEqual(self.win.viewer.rotate_speed, 0.5 * 1.5)

        # パン感度
        self.win.slider_pan_speed.setValue(8)  # 0.8x
        self.assertEqual(self.win.lbl_pan_speed.text(), "0.8x")
        self.assertAlmostEqual(self.win.viewer.pan_speed_factor, 0.8)

        # ズーム感度
        self.win.slider_zoom_speed.setValue(20)  # 2.0x
        self.assertEqual(self.win.lbl_zoom_speed.text(), "2.0x")
        self.assertAlmostEqual(self.win.viewer.zoom_speed_factor, 2.0)

        # WASD感度
        self.win.slider_wasd_speed.setValue(5)  # 0.5x
        self.assertEqual(self.win.lbl_wasd_speed.text(), "0.5x")
        self.assertAlmostEqual(self.win.viewer.wasd_speed_factor, 0.5)

        # ピッチ反転
        self.win.chk_invert_orbit.setChecked(True)
        self.assertTrue(self.win.viewer.invert_orbit_y)
        self.win.chk_invert_orbit.setChecked(False)
        self.assertFalse(self.win.viewer.invert_orbit_y)

    def test_settings_save_and_reset(self):
        """設定保存・復元・初期化リセットの不変条件テスト"""
        # 値を変更
        self.win.slider_rot_speed.setValue(18)
        self.win.slider_pan_speed.setValue(16)
        self.win.slider_zoom_speed.setValue(14)
        self.win.slider_wasd_speed.setValue(12)
        self.win.chk_invert_orbit.setChecked(True)
        self.win.size_slider.setValue(5)
        self.win.slider_fov.setValue(60)
        self.win.chk_depth_fog.setChecked(True)
        self.win.combo_sec_step.setCurrentIndex(4)  # 10cm
        self.win.combo_undo_limit.setCurrentIndex(2) # 100回
        self.win.settings_default_color_combo.setCurrentIndex(2) # Viridis

        # 保存実行
        self.win._save_app_settings(silent=True)

        # リセット実行
        self.win._reset_app_settings()

        # リセット後の値が全て初期値に戻っているか
        self.assertEqual(self.win.slider_rot_speed.value(), 10)
        self.assertEqual(self.win.slider_pan_speed.value(), 10)
        self.assertEqual(self.win.slider_zoom_speed.value(), 10)
        self.assertEqual(self.win.slider_wasd_speed.value(), 10)
        self.assertFalse(self.win.chk_invert_orbit.isChecked())
        self.assertEqual(self.win.size_slider.value(), 2)
        self.assertEqual(self.win.slider_fov.value(), 45)
        self.assertFalse(self.win.chk_depth_fog.isChecked())
        self.assertEqual(self.win.combo_sec_step.currentIndex(), 2)
        self.assertEqual(self.win.combo_undo_limit.currentIndex(), 1)
        self.assertEqual(self.win.settings_default_color_combo.currentIndex(), 0)

        self.assertAlmostEqual(self.win.viewer.rotate_speed, 0.5)
        self.assertAlmostEqual(self.win.viewer.pan_speed_factor, 1.0)
        self.assertAlmostEqual(self.win.viewer.zoom_speed_factor, 1.0)
        self.assertAlmostEqual(self.win.viewer.wasd_speed_factor, 1.0)
        self.assertFalse(self.win.viewer.invert_orbit_y)
        self.assertAlmostEqual(self.win.viewer.fov, 45.0)
        self.assertFalse(self.win.viewer.depth_fog_enabled)

        # 保存された設定を再ロード
        self.win._load_app_settings()
        self.assertEqual(self.win.slider_rot_speed.value(), 18)
        self.assertEqual(self.win.slider_pan_speed.value(), 16)
        self.assertEqual(self.win.slider_zoom_speed.value(), 14)
        self.assertEqual(self.win.slider_wasd_speed.value(), 12)
        self.assertTrue(self.win.chk_invert_orbit.isChecked())
        self.assertEqual(self.win.size_slider.value(), 5)
        self.assertEqual(self.win.slider_fov.value(), 60)
        self.assertTrue(self.win.chk_depth_fog.isChecked())
        self.assertEqual(self.win.combo_sec_step.currentIndex(), 4)
        self.assertEqual(self.win.combo_undo_limit.currentIndex(), 2)
        self.assertEqual(self.win.settings_default_color_combo.currentIndex(), 2)

        # 最後にテスト用の保存設定を初期化しておく
        self.win._reset_app_settings()
        self.win._save_app_settings(silent=True)

    # -------------------------------------------------------------
    # 2. 🗂 ワークフロー切替ナビゲーションバーのテスト
    # -------------------------------------------------------------
    def test_sidebar_workflow_tabs_visibility(self):
        """各タブに切り替えた時、対象のグループボックスのみが正確に表示・非表示制御されるか"""
        tabs_to_check = ["all", "edit", "section", "layer", "filter", "view", "settings", "clean"]
        for tab in tabs_to_check:
            self.win._set_sidebar_workflow_tab(tab)
            # どのタブでもファイル入出力グループは必ず表示されていること
            self.assertTrue(self.win.file_group.isVisible(), f"file_group should be visible in tab '{tab}'")

            if tab == "all":
                self.assertTrue(self.win.layer_group.isVisible())
                self.assertTrue(self.win.tools_group.isVisible())
                self.assertTrue(self.win.section_group.isVisible())
                self.assertTrue(self.win.auto_group.isVisible())
                self.assertTrue(self.win.view_group.isVisible())
                self.assertTrue(self.win.settings_group.isVisible())
            elif tab == "edit":
                self.assertFalse(self.win.layer_group.isVisible())
                self.assertTrue(self.win.tools_group.isVisible())
                self.assertFalse(self.win.section_group.isVisible())
                self.assertFalse(self.win.auto_group.isVisible())
                self.assertFalse(self.win.view_group.isVisible())
                self.assertFalse(self.win.settings_group.isVisible())
            elif tab == "section":
                self.assertFalse(self.win.layer_group.isVisible())
                self.assertFalse(self.win.tools_group.isVisible())
                self.assertTrue(self.win.section_group.isVisible())
                self.assertFalse(self.win.auto_group.isVisible())
                self.assertFalse(self.win.view_group.isVisible())
                self.assertFalse(self.win.settings_group.isVisible())
            elif tab == "layer":
                self.assertTrue(self.win.layer_group.isVisible())
                self.assertFalse(self.win.tools_group.isVisible())
                self.assertFalse(self.win.section_group.isVisible())
                self.assertFalse(self.win.auto_group.isVisible())
                self.assertFalse(self.win.view_group.isVisible())
                self.assertFalse(self.win.settings_group.isVisible())
            elif tab == "filter":
                self.assertFalse(self.win.layer_group.isVisible())
                self.assertFalse(self.win.tools_group.isVisible())
                self.assertFalse(self.win.section_group.isVisible())
                self.assertTrue(self.win.auto_group.isVisible())
                self.assertFalse(self.win.view_group.isVisible())
                self.assertFalse(self.win.settings_group.isVisible())
            elif tab == "view":
                self.assertFalse(self.win.layer_group.isVisible())
                self.assertFalse(self.win.tools_group.isVisible())
                self.assertFalse(self.win.section_group.isVisible())
                self.assertFalse(self.win.auto_group.isVisible())
                self.assertTrue(self.win.view_group.isVisible())
                self.assertFalse(self.win.settings_group.isVisible())
            elif tab == "settings":
                self.assertFalse(self.win.layer_group.isVisible())
                self.assertFalse(self.win.tools_group.isVisible())
                self.assertFalse(self.win.section_group.isVisible())
                self.assertFalse(self.win.auto_group.isVisible())
                self.assertFalse(self.win.view_group.isVisible())
                self.assertTrue(self.win.settings_group.isVisible())
            elif tab == "clean":
                self.assertFalse(self.win.layer_group.isVisible())
                self.assertTrue(self.win.tools_group.isVisible())
                self.assertTrue(self.win.section_group.isVisible())
                self.assertTrue(self.win.auto_group.isVisible())
                self.assertFalse(self.win.view_group.isVisible())
                self.assertFalse(self.win.settings_group.isVisible())

    # -------------------------------------------------------------
    # 3. 🎯 ノイズ専用モードの退避・復元テスト
    # -------------------------------------------------------------
    def test_clean_focus_mode_backup_and_restore(self):
        """ノイズ専用モード突入時に設定値が退避され、解除時に完全に復元されるか"""
        # 初期状態の設定
        self.win.bg_combo.setCurrentIndex(0)  # ダーク
        self.win.size_slider.setValue(2)

        # ノイズ専用モード ON
        self.win._toggle_clean_focus_mode(True)
        self.assertTrue(self.win.viewer.clean_focus_mode)
        self.assertEqual(self.win.bg_combo.currentIndex(), 1)  # ブラック
        self.assertGreaterEqual(self.win.size_slider.value(), 3)
        self.assertTrue(self.win.viewer.selection_mode)
        self.assertEqual(self.win.viewer.selection_shape, PointCloudViewer.SELECT_BRUSH)
        self.assertTrue(self.win.viewer.instant_delete_mode)

        # ノイズ専用モード OFF
        self.win._toggle_clean_focus_mode(False)
        self.assertFalse(self.win.viewer.clean_focus_mode)
        self.assertEqual(self.win.bg_combo.currentIndex(), 0)  # 元のダークへ復元
        self.assertEqual(self.win.size_slider.value(), 2)      # 元の2pxへ復元
        self.assertFalse(self.win.viewer.instant_delete_mode)

    # -------------------------------------------------------------
    # 4. 📐 断面スライサーの折りたたみ・リセットテスト
    # -------------------------------------------------------------
    def test_section_collapse_and_reset(self):
        """断面チェックONでスライダー行が展開し、リセットですべて折りたたまれるか"""
        # 新UIの既定タブは「①自動」で断面グループが隠れているため、断面タブへ切り替えてから検証する
        self.win._set_sidebar_workflow_tab("section")
        QApplication.processEvents()
        # 初期状態: すべて折りたたまれている (非表示)
        self.assertFalse(self.win.sec_x_detail.isVisible())
        self.assertFalse(self.win.sec_y_detail.isVisible())
        self.assertFalse(self.win.sec_z_detail.isVisible())
        self.assertFalse(self.win.sec_wall_detail.isVisible())

        # X断面 ON
        self.win.chk_sec_x.setChecked(True)
        self.assertTrue(self.win.sec_x_detail.isVisible())

        # Y断面 ON
        self.win.chk_sec_y.setChecked(True)
        self.assertTrue(self.win.sec_y_detail.isVisible())

        # Z断面 ON
        self.win.chk_sec_z.setChecked(True)
        self.assertTrue(self.win.sec_z_detail.isVisible())

        # 斜め壁スライス ON
        self.win.chk_sec_wall.setChecked(True)
        self.assertTrue(self.win.sec_wall_detail.isVisible())

        # 断面一括リセット実行
        self.win._reset_all_sections()

        # すべてチェックOFF＆折りたたみ非表示に戻ること
        self.assertFalse(self.win.chk_sec_x.isChecked())
        self.assertFalse(self.win.chk_sec_y.isChecked())
        self.assertFalse(self.win.chk_sec_z.isChecked())
        self.assertFalse(self.win.chk_sec_wall.isChecked())

        self.assertFalse(self.win.sec_x_detail.isVisible())
        self.assertFalse(self.win.sec_y_detail.isVisible())
        self.assertFalse(self.win.sec_z_detail.isVisible())
        self.assertFalse(self.win.sec_wall_detail.isVisible())

    # -------------------------------------------------------------
    # 5. 💾 常設保存ボタンと非ロード時・境界値テスト
    # -------------------------------------------------------------
    def test_save_button_without_data(self):
        """点群未ロード状態で常設保存ボタンを押してもクラッシュせず安全に復帰するか"""
        self.assertEqual(len(self.win.current_scans), 0)
        self.assertIsNone(self.win.viewer.layer_mask)

        # 全モードで例外が出ないことを確認
        for i in range(self.win.export_mode_combo.count()):
            self.win.export_mode_combo.setCurrentIndex(i)
            # QMessageBox がポップアップするので、クラッシュしないことの検証
            # _save_current_scans が安全に早期リターンすること
            try:
                # QMessageBox.warning をモックまたは何もしない関数に差し替えて実行
                from PySide6.QtWidgets import QMessageBox
                orig_warn = QMessageBox.warning
                QMessageBox.warning = lambda *args, **kwargs: QMessageBox.Ok
                self.win._save_current_scans()
            finally:
                QMessageBox.warning = orig_warn

    # -------------------------------------------------------------
    # 6. ✖ Escキー解除および各種ショートカット切替テスト
    # -------------------------------------------------------------
    def test_esc_and_shortcut_modes(self):
        """Escキーによる選択解除、計測解除、壁ピック解除の安定性テスト"""
        # 1. 2点壁ピックモード中のEsc
        self.win._toggle_wall_align_mode(True)
        self.assertTrue(self.win.viewer.wall_align_mode)
        self.win._clear_selection()
        self.assertFalse(self.win.viewer.wall_align_mode)

        # 2. 計測モード中のEsc
        self.win._set_interaction_mode("measure")
        self.assertTrue(self.win.viewer.measure_mode)
        self.win.viewer.measure_points.append((np.array([0, 0, 0]), np.array([0, 0, 0])))
        self.win._clear_selection()
        self.assertEqual(len(self.win.viewer.measure_points), 0)

        # 3. ブラシ選択モード中のEsc
        self.win._set_interaction_mode("brush")
        self.win.viewer.brush_stroke_points = [QApplication.primaryScreen().geometry().topLeft()]
        self.win._clear_selection()
        self.assertEqual(len(self.win.viewer.brush_stroke_points), 0)

        # 4. モード切替のシームレス性 (V, H, B, R, C, L, M)
        self.win._set_interaction_mode("orbit")
        self.assertFalse(self.win.viewer.pan_mode)
        self.assertFalse(self.win.viewer.selection_mode)
        self.assertFalse(self.win.viewer.measure_mode)

        self.win._set_interaction_mode("pan")
        self.assertTrue(self.win.viewer.pan_mode)
        self.assertFalse(self.win.viewer.selection_mode)
        self.assertFalse(self.win.viewer.measure_mode)

        self.win._set_interaction_mode("brush")
        self.assertTrue(self.win.viewer.selection_mode)
        self.assertEqual(self.win.viewer.selection_shape, PointCloudViewer.SELECT_BRUSH)

        self.win._set_interaction_mode("rect")
        self.assertTrue(self.win.viewer.selection_mode)
        self.assertEqual(self.win.viewer.selection_shape, PointCloudViewer.SELECT_RECT)

        self.win._set_interaction_mode("circle")
        self.assertTrue(self.win.viewer.selection_mode)
        self.assertEqual(self.win.viewer.selection_shape, PointCloudViewer.SELECT_CIRCLE)

        self.win._set_interaction_mode("polygon")
        self.assertTrue(self.win.viewer.selection_mode)
        self.assertEqual(self.win.viewer.selection_shape, PointCloudViewer.SELECT_POLYGON)

        self.win._set_interaction_mode("measure")
        self.assertTrue(self.win.viewer.measure_mode)


    # -------------------------------------------------------------
    # 7. 🎯 断面フォーカス・厚みプリセットおよび終了時処理テスト
    # -------------------------------------------------------------
    def test_section_focus_and_presets_without_data(self):
        """点群未ロード時でも断面基点取得や厚みプリセットが例外なく動作するか"""
        # 現在位置を基点にセット
        self.win._set_section_origin_from_current_view()
        self.assertEqual(self.win.spin_pos_x.value(), 0.0)
        self.assertEqual(self.win.spin_pos_y.value(), 0.0)
        self.assertEqual(self.win.spin_pos_z.value(), 0.0)

        # 厚みプリセット適用
        for t_val in [0.1, 0.5, 1.0, 2.0, 5.0]:
            self.win._apply_section_thickness_preset(t_val)
            self.assertEqual(self.win.spin_thick_x.value(), t_val)
            self.assertEqual(self.win.spin_thick_y.value(), t_val)
            self.assertEqual(self.win.spin_thick_z.value(), t_val)

    def test_close_event_safe_shutdown(self):
        """closeEvent呼び出し時に設定が保存され、例外なく正常終了するか"""
        from PySide6.QtGui import QCloseEvent
        event = QCloseEvent()
        self.win.closeEvent(event)
        self.assertTrue(event.isAccepted())


    # -------------------------------------------------------------
    # 8. 🎨 カラーパレット (高さ・反射強度) 拡充テスト
    # -------------------------------------------------------------
    def test_color_palettes_all_gradient_variations(self):
        """全8種類のカラーパレット(Turbo, Viridis, Plasma, Jet, Terrain, Heat, Cool-Warm, Grayscale)が正常計算されるか"""
        pts = np.random.uniform(-5, 5, size=(100, 3)).astype(np.float32)
        ints = np.random.uniform(0, 255, size=100).astype(np.float32)
        self.win.viewer.set_point_cloud(pts, intensities=ints)

        palettes = [
            PointCloudViewer.PALETTE_TURBO,
            PointCloudViewer.PALETTE_VIRIDIS,
            PointCloudViewer.PALETTE_PLASMA,
            PointCloudViewer.PALETTE_JET,
            PointCloudViewer.PALETTE_TERRAIN,
            PointCloudViewer.PALETTE_HEAT,
            PointCloudViewer.PALETTE_COOLWARM,
            PointCloudViewer.PALETTE_GRAYSCALE,
        ]

        # 1. 高さ(Z)モードでの全パレット検証
        self.win.viewer.set_color_mode(PointCloudViewer.COLOR_MODE_HEIGHT)
        for pal in palettes:
            self.win.viewer.set_color_palette(pal)
            cols = self.win.viewer.active_colors
            self.assertEqual(cols.shape, (100, 3))
            self.assertTrue(np.all(cols >= 0.0) and np.all(cols <= 1.0))

        # 2. 反射強度モードでの全パレット検証
        self.win.viewer.set_color_mode(PointCloudViewer.COLOR_MODE_INTENSITY)
        for pal in palettes:
            self.win.viewer.set_color_palette(pal)
            cols = self.win.viewer.active_colors
            self.assertEqual(cols.shape, (100, 3))
            self.assertTrue(np.all(cols >= 0.0) and np.all(cols <= 1.0))

        # 3. 以前あったカラーパレット(Viridis, Jet)および新パレットが直接選択で画面反映されるかテスト
        for mode in [
            PointCloudViewer.COLOR_MODE_HEIGHT,
            PointCloudViewer.COLOR_MODE_HEIGHT_VIRIDIS,
            PointCloudViewer.COLOR_MODE_HEIGHT_JET,
            PointCloudViewer.COLOR_MODE_HEIGHT_PLASMA,
            PointCloudViewer.COLOR_MODE_HEIGHT_TERRAIN,
            PointCloudViewer.COLOR_MODE_HEIGHT_HEAT,
            PointCloudViewer.COLOR_MODE_HEIGHT_COOLWARM,
            PointCloudViewer.COLOR_MODE_INTENSITY,
            PointCloudViewer.COLOR_MODE_INTENSITY_HEAT,
            PointCloudViewer.COLOR_MODE_INTENSITY_PLASMA,
            PointCloudViewer.COLOR_MODE_INTENSITY_TURBO,
            PointCloudViewer.COLOR_MODE_INTENSITY_VIRIDIS,
        ]:
            idx = self.win.color_combo.findText(mode)
            self.assertGreaterEqual(idx, 0, f"Mode '{mode}' should exist in color_combo")
            self.win.color_combo.setCurrentIndex(idx)
            cols = self.win.viewer.active_colors
            self.assertEqual(cols.shape, (100, 3))
            self.assertTrue(np.all(cols >= 0.0) and np.all(cols <= 1.0))


if __name__ == "__main__":
    unittest.main()
