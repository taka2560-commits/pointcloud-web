"""
断面シークのスライダーの範囲と刻み幅（gui/section_seek.py）、数値欄の幅（gui/spin_fit.py）のテスト

2026-11-13 の不具合の再発防止:
- 点群を読み込んだ後に設定の「断面シーク刻み」を変えると、スライダーの倍率だけが変わり、範囲は古い刻みのまま残った。
  高さ 7.20〜50.69m の点群で 1cm → 5cm に変えると、スライダーの左端が 720 ÷ 20 = 36.00m になり、地面（10m 付近）まで下げられなかった
- 点群を読み込むと、設定の刻みに関係なく倍率が 1cm に戻っていた
- 5mm・1mm の刻みは、数値欄の小数が 2 桁のままで、上下ボタンを押しても値が動かなかった
- 数値欄が幅 78px の固定で、桁の多い座標（-148769.06 m）が途中で切れて読めなかった

「真ん中が起点」（2026-11-13、利用者の提案「スライダー起点を真ん中に」「どちらのスライダーも」）:
- シーク: 基点がバーの真ん中。点群の最小・最大までは必ず届く
- 厚み幅: 真ん中が 1m。左端 1cm、右端 100m（以前は 0.1m〜敷地の大きさを等間隔に割り当て、1m が左端に張り付いていた）
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.settings_sandbox import use_sandbox_settings  # noqa: E402

use_sandbox_settings()  # この PC の本番アプリの保存設定を、テストで書き換えない

import numpy as np
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication

from gui.section_seek import (
    SLIDER_INT_LIMIT, THICK_SLIDER_MAX, centered_seek_span, seek_scale, seek_slider_range, seek_slider_value,
    spin_decimals_for, thick_from_slider, thick_to_slider,
)

# 利用者のファイル（平面直角座標・高さ 7.20〜50.69m）と同じ範囲
WORLD_MIN = np.array([-43551.46, -148819.79, 7.20])
WORLD_MAX = np.array([-43393.46, -148717.68, 50.69])
STEPS = (0.001, 0.005, 0.01, 0.05, 0.1)  # 設定「断面シーク刻み」の並び順


def _points(n=6000):
    rng = np.random.default_rng(3)
    pts = WORLD_MIN + rng.random((n, 3)) * (WORLD_MAX - WORLD_MIN)
    pts[0] = WORLD_MIN
    pts[1] = WORLD_MAX
    return pts


class TestSeekMath(unittest.TestCase):
    def test_scale_is_the_inverse_of_the_step(self):
        for step, want in zip(STEPS, (1000.0, 200.0, 100.0, 20.0, 10.0)):
            self.assertAlmostEqual(seek_scale(step, WORLD_MIN, WORLD_MAX), want, places=6)

    def test_scale_falls_back_for_bad_steps(self):
        for bad in (0, -1, None, float("nan"), "x"):
            self.assertAlmostEqual(seek_scale(bad, WORLD_MIN, WORLD_MAX), 100.0, places=6)

    def test_scale_is_reduced_only_when_the_slider_would_overflow(self):
        big_min, big_max = np.array([-1e7, -1e7, 0.0]), np.array([1e7, 1e7, 100.0])
        scale = seek_scale(0.001, big_min, big_max)
        self.assertLess(scale, 1000.0)
        self.assertLessEqual(scale * 1e7, SLIDER_INT_LIMIT)
        lo, hi = seek_slider_range(big_min[0], big_max[0], scale)
        self.assertLess(max(abs(lo), abs(hi)), 2**31 - 1)
        # 非有限値が混じっても落ちない
        self.assertGreater(seek_scale(0.01, [float("nan"), 0, 0], [float("inf"), 1, 1]), 0)

    def test_range_always_covers_the_cloud(self):
        for step in STEPS:
            scale = seek_scale(step, WORLD_MIN, WORLD_MAX)
            for axis in range(3):
                lo, hi = seek_slider_range(WORLD_MIN[axis], WORLD_MAX[axis], scale)
                self.assertLessEqual(lo / scale, WORLD_MIN[axis] + 1e-9, f"左端が点群の最小に届きません (step={step})")
                self.assertGreaterEqual(hi / scale, WORLD_MAX[axis] - 1e-9, f"右端が点群の最大に届きません (step={step})")
                self.assertGreater(lo / scale, WORLD_MIN[axis] - step - 1e-9, "左端は 1 刻みより外へ出ない")
                self.assertLess(hi / scale, WORLD_MAX[axis] + step + 1e-9, "右端は 1 刻みより外へ出ない")

    def test_range_of_a_flat_cloud_is_not_empty(self):
        lo, hi = seek_slider_range(5.0, 5.0, 100.0)
        self.assertLess(lo, hi)
        lo, hi = seek_slider_range(9.0, 3.0, 100.0)  # 逆順でも
        self.assertEqual((lo, hi), (300, 900))

    def test_value_and_decimals(self):
        self.assertEqual(seek_slider_value(36.0, 20.0), 720)
        self.assertEqual(seek_slider_value(-148769.06, 100.0), -14876906)
        self.assertEqual([spin_decimals_for(s) for s in STEPS], [3, 3, 2, 2, 2])
        self.assertEqual(spin_decimals_for(None), 2)


class _WinCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def setUp(self):
        from gui.main_window import MainWindow

        self.win = MainWindow()
        self.win._reset_app_settings()
        self.win.resize(1450, 900)
        self.win.show()
        self._pump()

    def tearDown(self):
        self.win._set_dirty(False)
        self.win.close()
        self.win.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        QApplication.processEvents()

    def _pump(self, n=5):
        for _ in range(n):
            QApplication.processEvents()

    def _load(self):
        self.win.viewer.set_point_cloud(_points())
        self._pump()

    def _ends(self, axis="z"):
        """スライダーを左端・右端へ動かしたときの位置 (m)"""
        win = self.win
        slider, spin = getattr(win, f"slider_pos_{axis}"), getattr(win, f"spin_pos_{axis}")
        keep = spin.value()
        slider.setValue(slider.minimum())
        left = spin.value()
        slider.setValue(slider.maximum())
        right = spin.value()
        spin.setValue(keep)
        return left, right


class TestSeekSliderRange(_WinCase):
    def test_changing_the_step_after_loading_keeps_the_slider_on_the_cloud(self):
        """利用者の手順: 1cm のまま読み込み → 設定で 5cm に変更 → Z シークを左端へ"""
        win = self.win
        self._load()
        self.assertEqual(win.combo_sec_step.currentData(), 0.01)
        win.chk_sec_z.setChecked(True)
        win.spin_pos_z.setValue(36.0)
        win.combo_sec_step.setCurrentIndex(3)  # 5cm
        self.assertAlmostEqual(win.spin_pos_z.value(), 36.0, places=6, msg="刻みを変えても、今の位置は動かない")
        self.assertEqual(win.slider_pos_z.value(), 720)
        left, right = self._ends("z")
        self.assertLess(left, 7.21, f"左端が点群のいちばん下に届きません（以前は 36.00m で止まった）: {left}")
        self.assertGreater(left, 7.20 - 0.05 - 1e-6)
        self.assertGreater(right, 50.68)
        self.assertLess(right, 50.69 + 0.05 + 1e-6, f"右端が点群の外へ大きく出ています（以前は 253m まで動いた）: {right}")

    def test_every_step_and_axis_reaches_both_ends(self):
        win = self.win
        self._load()
        for idx, step in enumerate(STEPS):
            win.combo_sec_step.setCurrentIndex(idx)
            self.assertAlmostEqual(win._sec_pos_scale, 1.0 / step, places=6)
            for axis_i, axis in enumerate("xyz"):
                left, right = self._ends(axis)
                tol = step + 0.006  # 数値欄の小数の丸めぶん
                self.assertLessEqual(left, WORLD_MIN[axis_i] + tol, f"{axis} 左端 step={step}: {left}")
                self.assertGreaterEqual(left, WORLD_MIN[axis_i] - tol, f"{axis} 左端 step={step}: {left}")
                self.assertGreaterEqual(right, WORLD_MAX[axis_i] - tol, f"{axis} 右端 step={step}: {right}")
                self.assertLessEqual(right, WORLD_MAX[axis_i] + tol, f"{axis} 右端 step={step}: {right}")

    def test_loading_does_not_reset_the_step(self):
        """刻みを 5cm にしてから読み込んでも、スライダーは 5cm 刻みのまま（以前は読み込みで 1cm に戻った）"""
        win = self.win
        win.combo_sec_step.setCurrentIndex(3)
        self._load()
        self.assertAlmostEqual(win._sec_pos_scale, 20.0, places=6)
        s = win.slider_pos_z
        self.assertEqual((s.minimum(), s.maximum()), (144, 1014))
        mid = 0.5 * (WORLD_MIN[2] + WORLD_MAX[2])
        start = win.spin_pos_z.value()
        self.assertAlmostEqual(start, mid, delta=0.006)
        self.assertEqual(s.value(), seek_slider_value(start, 20.0))
        s.setValue(s.value() + 1)
        self.assertAlmostEqual(win.spin_pos_z.value(), (seek_slider_value(start, 20.0) + 1) / 20.0, places=6)

    def test_seek_from_current_view_stays_consistent(self):
        win = self.win
        self._load()
        win.combo_sec_step.setCurrentIndex(4)  # 10cm
        win._set_section_origin_from_current_view()
        for axis in "xyz":
            slider, spin = getattr(win, f"slider_pos_{axis}"), getattr(win, f"spin_pos_{axis}")
            self.assertLessEqual(abs(slider.value() - seek_slider_value(spin.value(), win._sec_pos_scale)), 1)
            self.assertGreaterEqual(slider.value(), slider.minimum())
            self.assertLessEqual(slider.value(), slider.maximum())

    def test_fine_steps_can_be_entered_and_stepped(self):
        """5mm・1mm の刻み: 数値欄が小数 3 桁になり、上下ボタンで値が動く（以前は 2 桁に丸められて動かなかった）"""
        win = self.win
        self._load()
        for idx, step in ((0, 0.001), (1, 0.005)):
            win.combo_sec_step.setCurrentIndex(idx)
            for name in ("spin_pos_x", "spin_pos_y", "spin_pos_z", "spin_pos_wall"):
                self.assertEqual(getattr(win, name).decimals(), 3, name)
            win.spin_pos_z.setValue(12.0)
            win.spin_pos_z.stepUp()
            self.assertAlmostEqual(win.spin_pos_z.value(), 12.0 + step, places=6)
            self.assertEqual(win.slider_pos_z.value(), seek_slider_value(12.0 + step, 1.0 / step))
        win.combo_sec_step.setCurrentIndex(2)
        self.assertEqual(win.spin_pos_z.decimals(), 2)

    def test_reset_settings_restores_one_centimeter(self):
        win = self.win
        self._load()
        win.combo_sec_step.setCurrentIndex(4)
        win._reset_app_settings()
        self.assertAlmostEqual(win._sec_pos_scale, 100.0, places=6)
        left, right = self._ends("z")
        self.assertAlmostEqual(left, 7.20, places=2)
        self.assertAlmostEqual(right, 50.69, places=2)


class TestCenterMath(unittest.TestCase):
    def test_span_is_the_cloud_when_the_origin_is_its_middle(self):
        a, b = centered_seek_span(0.5 * (7.2 + 50.69), 7.2, 50.69)
        self.assertAlmostEqual(a, 7.2, places=9)
        self.assertAlmostEqual(b, 50.69, places=9)

    def test_span_keeps_the_origin_in_the_middle_and_covers_the_cloud(self):
        for c in (7.2, 10.5, 28.0, 45.0, 50.69):
            a, b = centered_seek_span(c, 7.2, 50.69)
            self.assertAlmostEqual(0.5 * (a + b), c, places=9, msg="基点が真ん中")
            self.assertLessEqual(a, 7.2 + 1e-9)
            self.assertGreaterEqual(b, 50.69 - 1e-9)
            self.assertLessEqual(b - a, 2 * (50.69 - 7.2) + 1e-9, "範囲は点群の幅の 2 倍まで")

    def test_span_handles_odd_inputs(self):
        self.assertEqual(centered_seek_span(99.0, 7.2, 50.69)[1] - 50.69, 50.69 - 7.2)  # 点群の外の基点は端へ寄せる
        a, b = centered_seek_span(float("nan"), 0.0, 10.0)
        self.assertEqual((a, b), (0.0, 10.0))
        a, b = centered_seek_span(3.0, 3.0, 3.0)  # 厚みの無い点群
        self.assertLess(a, 3.0)
        self.assertGreater(b, 3.0)
        self.assertEqual(centered_seek_span(5.0, 10.0, 0.0), (0.0, 10.0))  # 逆順

    def test_thickness_middle_is_one_meter(self):
        mid = THICK_SLIDER_MAX // 2
        self.assertEqual(thick_from_slider(mid), 1.0)
        self.assertEqual(thick_to_slider(1.0), mid)
        self.assertEqual(thick_from_slider(0), 0.01)
        self.assertEqual(thick_from_slider(THICK_SLIDER_MAX), 100.0)
        self.assertEqual(thick_from_slider(mid // 2), 0.1, "0.1m は左から 1/4")
        self.assertEqual(thick_from_slider(mid + mid // 2), 10.0, "10m は左から 3/4")

    def test_thickness_is_monotonic_and_round_trips(self):
        vals = [thick_from_slider(v) for v in range(THICK_SLIDER_MAX + 1)]
        self.assertTrue(all(b >= a for a, b in zip(vals, vals[1:])))
        self.assertGreater(len(set(vals)), 200, "細かく選べる")
        for t in (0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0):  # 「厚み一括」のボタンの値など
            self.assertAlmostEqual(thick_from_slider(thick_to_slider(t)), t, places=6, msg=f"{t}m")
        # 切りのよい値になる
        for v in vals:
            q = 0.01 if v < 1.0 else (0.05 if v < 10.0 else 0.5)
            self.assertAlmostEqual(round(v / q) * q, v, places=6)

    def test_thickness_out_of_range_goes_to_the_ends(self):
        self.assertEqual(thick_to_slider(5000.0), THICK_SLIDER_MAX)
        self.assertEqual(thick_to_slider(0.0), 0)
        self.assertEqual(thick_to_slider(-1.0), 0)
        self.assertEqual(thick_to_slider(float("nan")), 0)
        self.assertEqual(thick_from_slider(-5), 0.01)
        self.assertEqual(thick_from_slider(10**6), 100.0)


class TestCenteredSliders(_WinCase):
    def _assert_centered(self, slider, msg=""):
        self.assertLessEqual(abs(2 * slider.value() - (slider.minimum() + slider.maximum())), 2, f"つまみが真ん中にありません {msg}")

    def test_seek_starts_in_the_middle(self):
        win = self.win
        self._load()
        for axis_i, axis in enumerate("xyz"):
            slider = getattr(win, f"slider_pos_{axis}")
            self._assert_centered(slider, axis)
            left, right = self._ends(axis)
            self.assertAlmostEqual(left, WORLD_MIN[axis_i], delta=0.011)
            self.assertAlmostEqual(right, WORLD_MAX[axis_i], delta=0.011)

    def test_setting_the_origin_puts_it_in_the_middle(self):
        """地面の近く（点群の下端）を基点にしても、つまみは真ん中。左右どちらへも動かせ、点群の上端まで届く"""
        win = self.win
        self._load()
        pts = win.viewer.points_raw
        low = np.argsort(pts[:, 2])[:5]
        win.viewer.selected_indices = low.astype(np.int64)
        want = pts[low].mean(axis=0)
        win._set_section_origin_from_current_view()
        scale = win._sec_pos_scale
        for axis_i, axis in enumerate("xyz"):
            slider, spin = getattr(win, f"slider_pos_{axis}"), getattr(win, f"spin_pos_{axis}")
            self.assertAlmostEqual(spin.value(), want[axis_i], delta=0.006)
            self._assert_centered(slider, axis)
            self.assertLessEqual(slider.minimum() / scale, WORLD_MIN[axis_i] + 1e-6, "点群の最小まで届く")
            self.assertGreaterEqual(slider.maximum() / scale, WORLD_MAX[axis_i] - 1e-6, "点群の最大まで届く")
            self.assertLess(slider.minimum(), slider.value())
            self.assertGreater(slider.maximum(), slider.value())
            # スライダーで動かせる所は、数値欄にも入る
            self.assertLessEqual(spin.minimum(), slider.minimum() / scale + 1e-6)
            self.assertGreaterEqual(spin.maximum(), slider.maximum() / scale - 1e-6)
            slider.setValue(slider.minimum())
            self.assertAlmostEqual(spin.value(), slider.minimum() / scale, delta=0.006)
            slider.setValue(slider.maximum())
            self.assertAlmostEqual(spin.value(), slider.maximum() / scale, delta=0.006)
            spin.setValue(want[axis_i])
            self.assertIn("真ん中が基点", slider.toolTip())

        # 刻みを変えても、基点は真ん中のまま
        z_before = win.spin_pos_z.value()
        win.combo_sec_step.setCurrentIndex(3)
        self.assertAlmostEqual(win.spin_pos_z.value(), z_before, places=6)
        self._assert_centered(win.slider_pos_z, "刻み変更後")

        # つまみを動かしただけでは置き直さない（動かしている最中にバーの範囲が変わると使いにくい）
        rng_before = (win.slider_pos_z.minimum(), win.slider_pos_z.maximum())
        win.slider_pos_z.setValue(win.slider_pos_z.value() + 40)
        win.spin_pos_z.setValue(win.spin_pos_z.value() + 1.0)
        self.assertEqual((win.slider_pos_z.minimum(), win.slider_pos_z.maximum()), rng_before)

        # 別の点群を読み込むと、基点は新しい点群の中央に戻る
        self._load()
        left, right = self._ends("z")
        self.assertAlmostEqual(left, WORLD_MIN[2], delta=0.051)
        self.assertAlmostEqual(right, WORLD_MAX[2], delta=0.051)
        self._assert_centered(win.slider_pos_z, "読み込み直し後")

    def test_thickness_slider_middle_is_one_meter(self):
        win = self.win
        mid = THICK_SLIDER_MAX // 2
        sliders = [getattr(win, f"slider_thick_{a}") for a in ("x", "y", "z", "wall")]
        spins = [getattr(win, f"spin_thick_{a}") for a in ("x", "y", "z", "wall")]
        # 点群を読み込む前から、真ん中が 1m の目盛り
        for slider, spin in zip(sliders, spins):
            self.assertEqual((slider.minimum(), slider.maximum()), (0, THICK_SLIDER_MAX))
            self.assertEqual(slider.value(), thick_to_slider(spin.value()))
            self.assertIn("真ん中が 1m", slider.toolTip())
        self.assertEqual(win.slider_thick_wall.value(), mid, "斜め壁の既定 1m は真ん中")

        self._load()
        win.chk_sec_z.setChecked(True)
        for slider, spin in zip(sliders, spins):
            self.assertEqual((slider.minimum(), slider.maximum()), (0, THICK_SLIDER_MAX))
            spin.setValue(1.0)
            self.assertEqual(slider.value(), mid, "1m は真ん中（以前は左端に張り付いていた）")
            slider.setValue(mid // 2)
            self.assertAlmostEqual(spin.value(), 0.1, places=6)
            slider.setValue(0)
            self.assertAlmostEqual(spin.value(), 0.01, places=6)
            slider.setValue(THICK_SLIDER_MAX)
            self.assertAlmostEqual(spin.value(), 100.0, places=6)
            slider.setValue(mid + 30)
            self.assertAlmostEqual(spin.value(), 2.0, places=6)
        # スライダーで決めた厚みが、そのまま断面に使われる
        win.sec_origin_mode_combo.setCurrentIndex(0)  # シーク位置を中心
        win.spin_pos_z.setValue(20.0)
        win.slider_thick_z.setValue(mid - 30)  # 0.5m
        self.assertAlmostEqual(win.spin_thick_z.value(), 0.5, places=6)
        self.assertAlmostEqual(win.viewer.clip_z_range[0], 19.75, places=4)
        self.assertAlmostEqual(win.viewer.clip_z_range[1], 20.25, places=4)

    def test_thickness_presets_and_typed_values(self):
        win = self.win
        self._load()
        win.chk_sec_z.setChecked(True)
        for t in (0.1, 0.5, 1.0, 2.0, 5.0):
            win._apply_section_thickness_preset(t)
            self.assertAlmostEqual(win.spin_thick_z.value(), t, places=6)
            self.assertEqual(win.slider_thick_z.value(), thick_to_slider(t))
        # 数値欄には、スライダーの右端（100m）より大きい厚みも入る。つまみは右端に寄るだけで、値は変わらない
        win.spin_thick_z.setValue(250.0)
        self.assertEqual(win.slider_thick_z.value(), THICK_SLIDER_MAX)
        self.assertAlmostEqual(win.spin_thick_z.value(), 250.0, places=6)
        # 切りのよくない値を打っても、数値欄の値はそのまま
        win.spin_thick_z.setValue(2.37)
        self.assertAlmostEqual(win.spin_thick_z.value(), 2.37, places=6)
        self.assertEqual(win.slider_thick_z.value(), thick_to_slider(2.37))


class TestSeekSpinWidth(_WinCase):
    """数値欄に、座標が欠けずに入る（横 1,450px の画面・明るいテーマ）"""

    def _assert_fits(self, spin, text=None):
        from gui.spin_fit import TEXT_SLACK, spin_overhead

        fm = spin.lineEdit().fontMetrics()
        text = spin.text() if text is None else text
        room = spin.width() - spin_overhead(spin)
        self.assertGreaterEqual(room, fm.horizontalAdvance(text) + TEXT_SLACK - 2,
                                f"数値欄に「{text}」が入りません（欄 {spin.width()}px・文字の入る幅 {room}px）")

    def test_long_coordinates_are_fully_visible(self):
        win = self.win
        win.apply_ui_theme("kiseki")
        self._load()
        win._set_sidebar_workflow_tab("section")
        for chk in (win.chk_sec_x, win.chk_sec_y, win.chk_sec_z):
            chk.setChecked(True)
        self._pump(10)
        self.assertIn("-148768", win.spin_pos_y.text())
        widths = set()
        for axis in "xyz":
            pos, thick, slider = (getattr(win, f"{n}_{axis}") for n in ("spin_pos", "spin_thick", "slider_pos"))
            self._assert_fits(pos)
            self._assert_fits(pos, pos.textFromValue(pos.minimum()) + pos.suffix())
            self._assert_fits(thick)
            widths.update((pos.width(), thick.width()))
            self.assertGreaterEqual(slider.width(), 150, "数値欄を広げても、スライダーは使える長さを保つ")
        self.assertEqual(len(widths), 1, f"数値欄の幅がそろっていません: {widths}")

        # 1mm 刻み（小数 3 桁）にしても欠けない
        win.combo_sec_step.setCurrentIndex(0)
        self._pump(10)
        self.assertIn(".", win.spin_pos_y.text())
        self.assertEqual(win.spin_pos_y.decimals(), 3)
        self._assert_fits(win.spin_pos_y)

        # テーマを替えても欠けない
        win.apply_ui_theme("dark")
        self._pump(10)
        self._assert_fits(win.spin_pos_y)

    def test_small_coordinates_do_not_waste_width(self):
        win = self.win
        win.viewer.set_point_cloud(np.random.default_rng(1).random((2000, 3)) * [10.0, 8.0, 3.0])
        win._set_sidebar_workflow_tab("section")
        win.chk_sec_z.setChecked(True)
        self._pump(10)
        self._assert_fits(win.spin_pos_z)
        self.assertLess(win.spin_pos_z.width(), 110, "桁の少ない座標では、数値欄を広げすぎない")

    def test_overhead_matches_the_real_widget(self):
        from gui.spin_fit import spin_overhead

        win = self.win
        self._load()
        win._set_sidebar_workflow_tab("section")
        win.chk_sec_z.setChecked(True)
        self._pump(10)
        sp = win.spin_pos_z
        self.assertTrue(sp.isVisible())
        self.assertLessEqual(abs(spin_overhead(sp) - (sp.width() - sp.lineEdit().width())), 2)
        # 隠れている欄（斜め壁）でも同じ値が求められる
        self.assertLessEqual(abs(spin_overhead(win.spin_pos_wall) - spin_overhead(sp)), 2)


if __name__ == "__main__":
    unittest.main()
