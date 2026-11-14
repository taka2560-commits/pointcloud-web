# -*- coding: utf-8 -*-
"""
空中に浮いたノイズを探すための、反射強度の配色（gui/noise_palette.py）と、3D 画面への組み込みのテスト

配色そのもの
- どの色も、背景（ダーク・ブラック・ホワイト）とのコントラスト比 3 以上を保つこと（どの点も背景に溶けない）
- 弱い反射ほど背景から離れた明るさになること（これまでの配色と明暗が逆）
- 選択中の点の黄色・ノイズレイヤーの赤と、紛れない色だけを使うこと
- 順位で色を決めるので、強度の値の幅が違う点群でも「いちばん弱い 10%」が同じ色になること
- 実データで測った分布（2026-11-14、利用者の現場。合計の数字だけを写したもの）で、空中の点が見える割合が
  白黒の配色より大きく上がること

3D 画面
- 配色の一覧に 2 つ足されていて、切り替えると弱い点が明るくなること
- 陰影を掛けないこと、選択中の点は黄色・ノイズレイヤーは赤のままであること
- 背景を白にすると明暗が反転し、戻すと戻ること
- 重ね描き（元ファイルの読み直し・高精細表示）も同じ表で色が付くこと
- 反射強度の無い点群・点群の入れ替えで困らないこと
- 切り替えたときに色の読み方を知らせること、機能の検索から選べること
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.settings_sandbox import use_sandbox_settings  # noqa: E402
from tests.qt_cleanup import destroy_widget  # noqa: E402

use_sandbox_settings()  # この PC の本番アプリの保存設定を、テストで書き換えない

import numpy as np  # noqa: E402
from PySide6.QtGui import QColor  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from gui import noise_palette as NP  # noqa: E402
from gui.gl_viewer import PointCloudViewer  # noqa: E402

MODES = NP.NOISE_MODES


def lab(rgb) -> np.ndarray:
    """sRGB（0〜1）→ CIE L*a*b*（色の離れ具合を測るため）"""
    c = np.asarray(rgb, dtype=np.float64)
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = lin @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], axis=-1)


def delta_e(a, b) -> np.ndarray:
    return np.sqrt(((lab(a) - lab(b)) ** 2).sum(-1))


class TestPaletteDesign(unittest.TestCase):
    def test_every_color_stands_out_from_its_background(self):
        for mode in MODES:
            dark = NP.noise_lut(mode, light_background=False)
            light = NP.noise_lut(mode, light_background=True)
            self.assertEqual(dark.shape, (NP.LUT_SIZE, 3))
            self.assertEqual(dark.dtype, np.float32)
            for name, bg in (("ダーク", NP.BACKGROUND_DARK), ("ブラック", NP.BACKGROUND_BLACK)):
                self.assertGreaterEqual(float(NP.contrast_ratio(dark, bg).min()), NP.MIN_CONTRAST, f"{mode} / 背景 {name}")
            self.assertGreaterEqual(float(NP.contrast_ratio(light, NP.BACKGROUND_WHITE).min()), NP.MIN_CONTRAST, f"{mode} / 背景 ホワイト")

    def test_weaker_return_is_farther_from_the_background(self):
        for mode in MODES:
            lum_dark = NP.relative_luminance(NP.noise_lut(mode, False))
            lum_light = NP.relative_luminance(NP.noise_lut(mode, True))
            self.assertLessEqual(float(np.diff(lum_dark).max()), 0.005, f"{mode}: 暗い背景では、順位が上がるほど暗くなる")
            self.assertGreaterEqual(float(np.diff(lum_light).min()), -0.005, f"{mode}: 白い背景では、順位が上がるほど明るくなる")
            # いちばん弱い反射がいちばん背景から離れ、いちばん強い反射がいちばん近い
            cr = NP.contrast_ratio(NP.noise_lut(mode, False), NP.BACKGROUND_DARK)
            self.assertGreater(float(cr[0]), 15.0)
            self.assertLess(float(cr[-1]), 3.5)

    def test_the_weakest_band_is_bright_on_a_dark_background(self):
        n = int(NP.LUT_SIZE * NP.HIGHLIGHT_RANK)
        for mode in MODES:
            cr = NP.contrast_ratio(NP.noise_lut(mode, False)[:n], NP.BACKGROUND_DARK)
            self.assertGreaterEqual(float(cr.min()), 9.0, mode)

    def test_colors_are_not_mistaken_for_selection_or_the_noise_layer(self):
        """選択中の点は黄色、ノイズレイヤーは赤で描かれる。配色のどの色も、その 2 つとはっきり違うこと"""
        for mode in MODES:
            for light in (False, True):
                lut = NP.noise_lut(mode, light)
                self.assertGreater(float(delta_e(lut, NP.SELECTION_COLOR).min()), 50.0, f"{mode}: 選択の黄色に近い色があります")
                self.assertGreater(float(delta_e(lut, NP.NOISE_LAYER_COLOR).min()), 50.0, f"{mode}: ノイズレイヤーの赤に近い色があります")
                # 暖色（赤が青より強い色）を使わない
                self.assertTrue(bool(np.all(lut[:, 0] <= lut[:, 2] + 0.02)), f"{mode}: 暖色が混ざっています")

    def test_spot_mode_is_gray_outside_the_weak_band(self):
        lut = NP.noise_lut(NP.MODE_NOISE_SPOT, False)
        outside = lut[int(NP.LUT_SIZE * (NP.HIGHLIGHT_RANK + 0.04)):]
        self.assertLess(float((outside.max(axis=1) - outside.min(axis=1)).max()), 0.05, "弱い帯の外は、ほぼ無彩色")
        inside = lut[: int(NP.LUT_SIZE * NP.HIGHLIGHT_RANK)]
        self.assertGreater(float((inside.max(axis=1) - inside.min(axis=1))[int(NP.LUT_SIZE * 0.01):].min()), 0.25, "弱い帯は、色が付いている")
        # 弱い帯と、その外の灰色は、はっきり違う
        self.assertGreater(float(delta_e(inside[-1], outside[0])), 30.0)

    def test_stops_and_helpers(self):
        for mode in MODES:
            for light in (False, True):
                ranks = NP.stop_ranks(mode, light)
                self.assertEqual((ranks[0], ranks[-1]), (0.0, 1.0))
                self.assertEqual(ranks, sorted(ranks))
            self.assertTrue(NP.is_noise_mode(mode))
            self.assertIn("弱い", NP.legend_text(mode))
        self.assertFalse(NP.is_noise_mode("反射強度"))
        self.assertEqual(NP.legend_text("反射強度"), "")
        self.assertTrue(NP.is_light_background(*NP.BACKGROUND_WHITE))
        self.assertFalse(NP.is_light_background(*NP.BACKGROUND_DARK))
        self.assertFalse(NP.is_light_background(*NP.BACKGROUND_BLACK))
        # 知らない名前は、標準の表で受ける（落ちない）
        self.assertTrue(np.array_equal(NP.noise_lut("知らない配色"), NP.noise_lut(NP.MODE_NOISE_GLOW)))


class TestRank(unittest.TestCase):
    def test_rank_of_uniform_values_is_the_value(self):
        v = np.linspace(0.0, 1.0, 10001, dtype=np.float32)
        curve = NP.rank_curve(v)
        r = NP.to_rank(v, curve)
        self.assertLess(float(np.abs(r - v).max()), 0.005)

    def test_rank_does_not_depend_on_the_value_scale(self):
        """値の幅が違っても（0〜1 と 4,801〜43,667）、いちばん弱い 10% は同じ色になる"""
        rng = np.random.default_rng(1)
        base = rng.gamma(6.0, 1.0, 200_000).astype(np.float32)
        a = base / base.max()
        b = 4801.0 + base * 3000.0
        ca = NP.colorize(a, NP.MODE_NOISE_GLOW, NP.rank_curve(a))
        cb = NP.colorize(b, NP.MODE_NOISE_GLOW, NP.rank_curve(b))
        self.assertLess(float(np.abs(ca - cb).max()), 0.02)
        weakest = np.argsort(a)[: len(a) // 10]
        cr = NP.contrast_ratio(ca[weakest], NP.BACKGROUND_DARK)
        self.assertGreaterEqual(float(np.percentile(cr, 1)), 9.0)

    def test_skewed_values_spread_over_the_whole_ramp(self):
        """強度が狭い範囲に固まっていても（面が同じような強さ）、順位で色を決めるので色の全域を使う"""
        rng = np.random.default_rng(2)
        v = np.clip(rng.normal(0.72, 0.06, 100_000), 0, 1).astype(np.float32)
        r = NP.to_rank(v, NP.rank_curve(v))
        hist, _ = np.histogram(r, bins=10, range=(0, 1))
        self.assertGreater(int(hist.min()), 8000)
        self.assertLess(int(hist.max()), 12000)

    def test_clipped_ends_get_the_middle_of_their_band(self):
        """正規化で 0 と 1 に張り付いた点（各 2%）は、その帯の真ん中の順位になる"""
        v = np.concatenate([np.zeros(2000), np.linspace(0.001, 0.999, 96000), np.ones(2000)]).astype(np.float32)
        curve = NP.rank_curve(v)
        self.assertAlmostEqual(float(NP.to_rank(np.array([0.0], dtype=np.float32), curve)[0]), 0.01, delta=0.004)
        self.assertAlmostEqual(float(NP.to_rank(np.array([1.0], dtype=np.float32), curve)[0]), 0.99, delta=0.004)
        r = NP.to_rank(v, curve)
        self.assertTrue(bool(np.all(np.diff(r[np.argsort(v, kind="stable")]) >= -1e-6)), "順位は値の順を保つ")

    def test_degenerate_inputs(self):
        self.assertIsNone(NP.rank_curve(None))
        self.assertIsNone(NP.rank_curve(np.empty(0, dtype=np.float32)))
        self.assertIsNone(NP.rank_curve(np.full(100, 0.5, dtype=np.float32)))      # 値が 1 種類
        self.assertIsNone(NP.rank_curve(np.full(10, np.nan, dtype=np.float32)))
        # 表が無いときは、値を 0〜1 に切り詰めて順位の代わりにする
        r = NP.to_rank(np.array([-1.0, 0.25, 2.0, np.nan], dtype=np.float32), None)
        self.assertEqual(r.tolist(), [0.0, 0.25, 1.0, 0.0])
        self.assertEqual(NP.colorize(np.empty(0, dtype=np.float32), NP.MODE_NOISE_GLOW).shape, (0, 3))
        # 非数は、いちばん弱い側として色を付ける（落ちない）
        v = np.array([0.1, np.nan, 0.9], dtype=np.float32)
        c = NP.colorize(v, NP.MODE_NOISE_GLOW, NP.rank_curve(np.linspace(0, 1, 100)))
        self.assertTrue(bool(np.all(np.isfinite(c))))

    def test_one_table_from_intensity_to_color(self):
        """3D 画面は「正規化済みの強度 → 色」を 1 つの表にまとめて使う（数千万点でも、表を引くだけで済む）"""
        rng = np.random.default_rng(6)
        t = np.clip(rng.normal(0.7, 0.12, 300_000), 0, 1).astype(np.float32)
        curve = NP.rank_curve(t)
        for mode in MODES:
            for light in (False, True):
                table = NP.color_table(mode, curve, light)
                self.assertEqual(table.shape, (NP.TABLE_SIZE, 3))
                fast = NP.apply_table(t, table)
                exact = NP.colorize(t, mode, curve, light)
                self.assertLess(float(np.abs(fast - exact).max()), 0.03, mode)
                bg = NP.BACKGROUND_WHITE if light else NP.BACKGROUND_DARK
                self.assertGreaterEqual(float(NP.contrast_ratio(table, bg).min()), NP.MIN_CONTRAST)
        self.assertEqual(NP.apply_table(np.empty(0, dtype=np.float32), NP.color_table(NP.MODE_NOISE_GLOW, curve)).shape, (0, 3))
        # 範囲の外の値も、表の端の色になる（落ちない）
        table = NP.color_table(NP.MODE_NOISE_GLOW, curve)
        edge = NP.apply_table(np.array([-0.5, 1.5], dtype=np.float32), table)
        self.assertTrue(np.array_equal(edge, table[[0, -1]]))

    def test_large_arrays_are_sampled(self):
        v = np.random.default_rng(3).random(NP.RANK_SAMPLE_MAX * 2 + 7).astype(np.float32)
        xs, ys = NP.rank_curve(v)
        self.assertLessEqual(len(xs), NP.RANK_KNOTS)
        self.assertLess(float(np.abs(NP.to_rank(np.array([0.1, 0.5, 0.9], dtype=np.float32), (xs, ys)) - [0.1, 0.5, 0.9]).max()), 0.01)


class TestMeasuredDistribution(unittest.TestCase):
    """
    実データで測った分布で、空中の点が見える割合を比べる。
    数字は 2026-11-14 に利用者の現場（1 億 3,564 万点）で数えたもの。座標や点そのものは含まない。
      PCTS / RAW      … 反射強度の百分位と、その値
      BANDS / ISOLATED … 順位の帯と、「周りに点がほとんど無い点」（30cm の立方体に 6 点未満）のうち、その帯にある割合 (%)
    """
    PCTS = [0, 0.1, 1, 2, 5, 10, 25, 50, 75, 90, 95, 98, 99, 99.9, 100]
    RAW = [4801, 9874, 15347, 16619, 18218.55, 19572, 22158, 23863, 25125, 25914, 26374, 26870, 27327, 29743, 43667]
    BANDS = [0, 0.5, 1, 2, 5, 10, 20, 40, 60, 80, 90, 95, 98, 99, 99.5, 100]
    ISOLATED = [3.3, 4.5, 5.8, 9.6, 9.8, 23.3, 29.1, 9.7, 2.7, 0.8, 0.5, 0.3, 0.4, 0.1, 0.2]

    def _isolated_ranks(self) -> np.ndarray:
        """空中の点の順位の標本（帯の中は一様とみなす）"""
        rng = np.random.default_rng(4)
        out = []
        for lo, hi, share in zip(self.BANDS[:-1], self.BANDS[1:], self.ISOLATED):
            out.append(rng.uniform(lo / 100.0, hi / 100.0, int(round(share * 400))))
        return np.concatenate(out)

    def test_weak_returns_hold_most_of_the_airborne_points(self):
        cum = np.cumsum(self.ISOLATED)
        self.assertAlmostEqual(float(cum[-1]), 100.0, delta=0.5)
        self.assertGreater(float(cum[self.BANDS.index(10) - 1]), 30.0)      # いちばん弱い 10% に 3 割以上
        self.assertGreater(float(cum[self.BANDS.index(40) - 1]), 80.0)      # いちばん弱い 40% に 8 割以上
        self.assertLess(float(sum(self.ISOLATED[self.BANDS.index(80):])), 5.0)  # いちばん強い 20% には 5% 未満

    def test_airborne_points_become_visible(self):
        ranks = self._isolated_ranks()
        raw = np.interp(ranks * 100.0, self.PCTS, self.RAW)
        p2, p98 = self.RAW[self.PCTS.index(2)], self.RAW[self.PCTS.index(98)]
        t = np.clip((raw - p2) / (p98 - p2), 0.0, 1.0).astype(np.float32)   # 3D 画面と同じ正規化（2%〜98%）
        gray = np.repeat(t[:, None], 3, axis=1)                              # これまでの「反射強度」（白黒）
        visible_gray = float(np.mean(NP.contrast_ratio(gray, NP.BACKGROUND_DARK) >= NP.MIN_CONTRAST))
        self.assertLess(visible_gray, 0.70, "白黒の配色では、空中の点の 3 割以上が背景に溶ける")
        # 値 → 順位の表（全点の分布から作る。ここでは百分位の表そのもの）
        all_t = np.clip((np.interp(np.linspace(0, 100, 20001), self.PCTS, self.RAW) - p2) / (p98 - p2), 0.0, 1.0)
        curve = NP.rank_curve(all_t)
        for mode in MODES:
            cols = NP.colorize(t, mode, curve)
            cr = NP.contrast_ratio(cols, NP.BACKGROUND_DARK)
            self.assertEqual(float(np.mean(cr >= NP.MIN_CONTRAST)), 1.0, f"{mode}: 見えない点が残っています")
            self.assertGreater(float(np.mean(cr >= 4.5)), 0.90, mode)
            self.assertGreater(float(np.median(cr)), 2.0 * float(np.median(NP.contrast_ratio(gray, NP.BACKGROUND_DARK))), mode)


def _cloud(n: int = 20000, seed: int = 5):
    rng = np.random.default_rng(seed)
    pts = rng.random((n, 3)) * 10.0
    inten = rng.gamma(6.0, 1.0, n).astype(np.float32)
    return pts, inten


class TestViewer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def setUp(self):
        self.v = PointCloudViewer()
        self.pts, self.inten = _cloud()
        self.v.set_point_cloud(self.pts, None, self.inten)

    def tearDown(self):
        self.v.set_point_cloud(None)
        destroy_widget(self.v, close=False)

    def _lum(self, idx) -> np.ndarray:
        return NP.relative_luminance(self.v.active_colors[idx])

    def test_modes_are_listed_with_the_other_intensity_modes(self):
        modes = PointCloudViewer.ALL_COLOR_MODES
        for mode in MODES:
            self.assertIn(mode, modes)
            self.assertTrue(self.v._is_intensity_mode(mode))
            self.assertFalse(self.v._is_height_mode(mode))
        self.assertEqual(modes.index(NP.MODE_NOISE_GLOW), modes.index(PointCloudViewer.COLOR_MODE_INTENSITY_VIRIDIS) + 1)
        self.assertEqual(len(modes), len(set(modes)))
        # 名前が、ほかの配色の合図になる語を含まない（含むと、ほかの色の表が選ばれてしまう）
        for mode in MODES:
            for word in ("Heat", "温熱", "Plasma", "Turbo", "レインボー", "Viridis", "Jet", "地形", "Terrain", "Cool-Warm", "青白赤", "高さ"):
                self.assertNotIn(word, mode)

    def test_weak_points_are_bright_and_strong_points_are_dim(self):
        order = np.argsort(self.inten)
        weakest, strongest = order[:200], order[-200:]
        self.v.set_color_mode(PointCloudViewer.COLOR_MODE_INTENSITY)
        self.v.shading_enabled = False
        self.v._update_active_colors()
        self.assertLess(float(self._lum(weakest).mean()), float(self._lum(strongest).mean()), "これまでの白黒は、弱い点が暗い")
        for mode in MODES:
            self.v.set_color_mode(mode)
            self.assertGreater(float(self._lum(weakest).min()), float(self._lum(strongest).max()), f"{mode}: 弱い点のほうが明るい")
            cr = NP.contrast_ratio(self.v.active_colors, NP.BACKGROUND_DARK)
            self.assertGreaterEqual(float(cr.min()), NP.MIN_CONTRAST, f"{mode}: 背景に溶ける点があります")
            self.assertGreaterEqual(float(NP.contrast_ratio(self.v.active_colors[weakest], NP.BACKGROUND_DARK).min()), 12.0)
            self.v.validate_invariants()

    def test_colors_come_from_the_table_without_shading(self):
        self.assertTrue(self.v.shading_enabled, "陰影は標準で入っている")
        for mode in MODES:
            self.v.set_color_mode(mode)
            t = self.v._normalize_intensities()
            expect = NP.apply_table(t, self.v._noise_color_table(mode))
            self.assertTrue(np.array_equal(self.v.active_colors, expect), f"{mode}: 表の色そのまま（陰影で暗くしない）")
            self.assertTrue(np.array_equal(self.v.display_colors, expect[self.v.display_orig_indices]))
            # 「強度 → 色」を 1 つの表にまとめても、点ごとに順位を補間した色とほぼ同じ
            exact = NP.colorize(t, mode, NP.rank_curve(t), False)
            self.assertLess(float(np.abs(expect - exact).max()), 0.03, mode)
        # ふつうの反射強度には、これまでどおり陰影が掛かる
        self.v.set_color_mode(PointCloudViewer.COLOR_MODE_INTENSITY)
        plain = self.v._colormap_grayscale(self.v._normalize_intensities())
        self.assertFalse(np.array_equal(self.v.active_colors, plain))

    def test_selection_and_noise_layer_keep_their_own_colors(self):
        v = self.v
        v.set_color_mode(NP.MODE_NOISE_GLOW)
        noise_idx = np.arange(100, 200)
        v.move_indices_to_noise(noise_idx)
        v.selected_indices = np.arange(0, 50, dtype=np.int64)
        v._update_active_colors()
        self.assertTrue(np.allclose(v.active_colors[:50], NP.SELECTION_COLOR))
        self.assertTrue(np.allclose(v.active_colors[noise_idx], NP.NOISE_LAYER_COLOR))
        rest = v.active_colors[200:]
        self.assertGreater(float(delta_e(rest, NP.SELECTION_COLOR).min()), 50.0)
        self.assertGreater(float(delta_e(rest, NP.NOISE_LAYER_COLOR).min()), 50.0)

    def test_white_background_flips_the_table(self):
        v = self.v
        order = np.argsort(self.inten)
        weakest, strongest = order[:200], order[-200:]
        for mode in MODES:
            v.set_bg_color(QColor(20, 20, 26))
            v.set_color_mode(mode)
            dark_colors = v.active_colors.copy()
            v.set_bg_color(QColor(245, 245, 247))
            self.assertFalse(np.array_equal(v.active_colors, dark_colors), f"{mode}: 背景を白にしたら色の表が替わる")
            self.assertLess(float(self._lum(weakest).max()), float(self._lum(strongest).min()), f"{mode}: 白い背景では弱い点が濃い")
            self.assertGreaterEqual(float(NP.contrast_ratio(v.active_colors, NP.BACKGROUND_WHITE).min()), NP.MIN_CONTRAST)
            self.assertTrue(np.array_equal(v.display_colors, v.active_colors[v.display_orig_indices]), "描いている色も替わる")
            v.set_bg_color(QColor(0, 0, 0))
            self.assertTrue(np.array_equal(v.active_colors, dark_colors), f"{mode}: 暗い背景へ戻すと元の色")
        # ほかの配色は、背景を替えても色を作り直さない
        v.set_color_mode(PointCloudViewer.COLOR_MODE_INTENSITY)
        before = v.active_colors
        v.set_bg_color(QColor(245, 245, 247))
        self.assertIs(v.active_colors, before)

    def test_overlay_uses_the_same_table(self):
        """元ファイルから読み直した点（重ね描き）も、読み込み済みの点と同じ強さなら同じ色になる"""
        v = self.v
        v.set_color_mode(NP.MODE_NOISE_GLOW)
        pick = np.arange(0, 2000)
        v.set_view_overlay(v.points_centered[pick].copy(), None, self.inten[pick].copy(), n_total=len(pick))
        self.assertIsNotNone(v.overlay_colors)
        self.assertTrue(np.allclose(v.overlay_colors, v.active_colors[pick], atol=1e-6))
        v.set_color_mode(NP.MODE_NOISE_SPOT)
        self.assertTrue(np.allclose(v.overlay_colors, v.active_colors[pick], atol=1e-6))

    def test_cloud_without_intensity_falls_back(self):
        v = self.v
        v.set_point_cloud(self.pts, None, None)
        for mode in MODES:
            v.set_color_mode(mode)
            self.assertEqual(v.active_colors.shape, (len(self.pts), 3))
            self.assertTrue(bool(np.all(np.isfinite(v.active_colors))))
            v.validate_invariants()

    def test_constant_intensity_does_not_fail(self):
        v = self.v
        v.set_point_cloud(self.pts, None, np.full(len(self.pts), 0.4, dtype=np.float32))
        v.set_color_mode(NP.MODE_NOISE_GLOW)
        self.assertTrue(bool(np.all(np.isfinite(v.active_colors))))
        self.assertGreaterEqual(float(NP.contrast_ratio(v.active_colors, NP.BACKGROUND_DARK).min()), NP.MIN_CONTRAST)

    def test_a_new_cloud_gets_its_own_ranks(self):
        v = self.v
        v.set_color_mode(NP.MODE_NOISE_GLOW)
        first = v._noise_color_table(NP.MODE_NOISE_GLOW)
        self.assertIs(v._noise_color_table(NP.MODE_NOISE_GLOW), first, "同じ点群では作り直さない")
        self.assertIsNot(v._noise_color_table(NP.MODE_NOISE_SPOT), first, "配色ごとに別の表")
        pts, inten = _cloud(n=5000, seed=9)
        v.set_point_cloud(pts, None, inten * 1000.0 + 5000.0)
        v.set_color_mode(NP.MODE_NOISE_GLOW)
        self.assertIsNot(v._noise_color_table(NP.MODE_NOISE_GLOW), first, "点群が入れ替わったら作り直す")
        order = np.argsort(inten)
        self.assertGreater(float(self._lum(order[:50]).min()), float(self._lum(order[-50:]).max()))


class TestMainWindow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)
        from gui.main_window import MainWindow

        cls.win = MainWindow()
        cls.win._reset_app_settings()
        cls.win.resize(1450, 900)
        cls.win.show()
        cls.app.processEvents()
        pts, inten = _cloud(n=8000, seed=11)
        cls.win.viewer.set_point_cloud(pts, None, inten)
        cls.app.processEvents()

    @classmethod
    def tearDownClass(cls):
        cls.win._set_dirty(False)
        cls.win.viewer.set_point_cloud(None)
        destroy_widget(cls.win)

    def test_combos_list_the_modes(self):
        win = self.win
        for combo in (win.color_combo, win.settings_default_color_combo):
            texts = [combo.itemText(i) for i in range(combo.count())]
            for mode in MODES:
                self.assertIn(mode, texts)

    def test_switching_tells_how_to_read_the_colors(self):
        win = self.win
        win.color_combo.setCurrentText(NP.MODE_NOISE_GLOW)
        self.app.processEvents()
        self.assertEqual(win.viewer.color_mode, NP.MODE_NOISE_GLOW)
        self.assertTrue(win.toast_label.isVisible())
        toast = win.toast_label.text()
        self.assertIn("ノイズ探索", toast)
        self.assertIn("白〜水色 = いちばん弱い反射", toast)
        # ふつうの配色へ戻したときは、読み方を添えない
        win.color_combo.setCurrentText(PointCloudViewer.COLOR_MODE_RGB)
        self.app.processEvents()
        self.assertNotIn("弱い反射", win.toast_label.text())

    def test_commands_switch_to_the_modes(self):
        win = self.win
        cmds = {c.id: c for c in win.ui_extras.commands()}
        for cid, mode in (("view.color_noise", NP.MODE_NOISE_GLOW), ("view.color_noise_soft", NP.MODE_NOISE_SPOT)):
            self.assertIn(cid, cmds)
            cmds[cid].run(win)
            self.app.processEvents()
            self.assertEqual(win.color_combo.currentText(), mode)
            self.assertEqual(win.viewer.color_mode, mode)
        # これまでの「配色: 反射強度」は、これまでどおり白黒の反射強度を選ぶ
        cmds["view.color_intensity"].run(win)
        self.assertEqual(win.color_combo.currentText(), PointCloudViewer.COLOR_MODE_INTENSITY)

    def test_mode_can_be_the_default_at_load(self):
        win = self.win
        win.settings_default_color_combo.setCurrentText(NP.MODE_NOISE_SPOT)
        self.app.processEvents()
        self.assertEqual(win.viewer.default_color_mode, NP.MODE_NOISE_SPOT)
        win.settings_default_color_combo.setCurrentIndex(0)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
