# -*- coding: utf-8 -*-
"""
色の部分更新（gui/color_patch.py）のテスト

きっかけ（2026-11-14）: 選択が 1 点変わるだけでも、全点の色の配列を作り直し、表示用の色を全点ぶん写し直し、
色の VBO をまるごと GPU へ送り直していた（5,490 万点・全点表示で、選ぶ・解除するたびに約 1.7〜2 秒）。

色の式
- 一部の点だけを作った色が、全点ぶん作った色のその行と、1 つも違わないこと（どの配色・陰影・レイヤーの固有色でも）

部分更新の結果
- 選択・解除・反転・レイヤーの移動・配色の切り替え・表示の切り替えを、どんな順でくり返しても、
  色の配列と表示用の色の配列が「全部を作り直した結果」と 1 つも違わないこと
- 小さな選択では、全部を作り直す処理を呼ばないこと。変わる点が多いときは、全部を作り直す道を通ること
- 何も選んでいないときの「選択を解除」は、色に触らないこと

守り
- 読み込みの直後（色の配列が、元の色の配列と同じものを指している）に選択しても、元の色を壊さないこと
- 表示が間引きや断面で一部になっていても、表示している点の色が正しく変わること

GPU への転送
- 変わった区画だけを送った絵が、まるごと送り直した絵と同じであること。選んだ点が黄色で描かれ、解除すると消えること

画面
- 連結塊・Del・即削除・元に戻す、のあとも、色が「全部を作り直した結果」と同じであること
"""
import os
import sys
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.settings_sandbox import use_sandbox_settings  # noqa: E402
from tests.qt_cleanup import destroy_widget  # noqa: E402

use_sandbox_settings()  # この PC の本番アプリの保存設定を、テストで書き換えない

import numpy as np  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from gui import color_patch as CP  # noqa: E402
from gui.gl_viewer import PointCloudViewer  # noqa: E402
from gui.noise_palette import SELECTION_COLOR  # noqa: E402

V = PointCloudViewer
ALL_MODES = [getattr(V, name) for name in dir(V) if name.startswith("COLOR_MODE_")]
YELLOW = np.asarray(SELECTION_COLOR, dtype=np.float32)


def make_cloud(n=30000, seed=0, with_colors=True, with_intensity=True):
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-8.0, 8.0, (n, 3))
    colors = rng.uniform(0.0, 1.0, (n, 3)).astype(np.float32) if with_colors else None
    intens = rng.uniform(0.0, 4000.0, n).astype(np.float32) if with_intensity else None
    scan_ids = rng.integers(0, 4, n).astype(np.int32)
    return pts, colors, intens, scan_ids


def reference_colors(v) -> np.ndarray:
    """
    答え合わせ用: 2026-11-14 に _base_colors へ分ける前の、_update_active_colors の中身をそのまま写したもの。
    全点と一部で同じ書き間違いをしても、これと比べれば分かる。配色を足したら、ここにも足すこと。
    """
    from gui.noise_palette import is_noise_mode

    n = len(v.points_centered)
    mode = v.color_mode
    if mode in (v.COLOR_MODE_RGB, "RGB", "RGB (実色写真)"):
        if v.colors_raw is not None:
            colors = v.colors_raw.copy()
        elif v.intensities_raw is not None:
            colors = v._colormap_grayscale(v._normalize_intensities())
        else:
            colors = v._colormap_turbo(v._normalize_heights())
    elif v._is_height_mode(mode):
        colors = v._height_colormap(v._normalize_heights(), mode)
    elif v._is_intensity_mode(mode):
        if v.intensities_raw is not None:
            colors = v._intensity_colormap(v._normalize_intensities(), mode)
        else:
            colors = v._colormap_turbo(v._normalize_heights())
    elif mode in (v.COLOR_MODE_SCAN, "スキャン別 (器械点)") and v.scan_ids is not None:
        colors = v.SCAN_PALETTE[np.mod(np.maximum(0, v.scan_ids), len(v.SCAN_PALETTE))].copy()
    elif mode in (v.COLOR_MODE_LAYER, "レイヤー別色分け") and v.layer_ids is not None:
        colors = np.tile([0.22, 0.74, 0.97], (n, 1)).astype(np.float32)
        for lid, linfo in v.layers.items():
            mask = (v.layer_ids == lid)
            if np.any(mask):
                colors[mask] = linfo.color
    else:
        colors = np.tile([0.3, 0.8, 0.95], (n, 1)).astype(np.float32)
    if getattr(v, "shading_enabled", False) and not is_noise_mode(mode):
        shade = v._compute_shading_factors()
        factor = 1.0 + (shade - 1.0) * float(getattr(v, "shading_strength", 0.65))
        colors = np.clip(colors * factor[:, np.newaxis], 0.0, 1.0).astype(np.float32)
    if v.layer_ids is not None and v.color_mode != v.COLOR_MODE_LAYER:
        for lid, linfo in v.layers.items():
            if lid == v.LAYER_NOISE or linfo.use_layer_color:
                mask = (v.layer_ids == lid)
                if np.any(mask):
                    colors[mask] = linfo.color
    if len(v.selected_indices) > 0:
        valid = v.selected_indices[v.selected_indices < n]
        colors[valid] = [1.0, 0.9, 0.1]
    return np.ascontiguousarray(colors, dtype=np.float32)


def expected_colors(v) -> np.ndarray:
    """全部を作り直したときの色（ビューアの状態は変えない）"""
    colors = np.ascontiguousarray(v._base_colors(), dtype=np.float32)
    sel = np.asarray(v.selected_indices, dtype=np.int64)
    if len(sel):
        colors[sel] = YELLOW
    return colors


class _ViewerCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def setUp(self):
        self.v = PointCloudViewer()
        self.v.resize(480, 360)
        self.v.show()
        self.app.processEvents()

    def tearDown(self):
        self.v.set_point_cloud(None)
        destroy_widget(self.v)

    def load(self, n=30000, seed=0, **kw):
        pts, colors, intens, scan_ids = make_cloud(n, seed, **kw)
        self.v.set_point_cloud(pts, colors, intens, scan_ids=scan_ids)
        self.app.processEvents()
        return pts

    def assert_same_as_full(self, where=""):
        v = self.v
        exp = expected_colors(v)
        self.assertEqual(v.active_colors.dtype, np.float32)
        np.testing.assert_array_equal(v.active_colors, exp, err_msg=f"色の配列が、全部を作り直した結果と違う: {where}")
        shown = v.display_orig_indices
        self.assertEqual(len(shown), len(v.display_colors), where)
        self.assertEqual(len(shown), len(v.display_points), where)
        np.testing.assert_array_equal(v.display_colors, exp[shown], err_msg=f"表示用の色が、全部を作り直した結果と違う: {where}")
        np.testing.assert_array_equal(np.sort(v._colored_selection), np.sort(np.asarray(v.selected_indices, dtype=np.int64)),
                                      err_msg=f"黄色に塗った点の覚えが、選択と違う: {where}")
        v.validate_invariants()


class TestBaseColors(_ViewerCase):
    def test_subset_equals_rows_of_the_whole(self):
        v = self.v
        rng = np.random.default_rng(1)
        for with_colors, with_intensity in ((True, True), (False, True), (False, False), (True, False)):
            self.load(n=12000, seed=3, with_colors=with_colors, with_intensity=with_intensity)
            n = len(v.points_raw)
            custom = v.add_custom_layer("足場", color=(0.1, 0.9, 0.3))
            plain = v.add_custom_layer("色を使わないレイヤー")
            v.layers[plain].use_layer_color = False
            v.layer_ids[rng.choice(n, 3000, replace=False)] = V.LAYER_NOISE
            v.layer_ids[rng.choice(n, 2000, replace=False)] = custom
            v.layer_ids[rng.choice(n, 1000, replace=False)] = plain
            v._sync_mask_from_ids()
            for shading in (True, False):
                v.shading_enabled = shading
                for mode in ALL_MODES:
                    v.color_mode = mode
                    whole = v._base_colors()
                    self.assertEqual(whole.shape, (n, 3), mode)
                    idx = rng.integers(0, n, 700)                      # 順不同・重複あり
                    part = v._base_colors(idx)
                    np.testing.assert_array_equal(np.asarray(part, dtype=np.float32), np.asarray(whole, dtype=np.float32)[idx],
                                                  err_msg=f"{mode} 陰影={shading} 色={with_colors} 強度={with_intensity}")
                    self.assertEqual(len(v._base_colors(np.empty(0, dtype=np.int64))), 0)
            v.color_mode = V.COLOR_MODE_RGB

    def test_full_rebuild_equals_the_previous_implementation(self):
        # 式を _base_colors へ分ける前の実装（reference_colors）と、全部の配色で突き合わせる
        v = self.v
        rng = np.random.default_rng(2)
        for with_colors, with_intensity in ((True, True), (False, True), (False, False)):
            self.load(n=9000, seed=5, with_colors=with_colors, with_intensity=with_intensity)
            n = len(v.points_raw)
            custom = v.add_custom_layer("足場", color=(0.1, 0.9, 0.3))
            v.layer_ids[rng.choice(n, 2500, replace=False)] = V.LAYER_NOISE
            v.layer_ids[rng.choice(n, 1500, replace=False)] = custom
            v._sync_mask_from_ids()
            v.selected_indices = np.unique(rng.integers(0, n, 600)).astype(np.int64)
            for shading in (True, False):
                v.shading_enabled = shading
                for mode in ALL_MODES:
                    v.color_mode = mode
                    v._update_active_colors()
                    np.testing.assert_array_equal(v.active_colors, reference_colors(v),
                                                  err_msg=f"{mode} 陰影={shading} 色={with_colors} 強度={with_intensity}")
                    np.testing.assert_array_equal(expected_colors(v), reference_colors(v), err_msg=mode)
            v.color_mode = V.COLOR_MODE_RGB
            v.selected_indices = np.empty(0, dtype=np.int64)

    def test_full_rebuild_is_unchanged_by_the_refactoring(self):
        # 全部を作り直した結果が、式をそのまま書いた答えと同じ（RGB + 陰影 + ノイズの赤 + 選択の黄色）
        v = self.v
        self.load(n=5000, seed=4)
        n = len(v.points_raw)
        v.layer_ids[:500] = V.LAYER_NOISE
        v._sync_mask_from_ids()
        v.selected_indices = np.arange(400, 900, dtype=np.int64)
        v._update_active_colors()
        shade = v._compute_shading_factors()
        factor = 1.0 + (shade - 1.0) * float(v.shading_strength)
        exp = np.clip(v.colors_raw * factor[:, None], 0.0, 1.0).astype(np.float32)
        exp[:500] = v.layers[V.LAYER_NOISE].color
        exp[400:900] = YELLOW
        np.testing.assert_array_equal(v.active_colors, exp)
        self.assertEqual(n, len(v.active_colors))


class TestIncrementalUpdates(_ViewerCase):
    def _random_ops(self, seed, n=20000, budget=0, steps=120):
        v = self.v
        rng = np.random.default_rng(seed)
        self.load(n=n, seed=seed)
        v.set_point_budget(budget)
        custom = v.add_custom_layer("足場", color=(0.1, 0.9, 0.3))
        self.assert_same_as_full("読み込み直後")

        def some(k_max=400):
            return np.unique(rng.integers(0, n, int(rng.integers(1, k_max))))

        from PySide6.QtGui import QColor

        for step in range(steps):
            kind = int(rng.integers(0, 23))
            if kind <= 2:
                mods = [Qt.NoModifier, Qt.ShiftModifier, Qt.ControlModifier][kind]
                v._apply_selection_with_modifiers(some(), "テスト", mods)
            elif kind == 3:
                v.clear_selection()
            elif kind == 4:
                v.move_indices_to_noise(some())
            elif kind == 5:
                v.move_indices_to_layer(some(), custom)
            elif kind == 6:
                v.restore_indices_to_normal(some(2000))
            elif kind == 7:
                v.batch_move_indices_to_layers([(some(), V.LAYER_NOISE), (some(), custom), (some(), V.LAYER_NORMAL)])
            elif kind == 8:
                v.set_color_mode(str(rng.choice(ALL_MODES)))
            elif kind == 9:
                v.set_shading_enabled(bool(rng.integers(0, 2)))
            elif kind == 10:
                v.set_layer_visibility(True, bool(rng.integers(0, 2)))      # ノイズレイヤーの表示 / 非表示
            elif kind == 11:
                half = float(rng.uniform(2.0, 8.0))
                v.set_section_x(bool(rng.integers(0, 2)), -half, half)      # 断面で表示を一部にする
            elif kind == 12:
                v.set_layer_color(custom, tuple(rng.uniform(0, 1, 3)), use_layer_color=bool(rng.integers(0, 2)))
            elif kind == 13:
                # 選択中の点をノイズへ移して、選択を解除する（Del の流れ）
                if len(v.selected_indices):
                    v.move_indices_to_noise(v.selected_indices)
                v.clear_selection()
            elif kind == 14:
                if len(v.selected_indices) * 4 < n:                         # 小さな選択の反転は、全体の 1/4 を超える
                    v.invert_selection()
            elif kind == 15:
                v.set_layer_locked(int(rng.choice([V.LAYER_NORMAL, V.LAYER_NOISE, custom])), bool(rng.integers(0, 2)))
            elif kind == 16:
                v.set_point_budget(int(rng.choice([0, 1500, 6000])))
            elif kind == 17:
                v.set_bg_color(QColor(*[int(c) for c in rng.choice([[255, 255, 255], [15, 23, 42], [0, 0, 0]])]))
            elif kind == 18:
                v.set_scan_visibility(int(rng.integers(0, 4)), bool(rng.integers(0, 2)))
            elif kind == 19:
                lo, hi = sorted(rng.uniform(-8.0, 8.0, 2))
                v.set_custom_height_range(bool(rng.integers(0, 2)), float(lo), float(hi))
            elif kind == 20:
                v.set_color_palette(str(rng.choice(["turbo", "viridis", "jet", "plasma", "terrain", "heat", "coolwarm", "grayscale"])))
            elif kind == 21:
                v.set_single_layer_visibility(int(rng.choice([V.LAYER_NORMAL, V.LAYER_NOISE, custom])), bool(rng.integers(0, 2)))
            else:
                # 高さの範囲の下書き表示（確定なし。スライダーをキーやホイールで動かしたときの形）。
                # 色の配列は古いままになるので、次に選んだときに、全部を作り直して追いつくこと
                lo, hi = sorted(rng.uniform(-8.0, 8.0, 2))
                v.preview_custom_height_range(True, float(lo), float(hi))
                v._apply_selection_with_modifiers(some(), "テスト", Qt.NoModifier)
            for lid in (V.LAYER_NORMAL, V.LAYER_NOISE, custom):           # 次の操作のために、ロックは外しておく
                v.layers[lid].locked = False
            self.assert_same_as_full(f"seed={seed} step={step} kind={kind}")
        return v

    def test_any_sequence_matches_a_full_rebuild_when_everything_is_shown(self):
        for seed in (11, 12, 13, 14):
            self._random_ops(seed, budget=0, steps=150)

    def test_any_sequence_matches_a_full_rebuild_when_the_display_is_thinned(self):
        # 表示点数の上限で間引かれている（表示しているのは、数点に 1 点）
        for seed in (21, 22):
            v = self._random_ops(seed, n=20000, budget=3000)
            self.assertLess(len(v.display_orig_indices), 20000)

    def test_small_selection_never_rebuilds_everything(self):
        v = self.v
        self.load(n=50000, seed=31)
        calls = {"full": 0, "display": 0, "base_rows": 0}
        orig_full, orig_disp, orig_base = v._update_active_colors, v._update_display_colors_only, v._base_colors

        def full():
            calls["full"] += 1
            return orig_full()

        def disp():
            calls["display"] += 1
            return orig_disp()

        def base(idx=None):
            calls["base_rows"] += len(v.points_raw) if idx is None else len(idx)
            return orig_base(idx)

        v._update_active_colors, v._update_display_colors_only, v._base_colors = full, disp, base
        try:
            v._apply_selection_with_modifiers(np.arange(100, 400, dtype=np.int64), "テスト", Qt.NoModifier)
            self.assertEqual(calls, {"full": 0, "display": 0, "base_rows": 0}, "選ぶだけなら、元の色を作り直す点は無い")
            v._apply_selection_with_modifiers(np.arange(300, 500, dtype=np.int64), "テスト", Qt.NoModifier)
            self.assertEqual(calls["full"], 0)
            self.assertEqual(calls["display"], 0)
            self.assertEqual(calls["base_rows"], 300, "元の色へ戻すのは、前に選んでいた 300 点だけ")
            v.clear_selection()
            self.assertEqual(calls["base_rows"], 500)
            v.clear_selection()                                   # 何も選んでいないときの解除（即削除モードで毎回通る）
            self.assertEqual(calls, {"full": 0, "display": 0, "base_rows": 500}, "何も選んでいなければ、色に触らない")
            v.move_indices_to_noise(np.arange(1000, 1200, dtype=np.int64))
            self.assertEqual(calls["full"], 0)
            self.assertEqual(calls["base_rows"], 700, "レイヤーを移した 200 点だけを作り直す")
        finally:
            v._update_active_colors, v._update_display_colors_only, v._base_colors = orig_full, orig_disp, orig_base
        self.assert_same_as_full()

    def test_large_change_takes_the_full_rebuild(self):
        v = self.v
        self.load(n=20000, seed=32)
        calls = {"full": 0}
        orig_full = v._update_active_colors

        def full():
            calls["full"] += 1
            return orig_full()

        v._update_active_colors = full
        try:
            v._apply_selection_with_modifiers(np.arange(0, 100, dtype=np.int64), "テスト", Qt.NoModifier)
            self.assertEqual(calls["full"], 0)
            v.invert_selection()                                   # 19,900 点（全体の 1/4 を超える）
            self.assertEqual(calls["full"], 1)
            self.assert_same_as_full("反転")
            v.clear_selection()
            self.assertEqual(calls["full"], 2)
            self.assert_same_as_full("解除")
            v.batch_move_indices_to_layers([(np.arange(0, 9000, dtype=np.int64), V.LAYER_NOISE)])
            self.assertEqual(calls["full"], 3, "全体の 1/4 を超える点を移したら、全部を作り直す")
        finally:
            v._update_active_colors = orig_full
        self.assert_same_as_full("一括移動")

    def test_stale_colors_after_a_preview_are_rebuilt_at_the_next_change(self):
        # 高さの色分けの範囲を、スライダーのキーやホイールで動かすと、下書き（preview_custom_height_range）だけが走り、
        # 色の配列は古い範囲のまま残る。以前は、次の選択で全部を作り直して直っていた。部分更新でも、同じように直ること
        v = self.v
        self.load(n=20000, seed=34)
        v.set_color_mode(V.COLOR_MODE_HEIGHT)
        v.set_custom_height_range(True, -8.0, 8.0)
        self.assert_same_as_full("確定した範囲")
        v.preview_custom_height_range(True, -2.0, 1.0)             # 確定なし
        self.assertFalse(v._colors_are_private(), "設定が変わったので、色の配列は古い")
        v._apply_selection_with_modifiers(np.arange(100, 300, dtype=np.int64), "テスト", Qt.NoModifier)
        self.assert_same_as_full("下書きのあとの選択")
        v.preview_custom_height_range(True, -5.0, 0.5)
        v.move_indices_to_noise(np.arange(100, 300, dtype=np.int64))  # Del の流れ: レイヤーを移して、選択を解除
        v.clear_selection()
        self.assert_same_as_full("下書きのあとの Del")
        # 設定を直接書き換えた場合も、次の更新で追いつく
        v.shading_strength = 0.2
        v._apply_selection_with_modifiers(np.arange(0, 50, dtype=np.int64), "テスト", Qt.NoModifier)
        self.assert_same_as_full("陰影の強さを直接変えた")
        v.layers[V.LAYER_NOISE].color = (0.1, 0.1, 0.9)
        v.clear_selection()
        self.assert_same_as_full("レイヤーの色を直接変えた")
        # ノイズ探索の配色は、背景の明暗で色の表が変わる
        from PySide6.QtGui import QColor
        v.set_color_mode(V.COLOR_MODE_INTENSITY_NOISE)
        v.bg_color = QColor(255, 255, 255)
        v._apply_selection_with_modifiers(np.arange(0, 50, dtype=np.int64), "テスト", Qt.NoModifier)
        self.assert_same_as_full("背景を直接、白に変えた")

    def test_selection_while_the_section_preview_is_shown(self):
        # 断面のスライダーを動かしている間の下書き表示では、表示用の配列だけが短くなっている
        v = self.v
        self.load(n=20000, seed=35)
        v.set_section_x(True, -3.0, 3.0, is_preview=True)
        self.assertNotEqual(len(v.display_points), len(v.display_orig_indices), "前提: 下書きでは、長さが合っていない")
        v._apply_selection_with_modifiers(np.arange(0, 400, dtype=np.int64), "テスト", Qt.NoModifier)
        self.assert_same_as_full("下書き表示中の選択")
        v.set_section_x(True, -2.0, 4.0, is_preview=True)
        v.clear_selection()
        self.assert_same_as_full("下書き表示中の解除")

    def test_out_of_range_selection_is_ignored(self):
        v = self.v
        self.load(n=3000, seed=33)
        v.selected_indices = np.array([5, 6, 7], dtype=np.int64)
        v._sync_selection_colors()
        np.testing.assert_array_equal(v.active_colors[[5, 6, 7]], np.tile(YELLOW, (3, 1)))
        v.selected_indices = np.empty(0, dtype=np.int64)
        v._sync_selection_colors()
        self.assert_same_as_full()
        # 点群が無いとき
        v.set_point_cloud(None)
        v._sync_selection_colors()
        v._recolor_indices(np.array([1, 2]))


class TestSharedArraysAfterLoading(_ViewerCase):
    def _preloaded(self, n=6000, seed=41, budget=0):
        """gui/worker_thread.py: FileLoadWorker と同じ形（色の配列を、写しを作らずに使い回す）"""
        pts, colors, intens, scan_ids = make_cloud(n, seed)
        center = (pts.min(axis=0) + pts.max(axis=0)) / 2.0
        centered = np.ascontiguousarray(pts - center, dtype=np.float32)
        if budget and n > budget:
            shown = np.arange(0, n, int(np.ceil(n / budget)), dtype=np.int64)
            d_pts, d_cols = np.ascontiguousarray(centered[shown]), np.ascontiguousarray(colors[shown])
        else:
            shown, d_pts, d_cols = np.arange(n, dtype=np.int64), centered, colors
        return types.SimpleNamespace(
            scans=None, points_raw=pts, points_centered=centered, center_offset=center, colors_raw=colors, intensities_raw=intens,
            active_colors=colors, world_bounds_min=pts.min(axis=0), world_bounds_max=pts.max(axis=0),
            bounds_min=(pts.min(axis=0) - center).astype(np.float32), bounds_max=(pts.max(axis=0) - center).astype(np.float32),
            display_points=d_pts, display_colors=d_cols, display_orig_indices=shown, scan_ids=scan_ids,
        )

    def test_first_selection_after_loading_keeps_the_raw_colors(self):
        for budget in (0, 1500):
            v = self.v
            pre = self._preloaded(budget=budget)
            raw_before = pre.colors_raw.copy()
            v.set_preloaded_data(pre)
            self.app.processEvents()
            self.assertIs(v.active_colors, v.colors_raw, "前提: 読み込みの直後は、同じ配列を指している")
            self.assertFalse(v._colors_are_private())
            v._apply_selection_with_modifiers(np.arange(10, 300, dtype=np.int64), "テスト", Qt.NoModifier)
            np.testing.assert_array_equal(v.colors_raw, raw_before, err_msg="元の色に、選択の黄色を書き込んではいけない")
            self.assertFalse(np.may_share_memory(v.active_colors, v.colors_raw))
            self.assertFalse(np.may_share_memory(v.display_colors, v.colors_raw))
            self.assert_same_as_full(f"最初の選択 budget={budget}")
            v.clear_selection()
            v._apply_selection_with_modifiers(np.arange(200, 250, dtype=np.int64), "テスト", Qt.NoModifier)
            v.move_indices_to_noise(np.arange(0, 100, dtype=np.int64))
            v.clear_selection()
            np.testing.assert_array_equal(v.colors_raw, raw_before)
            self.assert_same_as_full(f"そのあと budget={budget}")

    def test_cloud_without_colors_replacing_a_cloud_with_a_selection(self):
        # 色（RGB）の無いファイルの読み込みは、単色の自前の配列を渡してくる（色の式で作った色ではない）。
        # 前の点群の選択の覚えが残っていても、範囲外を触らず、最初の更新で全部を作り直すこと
        v = self.v
        self.load(n=25000, seed=43)
        v._apply_selection_with_modifiers(np.arange(20000, 25000, dtype=np.int64), "テスト", Qt.NoModifier)
        self.assertEqual(len(v._colored_selection), 5000)
        for n_new, first in ((8000, "select"), (25000, "move")):
            pre = self._preloaded(n=n_new, seed=44)
            pre.colors_raw = None
            pre.active_colors = np.tile(np.array([0.3, 0.8, 0.95], dtype=np.float32), (n_new, 1))
            pre.display_colors = pre.active_colors
            v.set_preloaded_data(pre)
            self.assertEqual(len(v._colored_selection), 0, "前の点群の覚えは捨てる")
            self.assertFalse(v._colors_are_private(), "受け取った色は、式で作ったものではない")
            if first == "select":
                v._apply_selection_with_modifiers(np.arange(10, 200, dtype=np.int64), "テスト", Qt.NoModifier)
            else:
                v.move_indices_to_noise(np.arange(10, 200, dtype=np.int64))
            self.assert_same_as_full(f"色なしの点群 {n_new} 点、最初の操作 {first}")
            v.clear_selection()
            self.assert_same_as_full("解除")
            v._apply_selection_with_modifiers(np.arange(n_new - 3000, n_new, dtype=np.int64), "テスト", Qt.NoModifier)
        # 点群を空にしても、覚えが残らない
        v.set_point_cloud(None)
        self.assertEqual(len(v._colored_selection), 0)
        self.assertIsNone(v._colors_stamp)

    def test_shared_array_is_never_written_even_if_the_settings_look_current(self):
        # 設定の控えが合っていても、色の配列が元の色と同じものを指していたら、書き込まずに作り直す（二重の守り）
        v = self.v
        self.load(n=5000, seed=45)
        raw_before = v.colors_raw.copy()
        v._update_active_colors()
        v.shading_enabled = False
        v._update_active_colors()
        v.active_colors = v.colors_raw                                   # ほかの経路が、写しを作らずに差し替えた形
        self.assertFalse(v._colors_are_private())
        v._apply_selection_with_modifiers(np.arange(10, 300, dtype=np.int64), "テスト", Qt.NoModifier)
        np.testing.assert_array_equal(v.colors_raw, raw_before)
        v.active_colors = v.colors_raw
        v.move_indices_to_noise(np.arange(500, 900, dtype=np.int64))
        np.testing.assert_array_equal(v.colors_raw, raw_before)
        self.assert_same_as_full()

    def test_first_layer_move_after_loading_keeps_the_raw_colors(self):
        v = self.v
        pre = self._preloaded()
        raw_before = pre.colors_raw.copy()
        v.set_preloaded_data(pre)
        v.move_indices_to_noise(np.arange(0, 500, dtype=np.int64))        # 読み込んで、すぐ自動ノイズ処理をした形
        np.testing.assert_array_equal(v.colors_raw, raw_before)
        self.assert_same_as_full()

    def test_display_colors_sharing_the_full_array_is_not_patched_in_place(self):
        # 表示用の色が、全点ぶんの色と同じ配列を指しているとき（読み込みの直後の形）は、書き換えずに写し直す
        v = self.v
        self.load(n=4000, seed=42)
        v.display_colors = v.active_colors
        v.selected_indices = np.arange(0, 50, dtype=np.int64)
        v._sync_selection_colors()
        self.assertFalse(np.may_share_memory(v.display_colors, v.active_colors))
        self.assert_same_as_full()


class TestGpuUpload(_ViewerCase):
    def _picture(self):
        img = self.v.grabFramebuffer()
        if img.isNull() or img.width() == 0:
            self.skipTest("OpenGL の画面を取り込めない環境")
        img = img.convertToFormat(img.Format.Format_RGBA8888)
        arr = np.frombuffer(img.constBits(), dtype=np.uint8).reshape(img.height(), img.bytesPerLine() // 4, 4)
        return arr[:, : img.width(), :3].copy()

    @staticmethod
    def _yellow_pixels(pic) -> int:
        return int(np.count_nonzero((pic[:, :, 0] > 200) & (pic[:, :, 1] > 170) & (pic[:, :, 2] < 110)))

    def test_partial_upload_draws_the_same_picture_as_a_full_upload(self):
        v = self.v
        n = 400_000                                               # 区画（65,536 点）が 7 つ
        rng = np.random.default_rng(51)
        pts = rng.uniform(-5.0, 5.0, (n, 3))
        colors = np.tile(np.array([0.05, 0.15, 0.5], dtype=np.float32), (n, 1))     # 全部、暗い青
        v.set_shading_enabled(False)
        v.set_point_budget(0)
        v.set_point_cloud(pts, colors, None)
        self.app.processEvents()
        first = self._picture()
        self.assertEqual(self._yellow_pixels(first), 0)
        self.assertFalse(v._vbo_colors_dirty, "前提: 1 回描いたあとは、送り直す予定が無い")

        sent = []
        orig_sub, orig_data = CP.gl.glBufferSubData, CP.gl.glBufferData

        def sub(target, offset, size, data):
            sent.append(("部分", int(size)))
            return orig_sub(target, offset, size, data)

        def data(target, size, payload, usage):
            sent.append(("全部", int(size)))
            return orig_data(target, size, payload, usage)

        CP.gl.glBufferSubData, CP.gl.glBufferData = sub, data
        try:
            # 1 つ目と 4 つ目の区画の点を選ぶ
            sel = np.r_[np.arange(1000, 9000), np.arange(3 * CP.PATCH_BLOCK + 500, 3 * CP.PATCH_BLOCK + 9000)].astype(np.int64)
            v._apply_selection_with_modifiers(sel, "テスト", Qt.NoModifier)
            self.assertFalse(v._vbo_colors_dirty, "色の VBO を、まるごと送り直す予定にしない")
            self.assertIsNotNone(v._color_patch)
            np.testing.assert_array_equal(v._color_patch[1], [0, 3])
            partial = self._picture()
            self.assertIsNone(v._color_patch, "描いたら、覚えていた区画は消える")
            self.assertEqual([kind for kind, _ in sent], ["部分", "部分"], "離れた 2 区画を、2 回に分けて送る")
            self.assertEqual({size for _, size in sent}, {CP.PATCH_BLOCK * 12})
            self.assertGreater(self._yellow_pixels(partial), 50, "選んだ点が、黄色で描かれている")

            sent.clear()
            v._vbo_dirty = True                                       # まるごと送り直させる
            full = self._picture()
            np.testing.assert_array_equal(partial, full, err_msg="部分だけ送った絵が、まるごと送った絵と違う")

            # 解除: 黄色が消えて、最初の絵に戻る
            sent.clear()
            v.clear_selection()
            cleared = self._picture()
            self.assertEqual([kind for kind, _ in sent], ["部分", "部分"])
            self.assertEqual(self._yellow_pixels(cleared), 0)
            np.testing.assert_array_equal(cleared, first)

            # 隣り合う区画は、1 回にまとめて送る
            sent.clear()
            v._apply_selection_with_modifiers(np.arange(CP.PATCH_BLOCK - 100, 2 * CP.PATCH_BLOCK + 100, dtype=np.int64), "テスト", Qt.NoModifier)
            self._picture()
            self.assertEqual(sent, [("部分", 3 * CP.PATCH_BLOCK * 12)])

            # 描く前に選択を 2 回変えたら、両方の区画を覚えていて、1 回の描画でまとめて送る
            v.clear_selection()
            self._picture()
            sent.clear()
            v._apply_selection_with_modifiers(np.arange(100, 300, dtype=np.int64), "テスト", Qt.NoModifier)
            v._apply_selection_with_modifiers(np.arange(4 * CP.PATCH_BLOCK + 10, 4 * CP.PATCH_BLOCK + 500, dtype=np.int64), "テスト", Qt.NoModifier)
            np.testing.assert_array_equal(v._color_patch[1], [0, 4])
            twice = self._picture()
            self.assertEqual(len(sent), 2)
            v._vbo_dirty = True
            np.testing.assert_array_equal(twice, self._picture(), err_msg="1 回目の選択の区画（黄色を消す側）を送り忘れている")

            # 描いたあとで解除し、描く前に別の区画を選ぶ: 解除の区画（黄色を消す）と、選択の区画の両方を送る
            self._picture()
            v.clear_selection()                                           # 区画 4 を覚える（まだ描かない）
            np.testing.assert_array_equal(v._color_patch[1], [4])
            v._apply_selection_with_modifiers(np.arange(2 * CP.PATCH_BLOCK + 10, 2 * CP.PATCH_BLOCK + 500, dtype=np.int64), "テスト", Qt.NoModifier)
            np.testing.assert_array_equal(v._color_patch[1], [2, 4], err_msg="前に覚えた区画を、忘れてはいけない")
            merged = self._picture()
            v._vbo_dirty = True
            np.testing.assert_array_equal(merged, self._picture(), err_msg="解除した点の黄色が、GPU に残っている")

            # 大半の区画が変わるときは、まるごと送る
            sent.clear()
            v._apply_selection_with_modifiers(np.arange(0, n, 5000, dtype=np.int64), "テスト", Qt.NoModifier)
            self.assertTrue(v._vbo_colors_dirty)
            self.assertIsNone(v._color_patch)
        finally:
            CP.gl.glBufferSubData, CP.gl.glBufferData = orig_sub, orig_data

    def test_last_block_and_changed_display_are_safe(self):
        v = self.v
        n = 350_000                                               # 区画は 6 つ。最後の区画は、半端な長さ（22,320 点）
        rng = np.random.default_rng(52)
        v.set_shading_enabled(False)
        v.set_point_budget(0)
        v.set_point_cloud(rng.uniform(-5.0, 5.0, (n, 3)), np.tile(np.array([0.05, 0.15, 0.5], dtype=np.float32), (n, 1)), None)
        self.app.processEvents()
        self._picture()
        v._apply_selection_with_modifiers(np.arange(n - 300, n, dtype=np.int64), "テスト", Qt.NoModifier)
        np.testing.assert_array_equal(v._color_patch[1], [5])
        partial = self._picture()
        v._vbo_dirty = True
        np.testing.assert_array_equal(partial, self._picture())
        # 区画を覚えたあとで表示が入れ替わったら（点数が変わる）、覚えていた区画は使わずに、まるごと送る
        v._apply_selection_with_modifiers(np.arange(0, 200, dtype=np.int64), "テスト", Qt.NoModifier)
        self.assertIsNotNone(v._color_patch)
        v.set_point_budget(200_000)                                         # 間引くので、表示用の配列が作り直される
        self.assertLess(len(v.display_points), n)
        self.assertTrue(v._vbo_colors_dirty)
        after = self._picture()
        self.assertIsNone(v._color_patch)
        v._vbo_dirty = True
        np.testing.assert_array_equal(after, self._picture())


class TestMainWindowFlows(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from gui.main_window import MainWindow

        cls.app = QApplication.instance() or QApplication(sys.argv)
        cls.win = MainWindow()
        cls.win._reset_app_settings()
        cls.win.resize(1450, 900)
        cls.win.show()
        cls.app.processEvents()

    @classmethod
    def tearDownClass(cls):
        cls.win._set_dirty(False)
        cls.win.viewer.set_point_cloud(None)
        destroy_widget(cls.win)

    def setUp(self):
        win = self.win
        win._toggle_instant_delete_mode(False)
        pts, colors, intens, scan_ids = make_cloud(25000, seed=61)
        win.viewer.set_point_cloud(pts, colors, intens, scan_ids=scan_ids)
        self.app.processEvents()

    def tearDown(self):
        self.win._toggle_instant_delete_mode(False)

    def check(self, where):
        v = self.win.viewer
        exp = expected_colors(v)
        np.testing.assert_array_equal(v.active_colors, exp, err_msg=where)
        np.testing.assert_array_equal(v.display_colors, exp[v.display_orig_indices], err_msg=where)
        v.validate_invariants()

    def test_select_delete_undo_redo(self):
        win, v = self.win, self.win.viewer
        v._apply_selection_with_modifiers(np.arange(100, 900, dtype=np.int64), "テスト", Qt.NoModifier)
        self.check("選択")
        win._delete_selected_points()
        self.check("Del")
        self.assertEqual(len(v.selected_indices), 0)
        np.testing.assert_array_equal(v.active_colors[100:900], np.tile(np.asarray(v.layers[V.LAYER_NOISE].color, dtype=np.float32), (800, 1)),
                                      err_msg="ノイズへ移した点は、ノイズレイヤーの色になる")
        win._undo_action()
        self.check("元に戻す")
        win._redo_action()
        self.check("やり直し")
        win._undo_action()
        v._apply_selection_with_modifiers(np.arange(500, 1500, dtype=np.int64), "テスト", Qt.NoModifier)
        win._clear_selection()
        self.check("選択して解除")

    def test_undo_recolors_only_the_restored_points(self):
        win, v = self.win, self.win.viewer
        v._apply_selection_with_modifiers(np.arange(100, 900, dtype=np.int64), "テスト", Qt.NoModifier)
        win._delete_selected_points()
        calls = {"full": 0}
        orig_full = v._update_active_colors

        def full():
            calls["full"] += 1
            return orig_full()

        v._update_active_colors = full
        try:
            win._undo_action()
            self.assertEqual(calls["full"], 0, "点を戻すだけなら、全部を作り直さない")
            self.check("元に戻す")
            win._redo_action()
            self.assertEqual(calls["full"], 0)
            self.check("やり直し")
            # レイヤーの定義（新しいレイヤー・色）が変わる取り消しは、全部を作り直す
            win._push_layer_undo_state()
            lid = v.add_custom_layer("あとで消えるレイヤー", color=(0.9, 0.1, 0.9))
            v.move_indices_to_layer(np.arange(3000, 3300, dtype=np.int64), lid)
            self.check("新しいレイヤーへ移す")
            before = calls["full"]
            win._undo_action()
            self.assertGreater(calls["full"], before, "レイヤーの定義が変わったら、全部を作り直す")
            self.assertNotIn(lid, v.layers)
            self.check("レイヤーごと元に戻す")
        finally:
            v._update_active_colors = orig_full

    def test_instant_erase(self):
        win, v = self.win, self.win.viewer
        win._toggle_instant_delete_mode(True)
        v._apply_selection_with_modifiers(np.arange(2000, 2600, dtype=np.int64), "テスト", Qt.NoModifier)   # 即削除: そのままノイズへ
        self.check("即削除")
        self.assertEqual(len(v.selected_indices), 0)
        self.assertEqual(int(np.count_nonzero(v.layer_ids == V.LAYER_NOISE)), 600)
        win._undo_action()
        self.check("即削除を元に戻す")
        self.assertEqual(int(np.count_nonzero(v.layer_ids == V.LAYER_NOISE)), 0)

    def test_wand_selection(self):
        win, v = self.win, self.win.viewer
        v.selected_indices = np.arange(10, 14, dtype=np.int64)
        v._sync_selection_colors()
        win.cluster_eps_spin.setValue(0.5)
        win._on_magic_wand_cluster_select()
        self.assertTrue(win._wait_for_background())
        for _ in range(5):
            self.app.processEvents()
        self.check("連結塊")
        if len(v.selected_indices):
            np.testing.assert_array_equal(v.active_colors[v.selected_indices], np.tile(YELLOW, (len(v.selected_indices), 1)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
