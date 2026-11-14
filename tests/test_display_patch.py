# -*- coding: utf-8 -*-
"""
表示の部分更新（gui/display_patch.py）のテスト

きっかけ（2026-11-14）: 点を 1 つ消す・戻すだけでも、表示用の配列（点・色・番号）を全点ぶん作り直し、
点と色の VBO をまるごと GPU へ送り直していた（5,490 万点・全点表示で、消すたびに約 1.6 秒）。

全部を作り直す道
- np.take に替えたあとも、作る配列が、替える前の書き方（[番号の配列] での取り出し）と 1 つも違わないこと
- 一部の点だけについて作ったマスクが、全点のマスクのその行と同じこと（断面・レイヤーの表示・ロック・スキャンの表示）

部分更新の結果
- 消す・戻す・別のレイヤーへ移す・選ぶ、をどんな順でくり返しても、表示している点の集まりと座標・色が、
  全部を作り直した結果と同じこと。番号は昇順のまま、重複の枠の数が合っていること
- 消してすぐ戻すと、配列が、全部を作り直したものと 1 つも違わないこと
- 小さな変更では、全部を作り直す処理を呼ばないこと

守り（全部を作り直す道へ戻る場面）
- 読み込みの直後（表示用の配列が、元の配列そのもの）・表示点数の上限で間引いているとき・下書き表示の途中・
  表示の設定が変わっているとき・先頭の点が消えるとき・重複の枠が増えすぎたとき・戻す枠が足りないとき

GPU への転送
- 変わった区画だけを送った絵が、まるごと送り直した絵と同じであること（消したあと・戻したあと）

画面
- Del・即削除・元に戻す・やり直しのあとも、表示が「全部を作り直した結果」と同じであること
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
from gui import display_patch as DP  # noqa: E402
from gui.gl_viewer import PointCloudViewer  # noqa: E402
from tests.test_color_patch import expected_colors, make_cloud  # noqa: E402

V = PointCloudViewer


def reference_display(v):
    """
    答え合わせ用: 2026-11-14 に部分更新を入れる前の、_update_display_arrays の中身をそのまま写したもの。
    戻り値: (番号, 点, 色)。ビューアの状態は変えない。
    """
    import math

    in_sec = v._get_section_inlier_mask()
    layer_scan_vis = v._get_layer_and_scan_visibility_mask()
    surviving = np.where(in_sec & layer_scan_vis)[0]
    n_surv = len(surviving)
    budget = v.point_budget
    section_active = v.is_section_active()
    if section_active and v.section_full_detail and budget > 0:
        budget = max(budget, int(v.SECTION_DETAIL_MAX_POINTS))
    if budget <= 0 or n_surv <= budget:
        sampled = surviving
    else:
        step = max(1, int(math.ceil(n_surv / budget)))
        sampled = surviving[::step]
    colors = expected_colors(v)
    return sampled, np.ascontiguousarray(v.points_centered[sampled]), np.ascontiguousarray(colors[sampled])


class _ViewerCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def setUp(self):
        self.v = PointCloudViewer()
        self.v.resize(480, 360)
        self.v.show()
        self.app.processEvents()
        self.full_calls = 0
        orig = self.v._update_display_arrays

        def counted():
            self.full_calls += 1
            return orig()

        self.v._update_display_arrays = counted

    def tearDown(self):
        self.v.set_point_cloud(None)
        destroy_widget(self.v)

    def load(self, n=30000, seed=0, **kw):
        pts, colors, intens, scan_ids = make_cloud(n, seed, **kw)
        self.v.set_point_cloud(pts, colors, intens, scan_ids=scan_ids)
        self.app.processEvents()
        self.full_calls = 0
        return pts

    def assert_display_ok(self, where="", exact=False):
        """表示が、全部を作り直した結果と同じ（exact=True なら、配列そのものが同じ）"""
        v = self.v
        ref_idx, ref_pts, ref_cols = reference_display(v)
        shown = v.display_orig_indices
        self.assertEqual(len(shown), len(v.display_points), where)
        self.assertEqual(len(shown), len(v.display_colors), where)
        self.assertEqual(shown.dtype, np.int64, where)
        if len(shown) > 1:
            self.assertTrue(np.all(shown[1:] >= shown[:-1]), f"番号が昇順でない: {where}")
        np.testing.assert_array_equal(np.unique(shown), ref_idx, err_msg=f"表示している点の集まりが違う: {where}")
        self.assertEqual(len(shown) - len(ref_idx), v._display_dups, f"重複の枠の数が合わない: {where}")
        self.assertEqual(v.shown_point_count(), len(ref_idx), where)
        np.testing.assert_array_equal(v.display_points, v.points_centered[shown], err_msg=f"点の座標が違う: {where}")
        np.testing.assert_array_equal(v.display_colors, expected_colors(v)[shown], err_msg=f"色が違う: {where}")
        self.assertIs(v._preview_base_points, v.display_points, where)
        self.assertIs(v._preview_base_colors, v.display_colors, where)
        self.assertEqual(v.section_box_count, len(ref_idx) if v.is_section_active() else 0, where)
        if exact:
            np.testing.assert_array_equal(shown, ref_idx, err_msg=where)
            np.testing.assert_array_equal(v.display_points, ref_pts, err_msg=where)
            np.testing.assert_array_equal(v.display_colors, ref_cols, err_msg=where)
        v.validate_invariants()


class TestMasksForSubsets(_ViewerCase):
    def test_subset_masks_equal_rows_of_the_full_masks(self):
        v = self.v
        self.load(n=20000, seed=3)
        rng = np.random.default_rng(4)
        lid_a = v.add_custom_layer("表示するレイヤー", color=(0.1, 0.9, 0.2))
        lid_b = v.add_custom_layer("隠すレイヤー", color=(0.9, 0.2, 0.9))
        v.move_indices_to_layer(rng.choice(20000, 3000, replace=False), lid_a)
        v.move_indices_to_layer(rng.choice(20000, 3000, replace=False), lid_b)
        v.move_indices_to_noise(rng.choice(20000, 2500, replace=False))
        idx = np.sort(rng.choice(20000, 5000, replace=False)).astype(np.int64)
        unsorted = rng.permutation(idx)

        def check(where):
            for name in ("_get_section_inlier_mask", "_get_layer_and_scan_visibility_mask", "_get_unlocked_mask", "_get_selectable_mask"):
                full = getattr(v, name)()
                self.assertEqual(full.shape, (20000,), f"{name}: {where}")
                np.testing.assert_array_equal(getattr(v, name)(idx), full[idx], err_msg=f"{name}: {where}")
                np.testing.assert_array_equal(getattr(v, name)(unsorted), full[unsorted], err_msg=f"{name}（順不同）: {where}")
                self.assertEqual(getattr(v, name)(np.empty(0, dtype=np.int64)).shape, (0,), f"{name}（空）: {where}")

        check("最初")
        v.set_single_layer_visibility(lid_b, False)
        check("レイヤーを 1 つ隠す")
        v.set_layer_locked(lid_a, True)
        check("レイヤーを 1 つロック")
        v.set_scan_visibility(2, False)
        check("スキャンを 1 つ隠す")
        v.set_section_x(True, -3.0, 4.0)
        check("X 断面")
        v.set_section_z(True, -6.0, 1.5)
        check("X と Z の断面")
        v.set_section_y(True, -2.0, 2.0)
        check("X・Y・Z の断面")
        v.set_wall_align_points(np.array([-4.0, -1.0, 0.0]), np.array([5.0, 3.0, 0.0]))
        v.set_section_wall(True, -1.5, 2.0)
        check("斜めの壁の断面")
        v.set_layer_visibility(True, True)
        check("ノイズも表示")


class TestFullRebuild(_ViewerCase):
    def test_rebuilt_arrays_equal_the_previous_implementation(self):
        v = self.v
        self.load(n=30000, seed=5)
        v.move_indices_to_noise(np.arange(100, 4000, 3, dtype=np.int64))

        def check(where):
            v._update_display_arrays()
            ref_idx, ref_pts, ref_cols = reference_display(v)
            np.testing.assert_array_equal(v.display_orig_indices, ref_idx, err_msg=where)
            np.testing.assert_array_equal(v.display_points, ref_pts, err_msg=where)
            np.testing.assert_array_equal(v.display_colors, ref_cols, err_msg=where)
            for arr in (v.display_points, v.display_colors):
                self.assertEqual(arr.dtype, np.float32, where)
                self.assertTrue(arr.flags.c_contiguous and arr.flags.writeable, where)
            self.assertEqual(v._display_dups, 0, where)

        check("全点表示")
        v.set_point_budget(7000)
        check("間引き")
        self.assertIsNone(v._display_stamp, "間引いているときは、部分更新してよい印を付けない")
        v.set_section_z(True, -2.0, 2.0)
        check("断面")
        v.set_point_budget(0)
        check("断面・上限なし")
        self.assertIsNotNone(v._display_stamp)
        v.set_layer_visibility(False, True)
        check("ノイズだけ")

    def test_colors_only_update_also_replaces_the_preview_base(self):
        # 配色を変えたあと、断面のスライダーを動かしても、前の配色で描かれないこと（QA_REPORT backlog 22 (b)）
        v = self.v
        self.load(n=8000, seed=6)
        v.set_color_mode(V.COLOR_MODE_HEIGHT)
        self.assertIs(v._preview_base_colors, v.display_colors)
        v.set_section_z(True, -2.0, 2.0, is_preview=True)
        inside = (v.points_centered[:, 2] + v.center_offset[2] >= -2.0) & (v.points_centered[:, 2] + v.center_offset[2] <= 2.0)
        np.testing.assert_array_equal(v.display_colors, expected_colors(v)[inside])


class TestIncrementalUpdates(_ViewerCase):
    def test_delete_and_restore_do_not_rebuild_everything(self):
        v = self.v
        self.load(n=30000, seed=7)
        before = (v.display_orig_indices.copy(), v.display_points.copy(), v.display_colors.copy())
        pc_before = v.points_centered.copy()
        gone = np.r_[np.arange(500, 900), np.arange(12000, 12050), [29999]].astype(np.int64)
        v.move_indices_to_noise(gone)
        self.assertEqual(self.full_calls, 0, "消すだけなら、全部を作り直さない")
        self.assertEqual(v._display_dups, len(gone))
        self.assertEqual(len(v.display_points), 30000, "配列の長さは変わらない")
        self.assertEqual(v.shown_point_count(), 30000 - len(gone))
        self.assert_display_ok("消したあと")
        # 消えた点の枠には、すぐ手前の消えていない点が入る
        np.testing.assert_array_equal(v.display_orig_indices[500:900], np.full(400, 499))
        self.assertEqual(int(v.display_orig_indices[29999]), 29998)

        v.restore_indices_to_normal(gone)
        self.assertEqual(self.full_calls, 0, "戻すだけなら、全部を作り直さない")
        self.assertEqual(v._display_dups, 0)
        self.assert_display_ok("戻したあと", exact=True)
        for now, was in zip((v.display_orig_indices, v.display_points, v.display_colors), before):
            np.testing.assert_array_equal(now, was, err_msg="消してすぐ戻したら、元の配列と同じ")
        np.testing.assert_array_equal(v.points_centered, pc_before, err_msg="元の点の配列を書き換えてはいけない")

    def test_partial_restore_and_repeated_deletes(self):
        v = self.v
        self.load(n=20000, seed=8)
        v.move_indices_to_noise(np.arange(1000, 1400, dtype=np.int64))
        v.move_indices_to_noise(np.arange(900, 1000, dtype=np.int64))         # 重複の枠のすぐ手前を消す
        self.assert_display_ok("続けて消す")
        np.testing.assert_array_equal(v.display_orig_indices[900:1400], np.full(500, 899))
        v.move_indices_to_noise(np.array([899], dtype=np.int64))              # 重複の枠の元になっている点を消す
        self.assert_display_ok("写しの元を消す")
        np.testing.assert_array_equal(v.display_orig_indices[899:1400], np.full(501, 898))
        v.restore_indices_to_normal(np.array([950, 1200, 1201, 1399], dtype=np.int64))   # 一部だけ戻す
        self.assertEqual(self.full_calls, 0)
        self.assert_display_ok("一部だけ戻す")
        # すき間の中では、消える前と同じ枠に戻る（あとから戻る点の枠が残る）
        np.testing.assert_array_equal(v.display_orig_indices[[949, 950, 951, 1199, 1200, 1201, 1202, 1398, 1399, 1400]],
                                      [898, 950, 950, 950, 1200, 1201, 1201, 1201, 1399, 1400])
        v.restore_indices_to_normal(np.arange(899, 1400, dtype=np.int64))     # 残りも戻す（もう戻っている点が混じる）
        self.assertEqual(self.full_calls, 0)
        self.assert_display_ok("残りも戻す", exact=True)

    def test_undo_order_restores_adjacent_deletes_without_a_rebuild(self):
        # 隣り合う所を続けて消し、あとで消したほうから順に戻す（「元に戻す」をくり返した形）
        v = self.v
        self.load(n=20000, seed=14)
        steps = [np.arange(1000, 1400), np.arange(900, 1000), np.arange(1400, 1500), np.r_[np.arange(880, 900), np.arange(1500, 1520)],
                 np.arange(5000, 5001), np.arange(19990, 20000)]
        steps = [s.astype(np.int64) for s in steps]
        for i, idx in enumerate(steps):
            v.move_indices_to_noise(idx)
            self.assert_display_ok(f"{i} 回目に消す")
        for i, idx in reversed(list(enumerate(steps))):
            v.restore_indices_to_normal(idx)
            self.assert_display_ok(f"{i} 回目に消したものを戻す")
        self.assertEqual(self.full_calls, 0)
        self.assert_display_ok("すべて戻した", exact=True)
        # 消した順に戻しても同じ
        for idx in steps:
            v.move_indices_to_noise(idx)
        for i, idx in enumerate(steps):
            v.restore_indices_to_normal(idx)
            self.assert_display_ok(f"消した順に戻す {i}")
        self.assertEqual(self.full_calls, 0)
        self.assert_display_ok("すべて戻した（消した順）", exact=True)

    def test_restore_into_a_compacted_display_spreads_by_index(self):
        # 表示を詰め直したあと（番号と枠の位置がずれている）でも、番号の順を保って戻せること
        v = self.v
        self.load(n=20000, seed=15)
        v.move_indices_to_noise(np.arange(100, 3000, 2, dtype=np.int64))      # 1 つおきに消して
        v._update_display_arrays()                                            # 詰め直す
        self.full_calls = 0
        a = np.arange(4001, 4400, 3, dtype=np.int64)
        b = np.arange(4000, 4400, 3, dtype=np.int64)
        c = np.arange(4002, 4400, 3, dtype=np.int64)
        for idx in (a, b, c):
            v.move_indices_to_noise(idx)
        for k, idx in enumerate((b, c, a)):
            v.restore_indices_to_normal(idx)
            self.assert_display_ok(f"{k} 組目を戻す")
        self.assertEqual(self.full_calls, 0)
        self.assert_display_ok("すべて戻した", exact=True)

    def test_points_leaving_and_entering_the_same_gap_in_one_call(self):
        # 1 回の一括の移動で、同じすき間の中の点が、消える・戻る・色だけ変わる
        v = self.v
        self.load(n=20000, seed=16)
        lid = v.add_custom_layer("表示するレイヤー", color=(0.1, 0.9, 0.2))
        v.move_indices_to_noise(np.arange(1000, 1100, dtype=np.int64))        # ここで全部を作り直す（レイヤーを足した直後）
        v._update_display_arrays()
        v.move_indices_to_noise(np.arange(1150, 1250, dtype=np.int64))        # 重複の枠を作る（部分更新）
        self.full_calls = 0

        def span(a, b):
            return np.arange(a, b, dtype=np.int64)

        v.batch_move_indices_to_layers([
            (span(1150, 1200), V.LAYER_NORMAL),      # 戻る
            (span(1100, 1150), V.LAYER_NOISE),       # すぐ隣が消える
            (span(1250, 1300), lid),                 # 色だけ変わる
            (span(1200, 1210), lid),                 # 消えていた点が、別の表示レイヤーへ戻る
        ])
        self.assertEqual(self.full_calls, 0)
        self.assert_display_ok("消える・戻る・色だけ、が混じる")
        v.batch_move_indices_to_layers([
            (span(1100, 1150), V.LAYER_NORMAL), (span(1150, 1200), V.LAYER_NOISE),
            (span(1200, 1300), V.LAYER_NORMAL), (span(1300, 1320), V.LAYER_NOISE),
        ])
        # ここから先は、全部を作り直すことがある（詰め直した表示には、前から隠れていた点を戻す枠が無い）。結果は同じ
        self.assert_display_ok("逆向きに混ぜる")
        v.restore_indices_to_normal(span(1150, 1320))
        self.assert_display_ok("残りを戻す", exact=True)

    def test_unsorted_and_repeated_indices(self):
        v = self.v
        self.load(n=10000, seed=9)
        rng = np.random.default_rng(10)
        idx = rng.permutation(np.r_[np.arange(2000, 2300), np.arange(2000, 2100), [7000, 7000]]).astype(np.int64)
        v.batch_move_indices_to_layers([(idx[:200], V.LAYER_NOISE), (idx[200:], V.LAYER_NOISE)])
        self.assertEqual(self.full_calls, 0)
        self.assertEqual(v._display_dups, 301)
        self.assert_display_ok("順不同・同じ番号が 2 回")

    def test_selection_colors_reach_every_duplicate_slot(self):
        v = self.v
        self.load(n=10000, seed=11)
        v.move_indices_to_noise(np.arange(3000, 3600, dtype=np.int64))        # 2999 の写しが 600 枠
        v._apply_selection_with_modifiers(np.arange(2990, 3000, dtype=np.int64), "テスト", Qt.NoModifier)
        self.assert_display_ok("写しの元を選ぶ")
        np.testing.assert_array_equal(v.display_colors[2999:3600], np.tile(np.asarray(CP.SELECTION_COLOR, dtype=np.float32), (601, 1)),
                                      err_msg="同じ点の枠は、すべて黄色になる")
        v.clear_selection()
        self.assert_display_ok("解除")
        v.set_color_mode(V.COLOR_MODE_HEIGHT)                                 # 色だけを全部写し直す道
        self.assert_display_ok("配色を変える")
        v.move_indices_to_noise(np.arange(100, 120, dtype=np.int64))
        self.assert_display_ok("配色を変えたあとで消す")

    def test_moves_between_visible_layers_only_change_colors(self):
        v = self.v
        self.load(n=10000, seed=12)
        lid = v.add_custom_layer("表示するレイヤー", color=(0.1, 0.9, 0.2))
        v.move_indices_to_layer(np.arange(9000, 9010, dtype=np.int64), lid)
        self.assertEqual(self.full_calls, 1, "レイヤーを足した直後の 1 回は、色も表示も全部を作り直す（色の式が読む設定が変わった）")
        self.full_calls = 0
        v.move_indices_to_layer(np.arange(400, 900, dtype=np.int64), lid)
        self.assertEqual(self.full_calls, 0)
        self.assertEqual(v._display_dups, 0, "表示に残る点は、枠を動かさない")
        self.assert_display_ok("表示しているレイヤーへ移す", exact=True)
        np.testing.assert_array_equal(v.display_colors[400:900], np.tile(np.asarray((0.1, 0.9, 0.2), dtype=np.float32), (500, 1)))
        v.move_indices_to_noise(np.arange(600, 700, dtype=np.int64))
        v.move_indices_to_layer(np.arange(650, 1000, dtype=np.int64), V.LAYER_NORMAL)   # 消えた点と残っている点が混じる
        self.assertEqual(self.full_calls, 0)
        self.assert_display_ok("混ぜて移す")

    def test_noise_layer_visible_keeps_the_points_on_screen(self):
        v = self.v
        self.load(n=10000, seed=13)
        v.set_layer_visibility(True, True)
        self.full_calls = 0
        v.move_indices_to_noise(np.arange(0, 500, dtype=np.int64))           # 先頭の点も含む
        self.assertEqual(self.full_calls, 0)
        self.assertEqual(v._display_dups, 0)
        self.assert_display_ok("ノイズを表示したまま消す", exact=True)

    def test_random_operations_match_a_full_rebuild(self):
        rng = np.random.default_rng(2026)
        for trial in range(4):
            v = self.v
            n = 12000
            self.load(n=n, seed=20 + trial)
            lid_show = v.add_custom_layer("表示するレイヤー", color=(0.1, 0.9, 0.2))
            lid_hide = v.add_custom_layer("隠すレイヤー", color=(0.9, 0.2, 0.9))
            v.set_single_layer_visibility(lid_hide, False)
            if trial == 1:
                v.set_section_z(True, -5.0, 3.0)
            if trial == 2:
                v.set_layer_visibility(True, True)
                v.set_single_layer_visibility(lid_hide, False)
            if trial == 3:
                v.set_scan_visibility(1, False)
            targets = [V.LAYER_NOISE, V.LAYER_NORMAL, lid_show, lid_hide]
            for step in range(60):
                kind = int(rng.integers(0, 10))
                start = int(rng.integers(1, n - 600))
                if kind <= 4:
                    size = int(rng.integers(1, 500))
                    idx = np.arange(start, start + size, dtype=np.int64)
                    if rng.random() < 0.4:
                        idx = np.unique(rng.integers(1, n, size)).astype(np.int64)
                    v.move_indices_to_layer(idx, targets[int(rng.integers(0, len(targets)))])
                elif kind == 5:
                    hidden = np.flatnonzero(np.isin(v.layer_ids, [V.LAYER_NOISE, lid_hide]))
                    if len(hidden):
                        v.restore_indices_to_normal(rng.choice(hidden, min(len(hidden), int(rng.integers(1, 300))), replace=False))
                elif kind == 6:
                    v._apply_selection_with_modifiers(np.arange(start, start + int(rng.integers(1, 400)), dtype=np.int64), "テスト", Qt.NoModifier)
                elif kind == 7:
                    v.clear_selection()
                elif kind == 8:
                    v.batch_move_indices_to_layers([
                        (np.arange(start, start + 50, dtype=np.int64), V.LAYER_NOISE),
                        (np.arange(start + 100, start + 180, dtype=np.int64), lid_show),
                    ])
                else:
                    v.invert_selection() if rng.random() < 0.3 else v.set_color_mode(
                        [V.COLOR_MODE_RGB, V.COLOR_MODE_HEIGHT, V.COLOR_MODE_LAYER][int(rng.integers(0, 3))])
                self.assert_display_ok(f"試行 {trial} の {step} 手目（種類 {kind}）")
            before = self.full_calls
            v._update_display_arrays()
            self.assertEqual(self.full_calls, before + 1)
            self.assert_display_ok(f"試行 {trial}: 最後に全部を作り直す", exact=True)
            self.setUp_again()

    def setUp_again(self):
        self.tearDown()
        self.setUp()


class TestFallbacks(_ViewerCase):
    def _preloaded(self, n=6000, budget=0):
        import math

        pts, colors, intens, scan_ids = make_cloud(n, seed=31)
        center = (pts.min(axis=0) + pts.max(axis=0)) / 2.0
        pc = np.ascontiguousarray(pts - center, dtype=np.float32)
        if budget <= 0 or n <= budget:
            dp, dc, di = pc, colors, np.arange(n, dtype=np.int64)
        else:
            di = np.arange(0, n, max(1, int(math.ceil(n / budget))), dtype=np.int64)
            dp, dc = np.ascontiguousarray(pc[di]), np.ascontiguousarray(colors[di])
        return types.SimpleNamespace(
            scans=None, points_raw=pts, points_centered=pc, center_offset=center, colors_raw=colors, intensities_raw=intens,
            active_colors=colors, world_bounds_min=pts.min(axis=0), world_bounds_max=pts.max(axis=0),
            bounds_min=(pts.min(axis=0) - center).astype(np.float32), bounds_max=(pts.max(axis=0) - center).astype(np.float32),
            display_points=dp, display_colors=dc, display_orig_indices=di, scan_ids=scan_ids,
        )

    def test_right_after_loading_the_shared_arrays_are_not_touched(self):
        v = self.v
        pre = self._preloaded()
        pc_before, raw_before = pre.points_centered.copy(), pre.colors_raw.copy()
        v.set_preloaded_data(pre)
        self.assertIs(v.display_points, v.points_centered, "前提: 読み込みの直後は、表示用の配列が元の配列そのもの")
        self.assertFalse(v._display_patchable())
        self.full_calls = 0
        v.move_indices_to_noise(np.arange(100, 400, dtype=np.int64))
        self.assertEqual(self.full_calls, 1, "最初の 1 回は、全部を作り直す")
        np.testing.assert_array_equal(v.points_centered, pc_before, err_msg="元の点を書き換えてはいけない")
        np.testing.assert_array_equal(v.colors_raw, raw_before, err_msg="元の色を書き換えてはいけない")
        self.assert_display_ok("読み込みの直後に消す", exact=True)
        v.move_indices_to_noise(np.arange(1000, 1200, dtype=np.int64))
        self.assertEqual(self.full_calls, 1, "2 回目からは、部分更新")
        np.testing.assert_array_equal(v.points_centered, pc_before)
        np.testing.assert_array_equal(v.colors_raw, raw_before)
        self.assert_display_ok("2 回目")

    def test_a_new_point_cloud_forgets_the_previous_state(self):
        v = self.v
        self.load(n=6000, seed=32)
        v.move_indices_to_noise(np.arange(100, 400, dtype=np.int64))
        self.assertEqual(v._display_dups, 300)
        self.assertIsNotNone(v._display_stamp)
        v.set_preloaded_data(self._preloaded(n=6000))                        # 同じ点数の別の点群
        self.assertEqual(v._display_dups, 0)
        self.assertIsNone(v._display_stamp, "前の点群の印が残っていると、共有している配列を書き換えてしまう")
        self.assertEqual(v.shown_point_count(), 6000)

    def test_subsampled_display_always_rebuilds(self):
        v = self.v
        self.load(n=20000, seed=33)
        v.set_point_budget(5000)
        self.full_calls = 0
        v.move_indices_to_noise(np.arange(100, 400, dtype=np.int64))
        self.assertEqual(self.full_calls, 1)
        self.assert_display_ok("間引き中に消す", exact=True)
        v.restore_indices_to_normal(np.arange(100, 400, dtype=np.int64))
        self.assertEqual(self.full_calls, 2)
        self.assert_display_ok("間引き中に戻す", exact=True)

    def test_restoring_past_the_budget_rebuilds(self):
        v = self.v
        self.load(n=20000, seed=34)
        v.move_indices_to_noise(np.arange(100, 700, dtype=np.int64))          # 残り 19,400 点
        v.set_point_budget(19500)                                             # 収まるので、間引かない
        self.assertEqual(len(v.display_points), 19400)
        self.full_calls = 0
        v.move_indices_to_noise(np.arange(1000, 1050, dtype=np.int64))
        self.assertEqual(self.full_calls, 0)
        v.restore_indices_to_normal(np.arange(100, 700, dtype=np.int64))      # 19,950 点になり、上限を超える
        self.assertEqual(self.full_calls, 1, "入れる枠が無いので全部を作り直し、そこで上限に合わせて間引かれる")
        self.assert_display_ok("上限を超えて戻す", exact=True)
        self.assertLessEqual(len(v.display_points), 19500)

    def test_first_point_and_everything_gone(self):
        v = self.v
        self.load(n=5000, seed=35)
        v.move_indices_to_noise(np.arange(0, 30, dtype=np.int64))             # 先頭の点（手前の枠が無い）
        self.assertEqual(self.full_calls, 1)
        self.assert_display_ok("先頭を消す", exact=True)
        v.move_indices_to_noise(np.arange(30, 60, dtype=np.int64))            # いまの先頭（番号 30）
        self.assertEqual(self.full_calls, 2)
        self.assert_display_ok("もう一度、先頭を消す", exact=True)
        v.move_indices_to_noise(np.arange(0, 5000, dtype=np.int64))           # 全部
        self.assertEqual(len(v.display_points), 0)
        self.assertEqual(v.shown_point_count(), 0)
        v.restore_indices_to_normal(np.arange(10, 20, dtype=np.int64))
        self.assert_display_ok("何も無い所へ戻す", exact=True)

    def test_empty_display_forgets_the_partial_state(self):
        v = self.v
        self.load(n=5000, seed=46)
        v.move_indices_to_noise(np.arange(100, 400, dtype=np.int64))
        self.assertEqual(v._display_dups, 300)
        v.set_layer_visibility(False, False)
        self.assertEqual(len(v.display_points), 0)
        self.assertEqual(v._display_dups, 0)
        self.assertIsNone(v._display_stamp)
        self.assertEqual(v.shown_point_count(), 0)
        v.set_layer_visibility(True, False)
        self.assert_display_ok("", exact=True)

    def test_too_many_duplicates_trigger_a_compacting_rebuild(self):
        v = self.v
        self.load(n=8000, seed=36)
        for k in range(1, 9):                                                 # 300 点ずつ。7 回目で 1/4（2,000 枠）を超える
            v.move_indices_to_noise(np.arange(k * 800, k * 800 + 300, dtype=np.int64))
            self.assert_display_ok(f"{k} 回目")
            self.assertLessEqual(v._display_dups * DP.DUP_REBUILD_FRACTION, len(v.display_points))
        self.assertEqual(self.full_calls, 1, "増えすぎたら 1 回だけ詰め直す")
        self.assertLess(len(v.display_points), 8000)

    def test_big_moves_rebuild(self):
        v = self.v
        self.load(n=8000, seed=37)
        v.move_indices_to_noise(np.arange(100, 100 + 2100, dtype=np.int64))    # 表示の 1/4 を超える
        self.assertEqual(self.full_calls, 1)
        self.assert_display_ok("大きく消す", exact=True)

    def test_restore_without_free_slots_rebuilds(self):
        v = self.v
        self.load(n=8000, seed=38)
        v.move_indices_to_noise(np.arange(100, 200, dtype=np.int64))
        v._update_display_arrays()                                            # 詰め直す（空いた枠が無くなる）
        self.full_calls = 0
        v.move_indices_to_noise(np.arange(300, 310, dtype=np.int64))          # 別の所に、10 枠だけ空く
        v.restore_indices_to_normal(np.arange(100, 200, dtype=np.int64))
        self.assertEqual(self.full_calls, 1, "入れる枠が無ければ、全部を作り直す")
        self.assert_display_ok("枠の無い所へ戻す", exact=True)
        # 枠はあるが、足りない
        v.move_indices_to_noise(np.arange(500, 520, dtype=np.int64))
        v.layer_ids[520:560] = V.LAYER_NOISE                                  # 表示は直さずに、レイヤーだけ変えた形
        v._sync_mask_from_ids()
        self.full_calls = 0
        v.restore_indices_to_normal(np.arange(500, 520, dtype=np.int64))      # 20 枠に 20 点: 入る
        self.assertEqual(self.full_calls, 0)

    def test_changed_display_settings_rebuild(self):
        v = self.v
        self.load(n=8000, seed=39)
        lid = v.add_custom_layer("あとで隠すレイヤー", color=(0.2, 0.2, 0.9))
        v.move_indices_to_layer(np.arange(4000, 4500, dtype=np.int64), lid)
        v._update_display_arrays()
        self.full_calls = 0
        v.layers[lid].visible = False                                         # 設定だけ変えて、表示を作り直していない
        self.assertFalse(v._display_patchable())
        v.move_indices_to_noise(np.arange(100, 110, dtype=np.int64))
        self.assertEqual(self.full_calls, 1)
        self.assert_display_ok("レイヤーの表示が変わっていた", exact=True)

        v.clip_z_enabled = True                                               # 断面の設定だけ変えた形
        v.clip_z_range = [-3.0, 3.0]
        self.assertFalse(v._display_patchable())
        v.move_indices_to_noise(np.arange(200, 210, dtype=np.int64))
        self.assertEqual(self.full_calls, 2)
        self.assert_display_ok("断面が変わっていた", exact=True)

        v.scan_visibility[0] = False
        v.move_indices_to_noise(np.arange(300, 310, dtype=np.int64))
        self.assertEqual(self.full_calls, 3)
        self.assert_display_ok("スキャンの表示が変わっていた", exact=True)

        # 断面は有効のまま、範囲だけが変わった形
        v.clip_z_range = [-1.0, 2.0]
        self.assertFalse(v._display_patchable())
        v.move_indices_to_noise(np.arange(400, 410, dtype=np.int64))
        self.assertEqual(self.full_calls, 4)
        self.assert_display_ok("断面の範囲だけが変わっていた", exact=True)

        # 表示点数の上限・断面の中の全点表示の設定だけが変わった形
        v.point_budget = 7000
        self.assertFalse(v._display_patchable())
        v.move_indices_to_noise(np.arange(500, 510, dtype=np.int64))
        self.assertEqual(self.full_calls, 5)
        self.assert_display_ok("上限が変わっていた", exact=True)
        v.point_budget = 0
        v._update_display_arrays()
        self.assertTrue(v._display_patchable())
        v.section_full_detail = not v.section_full_detail
        self.assertFalse(v._display_patchable())

    def test_wall_section_flip_without_a_rebuild_is_noticed(self):
        # 斜めの壁の断面が有効のまま、向きを反転する操作は、表示を作り直さない（次に消すときに、全部を作り直して追いつく）
        v = self.v
        self.load(n=8000, seed=42)
        v.set_wall_align_points(np.array([-4.0, -1.0, 0.0]), np.array([5.0, 3.0, 0.0]))
        v.set_section_wall(True, -1.0, 3.0)
        self.assertTrue(v._display_patchable())
        v.flip_wall_aligned_view()
        self.assertFalse(v._display_patchable(), "壁の向きが変わったのに、前の表示のまま部分更新してはいけない")
        self.full_calls = 0
        v.move_indices_to_noise(np.flatnonzero(v._get_section_inlier_mask())[5:40])
        self.assertEqual(self.full_calls, 1)
        self.assert_display_ok("壁の向きを反転したあとで消す", exact=True)

    def test_display_arrays_sharing_the_originals_are_never_patched(self):
        # 表示用の配列が元の配列そのものなら、設定の控えが合っていても、書き換えずに全部を作り直す
        v = self.v
        self.load(n=6000, seed=43)
        v._update_active_colors()
        pc_before = v.points_centered.copy()
        stamp = v._display_stamp
        v.display_points = v._preview_base_points = v.points_centered
        self.assertEqual(v._display_stamp, stamp)
        self.assertFalse(v._display_patchable())
        self.full_calls = 0
        v.move_indices_to_noise(np.arange(100, 300, dtype=np.int64))
        self.assertEqual(self.full_calls, 1)
        np.testing.assert_array_equal(v.points_centered, pc_before, err_msg="元の点を書き換えてはいけない")
        self.assert_display_ok("点の配列を共有していた", exact=True)

        ac_before = v.active_colors.copy()
        v.display_colors = v._preview_base_colors = v.active_colors
        self.assertFalse(v._display_patchable())
        self.full_calls = 0
        v.move_indices_to_noise(np.arange(1000, 1200, dtype=np.int64))
        self.assertEqual(self.full_calls, 1)
        keep = np.ones(6000, dtype=bool)
        keep[1000:1200] = False
        np.testing.assert_array_equal(v.active_colors[keep], ac_before[keep], err_msg="移していない点の色を書き換えてはいけない")
        self.assert_display_ok("色の配列を共有していた", exact=True)

    def test_preview_base_follows_the_point_cloud(self):
        v = self.v
        self.load(n=6000, seed=44)
        v.move_indices_to_noise(np.arange(100, 300, dtype=np.int64))
        old_base = v._preview_base_points
        pre = self._preloaded(n=5000)
        v.set_preloaded_data(pre)
        self.assertIs(v._preview_base_points, v.display_points, "前の点群の表示用の配列を、持ち続けない")
        self.assertIs(v._preview_base_colors, v.display_colors)
        self.assertIsNot(v._preview_base_points, old_base)
        v.set_section_z(True, -2.0, 2.0, is_preview=True)                     # 読み込みの直後の、断面の下書き表示
        cz = float(v.center_offset[2])
        inside = (v.points_centered[:, 2] >= (-2.0 - cz)) & (v.points_centered[:, 2] <= (2.0 - cz))
        self.assertGreater(int(np.count_nonzero(inside)), 0)
        np.testing.assert_array_equal(v.display_points, v.points_centered[inside], err_msg="新しい点群から切り出している")
        v.set_point_cloud(None)
        self.assertIsNone(v._preview_base_points)
        self.assertIsNone(v._display_stamp)
        self.assertEqual(v._display_dups, 0)
        self.assertEqual(v.shown_point_count(), 0)

    def test_section_preview_skips_duplicate_slots(self):
        # 断面のスライダーを動かしている間（下書き表示）も、描いている点の数が合うこと
        v = self.v
        self.load(n=8000, seed=45)
        v.set_section_z(True, -8.0, 8.0)                                      # 全部が入る断面
        v.move_indices_to_noise(np.arange(500, 900, dtype=np.int64))          # 499 の写しが 400 枠
        self.assertEqual(v._display_dups, 400)
        z499 = float(v.points_raw[499, 2])
        v.set_section_z(True, z499 - 3.0, z499 + 3.0, is_preview=True)        # 499 が入る範囲
        mask = v._get_section_inlier_mask() & v._get_layer_and_scan_visibility_mask()
        self.assertTrue(mask[499])
        self.assertEqual(len(v.display_points), int(np.count_nonzero(mask)), "下書きの配列に、同じ点を 2 回入れない")
        self.assertEqual(v.shown_point_count(), int(np.count_nonzero(mask)))
        np.testing.assert_array_equal(v.display_points, v.points_centered[mask])
        np.testing.assert_array_equal(v.display_colors, expected_colors(v)[mask])
        v.set_section_z(True, z499 - 3.0, z499 + 3.0)                         # 確定
        self.assert_display_ok("確定", exact=True)

    def test_preview_states_rebuild(self):
        v = self.v
        self.load(n=8000, seed=40)
        v.set_section_z(True, -8.0, 8.0)
        v.set_section_z(True, -2.0, 2.0, is_preview=True)                     # 断面のスライダーを動かしている途中
        self.assertIsNot(v.display_points, v._preview_base_points)
        self.full_calls = 0
        v.move_indices_to_noise(np.arange(100, 200, dtype=np.int64))
        self.assertEqual(self.full_calls, 1)
        self.assert_display_ok("断面の下書き表示の途中", exact=True)

        v.set_color_mode(V.COLOR_MODE_HEIGHT)
        v.preview_custom_height_range(True, -1.0, 1.0)                        # 高さの範囲の下書き表示（表示用の色だけが替わる）
        self.assertFalse(v._display_patchable())
        self.full_calls = 0
        v.move_indices_to_noise(np.arange(300, 400, dtype=np.int64))
        self.assertGreaterEqual(self.full_calls, 1)
        self.assert_display_ok("高さの範囲の下書き表示のあと", exact=True)

    def test_section_counts_follow_the_partial_updates(self):
        v = self.v
        self.load(n=8000, seed=41)
        v.set_section_z(True, -4.0, 4.0)
        box = v.section_box_count
        self.assertGreater(box, 0)
        inside = np.flatnonzero(v._get_section_inlier_mask())
        outside = np.flatnonzero(~v._get_section_inlier_mask())
        self.full_calls = 0
        v.move_indices_to_noise(inside[10:60])
        v.move_indices_to_noise(outside[:40])                                 # 断面の外の点は、表示に関係しない
        self.assertEqual(self.full_calls, 0)
        self.assertEqual(v.section_box_count, box - 50)
        self.assert_display_ok("断面の中と外を消す")
        v.restore_indices_to_normal(np.r_[inside[10:60], outside[:40]])
        self.assertEqual(self.full_calls, 0)
        self.assertEqual(v.section_box_count, box)
        self.assert_display_ok("戻す", exact=True)


class TestHelpers(unittest.TestCase):
    def test_expand_ranges(self):
        rng = np.random.default_rng(50)
        for _ in range(200):
            k = int(rng.integers(1, 12))
            left = rng.integers(0, 50, k)
            right = left + rng.integers(0, 5, k)
            expect = np.concatenate([np.arange(a, b) for a, b in zip(left, right)]) if k else np.empty(0)
            got = CP.expand_ranges(left, right)
            np.testing.assert_array_equal(got, expect.astype(np.int64))
        self.assertEqual(len(CP.expand_ranges(np.array([3, 5]), np.array([3, 5]))), 0)

    def test_queue_blocks(self):
        b = CP.PATCH_BLOCK
        got = CP.queue_blocks(None, 10 * b, np.array([5, b + 1, b + 2]))
        self.assertEqual(got[0], 10 * b)
        np.testing.assert_array_equal(got[1], [0, 1])
        got = CP.queue_blocks(got, 10 * b, np.array([7 * b]))
        np.testing.assert_array_equal(got[1], [0, 1, 7])
        self.assertIsNone(CP.queue_blocks(got, 10 * b, np.array([3 * b, 4 * b])), "半分以上なら、まるごと")
        got = CP.queue_blocks((9 * b, np.array([8])), 10 * b, np.array([5]))
        np.testing.assert_array_equal(got[1], [0], err_msg="長さが違うときの覚えは、引き継がない")


class TestGpuUpload(_ViewerCase):
    def _picture(self):
        img = self.v.grabFramebuffer()
        if img.isNull() or img.width() == 0:
            self.skipTest("OpenGL の画面を取り込めない環境")
        img = img.convertToFormat(img.Format.Format_RGBA8888)
        arr = np.frombuffer(img.constBits(), dtype=np.uint8).reshape(img.height(), img.bytesPerLine() // 4, 4)
        return arr[:, : img.width(), :3].copy()

    @staticmethod
    def _red_pixels(pic) -> int:
        return int(np.count_nonzero((pic[:, :, 0] > 180) & (pic[:, :, 1] < 80) & (pic[:, :, 2] < 80)))

    def test_partial_upload_draws_the_same_picture_as_a_full_upload(self):
        v = self.v
        n = 400_000                                               # 区画（65,536 点）が 7 つ
        rng = np.random.default_rng(51)
        pts = rng.uniform(-5.0, 5.0, (n, 3))
        pts[: 3 * CP.PATCH_BLOCK, 0] -= 30.0                      # 前のほうの区画は、画面の左へ離す
        colors = np.tile(np.array([0.05, 0.15, 0.5], dtype=np.float32), (n, 1))     # 全部、暗い青
        red = np.arange(4 * CP.PATCH_BLOCK + 100, 4 * CP.PATCH_BLOCK + 30000, dtype=np.int64)
        colors[red] = (1.0, 0.0, 0.0)                             # 5 つ目の区画に、赤い点
        v.set_shading_enabled(False)
        v.set_point_budget(0)
        v.set_point_cloud(pts, colors, None)
        self.app.processEvents()
        first = self._picture()
        n_red = self._red_pixels(first)                           # 軸の印などの赤も数に入る
        self.assertGreater(n_red, 150, "前提: 赤い点が見えている")
        self.assertFalse(v._vbo_points_dirty or v._vbo_colors_dirty, "前提: 1 回描いたあとは、送り直す予定が無い")

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
            self.full_calls = 0
            v.move_indices_to_noise(red)                          # 赤い点を消す
            self.assertEqual(self.full_calls, 0)
            self.assertFalse(v._vbo_points_dirty or v._vbo_colors_dirty, "まるごと送り直す予定にしない")
            np.testing.assert_array_equal(v._point_patch[1], [4])
            np.testing.assert_array_equal(v._color_patch[1], [4])
            partial = self._picture()
            self.assertIsNone(v._point_patch, "描いたら、覚えていた区画は消える")
            self.assertEqual(sent, [("部分", CP.PATCH_BLOCK * 12)] * 2, "点と色を、1 区画ずつ送る")
            self.assertLess(self._red_pixels(partial), n_red - 100, "消した点は、描かれない")

            sent.clear()
            v._vbo_dirty = True                                   # まるごと送り直させる
            np.testing.assert_array_equal(partial, self._picture(), err_msg="部分だけ送った絵が、まるごと送った絵と違う")
            sent.clear()
            v._update_display_arrays()                            # 詰め直した配列で描く
            np.testing.assert_array_equal(partial, self._picture(), err_msg="重複の枠のある絵が、詰め直した絵と違う")

            # 戻す: 最初の絵に戻る
            v.move_indices_to_noise(red)                          # いちど戻して消し直す（重複の枠を作る）
            v.restore_indices_to_normal(red)
            v.move_indices_to_noise(red)
            self._picture()
            sent.clear()
            v.restore_indices_to_normal(red)
            self.assertFalse(v._vbo_points_dirty or v._vbo_colors_dirty)
            restored = self._picture()
            self.assertEqual(sent, [("部分", CP.PATCH_BLOCK * 12)] * 2)
            np.testing.assert_array_equal(restored, first, err_msg="戻した絵が、最初の絵と違う")

            # 離れた所を続けて消す（描く前に 2 回）: 両方の区画を送る
            sent.clear()
            v.move_indices_to_noise(red[:500])
            v.move_indices_to_noise(np.arange(6 * CP.PATCH_BLOCK + 10, 6 * CP.PATCH_BLOCK + 400, dtype=np.int64))
            np.testing.assert_array_equal(v._point_patch[1], [4, 6])
            twice = self._picture()
            v._vbo_dirty = True
            np.testing.assert_array_equal(twice, self._picture(), err_msg="1 回目に消した区画を送り忘れている")
        finally:
            CP.gl.glBufferSubData, CP.gl.glBufferData = orig_sub, orig_data

    def test_patch_queued_before_a_rebuild_is_dropped(self):
        v = self.v
        n = 350_000
        rng = np.random.default_rng(52)
        v.set_shading_enabled(False)
        v.set_point_budget(0)
        v.set_point_cloud(rng.uniform(-5.0, 5.0, (n, 3)), np.tile(np.array([0.05, 0.15, 0.5], dtype=np.float32), (n, 1)), None)
        self.app.processEvents()
        self._picture()
        v.move_indices_to_noise(np.arange(n - 300, n - 100, dtype=np.int64))  # 最後の（半端な）区画
        np.testing.assert_array_equal(v._point_patch[1], [5])
        partial = self._picture()
        v._vbo_dirty = True
        np.testing.assert_array_equal(partial, self._picture())
        v.move_indices_to_noise(np.arange(1000, 1200, dtype=np.int64))
        self.assertIsNotNone(v._point_patch)
        v.set_point_budget(200_000)                                           # 描く前に、表示用の配列が作り直される
        self.assertIsNone(v._point_patch)
        self.assertTrue(v._vbo_points_dirty)
        after = self._picture()
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
        win.viewer.set_point_budget(0)
        win.viewer.set_point_cloud(pts, colors, intens, scan_ids=scan_ids)
        self.app.processEvents()
        self.full_calls = 0
        self._orig_full = win.viewer._update_display_arrays

        def counted():
            self.full_calls += 1
            return self._orig_full()

        win.viewer._update_display_arrays = counted

    def tearDown(self):
        del self.win.viewer._update_display_arrays
        self.win._toggle_instant_delete_mode(False)

    def check(self, where, exact=False):
        v = self.win.viewer
        ref_idx, ref_pts, ref_cols = reference_display(v)
        shown = v.display_orig_indices
        np.testing.assert_array_equal(np.unique(shown), ref_idx, err_msg=where)
        np.testing.assert_array_equal(v.display_points, v.points_centered[shown], err_msg=where)
        np.testing.assert_array_equal(v.display_colors, expected_colors(v)[shown], err_msg=where)
        self.assertEqual(v.shown_point_count(), len(ref_idx), where)
        if exact:
            np.testing.assert_array_equal(shown, ref_idx, err_msg=where)
        v.validate_invariants()

    def test_delete_undo_redo(self):
        win, v = self.win, self.win.viewer
        v._apply_selection_with_modifiers(np.arange(100, 900, dtype=np.int64), "テスト", Qt.NoModifier)
        win._delete_selected_points()
        self.check("Del")
        self.assertEqual(v.shown_point_count(), 25000 - 800)
        win._undo_action()
        self.check("元に戻す", exact=True)
        win._redo_action()
        self.check("やり直し")
        win._undo_action()
        self.check("もう一度、元に戻す", exact=True)
        self.assertEqual(self.full_calls, 0, "消す・戻すだけなら、表示用の配列を全部は作り直さない")

    def test_instant_erase_and_undo(self):
        win, v = self.win, self.win.viewer
        win._toggle_instant_delete_mode(True)
        for start in (2000, 9000, 15000):
            v._apply_selection_with_modifiers(np.arange(start, start + 600, dtype=np.int64), "テスト", Qt.NoModifier)   # 即削除
            self.check(f"即削除 {start}")
        self.assertEqual(v.shown_point_count(), 25000 - 1800)
        for _ in range(3):
            win._undo_action()
            self.check("即削除を元に戻す")
        self.assertEqual(v.shown_point_count(), 25000)
        self.check("すべて戻した", exact=True)
        self.assertEqual(self.full_calls, 0)

    def test_undo_of_a_mixed_batch_move(self):
        # 自動処理の結果をまとめて反映した形（消える点と戻る点が混じる）を、元に戻す・やり直す
        win, v = self.win, self.win.viewer

        def span(a, b):
            return np.arange(a, b, dtype=np.int64)

        win._push_layer_undo_state()
        v.move_indices_to_noise(span(4000, 4200))
        win._push_layer_undo_state()
        v.batch_move_indices_to_layers([(span(4100, 4200), V.LAYER_NORMAL), (span(4200, 4300), V.LAYER_NOISE), (span(3950, 4000), V.LAYER_NOISE)])
        self.check("混ぜて移す")
        win._undo_action()
        self.check("混ぜた移動を元に戻す")
        win._redo_action()
        self.check("やり直し")
        win._undo_action()
        win._undo_action()
        self.check("すべて元に戻す", exact=True)
        self.assertEqual(self.full_calls, 0)

    def test_undo_that_changes_layer_definitions_rebuilds(self):
        win, v = self.win, self.win.viewer
        win._push_layer_undo_state()
        lid = v.add_custom_layer("あとで消えるレイヤー", color=(0.9, 0.1, 0.9))
        v.move_indices_to_layer(np.arange(3000, 3300, dtype=np.int64), lid)
        self.check("新しいレイヤーへ移す")
        before = self.full_calls
        win._undo_action()
        self.assertGreater(self.full_calls, before, "レイヤーの定義が変わる取り消しは、全部を作り直す")
        self.check("レイヤーごと元に戻す", exact=True)


if __name__ == "__main__":
    unittest.main()
