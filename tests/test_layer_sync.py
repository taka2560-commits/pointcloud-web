# -*- coding: utf-8 -*-
"""
レイヤーの整合の確認と、点数の数え直しの高速化（gui/layer_sync.py）のテスト

きっかけ（2026-11-14）: 点を消す・戻す 1 回の操作で、layer_mask と layer_ids の全点の比べ（_ensure_layer_sync）が 4〜8 回、
レイヤーごと・スキャンごとの点数の数え直し（全点の並べ替え）が 2 回ずつ走っていた（5,490 万点で、合わせて約 1 秒。
「自動除去:」のノイズレイヤーがあると、np.isin が遅い道に入って約 3 秒）。

結果が以前と同じこと
- 「ノイズ扱いのレイヤーの点か」が、np.isin と同じこと（ノイズ扱いのレイヤーが 1 つ・いくつか・たくさん）
- _sync_mask_from_ids / _ensure_layer_sync / get_layer_counts / get_scan_counts の結果が、以前の作り（テストの中に写してある）と同じこと
- どんな順で操作しても、覚えで「合っている」と見なした状態が、全点を比べても本当に合っていること

省くこと
- 何も変わっていなければ、全点の比べも、点数の数え直しもしないこと

守り（今までどおり全点を比べる場面）
- layer_mask を配列ごと差し替えた・その場で書き換えた・ノイズ扱いのレイヤーが変わった・点群が替わった・force=True
"""
import gc
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.settings_sandbox import use_sandbox_settings  # noqa: E402
from tests.qt_cleanup import destroy_widget  # noqa: E402

use_sandbox_settings()  # この PC の本番アプリの保存設定を、テストで書き換えない

import numpy as np  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from gui import layer_sync as LS  # noqa: E402
from gui.gl_viewer import PointCloudViewer  # noqa: E402
from tests.test_color_patch import make_cloud  # noqa: E402

V = PointCloudViewer


# ---------------------------------------------------------------- 答え合わせ用: 2026-11-14 に直す前の作りを、そのまま写したもの
def ref_noise_ids(v):
    noise_ids = [lid for lid in v.layers if v.is_noise_layer(lid)]
    if v.LAYER_NOISE not in noise_ids:
        noise_ids.append(v.LAYER_NOISE)
    return noise_ids


def ref_mask_from_ids(v):
    return ~np.isin(v.layer_ids, ref_noise_ids(v))


def ref_ensure(v):
    """以前の _ensure_layer_sync が返すはずの (layer_ids, layer_mask)。ビューアの状態は変えない"""
    if v.points_raw is None:
        return None, None
    n = len(v.points_raw)
    ids = None if v.layer_ids is None else v.layer_ids.copy()
    mask = None if v.layer_mask is None else v.layer_mask.copy()
    if ids is None or len(ids) != n:
        if mask is not None and len(mask) == n:
            ids = np.where(mask, v.LAYER_NORMAL, v.LAYER_NOISE).astype(np.int32)
        else:
            ids = np.full(n, v.LAYER_NORMAL, dtype=np.int32)
            mask = np.ones(n, dtype=bool)
        return ids, mask
    if mask is not None and len(mask) == n:
        noise_ids = ref_noise_ids(v)
        expected = ~np.isin(ids, noise_ids)
        if not np.array_equal(mask, expected):
            to_noise = (~mask) & np.isin(ids, noise_ids, invert=True)
            to_normal = mask & np.isin(ids, noise_ids)
            ids[to_noise] = v.LAYER_NOISE
            ids[to_normal] = v.LAYER_NORMAL
    return ids, mask


def ref_layer_counts(v):
    if v.layer_ids is None:
        return {lid: 0 for lid in v.layers}
    unique_ids, counts = np.unique(v.layer_ids, return_counts=True)
    count_map = {int(k): int(c) for k, c in zip(unique_ids, counts)}
    return {lid: count_map.get(lid, 0) for lid in v.layers}


def ref_scan_counts(v):
    if v.scan_ids is None:
        return {}
    unique_ids, counts = np.unique(v.scan_ids, return_counts=True)
    return {int(k): int(c) for k, c in zip(unique_ids, counts)}


class _ViewerCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def setUp(self):
        self.v = PointCloudViewer()
        self.v.resize(480, 360)
        self.v.show()
        self.app.processEvents()
        self.spy()

    def tearDown(self):
        self.v.set_point_cloud(None)
        destroy_widget(self.v)

    def spy(self):
        """全点を比べる処理（_check_layer_sync）が呼ばれた回数を数える"""
        self.full_checks = 0
        orig = self.v._check_layer_sync

        def counted():
            self.full_checks += 1
            return orig()

        self.v._check_layer_sync = counted

    def load(self, n=20000, seed=0):
        pts, colors, intens, scan_ids = make_cloud(n, seed)
        self.v.set_point_cloud(pts, colors, intens, scan_ids=scan_ids)
        self.app.processEvents()
        self.v._ensure_layer_sync()
        self.full_checks = 0
        return pts

    def assert_truly_synced(self, where=""):
        """いまの layer_mask / layer_ids / 点数が、以前の作りの結果と同じ。全点を比べ直しても、何も変わらない"""
        v = self.v
        np.testing.assert_array_equal(v.layer_mask, ref_mask_from_ids(v), err_msg=f"layer_mask が layer_ids と合っていない: {where}")
        self.assertEqual(v.get_layer_counts(), ref_layer_counts(v), f"覚えている点数が古い: {where}")   # 全点を比べ直す前に見る
        ids_before, mask_before = v.layer_ids.copy(), v.layer_mask.copy()
        v._ensure_layer_sync(force=True)
        np.testing.assert_array_equal(v.layer_ids, ids_before, err_msg=f"全点を比べ直したら layer_ids が変わった: {where}")
        np.testing.assert_array_equal(v.layer_mask, mask_before, err_msg=where)
        self.assertEqual(v.get_layer_counts(), ref_layer_counts(v), where)
        self.assertEqual(list(v.get_layer_counts()), list(v.layers), f"並びは self.layers の順: {where}")
        self.assertEqual(v.get_scan_counts(), ref_scan_counts(v), where)
        v.validate_invariants()


class TestNoisePoint(_ViewerCase):
    def test_same_as_isin_for_any_number_of_noise_layers(self):
        v = self.v
        self.load(n=5000, seed=1)
        rng = np.random.default_rng(2)
        v.layer_ids[:] = rng.integers(0, 14, 5000).astype(np.int32)
        for wanted in ([0], [0, 4], [4, 0, 4], [0, 2, 3, 5, 7, 9, 11, 13], list(range(0, 13)), [99], [0, 99, 5], [1]):
            got = v._is_noise_point(wanted)
            self.assertEqual(got.dtype, np.bool_)
            np.testing.assert_array_equal(got, np.isin(v.layer_ids, wanted), err_msg=str(wanted))
        self.assertGreater(len(list(range(0, 13))), LS.NOISE_COMPARE_MAX, "前提: 多いときの道（np.isin）も通している")

    def test_sync_mask_matches_the_previous_implementation(self):
        v = self.v
        self.load(n=8000, seed=3)
        rng = np.random.default_rng(4)
        made = []
        for k in range(11):                                          # ノイズ扱いのレイヤーを、1 つずつ増やす（0 → 11 個）
            v.layer_ids[rng.choice(8000, 600, replace=False)] = V.LAYER_NOISE
            if made:
                v.layer_ids[rng.choice(8000, 300, replace=False)] = made[-1]
            before = v.layer_mask
            v._sync_mask_from_ids()
            self.assertIsNot(v.layer_mask, before, "layer_mask は、新しい配列として作る（前の配列を書き換えない）")
            np.testing.assert_array_equal(v.layer_mask, ref_mask_from_ids(v), err_msg=f"ノイズ扱いのレイヤー {len(made) + 1} 個")
            self.assert_truly_synced(f"ノイズ扱いのレイヤー {len(made) + 1} 個")
            made.append(v.add_custom_layer(f"自動除去: テスト {k}", is_noise=True))
        self.assertGreater(len(ref_noise_ids(v)), LS.NOISE_COMPARE_MAX)


class TestEnsureSync(_ViewerCase):
    def test_unchanged_layers_are_not_compared_again(self):
        v = self.v
        self.load(n=20000, seed=5)
        for _ in range(5):
            v._ensure_layer_sync()
            v._get_selectable_mask()
            v.get_layer_counts()
            v.get_normal_indices()
        self.assertEqual(self.full_checks, 0, "何も変わっていなければ、全点を比べない")
        v.move_indices_to_noise(np.arange(100, 900, dtype=np.int64))
        v.move_indices_to_layer(np.arange(2000, 2500, dtype=np.int64), v.add_custom_layer("表示するレイヤー"))
        v.restore_indices_to_normal(np.arange(100, 500, dtype=np.int64))
        v.batch_move_indices_to_layers([(np.arange(5000, 5100, dtype=np.int64), V.LAYER_NOISE)])
        self.assertEqual(self.full_checks, 0, "ビューアが自分で移したあとは、合っていると分かっている")
        self.assert_truly_synced("移したあと")

    def test_replaced_mask_is_compared_and_the_ids_follow(self):
        v = self.v
        self.load(n=20000, seed=6)
        lid = v.add_custom_layer("表示するレイヤー")
        v.move_indices_to_layer(np.arange(3000, 4000, dtype=np.int64), lid)
        v.move_indices_to_noise(np.arange(0, 1000, dtype=np.int64))
        new_mask = v.layer_mask.copy()
        new_mask[500:1500] = True                                    # ノイズの一部を戻す
        new_mask[3500:4500] = False                                  # カスタムレイヤーの一部と、通常の点を消す
        v.layer_mask = new_mask                                      # main_window._restore_layer_snapshot と同じ形（配列ごと差し替え）
        exp_ids, exp_mask = ref_ensure(v)
        self.full_checks = 0
        v._ensure_layer_sync()
        self.assertEqual(self.full_checks, 1)
        np.testing.assert_array_equal(v.layer_ids, exp_ids)
        np.testing.assert_array_equal(v.layer_mask, exp_mask)
        self.assertTrue(np.all(v.layer_ids[500:1000] == V.LAYER_NORMAL))
        self.assertTrue(np.all(v.layer_ids[3500:4500] == V.LAYER_NOISE))
        v._ensure_layer_sync()
        self.assertEqual(self.full_checks, 1, "直したあとは、また省く")
        self.assert_truly_synced("差し替えのあと")

    def test_replaced_arrays_with_the_same_counts_are_noticed(self):
        # True の数が同じでも、配列ごと差し替えられたら、全点を比べる
        v = self.v
        self.load(n=20000, seed=20)
        v.move_indices_to_noise(np.arange(1000, 1500, dtype=np.int64))
        new_mask = v.layer_mask.copy()
        new_mask[1000:1500] = True                                   # 消した 500 点を戻し、
        new_mask[6000:6500] = False                                  # 別の 500 点を消す（True の数は同じ）
        self.assertEqual(int(np.count_nonzero(new_mask)), int(np.count_nonzero(v.layer_mask)))
        v.layer_mask = new_mask
        exp_ids, exp_mask = ref_ensure(v)
        self.full_checks = 0
        v._ensure_layer_sync()
        self.assertEqual(self.full_checks, 1)
        np.testing.assert_array_equal(v.layer_ids, exp_ids)
        self.assertTrue(np.all(v.layer_ids[6000:6500] == V.LAYER_NOISE) and np.all(v.layer_ids[1000:1500] == V.LAYER_NORMAL))
        self.assert_truly_synced("layer_mask を、同じ数のまま差し替えた")

        # layer_ids だけを配列ごと差し替えた形: 以前と同じく、layer_mask に合わせて直す
        v.layer_ids = np.full(20000, V.LAYER_NORMAL, dtype=np.int32)
        exp_ids, exp_mask = ref_ensure(v)
        self.full_checks = 0
        v._ensure_layer_sync()
        self.assertEqual(self.full_checks, 1)
        np.testing.assert_array_equal(v.layer_ids, exp_ids)
        self.assertTrue(np.all(v.layer_ids[6000:6500] == V.LAYER_NOISE))
        self.assert_truly_synced("layer_ids を差し替えた")

    def test_counts_follow_a_replaced_mask_without_an_explicit_check(self):
        # 点数を聞くだけで、差し替えられた layer_mask に layer_ids が合わせられること（以前と同じ）
        v = self.v
        self.load(n=20000, seed=21)
        self.assertEqual(v.get_layer_counts()[V.LAYER_NOISE], 0)
        new_mask = v.layer_mask.copy()
        new_mask[300:1000] = False
        v.layer_mask = new_mask
        self.assertEqual(v.get_layer_counts()[V.LAYER_NOISE], 700)
        self.assertEqual(v.get_layer_counts(), ref_layer_counts(v))

    def test_noise_layer_zero_counts_as_noise_even_if_it_is_not_listed(self):
        v = self.v
        self.load(n=6000, seed=22)
        v.move_indices_to_noise(np.arange(100, 400, dtype=np.int64))
        del v.layers[V.LAYER_NOISE]                                  # レイヤーの一覧に、ノイズレイヤー 0 が無い形
        self.assertIn(V.LAYER_NOISE, v._noise_layer_ids())
        v._sync_mask_from_ids()
        np.testing.assert_array_equal(v.layer_mask, ref_mask_from_ids(v))
        self.assertEqual(int(np.count_nonzero(~v.layer_mask)), 300)

    def test_mask_written_in_place_is_noticed(self):
        v = self.v
        self.load(n=20000, seed=7)
        v.layer_mask[100:300] = False                                # 配列はそのままで、中身だけ書き換える
        exp_ids, _ = ref_ensure(v)
        v._ensure_layer_sync()
        self.assertEqual(self.full_checks, 1, "True の数が変わったので、全点を比べる")
        np.testing.assert_array_equal(v.layer_ids, exp_ids)
        self.assertTrue(np.all(v.layer_ids[100:300] == V.LAYER_NOISE))
        self.assert_truly_synced("その場で書き換えたあと")

    def test_changed_noise_layers_are_noticed(self):
        v = self.v
        self.load(n=20000, seed=8)
        lid = v.add_custom_layer("あとでノイズ扱いにするレイヤー")
        v.move_indices_to_layer(np.arange(3000, 4000, dtype=np.int64), lid)
        v._ensure_layer_sync()
        self.full_checks = 0
        v.layers[lid].is_noise = True                                # 設定だけ変えて、layer_mask を作り直していない
        exp_ids, exp_mask = ref_ensure(v)
        v._ensure_layer_sync()
        self.assertEqual(self.full_checks, 1)
        np.testing.assert_array_equal(v.layer_ids, exp_ids)          # 以前の作りと同じ: layer_mask を正として、layer_ids を直す
        np.testing.assert_array_equal(v.layer_mask, exp_mask)
        v.rename_layer(lid, "自動除去: 名前で決まるノイズ")
        self.assert_truly_synced("名前を変えたあと")

    def test_force_always_compares(self):
        v = self.v
        self.load(n=20000, seed=9)
        v._ensure_layer_sync(force=True)
        v._ensure_layer_sync(force=True)
        self.assertEqual(self.full_checks, 2)
        # layer_ids をその場で書き換えて _sync_mask_from_ids を呼ばなかった形: force=True なら、以前と同じく layer_mask に合わせて直す
        v.layer_ids[700:900] = V.LAYER_NOISE
        exp_ids, _ = ref_ensure(v)
        v._ensure_layer_sync(force=True)
        np.testing.assert_array_equal(v.layer_ids, exp_ids)
        self.assertTrue(np.all(v.layer_ids[700:900] == V.LAYER_NORMAL))
        self.assert_truly_synced("force のあと")

    def test_missing_or_short_arrays_are_rebuilt(self):
        v = self.v
        self.load(n=6000, seed=10)
        v.move_indices_to_noise(np.arange(100, 300, dtype=np.int64))
        mask = v.layer_mask.copy()
        v.layer_ids = None                                           # layer_mask だけが残っている形
        v._ensure_layer_sync()
        np.testing.assert_array_equal(v.layer_ids, np.where(mask, V.LAYER_NORMAL, V.LAYER_NOISE))
        self.assertEqual(v.layer_ids.dtype, np.int32)
        self.assert_truly_synced("layer_ids を作り直したあと")
        v.layer_ids = None
        v.layer_mask = None
        v._ensure_layer_sync()
        self.assertTrue(np.all(v.layer_ids == V.LAYER_NORMAL) and np.all(v.layer_mask))
        self.assert_truly_synced("両方を作り直したあと")

    def test_point_count_change_rebuilds_the_arrays(self):
        # 点の数だけが変わった形（layer_ids / layer_mask の長さが合わない）: 以前と同じく、作り直す
        v = self.v
        self.load(n=6000, seed=23)
        v.move_indices_to_noise(np.arange(100, 300, dtype=np.int64))
        v.points_raw = np.vstack([v.points_raw, np.zeros((2, 3))])
        exp_ids, exp_mask = ref_ensure(v)
        v._ensure_layer_sync()
        self.assertEqual(self.full_checks, 1)
        self.assertEqual(len(v.layer_ids), 6002)
        np.testing.assert_array_equal(v.layer_ids, exp_ids)
        np.testing.assert_array_equal(v.layer_mask, exp_mask)
        v.points_raw = v.points_raw[:6000]

    def test_ids_written_without_a_sync_are_caught_by_the_invariants(self):
        # 決まり破り（layer_ids をその場で書いて _sync_mask_from_ids を呼ばない）は、品質ゲートの不変条件の確認が見つける
        v = self.v
        self.load(n=6000, seed=24)
        v.validate_invariants()
        v.layer_ids[700:900] = V.LAYER_NOISE
        with self.assertRaises(AssertionError):
            v.validate_invariants()
        v._sync_mask_from_ids()
        v.validate_invariants()
        # 書き換えの途中で失敗したときの形: 覚えを捨てれば、次の確認が全点を比べる（以前と同じく、layer_mask に合わせて直す）
        v.layer_ids[1000:1100] = V.LAYER_NOISE
        v._forget_layer_sync()
        self.full_checks = 0
        v._ensure_layer_sync()
        self.assertEqual(self.full_checks, 1)
        self.assertTrue(np.all(v.layer_ids[1000:1100] == V.LAYER_NORMAL))
        self.assert_truly_synced("覚えを捨てたあと")

    def test_batch_move_into_a_layer_created_on_the_way(self):
        # 一括の移動の途中でレイヤーが足されると、一覧の作り直し（点数の数え直し）が、書きかけの layer_ids を見る
        v = self.v
        self.load(n=6000, seed=25)
        v.layer_list_changed.connect(lambda: v.get_layer_counts())           # 画面のレイヤー一覧と同じ動き
        moved = v.batch_move_indices_to_layers([(np.arange(0, 300, dtype=np.int64), V.LAYER_NOISE), (np.arange(500, 600, dtype=np.int64), 9)])
        self.assertEqual(moved, 400)
        self.assertTrue(np.all(v.layer_ids[0:300] == V.LAYER_NOISE), "先に移した点が、途中の確認で取り消されてはいけない")
        self.assertTrue(np.all(v.layer_ids[500:600] == 9))
        self.assert_truly_synced("途中でレイヤーが足された")

    def test_a_new_point_cloud_is_compared_and_old_arrays_are_released(self):
        import weakref

        v = self.v
        self.load(n=6000, seed=11)
        v.move_indices_to_noise(np.arange(100, 300, dtype=np.int64))
        v.get_layer_counts()
        v.get_scan_counts()
        old_mask, old_ids, old_scans = weakref.ref(v.layer_mask), weakref.ref(v.layer_ids), weakref.ref(v.scan_ids)
        pts, colors, intens, scan_ids = make_cloud(6000, seed=12)    # 同じ点数の、別の点群
        v.set_point_cloud(pts, colors, intens, scan_ids=scan_ids)
        del pts, colors, intens, scan_ids
        gc.collect()
        self.assertIsNone(old_mask(), "前の点群の layer_mask を、持ち続けない")
        self.assertIsNone(old_ids())
        self.assertIsNone(old_scans())
        self.assertIs(v._layer_sync_key[0](), v.layer_mask)
        self.assertIs(v._layer_sync_key[1](), v.layer_ids)
        self.assertEqual(v.get_layer_counts(), {V.LAYER_NORMAL: 6000, V.LAYER_NOISE: 0})
        self.assertEqual(v.get_scan_counts(), ref_scan_counts(v))
        self.assert_truly_synced("新しい点群")
        # 点群を空にしたとき: 何も聞かれなくても、前の配列を持ち続けない（覚えは、弱い参照）
        v.move_indices_to_noise(np.arange(100, 300, dtype=np.int64))
        v.get_layer_counts()
        v.get_scan_counts()
        old_mask, old_ids, old_scans = weakref.ref(v.layer_mask), weakref.ref(v.layer_ids), weakref.ref(v.scan_ids)
        v.set_point_cloud(None)
        gc.collect()
        self.assertIsNone(old_mask(), "空にしたあとも、前の点群の layer_mask を持っている")
        self.assertIsNone(old_ids())
        self.assertIsNone(old_scans())
        v._ensure_layer_sync()
        self.assertIsNone(v.layer_mask)
        self.assertIsNone(v.layer_ids)
        self.assertEqual(v.get_scan_counts(), {})


class TestCounts(_ViewerCase):
    def test_counts_match_and_are_not_recounted(self):
        v = self.v
        self.load(n=20000, seed=13)
        v._sync_mask_from_ids()                                      # 覚えを捨てさせる
        first = v.get_layer_counts()                                 # 数えた直後の辞書
        self.assertEqual(first, ref_layer_counts(v))
        first[V.LAYER_NORMAL] = -1                                   # 返した辞書を書き換えても、覚えは壊れない
        self.assertEqual(v.get_layer_counts(), ref_layer_counts(v))
        cache = v._layer_counts_cache
        got = v.get_layer_counts()
        self.assertIs(v._layer_counts_cache, cache, "変わっていなければ、数え直さない")
        got[V.LAYER_NORMAL] = -1
        self.assertEqual(v.get_layer_counts(), ref_layer_counts(v))

        v.move_indices_to_noise(np.arange(100, 900, dtype=np.int64))
        self.assertEqual(v.get_layer_counts(), ref_layer_counts(v))
        self.assertIsNot(v._layer_counts_cache, cache, "点を移したら、数え直す")
        self.assertEqual(v.get_layer_counts()[V.LAYER_NOISE], 800)

        lid = v.add_custom_layer("まだ空のレイヤー")
        self.assertEqual(v.get_layer_counts(), ref_layer_counts(v))
        self.assertEqual(v.get_layer_counts()[lid], 0)

        # 元に戻す（gui/layer_history.py）と同じ形: layer_ids をその場で書き換えて、_sync_mask_from_ids
        v.layer_ids[100:500] = V.LAYER_NORMAL
        v._sync_mask_from_ids()
        self.assertEqual(v.get_layer_counts(), ref_layer_counts(v))
        self.assertEqual(v.get_layer_counts()[V.LAYER_NOISE], 400)

        # self.layers に無い番号の点は、数に出さない（以前と同じ）
        v.layer_ids[1000:1010] = 77
        v._sync_mask_from_ids()
        self.assertEqual(v.get_layer_counts(), ref_layer_counts(v))
        self.assertNotIn(77, v.get_layer_counts())

    def test_many_layers_use_the_sorting_path(self):
        v = self.v
        self.load(n=20000, seed=14)
        rng = np.random.default_rng(15)
        for k in range(LS.COUNT_COMPARE_MAX + 3):
            lid = v.add_custom_layer(f"レイヤー {k}")
            v.move_indices_to_layer(rng.choice(20000, 500, replace=False), lid)
            self.assertEqual(v.get_layer_counts(), ref_layer_counts(v), f"レイヤー {len(v.layers)} 個")
        self.assertGreater(len(v.layers), LS.COUNT_COMPARE_MAX)
        self.assert_truly_synced("レイヤーが多いとき")

    def test_scan_counts_are_cached_per_point_cloud(self):
        v = self.v
        self.load(n=20000, seed=16)
        self.assertEqual(v.get_scan_counts(), ref_scan_counts(v))
        self.assertEqual(len(v.get_scan_counts()), 4)
        cache = v._scan_counts_cache
        got = v.get_scan_counts()
        self.assertIs(v._scan_counts_cache, cache)
        got.clear()
        self.assertEqual(v.get_scan_counts(), ref_scan_counts(v))
        v.move_indices_to_noise(np.arange(100, 900, dtype=np.int64))  # レイヤーを移しても、スキャンごとの点数は変わらない
        self.assertIs(v._scan_counts_cache, cache)
        self.load(n=20000, seed=17)
        self.assertEqual(v.get_scan_counts(), ref_scan_counts(v))
        self.assertIsNot(v._scan_counts_cache, cache)

    def test_emitted_counts(self):
        v = self.v
        self.load(n=20000, seed=18)
        seen = []
        v.layers_updated.connect(lambda a, b: seen.append((a, b)))
        v.move_indices_to_noise(np.arange(100, 900, dtype=np.int64))
        self.assertEqual(seen[-1], (19200, 800))
        lid = v.add_custom_layer("自動除去: テスト", is_noise=True)
        v.move_indices_to_layer(np.arange(5000, 5300, dtype=np.int64), lid)
        self.assertEqual(seen[-1], (18900, 1100), "ノイズ扱いのレイヤーの点も、ノイズに数える")
        self.assertEqual(seen[-1], (int(np.sum(v.layer_mask)), int(np.sum(~v.layer_mask))))


class TestRandomOperations(_ViewerCase):
    def test_cached_state_is_always_really_in_sync(self):
        rng = np.random.default_rng(2026)
        v = self.v
        n = 15000
        self.load(n=n, seed=19)
        lids = [V.LAYER_NORMAL, V.LAYER_NOISE, v.add_custom_layer("表示するレイヤー")]
        for step in range(80):
            kind = int(rng.integers(0, 9))
            start = int(rng.integers(0, n - 600))
            idx = np.arange(start, start + int(rng.integers(1, 500)), dtype=np.int64)
            if kind <= 2:
                v.move_indices_to_layer(idx, lids[int(rng.integers(0, len(lids)))])
            elif kind == 3:
                v.batch_move_indices_to_layers([(idx, V.LAYER_NOISE), (idx + 300, lids[-1])])
            elif kind == 4 and len(lids) < 9:
                lids.append(v.add_custom_layer(f"自動除去: {step}" if rng.random() < 0.5 else f"レイヤー {step}",
                                               is_noise=bool(rng.random() < 0.3)))
            elif kind == 5:
                new_mask = v.layer_mask.copy()                       # 外から、配列ごと差し替える
                new_mask[idx] = ~new_mask[idx]
                v.layer_mask = new_mask
                exp_ids, exp_mask = ref_ensure(v)
                v._ensure_layer_sync()
                np.testing.assert_array_equal(v.layer_ids, exp_ids, err_msg=f"{step} 手目")
                np.testing.assert_array_equal(v.layer_mask, exp_mask, err_msg=f"{step} 手目")
            elif kind == 6:
                v.layer_ids[idx] = lids[int(rng.integers(0, len(lids)))]   # 元に戻す、と同じ形
                v._sync_mask_from_ids()
            elif kind == 7:
                v.set_layer_locked(lids[-1], bool(rng.random() < 0.5))
            else:
                v._apply_selection_with_modifiers(idx, "テスト", Qt.NoModifier)
            if kind == 4:
                v._ensure_layer_sync()                               # ノイズ扱いのレイヤーが増えた直後の 1 回は、全点を比べる
            before = self.full_checks
            v._ensure_layer_sync()
            v._get_selectable_mask()
            v.get_layer_counts()
            self.assertEqual(self.full_checks, before, f"{step} 手目（種類 {kind}）: 操作のあとの確認で、全点を比べ直している")
            self.assert_truly_synced(f"{step} 手目（種類 {kind}）")


class TestWritersAreKnown(unittest.TestCase):
    """
    layer_ids / layer_mask をその場で書き換える所が、知らないうちに増えていないこと。
    増やすときは、書き換えのあとに必ず _sync_mask_from_ids() を呼び（layer_mask は、配列ごと差し替える）、ここの数を直す。
    """

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    IDS_WRITES = {"gui/gl_viewer.py": 4, "gui/layer_history.py": 2}

    def _sources(self):
        for folder in ("gui", "core"):
            for name in sorted(os.listdir(os.path.join(self.ROOT, folder))):
                if name.endswith(".py"):
                    rel = f"{folder}/{name}"
                    with open(os.path.join(self.ROOT, folder, name), encoding="utf-8") as f:
                        yield rel, f.read()

    def test_in_place_writes_are_only_in_the_known_places(self):
        import re

        ids_write = re.compile(r"layer_ids\s*\[[^\]\n]*\]\s*(?:[-+*/|&^]?=)(?!=)")
        mask_write = re.compile(r"layer_mask\s*(?:\[[^\]\n]*\]\s*)?(?:[-+*/|&^]=|\[[^\]\n]*\]\s*=(?!=))")
        found_ids, found_mask = {}, {}
        for rel, text in self._sources():
            n_ids = len(ids_write.findall(text))
            n_mask = len([m for m in mask_write.finditer(text) if "[" in m.group(0) or m.group(0).rstrip()[-2] in "-+*/|&^"])
            if n_ids:
                found_ids[rel] = n_ids
            if n_mask:
                found_mask[rel] = n_mask
        self.assertEqual(found_ids, self.IDS_WRITES,
                         "layer_ids をその場で書き換える所が変わりました。書き換えのあとに _sync_mask_from_ids() を呼んでいるかを確かめて、この一覧を直してください")
        self.assertEqual(found_mask, {}, "layer_mask は、その場で書き換えず、配列ごと差し替えてください（gui/layer_sync.py）")

    def test_saving_always_compares_every_point(self):
        import inspect

        from gui.main_window import MainWindow

        self.assertIn("_ensure_layer_sync(force=True)", inspect.getsource(MainWindow._save_current_scans))
        self.assertIn("_ensure_layer_sync(force=True)", inspect.getsource(PointCloudViewer.save_layer_project))

    def test_undo_and_redo_forget_the_state_when_they_fail(self):
        import inspect

        from gui.main_window import MainWindow

        for meth in (MainWindow._undo_action, MainWindow._redo_action):
            self.assertIn("_forget_layer_sync()", inspect.getsource(meth), meth.__name__)


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
        self.full_checks = 0
        self._orig = win.viewer._check_layer_sync

        def counted():
            self.full_checks += 1
            return self._orig()

        win.viewer._check_layer_sync = counted

    def tearDown(self):
        del self.win.viewer._check_layer_sync
        self.win._toggle_instant_delete_mode(False)

    def check(self, where):
        win, v = self.win, self.win.viewer
        np.testing.assert_array_equal(v.layer_mask, ref_mask_from_ids(v), err_msg=where)
        n_norm = int(np.count_nonzero(v.layer_mask))
        self.assertEqual(win.lbl_layer_normal_count.text(), f"{n_norm:,} 点", where)
        self.assertEqual(win.lbl_layer_noise_count.text(), f"{len(v.layer_mask) - n_norm:,} 点", where)
        self.assertEqual(v.get_layer_counts(), ref_layer_counts(v), where)
        ids_before = v.layer_ids.copy()
        v._ensure_layer_sync(force=True)
        np.testing.assert_array_equal(v.layer_ids, ids_before, err_msg=f"全点を比べ直したら layer_ids が変わった: {where}")
        v.validate_invariants()

    def test_delete_undo_redo_and_instant_erase(self):
        win, v = self.win, self.win.viewer
        v._apply_selection_with_modifiers(np.arange(100, 900, dtype=np.int64), "テスト", Qt.NoModifier)
        win._delete_selected_points()
        self.check("Del")
        win._undo_action()
        self.check("元に戻す")
        win._redo_action()
        self.check("やり直し")
        win._toggle_instant_delete_mode(True)
        v._apply_selection_with_modifiers(np.arange(5000, 5600, dtype=np.int64), "テスト", Qt.NoModifier)
        self.check("即削除")
        win._undo_action()
        self.check("即削除を元に戻す")

    def test_operations_do_not_compare_every_point(self):
        win, v = self.win, self.win.viewer
        v._ensure_layer_sync()
        self.full_checks = 0
        win._toggle_instant_delete_mode(True)
        for start in (2000, 9000, 15000):
            v._apply_selection_with_modifiers(np.arange(start, start + 600, dtype=np.int64), "テスト", Qt.NoModifier)
        win._undo_action()
        win._undo_action()
        win._redo_action()
        self.assertEqual(self.full_checks, 0, "消す・戻す・やり直しでは、全点の比べは要らない")
        self.assertEqual(v.get_layer_counts()[V.LAYER_NOISE], 1200)

    def test_with_an_auto_removed_noise_layer(self):
        # 自動フィルタの結果（「自動除去:」のノイズレイヤー）があるときの、手作業の消す・戻す
        win, v = self.win, self.win.viewer
        win._apply_edge_veil_result(np.arange(12000, 13000, dtype=np.int64))
        self.assertEqual(len(ref_noise_ids(v)), 2)
        self.check("自動除去のあと")
        v._apply_selection_with_modifiers(np.arange(100, 900, dtype=np.int64), "テスト", Qt.NoModifier)
        win._delete_selected_points()
        self.check("Del")
        self.assertEqual(win.lbl_layer_noise_count.text(), "1,800 点")
        win._undo_action()
        self.check("元に戻す")
        win._undo_action()
        self.check("自動除去も元に戻す")
        self.assertEqual(win.lbl_layer_noise_count.text(), "0 点")

    def test_failed_undo_falls_back_to_a_full_compare(self):
        # 元に戻す処理が、layer_ids を書き換えたあとで失敗した形（メモリ不足など）
        win, v = self.win, self.win.viewer
        v._apply_selection_with_modifiers(np.arange(100, 900, dtype=np.int64), "テスト", Qt.NoModifier)
        win._delete_selected_points()
        orig = win.layer_history._take_shadow
        calls = {"n": 0}

        def failing(state):
            calls["n"] += 1
            if calls["n"] == 1:
                raise MemoryError("テスト用の失敗")
            return orig(state)

        win.layer_history._take_shadow = failing
        try:
            with self.assertRaises(MemoryError):
                win._undo_action()
        finally:
            del win.layer_history._take_shadow
        self.assertIsNone(v._layer_sync_key, "失敗したら、覚えを捨てる")
        self.full_checks = 0
        v._ensure_layer_sync()
        self.assertEqual(self.full_checks, 1)
        np.testing.assert_array_equal(v.layer_mask, ref_mask_from_ids(v), err_msg="layer_mask と layer_ids が、食い違ったまま残っている")
        self.assertEqual(v.get_layer_counts(), ref_layer_counts(v))
        v._update_active_colors()
        v._update_display_arrays()
        v.validate_invariants()

    def test_batch_move_like_the_auto_filters(self):
        win, v = self.win, self.win.viewer
        win._push_layer_undo_state()
        lid = v.add_custom_layer("自動除去: SOR", is_noise=True)
        lid2 = v.add_custom_layer("地面")
        v.batch_move_indices_to_layers([(np.arange(1000, 1400, dtype=np.int64), lid), (np.arange(3000, 3600, dtype=np.int64), lid2)])
        self.check("自動処理の結果をまとめて反映")
        self.assertEqual(win.lbl_layer_noise_count.text(), "400 点")
        win._undo_action()
        self.check("元に戻す")
        self.assertEqual(win.lbl_layer_noise_count.text(), "0 点")
        win._redo_action()
        self.check("やり直し")

    def test_restoring_a_mask_snapshot(self):
        win, v = self.win, self.win.viewer
        v.move_indices_to_noise(np.arange(100, 900, dtype=np.int64))
        snap = v.layer_mask.copy()
        v.restore_indices_to_normal(np.arange(100, 900, dtype=np.int64))
        v.move_indices_to_noise(np.arange(4000, 4300, dtype=np.int64))
        win._restore_layer_snapshot(snap)                            # 後方互換の形（真偽の配列だけ）
        self.assertEqual(int(np.count_nonzero(v.layer_ids == V.LAYER_NOISE)), 800)
        self.assertTrue(np.all(v.layer_ids[100:900] == V.LAYER_NOISE))
        self.check("真偽の配列から戻す")


if __name__ == "__main__":
    unittest.main()
