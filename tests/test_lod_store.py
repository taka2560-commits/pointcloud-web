"""
LOD ストア（core.lod_store）のテスト
- キャッシュ一括作成（ensure_section_cache）→ LOD 作成で、全点が 1 回ずつ（重複・欠落なし）収まること
- レベルの点数がほぼ 1/2 ずつになり、レベルを足すほど密になる（入れ子）こと
- 葉の点はその葉の箱の中にあり、葉の合計が全点数になること
- 視点による選択: 画面外の葉は描かない、予算を超えない、寄るほど細かいレベルになる、断面の箱に掛からない葉は描かない
- ビューア座標（座標系変換・中心オフセット）での読み出しと、箱での絞り込み
- 元ファイルが変わると開けない、中止で部分ファイルが残らない、キャッシュが無ければ作れない
- 非表示の点（ノイズ）の格子による隠し
"""

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import numpy as np

from core.e57_io import read_e57_scans
from core.lod_store import (
    LodScene,
    LodStore,
    build_lod_store,
    hidden_voxel_sets,
    level_count_for,
    lod_paths,
    morton_decode_prefix,
    morton_encode,
    not_hidden_mask,
    point_levels,
    remove_lod,
    select_leaf_levels,
)
from core.processor import transform_scan_coordinate_system
from core.section_cache import remove_cache
from core.section_reload import SectionBox, ensure_section_cache
from test_section_reload import _Scene, _as_set, _display_truth

BASE = 2_000      # レベル 0 の目標点数（57,803 点 → 6 レベル）
LEAF_MAX = 3_000  # 小さな葉に割る
CHUNK = 5_000


def _ortho_mvp(center, half_w, half_h, depth=1000.0):
    """真上から見下ろす正射影（x 右・y 上）。clip = mvp @ [x, y, z, 1]"""
    m = np.eye(4)
    m[0, 0] = 1.0 / half_w
    m[1, 1] = 1.0 / half_h
    m[2, 2] = -1.0 / depth
    m[0, 3] = -center[0] / half_w
    m[1, 3] = -center[1] / half_h
    m[2, 3] = center[2] / depth
    return m


class TestLodPrimitives(unittest.TestCase):
    def test_morton_roundtrip(self):
        rng = np.random.default_rng(1)
        ijk = rng.integers(0, 128, size=(1000, 3))
        m = morton_encode(ijk[:, 0], ijk[:, 1], ijk[:, 2])
        self.assertTrue(np.all((m >= 0) & (m < (1 << 21))))
        for (ix, iy, iz), code in list(zip(ijk.tolist(), m.tolist()))[:50]:
            self.assertEqual(morton_decode_prefix(code, 7), (ix, iy, iz))
        # 上位の接頭辞は 1 つ上の深さの格子番号になる
        self.assertEqual(morton_decode_prefix(int(m[0]) >> 3, 6), tuple(int(v) >> 1 for v in ijk[0]))

    def test_level_assignment_is_deterministic_and_halves(self):
        n = 400_000
        lv = point_levels(0, n, 6)
        np.testing.assert_array_equal(lv, np.concatenate([point_levels(0, 100_000, 6), point_levels(100_000, 300_000, 6)]))
        counts = np.bincount(lv, minlength=6)
        self.assertEqual(counts.sum(), n)
        for level, expect in zip(range(6), (1 / 32, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2)):
            self.assertAlmostEqual(counts[level] / n, expect, delta=expect * 0.08)
        self.assertEqual(level_count_for(57_803, 2_000), 6)
        self.assertEqual(level_count_for(1_000, 2_000), 1)
        self.assertEqual(level_count_for(538_377_435, 1_500_000), 10)

    def test_hidden_voxels(self):
        origin = np.zeros(3)
        # (0,0,0) 付近: 表示中と非表示が同じ 5cm 格子 → 隠さない。(1,1,1) 付近: 非表示だけ → 隠す。
        # (3,3,3) 付近: 非表示が 1 点だけ。同じ 80cm 格子に表示中の点は無い → 周り（読み込まれなかった仲間）も隠す。
        # (5,5,5): 表示中だけ → 隠さない
        pts = np.array([
            [0.01, 0.01, 0.01], [0.02, 0.02, 0.02], [1.0, 1.0, 1.0], [1.01, 1.0, 1.0], [3.0, 3.0, 3.0], [5.0, 5.0, 5.0]
        ])
        visible = np.array([True, False, False, False, False, True])
        sets = hidden_voxel_sets(pts, visible, origin)
        self.assertEqual([v for v, _ in sets], [0.05, 0.2, 0.8, 3.2])
        self.assertEqual(len(sets[3][1]), 0, "3.2m 格子には表示中の点も入るので、隠す格子は無い")
        self.assertEqual(len(sets[0][1]), 2, "5cm では (1,1,1) と (3,3,3) の 2 格子。(0,0,0) の格子は表示中の点があるので隠さない")
        probe = np.array([
            [1.02, 1.02, 1.02],   # 5cm 格子で隠れる
            [0.03, 0.03, 0.03],   # 表示中の点と同じ格子 → 残す
            [2.9, 2.9, 2.9],      # 5cm・20cm では別の格子だが、80cm 格子に非表示の点しか無い → 隠れる
            [5.0, 5.0, 5.0],      # 表示中の点の格子 → 残す
            [9.0, 9.0, 9.0],      # 読み込み済みの点が無い領域 → 残す
        ], dtype=np.float32)
        np.testing.assert_array_equal(not_hidden_mask(probe, sets, origin), [False, True, False, True, True])
        np.testing.assert_array_equal(not_hidden_mask(probe, sets[:1], origin), [False, True, True, True, True])
        self.assertEqual(hidden_voxel_sets(pts, np.ones(6, dtype=bool), origin), [])
        self.assertTrue(np.all(not_hidden_mask(probe, [], origin)))
        self.assertTrue(np.all(not_hidden_mask(probe, None, origin)))


class TestLodStore(_Scene):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        remove_cache(cls.path)
        cls.scans = read_e57_scans(cls.path, point_budget=cls.load_budget)
        info = ensure_section_cache(cls.scans, chunk_size=CHUNK)
        assert info["built"] == 3, info
        cls.summary = build_lod_store(cls.scans, base_points=BASE)[0]
        cls.store = LodStore.open(cls.path, leaf_max_points=LEAF_MAX)
        assert cls.store is not None
        # 正解: pose 適用後のワールド座標（座標系変換なし）
        cls.world, cls.world_cols, cls.world_ints, _ = _display_truth(cls.full)

    @classmethod
    def tearDownClass(cls):
        cls.store.close()
        cls.store = None
        remove_cache(cls.path)
        super().tearDownClass()

    def setUp(self):
        pass  # _Scene の「毎回キャッシュを消す」を止める（クラスで 1 回だけ作る）

    def tearDown(self):
        pass

    def test_all_points_exactly_once(self):
        st = self.store
        self.assertEqual(self.summary["n_points"], self.n_total)
        self.assertEqual(st.n_points, self.n_total)
        self.assertEqual(st.n_levels, 6)
        xyz, rgb, inten = st.gather(np.full(st.n_leaves, st.n_levels - 1))
        self.assertEqual(len(xyz), self.n_total, "全ての葉を最細レベルで読むと全点になる")
        world = xyz.astype(np.float64) + st.origin
        got = np.round(world, 3)
        want = np.round(self.world, 3)
        want_rgb = np.rint(self.world_cols * 255).astype(np.uint8)
        # 床の端と壁の下端は同じ座標の点が重なるので、色も並べ替えの鍵に入れて対応を一意にする
        order_g = np.lexsort((rgb[:, 2], rgb[:, 1], rgb[:, 0], got[:, 2], got[:, 1], got[:, 0]))
        order_w = np.lexsort((want_rgb[:, 2], want_rgb[:, 1], want_rgb[:, 0], want[:, 2], want[:, 1], want[:, 0]))
        np.testing.assert_array_equal(got[order_g], want[order_w])
        # 色・強度も同じ点のもの（色は 8bit そのまま、強度は float16）
        np.testing.assert_array_equal(rgb[order_g], want_rgb[order_w])
        np.testing.assert_allclose(inten[order_g].astype(np.float32), self.world_ints[order_w], rtol=2e-3, atol=1e-3)

    def test_levels_halve_and_nest(self):
        st = self.store
        sizes = np.diff(np.concatenate([[0], st.leaf_cum.sum(axis=0)]))
        self.assertEqual(int(sizes.sum()), self.n_total)
        for level in range(1, st.n_levels):
            expect = self.n_total / (2 ** (st.n_levels - level))
            self.assertAlmostEqual(sizes[level], expect, delta=expect * 0.15, msg=f"レベル {level}")
        prev = set()
        for level in range(st.n_levels):
            xyz, _, _ = st.gather(np.full(st.n_leaves, level))
            cur = _as_set(xyz.astype(np.float64) + st.origin, 3)
            self.assertTrue(prev <= cur, "粗いレベルの点は細かいレベルにも含まれる（足していく方式）")
            self.assertEqual(len(xyz), int(st.leaf_cum[:, level].sum()))
            prev = cur

    def test_leaf_points_are_inside_leaf_box(self):
        st = self.store
        self.assertGreater(st.n_leaves, 10)
        self.assertEqual(int(st.leaf_cum[:, -1].sum()), self.n_total)
        self.assertTrue(np.all(st.leaf_cum[:, -1] > 0))
        rng = np.random.default_rng(0)
        for leaf in rng.choice(st.n_leaves, size=min(12, st.n_leaves), replace=False):
            lv = np.full(st.n_leaves, -1)
            lv[leaf] = st.n_levels - 1
            xyz, _, _ = st.gather(lv)
            self.assertEqual(len(xyz), int(st.leaf_cum[leaf, -1]))
            world = xyz.astype(np.float64) + st.origin
            self.assertTrue(np.all(world >= st.leaf_min[leaf] - 1e-4))
            self.assertTrue(np.all(world <= st.leaf_max[leaf] + 1e-4))
        self.assertEqual(st.count_for(np.full(st.n_leaves, -1)), 0)

    def test_select_frustum_budget_and_zoom(self):
        scene = LodScene([self.store], "cartesian")
        center = np.array([2.0, 1.5, 0.0])
        scene.prepare(center)
        # スキャン A（床 4×3m）の周りを真上から。B (100,200) と C (-10..-8) は画面外
        mvp_wide = _ortho_mvp((0.0, 0.0, 0.0), 4.0, 3.0)
        lv, info = scene.select(mvp_wide, (800, 600), budget=10_000_000, px_density=1.0)
        cw = scene.corners_view + center
        in_a = (cw[:, :, 0].min(axis=1) < 4.5) & (cw[:, :, 0].max(axis=1) > -0.5) & (cw[:, :, 1].max(axis=1) > -0.5) & (cw[:, :, 1].min(axis=1) < 3.5)
        self.assertTrue(np.all(lv[~in_a] == -1), "画面の外の葉は描かない")
        self.assertTrue(np.any(lv[in_a] >= 0))
        self.assertEqual(int(info["n_points"]), self.store.count_for(lv))

        # 予算を絞ると超えない
        lv_small, info_small = scene.select(mvp_wide, (800, 600), budget=3_000, px_density=1.0)
        self.assertLessEqual(self.store.count_for(lv_small), 3_000)
        self.assertGreater(self.store.count_for(lv_small), 0)
        self.assertLess(info_small["scale"], 1.0)

        # 低い目標密度の広い画面より、同じ密度で 0.5m 四方へ寄った方が、画面内の葉は細かいレベルになる
        lv_coarse, _ = scene.select(mvp_wide, (800, 600), budget=10_000_000, px_density=0.002)
        mvp_zoom = _ortho_mvp((-0.5, -0.5, 0.0), 0.5, 0.375)
        lv_zoom, _ = scene.select(mvp_zoom, (800, 600), budget=10_000_000, px_density=0.002)
        seen = lv_zoom >= 0
        self.assertTrue(np.any(seen))
        self.assertTrue(np.all(lv_zoom[seen] >= lv_coarse[seen]))
        self.assertGreater(lv_zoom[seen].max(), lv_coarse[seen].min())
        self.assertLess(np.count_nonzero(seen), np.count_nonzero(lv_coarse >= 0))

    def test_gather_view_with_box_and_coord_mode(self):
        """ビューア座標での読み出し: 箱で絞ると箱の中の元ファイルの点と一致する（survey_swap でも）"""
        for mode in ("cartesian", "survey_swap"):
            full = read_e57_scans(self.path)
            truth, _, _, _ = _display_truth(full, mode)
            scene = LodScene([self.store], mode)
            center = (truth.min(axis=0) + truth.max(axis=0)) / 2.0
            scene.prepare(center)
            box = SectionBox(x_range=(1.0, 2.0)) if mode == "cartesian" else SectionBox(y_range=(1.0, 2.0))
            mvp_all = _ortho_mvp((0.0, 0.0, 0.0), 500.0, 500.0)
            lv, _ = scene.select(mvp_all, (800, 600), budget=10_000_000, px_density=1e9, box=box)
            self.assertTrue(np.any(lv == -1), "箱に掛からない葉は描かない")
            pts, rgb, inten, n_before = scene.gather_view(lv, box=box)
            self.assertGreaterEqual(n_before, len(pts))
            # 箱の端（1.0 / 2.0 の格子点の列）は float32 の丸めで出入りするので、端から 5mm より内側で比べる
            inner = SectionBox(x_range=(1.005, 1.995)) if mode == "cartesian" else SectionBox(y_range=(1.005, 1.995))
            got = pts.astype(np.float64) + center
            self.assertEqual(_as_set(got[inner.contains(got)], 3), _as_set(truth[inner.contains(truth)], 3), mode)
            self.assertTrue(np.all(box.contains(got) | ~inner.contains(got)))
            self.assertEqual(len(rgb), len(pts))
            self.assertEqual(len(inten), len(pts))

    def test_open_fails_when_source_changes_and_scene_open(self):
        self.assertIsNotNone(LodScene.open_for_scans(self.scans, leaf_max_points=LEAF_MAX))
        st = os.stat(self.path)
        os.utime(self.path, ns=(st.st_atime_ns, st.st_mtime_ns + 3_000_000_000))
        try:
            self.assertIsNone(LodStore.open(self.path))
            self.assertIsNone(LodScene.open_for_scans(self.scans))
        finally:
            os.utime(self.path, ns=(st.st_atime_ns, st.st_mtime_ns))
        self.assertIsNotNone(LodStore.open(self.path))


class TestLodBuildFailures(_Scene):
    def test_requires_cache_then_cancel_leaves_nothing(self):
        scans = read_e57_scans(self.path, point_budget=self.load_budget)
        with self.assertRaises(RuntimeError):
            build_lod_store(scans, base_points=BASE)  # キャッシュ未作成
        ensure_section_cache(scans, chunk_size=CHUNK)
        again = ensure_section_cache(scans, chunk_size=CHUNK)
        self.assertEqual((again["built"], again["already"]), (0, 3))

        calls = []

        def cancelled():
            calls.append(1)
            return len(calls) > 14  # 2 回目の走査の途中で止める

        with self.assertRaises(InterruptedError):
            build_lod_store(scans, base_points=BASE, is_cancelled=cancelled)
        paths = lod_paths(self.path)
        folder = os.path.dirname(paths["lod.json"])
        self.assertFalse(any(n.startswith("lod") for n in os.listdir(folder)), os.listdir(folder))
        self.assertIsNone(LodStore.open(self.path))

        progress = []
        res = build_lod_store(scans, base_points=BASE, progress_callback=lambda r, m: progress.append((r, m)))
        self.assertEqual(res[0]["n_points"], self.n_total)
        self.assertTrue(all(0.0 <= r <= 1.0 for r, _ in progress))
        self.assertGreaterEqual(progress[-1][0], 0.99)
        self.assertIsNotNone(LodStore.open(self.path))
        remove_lod(self.path)
        self.assertIsNone(LodStore.open(self.path))
        self.assertTrue(os.path.isfile(os.path.join(folder, "header.json")), "読み直しキャッシュは残る")

    def test_ensure_cache_cancel(self):
        scans = read_e57_scans(self.path, point_budget=self.load_budget)
        calls = []

        def cancelled():
            calls.append(1)
            return len(calls) > 2

        with self.assertRaises(InterruptedError):
            ensure_section_cache(scans, chunk_size=CHUNK, is_cancelled=cancelled)
        info = ensure_section_cache(scans, chunk_size=CHUNK)
        self.assertEqual(info["built"] + info["already"], 3)


if __name__ == "__main__":
    unittest.main()
