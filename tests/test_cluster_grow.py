# -*- coding: utf-8 -*-
"""
連結塊の追跡（core/cluster_grow.py）と、画面の「連結塊を選択」のテスト

きっかけ（2026-11-14）: 利用者の報告「連結塊の機能を使うと途中でフリーズした」。5,490 万点の点群で、たどり始める前の
準備だけで約 51 秒（進み具合なし・中止不可）、建物につながる所を選ぶと数十分かかる作りだった。

たどり方（答えは以前と同じで、求め方だけを変えた）
- 点を 1 つずつたどる以前の作りと、結果が 1 点も違わないこと（連結距離・起点・窓の大きさをいろいろ変えて突き合わせる）
- 連結距離 eps 以内にある点は必ずつながり、eps を少しでも超えるすき間はまたがないこと
- 格子の代表点どうしが離れていても、ほかの点どうしが eps 以内ならつながること（行き止まりでの確かめ）
- 同じ塊なら、どの点から始めても、窓がどの大きさでも、同じ結果になること
- 窓に収まらない長い塊は、窓を広げて最後までたどること

地面の扱い（以前の作りと同じ決まり）
- 物の上を選んだら、地面を伝って広がらないこと。地面の上を選んだら、地面も塊として扱うこと
- 空中の高い所の塊は、底まで丸ごと選べること（窓の中のいちばん低い点を、地面と見まちがえない）
- 地面の判定そのものが、以前の作りと 1 点も違わないこと

安全の決まり
- 選べない点（非表示・ロック・断面の外）は、起点にも、橋渡しにも、結果にもならないこと
- 上限を超えたら capped=True で知らせること（画面は、選択も即削除もしない）
- 進み具合が出ること、中止できること

画面
- 小さな塊: 選択される。即削除モードではノイズへ移る
- 建物や地面につながった: 選択を変えず、何も消さず、理由と次の手を知らせる
- 進捗の窓の「中止」で止まり、選択は変わらない
"""
import inspect
import os
import sys
import threading
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.settings_sandbox import use_sandbox_settings  # noqa: E402
from tests.qt_cleanup import destroy_widget  # noqa: E402

use_sandbox_settings()  # この PC の本番アプリの保存設定を、テストで書き換えない

import numpy as np  # noqa: E402

from core import cluster_grow as CG  # noqa: E402
from core.cluster_grow import ClusterGrowth, ground_point_mask, grow_connected_cluster  # noqa: E402
from core.filters import extract_connected_cluster_from_seeds, segment_ground_points  # noqa: E402


# ---------------------------------------------------------------- 答え合わせ用（以前の作り）
def ref_ground_mask(points, grid_size, height_threshold):
    """2026-11-14 まで core/filters.py: segment_ground_points にあった判定（2 列の np.unique で格子を見分ける）"""
    grid_size = max(0.05, float(grid_size))
    height_threshold = max(0.001, float(height_threshold))
    xy_cells = np.floor(points[:, :2] / grid_size).astype(np.int64)
    _, inv = np.unique(xy_cells, axis=0, return_inverse=True)
    inv = np.asarray(inv).ravel()
    cell_min = np.full(int(inv.max()) + 1, np.inf, dtype=np.float64)
    np.minimum.at(cell_min, inv, points[:, 2])
    return (points[:, 2] - cell_min[inv]) <= height_threshold


def ref_cluster(points, seeds, radius, ground_eps=None, selectable=None):
    """
    厳密な連結: 半径 radius 以内の点を 1 つずつたどる（以前の作りと同じ。上限なし）。
    ground_eps を渡すと、以前の作りと同じ決まりで地面を伝わないようにする（地面の格子は ground_eps から決める）。
    """
    from scipy.spatial import cKDTree

    cand = np.arange(len(points)) if selectable is None else np.flatnonzero(selectable)
    pts = points[cand]
    s = np.searchsorted(cand, np.unique(seeds))
    allowed = np.ones(len(pts), dtype=bool)
    if ground_eps is not None and len(pts) > 50:
        ground = ref_ground_mask(pts, max(0.8, ground_eps * 4.0), 0.12)
        if 0 < int(ground.sum()) < len(pts) and float(np.mean(ground[s])) < 0.5:
            allowed = ~ground
            allowed[s] = True
    valid = np.flatnonzero(allowed)
    sub = pts[valid]
    tree = cKDTree(sub)
    visited = np.zeros(len(valid), dtype=bool)
    sl = np.searchsorted(valid, s)
    visited[sl] = True
    frontier = list(sl)
    while frontier:
        nxt = []
        for nbrs in tree.query_ball_point(sub[frontier], r=radius):
            for nb in nbrs:
                if not visited[nb]:
                    visited[nb] = True
                    nxt.append(nb)
        frontier = nxt
    return cand[valid[visited]]


# ---------------------------------------------------------------- 合成の点群
def ball(rng, center, radius, n):
    """球の中に一様にばらまいた点（どの点も、近くに仲間がいる）"""
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1)[:, None]
    return np.asarray(center, dtype=np.float64) + v * (radius * np.cbrt(rng.uniform(size=n)))[:, None]


class Scene:
    """床 + 床に立つ人 + 床から 9m 上の浮遊塊 + 壁"""

    def __init__(self, seed=0, n_floor=60000, extent=12.0):
        rng = np.random.default_rng(seed)
        floor = np.c_[rng.uniform(-extent, extent, (n_floor, 2)), rng.normal(0.0, 0.003, n_floor)]
        person = np.c_[rng.normal([3.0, 2.0], 0.10, (1500, 2)), rng.uniform(0.0, 1.7, 1500)]
        blob = ball(rng, [-4.0, 1.0, 9.0], 0.4, 800)
        wall = np.c_[rng.uniform(-extent, extent, 6000), extent + rng.normal(0, 0.003, 6000), rng.uniform(0.0, 5.0, 6000)]
        self.points = np.vstack([floor, person, blob, wall])
        bounds = np.cumsum([0, len(floor), len(person), len(blob), len(wall)])
        self.floor, self.person, self.blob, self.wall = (np.arange(bounds[i], bounds[i + 1]) for i in range(4))
        self.n = len(self.points)


def patchy_cloud(seed):
    """いくつかの塊と、まばらな背景"""
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-3.0, 3.0, (12, 3))
    parts = [c + rng.normal(0.0, rng.uniform(0.1, 0.4), (int(rng.integers(50, 400)), 3)) for c in centers]
    parts.append(rng.uniform(-4.0, 4.0, (1500, 3)))
    return np.vstack(parts), rng


class TestConnectivity(unittest.TestCase):
    def test_separate_blobs_are_not_merged(self):
        # 品質ゲート（quality_gate.py）と同じ形: 大きな塊と、25m 離れた小さな塊
        rng = np.random.default_rng(1)
        pts = np.vstack([rng.normal(0.0, 0.3, (120, 3)), rng.normal(25.0, 0.2, (15, 3))])
        res = grow_connected_cluster(pts, np.array([120, 121]), eps=1.5, exclude_ground=False)
        self.assertIsInstance(res, ClusterGrowth)
        np.testing.assert_array_equal(res.indices, np.arange(120, 135))
        self.assertFalse(res.capped)
        self.assertEqual(res.n_seed, 2)
        self.assertEqual(res.indices.dtype, np.int64)
        # 以前からの入口（引数と戻り値の形はそのまま）
        old_style = extract_connected_cluster_from_seeds(pts, np.array([120, 121]), radius=1.5, exclude_ground=False)
        np.testing.assert_array_equal(old_style, np.arange(120, 135))
        self.assertIsInstance(old_style, np.ndarray)
        np.testing.assert_array_equal(
            extract_connected_cluster_from_seeds(pts, np.array([0]), eps=1.5, exclude_ground=False), np.arange(0, 120)
        )

    def test_points_within_eps_always_connect(self):
        rng = np.random.default_rng(2)
        for trial in range(40):
            eps = float(rng.uniform(0.05, 0.6))
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            origin = rng.uniform(-500.0, 500.0, 3)
            chain = origin + np.arange(300)[:, None] * direction * (eps * 0.99)
            far = origin + direction * (eps * 0.99 * 299) + direction * 50.0    # 離れた点はつながらない
            pts = np.vstack([chain, far[None, :]])
            res = grow_connected_cluster(pts, np.array([int(rng.integers(0, 300))]), eps=eps, exclude_ground=False,
                                         max_cluster_ratio=1.0)
            np.testing.assert_array_equal(res.indices, np.arange(300), err_msg=f"trial {trial} eps={eps}")

    def test_float32_points(self):
        rng = np.random.default_rng(3)
        chain = (np.array([10.0, -20.0, 5.0]) + np.arange(200)[:, None] * np.array([0.6, 0.64, 0.48]) * 0.19).astype(np.float32)
        other = (chain + np.array([0.0, 0.0, 3.0])).astype(np.float32)
        pts = np.vstack([chain, other])
        res = grow_connected_cluster(pts, np.array([int(rng.integers(0, 200))]), eps=0.2, exclude_ground=False,
                                     max_cluster_ratio=1.0)
        np.testing.assert_array_equal(res.indices, np.arange(200))

    def test_gap_just_over_eps_is_not_bridged_and_just_under_is(self):
        rng = np.random.default_rng(4)
        for trial in range(30):
            eps = float(rng.uniform(0.05, 0.5))
            axis = trial % 3
            side = eps * 6
            origin = rng.uniform(-200.0, 200.0, 3)
            a = origin + rng.uniform(0.0, 1.0, (400, 3)) * side
            pa = origin + side / 2.0
            pa[axis] = origin[axis] + side                     # a の、軸の向きの面の上の点
            for gap, bridged in ((eps * 1.001, False), (eps * 0.999, True)):
                shift = np.zeros(3)
                shift[axis] = side + gap
                b = a + shift
                pb = pa.copy()
                pb[axis] += gap                                # pa の真向かいに、すき間 gap をあけて置く
                pts = np.vstack([a, b, pa[None, :], pb[None, :]])
                res = grow_connected_cluster(pts, np.array([800]), eps=eps, exclude_ground=False, max_cluster_ratio=1.0)
                self.assertEqual(801 in res.indices, bridged, f"trial {trial}: eps={eps} gap={gap}")
                if not bridged:
                    self.assertFalse(((res.indices >= 400) & (res.indices < 800)).any(), "すき間の向こうの塊が入っています")
                np.testing.assert_array_equal(res.indices, np.sort(ref_cluster(pts, np.array([800]), eps)))

    def test_same_as_point_by_point_search(self):
        compared = checks = 0
        for seed in range(8):
            pts, rng = patchy_cloud(seed)
            for eps in (0.1, 0.2, 0.33):
                for s in rng.integers(0, len(pts), 4):
                    seeds = np.array([int(s)])
                    expected = np.sort(ref_cluster(pts, seeds, eps))
                    for margin in (0.3, None, 50.0):
                        res = grow_connected_cluster(pts, seeds, eps=eps, exclude_ground=False, max_cluster_ratio=1.0,
                                                     initial_margin=margin)
                        np.testing.assert_array_equal(res.indices, expected, err_msg=f"seed {seed}/{s} eps={eps} margin={margin}")
                        self.assertFalse(res.capped)
                        compared += 1
                        checks += res.checks
        self.assertGreater(compared, 250)
        self.assertGreater(checks, 0, "行き止まりでの確かめ（KD 木）を通る場合が、テストに含まれていること")

    def test_cells_connect_through_points_that_are_not_their_representatives(self):
        # 格子は 1 辺 eps/1.75 = 0.2m。x が [0, 0.2) の格子に 2 点、[0.4, 0.6) の格子に 2 点。
        # eps 以内なのは「左の格子の右の点」と「右の格子の左の点」（0.22m）だけ。代表点がどれになっても、つながること
        eps = 0.35
        base = np.array([[0.01, 0.1, 0.1], [0.19, 0.1, 0.1], [0.41, 0.1, 0.1], [0.59, 0.1, 0.1]])
        for shift in ([0.0, 0.0, 0.0], [20.0, -7.0, 3.0], [-123.4, 55.4, 9.8]):
            pts = base + np.array(shift)
            for order in ([0, 1, 2, 3], [1, 0, 3, 2], [3, 2, 1, 0], [2, 0, 3, 1]):
                p = pts[order]
                for seed in range(4):
                    res = grow_connected_cluster(p, [seed], eps=eps, exclude_ground=False, max_cluster_ratio=1.0)
                    np.testing.assert_array_equal(res.indices, np.arange(4), err_msg=f"shift {shift} order {order} seed {seed}")
        # 右の 2 点を 0.14m 遠ざけると（いちばん近い組が 0.36m）、つながらない
        far = base.copy()
        far[2:, 0] += 0.14
        res = grow_connected_cluster(far, [0], eps=eps, exclude_ground=False, max_cluster_ratio=1.0)
        np.testing.assert_array_equal(res.indices, [0, 1])

    def test_same_cluster_from_any_seed_and_any_window(self):
        for seed in range(4):
            pts, rng = patchy_cloud(100 + seed)
            for s in rng.integers(0, len(pts), 4):
                first = grow_connected_cluster(pts, np.array([int(s)]), eps=0.2, exclude_ground=False,
                                               max_cluster_ratio=1.0, initial_margin=0.25)
                other_seed = first.indices[[len(first.indices) // 2, -1]]
                for margin in (0.4, 3.0, 500.0):
                    again = grow_connected_cluster(pts, other_seed, eps=0.2, exclude_ground=False,
                                                   max_cluster_ratio=1.0, initial_margin=margin)
                    np.testing.assert_array_equal(again.indices, first.indices)

    def test_long_cluster_grows_the_window_until_it_fits(self):
        rng = np.random.default_rng(5)
        rod = np.c_[np.linspace(0.0, 80.0, 4000), rng.normal(0, 0.01, 4000), 5.0 + rng.normal(0, 0.01, 4000)]   # 80m の棒
        other = ball(rng, [40.0, 6.0, 5.0], 0.5, 500)                                                     # 棒から 6m 離れた塊
        pts = np.vstack([rod, other])
        res = grow_connected_cluster(pts, np.array([0, 1]), eps=0.15, exclude_ground=False, max_cluster_ratio=1.0)
        np.testing.assert_array_equal(res.indices, np.arange(4000))
        self.assertGreaterEqual(res.rounds, 3, "最初の窓（±4.5m）に収まらないので、窓を広げているはず")
        self.assertGreaterEqual(res.margin, 36.0)

    def test_small_cluster_only_looks_at_its_neighbourhood(self):
        # 300 万点の床の上の、小さな浮遊塊。相手にするのは窓の中の点だけで、窓は広げない
        rng = np.random.default_rng(6)
        n_floor = 3_000_000
        floor = np.c_[rng.uniform(-150.0, 150.0, (n_floor, 2)), rng.normal(0.0, 0.003, n_floor)]
        blob = ball(rng, [10.0, -20.0, 3.0], 0.4, 2000)
        pts = np.vstack([floor, blob])
        t0 = time.perf_counter()
        res = grow_connected_cluster(pts, np.array([n_floor, n_floor + 1]), eps=0.15)
        elapsed = time.perf_counter() - t0
        np.testing.assert_array_equal(res.indices, np.arange(n_floor, n_floor + 2000))
        self.assertEqual(res.rounds, 0)
        self.assertLess(res.window_points, 20000, "窓の中の点（床の一部 + 塊）だけを相手にする")
        self.assertLess(elapsed, 5.0, f"300 万点で {elapsed:.1f} 秒かかった（以前の作りは、準備だけで数秒〜）")

    def test_empty_and_out_of_range_inputs(self):
        pts = np.random.default_rng(7).uniform(0, 1, (100, 3))
        for points, seeds in ((None, [0]), (np.empty((0, 3)), [0]), (pts, []), (pts, None), (pts, [-5, 100, 9999])):
            res = grow_connected_cluster(points, seeds, eps=0.2)
            self.assertEqual(len(res.indices), 0)
            self.assertFalse(res.capped)
            self.assertEqual(res.indices.dtype, np.int64)
        self.assertEqual(len(extract_connected_cluster_from_seeds(pts, np.array([], dtype=np.int64))), 0)
        self.assertEqual(len(extract_connected_cluster_from_seeds(None, np.array([0]))), 0)
        # 範囲外の番号がまじっていても、範囲内の起点だけで動く
        res = grow_connected_cluster(pts, [-1, 3, 100], eps=0.5, exclude_ground=False, max_cluster_ratio=1.0)
        self.assertIn(3, res.indices)
        self.assertEqual(res.n_seed, 1)

    def test_points_without_coordinates_are_left_alone(self):
        rng = np.random.default_rng(9)
        blob = ball(rng, [5.0, 5.0, 5.0], 0.4, 500)
        pts = np.vstack([blob, [[np.nan, 0.0, 0.0], [np.inf, 1.0, 1.0], [5.0, 5.0, np.nan], [1e30, -1e30, 0.0]]])
        with np.errstate(invalid="ignore"):
            res = grow_connected_cluster(pts, [0, 500, 501, 502], eps=0.15, exclude_ground=False, max_cluster_ratio=1.0)
            np.testing.assert_array_equal(res.indices, np.arange(500))
            self.assertEqual(res.n_seed, 1)
            # 高さが数でない点が同じ場所にあっても、地面の判定は変わらない（その点が無いときと同じ結果）
            with_ground = grow_connected_cluster(pts, [0], eps=0.15, max_cluster_ratio=1.0).indices
            np.testing.assert_array_equal(with_ground, grow_connected_cluster(blob, [0], eps=0.15, max_cluster_ratio=1.0).indices)
            # 座標が数でない点だけを起点にしたときは、何も返さない（例外にしない）
            self.assertEqual(len(grow_connected_cluster(pts, [500, 501, 502], eps=0.15).indices), 0)

    def test_too_wide_for_the_grid_is_reported(self):
        pts = np.random.default_rng(8).uniform(0, 1, (200, 3))
        saved = CG._MAX_INDEX
        CG._MAX_INDEX = 20
        try:
            with self.assertRaises(ValueError) as ctx:
                grow_connected_cluster(pts, [0], eps=0.05, exclude_ground=False)
            self.assertIn("連結距離を大きく", str(ctx.exception))
        finally:
            CG._MAX_INDEX = saved


class TestGround(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene = Scene(seed=11)

    def test_person_on_the_floor_does_not_spread_over_the_floor(self):
        sc = self.scene
        seeds = sc.person[sc.points[sc.person, 2] > 1.0][:8]
        res = grow_connected_cluster(sc.points, seeds, eps=0.15)
        self.assertFalse(res.capped)
        self.assertTrue(np.isin(res.indices, sc.person).all(), "床や壁の点が入っています")
        body = sc.person[sc.points[sc.person, 2] > 0.2]
        self.assertGreater(np.isin(body, res.indices).mean(), 0.97, "人の体（床から 20cm より上）は、離れた外れ点を除いて入る")
        np.testing.assert_array_equal(res.indices, np.sort(ref_cluster(sc.points, seeds, 0.15, ground_eps=0.15)))
        self.assertEqual(res.rounds, 0)

    def test_floating_blob_high_above_is_selected_down_to_its_bottom(self):
        # 床から 9m 上。最初の窓（±4.5m）に床は入らない。窓の中のいちばん低い点（塊の底）を地面と見なすと、底が欠ける
        sc = self.scene
        res = grow_connected_cluster(sc.points, sc.blob[:3], eps=0.15)
        np.testing.assert_array_equal(res.indices, sc.blob)
        np.testing.assert_array_equal(res.indices, ref_cluster(sc.points, sc.blob[:3], 0.15, ground_eps=0.15))

    def test_seed_on_the_floor_takes_the_floor_and_hits_the_limit(self):
        sc = self.scene
        seeds = sc.floor[np.linalg.norm(sc.points[sc.floor, :2], axis=1) < 0.3]
        self.assertGreater(len(seeds), 3)
        res = grow_connected_cluster(sc.points, seeds, eps=0.15, max_cluster_ratio=0.45)
        self.assertTrue(res.capped, "床は全体の 8 割以上なので、上限（45%）を超える")
        self.assertEqual(res.limit, int(sc.n * 0.45))
        self.assertGreater(len(res.indices), res.limit)
        # 上限を外せば、床ぜんたい（と、床につながる人・壁）まで広がる
        res = grow_connected_cluster(sc.points, seeds, eps=0.15, max_cluster_ratio=1.0)
        self.assertFalse(res.capped)
        self.assertGreater(np.isin(sc.floor, res.indices).mean(), 0.99)
        self.assertFalse(np.isin(sc.blob, res.indices).any(), "9m 上の浮遊塊は、床とつながっていない")

    def test_same_as_the_previous_implementation_with_ground(self):
        for seed in (21, 22, 23):
            sc = Scene(seed=seed, n_floor=30000, extent=8.0)
            rng = np.random.default_rng(seed)
            for group in (sc.person[sc.points[sc.person, 2] > 0.5], sc.blob, sc.wall[sc.points[sc.wall, 2] > 1.0], sc.floor):
                seeds = rng.choice(group, 4, replace=False)
                for eps in (0.12, 0.15, 0.3):
                    expected = np.sort(ref_cluster(sc.points, seeds, eps, ground_eps=eps))
                    for margin in (0.5, None, 100.0):
                        got = grow_connected_cluster(sc.points, seeds, eps=eps, max_cluster_ratio=1.0, initial_margin=margin).indices
                        np.testing.assert_array_equal(got, expected, err_msg=f"seed {seed} eps {eps} margin {margin}")

    def test_exclude_ground_off_lets_it_run_over_the_floor(self):
        sc = self.scene
        seeds = sc.person[sc.points[sc.person, 2] > 1.0][:8]
        res = grow_connected_cluster(sc.points, seeds, eps=0.15, exclude_ground=False, max_cluster_ratio=1.0)
        self.assertGreater(np.isin(sc.floor, res.indices).mean(), 0.99)


class TestGroundMask(unittest.TestCase):
    def _clouds(self):
        rng = np.random.default_rng(31)
        yield "ふつう", np.c_[rng.uniform(-30, 30, (20000, 2)), rng.normal(0, 0.5, 20000)]
        yield "負の座標と段差", np.c_[rng.uniform(-900, -850, (20000, 2)), rng.choice([0.0, 2.5], 20000) + rng.normal(0, 0.05, 20000)]
        yield "float32", np.c_[rng.uniform(0, 40, (20000, 2)), rng.normal(10, 0.3, 20000)].astype(np.float32)
        yield "1 点", np.array([[1.0, 2.0, 3.0]])
        yield "遠くの外れ値つき", np.vstack([np.c_[rng.uniform(0, 10, (5000, 2)), rng.normal(0, 0.2, 5000)], [[9000.0, -7000.0, 3.0]]])

    def test_same_as_the_previous_implementation(self):
        for name, pts in self._clouds():
            for grid, height in ((1.0, 0.25), (0.8, 0.12), (0.05, 0.001), (3.3, 0.4)):
                expected = ref_ground_mask(pts, grid, height)
                got = ground_point_mask(pts, grid, height)
                self.assertEqual(got.dtype, np.bool_)
                np.testing.assert_array_equal(got, expected, err_msg=f"{name} grid={grid}")
                idx = segment_ground_points(pts, grid_size=grid, height_threshold=height)
                self.assertEqual(idx.dtype, np.int64)
                np.testing.assert_array_equal(idx, np.where(expected)[0], err_msg=f"{name} grid={grid}")

    def test_sparse_path_gives_the_same_answer(self):
        saved = CG.GROUND_DENSE_CELLS
        CG.GROUND_DENSE_CELLS = 10        # 格子の表を直接持たない道（広い現場）
        try:
            for name, pts in self._clouds():
                np.testing.assert_array_equal(ground_point_mask(pts, 1.0, 0.25), ref_ground_mask(pts, 1.0, 0.25), err_msg=name)
        finally:
            CG.GROUND_DENSE_CELLS = saved

    def test_broken_coordinates_do_not_crash(self):
        rng = np.random.default_rng(32)
        pts = np.c_[rng.uniform(0, 10, (2000, 2)), rng.normal(0, 0.2, 2000)]
        pts[5] = [1e30, -1e30, 0.0]
        with np.errstate(invalid="ignore"):
            got = ground_point_mask(pts, 1.0, 0.25)
            expected = ref_ground_mask(pts, 1.0, 0.25)
        np.testing.assert_array_equal(got, expected)

    def test_empty(self):
        self.assertEqual(len(ground_point_mask(np.empty((0, 3)), 1.0, 0.25)), 0)
        self.assertEqual(len(ground_point_mask(None, 1.0, 0.25)), 0)
        self.assertEqual(len(segment_ground_points(np.empty((0, 3)))), 0)
        self.assertEqual(len(segment_ground_points(None)), 0)


class TestSelectableAndLimits(unittest.TestCase):
    def _two_blobs_with_bridge(self):
        rng = np.random.default_rng(41)
        a = ball(rng, [0.0, 0.0, 5.0], 0.4, 600)
        b = ball(rng, [3.0, 0.0, 5.0], 0.4, 600)
        bridge = np.c_[np.linspace(0.3, 2.7, 60), np.zeros(60), np.full(60, 5.0)]
        pts = np.vstack([a, b, bridge])
        return pts, np.arange(0, 600), np.arange(600, 1200), np.arange(1200, 1260)

    def test_unselectable_points_do_not_bridge_and_are_not_returned(self):
        pts, a, b, bridge = self._two_blobs_with_bridge()
        everything = grow_connected_cluster(pts, a[:2], eps=0.15, exclude_ground=False, max_cluster_ratio=1.0)
        np.testing.assert_array_equal(everything.indices, np.arange(len(pts)))      # 橋があれば、ぜんぶつながる
        selectable = np.ones(len(pts), dtype=bool)
        selectable[bridge] = False                                                 # 橋を非表示（またはロック）にする
        res = grow_connected_cluster(pts, a[:2], eps=0.15, selectable_mask=selectable, exclude_ground=False, max_cluster_ratio=1.0)
        np.testing.assert_array_equal(res.indices, a)
        np.testing.assert_array_equal(res.indices, ref_cluster(pts, a[:2], 0.15, selectable=selectable))

    def test_unselectable_seeds_are_ignored(self):
        pts, a, b, bridge = self._two_blobs_with_bridge()
        selectable = np.ones(len(pts), dtype=bool)
        selectable[a] = False
        res = grow_connected_cluster(pts, a[:5], eps=0.15, selectable_mask=selectable, exclude_ground=False)
        self.assertEqual(len(res.indices), 0)
        self.assertEqual(res.n_seed, 0)
        # 選べる起点と選べない起点がまじっているときは、選べる起点だけを使う
        selectable[bridge] = False
        res = grow_connected_cluster(pts, np.r_[a[:5], b[:2]], eps=0.15, selectable_mask=selectable, exclude_ground=False,
                                     max_cluster_ratio=1.0)
        np.testing.assert_array_equal(res.indices, b)
        self.assertEqual(res.n_seed, 2)
        self.assertTrue(selectable[res.indices].all())

    def test_wrong_mask_length_is_an_error(self):
        pts, a, _b, _bridge = self._two_blobs_with_bridge()
        with self.assertRaises(ValueError):
            grow_connected_cluster(pts, a[:2], selectable_mask=np.ones(10, dtype=bool))

    def test_limit_counts_selectable_points_only(self):
        pts, a, b, bridge = self._two_blobs_with_bridge()
        selectable = np.zeros(len(pts), dtype=bool)
        selectable[a] = True
        selectable[b[:100]] = True
        res = grow_connected_cluster(pts, a[:2], eps=0.15, selectable_mask=selectable, exclude_ground=False, max_cluster_ratio=0.45)
        self.assertEqual(res.limit, int(700 * 0.45))
        self.assertTrue(res.capped, "塊 a（600 点）は、選べる 700 点の 45% を超える")
        res = grow_connected_cluster(pts, a[:2], eps=0.15, selectable_mask=selectable, exclude_ground=False, max_cluster_ratio=0.9)
        self.assertFalse(res.capped)
        np.testing.assert_array_equal(res.indices, a)

    def test_cancel_stops_with_interrupted_error(self):
        sc = Scene(seed=12)
        with self.assertRaises(InterruptedError):
            grow_connected_cluster(sc.points, sc.blob[:3], eps=0.15, is_cancelled=lambda: True)
        # 途中で中止になったとき
        calls = {"n": 0}

        def later():
            calls["n"] += 1
            return calls["n"] > 3

        with self.assertRaises(InterruptedError):
            grow_connected_cluster(sc.points, sc.floor[:5], eps=0.15, max_cluster_ratio=1.0, is_cancelled=later)
        self.assertGreater(calls["n"], 3)

        # 進み具合の知らせの中で中止の例外が出たとき（FunctionWorker のやり方）も、そのまま止まる
        def cb(_ratio, _message):
            raise InterruptedError("中止")

        with self.assertRaises(InterruptedError):
            grow_connected_cluster(sc.points, sc.blob[:3], eps=0.15, progress_callback=cb)

    def test_progress_messages(self):
        sc = Scene(seed=13)
        got = []
        saved = CG.PROGRESS_INTERVAL
        CG.PROGRESS_INTERVAL = 0.0     # テストでは、毎回知らせる
        try:
            res = grow_connected_cluster(sc.points, sc.floor[:5], eps=0.15, max_cluster_ratio=1.0,
                                         progress_callback=lambda r, m: got.append((r, m)))
        finally:
            CG.PROGRESS_INTERVAL = saved
        self.assertFalse(res.capped)
        self.assertTrue(all(0.0 <= r <= 1.0 for r, _ in got))
        text = "\n".join(m for _, m in got)
        for phrase in ("まわりの点を集めています", "地面を見分けています", "格子に分けています", "つながりをたどっています"):
            self.assertIn(phrase, text)
        self.assertRegex(text, r"つながりをたどっています\.\.\. [\d,]+ 点（まわり ±[\d.]+m）")
        self.assertIn("±4.5m", text, "最初の窓の余白 4.5m を、4m と丸めて出さない")
        # 床のまん中の近い点を起点にすると、窓を何度も広げてやり直す。それでも、進み具合の値は戻らない
        got2 = []
        center = sc.floor[np.linalg.norm(sc.points[sc.floor, :2], axis=1) < 0.3]
        CG.PROGRESS_INTERVAL = 0.0
        try:
            res2 = grow_connected_cluster(sc.points, center, eps=0.15, max_cluster_ratio=1.0,
                                          progress_callback=lambda r, m: got2.append(r))
        finally:
            CG.PROGRESS_INTERVAL = saved
        self.assertGreaterEqual(res2.rounds, 1)
        self.assertEqual(got2, sorted(got2), "窓を広げてやり直しても、進み具合の値は戻らない")
        self.assertLess(max(len(m) for _, m in got), 45, "進捗の窓の幅に収まる長さ")

    def test_worker_can_pass_progress_and_cancel(self):
        # gui/worker_thread.py: FunctionWorker は、この名前の引数があれば、進み具合と中止を自動で渡す
        params = inspect.signature(grow_connected_cluster).parameters
        self.assertIn("progress_callback", params)
        self.assertIn("is_cancelled", params)


class TestRulesThatKeepTheAnswer(unittest.TestCase):
    """
    答えを以前と同じに保っている決まりごと。2026-11-14 の独立レビューで、わざと壊した版をテストが見逃した所
    （格子の大きさ・窓の切り方・地面と選べる点の組み合わせ・起点の扱い・上限ちょうど）と、境目で食い違った所。
    """

    def test_cell_size_keeps_the_guarantees(self):
        self.assertGreater(CG.CELL_PER_EPS, 3.0 ** 0.5, "同じ格子の点が、必ず eps 以内になる大きさであること")
        self.assertGreaterEqual(CG.NEIGHBOR_CELLS, CG.CELL_PER_EPS, "eps 以内の点が、必ず調べる範囲の格子にあること")
        # 格子の対角の向きに、eps をほんの少し超えて離した 2 点は、どこに置いてもつながらない（同じ格子に入らない）
        rng = np.random.default_rng(61)
        diagonal = np.ones(3) / 3.0 ** 0.5
        for trial in range(60):
            eps = float(rng.choice([0.05, 0.15, 0.35, 1.0]))
            a = rng.uniform(-5.0, 5.0, 3)
            sign = rng.choice([-1.0, 1.0], 3)
            far = np.vstack([a, a + diagonal * sign * eps * 1.0005])
            near = np.vstack([a, a + diagonal * sign * eps * 0.9995])
            res = grow_connected_cluster(far, [0], eps=eps, exclude_ground=False, max_cluster_ratio=1.0)
            np.testing.assert_array_equal(res.indices, [0], err_msg=f"trial {trial} eps={eps}")
            res = grow_connected_cluster(near, [0], eps=eps, exclude_ground=False, max_cluster_ratio=1.0)
            np.testing.assert_array_equal(res.indices, [0, 1], err_msg=f"trial {trial} eps={eps}")

    def test_lattice_with_spacing_of_exactly_eps(self):
        for eps, origin in ((0.25, 0.0), (0.15, 1000.0), (0.5, -77.5)):
            g = origin + np.arange(5) * eps
            pts = np.array([[x, y, z] for x in g for y in g for z in g])
            expected = np.sort(ref_cluster(pts, np.array([0]), eps))
            res = grow_connected_cluster(pts, [0], eps=eps, exclude_ground=False, max_cluster_ratio=1.0)
            np.testing.assert_array_equal(res.indices, expected, err_msg=f"eps={eps} origin={origin}")

    def test_one_ulp_over_eps_is_judged_the_same_in_both_steps(self):
        # 3 点目は、2 点目から「eps の 2 乗より 1 目盛りだけ大きい」距離。近道と行き止まりの確かめで式が違うと、
        # 同じ格子にほかの点（1 点目）があるかどうかで、答えが変わってしまう
        pts = np.array([[0.0, 0.05, 0.0], [0.0, 0.0, 0.0], [0.15, 2.0 ** -29, 0.0]])
        expected = np.sort(ref_cluster(pts, np.array([0]), 0.15))
        res = grow_connected_cluster(pts, [0], eps=0.15, exclude_ground=False, max_cluster_ratio=1.0)
        np.testing.assert_array_equal(res.indices, expected)
        two = grow_connected_cluster(pts[1:], [0], eps=0.15, exclude_ground=False, max_cluster_ratio=1.0)
        self.assertEqual(2 in res.indices, 1 in two.indices, "3 点目が入るかどうかは、1 点目があっても無くても同じ")

    def test_window_is_cut_on_ground_grid_lines(self):
        # 床に穴（影）があり、その上に棒の端がある。地面の格子 [0, 0.8) の床は x < 0.2 にしか無い。
        # 窓を地面の格子にそろえずに切ると（窓の左端 x = 0.2）、その格子の「最低の高さ」が棒そのものになり、棒の端が欠ける
        rng = np.random.default_rng(0)
        fx = rng.uniform(-3.0, 8.0, 30000)
        fy = rng.uniform(-1.6, 1.6, 30000)
        hole = (fx >= 0.2) & (fx < 0.8) & (fy >= 0.0) & (fy < 0.8)
        floor = np.c_[fx[~hole], fy[~hole], np.zeros(int((~hole).sum()))]
        rod_x = np.arange(0.55, 5.0001, 0.05)
        rod = np.c_[rod_x, np.full(len(rod_x), 0.4), np.full(len(rod_x), 1.0)]
        pts = np.vstack([floor, rod])
        seeds = np.arange(len(floor), len(pts))[rod_x > 4.69]
        expected = np.sort(ref_cluster(pts, seeds, 0.15, ground_eps=0.15))
        res = grow_connected_cluster(pts, seeds, eps=0.15, max_cluster_ratio=1.0)
        np.testing.assert_array_equal(res.indices, expected)
        self.assertEqual(len(res.indices), len(rod), "棒は、端まで全部入る")

    def test_point_exactly_on_a_ground_grid_line(self):
        # 0.8m の格子で、floor(13.6 / 0.8) は 17 なのに 17 * 0.8 は 13.600000000000001。
        # 窓の端を 17 * 0.8 で切ると、x = 13.6 の点（その格子でいちばん低い点）が窓から落ち、地面の分類が変わる
        rng = np.random.default_rng(0)
        floor = np.c_[rng.uniform(14.4, 20.0, 4000), rng.uniform(0.0, 0.8, 4000), np.zeros(4000)]
        rod_x = np.arange(14.0, 18.5001, 0.05)
        rod = np.c_[rod_x, np.full(len(rod_x), 0.4), np.full(len(rod_x), 1.0)]
        pts = np.vstack([floor, rod, [[13.6, 0.4, 0.0]]])
        seeds = np.arange(len(floor), len(floor) + len(rod))[rod_x > 18.2]
        expected = np.sort(ref_cluster(pts, seeds, 0.15, ground_eps=0.15))
        for margin in (None, 50.0):
            res = grow_connected_cluster(pts, seeds, eps=0.15, max_cluster_ratio=1.0, initial_margin=margin)
            np.testing.assert_array_equal(res.indices, expected, err_msg=f"margin={margin}")

    def test_grid_edge_agrees_with_the_cell_number(self):
        # 窓に入る条件（端以上）と、地面の格子の番号（floor(x / 格子)）が、必ず一致すること
        for grid in (0.8, 0.84, 0.92, 1.2, 1.32, 1.4, 1.6, 1.8):
            for k in list(range(-3000, 3001)) + [-399996, 123456, 327677]:
                edge = CG._grid_edge(float(k), grid)
                self.assertGreaterEqual(np.floor(edge / grid), k)
                self.assertLess(np.floor(np.nextafter(edge, -np.inf) / grid), k)

    def test_ground_rule_looks_at_selectable_points_only(self):
        # 床を選べなくする（地面のレイヤーを隠す・ロックする）と、人の足もとが「その場のいちばん低い点」になる。以前の作りと同じ
        sc = Scene(seed=11)
        selectable = np.ones(sc.n, dtype=bool)
        selectable[sc.floor] = False
        seeds = sc.person[sc.points[sc.person, 2] > 1.0][:8]
        expected = np.sort(ref_cluster(sc.points, seeds, 0.15, ground_eps=0.15, selectable=selectable))
        res = grow_connected_cluster(sc.points, seeds, eps=0.15, selectable_mask=selectable, max_cluster_ratio=1.0)
        np.testing.assert_array_equal(res.indices, expected)
        with_floor = grow_connected_cluster(sc.points, seeds, eps=0.15, max_cluster_ratio=1.0)
        self.assertNotEqual(len(res.indices), len(with_floor.indices), "床が見えているときとは、地面の分類が違うはず")

    def test_seeds_on_the_ground_stay_in_the_result(self):
        # 起点の少数が地面の上（人の体 6 点 + 足もとの 2 点）。地面は伝わないが、起点そのものは外さない
        sc = Scene(seed=11)
        ground = ground_point_mask(sc.points, 0.8, 0.12)
        feet = sc.person[ground[sc.person]][:2]
        body = sc.person[sc.points[sc.person, 2] > 1.0][:6]
        self.assertEqual(len(feet), 2)
        seeds = np.r_[body, feet]
        expected = np.sort(ref_cluster(sc.points, seeds, 0.15, ground_eps=0.15))
        res = grow_connected_cluster(sc.points, seeds, eps=0.15, max_cluster_ratio=1.0)
        np.testing.assert_array_equal(res.indices, expected)
        self.assertTrue(np.isin(seeds, res.indices).all())
        self.assertTrue(np.isin(res.indices, sc.person).all())

    def test_ground_rule_is_decided_by_all_selectable_points_not_by_the_window(self):
        # 40 点の小さな塊（まわりに何も無い）+ 100m 先に 500 点。窓の中は 40 点でも、全体が 50 点を超えるので地面の判定をする
        rng = np.random.default_rng(0)
        blob = ball(rng, [0.3, 0.3, 5.0], 0.25, 40)
        pts = np.vstack([blob, ball(rng, [100.0, 100.0, 0.0], 2.0, 500)])
        seeds = np.argsort(-blob[:, 2])[:3]
        expected = np.sort(ref_cluster(pts, seeds, 0.15, ground_eps=0.15))
        for margin in (None, 0.5, 300.0):
            res = grow_connected_cluster(pts, seeds, eps=0.15, max_cluster_ratio=1.0, initial_margin=margin)
            np.testing.assert_array_equal(res.indices, expected, err_msg=f"margin={margin}")

    def test_selectable_and_ground_together(self):
        mismatches = 0
        for seed in range(6):
            r = np.random.default_rng(100 + seed)
            floor = np.c_[r.uniform(-8, 8, (20000, 2)), r.normal(0, 0.003, 20000)]
            person = np.c_[r.normal([3.0, 2.0], 0.10, (1500, 2)), r.uniform(0.0, 1.7, 1500)]
            box = np.c_[r.uniform(-3, -2, (3000, 2)), r.uniform(0.0, 1.0, 3000)]
            pts = np.vstack([floor, person, box])
            selectable = r.uniform(size=len(pts)) > 0.3
            selectable[: len(floor)] &= r.uniform(size=len(floor)) > 0.5      # 床の一部を選べなくすると、格子の最低点が変わる
            cand = np.flatnonzero(selectable)
            for lo_i, hi_i in ((20000, 21500), (21500, 24500), (0, 20000)):
                seeds = r.choice(cand[(cand >= lo_i) & (cand < hi_i)], 3, replace=False)
                expected = np.sort(ref_cluster(pts, seeds, 0.15, ground_eps=0.15, selectable=selectable))
                for margin in (None, 0.5):
                    res = grow_connected_cluster(pts, seeds, eps=0.15, selectable_mask=selectable, max_cluster_ratio=1.0,
                                                 initial_margin=margin)
                    mismatches += int(not np.array_equal(res.indices, expected))
        self.assertEqual(mismatches, 0)

    def test_cluster_of_exactly_the_limit_is_not_cut(self):
        rng = np.random.default_rng(5)
        blob = rng.normal(0, 0.05, (450, 3))
        rest = rng.normal(0, 0.05, (550, 3)) + [50.0, 0.0, 0.0]
        pts = np.vstack([blob, rest])
        res = grow_connected_cluster(pts, [0], eps=0.5, exclude_ground=False, max_cluster_ratio=0.45)
        self.assertEqual(res.limit, 450)
        self.assertFalse(res.capped, "上限ちょうどの点数は、打ち切らない")
        np.testing.assert_array_equal(res.indices, np.arange(450))
        res = grow_connected_cluster(pts, [450], eps=0.5, exclude_ground=False, max_cluster_ratio=0.45)
        self.assertTrue(res.capped, "550 点の塊は、上限 450 を超える")

    def test_old_entry_point_never_raises(self):
        # 起点が極端に離れているなど、範囲が広すぎるとき。以前からの入口は、例外を出さずに起点だけを返す
        rng = np.random.default_rng(62)
        pts = np.vstack([ball(rng, [0.0, 0.0, 5.0], 0.4, 300), [[1e9, 0.0, 0.0]]])
        with self.assertRaises(ValueError):
            grow_connected_cluster(pts, [0, 300], eps=0.15, exclude_ground=False)
        out = extract_connected_cluster_from_seeds(pts, np.array([300, 0, 0, 999]), eps=0.15, exclude_ground=False)
        np.testing.assert_array_equal(out, [0, 300])


class TestWandInWindow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        from gui.main_window import MainWindow

        cls.app = QApplication.instance() or QApplication(sys.argv)
        cls.win = MainWindow()
        cls.win._reset_app_settings()
        cls.win.resize(1450, 900)
        cls.win.show()
        cls.app.processEvents()
        cls.scene = Scene(seed=51, n_floor=40000)

    @classmethod
    def tearDownClass(cls):
        cls.win._set_dirty(False)
        cls.win.viewer.set_point_cloud(None)
        destroy_widget(cls.win)

    def setUp(self):
        win = self.win
        win._toggle_instant_delete_mode(False)
        win.viewer.set_point_cloud(self.scene.points.copy(), None, None)
        win.cluster_eps_spin.setValue(0.15)
        self.app.processEvents()

    def tearDown(self):
        self.win._toggle_instant_delete_mode(False)

    def _wand(self, seeds):
        win = self.win
        win.viewer.selected_indices = np.asarray(seeds, dtype=np.int64)
        win.viewer._update_active_colors()
        win._on_magic_wand_cluster_select()
        self.assertTrue(win._wait_for_background(), "連結塊の追跡が終わりません")
        for _ in range(5):
            self.app.processEvents()

    def _noise_count(self):
        from gui.gl_viewer import PointCloudViewer
        return len(self.win.viewer.get_layer_indices(PointCloudViewer.LAYER_NOISE))

    def test_small_blob_is_selected(self):
        sc, win = self.scene, self.win
        self._wand(sc.blob[:4])
        np.testing.assert_array_equal(np.sort(win.viewer.selected_indices), sc.blob)
        self.assertIn("一括選択", win.toast_label.text())
        self.assertIn(f"{len(sc.blob):,} 点", win.toast_label.text())
        self.assertEqual(self._noise_count(), 0)
        self.assertIsNone(win.progress_dialog)
        win.viewer.validate_invariants()

    def test_connected_to_the_floor_changes_nothing_and_says_why(self):
        sc, win = self.scene, self.win
        seeds = sc.floor[np.linalg.norm(sc.points[sc.floor, :2], axis=1) < 0.4]
        self.assertGreater(len(seeds), 3)
        for instant in (False, True):
            win._toggle_instant_delete_mode(instant)
            self._wand(seeds)
            np.testing.assert_array_equal(np.sort(win.viewer.selected_indices), np.sort(seeds), err_msg="選択は変えない")
            self.assertEqual(self._noise_count(), 0, "何も消さない")
            toast = win.toast_label.text()
            self.assertIn("大きすぎます", toast)
            self.assertIn("選択は変えていません", toast)
            self.assertEqual(win.toast_host.kind(), "warning")
            self.assertFalse(win.toast_host.has_action(), "点を動かしていないので「元に戻す」は出さない")
            status = win.status_label.text()
            self.assertIn("いま選べる点の 45%", status, "上限が何の何割かを伝える（表示中の点が少ないときにも当てはまる文）")
            self.assertIn("連結距離", status)
            self.assertIn("0.15m", status)
            self.assertIn("囲んで選んで", status)
            win.viewer.validate_invariants()

    def test_instant_delete_moves_the_blob_to_noise(self):
        from gui.gl_viewer import PointCloudViewer
        sc, win = self.scene, self.win
        win._toggle_instant_delete_mode(True)
        self._wand(sc.blob[:4])
        np.testing.assert_array_equal(np.sort(win.viewer.get_layer_indices(PointCloudViewer.LAYER_NOISE)), sc.blob)
        self.assertEqual(len(win.viewer.selected_indices), 0)
        self.assertIn("即消去", win.toast_label.text())
        win.viewer.validate_invariants()

    def test_cancel_button_stops_the_search(self):
        import gui.main_window as MW
        sc, win = self.scene, self.win
        real = MW.grow_connected_cluster
        started = threading.Event()

        def slow(points, seed_indices, progress_callback=None, is_cancelled=None, **kwargs):
            started.set()
            t0 = time.time()
            while not is_cancelled() and time.time() - t0 < 20.0:   # 「中止」が押されるまで待つ
                time.sleep(0.005)
            return real(points, seed_indices, progress_callback=progress_callback, is_cancelled=is_cancelled, **kwargs)

        MW.grow_connected_cluster = slow
        try:
            seeds = sc.blob[:4]
            win.viewer.selected_indices = np.asarray(seeds, dtype=np.int64)
            win._on_magic_wand_cluster_select()
            self.assertTrue(started.wait(10.0))
            self.assertIsNotNone(win.progress_dialog)
            win.progress_dialog.cancelled.emit()                    # 進捗の窓の「中止」
            self.assertTrue(win._wait_for_background())
            for _ in range(5):
                self.app.processEvents()
        finally:
            MW.grow_connected_cluster = real
        np.testing.assert_array_equal(win.viewer.selected_indices, seeds)
        self.assertEqual(self._noise_count(), 0)
        self.assertIn("中断", win.toast_label.text())
        self.assertIsNone(win.progress_dialog)

    def test_locked_or_hidden_seed_does_nothing(self):
        sc, win = self.scene, self.win
        saved = win.viewer._get_selectable_mask
        mask = np.ones(sc.n, dtype=bool)
        mask[sc.blob] = False
        win.viewer._get_selectable_mask = lambda: mask
        try:
            win.viewer.selected_indices = np.asarray(sc.blob[:4], dtype=np.int64)
            win._on_magic_wand_cluster_select()
            self.assertFalse(win._is_background_busy(), "選べない点だけが起点なら、何も始めない")
        finally:
            win.viewer._get_selectable_mask = saved
        np.testing.assert_array_equal(win.viewer.selected_indices, sc.blob[:4])


if __name__ == "__main__":
    unittest.main(verbosity=2)
