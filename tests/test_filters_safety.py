"""
ノイズフィルタの安全性テスト（全点消失ガード・代表点安全リミッター）

背景: 孤立クラスタ除去で「条件を満たす塊が1つも無い」場合に空集合を返し、点群が丸ごとノイズ化する
      事故が起こり得た（core/filters.py）。本テストはその再発防止と、代表点リミッターの上限遵守を検証する。
"""

import unittest
import numpy as np

from core.filters import (
    REPRESENTATIVE_POINT_LIMIT,
    _voxelize_with_limit,
    filter_isolated_clusters,
    keep_largest_clusters,
)


def _make_tiny_clusters(n_clusters: int, pts_per_cluster: int, spacing: float, seed: int = 7) -> np.ndarray:
    """互いに遠く離れた小さな塊を n_clusters 個生成する（どの塊も min_cluster_points 未満）"""
    rng = np.random.default_rng(seed)
    side = int(np.ceil(n_clusters ** (1.0 / 3.0)))
    centers = []
    for i in range(n_clusters):
        cx, cy, cz = i % side, (i // side) % side, i // (side * side)
        centers.append((cx * spacing, cy * spacing, cz * spacing))
    centers = np.asarray(centers, dtype=np.float64)
    pts = np.vstack([c + rng.normal(0.0, 0.01, (pts_per_cluster, 3)) for c in centers])
    return pts


class TestIsolatedClusterSafety(unittest.TestCase):
    """filter_isolated_clusters の全点消失ガード"""

    def test_small_path_no_qualifying_cluster_keeps_all_points(self):
        """1万点以下の直接DBSCAN経路: 全ての塊が最小点数未満でも全点を保持する（既定）"""
        pts = _make_tiny_clusters(n_clusters=20, pts_per_cluster=10, spacing=5.0)
        self.assertLessEqual(len(pts), 10000)
        keep = filter_isolated_clusters(pts, eps=0.15, min_cluster_points=30)
        self.assertEqual(len(keep), len(pts), "条件を満たす塊が無い場合に全点が消えてはならない")
        np.testing.assert_array_equal(np.sort(keep), np.arange(len(pts)))

    def test_small_path_allow_remove_all_restores_legacy_behavior(self):
        """allow_remove_all=True を明示した場合のみ、旧来どおり空集合（全点除去）を返す"""
        pts = _make_tiny_clusters(n_clusters=20, pts_per_cluster=10, spacing=5.0)
        keep = filter_isolated_clusters(pts, eps=0.15, min_cluster_points=30, allow_remove_all=True)
        self.assertEqual(len(keep), 0)

    def test_large_path_no_qualifying_cluster_keeps_all_points(self):
        """1万点超のボクセル代表DBSCAN経路でも同じく全点を保持する"""
        pts = _make_tiny_clusters(n_clusters=1200, pts_per_cluster=10, spacing=5.0)
        self.assertGreater(len(pts), 10000)
        keep = filter_isolated_clusters(pts, eps=0.15, min_cluster_points=30)
        self.assertEqual(len(keep), len(pts))

    def test_normal_case_still_removes_small_floating_cluster(self):
        """通常ケース（大きな塊 + 浮遊小塊）では従来どおり小塊だけが除去される"""
        rng = np.random.default_rng(3)
        main = rng.uniform(0.0, 0.5, (500, 3))  # 一辺0.5mの密な塊（点間隔 約6cm < eps）
        floating = rng.normal(20.0, 0.02, (8, 3))
        pts = np.vstack([main, floating])
        keep = filter_isolated_clusters(pts, eps=0.15, min_cluster_points=30)
        self.assertEqual(len(keep), 500)
        self.assertTrue(np.all(keep < 500))

    def test_consistency_with_keep_largest_clusters(self):
        """keep_largest_clusters は元から「見つからなければ全点保持」。両関数の安全側の挙動が揃っていること"""
        pts = _make_tiny_clusters(n_clusters=20, pts_per_cluster=10, spacing=5.0)
        keep_a = filter_isolated_clusters(pts, eps=0.15, min_cluster_points=30)
        keep_b = keep_largest_clusters(pts, eps=0.15, top_k=1, min_points=30)
        # keep_largest_clusters は最大塊(10点)のみ or 全点のどちらかだが、少なくとも空にはならない
        self.assertGreater(len(keep_b), 0)
        self.assertEqual(len(keep_a), len(pts))


class TestRepresentativePointLimiter(unittest.TestCase):
    """代表点安全リミッター（REPRESENTATIVE_POINT_LIMIT）"""

    def test_limit_constant_matches_documentation(self):
        self.assertEqual(REPRESENTATIVE_POINT_LIMIT, 200_000)

    def test_dense_random_cloud_is_reduced_below_limit(self):
        """初期推定では上限を超える密な点群でも、再量子化で代表点数が上限以下に収まる"""
        rng = np.random.default_rng(11)
        n = 260_000
        pts = rng.uniform(0.0, 2.0, (n, 3))
        unique_coords, inverse_idx, counts, voxel_size = _voxelize_with_limit(pts, eps=0.02)
        self.assertLessEqual(len(unique_coords), REPRESENTATIVE_POINT_LIMIT)
        self.assertEqual(len(inverse_idx), n)
        self.assertEqual(int(np.sum(counts)), n)
        self.assertGreater(voxel_size, 0.0)

    def test_small_cloud_is_untouched(self):
        """上限未満の点群ではボクセルサイズは基準値 (max(0.02, eps*0.45)) のまま"""
        rng = np.random.default_rng(5)
        pts = rng.uniform(0.0, 1.0, (5000, 3))
        _, _, _, voxel_size = _voxelize_with_limit(pts, eps=0.30)
        self.assertAlmostEqual(voxel_size, 0.30 * 0.45)


if __name__ == "__main__":
    unittest.main()
