"""
点群処理コアモジュール（e57_io, filters, processor）の単体テスト
"""

import os
import unittest
import numpy as np
import pye57

from core.e57_io import E57ScanData, read_e57_scans, write_e57_scans
from core.filters import (
    filter_statistical_outlier,
    filter_radius_outlier,
    filter_distance,
    filter_intensity,
    voxel_downsample_indices,
)
from core.processor import FilterConfig, PointCloudProcessor


class TestPointCloudCore(unittest.TestCase):
    """E57入出力およびフィルタ処理のテストケース"""

    def setUp(self):
        """テストデータの準備"""
        self.test_input = "test_temp_input.e57"
        self.test_output = "test_temp_output.e57"

        # 密な球体点群（インライア: 1000点）
        np.random.seed(42)
        n_inliers = 1000
        inliers = np.random.normal(loc=0.0, scale=1.0, size=(n_inliers, 3))

        # 散乱した外れ値ノイズ（アウトライア: 50点）
        n_outliers = 50
        outliers = np.random.uniform(low=-15.0, high=15.0, size=(n_outliers, 3))

        self.all_points = np.vstack([inliers, outliers]).astype(np.float64)
        self.colors = np.random.randint(0, 255, size=(len(self.all_points), 3), dtype=np.uint8)
        self.intensities = np.random.uniform(0.1, 1.0, size=len(self.all_points)).astype(np.float32)

        # テスト用E57ファイルを書き込み
        with pye57.E57(self.test_input, mode="w") as e57:
            data = {
                "cartesianX": self.all_points[:, 0],
                "cartesianY": self.all_points[:, 1],
                "cartesianZ": self.all_points[:, 2],
                "colorRed": self.colors[:, 0],
                "colorGreen": self.colors[:, 1],
                "colorBlue": self.colors[:, 2],
                "intensity": self.intensities,
            }
            e57.write_scan_raw(data)

    def tearDown(self):
        """テスト後の一時ファイル削除"""
        for f in [self.test_input, self.test_output]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except Exception:
                    pass

    def test_e57_read(self):
        """E57読み込みテスト"""
        scans = read_e57_scans(self.test_input)
        self.assertEqual(len(scans), 1)
        scan = scans[0]
        self.assertEqual(scan.point_count, 1050)
        self.assertIn("colorRed", scan.raw_fields)
        self.assertIn("intensity", scan.raw_fields)

    def test_filter_statistical_outlier(self):
        """統計的外れ値除去 (SOR) のテスト"""
        inliers_idx = filter_statistical_outlier(self.all_points, nb_neighbors=20, std_ratio=2.0)
        # 外れ値ノイズが除去され、点数が減少していることを確認
        self.assertLess(len(inliers_idx), 1050)
        self.assertGreater(len(inliers_idx), 950)

    def test_filter_radius_outlier(self):
        """半径ベース外れ値除去 (ROR) のテスト"""
        inliers_idx = filter_radius_outlier(self.all_points, nb_points=10, radius=0.8)
        self.assertLess(len(inliers_idx), 1050)
        self.assertGreater(len(inliers_idx), 700)

    def test_filter_distance(self):
        """距離フィルタのテスト"""
        # 原点から2m以内の点を抽出
        indices = filter_distance(self.all_points, min_dist=0.0, max_dist=2.0)
        self.assertGreater(len(indices), 0)
        self.assertLess(len(indices), 1050)

    def test_voxel_downsample(self):
        """ボクセル間引きのテスト"""
        indices = voxel_downsample_indices(self.all_points, voxel_size=0.1)
        self.assertLess(len(indices), 1050)
        self.assertGreater(len(indices), 100)

    def test_full_pipeline(self):
        """ファイル全体の処理パイプラインおよび再書き出しテスト"""
        config = FilterConfig(
            use_sor=True,
            sor_neighbors=20,
            sor_std_ratio=2.0,
            use_ror=False,
            use_voxel_downsample=False,
        )
        processor = PointCloudProcessor(config=config)
        summary = processor.process_file(self.test_input, self.test_output)

        self.assertEqual(summary.total_initial_points, 1050)
        self.assertLess(summary.total_final_points, 1050)
        self.assertGreater(summary.total_removed_points, 0)
        self.assertTrue(os.path.exists(self.test_output))

        # 書き出されたE57を再読み込みして検証
        reloaded_scans = read_e57_scans(self.test_output)
        self.assertEqual(len(reloaded_scans), 1)
        self.assertEqual(reloaded_scans[0].point_count, summary.total_final_points)
        # 属性フィールドの要素数が一致していることを確認
        self.assertEqual(len(reloaded_scans[0].raw_fields["colorRed"]), summary.total_final_points)
        self.assertEqual(len(reloaded_scans[0].raw_fields["intensity"]), summary.total_final_points)

    def test_multiple_scans(self):
        """複数スキャン（マルチステーション）のE57ファイル処理テスト"""
        multi_input = "test_multi_input.e57"
        multi_output = "test_multi_output.e57"

        # 2つのスキャンを持つファイルを作成
        with pye57.E57(multi_input, mode="w") as e57:
            for s_idx in range(2):
                pts = self.all_points + (s_idx * 5.0)  # 位置をずらす
                data = {
                    "cartesianX": pts[:, 0],
                    "cartesianY": pts[:, 1],
                    "cartesianZ": pts[:, 2],
                    "colorRed": self.colors[:, 0],
                    "colorGreen": self.colors[:, 1],
                    "colorBlue": self.colors[:, 2],
                    "intensity": self.intensities,
                }
                e57.write_scan_raw(data)

        try:
            processor = PointCloudProcessor(config=FilterConfig(use_sor=True))
            summary = processor.process_file(multi_input, multi_output)
            self.assertEqual(len(summary.scan_results), 2)
            self.assertEqual(summary.total_initial_points, 1050 * 2)
            self.assertTrue(os.path.exists(multi_output))

            reloaded = read_e57_scans(multi_output)
            self.assertEqual(len(reloaded), 2)
            self.assertEqual(reloaded[0].point_count, summary.scan_results[0].final_points)
            self.assertEqual(reloaded[1].point_count, summary.scan_results[1].final_points)
        finally:
            for f in [multi_input, multi_output]:
                if os.path.exists(f):
                    os.remove(f)


if __name__ == "__main__":
    unittest.main()
