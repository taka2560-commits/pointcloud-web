"""
E57 読み込み点数の整合性テスト

背景: 全点読み込みでは np.empty で宣言点数分のバッファを確保するため、読み込みが途中で止まると
      未初期化メモリがそのまま座標として残り得た。_reconcile_read_count による照合・切り詰めを検証する。
"""

import os
import unittest
import numpy as np

from core.e57_io import _reconcile_read_count, read_e57_scans

SAMPLE_E57 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "samples", "サンプル点群.e57")


class TestReconcileReadCount(unittest.TestCase):
    def _stores(self, n):
        raw_data = {"intensity": np.arange(n, dtype=np.float32), "cartesianX": np.arange(n, dtype=np.float64)}
        raw_unscaled = {"cartesianY": np.arange(n, dtype=np.int64)}
        return raw_data, raw_unscaled

    def test_exact_count_leaves_arrays_untouched(self):
        raw_data, raw_unscaled = self._stores(100)
        ref = raw_data["intensity"]
        _reconcile_read_count(0, 100, 100, raw_data, raw_unscaled)
        self.assertIs(raw_data["intensity"], ref)
        self.assertEqual(len(raw_unscaled["cartesianY"]), 100)

    def test_short_read_trims_all_arrays(self):
        raw_data, raw_unscaled = self._stores(100)
        _reconcile_read_count(0, 100, 73, raw_data, raw_unscaled)
        self.assertEqual(len(raw_data["intensity"]), 73)
        self.assertEqual(len(raw_data["cartesianX"]), 73)
        self.assertEqual(len(raw_unscaled["cartesianY"]), 73)
        np.testing.assert_array_equal(raw_unscaled["cartesianY"], np.arange(73))

    def test_zero_read_raises(self):
        raw_data, raw_unscaled = self._stores(100)
        with self.assertRaises(ValueError):
            _reconcile_read_count(0, 100, 0, raw_data, raw_unscaled)

    def test_zero_declared_is_noop(self):
        raw_data, raw_unscaled = {}, {}
        _reconcile_read_count(0, 0, 0, raw_data, raw_unscaled)


@unittest.skipUnless(os.path.exists(SAMPLE_E57), "samples/サンプル点群.e57 が見つかりません")
class TestSampleFileReadsCompletely(unittest.TestCase):
    """同梱サンプル点群がヘッダー宣言点数どおりに読み込めること（回帰確認）"""

    def test_sample_point_count_matches_header(self):
        scans = read_e57_scans(SAMPLE_E57)
        self.assertGreaterEqual(len(scans), 1)
        for s in scans:
            self.assertGreater(s.point_count, 0)
            self.assertEqual(s.point_count, s.original_point_count)
            self.assertTrue(np.isfinite(s.points).all())


if __name__ == "__main__":
    unittest.main()
