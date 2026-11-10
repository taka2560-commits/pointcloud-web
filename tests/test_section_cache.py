"""
断面ボックスの読み直しキャッシュ（core.section_cache）のテスト
- 初回の読み直しでキャッシュが作られ、2 回目はキャッシュから同じ点（座標・色・強度）が返ること
- 箱に掛かるチャンクだけを読むこと（チャンク数 < 総チャンク数）
- 範囲外で読まなかったスキャンは後で必要になったときに追加されること（スキャン単位の追加）
- 元ファイルが変わると（署名不一致）作り直すこと
- 中止したら部分ファイルが残らず、次回は普通に作れること
- 書けない場所なら警告だけでキャッシュ無しの結果になること、remove_cache で消えること
- MainWindow から 2 回押すと 2 回目はキャッシュを使うこと
"""

import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from tests.settings_sandbox import use_sandbox_settings  # noqa: E402

use_sandbox_settings()  # この PC の本番アプリの保存設定を、テストで書き換えない

import numpy as np
from PySide6.QtWidgets import QApplication

from core.e57_io import read_e57_scans
from core.section_cache import SectionCache, cache_dir_for, remove_cache
from core.section_reload import SectionBox, reload_section_points
from test_section_reload import _Scene, _as_set

CHUNK = 5_000  # 小さなチャンクで多数のチャンクを作り、読み飛ばしを確かめる


class TestSectionCache(_Scene):
    def setUp(self):
        remove_cache(self.path)
        self.cache_tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        remove_cache(self.path)
        self.cache_tmp.cleanup()

    def _reload(self, box, **kw):
        kw.setdefault("max_points", 0)
        kw.setdefault("chunk_size", CHUNK)
        return reload_section_points(self._subsampled(), box, **kw)

    def _assert_same_result(self, a, b):
        """座標はビット単位で同じ（生整数から同じ式で復元）。色は 8bit、強度は float16 に落としている"""
        self.assertEqual(a.n_in_box, b.n_in_box)
        self.assertEqual(a.n_points, b.n_points)
        order_a = np.lexsort((a.points[:, 2], a.points[:, 1], a.points[:, 0]))
        order_b = np.lexsort((b.points[:, 2], b.points[:, 1], b.points[:, 0]))
        np.testing.assert_array_equal(a.points[order_a], b.points[order_b])
        np.testing.assert_allclose(a.colors[order_a], b.colors[order_b], atol=1.0 / 255 + 1e-6)
        np.testing.assert_allclose(a.intensities[order_a], b.intensities[order_b], rtol=2e-3, atol=1e-3)
        np.testing.assert_array_equal(a.scan_ids[order_a], b.scan_ids[order_b])

    def test_build_then_use_cache(self):
        """初回はデコードしてキャッシュを作り、2 回目はキャッシュから同じ結果を返す（箱に掛かるチャンクだけ読む）"""
        box = SectionBox(x_range=(1.0, 2.0), z_range=(-0.01, 0.01))
        first = self._reload(box)
        self.assertEqual(first.cache_used_scans, 0)
        self.assertEqual(first.cache_built_scans, 2, "A と（bounds の無い）C が作られる")
        cdir = cache_dir_for(self.path)
        self.assertEqual(first.cache_dir, cdir)
        self.assertTrue(os.path.isdir(cdir))
        self.assertTrue(os.path.isfile(os.path.join(cdir, "header.json")))
        self.assertTrue(os.path.isfile(os.path.join(cdir, "scan00_xyz.npy")))
        self.assertTrue(os.path.isfile(os.path.join(cdir, "scan00_rgb.npy")))
        self.assertTrue(os.path.isfile(os.path.join(cdir, "scan00_int.npy")))
        self.assertTrue(os.path.isfile(os.path.join(cdir, "scan00_chunks.npy")))
        self.assertFalse(any(n.endswith(".part.npy") for n in os.listdir(cdir)))
        self.assertGreater(first.cache_bytes, 17 * 40_000)

        second = self._reload(box)
        self.assertEqual(second.cache_used_scans, 2)
        self.assertEqual(second.cache_built_scans, 0)
        self.assertGreater(second.n_chunks_total, 8)
        self.assertLess(second.n_chunks_read, second.n_chunks_total, "箱に掛からないチャンクは読まない")
        self.assertLess(second.n_read, first.n_read)
        self._assert_same_result(first, second)
        # 正解（全点読込）とも一致
        idx = self._truth_in(box)
        self.assertEqual(second.n_in_box, len(idx))
        self.assertEqual(_as_set(second.points, 3), _as_set(self.truth_pts[idx], 3))

    def test_scan_added_later(self):
        """範囲外で読まなかったスキャンは、後で箱に入ったときにキャッシュへ追加される"""
        box_b = SectionBox(x_range=(97.0, 101.0), y_range=(199.0, 203.0))
        r1 = self._reload(box_b)
        cache = SectionCache.open(self.path, create=False)
        self.assertIsNotNone(cache)
        self.assertEqual(sorted(cache.scans.keys()), [1, 2], "B と bounds の無い C。A は範囲外で読まない")
        self.assertEqual(r1.cache_built_scans, 2)

        box_a = SectionBox(x_range=(1.0, 2.0))
        r2 = self._reload(box_a)
        self.assertEqual(r2.cache_built_scans, 1, "A を追加")
        self.assertEqual(r2.cache_used_scans, 1, "C はキャッシュから")
        cache = SectionCache.open(self.path, create=False)
        self.assertEqual(sorted(cache.scans.keys()), [0, 1, 2])

        r3 = self._reload(box_a)
        self.assertEqual(r3.cache_used_scans, 2)
        self.assertEqual(r3.cache_built_scans, 0)
        self._assert_same_result(r2, r3)
        # pose 付き B もキャッシュ経由で正解と一致
        r4 = self._reload(box_b)
        self.assertEqual(r4.cache_used_scans, 2)
        self.assertEqual(_as_set(r4.points, 3), _as_set(self.truth_pts[self._truth_in(box_b)], 3))

    def test_rebuild_when_source_changes(self):
        """元ファイルの更新時刻が変わると古いキャッシュを捨てて作り直す"""
        box = SectionBox(x_range=(1.0, 2.0))
        self._reload(box)
        cdir = cache_dir_for(self.path)
        marker = os.path.join(cdir, "stale.txt")
        with open(marker, "w", encoding="utf-8") as f:
            f.write("old")
        st = os.stat(self.path)
        os.utime(self.path, ns=(st.st_atime_ns, st.st_mtime_ns + 2_000_000_000))
        try:
            scans = read_e57_scans(self.path, point_budget=self.load_budget)  # 新しい署名で読み直す
            res = reload_section_points(scans, box, max_points=0, chunk_size=CHUNK)
            self.assertEqual(res.cache_used_scans, 0)
            self.assertEqual(res.cache_built_scans, 2)
            self.assertFalse(os.path.exists(marker), "古いフォルダごと捨てられる")
            cache = SectionCache.open(self.path, create=False)
            self.assertEqual(tuple(cache.signature), (st.st_size, st.st_mtime_ns + 2_000_000_000))
        finally:
            os.utime(self.path, ns=(st.st_atime_ns, st.st_mtime_ns))

    def test_cancel_leaves_no_partial_files(self):
        """途中で中止すると部分ファイルは消え、header にスキャンは登録されない。次回は普通に作れる"""
        box = SectionBox(x_range=(1.0, 2.0))
        n_checks = []

        def cancelled():
            n_checks.append(1)
            return len(n_checks) > 3

        with self.assertRaises(InterruptedError):
            self._reload(box, is_cancelled=cancelled)
        cdir = cache_dir_for(self.path)
        names = os.listdir(cdir) if os.path.isdir(cdir) else []
        self.assertFalse(any(n.endswith(".part.npy") for n in names), names)
        cache = SectionCache.open(self.path, create=False)
        self.assertTrue(cache is None or len(cache.scans) == 0)
        res = self._reload(box)
        self.assertEqual(res.cache_built_scans, 2)
        self.assertEqual(res.n_in_box, len(self._truth_in(box)))

    def test_unwritable_location_falls_back(self):
        """キャッシュフォルダが作れない場所でも、結果は正しくキャッシュ無しで返る"""
        blocker = os.path.join(self.cache_tmp.name, "blocked")
        with open(blocker, "w", encoding="utf-8") as f:
            f.write("not a directory")
        box = SectionBox(x_range=(1.0, 2.0))
        res = self._reload(box, cache_dir=blocker)
        self.assertEqual(res.cache_built_scans, 0)
        self.assertEqual(res.cache_used_scans, 0)
        self.assertIsNone(res.cache_dir)
        self.assertEqual(res.n_in_box, len(self._truth_in(box)))
        # 別フォルダ指定でも作れて使える
        res2 = self._reload(box, cache_dir=self.cache_tmp.name)
        self.assertEqual(res2.cache_built_scans, 2)
        self.assertTrue(res2.cache_dir.startswith(self.cache_tmp.name))
        res3 = self._reload(box, cache_dir=self.cache_tmp.name)
        self.assertEqual(res3.cache_used_scans, 2)
        self._assert_same_result(res2, res3)

    def test_header_records_raw_integer_storage(self):
        """ScaledInteger の元ファイルは生整数 int32 と scale/offset で持ち、復元が全点読込と一致する"""
        box = SectionBox(x_range=(1.0, 2.0))
        self._reload(box)
        cache = SectionCache.open(self.path, create=False)
        meta = cache.scans[0]
        self.assertEqual(meta["xyz_kind"], "int32")
        self.assertEqual(meta["xyz_scale"], [0.001, 0.001, 0.001])
        self.assertEqual(meta["xyz_offset"], [10.0, -5.0, 2.0])
        cs = cache.scan(0)
        self.assertEqual(cs.xyz.dtype, np.int32)
        whole = np.vstack([cs.load_chunk(k)[0] for k in range(len(cs.chunks))])
        np.testing.assert_array_equal(whole, self.full[0].points, "キャッシュから復元した座標が全点読込とビット単位で一致する")

    def test_float_double_source_is_exact(self):
        """倍精度 Float XYZ の元ファイルは float64 のまま持ち、キャッシュ経由でも結果が一致する"""
        from test_section_reload import _grid, _write_e57, _display_truth

        rng = np.random.default_rng(21)
        pts_a = _grid(0.0, 2.0, 0.0, 1.0, 0.0, 0.01) + rng.normal(0, 1e-4, (20_301, 3))
        pts_b = _grid(0.0, 1.0, 0.0, 1.0, 0.0, 0.01)
        path = os.path.join(self.cache_tmp.name, "float.e57")
        specs = []
        for name, guid, pts, pose in (
            ("FA", "{dddddddd-0000-0000-0000-000000000001}", pts_a, None),
            ("FB", "{eeeeeeee-0000-0000-0000-000000000002}", pts_b, ((np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)), (50.0, 60.0, 1.0))),
        ):
            specs.append(dict(
                points=pts, name=name, guid=guid, pose=pose, bounds=True,
                colors=(rng.random((len(pts), 3)) * 255).astype(np.uint8),
                intensity=rng.random(len(pts)).astype(np.float32),
            ))
        _write_e57(path, specs, float_xyz=True)
        full = read_e57_scans(path)
        np.testing.assert_allclose(full[0].points, pts_a, atol=1e-12)
        truth_pts, _, _, _ = _display_truth(full)
        scans = read_e57_scans(path, point_budget=10_000)
        self.assertTrue(all(s.is_subsampled for s in scans))
        box = SectionBox(x_range=(0.5, 1.0), y_range=(0.0, 0.5))
        r1 = reload_section_points(scans, box, max_points=0, chunk_size=CHUNK)
        self.assertEqual(r1.cache_built_scans, 1)
        cache = SectionCache.open(path, create=False)
        self.assertEqual(cache.scans[0]["xyz_kind"], "float64")
        r2 = reload_section_points(scans, box, max_points=0, chunk_size=CHUNK)
        self.assertEqual(r2.cache_used_scans, 1)
        self._assert_same_result(r1, r2)
        idx = np.flatnonzero(box.contains(truth_pts))
        self.assertEqual(r2.n_in_box, len(idx))
        self.assertEqual(_as_set(r2.points, 6), _as_set(truth_pts[idx], 6))
        # pose 付き FB は範囲外なので読まない
        self.assertEqual(r2.n_scans_skipped, 1)
        remove_cache(path)

    def test_use_cache_false_and_remove(self):
        box = SectionBox(x_range=(1.0, 2.0))
        res = self._reload(box, use_cache=False)
        self.assertIsNone(res.cache_dir)
        self.assertFalse(os.path.isdir(cache_dir_for(self.path)))
        self._reload(box)
        self.assertTrue(os.path.isdir(cache_dir_for(self.path)))
        self.assertTrue(remove_cache(self.path))
        self.assertFalse(os.path.isdir(cache_dir_for(self.path)))
        self.assertFalse(remove_cache(self.path))


class TestMainWindowCache(_Scene):
    def setUp(self):
        from gui.main_window import MainWindow

        remove_cache(self.path)
        self.win = MainWindow()
        self.win._reset_app_settings()
        self.win.show()
        QApplication.processEvents()

    def tearDown(self):
        self.win.close()
        self.win.deleteLater()
        QApplication.processEvents()
        remove_cache(self.path)

    def test_second_click_uses_cache(self):
        win = self.win
        win.combo_load_budget.addItem("テスト用 2 万点", self.load_budget)
        win.combo_load_budget.setCurrentIndex(win.combo_load_budget.count() - 1)
        self.assertTrue(win._load_file_to_viewer(self.path) is not False)
        self.assertTrue(win._wait_for_background(60_000))
        QApplication.processEvents()
        win.spin_pos_z.setValue(0.0)
        win.spin_thick_z.setValue(0.1)
        win.chk_sec_z.setChecked(True)

        win.btn_sec_reload.click()
        self.assertTrue(win._wait_for_background(60_000))
        QApplication.processEvents()
        log1 = win.log_text.toPlainText()
        self.assertIn("キャッシュ作成", log1)
        self.assertTrue(os.path.isdir(cache_dir_for(self.path)))
        n1 = len(win.viewer.overlay_points)

        win.btn_sec_reload_clear.click()
        win.btn_sec_reload.click()
        self.assertTrue(win._wait_for_background(60_000))
        QApplication.processEvents()
        log2 = win.log_text.toPlainText()[len(log1):]
        self.assertIn("キャッシュから", log2)
        self.assertEqual(len(win.viewer.overlay_points), n1)
        win.viewer.validate_invariants()


if __name__ == "__main__":
    unittest.main()
