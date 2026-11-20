# -*- coding: utf-8 -*-
"""
全点表示（断面の「元ファイルから全点で読み直す」）の点を、選んで消す
（core/fullres_marks.py・gui/overlay_edit.py・core/section_reload.py の rows・全点で保存への反映・元に戻す）

2026-11-20、利用者の報告「断面の全点読み込みをすると、ブラシで消えないノイズがあった」から:
- 1 億点超のファイルは「k 点に 1 点」で読み込まれ、ブラシが選べるのは読み込んだ点だけだった
- 全点表示で見える残りの点は表示専用で、面のすぐそばのノイズは消す手段が無かった

守ること:
- 読み直した点は、元ファイルでの行の番号を持つ（デコードでも、キャッシュからでも、上限で間引いても、同じ番号）
- 読み込んだ点と同じ行は、重ね描きでは描かない・選ばない（本体がレイヤーで扱う）
- 同じ形で選ばれる点は、形の中の点の全部（候補のふるいで落とさない）。本体で選ばれる点と重ならない
- 消した点は、箱を作り直しても消えたまま。「全点で保存」で必ず除かれ、ほかの点には影響しない
- 元に戻す・やり直しは、レイヤーと同じ 1 手
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.settings_sandbox import use_sandbox_settings  # noqa: E402

use_sandbox_settings()  # この PC の本番アプリの保存設定を、テストで書き換えない

from PySide6.QtCore import QCoreApplication, QEvent, QPoint, Qt  # noqa: E402
from PySide6.QtGui import QColor  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from core.e57_io import read_e57_scans  # noqa: E402
from core.fullres_marks import FullResMarks, loaded_row_mask  # noqa: E402
from core.section_reload import SectionBox, reload_section_points, to_display_coordinates  # noqa: E402
from gui.layer_history import LayerState, LayerUndoHistory  # noqa: E402
from gui.overlay_edit import (  # noqa: E402
    apply_marks_to_save_worker, hit_brush, hit_circle, hit_polygon, hit_rect, marks_label_for_export,
)
from gui.worker_thread import FullResolutionSaveWorker  # noqa: E402
from tests.test_fullres_save import CAT_SURFACE, STRIDE, _Scene, _scan_keys  # noqa: E402

NO_MOD = Qt.KeyboardModifier(0)
# _Scene (tests/test_fullres_save.py) は QCoreApplication を作る。先に作られると、あとから QApplication を作れず、
# ウィジェットを作った所でプロセスが止まる。このファイルだけを実行するときのために、ここで QApplication を作っておく
_APP = QApplication.instance() or QApplication(sys.argv)


# ============================================================================= 印（消した点）の集まり
class TestMarks(unittest.TestCase):
    def test_add_contains_and_counts(self):
        m = FullResMarks()
        self.assertTrue(m.is_empty())
        sids = np.array([0, 0, 1, 0, 1], dtype=np.int32)
        rows = np.array([5, 2, 5, 5, -1], dtype=np.int64)
        self.assertEqual(m.add(sids, rows), 3, "重なりと、番号が負の物は数えない")
        self.assertEqual(m.count(), 3)
        np.testing.assert_array_equal(m.rows_of(0), [2, 5])
        np.testing.assert_array_equal(m.rows_of(1), [5])
        self.assertEqual(m.add(sids, rows), 0, "同じ点をもう一度消しても増えない")
        got = m.contains(np.array([0, 0, 1, 1, 2]), np.array([2, 3, 5, 2, 5]))
        np.testing.assert_array_equal(got, [True, False, True, False, False])
        self.assertEqual(len(m.contains(np.empty(0, int), np.empty(0, int))), 0)
        # 印より大きい番号を調べても、範囲を外れない
        np.testing.assert_array_equal(m.contains(np.array([0, 0]), np.array([10**12, 5])), [False, True])

    def test_mask_for_chunk(self):
        m = FullResMarks()
        m.add(np.zeros(4, int), np.array([0, 9, 10, 25]))
        np.testing.assert_array_equal(np.flatnonzero(m.mask_for_chunk(0, 0, 10)), [0, 9])
        np.testing.assert_array_equal(np.flatnonzero(m.mask_for_chunk(0, 10, 10)), [0])
        self.assertIsNone(m.mask_for_chunk(0, 11, 10), "印の無い区間は None（保存は何もしない）")
        np.testing.assert_array_equal(np.flatnonzero(m.mask_for_chunk(0, 20, 10)), [5])
        self.assertIsNone(m.mask_for_chunk(1, 0, 100), "ほかのスキャンには付かない")
        self.assertIsNone(m.mask_for_chunk(0, 0, 0))
        # 区間に分けて集めると、全部の印になる
        got = []
        for start in range(0, 30, 7):
            mask = m.mask_for_chunk(0, start, 7)
            if mask is not None:
                got += (np.flatnonzero(mask) + start).tolist()
        self.assertEqual(got, [0, 9, 10, 25])

    def test_mask_for_rows_with_gaps(self):
        """保存の流し読みは「数でない座標の点」を除いて返すので、行の番号は続き番号とは限らない"""
        m = FullResMarks()
        m.add(np.zeros(5, int), np.array([3, 7, 8, 20, 99]))
        rows = np.array([0, 1, 2, 3, 6, 8, 9, 19, 21, 50])       # 4, 5, 7, 20 は除かれている
        np.testing.assert_array_equal(np.flatnonzero(m.mask_for_rows(0, rows)), [3, 5])
        self.assertIsNone(m.mask_for_rows(0, np.array([4, 5, 6, 9, 10])))
        self.assertIsNone(m.mask_for_rows(0, np.array([100, 101])))
        self.assertIsNone(m.mask_for_rows(0, np.empty(0, int)))
        self.assertIsNone(m.mask_for_rows(1, rows))
        # 続き番号なら mask_for_chunk と同じ
        rng = np.random.default_rng(0)
        big = FullResMarks()
        big.add(np.zeros(300, int), rng.integers(0, 2000, 300))
        for start in range(0, 2000, 137):
            a = big.mask_for_chunk(0, start, 137)
            b = big.mask_for_rows(0, np.arange(start, start + 137))
            self.assertEqual(a is None, b is None)
            if a is not None:
                np.testing.assert_array_equal(a, b)

    def test_copy_does_not_follow_later_changes(self):
        m = FullResMarks()
        m.add(np.array([0]), np.array([1]))
        c = m.copy()
        m.add(np.array([0]), np.array([2]))
        m.clear()
        np.testing.assert_array_equal(c.rows_of(0), [1])

    def test_toggle_diff_and_snapshot(self):
        m = FullResMarks()
        before = m.snapshot()
        m.add(np.array([0, 0, 2]), np.array([3, 4, 7]))
        after = m.snapshot()
        d = FullResMarks.diff(before, after)
        self.assertEqual(FullResMarks.diff_count(d), 3)
        self.assertGreater(FullResMarks.diff_nbytes(d), 0)
        self.assertIsNone(FullResMarks.diff(after, m.snapshot()), "変わっていなければ None")
        m.toggle(d)
        self.assertTrue(m.is_empty(), "同じ差分を当てると、元に戻る")
        m.toggle(d)
        self.assertEqual(m.count(), 3, "もう一度当てると、やり直しになる")
        # 写しは、あとから変えても変わらない
        m.add(np.array([0]), np.array([99]))
        np.testing.assert_array_equal(after[0], [3, 4])
        with self.assertRaises(ValueError):
            m.rows_of(0)[0] = 1   # 配列は書き換え不可

    def test_listener_version_and_clear(self):
        m = FullResMarks()
        calls = []
        m.add_listener(lambda: calls.append(m.count()))
        m.add(np.array([0]), np.array([1]))
        m.add(np.array([0]), np.array([1]))   # 変わらないので、知らせない
        v = m.version
        self.assertEqual(m.clear(), 1)
        self.assertEqual(m.clear(), 0)
        self.assertEqual(calls, [1, 0])
        self.assertEqual(m.version, v + 1)

    def test_loaded_rows(self):
        rows = np.arange(10)
        np.testing.assert_array_equal(np.flatnonzero(loaded_row_mask(rows, 3)), [0, 3, 6, 9])
        self.assertTrue(loaded_row_mask(rows, 1).all(), "間引いていなければ、全部が読み込んだ点")
        self.assertTrue(loaded_row_mask(rows, None).all())


# ============================================================================= 元に戻す（レイヤーと同じ 1 手）
class TestHistoryWithMarks(unittest.TestCase):
    def _setup(self):
        ids = np.zeros(10, dtype=np.int32)
        layers = {0: "normal", 1: "noise"}
        h = LayerUndoHistory()
        m = FullResMarks()
        h.attach_extra(m)
        return ids, layers, h, m

    def test_marks_only_step(self):
        ids, layers, h, m = self._setup()
        h.push(LayerState(ids, layers))
        m.add(np.array([0, 0]), np.array([4, 5]))
        self.assertTrue(h.can_undo())
        st = h.undo(LayerState(ids, layers))
        self.assertIsNotNone(st, "レイヤーが変わっていなくても、1 手として記録する")
        self.assertEqual(len(st.changed), 0)
        self.assertTrue(m.is_empty())
        self.assertTrue(h.can_redo())
        h.redo(LayerState(ids, layers))
        self.assertEqual(m.count(), 2)
        self.assertTrue((ids == 0).all())

    def test_layer_and_marks_in_one_step(self):
        ids, layers, h, m = self._setup()
        h.push(LayerState(ids, layers))
        ids[2:4] = 1
        m.add(np.array([0]), np.array([7]))
        h.push(LayerState(ids, layers))      # 次の操作
        ids[8] = 1
        m.add(np.array([1]), np.array([3]))
        self.assertEqual(len(h.undo_stack), 1)
        h.undo(LayerState(ids, layers))
        self.assertEqual(int(ids[8]), 0)
        np.testing.assert_array_equal(ids[2:4], [1, 1])
        self.assertEqual(m.count(), 1)
        h.undo(LayerState(ids, layers))
        self.assertTrue((ids == 0).all())
        self.assertTrue(m.is_empty())
        h.redo(LayerState(ids, layers))
        h.redo(LayerState(ids, layers))
        self.assertEqual(m.count(), 2)
        self.assertEqual(int(ids[8]), 1)

    def test_nothing_changed_is_not_a_step(self):
        ids, layers, h, m = self._setup()
        m.add(np.array([0]), np.array([1]))   # 記録を始める前の印は、手に数えない
        h.push(LayerState(ids, layers))
        self.assertIsNone(h.undo(LayerState(ids, layers)))
        self.assertEqual(m.count(), 1)

    def test_memory_counts_the_marks(self):
        ids, layers, h, m = self._setup()
        h.push(LayerState(ids, layers))
        base = h.memory_bytes()
        m.add(np.zeros(1000, int), np.arange(1000))
        h.push(LayerState(ids, layers))
        self.assertGreaterEqual(h.memory_bytes() - base, 8000)

    def test_without_extra_behaves_as_before(self):
        ids = np.zeros(5, dtype=np.int32)
        h = LayerUndoHistory()
        h.push(LayerState(ids, {}))
        ids[1] = 1
        st = h.undo(LayerState(ids, {}))
        self.assertEqual(int(ids[1]), 0)
        self.assertIsNone(h.undo_stack[-1].extra if h.undo_stack else None)
        self.assertIsNotNone(st)


# ============================================================================= 読み直しが返す行の番号・保存への反映
class TestRowsAndSave(_Scene):
    def setUp(self):
        self.pre = self._load(max_load_points=self.load_budget)

    def test_load_records_the_step(self):
        for s in self.pre.scans:
            self.assertEqual(s.subsample_step, STRIDE)
        full = read_e57_scans(self.path)
        self.assertEqual([s.subsample_step for s in full], [1, 1])

    def _check_rows(self, result):
        """rows の行にある元ファイルの点が、返ってきた点と同じ位置"""
        self.assertEqual(len(result.rows), result.n_points)
        self.assertTrue((result.rows >= 0).all())
        for s_idx, s in enumerate(self.pre.scans):
            sel = result.scan_ids == s_idx
            if not sel.any():
                continue
            expect = to_display_coordinates(s, self.truth[s.name]["points"][result.rows[sel]])
            np.testing.assert_allclose(result.points[sel], expect, atol=1e-4)

    def test_rows_point_at_the_file_rows(self):
        cache_dir = os.path.join(self.tmp.name, "cache_rows")
        box = SectionBox(x_range=(0.5, 2.5), z_range=(-0.5, 3.0))
        first = reload_section_points(self.pre.scans, box, use_cache=True, cache_dir=cache_dir)   # デコード + キャッシュ作成
        self.assertGreater(first.n_points, 5000)
        self.assertGreaterEqual(first.cache_built_scans, 1)
        self._check_rows(first)
        second = reload_section_points(self.pre.scans, box, use_cache=True, cache_dir=cache_dir)  # キャッシュから
        self.assertGreaterEqual(second.cache_used_scans, 1)
        self._check_rows(second)
        key = lambda r: set(zip(r.scan_ids.tolist(), r.rows.tolist()))  # noqa: E731
        self.assertEqual(key(first), key(second), "デコードでも、キャッシュからでも、同じ行")
        plain = reload_section_points(self.pre.scans, box, use_cache=False)
        self.assertEqual(key(first), key(plain))
        self.assertEqual(len(key(first)), first.n_points, "同じ行が 2 回出ない")

    def test_rows_survive_the_cap(self):
        box = SectionBox(z_range=(-0.5, 3.0))
        capped = reload_section_points(self.pre.scans, box, max_points=4000, use_cache=False)
        self.assertTrue(capped.capped)
        self.assertLessEqual(capped.n_points, 4000)
        self._check_rows(capped)

    def _save(self, name, layer_mask, marks=None, label=0):
        out = os.path.join(self.tmp.name, name)
        worker = FullResolutionSaveWorker(out, self.pre.scans, layer_mask)
        if marks is not None:
            worker.set_marks(marks, label)
        ok, err = self._run(worker)
        self.assertFalse(err, err)
        return worker, read_e57_scans(out)

    def test_marked_rows_are_removed_and_nothing_else_changes(self):
        cat_j, _ = self._judged_truth(self.pre)
        layer_mask = cat_j == CAT_SURFACE
        t = self.truth["A"]
        p = t["points"]
        rows_all = np.arange(len(p))
        # 床の一部（面の点なので、印が無ければ残る）のうち、読み込みで飛ばされた行だけに印を付ける
        region = (p[:, 0] > 2.0) & (p[:, 0] < 2.3) & (p[:, 1] > 2.0) & (p[:, 1] < 2.3) & (np.abs(p[:, 2]) < 0.01)
        target = rows_all[region & (rows_all % STRIDE != 0)]
        self.assertGreater(len(target), 300)
        marks = FullResMarks()
        marks.add(np.zeros(len(target), np.int32), target)

        _, base = self._save("marks_base.e57", layer_mask)
        worker, got = self._save("marks_on.e57", layer_mask, marks, 0)
        self.assertEqual(worker.total_marked, len(target))
        base_keys = {s.name: set(_scan_keys(s).tolist()) for s in base}
        got_keys = {s.name: set(_scan_keys(s).tolist()) for s in got}
        marked_keys = set(t["keys"][target].tolist())
        self.assertTrue(marked_keys <= base_keys["A"], "印が無ければ、残っていた点")
        self.assertEqual(got_keys["A"], base_keys["A"] - marked_keys, "印の付いた行だけが消える")
        self.assertEqual(got_keys["B"], base_keys["B"], "ほかのスキャンは変わらない")
        # 同じ場所の、読み込んだ行（印を付けない行）は残っている
        loaded_in_region = rows_all[region & (rows_all % STRIDE == 0)]
        self.assertTrue(set(t["keys"][loaded_in_region].tolist()) <= got_keys["A"])

    def test_noise_only_export_includes_the_marked_rows(self):
        cat_j, _ = self._judged_truth(self.pre)
        noise_mask = cat_j != CAT_SURFACE      # ノイズだけを保存する
        t = self.truth["A"]
        p = t["points"]
        rows_all = np.arange(len(p))
        region = (p[:, 0] > 2.0) & (p[:, 0] < 2.3) & (p[:, 1] > 2.0) & (p[:, 1] < 2.3) & (np.abs(p[:, 2]) < 0.01)
        target = rows_all[region & (rows_all % STRIDE != 0)]
        marks = FullResMarks()
        marks.add(np.zeros(len(target), np.int32), target)
        _, base = self._save("noise_base.e57", noise_mask)
        _, got = self._save("noise_marks.e57", noise_mask, marks, FullResolutionSaveWorker.KEEP_LABEL)
        base_a = set(_scan_keys(base[0]).tolist())
        got_a = set(_scan_keys(got[0]).tolist())
        self.assertEqual(got_a, base_a | set(t["keys"][target].tolist()), "消した点は、ノイズとして出る")

    def test_empty_marks_change_nothing(self):
        cat_j, _ = self._judged_truth(self.pre)
        worker = FullResolutionSaveWorker(os.path.join(self.tmp.name, "x.e57"), self.pre.scans, cat_j == CAT_SURFACE)
        worker.set_marks(FullResMarks(), 0)
        self.assertIsNone(worker.marks)


class TestRowsWithInvalidPoints(unittest.TestCase):
    """
    座標が数でない点（NaN）を含む Float 座標のスキャンでも、消した点だけが保存から消える。
    2026-11-20 の独立レビューで再現: 読み込みと読み直しは NaN の行も数えるのに、保存は NaN を除いてから数えていて、
    行の番号がずれ、消していない点が消えていた
    """

    N = 3000
    BAD = (4, 5, 100, 1001, 2500)

    @classmethod
    def setUpClass(cls):
        import tempfile

        from pye57 import libe57

        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = os.path.join(cls.tmp.name, "nan.e57")
        rng = np.random.default_rng(5)
        gx, gy = np.meshgrid(np.arange(60) * 0.01, np.arange(50) * 0.01)
        xyz = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(cls.N)]) + rng.normal(0, 0.0003, (cls.N, 3))
        cls.xyz_true = xyz.copy()
        xyz[list(cls.BAD), 1] = np.nan
        imf = libe57.ImageFile(cls.path, "w")
        root = imf.root()
        imf.extensionsAdd("", libe57.E57_V1_0_URI)
        root.set("formatName", libe57.StringNode(imf, "ASTM E57 3D Imaging Data File"))
        root.set("guid", libe57.StringNode(imf, "{nan}"))
        d3 = libe57.VectorNode(imf, True)
        root.set("data3D", d3)
        scan = libe57.StructureNode(imf)
        scan.set("guid", libe57.StringNode(imf, "{nan-scan}"))
        proto = libe57.StructureNode(imf)
        for ax in "XYZ":
            proto.set(f"cartesian{ax}", libe57.FloatNode(imf, 0.0, libe57.E57_DOUBLE, -1e9, 1e9))
        cv = libe57.CompressedVectorNode(imf, proto, libe57.VectorNode(imf, True))
        scan.set("points", cv)
        d3.append(scan)
        bufs = libe57.VectorSourceDestBuffer()
        cols = [np.ascontiguousarray(xyz[:, i]) for i in range(3)]
        for i, ax in enumerate("XYZ"):
            bufs.append(libe57.SourceDestBuffer(imf, f"cartesian{ax}", cols[i], cls.N, True, True))
        wr = cv.writer(bufs)
        wr.write(cls.N)
        wr.close()
        imf.close()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _rows_of(self, pts):
        """点の座標から、元ファイルの行を引く（座標は倍精度のまま書いてあるので、ぴったり一致する）"""
        key = {tuple(p): i for i, p in enumerate(self.xyz_true)}
        return np.array([key[tuple(p)] for p in pts])

    def test_reload_rows_count_the_invalid_points(self):
        scans = read_e57_scans(self.path, point_budget=self.N // 3 + 1)
        self.assertEqual(scans[0].subsample_step, 3)
        box = SectionBox(z_range=(-1.0, 1.0))
        res = reload_section_points(scans, box, use_cache=False)
        self.assertEqual(res.n_points, self.N - len(self.BAD))
        np.testing.assert_array_equal(res.rows, self._rows_of(res.points), "除いた点も数えた番号")
        # キャッシュは詰めて持つので、除いた点があるスキャンは行が分からない（-1 = 表示専用のまま）
        cache_dir = os.path.join(self.tmp.name, "cache")
        reload_section_points(scans, box, use_cache=True, cache_dir=cache_dir)
        cached = reload_section_points(scans, box, use_cache=True, cache_dir=cache_dir)
        self.assertGreaterEqual(cached.cache_used_scans, 1)
        self.assertTrue((cached.rows == -1).all())

    def test_save_removes_exactly_the_marked_rows(self):
        scans = read_e57_scans(self.path, point_budget=self.N // 3 + 1)
        target = np.array([7, 8, 103, 1003, 2600, 2999])
        marks = FullResMarks()
        marks.add(np.zeros(len(target), np.int32), target)
        layer_mask = np.ones(len(scans[0].points), dtype=bool)
        for chunk in (1_000_000, 64):          # 1 回で読む場合と、細かく分けて読む場合
            out = os.path.join(self.tmp.name, f"out_{chunk}.e57")
            worker = FullResolutionSaveWorker(out, scans, layer_mask)
            worker.stream_chunk_size = chunk
            worker.set_marks(marks, 0)
            errors = []
            worker.finished_error.connect(errors.append)
            worker.run()
            self.assertFalse(errors, errors)
            self.assertEqual(worker.total_marked, len(target))
            got = set(self._rows_of(read_e57_scans(out)[0].points).tolist())
            want = set(range(self.N)) - set(self.BAD) - set(target.tolist())
            self.assertEqual(got, want, f"チャンク {chunk}: 消した点だけが消える")


# ============================================================================= ビューア: 選ぶ・消す・描く
class _ViewerCase(unittest.TestCase):
    STEP = 3

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def setUp(self):
        from gui.gl_viewer import PointCloudViewer

        self.Viewer = PointCloudViewer
        self.v = PointCloudViewer()
        self.v.resize(800, 600)
        self.v.show()
        self._pump()
        # 3m × 2m の面（行の順に並ぶ）。6 万点。読み込んだ点は 3 点に 1 点
        gx, gy = np.meshgrid(np.arange(300) * 0.01, np.arange(200) * 0.01)
        self.full = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)]).astype(np.float64)
        self.n = len(self.full)
        self.rows = np.arange(self.n, dtype=np.int64)
        self.v.set_point_cloud(self.full[:: self.STEP].copy(), None, None)
        self.v.set_section_z(True, -1.0, 1.0)
        self.v.set_view_top()
        self._pump()
        self._attach()

    def tearDown(self):
        self.v.close()
        self.v.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        self._pump()

    def _pump(self, n=6):
        for _ in range(n):
            QApplication.processEvents()

    def _attach(self, steps=None, rows="default"):
        self.v.set_section_overlay(
            self.full, None, None, np.zeros(self.n, dtype=np.int32), n_in_box=self.n,
            rows=self.rows if isinstance(rows, str) else rows,
            scan_steps=[self.STEP] if steps is None else steps,
        )
        self._pump()

    def _screen(self):
        return self.v._project_to_screen(self.v._overlay_source[0])

    def _brush(self, pts, radius=18, mods=NO_MOD):
        v = self.v
        v.selection_shape = v.SELECT_BRUSH
        v.brush_radius = radius
        v.brush_stroke_points = [QPoint(int(x), int(y)) for x, y in pts]
        v._last_selection_modifiers = mods
        v._process_brush_selection()
        stroke = np.array([[p.x(), p.y()] for p in v.brush_stroke_points], dtype=np.float32)   # 画面の位置は整数に丸められる
        v.brush_stroke_points = []
        return stroke, float(max(4, radius))

    def _center_stroke(self):
        sx, sy, _ = self._screen()
        i = self.n // 2 + 150
        return [(sx[i] - 20, sy[i] - 6), (sx[i] + 25, sy[i] + 9)]


class TestViewerPick(_ViewerCase):
    def test_loaded_rows_are_left_to_the_main_cloud(self):
        v = self.v
        loaded = self.rows % self.STEP == 0
        self.assertFalse(v._ov_visible[loaded].any(), "読み込んだ点と同じ行は、重ね描きでは描かない")
        self.assertTrue(v._ov_visible[~loaded].all())
        self.assertEqual(v.overlay_editable_count(), int((~loaded).sum()))

    def test_brush_picks_every_point_in_the_shape_once(self):
        v = self.v
        stroke, r = self._brush(self._center_stroke())
        sx, sy, vz = self._screen()
        inside = hit_brush(sx, sy, vz, stroke, r)
        self.assertGreater(int(inside.sum()), 200)
        loaded = self.rows % self.STEP == 0
        np.testing.assert_array_equal(v.overlay_selection(), np.flatnonzero(inside & ~loaded), "形の中の、飛ばされた点の全部")
        np.testing.assert_array_equal(v.selected_indices, np.flatnonzero(inside & loaded) // self.STEP, "読み込んだ点は、本体で")
        self.assertEqual(v.overlay_selection_count() + len(v.selected_indices), int(inside.sum()), "重ならず、漏れない")

    def test_rect_circle_polygon(self):
        v = self.v
        sx, sy, vz = self._screen()
        loaded = self.rows % self.STEP == 0
        i = self.n // 2 + 150
        cx, cy = float(sx[i]), float(sy[i])
        v._last_selection_modifiers = NO_MOD
        # 矩形
        v.selection_shape = v.SELECT_RECT
        v.selection_start, v.selection_end = QPoint(int(cx - 30), int(cy - 20)), QPoint(int(cx + 30), int(cy + 20))
        v._process_selection()
        from PySide6.QtCore import QRect

        rect = QRect(v.selection_start, v.selection_end).normalized()
        want = hit_rect(sx, sy, vz, rect.left(), rect.right(), rect.top(), rect.bottom())
        np.testing.assert_array_equal(v.overlay_selection(), np.flatnonzero(want & ~loaded))
        # 円
        v.selection_shape = v.SELECT_CIRCLE
        v._process_selection()
        radius = float(np.hypot(v.selection_end.x() - v.selection_start.x(), v.selection_end.y() - v.selection_start.y()))
        want = hit_circle(sx, sy, vz, v.selection_start.x(), v.selection_start.y(), radius)
        np.testing.assert_array_equal(v.overlay_selection(), np.flatnonzero(want & ~loaded))
        # 多角形
        v.selection_shape = v.SELECT_POLYGON
        v.polygon_points = [QPoint(int(cx - 40), int(cy - 25)), QPoint(int(cx + 35), int(cy - 30)), QPoint(int(cx + 10), int(cy + 40))]
        poly = np.array([[p.x(), p.y()] for p in v.polygon_points], dtype=np.float32)
        v._process_polygon_selection()
        want = hit_polygon(sx, sy, vz, poly)
        self.assertGreater(int(want.sum()), 100)
        np.testing.assert_array_equal(v.overlay_selection(), np.flatnonzero(want & ~loaded))

    def test_shift_adds_and_ctrl_removes(self):
        v = self.v
        a = self._center_stroke()
        self._brush(a)
        first = v.overlay_selection().copy()
        b = [(x + 60, y) for x, y in a]
        self._brush(b, mods=Qt.ShiftModifier)
        both = v.overlay_selection().copy()
        self.assertGreater(len(both), len(first))
        self.assertTrue(set(first.tolist()) <= set(both.tolist()))
        self._brush(a, mods=Qt.ControlModifier)
        left = v.overlay_selection()
        self.assertEqual(set(left.tolist()), set(both.tolist()) - set(first.tolist()))

    def test_selection_is_dropped_when_the_main_selection_is_replaced(self):
        v = self.v
        self._brush(self._center_stroke())
        self.assertGreater(v.overlay_selection_count(), 0)
        v.selected_indices = np.arange(5, dtype=np.int64)   # ほかの操作（連結塊・反転など）が、本体の選択を入れ替えた
        self.assertEqual(v.overlay_selection_count(), 0, "古い選択が残って、あとで消されることが無い")
        self.assertEqual(v.erase_overlay_selection(), 0)
        self._brush(self._center_stroke())
        v.clear_selection()
        self.assertEqual(v.overlay_selection_count(), 0)

    def test_erase_marks_hides_and_survives_a_new_overlay(self):
        v = self.v
        self._brush(self._center_stroke())
        picked = v.overlay_selection().copy()
        self.assertEqual(v.erase_overlay_selection(), len(picked))
        self.assertEqual(v.fullres_marks_count(), len(picked))
        self.assertFalse(v._ov_visible[picked].any())
        np.testing.assert_array_equal(v.fullres_marks.rows_of(0), np.sort(self.rows[picked]))
        self._brush(self._center_stroke())
        self.assertEqual(v.overlay_selection_count(), 0, "消した点は、もう選ばれない")
        # 箱を動かして重ね描きを作り直しても、消えたまま
        v.clear_section_overlay()
        self.assertEqual(v.fullres_marks_count(), len(picked))
        self._attach()
        self.assertFalse(v._ov_visible[picked].any())
        # 全部戻す
        self.assertEqual(v.restore_all_fullres_marks(), len(picked))
        self.assertTrue(v._ov_visible[picked].all())

    def test_instant_delete_hands_the_overlay_points_to_the_window(self):
        v = self.v
        got = []
        v.instant_erase_requested.connect(lambda idx: got.append(np.asarray(idx).copy()))
        v.instant_delete_mode = True
        self._brush(self._center_stroke())
        self.assertEqual(len(got), 1)
        self.assertGreater(len(got[0]), 0)
        n_wait = v.overlay_pending_erase_count()
        self.assertGreater(n_wait, 0)
        self.assertEqual(v.commit_overlay_erase(), n_wait)
        self.assertEqual(v.overlay_pending_erase_count(), 0)
        self.assertEqual(v.fullres_marks_count(), n_wait)
        # 何も無い所をなぞっても、何も起きない
        self._brush([(3, 3), (5, 5)], radius=4)
        self.assertEqual(len(got), 1)

    def test_instant_delete_fires_even_when_only_overlay_points_are_hit(self):
        """読み込んだ点が 1 つも無い所の、飛ばされた点だけのノイズ（今回の報告の場面）"""
        v = self.v
        extra = np.array([[1.5, 1.0, 0.5], [1.5005, 1.0, 0.5]])   # 面から 0.5m 浮いた 2 点。行は 3 の倍数でない
        full = np.vstack([self.full, extra])
        rows = np.concatenate([self.rows, [self.n + 1, self.n + 2]])
        v.set_section_overlay(full, None, None, np.zeros(len(full), np.int32), rows=rows, scan_steps=[self.STEP])
        v.set_view_front()
        self._pump()
        sx, sy, _ = v._project_to_screen(v._overlay_source[0])
        got = []
        v.instant_erase_requested.connect(lambda idx: got.append(len(idx)))
        v.instant_delete_mode = True
        self._brush([(sx[-1], sy[-1])], radius=6)
        self.assertEqual(got, [0], "本体の点が無くても、知らせる")
        self.assertEqual(v.overlay_pending_erase_count(), 2)
        self.assertEqual(v.commit_overlay_erase(), 2)

    def test_marks_travel_with_the_saved_work_state(self):
        """「作業の状態を保存 (.npz)」に、消した点も入る。読み込むと戻る。印の無いファイルを読むと、印は空になる"""
        import tempfile

        v = self.v
        with tempfile.TemporaryDirectory() as d:
            empty = os.path.join(d, "empty.npz")
            v.save_layer_project(empty)
            with np.load(empty) as data:
                self.assertNotIn("fullres_mark_rows", data.files, "印が無ければ、今までと同じ中身")
            self._brush(self._center_stroke())
            v.erase_overlay_selection()
            want = v.fullres_marks.rows_of(0).copy()
            self.assertGreater(len(want), 100)
            full = os.path.join(d, "with_marks.npz")
            v.save_layer_project(full)
            v.restore_all_fullres_marks()
            v.load_layer_project(full)
            np.testing.assert_array_equal(v.fullres_marks.rows_of(0), want)
            picked = np.flatnonzero(np.isin(self.rows, want))
            self.assertFalse(v._ov_visible[picked].any(), "読み込んだら、画面からも消える")
            v.load_layer_project(empty)
            self.assertEqual(v.fullres_marks_count(), 0)

    def test_marks_are_cleared_with_the_point_cloud(self):
        v = self.v
        self._brush(self._center_stroke())
        v.erase_overlay_selection()
        self.assertGreater(v.fullres_marks_count(), 0)
        v.set_point_cloud(self.full[::2].copy(), None, None)
        self.assertEqual(v._ov_marks.count(), 0, "入れ替えた、その場で空にする（次に見たときに、ではない）")
        self.assertEqual(v.fullres_marks_count(), 0, "別の点群の番号を、持ち越さない")

    def test_old_marks_do_not_come_back_through_undo(self):
        """前の点群の印が、次の点群で「元に戻す」と生き返らない（2026-11-20、独立レビューで再現）"""
        v = self.v
        h = LayerUndoHistory()
        h.attach_extra(v.fullres_marks_tracker())
        self._brush(self._center_stroke())
        v.erase_overlay_selection()
        self.assertGreater(v.fullres_marks_count(), 100)
        # 別の点群を開く（画面は、履歴を空にしてから点群を入れ替える）
        h.undo_stack.clear()
        h.redo_stack.clear()
        v.set_point_cloud(self.full[::2].copy(), None, None)
        ids = v.layer_ids
        h.push(LayerState(ids, v.layers))
        ids[:10] = v.LAYER_NOISE
        self.assertEqual(v.fullres_marks_count(), 0)
        st = h.undo(LayerState(ids, v.layers))
        self.assertIsNotNone(st)
        self.assertEqual(v.fullres_marks_count(), 0, "前の点群の印が入ってはいけない")

    def test_overlay_points_can_be_picked_when_no_loaded_point_is_selectable(self):
        """箱の中の読み込んだ点を先に全部消すと、残った「飛ばされた点」を選べなくなっていた（独立レビュー）"""
        v = self.v
        v.move_indices_to_noise(np.arange(len(v.points_raw), dtype=np.int64))   # ノイズのレイヤーは非表示（既定）
        self._pump()
        self.assertFalse(v._get_selectable_mask().any())
        stroke, r = self._brush(self._center_stroke())
        sx, sy, vz = self._screen()
        inside = hit_brush(sx, sy, vz, stroke, r)
        loaded = self.rows % self.STEP == 0
        np.testing.assert_array_equal(v.overlay_selection(), np.flatnonzero(inside & ~loaded))
        self.assertEqual(len(v.selected_indices), 0)
        # 小さすぎる矩形（ただのクリック）では、選択を消さない
        n_before = v.overlay_selection_count()
        v.selection_shape = v.SELECT_RECT
        v.selection_start, v.selection_end = QPoint(100, 100), QPoint(101, 101)
        v._process_selection()
        self.assertEqual(v.overlay_selection_count(), n_before)
        # 即削除でも知らせが出る
        got = []
        v.instant_erase_requested.connect(lambda idx: got.append(len(idx)))
        v.instant_delete_mode = True
        self._brush(self._center_stroke())
        self.assertEqual(got, [0])
        self.assertGreater(v.commit_overlay_erase(), 100)

    def test_ctrl_only_removes_when_only_overlay_points_are_selected(self):
        """全点表示の点だけを選んでいるときの Ctrl は「外す」。本体の点を新しく選ばない（独立レビュー）"""
        v = self.v
        v.move_indices_to_noise(np.arange(len(v.points_raw), dtype=np.int64))
        a = self._center_stroke()
        self._brush(a)
        n_ov = v.overlay_selection_count()
        self.assertGreater(n_ov, 100)
        v.restore_indices_to_normal(np.arange(len(v.points_raw), dtype=np.int64))   # 本体の点を、また選べるようにする
        v._ov_sel_owner = v.selected_indices
        self._brush([(x + 120, y) for x, y in a], mods=Qt.ControlModifier)
        self.assertEqual(len(v.selected_indices), 0, "外すつもりの所の点が、新しく選ばれてはいけない")
        self.assertEqual(v.overlay_selection_count(), n_ov)

    def test_locked_or_hidden_normal_layer_and_hidden_scan_are_not_pickable(self):
        """飛ばされた点は、通常のレイヤーの点として扱う。ロック・非表示・隠したスキャンでは選べない（独立レビュー）"""
        v = self.v
        v.set_layer_locked(v.LAYER_NORMAL, True)
        self._brush(self._center_stroke())
        self.assertEqual(v.overlay_selection_count(), 0)
        v.set_layer_locked(v.LAYER_NORMAL, False)
        self._brush(self._center_stroke())
        self.assertGreater(v.overlay_selection_count(), 100)
        v.clear_selection()
        v.scan_visibility[0] = False
        self.assertFalse(v._overlay_pickable().any())
        v.scan_visibility[0] = True
        v.layers[v.LAYER_NORMAL].visible = False
        self.assertFalse(v._overlay_pickable().any())

    def test_loaded_rows_are_drawn_while_the_main_cloud_is_thinned(self):
        """本体が箱の中を間引いて描いているあいだは、重ね描きが「読み込んだ点と同じ行」も描く（どこにも描かれない点を作らない）"""
        v = self.v
        loaded = self.rows % self.STEP == 0
        self.assertTrue(v._overlay_main_draws_all())
        v.section_full_detail = False
        v.set_point_budget(2000)
        self._pump()
        self.assertFalse(v._overlay_main_draws_all())
        v.grabFramebuffer()                      # 描くときに、描く点を決め直す
        self.assertTrue(v._ov_visible[loaded].all())
        self.assertFalse(v._ov_editable[loaded].any(), "描いても、選べないまま（本体がレイヤーで扱う）")
        v.set_point_budget(0)
        self._pump()
        v.grabFramebuffer()
        self.assertFalse(v._ov_visible[loaded].any())

    def test_failed_work_state_load_keeps_the_marks(self):
        """作業の状態の読み込みが途中で失敗したら、印も変えない（独立レビュー）"""
        import json
        import tempfile

        v = self.v
        self._brush(self._center_stroke())
        v.erase_overlay_selection()
        want = v.fullres_marks.rows_of(0).copy()
        with tempfile.TemporaryDirectory() as d:
            bad = os.path.join(d, "bad.npz")
            meta = json.dumps({"point_count": len(v.points_raw), "layers": [{"name": "番号の無いレイヤー"}], "scan_visibility": {}})
            np.savez_compressed(
                bad, layer_ids=v.layer_ids, meta_json=np.array(meta),
                fullres_mark_scans=np.zeros(2, np.int32), fullres_mark_rows=np.array([1, 2], np.int64),
            )
            with self.assertRaises(KeyError):
                v.load_layer_project(bad)
        np.testing.assert_array_equal(v.fullres_marks.rows_of(0), want)

    def test_display_only_cases_stay_display_only(self):
        v = self.v
        # 間引きの間隔が分からない → 選べない・全部描く（今までどおり）
        self._attach(steps=[None])
        self.assertEqual(v.overlay_editable_count(), 0)
        self.assertTrue(v._ov_all_visible)
        self._brush(self._center_stroke())
        self.assertEqual(v.overlay_selection_count(), 0)
        # 行の番号を渡さない（古い呼び方）
        self._attach(rows=None)
        self.assertEqual(v.overlay_editable_count(), 0)
        # 行の番号が分からない点（-1）は選べない
        rows = self.rows.copy()
        rows[1::3] = -1
        self._attach(rows=rows)
        self.assertEqual(v.overlay_editable_count(), int((self.rows % 3 == 2).sum()))
        # 高精細表示（LOD）の重ね描き
        v.set_view_overlay(v._overlay_source[0].copy())
        self.assertEqual(v.overlay_editable_count(), 0)
        self._brush(self._center_stroke())
        self.assertEqual(v.overlay_selection_count(), 0)
        v.validate_invariants()


class TestViewerDraw(_ViewerCase):
    """実際に描いた絵で確かめる: 消した点は描かれない・選んだ点は選択の色"""

    def setUp(self):
        super().setUp()
        v = self.v
        # 大きな点 6 つ（行 0〜5）。行 0 と 3 は読み込んだ点（本体）。あとは飛ばされた点
        self.pts6 = np.array([[0.0, 0, 0], [0.2, 0.1, 0], [0.4, 0.2, 0], [0.6, 0.3, 0], [0.8, 0.12, 0], [1.0, 0.22, 0]])
        v.set_point_cloud(self.pts6[::3].copy(), np.tile([[0.2, 0.9, 0.2]], (2, 1)).astype(np.float32), None)
        v.set_section_z(True, -1.0, 1.0)
        v.set_bg_color(QColor(0, 0, 0))
        v.set_show_grid(False)
        v.set_show_section_box(False)             # 枠の線が点に重なると、点の色を読めない
        v.set_show_section_origin_plane(False)
        v.set_point_size(9.0)
        v.set_view_top()
        v.set_section_overlay(
            self.pts6, np.tile([[1.0, 1.0, 1.0]], (6, 1)).astype(np.float32), None, np.zeros(6, np.int32),
            rows=np.arange(6), scan_steps=[3],
        )
        self._pump(10)

    def _color_at(self, row):
        v = self.v
        img = v.grabFramebuffer()
        k = img.width() / v.width()
        sx, sy, _ = v._project_to_screen(v._overlay_source[0])
        c = img.pixelColor(int(round(float(sx[row]) * k)), int(round(float(sy[row]) * k)))
        return c.red(), c.green(), c.blue()

    def test_what_is_drawn(self):
        v = self.v
        self.assertEqual(self._color_at(1), (255, 255, 255), "飛ばされた点は、重ね描きが描く")
        r, g, b = self._color_at(0)
        self.assertTrue(g > 150 and r < 120, f"読み込んだ点は、本体の色で描かれる（重ね描きの白を重ねない）: {(r, g, b)}")
        # 行 1 を消す
        self.assertEqual(v._overlay_erase(np.array([1])), 1)
        self._pump()
        self.assertEqual(self._color_at(1), (0, 0, 0), "消した点は描かれない")
        self.assertEqual(self._color_at(2), (255, 255, 255), "ほかの点は、そのまま")
        # 行 4 を選ぶ → 選択の色
        v._ov_selected = np.array([4], dtype=np.int64)
        v._ov_sel_owner = v.selected_indices
        v.update()
        self._pump()
        r, g, b = self._color_at(4)
        self.assertNotEqual((r, g, b), (255, 255, 255))
        self.assertNotEqual((r, g, b), (0, 0, 0))
        # 戻す
        v.restore_all_fullres_marks()
        self._pump()
        self.assertEqual(self._color_at(1), (255, 255, 255))
        v.validate_invariants()


# ============================================================================= 実際の画面: なぞる → Del → 元に戻す → 全点で保存
class TestWindowFlow(_Scene):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.qapp = QApplication.instance() or QApplication(sys.argv)

    def setUp(self):
        from gui.main_window import MainWindow

        self.pre = self._load(max_load_points=self.load_budget)
        self.win = MainWindow()
        self.win._reset_app_settings()
        self.win.resize(1450, 900)
        self.win.show()
        self._pump()
        self.win.current_scans = self.pre.scans
        self.win.viewer.set_preloaded_data(self.pre, reset_camera=True)
        self._pump()

    def tearDown(self):
        self.win._set_dirty(False)
        self.win.close()
        self.win.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        self._pump()

    def _pump(self, n=8):
        for _ in range(n):
            QApplication.processEvents()

    def _reload(self):
        win, v = self.win, self.win.viewer
        v.set_section_z(True, -0.5, 3.0)       # スキャン A だけが入る（B は離れた所にある）
        v.set_view_top()
        # スキャン A（4m × 3m）が画面いっぱいに映る所まで寄る（B が 200m 先にあるので、そのままだと A は数ピクセルになる）
        from PySide6.QtGui import QVector3D

        c = np.array([2.0, 1.5, 0.0]) - v.center_offset
        v.camera_target = QVector3D(float(c[0]), float(c[1]), float(c[2]))
        v.camera_distance = 6.0
        self._pump()
        win._on_section_reload_clicked()
        for _ in range(400):
            QTest.qWait(25)
            if v.has_section_overlay and not win._is_background_busy():
                break
        self._pump()
        self.assertTrue(v.has_section_overlay, "読み直しが終わっていません")

    def _brush_at_world(self, world_xyz, radius=14):
        v = self.win.viewer
        c = np.asarray([world_xyz], dtype=np.float64) - v.center_offset
        sx, sy, _ = v._project_to_screen(c.astype(np.float32))
        v.selection_shape = v.SELECT_BRUSH
        v.brush_radius = radius
        v.brush_stroke_points = [QPoint(int(sx[0]), int(sy[0]))]
        v._last_selection_modifiers = NO_MOD
        v._process_brush_selection()
        stroke = np.array([[int(sx[0]), int(sy[0])]], dtype=np.float32)
        v.brush_stroke_points = []
        return stroke, float(max(4, radius))

    def test_brush_delete_undo_redo_and_save(self):
        win, v = self.win, self.win.viewer
        self._reload()
        n_edit = v.overlay_editable_count()
        self.assertGreater(n_edit, 10000)
        t = self.truth["A"]

        # 床の上の 1 か所をなぞる（面の点。印が無ければ保存で残る所）
        stroke, r = self._brush_at_world((2.2, 2.0, 0.0))
        sx, sy, vz = v._project_to_screen(v._overlay_source[0])
        inside = hit_brush(sx, sy, vz, stroke, r)
        self.assertLess(int(inside.sum()), 5000, "なぞった所は、全体のごく一部")
        ov_rows_inside = v._ov_rows[inside]
        expect_marks = np.sort(ov_rows_inside[ov_rows_inside % STRIDE != 0])
        self.assertGreater(len(expect_marks), 30)
        n_main = len(v.selected_indices)
        self.assertGreater(n_main, 10)
        self.assertEqual(v.overlay_selection_count(), len(expect_marks))
        self.assertIn("全点表示", win.manual_status_label.text())

        noise_before = int(np.sum(~v.layer_mask))
        win._delete_selected_points()
        self._pump()
        np.testing.assert_array_equal(v.fullres_marks.rows_of(0), expect_marks)
        self.assertEqual(int(np.sum(~v.layer_mask)), noise_before + n_main, "読み込んだ点は、今までどおりノイズのレイヤーへ")
        self.assertEqual(v.overlay_selection_count(), 0)

        # 元に戻す・やり直しは、1 手で両方
        win._undo_action()
        self._pump()
        self.assertEqual(v.fullres_marks_count(), 0)
        self.assertEqual(int(np.sum(~v.layer_mask)), noise_before)
        win._redo_action()
        self._pump()
        np.testing.assert_array_equal(v.fullres_marks.rows_of(0), expect_marks)
        self.assertEqual(int(np.sum(~v.layer_mask)), noise_before + n_main)

        # 全点で保存: なぞった所の点は、読み込んだ点も、飛ばされた点も、出力に無い。離れた所の面は残る
        out = os.path.join(self.tmp.name, "flow_out.e57")
        worker = FullResolutionSaveWorker(out, win.current_scans, v.layer_mask.copy())
        self.assertEqual(apply_marks_to_save_worker(worker, v, "normal"), len(expect_marks))
        ok, err = self._run(worker)
        self.assertFalse(err, err)
        self.assertEqual(worker.total_marked, len(expect_marks))
        out_a = [s for s in read_e57_scans(out) if s.name == "A"][0]
        out_keys = set(_scan_keys(out_a).tolist())
        brushed = set(t["keys"][ov_rows_inside].tolist())
        self.assertFalse(out_keys & brushed, "なぞって消した点が、保存したファイルに残っています")
        p = t["points"]
        far = (np.abs(p[:, 2]) < 0.01) & (p[:, 0] < 1.5) & (p[:, 1] < 0.5)
        kept = np.isin(t["keys"][far], np.fromiter(out_keys, dtype=np.int64))
        self.assertGreaterEqual(kept.mean(), 0.999, "離れた所の面の点が失われています")

    def test_instant_delete_and_restore_command(self):
        win, v = self.win, self.win.viewer
        self._reload()
        v.instant_delete_mode = True
        self._brush_at_world((1.2, 0.6, 0.0))
        self._pump()
        n = v.fullres_marks_count()
        self.assertGreater(n, 30)
        self.assertEqual(v.overlay_pending_erase_count(), 0)
        v.instant_delete_mode = False
        # コマンド「全点表示で消した点をすべて戻す」は 1 手として記録され、元に戻せる
        from gui.overlay_edit import restore_all_marks_command

        restore_all_marks_command(win)
        self.assertEqual(v.fullres_marks_count(), 0)
        win._undo_action()
        self.assertEqual(v.fullres_marks_count(), n)
        win._undo_action()
        self.assertEqual(v.fullres_marks_count(), 0)

    def test_delete_command_works_with_overlay_only_selection(self):
        """コマンド「選択点をノイズへ移す」は、全点表示の点だけを選んでいても使える（独立レビュー）"""
        from gui.commands import _need_selection, _need_selection_or_overlay

        win, v = self.win, self.win.viewer
        self._reload()
        self.assertIsNotNone(_need_selection_or_overlay(win), "何も選んでいなければ、使えない")
        v.move_indices_to_noise(np.arange(len(v.points_raw), dtype=np.int64))   # 読み込んだ点を、先に全部消してある場面
        self._brush_at_world((2.2, 2.0, 0.0))
        self.assertEqual(len(v.selected_indices), 0)
        self.assertGreater(v.overlay_selection_count(), 30)
        self.assertIsNotNone(_need_selection(win))
        self.assertIsNone(_need_selection_or_overlay(win))
        self.assertTrue(win.floating_del_btn.isEnabled(), "下の帯の「ノイズ削除」も押せる")
        n = v.overlay_selection_count()
        win._delete_selected_points()
        self.assertEqual(v.fullres_marks_count(), n)

    def test_visible_export_leaves_out_the_marks_of_hidden_scans(self):
        """表示中だけを保存するとき、隠したスキャンの「消した点」だけが出力に入ることの無いように（独立レビュー）"""
        v = self.win.viewer
        v.fullres_marks.add(np.array([0, 1], np.int32), np.array([4, 5], np.int64))
        v.layers[v.LAYER_NOISE].visible = True
        v.scan_visibility[0] = False

        class Fake:
            KEEP_LABEL = 1

            def set_marks(self, marks, label, scan_labels=None):
                self.got = (marks.count(), label, dict(scan_labels or {}))

        w = Fake()
        apply_marks_to_save_worker(w, v, "visible")
        self.assertEqual(w.got, (2, 1, {0: 0}), "隠したスキャン 0 の印は除く。表示中のスキャン 1 の印は、ノイズとして出す")
        w2 = Fake()
        apply_marks_to_save_worker(w2, v, "normal")
        self.assertEqual(w2.got, (2, 0, {}))

    def test_save_worker_keeps_its_own_copy_of_the_marks(self):
        v = self.win.viewer
        v.fullres_marks.add(np.array([0], np.int32), np.array([4], np.int64))
        worker = FullResolutionSaveWorker(os.path.join(self.tmp.name, "c.e57"), self.win.current_scans, v.layer_mask.copy())
        apply_marks_to_save_worker(worker, v, "normal")
        v.fullres_marks.clear()
        self.assertEqual(worker.marks.count(), 1, "保存のあいだに画面で印が変わっても、保存は影響を受けない")

    def test_export_label_follows_the_noise_layer(self):
        v = self.win.viewer
        keep, orphan = 1, -1
        self.assertEqual(marks_label_for_export(v, "normal", None, keep, orphan), 0)
        self.assertEqual(marks_label_for_export(v, "noise_only", None, keep, orphan), keep)
        self.assertEqual(marks_label_for_export(v, "split_all", None, keep, orphan), orphan)
        self.assertEqual(marks_label_for_export(v, "single_layer", v.LAYER_NOISE, keep, orphan), keep)
        self.assertEqual(marks_label_for_export(v, "single_layer", v.LAYER_NORMAL, keep, orphan), 0)
        v.layers[v.LAYER_NOISE].visible = True
        self.assertEqual(marks_label_for_export(v, "visible", None, keep, orphan), keep)
        v.layers[v.LAYER_NOISE].visible = False
        self.assertEqual(marks_label_for_export(v, "visible", None, keep, orphan), 0)


if __name__ == "__main__":
    unittest.main()
