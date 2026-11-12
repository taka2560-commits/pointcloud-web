"""
全点で保存の流し方式（core/e57_stream.py と、それを使う FullResolutionSaveWorker）のテスト

2026-11-12 に、1 スキャンを全点メモリへ読んでから書く方式（5.4 億点・1 スキャンでは約 89GB が要る）を、
100 万点ずつ「読む → 転写 → 書く」を流す方式に置き換えた。ここでは次を確かめる。
- 出力が以前の方式と同じ: フィールドの並び・型・プロトタイプの範囲・scale/offset、全点の生の値、
  cartesianBounds、スキャンのメタデータ（名前・guid・pose）が、以前の方式で書いたファイルと一致する
- チャンクの境目に依らない（小さなチャンクで流しても同じ）
- 座標が NaN の点は読み込みと同じく除かれる（Float 座標のファイル）
- 元ファイルの値がプロトタイプの範囲を外れていたら、範囲を測ってから書き直して、値は失わない
- 分割保存・中止でも、一時ファイルが残らない
"""

import glob
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from pye57 import libe57
from pye57.utils import get_node

from core.e57_io import E57StreamWriter, read_e57_scans
from core.e57_stream import (
    E57SchemaBoundsError,
    ScanStreamSink,
    iter_scan_stream_chunks,
    measure_field_ranges,
    scan_stream_template,
)
from core.fullres_transfer import estimate_stride, transfer_labels
from core.processor import inverse_transform_coordinate_system, is_identity_coord_mode
from gui.worker_thread import FileLoadWorker, FullResolutionSaveWorker
from test_fullres_save import ROT_90Z, _write_e57, make_scene


def _reference_save(out_path, scans, layer_mask):
    """以前の方式（1 スキャンの全点を読んでから、転写して、まとめて書く）。比べるための基準"""
    labels_all = layer_mask.astype(np.int32)
    offset = 0
    w = E57StreamWriter(out_path)
    for s in scans:
        n_j = len(s.points)
        labels_j = labels_all[offset: offset + n_j]
        offset += n_j
        full = read_e57_scans(s.source_file, point_budget=None, scan_indices=[int(s.source_scan_index)])[0]
        if s._name:
            full.name = s._name
        judged = s.points if is_identity_coord_mode(getattr(s, "coord_mode", "cartesian")) else \
            inverse_transform_coordinate_system(s.points, s.coord_mode)
        labels, _ = transfer_labels(judged, labels_j, full.points, stride=estimate_stride(s.original_point_count, n_j))
        keep = np.flatnonzero(labels == 1)
        if len(keep):
            w.add_scan(full.filter_by_indices(keep))
    w.close()


def _describe_file(path):
    """ファイルの中身を、比べられる形に書き出す（スキャンごとの定義と全フィールドの生の値）"""
    imf = libe57.ImageFile(path, "r")
    out = []
    try:
        d3 = get_node(imf.root(), "data3D")
        for i in range(d3.childCount()):
            scan = get_node(d3, i)
            cv = get_node(scan, "points")
            proto = libe57.StructureNode(cv.prototype())
            n = cv.childCount()
            fields, bufs = [], libe57.VectorSourceDestBuffer()
            arrays = {}
            for f in range(proto.childCount()):
                node = get_node(proto, f)
                name = node.elementName()
                if isinstance(node, libe57.ScaledIntegerNode):
                    fields.append((name, "SI", node.minimum(), node.maximum(), node.scale(), node.offset(), node.rawValue()))
                    arr = np.empty(n, dtype=np.int64)
                    bufs.append(libe57.SourceDestBuffer(imf, name, arr, n, True, False))
                elif isinstance(node, libe57.FloatNode):
                    fields.append((name, "F", node.precision(), node.minimum(), node.maximum(), node.value()))
                    arr = np.empty(n, dtype=np.float32 if node.precision() == libe57.E57_SINGLE else np.float64)
                    bufs.append(libe57.SourceDestBuffer(imf, name, arr, n, True, True))
                else:
                    fields.append((name, "I", node.minimum(), node.maximum(), node.value()))
                    arr = np.empty(n, dtype=np.int64)
                    bufs.append(libe57.SourceDestBuffer(imf, name, arr, n, True, True))
                arrays[name] = arr
            if n:
                r = cv.reader(bufs)
                while r.read() > 0:
                    pass
                r.close()
            meta = {}
            for key in ("name", "guid"):
                if scan.isDefined(key):
                    meta[key] = get_node(scan, key).value()
            if scan.isDefined("cartesianBounds"):
                b = get_node(scan, "cartesianBounds")
                meta["bounds"] = tuple(get_node(b, k).value() for k in
                                       ("xMinimum", "xMaximum", "yMinimum", "yMaximum", "zMinimum", "zMaximum"))
            meta["has_pose"] = scan.isDefined("pose")
            out.append((n, fields, meta, arrays))
    finally:
        imf.close()
    return out


class _StreamScene(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = os.path.join(cls.tmp.name, "現場.e57")
        rng = np.random.default_rng(11)
        specs = []
        for name, guid, pose, floor in (
            ("A", "{aaaaaaaa-0000-0000-0000-000000000001}", None, (3.0, 2.0)),
            ("B", "{bbbbbbbb-0000-0000-0000-000000000002}", (ROT_90Z, (100.0, 200.0, 5.0)), (2.0, 2.0)),
        ):
            pts, _cat, _wall = make_scene(rng, floor=floor, blobs=(300, 800), n_stray=150)
            colors = (rng.random((len(pts), 3)) * 255).astype(np.uint8)
            inten = rng.random(len(pts)).astype(np.float32)
            specs.append(dict(points=pts, name=name, guid=guid, pose=pose, colors=colors, intensity=inten))
        _write_e57(cls.path, specs)
        cls.n_total = sum(len(sp["points"]) for sp in specs)
        cls.load_budget = -(-cls.n_total // 3) + 1

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _load(self, coord_mode="cartesian"):
        holder, errors = [], []
        w = FileLoadWorker(self.path, point_budget=50_000, max_load_points=self.load_budget, coord_mode=coord_mode)
        w.finished_success.connect(holder.append)
        w.finished_error.connect(errors.append)
        w.run()
        self.assertFalse(errors, errors)
        return holder[0]

    def _run(self, worker):
        ok, err = [], []
        worker.finished_success.connect(lambda a, b, c: ok.append((a, b, c)))
        worker.finished_error.connect(err.append)
        worker.run()
        return ok, err

    def _mask(self, pre):
        rng = np.random.default_rng(5)
        return rng.random(len(pre.points_raw)) > 0.2

    def assertSameFile(self, a, b):
        da, db = _describe_file(a), _describe_file(b)
        self.assertEqual(len(da), len(db), "スキャン数が違います")
        for (na, fa, ma, aa), (nb, fb, mb, ab) in zip(da, db):
            self.assertEqual(na, nb, "点数が違います")
            self.assertEqual(fa, fb, "フィールドの定義（並び・型・範囲・scale/offset）が違います")
            self.assertEqual(ma, mb, "スキャンのメタデータ（名前・guid・範囲・pose）が違います")
            for name in aa:
                np.testing.assert_array_equal(aa[name], ab[name], f"{name} の値が違います")


class TestStreamEqualsReference(_StreamScene):
    def _compare(self, coord_mode, chunk=None):
        pre = self._load(coord_mode)
        mask = self._mask(pre)
        ref = os.path.join(self.tmp.name, f"ref_{coord_mode}_{chunk}.e57")
        new = os.path.join(self.tmp.name, f"new_{coord_mode}_{chunk}.e57")
        _reference_save(ref, pre.scans, mask)
        worker = FullResolutionSaveWorker(new, pre.scans, mask)
        if chunk is not None:
            worker.stream_chunk_size = chunk
        ok, err = self._run(worker)
        self.assertFalse(err, err)
        self.assertSameFile(ref, new)
        self.assertFalse(glob.glob(os.path.join(self.tmp.name, "*.tmp_*")), "一時ファイルが残っています")
        return worker, ok

    def test_same_output_as_reference(self):
        worker, ok = self._compare("cartesian")
        saved, excluded, _ = ok[0]
        self.assertEqual(saved + excluded, self.n_total)
        self.assertGreater(worker.total_orphans, 0)

    def test_same_output_with_small_chunks(self):
        """チャンクの境目に依らない（境目がスキャンの中で何百回も来る大きさで流す）"""
        self._compare("cartesian", chunk=777)

    def test_same_output_with_survey_swap(self):
        self._compare("survey_swap", chunk=5000)

    def test_keep_everything_keeps_original_bounds(self):
        """全点を残したときは、元ファイルの cartesianBounds をそのまま書く（以前の方式と同じ）"""
        from core.e57_io import write_e57_scans

        src_path = os.path.join(self.tmp.name, "範囲あり.e57")
        write_e57_scans(src_path, read_e57_scans(self.path))  # cartesianBounds を持つ元ファイルを作る
        src = _describe_file(src_path)[0]
        self.assertIn("bounds", src[2])
        tpl = scan_stream_template(src_path, 0)
        self.assertTrue(tpl.has_orig_bounds)
        out = os.path.join(self.tmp.name, "all.e57")
        w = E57StreamWriter(out)
        sink = ScanStreamSink(w, tpl, capacity=1000)
        for raw, xyz in iter_scan_stream_chunks(tpl, 1000):
            sink.write(raw, xyz, np.arange(len(xyz)))
        self.assertEqual(sink.finish(), tpl.n_declared)
        w.close()
        got = _describe_file(out)[0]
        self.assertEqual(src[2]["bounds"], got[2]["bounds"])
        self.assertEqual(src[1], got[1])
        for name in src[3]:
            np.testing.assert_array_equal(src[3][name], got[3][name])


class TestStreamEdgeCases(_StreamScene):
    def test_nan_points_are_dropped_like_the_loader(self):
        """Float 座標のファイルで座標が NaN の点は、読み込み（read_e57_scans）と同じく除かれる"""
        path = os.path.join(self.tmp.name, "nan.e57")
        rng = np.random.default_rng(2)
        n = 2500
        xyz = rng.random((n, 3))
        xyz[::97, 1] = np.nan
        imf = libe57.ImageFile(path, "w")
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
            bufs.append(libe57.SourceDestBuffer(imf, f"cartesian{ax}", cols[i], n, True, True))
        wr = cv.writer(bufs)
        wr.write(n)
        wr.close()
        imf.close()

        loaded = read_e57_scans(path)[0]
        tpl = scan_stream_template(path, 0)
        pts = np.vstack([p for _r, p in iter_scan_stream_chunks(tpl, 300)])
        self.assertEqual(len(pts), len(loaded.points))
        self.assertLess(len(pts), n)
        np.testing.assert_array_equal(pts, loaded.points)

    def test_out_of_range_values_are_rewritten_with_measured_ranges(self):
        """プロトタイプの範囲を外れた値は、範囲を測ってから書けば失わない"""
        tpl = scan_stream_template(self.path, 0)
        sch = tpl.field_schemas["colorRed"]
        sch_min, sch_max = sch.minimum, sch.maximum
        try:
            sch.maximum = 100  # 元ファイルの値（0〜255）が範囲を外れる状況を作る
            out = os.path.join(self.tmp.name, "range.e57")
            w = E57StreamWriter(out)
            sink = ScanStreamSink(w, tpl, capacity=2000)
            with self.assertRaises(E57SchemaBoundsError):
                for raw, xyz in iter_scan_stream_chunks(tpl, 2000):
                    sink.write(raw, xyz, np.arange(len(xyz)))
            sink.close_quietly()
            w.abort()
            self.assertFalse(os.path.exists(out))

            tpl.bounds_override = measure_field_ranges(tpl, 2000)
            self.assertGreater(tpl.bounds_override["colorRed"][1], 100)
            w = E57StreamWriter(out)
            sink = ScanStreamSink(w, tpl, capacity=2000)
            for raw, xyz in iter_scan_stream_chunks(tpl, 2000):
                sink.write(raw, xyz, np.arange(len(xyz)))
            sink.finish()
            w.close()
            src = _describe_file(self.path)[0][3]
            got = _describe_file(out)[0][3]
            for name in src:
                np.testing.assert_array_equal(src[name], got[name], name)
        finally:
            sch.minimum, sch.maximum = sch_min, sch_max

    def test_worker_retries_with_measured_ranges(self):
        import gui.worker_thread as wt

        pre = self._load()
        mask = self._mask(pre)
        orig = wt.scan_stream_template
        calls = []

        def narrowed(path, idx):
            tpl = orig(path, idx)
            tpl.field_schemas["colorGreen"].maximum = 10
            calls.append(idx)
            return tpl

        wt.scan_stream_template = narrowed
        try:
            out = os.path.join(self.tmp.name, "retry.e57")
            ok, err = self._run(FullResolutionSaveWorker(out, pre.scans, mask))
        finally:
            wt.scan_stream_template = orig
        self.assertFalse(err, err)
        self.assertGreater(len(calls), 2, "範囲を測ってから最初から書き直していません")
        ref = os.path.join(self.tmp.name, "retry_ref.e57")
        _reference_save(ref, pre.scans, mask)
        for (_n, _f, _m, a), (_n2, _f2, _m2, b) in zip(_describe_file(ref), _describe_file(out)):
            for name in a:
                np.testing.assert_array_equal(a[name], b[name], f"{name}: 書き直しで値が変わりました")
        self.assertFalse(glob.glob(os.path.join(self.tmp.name, "*.tmp_*")))

    def test_split_and_cancel(self):
        pre = self._load()
        n = len(pre.points_raw)
        rng = np.random.default_rng(9)
        masks = {"床": rng.random(n) > 0.5}
        masks["壁"] = ~masks["床"] & (rng.random(n) > 0.3)
        out = os.path.join(self.tmp.name, "分割.e57")
        worker = FullResolutionSaveWorker(out, pre.scans, np.ones(n, dtype=bool), split_masks=masks)
        worker.stream_chunk_size = 3000
        ok, err = self._run(worker)
        self.assertFalse(err, err)
        files = sorted(glob.glob(os.path.join(self.tmp.name, "分割*.e57")))
        self.assertEqual(len(files), 2, files)
        total = sum(sum(s.point_count for s in read_e57_scans(f)) for f in files)
        self.assertEqual(total, ok[0][0])

        out2 = os.path.join(self.tmp.name, "中止.e57")
        w2 = FullResolutionSaveWorker(out2, pre.scans, np.ones(n, dtype=bool))
        w2.stream_chunk_size = 2000
        seen = []

        def on_progress(_pct, msg, sub):
            seen.append(msg)
            if "転写中" in msg and len(seen) > 3:
                w2.cancel()

        w2.progress_changed.connect(on_progress)
        ok2, err2 = self._run(w2)
        self.assertFalse(ok2)
        self.assertIn("中止", err2[0])
        self.assertFalse(os.path.exists(out2))
        self.assertFalse(glob.glob(os.path.join(self.tmp.name, "*.tmp_*")))


if __name__ == "__main__":
    unittest.main()
