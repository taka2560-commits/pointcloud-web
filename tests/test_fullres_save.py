"""
全点転写保存（FullResolutionSaveWorker / core.fullres_transfer / E57StreamWriter）のテスト

- 転写規則: 判定点の近くの点は判定点のラベルを引き継ぎ、離れた点は孤立点として除外されること
- 間引き読込（k=3）で決めた判定が元ファイルの全点へ写って保存されること:
    浮遊塊は判定されなかった点も含めて全点除外、面の点は残る、離れた単独点は除外、
    手動で消した範囲は全点で消える、保存された点の生整数・色・強度・pose・スキャン名・guid が元と一致（ロスレス）
- 座標系変換（測量 X/Y 入替）で取り込んでも同じ結果になること
- 全レイヤー分割保存でも各ファイルへ写ること
- 元ファイルが読込後に変更されていたら保存を拒否すること
- 中止すると出力も一時ファイルも残らないこと
- 間引きなしで読み込んだデータでは FileSaveWorker と同じ点が保存されること
- E57StreamWriter: 1 スキャンずつ追記しても write_e57_scans と同じ内容になり、abort で一時ファイルが残らないこと
"""

import glob
import os
import sys
import tempfile
import unittest

import numpy as np
from PySide6.QtCore import QCoreApplication
from pye57 import libe57

from core.e57_io import E57StreamWriter, read_e57_scans, write_e57_scans
from core.fullres_transfer import ORPHAN_LABEL, labels_from_masks, radius_factor, transfer_labels
from core.processor import inverse_transform_coordinate_system, is_identity_coord_mode
from gui.worker_thread import FileLoadWorker, FileSaveWorker, FullResolutionSaveWorker

SCALE = 0.0001  # 0.1mm
OFFSETS = (12.5, -3.25, 0.75)
STRIDE = 3

CAT_SURFACE, CAT_BLOB, CAT_STRAY = 0, 1, 2


def _grid(x0, x1, y0, y1, z, spacing, rng, axis_z=True):
    xs = np.arange(x0, x1, spacing)
    ys = np.arange(y0, y1, spacing)
    gx, gy = np.meshgrid(xs, ys)
    n = gx.size
    jitter = rng.normal(0.0, 0.0005, (n, 3))
    if axis_z:
        pts = np.column_stack([gx.ravel(), gy.ravel(), np.full(n, z)])
    else:  # 壁: x-z 面、y 固定
        pts = np.column_stack([gx.ravel(), np.full(n, z), gy.ravel()])
    return pts + jitter


def make_scene(rng, floor=(4.0, 3.0), wall_h=2.0, blobs=(500, 1500, 3000), n_stray=300, spacing=0.01):
    """
    床 + 壁 + 浮遊塊 + 離れた単独点。戻り値: (points (N,3), category (N,)) ランダム順。
    塊と単独点は面から 0.3m 以上離す。
    """
    floor_pts = _grid(0.0, floor[0], 0.0, floor[1], 0.0, spacing, rng)
    wall_pts = _grid(0.0, floor[0], 0.0, wall_h, floor[1], spacing, rng, axis_z=False)
    surface = np.vstack([floor_pts, wall_pts])

    blob_pts = []
    centers = [(1.0, 1.0, 1.0), (2.5, 1.5, 0.8), (3.2, 0.8, 1.3)]
    for c, n in zip(centers, blobs):
        blob_pts.append(rng.normal(0.0, 0.05, (n, 3)) + np.asarray(c))
    blob_pts = np.vstack(blob_pts)

    strays = []
    while len(strays) < n_stray:
        p = rng.random(3) * [floor[0], floor[1] - 0.4, 1.2] + [0.0, 0.0, 0.4]
        if min(np.linalg.norm(p - np.asarray(c)) for c in centers) < 0.4:
            continue
        strays.append(p)
    strays = np.asarray(strays)

    pts = np.vstack([surface, blob_pts, strays])
    cat = np.concatenate([
        np.full(len(surface), CAT_SURFACE),
        np.full(len(blob_pts), CAT_BLOB),
        np.full(len(strays), CAT_STRAY),
    ])
    is_wall = np.zeros(len(pts), dtype=bool)
    is_wall[len(floor_pts) : len(surface)] = True
    order = rng.permutation(len(pts))
    return pts[order], cat[order], is_wall[order]


def _to_raw(pts):
    return [np.round((pts[:, i] - OFFSETS[i]) / SCALE).astype(np.int64) for i in range(3)]


def _write_e57(path, scans):
    """
    scans: list of dict(points, name, guid, pose=(rot_wxyz, trans) or None)
    ScaledInteger XYZ（軸ごとに offset が異なる）+ color (Integer 0..255) + intensity (Float single)
    """
    imf = libe57.ImageFile(path, "w")
    imf.extensionsAdd("", libe57.E57_V1_0_URI)
    root = imf.root()
    root.set("formatName", libe57.StringNode(imf, "ASTM E57 3D Imaging Data File"))
    root.set("guid", libe57.StringNode(imf, "{11111111-2222-3333-4444-555555555555}"))
    root.set("coordinateMetadata", libe57.StringNode(imf, "JGD2011 / Plane 9"))
    data3d = libe57.VectorNode(imf, True)
    root.set("data3D", data3d)

    for spec in scans:
        pts = spec["points"]
        n = len(pts)
        scan = libe57.StructureNode(imf)
        scan.set("guid", libe57.StringNode(imf, spec["guid"]))
        scan.set("name", libe57.StringNode(imf, spec["name"]))
        if spec.get("pose") is not None:
            rot, tr = spec["pose"]
            pose = libe57.StructureNode(imf)
            rn = libe57.StructureNode(imf)
            for k, v in zip("wxyz", rot):
                rn.set(k, libe57.FloatNode(imf, float(v), libe57.E57_DOUBLE, -1.0, 1.0))
            pose.set("rotation", rn)
            tn = libe57.StructureNode(imf)
            for k, v in zip("xyz", tr):
                tn.set(k, libe57.FloatNode(imf, float(v), libe57.E57_DOUBLE, -1e6, 1e6))
            pose.set("translation", tn)
            scan.set("pose", pose)
        proto = libe57.StructureNode(imf)
        for i, ax in enumerate("XYZ"):
            proto.set(f"cartesian{ax}", libe57.ScaledIntegerNode(imf, 0, -100000000, 100000000, SCALE, OFFSETS[i]))
        for c in ("colorRed", "colorGreen", "colorBlue"):
            proto.set(c, libe57.IntegerNode(imf, 0, 0, 255))
        proto.set("intensity", libe57.FloatNode(imf, 0.0, libe57.E57_SINGLE, 0.0, 1.0))
        codecs = libe57.VectorNode(imf, True)
        cv = libe57.CompressedVectorNode(imf, proto, codecs)
        scan.set("points", cv)
        data3d.append(scan)

        raw = _to_raw(pts)
        colors = spec["colors"]
        inten = spec["intensity"]
        bufs = libe57.VectorSourceDestBuffer()
        for i, ax in enumerate("XYZ"):
            bufs.append(libe57.SourceDestBuffer(imf, f"cartesian{ax}", raw[i], n, True, False))
        color_cols = [np.ascontiguousarray(colors[:, i]) for i in range(3)]  # 列ビューのままだと stride を無視して読まれる
        for i, c in enumerate(("colorRed", "colorGreen", "colorBlue")):
            bufs.append(libe57.SourceDestBuffer(imf, c, color_cols[i], n, True, True))
        bufs.append(libe57.SourceDestBuffer(imf, "intensity", inten, n, True, True))
        w = cv.writer(bufs)
        w.write(n)
        w.close()
    imf.close()


def _key(raw_xyz):
    """生整数 (ix, iy, iz) を 1 本の照合キーにする（各軸 21bit に収めて int64 をあふれさせない）"""
    ix, iy, iz = (np.asarray(a, dtype=np.int64) for a in raw_xyz)
    off = np.int64(1 << 20)
    base = np.int64(1 << 21)
    assert np.all(np.abs(ix) < off) and np.all(np.abs(iy) < off) and np.all(np.abs(iz) < off)
    return ((ix + off) * base + (iy + off)) * base + (iz + off)


def _scan_keys(scan, coord_mode="cartesian"):
    """スキャンの点の照合キー。座標系変換して読み込んだスキャンは元の座標系へ戻してから作る"""
    if is_identity_coord_mode(coord_mode):
        return _key([scan.raw_unscaled_fields[f"cartesian{ax}"] for ax in "XYZ"])
    return _key(_to_raw(inverse_transform_coordinate_system(scan.points, coord_mode)))


ROT_90Z = (np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5))


class _Scene(unittest.TestCase):
    """合成 2 スキャン E57（A: pose なし、B: Z 軸 90° 回転 + 平行移動）を作って読み込むベース"""

    @classmethod
    def setUpClass(cls):
        cls.app = QCoreApplication.instance() or QCoreApplication(sys.argv)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = os.path.join(cls.tmp.name, "現場.e57")
        rng = np.random.default_rng(7)
        pts_a, cat_a, wall_a = make_scene(rng)
        pts_b, cat_b, wall_b = make_scene(rng, floor=(3.0, 2.0), blobs=(400, 1200), n_stray=200)
        cls.truth = {}
        specs = []
        for name, guid, pts, cat, wall, pose in (
            ("A", "{aaaaaaaa-0000-0000-0000-000000000001}", pts_a, cat_a, wall_a, None),
            ("B", "{bbbbbbbb-0000-0000-0000-000000000002}", pts_b, cat_b, wall_b, (ROT_90Z, (100.0, 200.0, 5.0))),
        ):
            colors = (rng.random((len(pts), 3)) * 255).astype(np.uint8)
            inten = rng.random(len(pts)).astype(np.float32)
            cls.truth[name] = dict(points=pts, cat=cat, is_wall=wall, colors=colors, intensity=inten, keys=_key(_to_raw(pts)))
            specs.append(dict(points=pts, name=name, guid=guid, pose=pose, colors=colors, intensity=inten))
        _write_e57(cls.path, specs)
        cls.n_total = sum(len(t["points"]) for t in cls.truth.values())
        # 総点数 > 上限 × 1.5 のとき ceil(総点数 / 上限) 点おきに読まれる。ちょうど STRIDE になる上限にする
        cls.load_budget = -(-cls.n_total // STRIDE) + 1

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _load(self, max_load_points, coord_mode="cartesian"):
        holder, errors = [], []
        worker = FileLoadWorker(self.path, point_budget=50_000, max_load_points=max_load_points, coord_mode=coord_mode)
        worker.finished_success.connect(holder.append)
        worker.finished_error.connect(errors.append)
        worker.run()
        self.assertFalse(errors, errors)
        return holder[0]

    def _judged_truth(self, pre, coord_mode="cartesian"):
        """読み込んだ（間引き済み）各点の正解 (カテゴリ, 壁かどうか)。間引きは各スキャン先頭から STRIDE 点おき"""
        cats, walls = [], []
        for s in pre.scans:
            t = self.truth[s.name]
            step = max(1, int(round(s.original_point_count / len(s.points))))
            idx = np.arange(0, len(t["points"]), step)
            self.assertEqual(len(idx), len(s.points), "間引きの点数が想定と違います")
            np.testing.assert_array_equal(_scan_keys(s, coord_mode), t["keys"][idx], "間引き読込の点が想定の位置と違います")
            cats.append(t["cat"][idx])
            walls.append(t["is_wall"][idx])
        return np.concatenate(cats), np.concatenate(walls)

    def _run(self, worker):
        ok, err = [], []
        worker.finished_success.connect(lambda a, b, c: ok.append((a, b, c)))
        worker.finished_error.connect(err.append)
        worker.run()
        return ok, err

    def _assert_lossless(self, out_scan, name, expect_cat_removed=(CAT_BLOB, CAT_STRAY)):
        """出力スキャンの全点が元ファイルの点と生整数・色・強度まで一致すること。戻り値: 元ファイル側の index"""
        t = self.truth[name]
        order = np.argsort(t["keys"])
        pos = np.searchsorted(t["keys"], _scan_keys(out_scan), sorter=order)
        src_idx = order[np.clip(pos, 0, len(order) - 1)]
        np.testing.assert_array_equal(t["keys"][src_idx], _scan_keys(out_scan), "元ファイルに無い点が出力されています")
        for i, c in enumerate(("colorRed", "colorGreen", "colorBlue")):
            np.testing.assert_array_equal(out_scan.raw_fields[c], t["colors"][src_idx, i], c)
        np.testing.assert_array_equal(out_scan.raw_fields["intensity"], t["intensity"][src_idx], "intensity")
        return src_idx


class TestTransferRule(unittest.TestCase):
    def test_inherit_and_orphan(self):
        judged = np.array([[i * 0.1, 0.0, 0.0] for i in range(10)], dtype=np.float64)
        labels = np.array([i % 2 for i in range(10)], dtype=np.int32)  # 0,1,0,1,...
        full = np.array([
            [0.0, 0.0, 0.0],     # 判定点そのもの → 0
            [0.12, 0.0, 0.0],    # 判定点 1 (x=0.1) に最近傍、距離 0.02 → 1
            [0.3, 0.25, 0.0],    # 最近傍 x=0.3 (label 1)、距離 0.25 ≤ 0.3 → 1（k=2: 倍率 3 × 間隔 0.1）
            [0.5, 0.35, 0.0],    # 距離 0.35 > 0.3 → 孤立点
            [5.0, 5.0, 5.0],     # 遠い → 孤立点
        ])
        out, st = transfer_labels(judged, labels, full, stride=2)
        self.assertEqual(out.tolist(), [0, 1, 1, ORPHAN_LABEL, ORPHAN_LABEL])
        self.assertEqual(st.n_orphan, 2)
        self.assertEqual(st.n_inherited, 3)
        self.assertEqual(st.stride, 2)
        self.assertEqual(st.factor, 3.0)
        self.assertAlmostEqual(st.max_inherit_distance, 0.25, places=6)

    def test_factor_grows_with_stride(self):
        self.assertEqual(radius_factor(1), 3.0)
        self.assertEqual(radius_factor(2), 3.0)
        self.assertEqual(radius_factor(5), 6.0)
        judged = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
        full = np.array([[0.55, 0.0, 0.0]])  # 距離 0.45: k=2 (半径 0.3) なら孤立、k=5 (半径 0.6) なら引き継ぐ
        self.assertEqual(transfer_labels(judged, [1, 1], full, stride=2)[0].tolist(), [ORPHAN_LABEL])
        self.assertEqual(transfer_labels(judged, [1, 1], full, stride=5)[0].tolist(), [1])

    def test_min_radius_for_duplicate_judged_points(self):
        judged = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])  # 間隔 0 → 下限 1cm
        full = np.array([[0.005, 0.0, 0.0], [0.02, 0.0, 0.0]])
        self.assertEqual(transfer_labels(judged, [1, 1], full)[0].tolist(), [1, ORPHAN_LABEL])

    def test_edge_cases(self):
        out, st = transfer_labels(np.empty((0, 3)), np.empty(0), np.array([[0.0, 0.0, 0.0]]))
        self.assertEqual(out.tolist(), [ORPHAN_LABEL])
        self.assertEqual(st.n_orphan, 1)
        out, st = transfer_labels(np.array([[0.0, 0.0, 0.0]]), [1], np.empty((0, 3)))
        self.assertEqual(len(out), 0)
        # 判定点が 1 つだけなら間隔は無限大 → どこでも引き継ぐ
        out, _ = transfer_labels(np.array([[0.0, 0.0, 0.0]]), [1], np.array([[9.0, 9.0, 9.0]]))
        self.assertEqual(out.tolist(), [1])
        with self.assertRaises(ValueError):
            transfer_labels(np.zeros((2, 3)), [1], np.zeros((1, 3)))

    def test_labels_from_masks(self):
        m0 = np.array([True, False, False, True])
        m1 = np.array([False, True, False, False])
        self.assertEqual(labels_from_masks([m0, m1], 4).tolist(), [0, 1, ORPHAN_LABEL, 0])
        with self.assertRaises(ValueError):
            labels_from_masks([m0[:3]], 4)

    def test_cancel_and_progress(self):
        judged = np.random.default_rng(0).random((100, 3))
        full = np.random.default_rng(1).random((5000, 3))
        seen = []
        transfer_labels(judged, np.ones(100), full, chunk_size=1000, progress=lambda d, n: seen.append((d, n)))
        self.assertEqual(seen[-1], (5000, 5000))
        self.assertEqual(len(seen), 5)
        with self.assertRaises(InterruptedError):
            transfer_labels(judged, np.ones(100), full, chunk_size=1000, is_cancelled=lambda: True)


class TestFullResolutionSave(_Scene):
    def test_subsampled_load_is_stride_3(self):
        pre = self._load(max_load_points=self.load_budget)
        self.assertTrue(pre.is_downsampled)
        for s in pre.scans:
            self.assertTrue(s.is_subsampled)
            self.assertIsNotNone(s.source_file)
            self.assertEqual(os.path.normcase(s.source_file), os.path.normcase(os.path.abspath(self.path)))
        self.assertEqual(pre.scans[0].source_scan_index, 0)
        self.assertEqual(pre.scans[1].source_scan_index, 1)
        self._judged_truth(pre)

    def _check_full_save(self, coord_mode):
        pre = self._load(max_load_points=self.load_budget, coord_mode=coord_mode)
        cat_j, _ = self._judged_truth(pre, coord_mode)
        layer_mask = cat_j == CAT_SURFACE
        # 手動削除の模擬: A の床の一部（x 1.0〜1.5, y 1.0〜1.5）を判定点で除外する
        a = pre.scans[0]
        a_local = a.points if is_identity_coord_mode(coord_mode) else inverse_transform_coordinate_system(a.points, coord_mode)
        cut = (a_local[:, 0] > 1.0) & (a_local[:, 0] < 1.5) & (a_local[:, 1] > 1.0) & (a_local[:, 1] < 1.5) & (np.abs(a_local[:, 2]) < 0.01)
        layer_mask[: len(a.points)][cut] = False
        self.assertGreater(int(np.sum(cut)), 500)

        out = os.path.join(self.tmp.name, f"現場_clean_{coord_mode}.e57")
        worker = FullResolutionSaveWorker(out, pre.scans, layer_mask)
        ok, err = self._run(worker)
        self.assertFalse(err, err)
        saved, excluded, path = ok[0]
        self.assertEqual(path, out)
        self.assertEqual(saved + excluded, self.n_total, "保存 + 除外 が元の総点数と一致しません")
        self.assertGreater(worker.total_orphans, 0)
        self.assertFalse(glob.glob(os.path.join(self.tmp.name, "*.tmp_*")), "一時ファイルが残っています")

        result = read_e57_scans(out)
        self.assertEqual([s.name for s in result], ["A", "B"])
        self.assertEqual(result[0].header.guid, "{aaaaaaaa-0000-0000-0000-000000000001}")
        np.testing.assert_allclose(result[1].rotation, ROT_90Z)
        np.testing.assert_allclose(result[1].translation, (100.0, 200.0, 5.0))
        self.assertIsNone(result[0].rotation)
        self.assertEqual(result[0].field_schemas["cartesianX"].node_type, "ScaledInteger")
        self.assertEqual(result[0].field_schemas["cartesianX"].offset, OFFSETS[0])
        self.assertEqual(result[0].field_schemas["intensity"].precision, libe57.E57_SINGLE)

        for out_scan in result:
            t = self.truth[out_scan.name]
            src_idx = self._assert_lossless(out_scan, out_scan.name)
            out_cat = t["cat"][src_idx]
            n_blob = int(np.sum(t["cat"] == CAT_BLOB))
            n_stray = int(np.sum(t["cat"] == CAT_STRAY))
            self.assertEqual(int(np.sum(out_cat == CAT_BLOB)), 0, f"{out_scan.name}: 浮遊塊の点が残っています（全点 {n_blob:,}）")
            self.assertLessEqual(int(np.sum(out_cat == CAT_STRAY)), n_stray * 0.01, f"{out_scan.name}: 単独点が残っています")
            kept_surface = np.zeros(len(t["points"]), dtype=bool)
            kept_surface[src_idx] = True
            surface = t["cat"] == CAT_SURFACE
            p = t["points"]
            if out_scan.name == "A":
                deep_in_cut = (p[:, 0] > 1.03) & (p[:, 0] < 1.47) & (p[:, 1] > 1.03) & (p[:, 1] < 1.47) & (np.abs(p[:, 2]) < 0.01)
                near_cut = (p[:, 0] > 0.97) & (p[:, 0] < 1.53) & (p[:, 1] > 0.97) & (p[:, 1] < 1.53) & (np.abs(p[:, 2]) < 0.01)
                self.assertEqual(int(np.sum(kept_surface & deep_in_cut)), 0, "手動で消した範囲の点が全点側で残っています")
                self.assertGreater(int(np.sum(deep_in_cut)), 500)
                surface_outside = surface & ~near_cut
            else:
                surface_outside = surface
            ratio = np.sum(kept_surface & surface_outside) / max(1, np.sum(surface_outside))
            self.assertGreaterEqual(ratio, 0.999, f"{out_scan.name}: 面の点が失われています ({ratio:.5f})")
        return worker

    def test_full_save_cartesian(self):
        worker = self._check_full_save("cartesian")
        for name, st in worker.scan_stats:
            self.assertEqual(st.stride, STRIDE, name)
            self.assertEqual(st.factor, 4.0, name)

    def test_full_save_with_survey_swap(self):
        self._check_full_save("survey_swap")

    def test_split_save_transfers_each_layer(self):
        pre = self._load(max_load_points=self.load_budget)
        cat_j, wall_j = self._judged_truth(pre)
        surf = cat_j == CAT_SURFACE
        masks = {"床": surf & ~wall_j, "壁": surf & wall_j, "空": np.zeros(len(cat_j), dtype=bool)}
        out = os.path.join(self.tmp.name, "分割.e57")
        worker = FullResolutionSaveWorker(out, pre.scans, surf, split_masks=masks)
        ok, err = self._run(worker)
        self.assertFalse(err, err)
        saved, excluded, summary = ok[0]
        self.assertIn("2 ファイル", summary)
        files = sorted(glob.glob(os.path.join(self.tmp.name, "分割_*.e57")))
        self.assertEqual([os.path.basename(f) for f in files], ["分割_壁.e57", "分割_床.e57"])
        seen = {}
        total = 0
        for f in files:
            is_wall_file = "壁" in os.path.basename(f)
            for out_scan in read_e57_scans(f):
                src_idx = self._assert_lossless(out_scan, out_scan.name)
                t = self.truth[out_scan.name]
                self.assertEqual(int(np.sum(t["cat"][src_idx] != CAT_SURFACE)), 0, "浮遊塊・単独点が分割ファイルに入っています")
                # 床と壁が接する角の 1 列は最近傍が相手側の判定点になりうるので、わずかな混入は許す
                ratio = float(np.mean(t["is_wall"][src_idx] == is_wall_file))
                self.assertGreaterEqual(ratio, 0.99, f"{os.path.basename(f)}: 床と壁の振り分けが崩れています ({ratio:.4f})")
                keys = set(_scan_keys(out_scan).tolist())
                self.assertFalse(keys & seen.get(out_scan.name, set()), "同じ点が 2 つのファイルに入っています")
                seen.setdefault(out_scan.name, set()).update(keys)
                total += len(keys)
        self.assertEqual(total, saved)
        self.assertEqual(saved + excluded, self.n_total)

    def test_refuses_when_source_changed(self):
        pre = self._load(max_load_points=self.load_budget)
        st = os.stat(self.path)
        os.utime(self.path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
        try:
            out = os.path.join(self.tmp.name, "変更後.e57")
            ok, err = self._run(FullResolutionSaveWorker(out, pre.scans, np.ones(len(pre.points_raw), dtype=bool)))
            self.assertFalse(ok)
            self.assertEqual(len(err), 1)
            self.assertIn("変更", err[0])
            self.assertFalse(os.path.exists(out))
        finally:
            os.utime(self.path, ns=(st.st_atime_ns, st.st_mtime_ns))

    def test_refuses_scans_without_source(self):
        pre = self._load(max_load_points=self.load_budget)
        pre.scans[0].source_file = None
        out = os.path.join(self.tmp.name, "元なし.e57")
        ok, err = self._run(FullResolutionSaveWorker(out, pre.scans, np.ones(len(pre.points_raw), dtype=bool)))
        self.assertFalse(ok)
        self.assertIn("間引いたまま保存", err[0])

    def test_cancel_leaves_nothing(self):
        pre = self._load(max_load_points=self.load_budget)
        out = os.path.join(self.tmp.name, "中止.e57")
        worker = FullResolutionSaveWorker(out, pre.scans, np.ones(len(pre.points_raw), dtype=bool))
        progress = []

        def on_progress(pct, msg, sub):
            progress.append(msg)
            if "転写中" in msg:
                worker.cancel()

        worker.progress_changed.connect(on_progress)
        ok, err = self._run(worker)
        self.assertFalse(ok)
        self.assertIn("中止", err[0])
        self.assertTrue(any("転写中" in m for m in progress))
        self.assertFalse(os.path.exists(out))
        self.assertFalse(glob.glob(os.path.join(self.tmp.name, "*.tmp_*")))

    def test_empty_mask_is_rejected(self):
        pre = self._load(max_load_points=self.load_budget)
        out = os.path.join(self.tmp.name, "空.e57")
        ok, err = self._run(FullResolutionSaveWorker(out, pre.scans, np.zeros(len(pre.points_raw), dtype=bool)))
        self.assertFalse(ok)
        self.assertIn("0 点", err[0])
        self.assertFalse(os.path.exists(out))

    def test_full_load_matches_file_save_worker(self):
        """間引きなしで読み込んだときは、全点転写保存と従来の保存が同じ点を出力する"""
        pre = self._load(max_load_points=None)
        self.assertFalse(pre.is_downsampled)
        rng = np.random.default_rng(3)
        layer_mask = rng.random(len(pre.points_raw)) > 0.3
        out_a = os.path.join(self.tmp.name, "従来.e57")
        out_b = os.path.join(self.tmp.name, "転写.e57")
        ok, err = self._run(FileSaveWorker(out_a, pre.scans, layer_mask))
        self.assertFalse(err, err)
        ok2, err2 = self._run(FullResolutionSaveWorker(out_b, pre.scans, layer_mask))
        self.assertFalse(err2, err2)
        self.assertEqual(ok[0][:2], ok2[0][:2])
        for sa, sb in zip(read_e57_scans(out_a), read_e57_scans(out_b)):
            np.testing.assert_array_equal(_scan_keys(sa), _scan_keys(sb))
            np.testing.assert_array_equal(sa.raw_fields["colorRed"], sb.raw_fields["colorRed"])


class TestStreamWriter(_Scene):
    def test_incremental_write_equals_batch_write(self):
        scans = read_e57_scans(self.path)
        out_batch = os.path.join(self.tmp.name, "batch.e57")
        out_stream = os.path.join(self.tmp.name, "stream.e57")
        write_e57_scans(out_batch, scans)
        with E57StreamWriter(out_stream) as w:
            for s in scans:
                w.add_scan(s)
            self.assertEqual(w.scan_count, 2)
            self.assertEqual(w.point_count, self.n_total)
        ra, rb = read_e57_scans(out_batch), read_e57_scans(out_stream)
        self.assertEqual(len(rb), 2)
        for sa, sb in zip(ra, rb):
            np.testing.assert_array_equal(_scan_keys(sa), _scan_keys(sb))
            self.assertEqual(sa.name, sb.name)
            self.assertEqual(sa.coordinate_metadata, "JGD2011 / Plane 9")
            self.assertEqual(sb.coordinate_metadata, "JGD2011 / Plane 9")
            self.assertEqual(sa.header.guid, sb.header.guid)
        self.assertFalse(glob.glob(os.path.join(self.tmp.name, "*.tmp_*")))

    def test_abort_and_errors_leave_no_temp_file(self):
        scans = read_e57_scans(self.path)
        out = os.path.join(self.tmp.name, "abort.e57")
        w = E57StreamWriter(out)
        w.add_scan(scans[0])
        self.assertTrue(os.path.exists(w.temp_path))
        w.abort()
        self.assertFalse(os.path.exists(w.temp_path))
        self.assertFalse(os.path.exists(out))
        with self.assertRaises(RuntimeError):
            w.add_scan(scans[1])
        # 1 スキャンも書かずに close すると失敗し、一時ファイルも残らない
        w2 = E57StreamWriter(out)
        with self.assertRaises(ValueError):
            w2.close()
        self.assertFalse(os.path.exists(w2.temp_path))
        self.assertFalse(os.path.exists(out))

    def test_read_single_scan_by_index(self):
        only_b = read_e57_scans(self.path, scan_indices=[1])
        self.assertEqual(len(only_b), 1)
        self.assertEqual(only_b[0].name, "B")
        self.assertEqual(only_b[0].source_scan_index, 1)
        self.assertFalse(only_b[0].is_subsampled)
        np.testing.assert_array_equal(_scan_keys(only_b[0]), self.truth["B"]["keys"])


if __name__ == "__main__":
    unittest.main()
