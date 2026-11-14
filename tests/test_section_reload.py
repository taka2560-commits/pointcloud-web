"""
断面ボックスの元ファイル読み直し（core.section_reload / iter_e57_point_chunks / ビューアの重ね描き）のテスト
- チャンク流し読みが全点読みと同じ座標・色・強度を返すこと
- 断面ボックス内の点だけが、表示座標（pose 適用後・座標系変換後）で元ファイルから拾われること
- cartesianBounds がボックスに掛からないスキャンは読まずに飛ばすこと（無いスキャンは読む）
- 間引き読込（k 点に 1 点）でも、読み直しでボックス内の密度が元ファイルどおりに戻ること
- 上限で間引くときは一様に薄くなり、ボックス内の総点数は正確に数えること
- 斜め壁・座標系変換（survey_swap）でも同じこと
- 中止・進捗・読み込み元の照合（記録なし / 変更あり / 断面なし）
- ビューア: 重ね描きの設定・自動クリア（ボックス変更 / 断面解除 / 点群入替）・配色モードに合わせた再着色・不変条件
- MainWindow: ボタンの前提チェックと、実際の読込経路（間引き読込 → Z 断面 → 読み直し → 重ね描き）
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.settings_sandbox import use_sandbox_settings  # noqa: E402
from tests.qt_cleanup import destroy_widget  # noqa: E402

use_sandbox_settings()  # この PC の本番アプリの保存設定を、テストで書き換えない

import numpy as np
from pye57 import libe57
from PySide6.QtWidgets import QApplication

from core.e57_io import iter_e57_point_chunks, read_e57_scans
from core.processor import scan_display_points, transform_scan_coordinate_system
from core.section_cache import remove_cache
from core.section_reload import (
    SectionBox,
    color_divisor,
    reload_section_points,
    scan_bounds_corners,
)
from gui.gl_viewer import PointCloudViewer

SCALE = 0.001
OFFSETS = (10.0, -5.0, 2.0)
ROT_90Z = (np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5))  # Z 軸まわり 90°


def _grid(x0, x1, y0, y1, z, spacing):
    xs = np.arange(x0, x1 + spacing / 2, spacing)
    ys = np.arange(y0, y1 + spacing / 2, spacing)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    return np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, z, dtype=np.float64)])


def _wall_x(x, y0, y1, z0, z1, spacing):
    ys = np.arange(y0, y1 + spacing / 2, spacing)
    zs = np.arange(z0, z1 + spacing / 2, spacing)
    gy, gz = np.meshgrid(ys, zs, indexing="ij")
    return np.column_stack([np.full(gy.size, x, dtype=np.float64), gy.ravel(), gz.ravel()])


def _to_raw(pts):
    return [np.round((pts[:, i] - OFFSETS[i]) / SCALE).astype(np.int64) for i in range(3)]


def _write_e57(path, scans, float_xyz=False):
    """
    scans: list of dict(points, name, guid, pose=(rot_wxyz, trans) or None, colors uint8 (N,3), intensity float32 (N,),
                        bounds=True/False)
    ScaledInteger XYZ（float_xyz=True なら倍精度 Float XYZ）+ color (Integer 0..255) + intensity (Float single)。
    bounds=True なら cartesianBounds を書く。
    """
    imf = libe57.ImageFile(path, "w")
    imf.extensionsAdd("", libe57.E57_V1_0_URI)
    root = imf.root()
    root.set("formatName", libe57.StringNode(imf, "ASTM E57 3D Imaging Data File"))
    root.set("guid", libe57.StringNode(imf, "{22222222-2222-3333-4444-555555555555}"))
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
        if spec.get("bounds", False):
            bb = libe57.StructureNode(imf)
            for ax_i, ax in enumerate("xyz"):
                bb.set(f"{ax}Minimum", libe57.FloatNode(imf, float(np.min(pts[:, ax_i])), libe57.E57_DOUBLE, -1e6, 1e6))
                bb.set(f"{ax}Maximum", libe57.FloatNode(imf, float(np.max(pts[:, ax_i])), libe57.E57_DOUBLE, -1e6, 1e6))
            scan.set("cartesianBounds", bb)
        proto = libe57.StructureNode(imf)
        for i, ax in enumerate("XYZ"):
            if float_xyz:
                proto.set(f"cartesian{ax}", libe57.FloatNode(imf, 0.0, libe57.E57_DOUBLE, -1e6, 1e6))
            else:
                proto.set(f"cartesian{ax}", libe57.ScaledIntegerNode(imf, 0, -100000000, 100000000, SCALE, OFFSETS[i]))
        for c in ("colorRed", "colorGreen", "colorBlue"):
            proto.set(c, libe57.IntegerNode(imf, 0, 0, 255))
        proto.set("intensity", libe57.FloatNode(imf, 0.0, libe57.E57_SINGLE, 0.0, 1.0))
        codecs = libe57.VectorNode(imf, True)
        cv = libe57.CompressedVectorNode(imf, proto, codecs)
        scan.set("points", cv)
        data3d.append(scan)

        if float_xyz:
            raw = [np.ascontiguousarray(pts[:, i], dtype=np.float64) for i in range(3)]
        else:
            raw = _to_raw(pts)
        colors = spec["colors"]
        inten = np.ascontiguousarray(spec["intensity"], dtype=np.float32)
        bufs = libe57.VectorSourceDestBuffer()
        for i, ax in enumerate("XYZ"):
            # ScaledInteger は生整数をそのまま書く（doScaling=False）。Float は値のまま
            bufs.append(libe57.SourceDestBuffer(imf, f"cartesian{ax}", raw[i], n, True, bool(float_xyz)))
        color_cols = [np.ascontiguousarray(colors[:, i]) for i in range(3)]
        for i, c in enumerate(("colorRed", "colorGreen", "colorBlue")):
            bufs.append(libe57.SourceDestBuffer(imf, c, color_cols[i], n, True, True))
        bufs.append(libe57.SourceDestBuffer(imf, "intensity", inten, n, True, True))
        w = cv.writer(bufs)
        w.write(n)
        w.close()
    imf.close()


def _as_set(pts, decimals=5):
    return set(map(tuple, np.round(np.asarray(pts, dtype=np.float64), decimals)))


def _display_truth(scans, coord_mode="cartesian"):
    """全点読込したスキャンの表示座標・色・強度・スキャン番号を連結して返す"""
    pts, cols, ints, sids = [], [], [], []
    for s_idx, s in enumerate(scans):
        if coord_mode != "cartesian":
            transform_scan_coordinate_system(s, coord_mode)
        pts.append(scan_display_points(s))
        c = np.column_stack([s.raw_fields[k] for k in ("colorRed", "colorGreen", "colorBlue")]).astype(np.float32) / 255.0
        cols.append(c)
        ints.append(np.asarray(s.raw_fields["intensity"], dtype=np.float32))
        sids.append(np.full(len(s.points), s_idx, dtype=np.int32))
    return np.vstack(pts), np.vstack(cols), np.concatenate(ints), np.concatenate(sids)


class _Scene(unittest.TestCase):
    """合成 3 スキャン E57: A 床+壁 (bounds あり)、B 床 (pose: Z90° + 平行移動、bounds あり)、C 小さな塊 (bounds なし)"""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = os.path.join(cls.tmp.name, "現場_断面.e57")
        rng = np.random.default_rng(11)
        pts_a = np.vstack([_grid(0.0, 4.0, 0.0, 3.0, 0.0, 0.02), _wall_x(4.0, 0.0, 3.0, 0.0, 2.0, 0.02)])
        pts_b = _grid(0.0, 2.0, 0.0, 2.0, 0.0, 0.02)
        pts_c = np.column_stack([
            rng.uniform(-10.0, -8.0, 2000), rng.uniform(0.0, 2.0, 2000), rng.uniform(0.0, 1.0, 2000)
        ])
        specs = []
        for name, guid, pts, pose, bounds in (
            ("A", "{aaaaaaaa-0000-0000-0000-000000000001}", pts_a, None, True),
            ("B", "{bbbbbbbb-0000-0000-0000-000000000002}", pts_b, (ROT_90Z, (100.0, 200.0, 5.0)), True),
            ("C", "{cccccccc-0000-0000-0000-000000000003}", pts_c, None, False),
        ):
            specs.append(dict(
                points=pts, name=name, guid=guid, pose=pose, bounds=bounds,
                colors=(rng.random((len(pts), 3)) * 255).astype(np.uint8),
                intensity=rng.random(len(pts)).astype(np.float32),
            ))
        _write_e57(cls.path, specs)
        cls.n_total = sum(len(s["points"]) for s in specs)
        cls.full = read_e57_scans(cls.path)
        cls.truth_pts, cls.truth_cols, cls.truth_ints, cls.truth_sids = _display_truth(cls.full)
        cls.load_budget = 20_000  # 総点数 57,803 → ceil(57803/20000) = 3 点に 1 点

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        # 読み直しキャッシュ（<ファイル>.cache）がテスト間で持ち越されないようにする。キャッシュ自体の検証は test_section_cache.py
        remove_cache(self.path)

    def tearDown(self):
        remove_cache(self.path)

    def _subsampled(self, coord_mode="cartesian"):
        scans = read_e57_scans(self.path, point_budget=self.load_budget)
        self.assertTrue(all(s.is_subsampled for s in scans))
        if coord_mode != "cartesian":
            for s in scans:
                transform_scan_coordinate_system(s, coord_mode)
        return scans

    def _truth_in(self, box, pts=None):
        pts = self.truth_pts if pts is None else pts
        return np.flatnonzero(box.contains(pts))


class TestPointChunks(_Scene):
    def test_chunks_match_full_read(self):
        """チャンク流し読みを連結すると全点読みと同じ座標・色・強度になる（チャンクは指定サイズ以下）"""
        s = self.full[0]
        chunks = list(iter_e57_point_chunks(self.path, 0, ["cartesianX", "cartesianY", "cartesianZ", "colorRed", "intensity", "nonexistent"], chunk_size=7_000))
        self.assertGreater(len(chunks), 5)
        self.assertTrue(all(len(c["cartesianX"]) <= 7_000 for c in chunks))
        self.assertTrue(all("nonexistent" not in c for c in chunks))
        xyz = np.column_stack([np.concatenate([c[k] for c in chunks]) for k in ("cartesianX", "cartesianY", "cartesianZ")])
        np.testing.assert_allclose(xyz, s.points, atol=1e-9)
        np.testing.assert_array_equal(np.concatenate([c["colorRed"] for c in chunks]), s.raw_fields["colorRed"])
        np.testing.assert_array_equal(np.concatenate([c["intensity"] for c in chunks]), s.raw_fields["intensity"])
        self.assertEqual(chunks[0]["intensity"].dtype, np.float32)

    def test_invalid_scan_index(self):
        with self.assertRaises(ValueError):
            list(iter_e57_point_chunks(self.path, 99))
        with self.assertRaises(FileNotFoundError):
            list(iter_e57_point_chunks(self.path + ".missing", 0))


class TestSectionBox(unittest.TestCase):
    def test_contains_and_intersect(self):
        box = SectionBox(x_range=(1.0, 2.0), z_range=(-0.5, 0.5))
        pts = np.array([[1.5, 0.0, 0.0], [0.5, 0.0, 0.0], [1.5, 0.0, 1.0], [2.0, 9.0, -0.5]])
        np.testing.assert_array_equal(box.contains(pts), [True, False, False, True])
        self.assertTrue(box.is_active())
        self.assertFalse(SectionBox().is_active())
        corners_out = np.array([[3.0, 0.0, 0.0], [4.0, 1.0, 1.0]])
        corners_in = np.array([[0.0, 0.0, 0.0], [4.0, 1.0, 1.0]])
        self.assertFalse(box.may_intersect(corners_out))
        self.assertTrue(box.may_intersect(corners_in))
        self.assertTrue(box.may_intersect(None))
        self.assertIn("X 1.00〜2.00m", box.describe())

    def test_wall_distance(self):
        # 角度 90° の壁（Y 軸に沿う）: 直交距離 d = -(x - cx)。|x - 2| <= 0.3 が内側
        box = SectionBox(wall=(90.0, 2.0, 1.5, -0.3, 0.3))
        pts = np.array([[2.0, 0.0, 0.0], [2.29, 5.0, 0.0], [1.71, -5.0, 0.0], [2.4, 0.0, 0.0], [1.6, 0.0, 0.0]])
        np.testing.assert_array_equal(box.contains(pts), [True, True, True, False, False])
        self.assertFalse(box.may_intersect(np.array([[3.0, 0.0, 0.0], [5.0, 1.0, 1.0]])))
        self.assertTrue(box.may_intersect(np.array([[0.0, 0.0, 0.0], [5.0, 1.0, 1.0]])))


class TestReload(_Scene):
    def test_x_box_matches_truth_and_skips_far_scan(self):
        """X 範囲の箱: A の床の帯だけが元ファイルから拾われ、範囲外の B は読まずに飛ばし、bounds の無い C は読む"""
        box = SectionBox(x_range=(1.0, 2.0))
        res = reload_section_points(self._subsampled(), box, max_points=0)
        idx = self._truth_in(box)
        self.assertGreater(len(idx), 5000)
        self.assertEqual(res.n_in_box, len(idx))
        self.assertEqual(res.n_points, len(idx))
        self.assertFalse(res.capped)
        self.assertEqual(_as_set(res.points), _as_set(self.truth_pts[idx]))
        self.assertEqual(res.n_scans_skipped, 1, "B は bounds が範囲外なので飛ばす")
        self.assertEqual(res.n_scans_read, 2, "A と（bounds の無い）C を読む")
        self.assertEqual(res.n_read, len(self.full[0].points) + len(self.full[2].points))
        self.assertTrue(np.all(res.scan_ids == 0))
        # 色・強度は元ファイルの同じ点のもの
        truth_map = {k: (c, i) for k, c, i in zip(map(tuple, np.round(self.truth_pts[idx], 5)), self.truth_cols[idx], self.truth_ints[idx])}
        for p, c, i in list(zip(np.round(res.points, 5), res.colors, res.intensities))[:200]:
            tc, ti = truth_map[tuple(p)]
            np.testing.assert_allclose(c, tc, atol=1e-6)
            self.assertEqual(float(i), float(ti))

    def test_posed_scan_box_in_display_coordinates(self):
        """pose（回転＋平行移動）のある B を、表示座標の箱で拾える（A は bounds で飛ばす）"""
        box = SectionBox(x_range=(97.0, 101.0), y_range=(199.0, 203.0))
        res = reload_section_points(self._subsampled(), box, max_points=0)
        idx = self._truth_in(box)
        self.assertEqual(len(idx), len(self.full[1].points), "テスト前提: B の全点が箱に入る")
        self.assertEqual(res.n_in_box, len(idx))
        self.assertEqual(_as_set(res.points), _as_set(self.truth_pts[idx]))
        self.assertTrue(np.all(res.scan_ids == 1))
        self.assertEqual(res.n_scans_skipped, 1)
        corners = scan_bounds_corners(self.full[1])
        self.assertIsNotNone(corners)
        self.assertTrue(box.may_intersect(corners))
        self.assertFalse(SectionBox(x_range=(0.0, 4.0)).may_intersect(corners))
        self.assertIsNone(scan_bounds_corners(self.full[2]), "C には cartesianBounds が無い")

    def test_subsampled_load_restores_density(self):
        """3 点に 1 点の間引き読込でも、読み直しで箱の中は元ファイルの密度に戻る（読み込み済みの点を含む）"""
        box = SectionBox(x_range=(1.0, 2.0), z_range=(-0.01, 0.01))
        scans = self._subsampled()
        loaded_pts = np.vstack([scan_display_points(s) for s in scans])
        loaded_in = loaded_pts[box.contains(loaded_pts)]
        res = reload_section_points(scans, box, max_points=0)
        idx = self._truth_in(box)
        self.assertEqual(res.n_points, len(idx))
        self.assertGreater(res.n_points, 2 * len(loaded_in))
        self.assertTrue(_as_set(loaded_in) <= _as_set(res.points))

    def test_cap_thins_uniformly_and_counts_exactly(self):
        """上限 1,000 点: 返す点数は上限以下、ボックス内の総点数は正確、点は箱の全域に散らばる"""
        box = SectionBox(x_range=(1.0, 2.0))
        res = reload_section_points(self._subsampled(), box, max_points=1_000)
        idx = self._truth_in(box)
        self.assertTrue(res.capped)
        self.assertLessEqual(res.n_points, 1_000)
        self.assertGreater(res.n_points, 400)
        self.assertEqual(res.n_in_box, len(idx))
        self.assertTrue(_as_set(res.points) <= _as_set(self.truth_pts[idx]))
        self.assertLess(abs(float(np.min(res.points[:, 1])) - 0.0), 0.1)
        self.assertLess(abs(float(np.max(res.points[:, 1])) - 3.0), 0.1)
        self.assertEqual(len(res.colors), res.n_points)
        self.assertEqual(len(res.intensities), res.n_points)
        self.assertEqual(len(res.scan_ids), res.n_points)

    def test_wall_box(self):
        """斜め壁の直交スライスでも表示座標で正しく拾う（90° の壁 → d = -(x-2) が -0.31〜0.29 ⇔ x が 1.71〜2.31 の帯。格子点の境界を避ける）"""
        box = SectionBox(wall=(90.0, 2.0, 1.5, -0.31, 0.29), z_range=(-0.01, 0.01))
        res = reload_section_points(self._subsampled(), box, max_points=0)
        direct = np.flatnonzero(
            (self.truth_pts[:, 0] >= 1.71) & (self.truth_pts[:, 0] <= 2.31) & (np.abs(self.truth_pts[:, 2]) <= 0.01)
        )
        self.assertEqual(res.n_in_box, len(direct))
        self.assertEqual(_as_set(res.points), _as_set(self.truth_pts[direct]))

    def test_coord_mode_survey_swap(self):
        """survey_swap（X↔Y）で読み込んだときは、表示座標の箱（Y 範囲）で元ファイルの X の帯が拾われ、bounds の飛ばしも変換後で判定"""
        full = read_e57_scans(self.path)
        truth_pts, _, _, _ = _display_truth(full, "survey_swap")
        box = SectionBox(y_range=(1.0, 2.0), z_range=(-0.01, 0.01))
        res = reload_section_points(self._subsampled("survey_swap"), box, max_points=0)
        idx = np.flatnonzero(box.contains(truth_pts))
        self.assertGreater(len(idx), 5000)
        self.assertEqual(res.n_in_box, len(idx))
        self.assertEqual(_as_set(res.points), _as_set(truth_pts[idx]))
        self.assertEqual(res.n_scans_skipped, 1)

    def test_cancel_and_progress(self):
        box = SectionBox(x_range=(1.0, 2.0))
        calls = []

        def progress(ratio, msg):
            calls.append((ratio, msg))

        res = reload_section_points(self._subsampled(), box, max_points=0, chunk_size=5_000, progress_callback=progress)
        self.assertGreater(len(calls), 3)
        self.assertTrue(all(0.0 <= r <= 1.0 for r, _ in calls))
        self.assertTrue(all(isinstance(m, str) and m for _, m in calls))
        self.assertGreaterEqual(calls[-1][0], 0.99)
        self.assertEqual(res.n_in_box, len(self._truth_in(box)))

        n_checks = []

        def cancelled():
            n_checks.append(1)
            return len(n_checks) > 2

        with self.assertRaises(InterruptedError):
            reload_section_points(self._subsampled(), box, max_points=0, chunk_size=5_000, is_cancelled=cancelled)

    def test_source_validation(self):
        scans = self._subsampled()
        with self.assertRaises(ValueError):
            reload_section_points(scans, SectionBox(), max_points=0)
        scans[0].source_file = None
        with self.assertRaises(ValueError):
            reload_section_points(scans, SectionBox(x_range=(1.0, 2.0)), max_points=0)
        scans = self._subsampled()
        scans[1].source_signature = (scans[1].source_signature[0] + 1, scans[1].source_signature[1])
        with self.assertRaises(RuntimeError):
            reload_section_points(scans, SectionBox(x_range=(1.0, 2.0)), max_points=0)

    def test_color_divisor_follows_loaded_colors(self):
        s = self.full[0]
        self.assertEqual(color_divisor(s), 255.0)
        s16 = type("S", (), {})()
        s16.raw_fields = {"colorRed": np.array([65000]), "colorGreen": np.array([1]), "colorBlue": np.array([2])}
        self.assertEqual(color_divisor(s16), 65535.0)
        s_none = type("S", (), {})()
        s_none.raw_fields = {}
        s_none.field_schemas = {}
        self.assertEqual(color_divisor(s_none), 1.0)


class TestViewerOverlay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def setUp(self):
        rng = np.random.default_rng(3)
        self.pts = rng.random((50_000, 3)) * 10.0
        self.cols = rng.random((50_000, 3)).astype(np.float32)
        self.ints = rng.random(50_000).astype(np.float32)
        self.v = PointCloudViewer()
        self.v.set_shading_enabled(False)
        self.v.set_point_cloud(self.pts, self.cols, self.ints)
        # 箱の中に「元ファイルから読み直した」体の密な点（表示座標）
        rng2 = np.random.default_rng(5)
        n = 20_000
        self.ov_pts = np.column_stack([rng2.random(n) * 10.0, rng2.random(n) * 10.0, 2.0 + rng2.random(n) * 2.0])
        self.ov_cols = rng2.random((n, 3)).astype(np.float32)
        self.ov_ints = rng2.random(n).astype(np.float32)
        self.ov_sids = np.zeros(n, dtype=np.int32)

    def tearDown(self):
        self.v.set_point_cloud(None)
        destroy_widget(self.v, close=False)

    def _set_overlay(self):
        self.v.set_section_overlay(self.ov_pts, self.ov_cols, self.ov_ints, self.ov_sids, n_in_box=len(self.ov_pts), capped=False)

    def test_spec_and_lifecycle(self):
        v = self.v
        self.assertIsNone(v.section_box_spec())
        v.set_section_z(True, 2.0, 4.0)
        spec = v.section_box_spec()
        self.assertEqual(spec, {"x_range": None, "y_range": None, "z_range": (2.0, 4.0), "wall": None})
        self.assertEqual(SectionBox(**spec).z_range, (2.0, 4.0))

        self._set_overlay()
        self.assertTrue(v.has_section_overlay)
        self.assertEqual(len(v.overlay_points), len(self.ov_pts))
        np.testing.assert_allclose(v.overlay_points, (self.ov_pts - v.center_offset).astype(np.float32), atol=1e-6)
        self.assertEqual(v.overlay_box_total, len(self.ov_pts))
        v.validate_invariants()
        # 読み込み済みの点は変わらない（表示専用）
        self.assertEqual(len(v.points_raw), len(self.pts))

        # 箱が変わると消える
        v.set_section_z(True, 2.5, 4.0)
        self.assertFalse(v.has_section_overlay)
        v.validate_invariants()
        # プレビュー（ドラッグ中）でも消える
        self._set_overlay()
        v.set_section_z(True, 2.6, 4.0, is_preview=True)
        self.assertFalse(v.has_section_overlay)
        # 同じ箱のまま表示予算やレイヤーが変わっても残る
        v.set_section_z(True, 2.6, 4.0)
        self._set_overlay()
        v.set_point_budget(10_000)
        self.assertTrue(v.has_section_overlay)
        v.set_section_full_detail(False)
        self.assertTrue(v.has_section_overlay)
        v.validate_invariants()
        # 断面解除で消える
        v.set_section_z(False, 2.6, 4.0)
        self.assertFalse(v.has_section_overlay)
        # 断面が無効なら設定しても付かない
        self._set_overlay()
        self.assertFalse(v.has_section_overlay)
        # 点群入替で消える
        v.set_section_z(True, 2.0, 4.0)
        self._set_overlay()
        self.assertTrue(v.has_section_overlay)
        v.set_point_cloud(self.pts[:1000])
        self.assertFalse(v.has_section_overlay)
        v.validate_invariants()

    def test_wall_spec(self):
        v = self.v
        angle = v.set_wall_align_points(np.array([2.0, 0.0, 0.0]), np.array([2.0, 3.0, 0.0]))
        self.assertAlmostEqual(angle, 90.0)
        v.set_section_wall(True, -0.3, 0.3)
        spec = v.section_box_spec()
        self.assertIsNotNone(spec["wall"])
        self.assertAlmostEqual(spec["wall"][0], 90.0)
        self.assertAlmostEqual(spec["wall"][1], 2.0)
        self.assertAlmostEqual(spec["wall"][2], 1.5)
        self.assertEqual(spec["wall"][3:], (-0.3, 0.3))
        self._set_overlay()
        self.assertTrue(v.has_section_overlay)
        v.clear_wall_alignment()
        self.assertFalse(v.has_section_overlay)
        v.validate_invariants()

    def test_recolor_follows_color_mode(self):
        v = self.v
        v.set_section_z(True, 2.0, 4.0)
        v.set_color_mode(PointCloudViewer.COLOR_MODE_RGB)
        self._set_overlay()
        np.testing.assert_allclose(v.overlay_colors, self.ov_cols, atol=1e-6)

        v.set_color_mode(PointCloudViewer.COLOR_MODE_HEIGHT)
        self.assertTrue(v.has_section_overlay)
        h_cols = v.overlay_colors.copy()
        self.assertEqual(h_cols.shape, self.ov_cols.shape)
        self.assertFalse(np.allclose(h_cols, self.ov_cols))
        self.assertTrue(np.all((h_cols >= 0.0) & (h_cols <= 1.0)))
        # 高さは本体と同じ範囲で正規化される: 同じ高さの点は本体と同じ色
        z_min, z_max = v._height_range()
        t = np.clip((v.overlay_points[:, 2] - z_min) / (z_max - z_min), 0.0, 1.0).astype(np.float32)
        np.testing.assert_allclose(h_cols, v._colormap_turbo(t), atol=1e-6)

        v.set_color_mode(PointCloudViewer.COLOR_MODE_INTENSITY)
        i_cols = v.overlay_colors.copy()
        self.assertFalse(np.allclose(i_cols, h_cols))
        p_low, p_high = v._intensity_stretch()
        t_i = np.clip((self.ov_ints - p_low) / (p_high - p_low), 0.0, 1.0).astype(np.float32)
        np.testing.assert_allclose(i_cols, v._colormap_grayscale(t_i), atol=1e-6)

        v.set_color_mode(PointCloudViewer.COLOR_MODE_SCAN)
        self.assertTrue(np.allclose(v.overlay_colors, v.SCAN_PALETTE[0]))

        # 陰影 ON でも長さと範囲は保たれる
        v.set_shading_enabled(True)
        v.set_color_mode(PointCloudViewer.COLOR_MODE_RGB)
        self.assertEqual(len(v.overlay_colors), len(self.ov_pts))
        self.assertTrue(np.all((v.overlay_colors >= 0.0) & (v.overlay_colors <= 1.0)))
        v.validate_invariants()

    def test_overlay_without_colors_uses_fallback(self):
        v = self.v
        v.set_section_z(True, 2.0, 4.0)
        v.set_color_mode(PointCloudViewer.COLOR_MODE_RGB)
        v.set_section_overlay(self.ov_pts, None, None, None, n_in_box=123, capped=True)
        self.assertTrue(v.has_section_overlay)
        self.assertTrue(v.overlay_capped)
        self.assertEqual(v.overlay_box_total, 123)
        self.assertEqual(v.overlay_colors.shape, (len(self.ov_pts), 3))
        v.validate_invariants()


class TestMainWindowReload(_Scene):
    def setUp(self):
        from gui.main_window import MainWindow

        super().setUp()
        self.win = MainWindow()
        self.win._reset_app_settings()
        self.win.show()
        QApplication.processEvents()

    def tearDown(self):
        destroy_widget(self.win)
        super().tearDown()

    def test_button_guards_without_data_or_section(self):
        win = self.win
        self.assertTrue(hasattr(win, "btn_sec_reload") and hasattr(win, "btn_sec_reload_clear"))
        win.btn_sec_reload.click()
        self.assertFalse(win._is_background_busy())
        self.assertFalse(win.viewer.has_section_overlay)
        win.btn_sec_reload_clear.click()  # 何も無くても落ちない
        # 合成データ（読み込み元なし）では読み直さない
        win.viewer.set_point_cloud(np.random.default_rng(1).random((1000, 3)))
        win.chk_sec_z.setChecked(True)
        win.btn_sec_reload.click()
        self.assertFalse(win._is_background_busy())
        self.assertFalse(win.viewer.has_section_overlay)

    def test_real_load_path_subsampled_then_reload(self):
        """実際の読込経路: 間引き読込 → Z 断面（床 ±0.05m）→ 読み直し → 床の全点が重ね描きされる"""
        win = self.win
        win.combo_load_budget.addItem("テスト用 2 万点", self.load_budget)
        win.combo_load_budget.setCurrentIndex(win.combo_load_budget.count() - 1)
        self.assertTrue(win._load_file_to_viewer(self.path) is not False)
        self.assertTrue(win._wait_for_background(60_000), "読込が終わりませんでした")
        QApplication.processEvents()
        self.assertTrue(win.current_scans and all(s.is_subsampled for s in win.current_scans))
        n_loaded = len(win.viewer.points_raw)
        self.assertLess(n_loaded, self.n_total)

        win.sec_origin_mode_combo.setCurrentIndex(max(0, win.sec_origin_mode_combo.findData("center")))
        win.spin_pos_z.setValue(0.0)
        win.spin_thick_z.setValue(0.1)
        win.chk_sec_z.setChecked(True)
        self.assertTrue(win.viewer.is_section_active())
        spec = win.viewer.section_box_spec()
        self.assertAlmostEqual(spec["z_range"][0], -0.05)
        self.assertAlmostEqual(spec["z_range"][1], 0.05)

        win.btn_sec_reload.click()
        self.assertTrue(win._wait_for_background(60_000), "読み直しが終わりませんでした")
        QApplication.processEvents()
        v = win.viewer
        self.assertTrue(v.has_section_overlay)
        idx = self._truth_in(SectionBox(**spec))
        self.assertEqual(len(v.overlay_points), len(idx), "箱の中の元ファイルの全点が重ね描きされる")
        self.assertEqual(v.overlay_box_total, len(idx))
        self.assertGreater(len(v.overlay_points), v.section_box_count * 2, "読み込み済み（3 点に 1 点）より密")
        self.assertEqual(len(v.points_raw), n_loaded, "読み込み済みの点は変わらない")
        v.validate_invariants()

        # 箱を動かすと消え、「重ね描きを消す」も効く
        win.spin_pos_z.setValue(0.3)
        self.assertFalse(v.has_section_overlay)
        win.spin_pos_z.setValue(0.0)
        win.btn_sec_reload.click()
        self.assertTrue(win._wait_for_background(60_000))
        QApplication.processEvents()
        self.assertTrue(v.has_section_overlay)
        win.btn_sec_reload_clear.click()
        self.assertFalse(v.has_section_overlay)
        v.validate_invariants()


if __name__ == "__main__":
    unittest.main()
