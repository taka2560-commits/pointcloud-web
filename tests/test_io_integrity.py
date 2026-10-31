"""
E57入出力の完全性（ロスレス・座標値不変・元の形式保持・全属性データ無欠損・異常系耐性）を検証する徹底テストスイート
"""

import os
import tempfile
import unittest
import numpy as np
from pye57 import libe57
from pye57.utils import get_node

from core.e57_io import read_e57_scans, write_e57_scans
from core.processor import FilterConfig, PointCloudProcessor


class TestE57IOIntegrity(unittest.TestCase):
    """入力・出力時のエラーバグ、形式維持、座標値の完全一致、データ無欠損を検証するテスト"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="e57_integrity_test_")

    def tearDown(self):
        for root_dir, _, files in os.walk(self.temp_dir, topdown=False):
            for name in files:
                try:
                    os.remove(os.path.join(root_dir, name))
                except Exception:
                    pass
            try:
                os.rmdir(root_dir)
            except Exception:
                pass

    def test_1_double_precision_exact_coordinates(self):
        """
        1. 倍精度 (FloatNode E57_DOUBLE) の巨大な公共座標値が1ビットも劣化せず完全一致することを検証
        （従来の pye57.write_scan_raw では E57_SINGLE に丸められて数センチの誤差が出ていた問題の解消確認）
        """
        in_path = os.path.join(self.temp_dir, "double_in.e57")
        out_path = os.path.join(self.temp_dir, "double_out.e57")

        n = 500
        np.random.seed(101)
        # 公共座標系のような大きな値 + サブミリの精密な小数部
        orig_x = 54321.12345678901 + np.random.uniform(-10.0, 10.0, n)
        orig_y = -128765.98765432109 + np.random.uniform(-10.0, 10.0, n)
        orig_z = 345.67890123456 + np.random.uniform(-5.0, 5.0, n)

        imf = libe57.ImageFile(in_path, "w")
        imf.extensionsAdd("", libe57.E57_V1_0_URI)
        root = imf.root()
        data3d = libe57.VectorNode(imf, True)
        root.set("data3D", data3d)

        scan = libe57.StructureNode(imf)
        scan.set("guid", libe57.StringNode(imf, "{11111111-2222-3333-4444-555555555555}"))
        proto = libe57.StructureNode(imf)
        proto.set("cartesianX", libe57.FloatNode(imf, 0.0, libe57.E57_DOUBLE))
        proto.set("cartesianY", libe57.FloatNode(imf, 0.0, libe57.E57_DOUBLE))
        proto.set("cartesianZ", libe57.FloatNode(imf, 0.0, libe57.E57_DOUBLE))

        codecs = libe57.VectorNode(imf, True)
        pts = libe57.CompressedVectorNode(imf, proto, codecs)
        scan.set("points", pts)
        data3d.append(scan)

        bufs = libe57.VectorSourceDestBuffer()
        bufs.append(libe57.SourceDestBuffer(imf, "cartesianX", orig_x, n, True, True))
        bufs.append(libe57.SourceDestBuffer(imf, "cartesianY", orig_y, n, True, True))
        bufs.append(libe57.SourceDestBuffer(imf, "cartesianZ", orig_z, n, True, True))
        w = pts.writer(bufs)
        w.write(n)
        w.close()
        imf.close()

        # 読み込み -> インデックス抽出 -> 書き出し -> 再読み込み
        scans = read_e57_scans(in_path)
        self.assertEqual(len(scans), 1)
        keep_idx = np.arange(0, n, 2)  # 偶数インデックスのみ抽出
        filtered_scan = scans[0].filter_by_indices(keep_idx)
        write_e57_scans(out_path, [filtered_scan])

        reloaded = read_e57_scans(out_path)[0]
        self.assertEqual(reloaded.point_count, len(keep_idx))

        # 形式が E57_DOUBLE のままであることを確認
        self.assertEqual(reloaded.field_schemas["cartesianX"].node_type, "Float")
        self.assertEqual(reloaded.field_schemas["cartesianX"].precision, libe57.E57_DOUBLE)

        # 座標値の誤差が完全に 0.0 (ビット完全一致) であることを確認
        np.testing.assert_array_equal(reloaded.points[:, 0], orig_x[keep_idx])
        np.testing.assert_array_equal(reloaded.points[:, 1], orig_y[keep_idx])
        np.testing.assert_array_equal(reloaded.points[:, 2], orig_z[keep_idx])

    def test_2_scaled_integer_exact_preservation(self):
        """
        2. ScaledIntegerNode 形式（地上型レーザースキャナー標準）の形式・スケール・オフセット・座標値が
        1ビットの丸め誤差もなく完全保持されることを検証
        """
        in_path = os.path.join(self.temp_dir, "scaled_in.e57")
        out_path = os.path.join(self.temp_dir, "scaled_out.e57")

        n = 300
        scale = 0.0001
        offset_x, offset_y, offset_z = 10000.0, -20000.0, 50.0
        raw_ix = np.arange(-150, 150, dtype=np.int64) * 123
        raw_iy = np.arange(-150, 150, dtype=np.int64) * -456
        raw_iz = np.arange(-150, 150, dtype=np.int64) * 78

        imf = libe57.ImageFile(in_path, "w")
        imf.extensionsAdd("", libe57.E57_V1_0_URI)
        root = imf.root()
        data3d = libe57.VectorNode(imf, True)
        root.set("data3D", data3d)

        scan = libe57.StructureNode(imf)
        scan.set("guid", libe57.StringNode(imf, "{22222222-3333-4444-5555-666666666666}"))
        proto = libe57.StructureNode(imf)
        proto.set("cartesianX", libe57.ScaledIntegerNode(imf, 0, -10000000, 10000000, scale, offset_x))
        proto.set("cartesianY", libe57.ScaledIntegerNode(imf, 0, -10000000, 10000000, scale, offset_y))
        proto.set("cartesianZ", libe57.ScaledIntegerNode(imf, 0, -10000000, 10000000, scale, offset_z))

        codecs = libe57.VectorNode(imf, True)
        pts = libe57.CompressedVectorNode(imf, proto, codecs)
        scan.set("points", pts)
        data3d.append(scan)

        bufs = libe57.VectorSourceDestBuffer()
        bufs.append(libe57.SourceDestBuffer(imf, "cartesianX", raw_ix, n, True, False))
        bufs.append(libe57.SourceDestBuffer(imf, "cartesianY", raw_iy, n, True, False))
        bufs.append(libe57.SourceDestBuffer(imf, "cartesianZ", raw_iz, n, True, False))
        w = pts.writer(bufs)
        w.write(n)
        w.close()
        imf.close()

        # 読み込み -> フィルタ抽出 -> 書き出し -> 再読み込み
        scans = read_e57_scans(in_path)
        keep_idx = np.array([0, 5, 10, 50, 123, 299], dtype=np.int64)
        filtered = scans[0].filter_by_indices(keep_idx)
        write_e57_scans(out_path, [filtered])

        reloaded = read_e57_scans(out_path)[0]
        # 形式が ScaledInteger のままであることを確認
        self.assertEqual(reloaded.field_schemas["cartesianX"].node_type, "ScaledInteger")
        self.assertEqual(reloaded.field_schemas["cartesianX"].scale, scale)
        self.assertEqual(reloaded.field_schemas["cartesianX"].offset, offset_x)

        # 生の整数値が1ビットの狂いもなく完全一致することを確認
        np.testing.assert_array_equal(reloaded.raw_unscaled_fields["cartesianX"], raw_ix[keep_idx])
        np.testing.assert_array_equal(reloaded.raw_unscaled_fields["cartesianY"], raw_iy[keep_idx])
        np.testing.assert_array_equal(reloaded.raw_unscaled_fields["cartesianZ"], raw_iz[keep_idx])

        # スケーリング後の実座標も完全一致することを確認
        np.testing.assert_array_equal(reloaded.points, scans[0].points[keep_idx])

    def test_3_all_extended_fields_and_large_indices(self):
        """
        3. 法線(normalX/Y/Z)、タイムスタンプ(timeStamp)、マルチリターン(returnIndex/returnCount)、
        65535を超える巨大なrowIndex/columnIndex、無効フラグ等がエラーなく読み書きでき、1つも欠損しないことを検証
        """
        in_path = os.path.join(self.temp_dir, "ext_in.e57")
        out_path = os.path.join(self.temp_dir, "ext_out.e57")

        n = 100
        x = np.linspace(-5.0, 5.0, n, dtype=np.float64)
        y = np.linspace(-10.0, 10.0, n, dtype=np.float64)
        z = np.linspace(0.0, 2.0, n, dtype=np.float64)
        nx = np.ones(n, dtype=np.float32) * 0.577
        ny = np.ones(n, dtype=np.float32) * 0.577
        nz = np.ones(n, dtype=np.float32) * 0.577
        ts = np.linspace(1000.0, 2000.0, n, dtype=np.float64)
        # uint16 (65535) を超える巨大な行インデックス
        row_idx = np.arange(100000, 100000 + n, dtype=np.int64)
        col_idx = np.arange(200000, 200000 + n, dtype=np.int64)
        ret_idx = np.ones(n, dtype=np.int64)
        ret_cnt = np.full(n, 2, dtype=np.int64)
        r = np.random.randint(0, 256, n, dtype=np.int64)
        g = np.random.randint(0, 256, n, dtype=np.int64)
        b = np.random.randint(0, 256, n, dtype=np.int64)

        imf = libe57.ImageFile(in_path, "w")
        imf.extensionsAdd("", libe57.E57_V1_0_URI)
        imf.extensionsAdd("nor", "http://www.libe57.org/E57_NOR_surface_normals.txt")
        root = imf.root()
        root.set("coordinateMetadata", libe57.StringNode(imf, "JGD2011 / Japan Plane Rectangular CS IX"))
        data3d = libe57.VectorNode(imf, True)
        root.set("data3D", data3d)

        scan = libe57.StructureNode(imf)
        scan.set("guid", libe57.StringNode(imf, "{33333333-4444-5555-6666-777777777777}"))
        scan.set("name", libe57.StringNode(imf, "Station_01_Full"))
        scan.set("sensorVendor", libe57.StringNode(imf, "TestScannerCorp"))

        # あえて x, y, z, w の順番で rotation ノードを構築（pye57の順番依存バグが起きないか検証）
        pose = libe57.StructureNode(imf)
        rot = libe57.StructureNode(imf)
        rot.set("x", libe57.FloatNode(imf, 0.1, libe57.E57_DOUBLE))
        rot.set("y", libe57.FloatNode(imf, 0.2, libe57.E57_DOUBLE))
        rot.set("z", libe57.FloatNode(imf, 0.3, libe57.E57_DOUBLE))
        rot.set("w", libe57.FloatNode(imf, 0.9273618495495704, libe57.E57_DOUBLE))
        pose.set("rotation", rot)
        tr = libe57.StructureNode(imf)
        tr.set("x", libe57.FloatNode(imf, 12.345, libe57.E57_DOUBLE))
        tr.set("y", libe57.FloatNode(imf, -67.890, libe57.E57_DOUBLE))
        tr.set("z", libe57.FloatNode(imf, 3.456, libe57.E57_DOUBLE))
        pose.set("translation", tr)
        scan.set("pose", pose)

        proto = libe57.StructureNode(imf)
        proto.set("cartesianX", libe57.FloatNode(imf, 0.0, libe57.E57_DOUBLE))
        proto.set("cartesianY", libe57.FloatNode(imf, 0.0, libe57.E57_DOUBLE))
        proto.set("cartesianZ", libe57.FloatNode(imf, 0.0, libe57.E57_DOUBLE))
        proto.set("nor:normalX", libe57.FloatNode(imf, 0.0, libe57.E57_SINGLE))
        proto.set("nor:normalY", libe57.FloatNode(imf, 0.0, libe57.E57_SINGLE))
        proto.set("nor:normalZ", libe57.FloatNode(imf, 0.0, libe57.E57_SINGLE))
        proto.set("timeStamp", libe57.FloatNode(imf, 1000.0, libe57.E57_DOUBLE))
        proto.set("rowIndex", libe57.IntegerNode(imf, 100000, 0, 1000000))
        proto.set("columnIndex", libe57.IntegerNode(imf, 200000, 0, 1000000))
        proto.set("returnIndex", libe57.IntegerNode(imf, 1, 1, 10))
        proto.set("returnCount", libe57.IntegerNode(imf, 2, 1, 10))
        proto.set("colorRed", libe57.IntegerNode(imf, 0, 0, 255))
        proto.set("colorGreen", libe57.IntegerNode(imf, 0, 0, 255))
        proto.set("colorBlue", libe57.IntegerNode(imf, 0, 0, 255))

        codecs = libe57.VectorNode(imf, True)
        pts = libe57.CompressedVectorNode(imf, proto, codecs)
        scan.set("points", pts)
        data3d.append(scan)

        bufs = libe57.VectorSourceDestBuffer()
        bufs.append(libe57.SourceDestBuffer(imf, "cartesianX", x, n, True, True))
        bufs.append(libe57.SourceDestBuffer(imf, "cartesianY", y, n, True, True))
        bufs.append(libe57.SourceDestBuffer(imf, "cartesianZ", z, n, True, True))
        bufs.append(libe57.SourceDestBuffer(imf, "nor:normalX", nx, n, True, True))
        bufs.append(libe57.SourceDestBuffer(imf, "nor:normalY", ny, n, True, True))
        bufs.append(libe57.SourceDestBuffer(imf, "nor:normalZ", nz, n, True, True))
        bufs.append(libe57.SourceDestBuffer(imf, "timeStamp", ts, n, True, True))
        bufs.append(libe57.SourceDestBuffer(imf, "rowIndex", row_idx, n, True, True))
        bufs.append(libe57.SourceDestBuffer(imf, "columnIndex", col_idx, n, True, True))
        bufs.append(libe57.SourceDestBuffer(imf, "returnIndex", ret_idx, n, True, True))
        bufs.append(libe57.SourceDestBuffer(imf, "returnCount", ret_cnt, n, True, True))
        bufs.append(libe57.SourceDestBuffer(imf, "colorRed", r, n, True, True))
        bufs.append(libe57.SourceDestBuffer(imf, "colorGreen", g, n, True, True))
        bufs.append(libe57.SourceDestBuffer(imf, "colorBlue", b, n, True, True))
        w = pts.writer(bufs)
        w.write(n)
        w.close()
        imf.close()

        # 読み込み検証
        scans = read_e57_scans(in_path)
        s = scans[0]
        # ファイルクローズ後の header アクセス安全性検証 (bad_weak_ptr にならないこと)
        self.assertEqual(s.header.point_count, n)
        self.assertEqual(s.header.name, "Station_01_Full")
        self.assertEqual(s.header.guid, "{33333333-4444-5555-6666-777777777777}")
        self.assertTrue(s.header.has_pose())
        # rotation の w, x, y, z が XML の並び順に影響されず正しく読み込まれていること
        np.testing.assert_allclose(s.rotation, [0.9273618495495704, 0.1, 0.2, 0.3])
        np.testing.assert_allclose(s.translation, [12.345, -67.890, 3.456])

        # フィルタリングして書き出し -> 再読み込み
        keep_idx = np.arange(10, 90)
        write_e57_scans(out_path, [s.filter_by_indices(keep_idx)])

        reloaded = read_e57_scans(out_path)[0]
        self.assertEqual(reloaded.point_count, len(keep_idx))
        self.assertEqual(reloaded.coordinate_metadata, "JGD2011 / Japan Plane Rectangular CS IX")
        self.assertEqual(reloaded.header.name, "Station_01_Full")
        np.testing.assert_allclose(reloaded.rotation, [0.9273618495495704, 0.1, 0.2, 0.3])
        np.testing.assert_allclose(reloaded.translation, [12.345, -67.890, 3.456])

        # 全属性フィールドが欠損なく完全一致していることを確認
        np.testing.assert_array_equal(reloaded.raw_fields["nor:normalX"], nx[keep_idx])
        np.testing.assert_array_equal(reloaded.raw_fields["timeStamp"], ts[keep_idx])
        np.testing.assert_array_equal(reloaded.raw_fields["rowIndex"], row_idx[keep_idx])
        np.testing.assert_array_equal(reloaded.raw_fields["columnIndex"], col_idx[keep_idx])
        np.testing.assert_array_equal(reloaded.raw_fields["returnIndex"], ret_idx[keep_idx])
        np.testing.assert_array_equal(reloaded.raw_fields["returnCount"], ret_cnt[keep_idx])
        np.testing.assert_array_equal(reloaded.raw_fields["colorRed"], r[keep_idx])

    def test_4_processor_apply_filters_to_points_all_filters(self):
        """
        4. PointCloudProcessor.apply_filters_to_points で距離フィルタ・反射強度フィルタ・ボクセル・SOR・RORの
        すべてがエラーなく正常動作することを検証
        """
        np.random.seed(42)
        pts = np.random.normal(0.0, 1.0, (200, 3))
        intensities = np.linspace(0.0, 1.0, 200)

        config = FilterConfig(
            use_distance_filter=True,
            min_distance=0.1,
            max_distance=5.0,
            use_intensity_filter=True,
            min_intensity=0.2,
            max_intensity=0.9,
            use_voxel_downsample=True,
            voxel_size=0.05,
            use_sor=True,
            sor_neighbors=10,
            sor_std_ratio=2.0,
            use_ror=True,
            ror_radius=1.5,
            ror_min_points=3,
        )
        proc = PointCloudProcessor(config)
        keep_idx, rem_idx = proc.apply_filters_to_points(pts, intensities=intensities)
        self.assertGreater(len(keep_idx), 0)
        self.assertEqual(len(keep_idx) + len(rem_idx), 200)

    def test_5_inplace_coordinate_modification_and_bounds_update(self):
        """
        5. ScaledIntegerNode スキャンで既存の cartesianBounds がある状態から：
           (A) 外れ値除去で範囲が縮小した際に cartesianBounds が ErrorSetTwice なく最新化されること
           (B) scan.points をインプレース変更した際に変更後の座標値が正確に保存されること
        """
        in_path = os.path.join(self.temp_dir, "bounds_scaled_in.e57")
        out_filtered_path = os.path.join(self.temp_dir, "bounds_scaled_filtered.e57")
        out_modified_path = os.path.join(self.temp_dir, "bounds_scaled_modified.e57")

        n = 100
        scale = 0.001
        raw_ix = np.arange(0, n, dtype=np.int64) * 1000  # 0.0m ~ 99.0m
        raw_iy = np.arange(0, n, dtype=np.int64) * 500   # 0.0m ~ 49.5m
        raw_iz = np.arange(0, n, dtype=np.int64) * 200   # 0.0m ~ 19.8m

        imf = libe57.ImageFile(in_path, "w")
        imf.extensionsAdd("", libe57.E57_V1_0_URI)
        root = imf.root()
        data3d = libe57.VectorNode(imf, True)
        root.set("data3D", data3d)

        scan = libe57.StructureNode(imf)
        scan.set("guid", libe57.StringNode(imf, "{55555555-6666-7777-8888-999999999999}"))
        bbox = libe57.StructureNode(imf)
        bbox.set("xMinimum", libe57.FloatNode(imf, -999.0, libe57.E57_DOUBLE))
        bbox.set("xMaximum", libe57.FloatNode(imf, 999.0, libe57.E57_DOUBLE))
        bbox.set("yMinimum", libe57.FloatNode(imf, -999.0, libe57.E57_DOUBLE))
        bbox.set("yMaximum", libe57.FloatNode(imf, 999.0, libe57.E57_DOUBLE))
        bbox.set("zMinimum", libe57.FloatNode(imf, -999.0, libe57.E57_DOUBLE))
        bbox.set("zMaximum", libe57.FloatNode(imf, 999.0, libe57.E57_DOUBLE))
        scan.set("cartesianBounds", bbox)

        proto = libe57.StructureNode(imf)
        proto.set("cartesianX", libe57.ScaledIntegerNode(imf, 0, -10000000, 10000000, scale, 0.0))
        proto.set("cartesianY", libe57.ScaledIntegerNode(imf, 0, -10000000, 10000000, scale, 0.0))
        proto.set("cartesianZ", libe57.ScaledIntegerNode(imf, 0, -10000000, 10000000, scale, 0.0))

        codecs = libe57.VectorNode(imf, True)
        pts = libe57.CompressedVectorNode(imf, proto, codecs)
        scan.set("points", pts)
        data3d.append(scan)

        bufs = libe57.VectorSourceDestBuffer()
        bufs.append(libe57.SourceDestBuffer(imf, "cartesianX", raw_ix, n, True, False))
        bufs.append(libe57.SourceDestBuffer(imf, "cartesianY", raw_iy, n, True, False))
        bufs.append(libe57.SourceDestBuffer(imf, "cartesianZ", raw_iz, n, True, False))
        w = pts.writer(bufs)
        w.write(n)
        w.close()
        imf.close()

        # (A) 先頭10点 (X: 0.0m ~ 9.0m) だけを抽出して書き出し -> cartesianBounds が 0.0~9.0 に更新されること
        loaded = read_e57_scans(in_path)[0]
        sub_scan = loaded.filter_by_indices(np.arange(0, 10))
        write_e57_scans(out_filtered_path, [sub_scan])

        re_filtered = read_e57_scans(out_filtered_path)[0]
        b_children = re_filtered.meta_snapshot["children"]["cartesianBounds"]["children"]
        self.assertAlmostEqual(b_children["xMinimum"]["value"], 0.0)
        self.assertAlmostEqual(b_children["xMaximum"]["value"], 9.0)

        # (B) インプレースで X 座標に +25.0m 加算して書き出し -> 変更後の座標が保存されていること
        loaded.points[:, 0] += 25.0
        self.assertTrue(loaded.is_coordinates_modified())
        write_e57_scans(out_modified_path, [loaded])
        re_mod = read_e57_scans(out_modified_path)[0]
        np.testing.assert_allclose(re_mod.points[:, 0], (raw_ix * scale) + 25.0, atol=1e-6)

    def test_6_non_finite_nan_inf_sanitization_and_strict_validation(self):
        """
        6. E57に NaN / Inf 座標が含まれている場合の安全な自動除外と、
           メモリ上で NaN が混入した破損データの書き込み阻止 (validate) を検証
        """
        in_path = os.path.join(self.temp_dir, "nan_in.e57")
        out_path = os.path.join(self.temp_dir, "nan_out.e57")

        n = 50
        x = np.linspace(0.0, 10.0, n, dtype=np.float64)
        y = np.linspace(0.0, 10.0, n, dtype=np.float64)
        z = np.linspace(0.0, 10.0, n, dtype=np.float64)
        col_r = np.arange(n, dtype=np.uint8)
        # 3点に NaN / Inf を混入
        x[5] = np.nan
        y[15] = np.inf
        z[25] = -np.inf

        imf = libe57.ImageFile(in_path, "w")
        imf.extensionsAdd("", libe57.E57_V1_0_URI)
        root = imf.root()
        data3d = libe57.VectorNode(imf, True)
        root.set("data3D", data3d)

        scan = libe57.StructureNode(imf)
        scan.set("guid", libe57.StringNode(imf, "{66666666-7777-8888-9999-aaaaaaaaaaaa}"))
        proto = libe57.StructureNode(imf)
        proto.set("cartesianX", libe57.FloatNode(imf, 0.0, libe57.E57_DOUBLE))
        proto.set("cartesianY", libe57.FloatNode(imf, 0.0, libe57.E57_DOUBLE))
        proto.set("cartesianZ", libe57.FloatNode(imf, 0.0, libe57.E57_DOUBLE))
        proto.set("colorRed", libe57.IntegerNode(imf, 0, 0, 255))

        codecs = libe57.VectorNode(imf, True)
        pts = libe57.CompressedVectorNode(imf, proto, codecs)
        scan.set("points", pts)
        data3d.append(scan)

        bufs = libe57.VectorSourceDestBuffer()
        bufs.append(libe57.SourceDestBuffer(imf, "cartesianX", x, n, True, True))
        bufs.append(libe57.SourceDestBuffer(imf, "cartesianY", y, n, True, True))
        bufs.append(libe57.SourceDestBuffer(imf, "cartesianZ", z, n, True, True))
        bufs.append(libe57.SourceDestBuffer(imf, "colorRed", col_r, n, True, True))
        w = pts.writer(bufs)
        w.write(n)
        w.close()
        imf.close()

        # 読み込み時に非有限値3点が安全に除外され、47点の有効データのみが揃うこと
        scans = read_e57_scans(in_path)
        self.assertEqual(scans[0].point_count, 47)
        self.assertTrue(np.isfinite(scans[0].points).all())
        self.assertEqual(len(scans[0].raw_fields["colorRed"]), 47)

        # 故意に NaN を混入させて write_e57_scans を呼ぶと ValueError で拒否されること
        scans[0].points[0, 0] = np.nan
        with self.assertRaises(ValueError):
            write_e57_scans(out_path, scans)
        self.assertFalse(os.path.exists(out_path))

    def test_7_file_load_worker_preserves_all_points_above_display_budget(self):
        """
        7. 表示用 Point Budget (例: 300点) より多い点数 (例: 1,200点) のE57ファイルを
           FileLoadWorker で読み込んだ際、表示配列は間引かれるが原本データ (points_raw / scans) は
           1,200点すべて無欠損で保持され、保存時に1,200点すべて出力されることを検証
        """
        from gui.worker_thread import FileLoadWorker, FileSaveWorker

        in_path = os.path.join(self.temp_dir, "budget_full_in.e57")
        out_path = os.path.join(self.temp_dir, "budget_full_out.e57")

        n = 1200
        pts_orig = np.random.uniform(-20.0, 20.0, (n, 3)).astype(np.float64)
        orig_x = np.ascontiguousarray(pts_orig[:, 0])
        orig_y = np.ascontiguousarray(pts_orig[:, 1])
        orig_z = np.ascontiguousarray(pts_orig[:, 2])
        imf = libe57.ImageFile(in_path, "w")
        imf.extensionsAdd("", libe57.E57_V1_0_URI)
        root = imf.root()
        data3d = libe57.VectorNode(imf, True)
        root.set("data3D", data3d)
        scan = libe57.StructureNode(imf)
        scan.set("guid", libe57.StringNode(imf, "{77777777-8888-9999-aaaa-bbbbbbbbbbbb}"))
        proto = libe57.StructureNode(imf)
        proto.set("cartesianX", libe57.FloatNode(imf, 0.0, libe57.E57_DOUBLE))
        proto.set("cartesianY", libe57.FloatNode(imf, 0.0, libe57.E57_DOUBLE))
        proto.set("cartesianZ", libe57.FloatNode(imf, 0.0, libe57.E57_DOUBLE))
        codecs = libe57.VectorNode(imf, True)
        pts = libe57.CompressedVectorNode(imf, proto, codecs)
        scan.set("points", pts)
        data3d.append(scan)
        bufs = libe57.VectorSourceDestBuffer()
        bufs.append(libe57.SourceDestBuffer(imf, "cartesianX", orig_x, n, True, True))
        bufs.append(libe57.SourceDestBuffer(imf, "cartesianY", orig_y, n, True, True))
        bufs.append(libe57.SourceDestBuffer(imf, "cartesianZ", orig_z, n, True, True))
        w = pts.writer(bufs)
        w.write(n)
        w.close()
        imf.close()

        holder = []
        worker = FileLoadWorker(in_path, point_budget=300)
        worker.finished_success.connect(lambda pre: holder.append(pre))
        worker.run()
        self.assertEqual(len(holder), 1)
        preloaded = holder[0]

        # 表示用配列は Point Budget (300点) にサンプリングされているが、原本は 1,200 点すべて保持されていること
        self.assertLessEqual(len(preloaded.display_points), 300)
        self.assertEqual(len(preloaded.points_raw), n)
        self.assertEqual(preloaded.scans[0].point_count, n)
        self.assertFalse(preloaded.is_downsampled)

        # そのまま保存した際、1,200点すべてが1ビットの劣化もなく完全一致で出力されること
        mask = np.ones(n, dtype=bool)
        saver = FileSaveWorker(out_path, preloaded.scans, mask)
        saver.run()
        reloaded = read_e57_scans(out_path)[0]
        self.assertEqual(reloaded.point_count, n)
        np.testing.assert_array_equal(reloaded.points, pts_orig)


if __name__ == "__main__":
    unittest.main()
