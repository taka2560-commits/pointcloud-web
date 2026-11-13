"""
読み込みに要るメモリの見積もり（core/memguard.py）のテスト

- 式が読み込みの実装とずれていないこと: 合成の E57 を本物の FileLoadWorker で読み、できた配列の大きさの合計が
  見積もりとぴったり合うこと（項目の並び 3 通り × 全点 / 間引き × 描画点数、フォルダ読み込み）
- 間引きで読み込む点数が、実際の読み込みと一致すること
- 載っているメモリに対する判定と、おすすめの上限（利用者の実データと同じ並び・点数で確かめる）
- 読みやすい区切りへの切り下げ、フォルダのヘッダ読みの時間切れと推定
"""

import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
from pye57 import libe57

from core.e57_io import e57_scan_layouts, subsample_step, subsampled_count
from core.memguard import (
    AUTO_TIER_SMALL, DRAW_BYTES, SAFETY_FACTOR, FileCost, ScanCost, auto_load_limit, check_load, fit_max_points,
    memory_limit_bytes, nice_floor_points, plan_load, read_file_cost, read_file_costs, scan_bytes_per_point,
)
from core.sysinfo import GB
from gui.worker_thread import FileLoadWorker, collect_e57_files

LAYOUT_SCALED_NORMALS = "scaled_normals"  # 利用者の実データと同じ並び: ScaledInteger の XYZ + 反射強度 + 色 + 法線
LAYOUT_FLS = "fls"                        # スキャナの書き出し: 単精度の XYZ + 反射強度 + 色 + 行・列・無効フラグ
LAYOUT_DOUBLE = "double"                  # 倍精度の XYZ だけ
NOR_URI = "http://www.libe57.org/E57_NOR_surface_normals.txt"
POSE_TILT = ((0.9238795325, 0.0, 0.0, 0.3826834324), (10.0, -20.0, 1.5))  # Z まわり 45 度 + 平行移動


def write_synthetic_e57(path: str, layout: str, counts, pose=None, seed: int = 0) -> None:
    """合成の E57 を書く（counts: スキャンごとの点数。pose: (回転 wxyz, 平行移動) か None）"""
    rng = np.random.default_rng(seed)
    imf = libe57.ImageFile(path, "w")
    try:
        imf.extensionsAdd("", libe57.E57_V1_0_URI)
        if layout == LAYOUT_SCALED_NORMALS:
            imf.extensionsAdd("nor", NOR_URI)
        root = imf.root()
        root.set("formatName", libe57.StringNode(imf, "ASTM E57 3D Imaging Data File"))
        root.set("guid", libe57.StringNode(imf, "{00000000-0000-0000-0000-00000000aaaa}"))
        data3d = libe57.VectorNode(imf, True)
        root.set("data3D", data3d)
        for i, n in enumerate(counts):
            n = int(n)
            scan = libe57.StructureNode(imf)
            scan.set("guid", libe57.StringNode(imf, "{00000000-0000-0000-0000-%012d}" % i))
            scan.set("name", libe57.StringNode(imf, f"Scan {i}"))
            if pose is not None:
                rot, tr = pose
                pn = libe57.StructureNode(imf)
                rn = libe57.StructureNode(imf)
                for k, v in zip("wxyz", rot):
                    rn.set(k, libe57.FloatNode(imf, float(v), libe57.E57_DOUBLE, -1.0, 1.0))
                pn.set("rotation", rn)
                tn = libe57.StructureNode(imf)
                for k, v in zip("xyz", tr):
                    tn.set(k, libe57.FloatNode(imf, float(v), libe57.E57_DOUBLE, -1e6, 1e6))
                pn.set("translation", tn)
                scan.set("pose", pn)

            proto = libe57.StructureNode(imf)
            cols = []  # (名前, 配列, 物理値へ換算するか)
            xyz = rng.uniform(-20.0, 20.0, size=(n, 3))
            if layout == LAYOUT_SCALED_NORMALS:
                for a, ax in enumerate("XYZ"):
                    proto.set(f"cartesian{ax}", libe57.ScaledIntegerNode(imf, 0, -2147483647, 2147483647, 1e-5, 0.0))
                    cols.append((f"cartesian{ax}", np.round(xyz[:, a] / 1e-5).astype(np.int64), False))
            elif layout == LAYOUT_FLS:
                for a, ax in enumerate("XYZ"):
                    proto.set(f"cartesian{ax}", libe57.FloatNode(imf, 0.0, libe57.E57_SINGLE, -1e6, 1e6))
                    cols.append((f"cartesian{ax}", xyz[:, a].astype(np.float32), True))
            else:
                for a, ax in enumerate("XYZ"):
                    proto.set(f"cartesian{ax}", libe57.FloatNode(imf, 0.0, libe57.E57_DOUBLE, -1e9, 1e9))
                    cols.append((f"cartesian{ax}", np.ascontiguousarray(xyz[:, a]), True))
            if layout in (LAYOUT_SCALED_NORMALS, LAYOUT_FLS):
                proto.set("intensity", libe57.FloatNode(imf, 0.0, libe57.E57_SINGLE, 0.0, 1.0))
                cols.append(("intensity", rng.random(n).astype(np.float32), True))
                for c in ("colorRed", "colorGreen", "colorBlue"):
                    proto.set(c, libe57.IntegerNode(imf, 0, 0, 255))
                    cols.append((c, rng.integers(0, 256, size=n).astype(np.uint8), True))
            if layout == LAYOUT_SCALED_NORMALS:
                for ax in "XYZ":
                    proto.set(f"nor:normal{ax}", libe57.FloatNode(imf, 0.0, libe57.E57_SINGLE, -1.0, 1.0))
                    cols.append((f"nor:normal{ax}", rng.uniform(-1.0, 1.0, size=n).astype(np.float32), True))
            if layout == LAYOUT_FLS:
                proto.set("rowIndex", libe57.IntegerNode(imf, 0, 0, 852))
                cols.append(("rowIndex", rng.integers(0, 853, size=n).astype(np.uint16), True))
                proto.set("columnIndex", libe57.IntegerNode(imf, 0, 0, 2029))
                cols.append(("columnIndex", rng.integers(0, 2030, size=n).astype(np.uint16), True))
                proto.set("cartesianInvalidState", libe57.IntegerNode(imf, 0, 0, 2))
                cols.append(("cartesianInvalidState", np.zeros(n, dtype=np.uint8), True))

            cv = libe57.CompressedVectorNode(imf, proto, libe57.VectorNode(imf, True))
            scan.set("points", cv)
            data3d.append(scan)
            if n > 0:
                bufs = libe57.VectorSourceDestBuffer()
                for name, arr, scaled in cols:
                    bufs.append(libe57.SourceDestBuffer(imf, name, arr, n, True, scaled))
                w = cv.writer(bufs)
                w.write(n)
                w.close()
    finally:
        imf.close()


def load_with_worker(path: str, max_load_points, display_budget: int, coord_mode: str = "cartesian"):
    """本物の FileLoadWorker を、このスレッドでそのまま走らせて結果を受け取る"""
    worker = FileLoadWorker(path, point_budget=display_budget, max_load_points=max_load_points, coord_mode=coord_mode)
    got, errors = [], []
    worker.finished_success.connect(got.append)
    worker.finished_error.connect(errors.append)
    worker.run()
    if errors or not got:
        raise AssertionError(f"読み込みに失敗: {errors}")
    return got[0]


def owned_array_bytes(arrays) -> int:
    """配列が実際に持っているメモリの合計（同じ領域を見ている配列は 1 回だけ数える）"""
    seen = {}
    for a in arrays:
        if not isinstance(a, np.ndarray):
            continue
        root = a
        while isinstance(root.base, np.ndarray):
            root = root.base
        seen[id(root)] = int(root.nbytes)
    return sum(seen.values())


def _arrays_in(value, out, depth=0):
    """value の中にある配列を、辞書・リスト・オブジェクトの属性をたどって全部集める"""
    if isinstance(value, np.ndarray):
        if value.size >= 100:
            out.append(value)
    elif depth > 4:
        return
    elif isinstance(value, dict):
        for v in value.values():
            _arrays_in(v, out, depth + 1)
    elif isinstance(value, (list, tuple)):
        for v in value:
            _arrays_in(v, out, depth + 1)
    elif hasattr(value, "__dict__") and not isinstance(value, type):
        for v in vars(value).values():
            _arrays_in(v, out, depth + 1)


def preloaded_arrays(pre):
    """
    読み込み結果が持っている配列を全部集める。決まった名前だけを数えるのではなく、属性をたどって集める
    （スキャンが「読まれない属性」で古い座標の配列を持ち続けていたのを、名前を決め打ちした数え方では見逃した）。
    """
    out = []
    _arrays_in(pre, out)
    return out


class _TempDirCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="e57_memguard_")
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def path(self, name: str) -> str:
        return os.path.join(self.tmp, name)


class TestFormulaMatchesLoader(_TempDirCase):
    """見積もりの式と、本物の読み込みが作る配列の大きさ"""

    CASES = (
        # (名前, 並び, スキャンごとの点数, pose)
        ("実データと同じ並び・1 スキャン・pose あり", LAYOUT_SCALED_NORMALS, (12000,), POSE_TILT),
        ("実データと同じ並び・1 スキャン・pose なし", LAYOUT_SCALED_NORMALS, (12000,), None),
        ("スキャナの書き出し・4 スキャン・pose あり", LAYOUT_FLS, (4000, 3000, 2500, 2500), POSE_TILT),
        ("倍精度の XYZ だけ・1 スキャン・pose なし", LAYOUT_DOUBLE, (12000,), None),
        ("倍精度の XYZ だけ・2 スキャン", LAYOUT_DOUBLE, (7000, 5000), None),
    )
    SETTINGS = (
        # (読み込み上限, 描画点数)
        (None, 0),        # 全点・全点表示
        (None, 3000),     # 全点・描画は間引く
        (5000, 0),        # 間引いて読む・全点表示
        (5000, 2000),     # 間引いて読む・描画も間引く
        (9000, 0),        # 上限の 1.5 倍に届かないので、全点を読む
    )

    def _check(self, path: str, max_load, display_budget: int, label: str, coord_mode: str = "cartesian"):
        files = [read_file_cost(p) for p in collect_e57_files(path)]
        plan = plan_load(files, max_load, display_budget)
        pre = load_with_worker(path, max_load, display_budget, coord_mode)
        n = len(pre.points_raw)
        self.assertEqual(plan.n_loaded, n, f"{label}: 読み込む点数")
        self.assertEqual(plan.n_drawn, len(pre.display_points), f"{label}: 描く点数")
        self.assertEqual(plan.subsampled, bool(pre.is_downsampled), f"{label}: 間引いたか")
        # 読み込みの直後は、全点表示なら描画用の配列を共有していて、点の番号（8）だけを別に持つ。
        # 描画を間引くときは、描く点の番号（8）と写し（24）を持つ
        draw_now = 8 * n if plan.n_drawn == n else 32 * plan.n_drawn
        self.assertEqual(owned_array_bytes(preloaded_arrays(pre)), plan.base_bytes + draw_now, f"{label}: 配列の大きさの合計")
        # 見込みは、読み込み中の山（中心を引く一時配列 24）と、読み込み後に使う量（レイヤー 5 + 描画 56）の大きいほうに余裕を掛けたもの
        expect = max(plan.base_bytes + 24 * n, plan.base_bytes + 5 * n + DRAW_BYTES * plan.n_drawn)
        self.assertEqual(plan.peak_bytes, plan.base_bytes + 24 * n, label)
        self.assertEqual(plan.need_bytes, int(expect * SAFETY_FACTOR), label)

    def test_single_file_layouts(self):
        for idx, (name, layout, counts, pose) in enumerate(self.CASES):
            p = self.path(f"case{idx}.e57")
            write_synthetic_e57(p, layout, counts, pose=pose, seed=idx)
            for max_load, budget in self.SETTINGS:
                with self.subTest(case=name, max_load=max_load, budget=budget):
                    self._check(p, max_load, budget, f"{name} / 上限 {max_load} / 描画 {budget}")

    def test_coordinate_modes_do_not_keep_extra_arrays(self):
        """座標系を変えて読み込んでも、見積もりどおりの量になる（以前は、古い座標の配列 1 点 24 バイトを持ち続けていた）"""
        for idx, (name, layout, counts, pose) in enumerate(self.CASES):
            p = self.path(f"coord{idx}.e57")
            write_synthetic_e57(p, layout, counts, pose=pose, seed=20 + idx)
            for coord_mode in ("survey_swap", "survey_swap_inv", "y_up_to_z_up", "z_invert"):
                for max_load, budget in ((None, 0), (5000, 2000)):
                    with self.subTest(case=name, coord_mode=coord_mode, max_load=max_load):
                        self._check(p, max_load, budget, f"{name} / 座標系 {coord_mode} / 上限 {max_load}", coord_mode)

    def test_folder_load_splits_the_limit_per_file(self):
        folder = self.path("folder")
        os.makedirs(folder)
        for i, n in enumerate((6000, 1000, 5000)):
            write_synthetic_e57(os.path.join(folder, f"s{i}.e57"), LAYOUT_FLS, (n,), pose=POSE_TILT, seed=10 + i)
        for max_load, budget in ((None, 0), (6000, 0), (6000, 1500), (3000, 0)):
            with self.subTest(max_load=max_load, budget=budget):
                self._check(folder, max_load, budget, f"フォルダ / 上限 {max_load} / 描画 {budget}")
        # 上限 6,000 を 3 ファイルで割ると 2,000 ずつ。6,000 点と 5,000 点のファイルは間引き、1,000 点のファイルは全点
        files = [read_file_cost(p) for p in collect_e57_files(folder)]
        self.assertEqual(plan_load(files, 6000, 0).n_loaded, 2000 + 1000 + 1667)

    def test_scan_without_points_is_counted_as_zero(self):
        p = self.path("empty_scan.e57")
        write_synthetic_e57(p, LAYOUT_FLS, (0, 3000), pose=None, seed=3)
        self.assertEqual([lay.n_points for lay in e57_scan_layouts(p)], [0, 3000])
        self._check(p, None, 0, "点の無いスキャンを含む")


class TestBytesPerPoint(_TempDirCase):
    def _bpp(self, layout: str) -> int:
        p = self.path(f"{layout}.e57")
        write_synthetic_e57(p, layout, (10,), pose=None)
        return scan_bytes_per_point(e57_scan_layouts(p)[0].schemas)

    def test_known_layouts(self):
        # ScaledInteger の XYZ は生の整数 8×3、反射強度 4、色 1×3、法線 4×3、倍精度の座標 24
        self.assertEqual(self._bpp(LAYOUT_SCALED_NORMALS), 24 + 4 + 3 + 12 + 24)
        # 単精度の XYZ 4×3、反射強度 4、色 3、行・列 2×2、無効フラグ 1、倍精度の座標 24
        self.assertEqual(self._bpp(LAYOUT_FLS), 12 + 4 + 3 + 4 + 1 + 24)
        # 倍精度の XYZ は座標の列を見るだけ
        self.assertEqual(self._bpp(LAYOUT_DOUBLE), 24)

    def test_file_cost_reads_pose_and_fields(self):
        p = self.path("pose.e57")
        write_synthetic_e57(p, LAYOUT_SCALED_NORMALS, (10, 20), pose=POSE_TILT)
        cost = read_file_cost(p)
        self.assertEqual(cost.n_points, 30)
        self.assertEqual([s.pose_identity for s in cost.scans], [False, False])
        self.assertTrue(all(s.has_color and s.has_intensity for s in cost.scans))
        q = self.path("plain.e57")
        write_synthetic_e57(q, LAYOUT_DOUBLE, (10,), pose=None)
        cost = read_file_cost(q)
        self.assertEqual([(s.pose_identity, s.has_color, s.has_intensity) for s in cost.scans], [(True, False, False)])


def user_file_cost() -> FileCost:
    """利用者の実データ（16.8GB・1 スキャン）と同じ点数・並び"""
    return FileCost("keisathu.e57", (ScanCost(538_377_435, 67, True, True, False),))


class TestLimitsAndCheck(unittest.TestCase):
    PC_32GB = (int(31.1 * GB), int(15.0 * GB))
    PC_16GB = (int(15.9 * GB), int(6.0 * GB))
    PC_8GB = (int(7.9 * GB), int(2.5 * GB))

    def test_auto_load_limit_keeps_the_old_tiers(self):
        self.assertEqual(auto_load_limit(int(63.9 * GB)), 120_000_000)
        self.assertEqual(auto_load_limit(int(31.1 * GB)), 60_000_000)
        self.assertEqual(auto_load_limit(int(15.9 * GB)), AUTO_TIER_SMALL)
        self.assertEqual(auto_load_limit(None), AUTO_TIER_SMALL)

    def test_memory_limit(self):
        self.assertEqual(memory_limit_bytes(30 * GB), 20 * GB)   # 3 分の 2
        self.assertEqual(memory_limit_bytes(9 * GB), 5 * GB)     # 少なくとも 4GB は残す
        self.assertEqual(memory_limit_bytes(8 * GB), 4 * GB)
        self.assertEqual(memory_limit_bytes(4 * GB), 2 * GB)     # それでも半分は使える
        self.assertEqual(memory_limit_bytes(0), 0)

    def test_user_file_unlimited_does_not_fit_32gb(self):
        files = [user_file_cost()]
        chk = check_load(files, None, 0, memory=self.PC_32GB)
        self.assertFalse(chk.ok)
        self.assertEqual(chk.plan.n_loaded, 538_377_435)
        self.assertGreater(chk.plan.need_bytes, 95 * GB)   # 約 100GB
        self.assertLess(chk.plan.need_bytes, 110 * GB)
        self.assertEqual(chk.avail_bytes, self.PC_32GB[1])
        # おすすめは「自動最適化」と同じ 6,000 万点（9 点に 1 点で 5,982 万点）
        self.assertEqual(chk.recommended_points, 60_000_000)
        self.assertEqual(chk.recommended_plan.n_loaded, 59_819_715)
        self.assertEqual(chk.recommended_plan.max_step, 9)
        self.assertLessEqual(chk.recommended_plan.need_bytes, chk.limit_bytes)

    def test_user_file_usual_settings_fit_32gb(self):
        files = [user_file_cost()]
        for max_load in (60_000_000, 100_000_000, 30_000_000):
            for budget in (0, 2_000_000):
                with self.subTest(max_load=max_load, budget=budget):
                    chk = check_load(files, max_load, budget, memory=self.PC_32GB)
                    self.assertTrue(chk.ok)
                    self.assertIsNone(chk.recommended_points)
        # いつもの設定（自動最適化・全点表示）は約 11GB
        chk = check_load(files, 60_000_000, 0, memory=self.PC_32GB)
        self.assertGreater(chk.plan.need_bytes, 10 * GB)
        self.assertLess(chk.plan.need_bytes, 12.5 * GB)

    def test_smaller_pcs_get_a_smaller_recommendation(self):
        files = [user_file_cost()]
        chk = check_load(files, 60_000_000, 0, memory=self.PC_16GB)
        self.assertFalse(chk.ok)
        self.assertEqual(chk.recommended_points, AUTO_TIER_SMALL)
        self.assertLessEqual(chk.recommended_plan.need_bytes, chk.limit_bytes)
        # 8GB の PC では「自動最適化」の 3,000 万点も収まらないので、収まる上限まで下げる
        chk = check_load(files, AUTO_TIER_SMALL, 0, memory=self.PC_8GB)
        self.assertFalse(chk.ok)
        self.assertLess(chk.recommended_points, AUTO_TIER_SMALL)
        self.assertGreaterEqual(chk.recommended_points, 15_000_000)
        self.assertLessEqual(chk.recommended_plan.need_bytes, chk.limit_bytes)
        self.assertEqual(chk.recommended_points % 1_000_000, 0)  # 読みやすい区切り

    def test_small_file_always_fits(self):
        files = [FileCost("small.e57", (ScanCost(8200, 43, True, True, False),))]
        for max_load in (None, 0, 30_000_000):
            chk = check_load(files, max_load, 0, memory=self.PC_8GB)
            self.assertTrue(chk.ok)
            self.assertEqual(chk.plan.n_loaded, 8200)
            self.assertFalse(chk.plan.subsampled)

    def test_unknown_memory_does_not_block(self):
        self.assertIsNone(check_load([user_file_cost()], None, 0, memory=(0, 0)))

    def test_no_points(self):
        chk = check_load([FileCost("e.e57", ())], None, 0, memory=self.PC_8GB)
        self.assertTrue(chk.ok)
        self.assertEqual((chk.plan.n_loaded, chk.plan.need_bytes), (0, 0))

    def test_fit_max_points_is_the_largest_that_fits(self):
        files = [user_file_cost()]
        limit = memory_limit_bytes(self.PC_32GB[0])
        fit = fit_max_points(files, 0, limit)
        self.assertLessEqual(plan_load(files, fit, 0).need_bytes, limit)
        # 1 段細かい間引き（点数が増える側）は収まらない
        step = plan_load(files, fit, 0).max_step
        self.assertGreater(step, 1)
        finer = subsampled_count(538_377_435, step - 1)
        self.assertGreater(plan_load(files, finer, 0).need_bytes, limit)
        # 全点が収まるなら、点数そのもの
        self.assertEqual(fit_max_points([FileCost("s.e57", (ScanCost(5000, 43, True, True, True),))], 0, limit), 5000)

    def test_nice_floor_points(self):
        self.assertEqual(nice_floor_points(114_560_000), 110_000_000)
        self.assertEqual(nice_floor_points(57_399_999), 57_000_000)
        self.assertEqual(nice_floor_points(8_439_000), 8_400_000)
        self.assertEqual(nice_floor_points(999_999), 999_999)
        self.assertEqual(nice_floor_points(-5), 0)

    def test_step_rule_is_shared_with_the_loader(self):
        self.assertEqual(subsample_step(538_377_435, 60_000_000), 9)
        self.assertEqual(subsample_step(89_000_000, 60_000_000), 1)   # 1.5 倍までは全点
        self.assertEqual(subsample_step(90_000_001, 60_000_000), 2)
        self.assertEqual(subsample_step(100, None), 1)
        self.assertEqual(subsample_step(100, 0), 1)
        self.assertEqual(subsampled_count(10, 3), 4)
        self.assertEqual(subsampled_count(9, 3), 3)
        self.assertEqual(subsampled_count(0, 3), 0)


class TestHeaderTimeLimit(_TempDirCase):
    def test_rest_is_estimated_from_file_size(self):
        paths = []
        for i in range(4):
            p = self.path(f"f{i}.e57")
            with open(p, "wb") as f:
                f.write(b"\0" * (1000 * (i + 1)))
            paths.append(p)
        ticks = iter([0.0, 0.5, 2.5, 9.0, 9.0, 9.0])
        read = []

        def reader(p):
            read.append(p)
            # 1,000 バイトあたり 100 点
            return FileCost(p, (ScanCost(os.path.getsize(p) // 10, 48, True, True, False),))

        costs = read_file_costs(paths, time_limit=2.0, clock=lambda: next(ticks), reader=reader)
        self.assertEqual(read, paths[:2])                      # 2 つ読んだところで時間切れ
        self.assertEqual([c.estimated for c in costs], [False, False, True, True])
        self.assertEqual([c.n_points for c in costs], [100, 200, 300, 400])
        self.assertEqual(costs[3].scans[0].bytes_per_point, 48)
        self.assertTrue(plan_load(costs, None, 0).estimated)

    def test_rest_is_read_when_nothing_can_be_estimated(self):
        """読めた分がどれも 0 点だと、大きさからの推定ができない。残りを見積もりから落とさず、時間が切れていても読む"""
        paths = []
        for i in range(3):
            p = self.path(f"z{i}.e57")
            with open(p, "wb") as f:
                f.write(b"\0" * 1000)
            paths.append(p)
        ticks = iter([0.0, 5.0, 9.0, 9.0, 9.0])
        read = []

        def reader(p):
            read.append(p)
            n = 0 if p == paths[0] else 700
            return FileCost(p, (ScanCost(n, 48, True, True, False),))

        costs = read_file_costs(paths, time_limit=2.0, clock=lambda: next(ticks), reader=reader)
        self.assertEqual(read, paths)
        self.assertEqual([c.n_points for c in costs], [0, 700, 700])
        self.assertEqual([c.estimated for c in costs], [False, False, False])

    def test_all_read_within_time(self):
        p = self.path("one.e57")
        write_synthetic_e57(p, LAYOUT_FLS, (500,), pose=None)
        costs = read_file_costs([p, p])
        self.assertEqual([c.estimated for c in costs], [False, False])
        self.assertFalse(plan_load(costs, None, 0).estimated)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
