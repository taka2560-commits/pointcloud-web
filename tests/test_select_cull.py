# -*- coding: utf-8 -*-
"""
選択の候補のふるい（gui/select_cull.py）のテスト

きっかけ（2026-11-14）: 矩形・円・多角形・ブラシで選ぶたびに、選べる点の全部について画面の位置を計算していた
（5,490 万点・全点表示で、ブラシでなぞるたびに約 2.1 秒。なぞった所が画面のごく一部でも同じ）。

選ばれる点
- ふるったときに選ばれる点が、ふるわないとき（全部の点から探す、以前の道）と 1 つも違わないこと
  （矩形・円・多角形・ブラシ / 透視・正射影 / 視点が点群の外・中 / 断面・レイヤーの表示・ロック・スキャンの表示）
- 形を囲む四角の中に映る、選べる点は、候補に必ず入っていること

ふるいの効き目と守り
- 近い点が近い番号に並んでいる点群では、候補が全体よりずっと少ないこと
- 点の並びがばらばらの点群・小さな点群では、ふるわずに、全部の点を候補にすること
- 選べる点が 1 つも無いときは何もしない・形の中に点が無いときは「0 点」として扱う、という以前の動きが同じこと
- 区画の箱は、点群が替わったら作り直すこと
"""
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.settings_sandbox import use_sandbox_settings  # noqa: E402
from tests.qt_cleanup import destroy_widget  # noqa: E402

use_sandbox_settings()  # この PC の本番アプリの保存設定を、テストで書き換えない

import numpy as np  # noqa: E402
from PySide6.QtCore import QPoint, Qt  # noqa: E402
from PySide6.QtGui import QVector3D  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from gui import select_cull as SC  # noqa: E402
from gui.gl_viewer import PointCloudViewer  # noqa: E402

V = PointCloudViewer


def ordered_cloud(n=150_000, seed=0, cell=4.0):
    """近い点が近い番号に並んだ点群（実データのファイルと同じ形）。升目ごとにまとめて並べる"""
    rng = np.random.default_rng(seed)
    pts = np.empty((n, 3))
    pts[:, 0] = rng.uniform(-20.0, 20.0, n)
    pts[:, 1] = rng.uniform(-20.0, 20.0, n)
    pts[:, 2] = rng.uniform(-3.0, 6.0, n)
    key = np.floor(pts[:, 0] / cell) * 1000 + np.floor(pts[:, 1] / cell)
    pts = pts[np.argsort(key, kind="stable")]
    scan_ids = (np.arange(n) * 4 // n).astype(np.int32)
    return pts, scan_ids


def old_candidates(v):
    """以前の道: 選べる点の全部を候補にする（1 つも無ければ None）"""
    cand = np.where(v._get_selectable_mask())[0]
    return cand if len(cand) else None


class _Case(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def setUp(self):
        self._saved = (SC.CULL_BLOCK, SC.CULL_MIN_POINTS)
        self.v = PointCloudViewer()
        self.v.resize(640, 480)
        self.v.show()
        self.app.processEvents()

    def tearDown(self):
        SC.CULL_BLOCK, SC.CULL_MIN_POINTS = self._saved
        self.v.set_point_cloud(None)
        destroy_widget(self.v)

    def load(self, n=150_000, seed=0, shuffle=False):
        pts, scan_ids = ordered_cloud(n, seed)
        if shuffle:
            pts = pts[np.random.default_rng(seed + 1).permutation(n)]
        self.v.set_point_budget(0)
        self.v.set_point_cloud(pts, None, None, scan_ids=scan_ids)
        self.app.processEvents()
        return pts

    def camera(self, yaw=45.0, pitch=30.0, distance=60.0, ortho=False, target=(0.0, 0.0, 0.0), fov=45.0):
        v = self.v
        v.yaw, v.pitch, v.camera_distance, v.is_orthographic, v.fov = yaw, pitch, distance, ortho, fov
        v.camera_target = QVector3D(*target)

    def run_shape(self, shape, culled=True):
        """形で選び、_apply_selection_with_modifiers に渡った番号を返す（呼ばれなければ None）。使った候補の数も返す"""
        v = self.v
        got = {"sel": None, "cand": None}
        orig_apply = v._apply_selection_with_modifiers
        orig_cand = type(v)._selection_candidates

        def apply(new_selected, *a, **k):
            got["sel"] = np.array(new_selected, dtype=np.int64)

        def candidates(*box):
            cand = orig_cand(v, *box) if culled else old_candidates(v)
            got["cand"] = None if cand is None else len(cand)
            return cand

        v._apply_selection_with_modifiers = apply
        v._selection_candidates = candidates
        try:
            kind = shape[0]
            if kind in ("rect", "circle"):
                v.selection_shape = V.SELECT_RECT if kind == "rect" else V.SELECT_CIRCLE
                v.selection_start, v.selection_end = QPoint(*shape[1]), QPoint(*shape[2])
                v._process_selection()
            elif kind == "polygon":
                v.polygon_points = [QPoint(*p) for p in shape[1]]
                v._process_polygon_selection()
                v.polygon_points = []
            else:
                v.brush_stroke_points = [QPoint(*p) for p in shape[1]]
                v.brush_radius = shape[2]
                v._process_brush_selection()
                v.brush_stroke_points = []
        finally:
            del v._apply_selection_with_modifiers
            del v._selection_candidates
        return got["sel"], got["cand"]

    def random_shape(self, rng):
        w, h = self.v.width(), self.v.height()
        kind = ("rect", "circle", "polygon", "brush")[int(rng.integers(0, 4))]
        cx, cy = int(rng.integers(-40, w + 40)), int(rng.integers(-40, h + 40))
        if kind == "rect":
            return ("rect", (cx, cy), (cx + int(rng.integers(-200, 200)), cy + int(rng.integers(-150, 150))))
        if kind == "circle":
            return ("circle", (cx, cy), (cx + int(rng.integers(-90, 90)), cy + int(rng.integers(-90, 90))))
        if kind == "polygon":
            k = int(rng.integers(3, 7))
            ang = np.sort(rng.uniform(0, 2 * math.pi, k))
            rad = rng.uniform(15, 140, k)
            return ("polygon", [(int(cx + r * math.cos(a)), int(cy + r * math.sin(a))) for a, r in zip(ang, rad)])
        k = int(rng.integers(1, 6))
        pts = [(cx, cy)]
        for _ in range(k - 1):
            pts.append((pts[-1][0] + int(rng.integers(-60, 60)), pts[-1][1] + int(rng.integers(-60, 60))))
        return ("brush", pts, int(rng.integers(2, 40)))

    def assert_same_selection(self, shape, where):
        new, n_cand = self.run_shape(shape, culled=True)
        old, _ = self.run_shape(shape, culled=False)
        if old is None:
            self.assertIsNone(new, f"以前は何もしなかった場面: {where} {shape}")
        else:
            self.assertIsNotNone(new, f"以前は選択を変えていた場面: {where} {shape}")
            np.testing.assert_array_equal(new, old, err_msg=f"選ばれる点が違う: {where} {shape}")
        return new, n_cand


CAMERAS = [
    ("透視・斜め上から", dict(yaw=45.0, pitch=30.0, distance=60.0)),
    ("透視・真上から", dict(yaw=10.0, pitch=89.9, distance=70.0)),
    ("透視・水平", dict(yaw=200.0, pitch=0.0, distance=45.0)),
    ("透視・視点が点群の中", dict(yaw=120.0, pitch=10.0, distance=3.0, target=(4.0, -6.0, 1.0))),
    ("透視・点のすぐそば", dict(yaw=300.0, pitch=45.0, distance=0.6, target=(-9.0, 9.0, 0.5))),
    ("透視・広い視野", dict(yaw=70.0, pitch=20.0, distance=25.0, fov=90.0)),
    ("正射影・斜め", dict(yaw=45.0, pitch=30.0, distance=60.0, ortho=True)),
    ("正射影・真上", dict(yaw=0.0, pitch=89.9, distance=40.0, ortho=True)),
    ("正射影・近く", dict(yaw=160.0, pitch=5.0, distance=6.0, ortho=True, target=(5.0, 5.0, 0.0))),
]


class TestSameSelection(_Case):
    def test_all_shapes_and_cameras_select_the_same_points(self):
        v = self.v
        SC.CULL_BLOCK, SC.CULL_MIN_POINTS = 2048, 4096              # 区画を細かくして、端の場面を増やす
        self.load(n=120_000, seed=1)
        rng = np.random.default_rng(100)
        some_selected = 0
        for name, cam in CAMERAS:
            self.camera(**cam)
            for k in range(25):
                shape = self.random_shape(rng)
                new, _ = self.assert_same_selection(shape, f"{name} の {k} 個目")
                some_selected += int(new is not None and len(new) > 0)
        self.assertGreater(some_selected, 80, "前提: 点が選ばれる場面が、十分にある")

    def test_with_sections_hidden_layers_locks_and_scans(self):
        v = self.v
        SC.CULL_BLOCK, SC.CULL_MIN_POINTS = 2048, 4096
        self.load(n=120_000, seed=2)
        rng = np.random.default_rng(101)
        lid_show = v.add_custom_layer("表示するレイヤー", color=(0.1, 0.9, 0.2))
        lid_lock = v.add_custom_layer("ロックするレイヤー", color=(0.9, 0.2, 0.9))
        v.move_indices_to_layer(rng.choice(120_000, 20_000, replace=False), lid_show)
        v.move_indices_to_layer(rng.choice(120_000, 20_000, replace=False), lid_lock)
        v.move_indices_to_noise(np.arange(30_000, 50_000, dtype=np.int64))
        steps = [
            ("ノイズを消したあと", lambda: None),
            ("レイヤーをロック", lambda: v.set_layer_locked(lid_lock, True)),
            ("スキャンを 1 つ隠す", lambda: v.set_scan_visibility(1, False)),
            ("Z の断面", lambda: v.set_section_z(True, -1.0, 2.5)),
            ("X の断面も", lambda: v.set_section_x(True, -12.0, 3.0)),
            ("ノイズも表示", lambda: v.set_layer_visibility(True, True)),
        ]
        selected = 0
        for label, change in steps:
            change()
            for name, cam in CAMERAS[::2]:
                self.camera(**cam)
                for k in range(12):
                    new, _ = self.assert_same_selection(self.random_shape(rng), f"{label} / {name} の {k} 個目")
                    selected += int(new is not None and len(new) > 0)
        self.assertGreater(selected, 60)

    def test_default_block_size(self):
        self.load(n=150_000, seed=3)
        rng = np.random.default_rng(102)
        for name, cam in CAMERAS:
            self.camera(**cam)
            for k in range(8):
                self.assert_same_selection(self.random_shape(rng), f"{name} の {k} 個目")

    def test_points_inside_the_box_are_always_candidates(self):
        v = self.v
        SC.CULL_BLOCK, SC.CULL_MIN_POINTS = 1024, 2048
        self.load(n=100_000, seed=4)
        rng = np.random.default_rng(103)
        w, h = v.width(), v.height()
        for name, cam in CAMERAS:
            self.camera(**cam)
            sx, sy, ok = v._project_to_screen(v.points_centered)
            for k in range(20):
                x0, y0 = float(rng.uniform(-30, w)), float(rng.uniform(-30, h))
                x1, y1 = x0 + float(rng.uniform(1, 220)), y0 + float(rng.uniform(1, 180))
                cand = v._selection_candidates(x0, x1, y0, y1)
                inside = np.flatnonzero(ok & (sx >= x0) & (sx <= x1) & (sy >= y0) & (sy <= y1))
                if cand is None:
                    self.fail("選べる点はあるので、None にはならない")
                self.assertTrue(np.all(cand[1:] > cand[:-1]), "候補は昇順")
                missing = np.setdiff1d(inside, cand)
                self.assertEqual(len(missing), 0, f"四角の中の点が、候補から落ちている: {name} の {k} 個目（{len(missing)} 点）")


class TestCullingEffect(_Case):
    def test_ordered_cloud_gets_far_fewer_candidates(self):
        v = self.v
        n = 200_000
        self.load(n=n, seed=5)                                       # 区画（16,384 点）は 13 個。X の向きに細い帯になる
        self.camera(yaw=0.0, pitch=89.9, distance=70.0)             # 真上から
        cx, cy = v.width() // 2, v.height() // 2
        sel, n_cand = self.assert_same_selection(("brush", [(cx - 20, cy), (cx + 20, cy)], 10), "まん中をなぞる")
        self.assertGreater(len(sel), 0)
        self.assertLess(n_cand, n // 2, "近くの区画の点だけが候補になる")
        sel, n_cand = self.assert_same_selection(("rect", (cx - 30, cy - 30), (cx + 30, cy + 30)), "まん中を囲む")
        self.assertLess(n_cand, n // 2)

    def test_shuffled_and_small_clouds_use_every_point(self):
        v = self.v
        n = 100_000
        self.load(n=n, seed=6, shuffle=True)                       # 点の並びがばらばら: どの区画の箱も、点群の全体になる
        self.camera(yaw=30.0, pitch=60.0, distance=70.0)
        cx, cy = v.width() // 2, v.height() // 2
        calls = []
        orig = v._get_selectable_mask
        v._get_selectable_mask = lambda idx=None: (calls.append(idx is None), orig(idx))[1]
        try:
            sel, n_cand = self.assert_same_selection(("rect", (cx - 30, cy - 30), (cx + 30, cy + 30)), "ばらばらの点群")
            self.assertEqual(n_cand, n, "ふるえないので、全部の点が候補")
            self.assertTrue(all(calls), "一部の点だけのマスクは作らない")
            self.load(n=3000, seed=7)                              # 小さな点群
            v._get_selectable_mask = lambda idx=None: (calls.append(idx is None), orig(idx))[1]
            called = []
            v._blocks_near_screen_box = lambda *a: called.append(a)
            sel, n_cand = self.assert_same_selection(("rect", (cx - 30, cy - 30), (cx + 30, cy + 30)), "小さな点群")
            self.assertEqual(n_cand, 3000)
            self.assertEqual(called, [], "小さな点群では、区画の箱を使わない")
        finally:
            v.__dict__.pop("_get_selectable_mask", None)
            v.__dict__.pop("_blocks_near_screen_box", None)

    def test_nothing_selectable_and_empty_region_behave_as_before(self):
        v = self.v
        self.load(n=100_000, seed=8)
        self.camera(yaw=30.0, pitch=60.0, distance=70.0)
        cx, cy = v.width() // 2, v.height() // 2
        # 画面の端の、点の無い所: 候補は空だが、選べる点はある → 「0 点」として扱う（選択が外れる・即削除なら案内が出る）
        self.camera(yaw=30.0, pitch=60.0, distance=400.0)
        sel, n_cand = self.assert_same_selection(("rect", (2, 2), (30, 30)), "点の無い所")
        self.assertIsNotNone(sel)
        self.assertEqual(len(sel), 0)
        self.assertEqual(n_cand, 0)
        # 選べる点が 1 つも無い → 何もしない
        v.set_layer_locked(V.LAYER_NORMAL, True)
        for shape in (("rect", (cx - 30, cy - 30), (cx + 30, cy + 30)), ("rect", (2, 2), (30, 30)),
                      ("brush", [(cx, cy)], 12), ("polygon", [(cx - 40, cy - 40), (cx + 40, cy - 40), (cx, cy + 40)])):
            sel, n_cand = self.assert_same_selection(shape, "全部ロック")
            self.assertIsNone(sel)
            self.assertIsNone(n_cand)

    def test_selection_through_the_real_handlers(self):
        # 差し替えなしで通す: 選ぶ → 追加（Shift）→ 即削除
        v = self.v
        self.load(n=120_000, seed=9)
        self.camera(yaw=30.0, pitch=60.0, distance=70.0)
        cx, cy = v.width() // 2, v.height() // 2
        sx, sy, ok = v._project_to_screen(v.points_centered)
        v.selection_shape = V.SELECT_RECT
        v.selection_start, v.selection_end = QPoint(cx - 25, cy - 25), QPoint(cx + 25, cy + 25)
        v._process_selection()
        expect = np.flatnonzero(ok & (sx >= cx - 25) & (sx <= cx + 25) & (sy >= cy - 25) & (sy <= cy + 25))
        np.testing.assert_array_equal(np.sort(v.selected_indices), expect)
        self.assertGreater(len(expect), 0)
        v._last_selection_modifiers = Qt.ShiftModifier
        v.selection_start, v.selection_end = QPoint(cx + 40, cy - 25), QPoint(cx + 70, cy + 25)
        v._process_selection()
        more = np.flatnonzero(ok & (sx >= cx + 40) & (sx <= cx + 70) & (sy >= cy - 25) & (sy <= cy + 25))
        np.testing.assert_array_equal(np.sort(v.selected_indices), np.union1d(expect, more))
        v._last_selection_modifiers = Qt.NoModifier
        erased = []
        v.instant_erase_requested.connect(lambda idx: erased.append(np.array(idx)))
        v.instant_delete_mode = True
        v.brush_stroke_points = [QPoint(cx - 60, cy + 60), QPoint(cx - 20, cy + 60)]
        v.brush_radius = 9
        v._process_brush_selection()
        v.instant_delete_mode = False
        self.assertEqual(len(erased), 1)
        d2 = np.minimum((sx - (cx - 60)) ** 2, (sx - (cx - 20)) ** 2) + (sy - (cy + 60)) ** 2
        on_line = (sx >= cx - 60) & (sx <= cx - 20) & (np.abs(sy - (cy + 60)) <= 9)
        np.testing.assert_array_equal(np.sort(erased[0]), np.flatnonzero(ok & ((d2 <= 81) | on_line)))


class TestBlockBoxes(_Case):
    def test_boxes_enclose_their_points_and_follow_the_point_cloud(self):
        v = self.v
        SC.CULL_BLOCK = 1000
        self.load(n=10_500, seed=10)                                # 最後の区画は、半端な長さ（500 点）
        lo, hi = v._block_boxes()
        self.assertEqual(lo.shape, (11, 3))
        pc = v.points_centered.astype(np.float64)
        for b in range(11):
            part = pc[b * 1000:(b + 1) * 1000]
            np.testing.assert_array_equal(lo[b], part.min(axis=0))
            np.testing.assert_array_equal(hi[b], part.max(axis=0))
        self.assertIs(v._block_boxes()[0], lo, "同じ点群なら、作ってあるものを使う")
        self.load(n=10_500, seed=11)                                # 同じ点数の、別の点群
        lo2, _ = v._block_boxes()
        self.assertIsNot(lo2, lo)
        np.testing.assert_array_equal(lo2[3], v.points_centered[3000:4000].astype(np.float64).min(axis=0))
        self.assertIs(v._cull_boxes[0](), v.points_centered, "箱が、点群の配列を生かし続けない（弱い参照）")

    def test_mvp_matrix_is_the_one_used_for_projection(self):
        v = self.v
        self.load(n=5000, seed=12)
        for name, cam in CAMERAS:
            self.camera(**cam)
            mvp = v._mvp_matrix()
            self.assertEqual(mvp.dtype, np.float32)
            pts = v.points_centered[:200]
            clip = np.hstack([pts, np.ones((200, 1), dtype=np.float32)]) @ mvp
            sx, sy, ok = v._project_to_screen(pts)
            w = clip[:, 3]
            np.testing.assert_array_equal(ok, w > 0.001, err_msg=name)
            np.testing.assert_allclose(sx[ok], (clip[ok, 0] / w[ok] + 1.0) * 0.5 * v.width(), rtol=1e-5, atol=1e-3, err_msg=name)
            np.testing.assert_allclose(sy[ok], (1.0 - clip[ok, 1] / w[ok]) * 0.5 * v.height(), rtol=1e-5, atol=1e-3, err_msg=name)


if __name__ == "__main__":
    unittest.main()
