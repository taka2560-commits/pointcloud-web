# -*- coding: utf-8 -*-
"""
点群を描く間の OpenGL の状態（gui/gl_state.py）のテスト

きっかけ（2026-11-15）: 丸い点が遅い理由を調べて分かった。paintGL の最後に QPainter で文字や枠を描くと、
QPainter が OpenGL の状態を「ブレンド ON・奥行きの判定 OFF」にして返す。2 回目からの描画では、点群が
  - 手前・奥に関係なく、配列であとにある点が上に描かれ
  - 丸い点の縁の色が下の色へ足されて、点が重なる所が白く飛んでいた（全点表示の 5,490 万点で、建物がほぼ白一色）

確かめること
- 何回描いても、手前の点が奥の点を隠す（配列の順番に関係しない。丸・四角とも）
- 丸い点を重ねても、点の色より明るい画素ができない
- 丸い点でも、取り込んだ絵・スクリーンショットに半透明の画素が混ざらない
- 点群のあとに描く部品（断面の枠など）へ渡る状態は、今までと同じ（begin の前へ戻す）
- begin / end が、状態を決まった形にして、元へ戻す
"""
import inspect
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.settings_sandbox import use_sandbox_settings  # noqa: E402
from tests.qt_cleanup import destroy_widget  # noqa: E402

use_sandbox_settings()  # この PC の本番アプリの保存設定を、テストで書き換えない

import numpy as np  # noqa: E402
from OpenGL import GL as gl  # noqa: E402
from PySide6.QtGui import QColor  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from gui import gl_state as GS  # noqa: E402
from gui.gl_viewer import PointCloudViewer  # noqa: E402

RED = (0.9, 0.05, 0.05)
BLUE = (0.05, 0.05, 0.9)


def plane(y: float, n: int = 160, half: float = 5.0) -> np.ndarray:
    """XZ 平面に平行な、点の並んだ板（正面から見ると、y が小さいほど手前）"""
    xs, zs = np.meshgrid(np.linspace(-half, half, n), np.linspace(-half, half, n))
    return np.stack([xs.ravel(), np.full(xs.size, y), zs.ravel()], axis=1).astype(np.float64)


class _Case(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def setUp(self):
        v = self.v = PointCloudViewer()
        v.resize(480, 360)
        v.show()
        self.app.processEvents()
        v.set_shading_enabled(False)
        v.set_point_budget(0)
        v.depth_fog_enabled = False
        v.adaptive_point_size = False
        v.show_grid = False
        v.bg_color = QColor(0, 0, 0)
        v._hide_hud_for_capture = True     # 軸の印・文字を描かない（点群の色だけを見る）

    def tearDown(self):
        self.v.set_point_cloud(None)
        destroy_widget(self.v)

    def front_view(self):
        v = self.v
        v.pitch, v.yaw = 0.0, 0.0          # 正面: 画面の右が X、上が Z、奥が Y
        v.is_orthographic = False

    def draw(self, times: int = 3):
        """QPainter が状態を変えたあとの描画を見るため、何回か描く"""
        v = self.v
        v._hide_hud_for_capture = False    # 1 回は、文字（QPainter）も描かせる
        v.repaint()
        self.app.processEvents()
        v._hide_hud_for_capture = True
        for _ in range(times):
            v.repaint()
            self.app.processEvents()

    def picture(self, img=None):
        img = self.v.grabFramebuffer() if img is None else img
        if img.isNull() or img.width() == 0:
            self.skipTest("OpenGL の画面を取り込めない環境")
        img = img.convertToFormat(img.Format.Format_RGBA8888)
        arr = np.frombuffer(img.constBits(), dtype=np.uint8).reshape(img.height(), img.bytesPerLine() // 4, 4)
        return arr[:, : img.width(), :].copy()

    @staticmethod
    def middle(pic):
        h, w = pic.shape[:2]
        return pic[int(h * 0.35): int(h * 0.65), int(w * 0.35): int(w * 0.65)]


class TestDepthOrder(_Case):
    def _two_planes(self, near_first: bool, shape: str):
        v = self.v
        near, far = plane(-2.0), plane(2.0)
        c_near = np.tile(np.array(RED, dtype=np.float32), (len(near), 1))
        c_far = np.tile(np.array(BLUE, dtype=np.float32), (len(far), 1))
        if near_first:
            pts, cols = np.vstack([near, far]), np.vstack([c_near, c_far])
        else:
            pts, cols = np.vstack([far, near]), np.vstack([c_far, c_near])
        v.set_point_cloud(pts, cols, None)
        v.point_shape = shape
        v.point_size = 6.0
        self.front_view()
        self.draw()
        mid = self.middle(self.picture()).astype(np.int16)
        red = int(np.count_nonzero((mid[..., 0] > 150) & (mid[..., 2] < 90)))
        blue = int(np.count_nonzero((mid[..., 2] > 150) & (mid[..., 0] < 90)))
        return red, blue, mid.shape[0] * mid.shape[1]

    def test_near_points_hide_far_points_in_any_array_order(self):
        for shape in ("circle", "square"):
            for near_first in (True, False):
                with self.subTest(shape=shape, near_first=near_first):
                    red, blue, total = self._two_planes(near_first, shape)
                    self.assertGreater(red, total * 0.95, "手前の板（赤）が、まん中を覆っている")
                    self.assertEqual(blue, 0, "奥の板（青）が、手前の板の上に出ていない")

    def test_points_stay_visible_when_depth_writes_were_left_off(self):
        """前の描画が「奥行きの書き込み OFF」を残していても、視点を変えたあとに点が消えない（RTX で起きた）"""
        v = self.v
        orig = v.paintGL

        def paint():
            gl.glDepthMask(gl.GL_FALSE)        # 描画の頭の glClear が、奥行きを消せない状態
            orig()

        v.paintGL = paint
        pts = plane(0.0)
        v.set_point_cloud(pts, np.tile(np.array(RED, dtype=np.float32), (len(pts), 1)), None)
        v.point_shape = "square"
        v.point_size = 6.0
        for pitch, yaw in ((30.0, 45.0), (0.0, 0.0), (60.0, -30.0), (0.0, 0.0)):
            v.pitch, v.yaw = pitch, yaw
            for _ in range(2):
                v.repaint()
                self.app.processEvents()
        mid = self.middle(self.picture()).astype(np.int16)
        red = int(np.count_nonzero((mid[..., 0] > 150) & (mid[..., 2] < 90)))
        self.assertGreater(red, mid.shape[0] * mid.shape[1] * 0.95, "視点を変えたあとも、板が見えている")

    def test_the_order_stays_right_after_many_frames(self):
        red, blue, total = self._two_planes(True, "circle")
        for _ in range(3):
            self.draw(times=2)
        mid = self.middle(self.picture()).astype(np.int16)
        self.assertEqual(int(np.count_nonzero((mid[..., 2] > 150) & (mid[..., 0] < 90))), 0)


class TestRoundPointsKeepTheirColor(_Case):
    def _dense_one_color(self, shape: str, size: float):
        v = self.v
        rng = np.random.default_rng(3)
        pts = rng.uniform(-5.0, 5.0, (150_000, 3))
        pts[:, 1] *= 0.2                                     # 薄い板にたくさん重ねる
        color = np.array([0.40, 0.30, 0.20], dtype=np.float32)
        v.set_point_cloud(pts, np.tile(color, (len(pts), 1)), None)
        v.point_shape = shape
        v.point_size = size
        self.front_view()
        self.draw()
        return self.picture(), (color * 255.0)

    def test_overlapping_round_points_do_not_get_brighter(self):
        for size in (1.3, 2.0, 6.0):
            with self.subTest(size=size):
                pic, color = self._dense_one_color("circle", size)
                rgb = pic[..., :3].astype(np.int16)
                self.assertGreater(int(np.count_nonzero(rgb[..., 0] > 60)), 1000, "前提: 点が見えている")
                for ch in range(3):
                    self.assertLessEqual(int(rgb[..., ch].max()), int(color[ch]) + 3,
                                         f"点の色より明るい画素がある（色が足されている）: 色 {ch}")

    def test_round_and_square_points_have_the_same_colors(self):
        circle, _ = self._dense_one_color("circle", 4.0)
        square, _ = self._dense_one_color("square", 4.0)
        mc, ms = self.middle(circle)[..., :3].astype(np.int16), self.middle(square)[..., :3].astype(np.int16)
        self.assertLessEqual(int(np.abs(mc - ms).max()), 3, "点で埋まったまん中は、丸でも四角でも同じ色")

    def test_no_translucent_pixels_in_grab_and_screenshot(self):
        pic, _ = self._dense_one_color("circle", 5.0)
        self.assertEqual(int(pic[..., 3].min()), 255, "取り込んだ絵に、半透明の画素が無い")
        shot = self.v.capture_screenshot(save_path=None, copy_to_clipboard=False)
        self.assertIsNotNone(shot)
        self.assertEqual(int(self.picture(shot)[..., 3].min()), 255, "スクリーンショットに、半透明の画素が無い")


class TestOtherPartsSeeTheSameState(_Case):
    def test_state_after_the_points_equals_state_before(self):
        v = self.v
        v.set_point_cloud(plane(0.0, n=60), None, None)
        seen = {}
        orig_box, orig_begin = v._draw_section_box, GS.begin_point_pass
        import gui.gl_viewer as GV

        def begin():
            seen["before"] = GS.read_point_pass_state()
            saved = orig_begin()
            seen["inside"] = GS.read_point_pass_state()
            return saved

        def box():
            seen["after"] = GS.read_point_pass_state()
            return orig_box()

        GV.begin_point_pass = begin
        v._draw_section_box = box
        try:
            self.draw()
        finally:
            GV.begin_point_pass = orig_begin
        self.assertEqual(seen["after"], seen["before"], "点群のあとの部品へ渡る状態は、点群の前と同じ")
        blend, depth, func, dmask, cmask = seen["inside"]
        self.assertFalse(blend)
        self.assertTrue(depth)
        self.assertEqual(func, int(gl.GL_LEQUAL))
        self.assertTrue(dmask)
        self.assertEqual(cmask, (True, True, True, False))

    def test_begin_and_end_restore_whatever_was_there(self):
        v = self.v
        v.makeCurrent()
        try:
            for blend, depth, func in ((True, False, gl.GL_LESS), (False, True, gl.GL_ALWAYS), (True, True, gl.GL_LEQUAL)):
                (gl.glEnable if blend else gl.glDisable)(gl.GL_BLEND)
                (gl.glEnable if depth else gl.glDisable)(gl.GL_DEPTH_TEST)
                gl.glDepthFunc(func)
                gl.glDepthMask(gl.GL_FALSE)
                gl.glColorMask(gl.GL_TRUE, gl.GL_FALSE, gl.GL_TRUE, gl.GL_TRUE)
                before = GS.read_point_pass_state()
                saved = GS.begin_point_pass()
                self.assertEqual(saved, before)
                self.assertEqual(GS.read_point_pass_state(), (False, True, int(gl.GL_LEQUAL), True, (True, True, True, False)))
                GS.end_point_pass(saved)
                self.assertEqual(GS.read_point_pass_state(), before)
            GS.end_point_pass(None)                                   # 読めなかったとき: 色の書き込みだけは全部に戻す
            self.assertEqual(GS.read_point_pass_state()[4], (True, True, True, True))
        finally:
            gl.glDepthMask(gl.GL_TRUE)
            gl.glColorMask(gl.GL_TRUE, gl.GL_TRUE, gl.GL_TRUE, gl.GL_TRUE)
            v.doneCurrent()


class TestWiring(unittest.TestCase):
    def test_paint_wraps_the_point_drawing(self):
        src = inspect.getsource(PointCloudViewer.paintGL)
        begin = src.index("begin_point_pass(")
        end = src.index("end_point_pass(")
        self.assertLess(begin, src.index("glDrawArrays("), "点群を描く前に、状態をそろえる")
        self.assertLess(src.index("_draw_section_overlay("), end, "重ね描きの点も、同じ状態で描く")
        self.assertLess(end, src.index("_draw_section_box("), "断面の枠の前に、状態を戻す")
        self.assertEqual(src.count("begin_point_pass("), 1)
        self.assertEqual(src.count("end_point_pass("), 1)

    def test_never_raises_without_a_context(self):
        calls = []
        names = ("glIsEnabled", "glGetIntegerv", "glGetBooleanv", "glEnable", "glDisable", "glDepthFunc", "glDepthMask", "glColorMask")
        keep = {n: getattr(GS.gl, n) for n in names}

        def boom(*a, **k):
            calls.append(1)
            raise RuntimeError("OpenGL が使えない")

        for n in names:
            setattr(GS.gl, n, boom)
        try:
            saved = GS.begin_point_pass()
            self.assertIsNone(saved)
            GS.end_point_pass(saved)
            GS.end_point_pass((True, False, 513, True, (True, True, True, True)))
        finally:
            for n, f in keep.items():
                setattr(GS.gl, n, f)
        self.assertGreater(len(calls), 0)


if __name__ == "__main__":
    unittest.main()
