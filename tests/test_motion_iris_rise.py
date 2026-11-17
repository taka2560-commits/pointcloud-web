# -*- coding: utf-8 -*-
"""
2026-11-17 に足した動き（gui/motion.py）: アイリスオープン（テーマの切り替え）・ソフトライズ（起動した所のカードの頭）

利用者の依頼「（モーションリファレンスから）面白そうなモーションも採用したい」。
- どちらも 1 回きりで短い。終わったら、重ねた物・描画の効果・ずらした位置が残らない
- 動きを OFF にすると、何も重ねず、すぐに切り替わる
- 操作を妨げない（重ねた物は、マウスを通す）
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.settings_sandbox import use_sandbox_settings  # noqa: E402

use_sandbox_settings()  # この PC の本番アプリの保存設定を、テストで書き換えない

import shiboken6  # noqa: E402
from PySide6.QtCore import QCoreApplication, QEvent, QPoint, Qt  # noqa: E402
from PySide6.QtGui import QColor, QPixmap  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget  # noqa: E402

from gui.motion import MOTION, iris_reveal, set_motion_enabled, staged_rise  # noqa: E402


def _host(color="#204060", size=(300, 200)) -> QWidget:
    w = QWidget()
    w.setAttribute(Qt.WA_StyledBackground, True)
    w.setStyleSheet(f"background-color: {color};")
    w.resize(*size)
    w.show()
    QApplication.processEvents()
    return w


class TestIrisReveal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def setUp(self):
        set_motion_enabled(True)

    def tearDown(self):
        set_motion_enabled(True)

    def _snapshot(self, host, color="#ff0000") -> QPixmap:
        pm = QPixmap(host.size())
        pm.fill(QColor(color))
        return pm

    def test_lengths_are_short(self):
        self.assertLessEqual(MOTION.duration_iris_ms, 600)
        self.assertLessEqual(MOTION.duration_ripple_ms, 450)
        self.assertLessEqual(MOTION.duration_rise_ms + MOTION.rise_step_ms * 2, 600)

    def test_old_picture_opens_from_the_center_and_disappears(self):
        host = _host()
        try:
            ov = iris_reveal(host, self._snapshot(host), QPoint(150, 100), duration_ms=300)
            self.assertIsNotNone(ov)
            self.assertTrue(ov.testAttribute(Qt.WA_TransparentForMouseEvents), "操作を妨げない")
            self.assertEqual(ov.geometry(), host.rect())
            # 途中: 中心は新しい画面（下の色）、隅はまだ前の画面（赤）
            ov.set_progress(0.4)
            img = host.grab().toImage()
            k = img.width() / host.width()  # 写しは画面の拡大率ぶん細かい（この PC は 1.75 倍）
            at = lambda x, y: img.pixelColor(int(x * k), int(y * k))  # noqa: E731
            center, corner = at(150, 100), at(3, 3)
            red = (255, 0, 0)
            self.assertNotEqual((center.red(), center.green(), center.blue()), red, "中心は、もう新しい画面")
            self.assertEqual((corner.red(), corner.green(), corner.blue()), red, "隅は、まだ前の画面")
            # 最後まで開くと、どこにも前の画面が残らない
            ov.set_progress(1.0)
            img = host.grab().toImage()
            for x, y in ((3, 3), (296, 3), (3, 196), (296, 196), (150, 100)):
                c = at(x, y)
                self.assertNotEqual((c.red(), c.green(), c.blue()), red, (x, y))
            QTest.qWait(450)
            QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
            self.assertFalse(shiboken6.isValid(ov) and ov.isVisible(), "終わったら消える")
            self.assertIsNone(getattr(host, "_motion_iris_overlay", None))
        finally:
            host.close()

    def test_hole_reaches_the_farthest_corner_from_an_off_center_start(self):
        host = _host()
        try:
            ov = iris_reveal(host, self._snapshot(host), QPoint(10, 10), duration_ms=300)
            ov.set_progress(1.0)
            self.assertGreaterEqual(ov.hole_radius(), ((300 - 10) ** 2 + (200 - 10) ** 2) ** 0.5)
        finally:
            host.close()

    def test_second_switch_replaces_the_first(self):
        host = _host()
        try:
            a = iris_reveal(host, self._snapshot(host), duration_ms=300)
            b = iris_reveal(host, self._snapshot(host, "#00ff00"), duration_ms=300)
            self.assertIsNot(a, b)
            self.assertFalse(a.isVisible(), "前の動きは、すぐ片付ける（2 枚重ならない）")
            self.assertIs(host._motion_iris_overlay, b)
        finally:
            host.close()

    def test_nothing_happens_when_motion_is_off_or_there_is_nothing_to_show(self):
        host = _host()
        try:
            set_motion_enabled(False)
            self.assertIsNone(iris_reveal(host, self._snapshot(host)))
            set_motion_enabled(True)
            self.assertIsNone(iris_reveal(host, QPixmap()))
            host.hide()
            self.assertIsNone(iris_reveal(host, self._snapshot(host)))
        finally:
            host.close()


class TestRipple(unittest.TestCase):
    """ボタンを押した所から、波紋が広がって消える"""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def setUp(self):
        set_motion_enabled(True)

    def tearDown(self):
        set_motion_enabled(True)

    def _button(self, cls=QPushButton):
        host = QWidget()
        btn = cls("押す", host)
        btn.setGeometry(20, 20, 160, 40)
        host.resize(200, 80)
        host.show()
        QApplication.processEvents()
        return host, btn

    def test_press_starts_a_ripple_at_the_pressed_point_and_it_goes_away(self):
        from gui.motion import ButtonPressMotion

        host, btn = self._button()
        flt = ButtonPressMotion()
        btn.installEventFilter(flt)
        clicked = []
        btn.clicked.connect(lambda: clicked.append(1))
        try:
            QTest.mousePress(btn, Qt.LeftButton, pos=QPoint(30, 12))
            ov = getattr(btn, "_motion_ripple", None)
            self.assertIsNotNone(ov)
            self.assertEqual(ov.size(), btn.size(), "ボタンの内側だけ")
            self.assertTrue(ov.testAttribute(Qt.WA_TransparentForMouseEvents))
            self.assertEqual((ov._origin.x(), ov._origin.y()), (30.0, 12.0))
            QTest.mouseRelease(btn, Qt.LeftButton, pos=QPoint(30, 12))
            self.assertEqual(clicked, [1], "クリックは今までどおり届く")
            QTest.qWait(MOTION.duration_ripple_ms + 200)
            QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
            self.assertIsNone(getattr(btn, "_motion_ripple", None))
        finally:
            host.close()

    def test_ripple_is_gone_when_fully_spread(self):
        from gui.motion import ripple

        host, btn = self._button()
        try:
            ov = ripple(btn, QPoint(80, 20), duration_ms=300)
            ov.set_progress(0.3)
            mid = btn.grab().toImage()
            ov.set_progress(1.0)
            a = btn.grab().toImage()
            ov.hide()
            b = btn.grab().toImage()
            self.assertEqual(a, b, "広がりきったら、何も残らない（透明）")
            self.assertNotEqual(mid, b, "途中は、波紋が見えている")
        finally:
            host.close()

    def test_no_ripple_on_checkbox_or_when_motion_is_off(self):
        from PySide6.QtWidgets import QCheckBox

        from gui.motion import ButtonPressMotion

        host, chk = self._button(QCheckBox)
        flt = ButtonPressMotion()
        chk.installEventFilter(flt)
        try:
            QTest.mouseClick(chk, Qt.LeftButton, pos=QPoint(8, 20))
            self.assertIsNone(getattr(chk, "_motion_ripple", None))
        finally:
            host.close()
        host, btn = self._button()
        btn.installEventFilter(flt)
        try:
            set_motion_enabled(False)
            QTest.mouseClick(btn, Qt.LeftButton)
            self.assertIsNone(getattr(btn, "_motion_ripple", None))
        finally:
            host.close()

    def test_button_thrown_away_mid_ripple_is_safe(self):
        import gc

        from gui.motion import ripple

        def run():
            host, btn = self._button()
            ripple(btn, QPoint(10, 10), duration_ms=200)
            host.close()

        run()
        gc.collect()
        QTest.qWait(350)
        gc.collect()


class TestIrisThrownAway(unittest.TestCase):
    def test_window_thrown_away_mid_iris_is_safe(self):
        """動きの途中で画面が捨てられても落ちない（2026-11-17、作っている途中で見つけた）"""
        import gc

        QApplication.instance() or QApplication(sys.argv)
        set_motion_enabled(True)

        def run():
            host = _host()
            pm = QPixmap(host.size())
            pm.fill(QColor("#ff0000"))
            iris_reveal(host, pm, QPoint(10, 10), duration_ms=200)
            host.close()

        run()
        gc.collect()
        QTest.qWait(350)
        gc.collect()


class TestStagedRise(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def setUp(self):
        set_motion_enabled(True)

    def tearDown(self):
        set_motion_enabled(True)

    def _column(self):
        host = QWidget()
        lay = QVBoxLayout(host)
        labels = [QLabel(t, host) for t in ("絵", "名前", "説明")]
        for lb in labels:
            lay.addWidget(lb)
        host.resize(200, 150)
        host.show()
        QApplication.processEvents()
        return host, labels

    def test_items_rise_in_order_and_end_where_they_started(self):
        host, labels = self._column()
        try:
            home = [lb.pos() for lb in labels]
            n = staged_rise(labels, step_ms=30, duration_ms=120)
            self.assertEqual(n, 3)
            QApplication.processEvents()
            self.assertIsNotNone(labels[2].graphicsEffect(), "後ろの部品は、順番が来るまで透明にしてある")
            QTest.qWait(400)
            for lb, p in zip(labels, home):
                self.assertEqual(lb.pos(), p, "終わったら元の位置")
                self.assertIsNone(lb.graphicsEffect(), "終わったら描画の効果を外す")
                self.assertTrue(lb.isVisible())
        finally:
            host.close()

    def test_off_means_no_motion(self):
        host, labels = self._column()
        try:
            set_motion_enabled(False)
            self.assertEqual(staged_rise(labels), 0)
            for lb in labels:
                self.assertIsNone(lb.graphicsEffect())
        finally:
            host.close()

    def test_hidden_items_are_skipped(self):
        host, labels = self._column()
        try:
            labels[0].hide()
            self.assertEqual(staged_rise(labels, step_ms=10, duration_ms=60), 2)
            QTest.qWait(200)
            self.assertFalse(labels[0].isVisible())
        finally:
            host.close()


class TestInTheWindow(unittest.TestCase):
    """実際の画面: テーマを切り替えると円く開き、終わると何も残らない。カードの頭は、最初に 1 回だけ動く"""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)
        set_motion_enabled(True)
        from gui.main_window import MainWindow

        cls.win = MainWindow()
        cls.win._reset_app_settings()
        cls.win.resize(1450, 900)
        cls.win.show()
        for _ in range(8):
            QApplication.processEvents()

    @classmethod
    def tearDownClass(cls):
        set_motion_enabled(True)
        cls.win._set_dirty(False)
        cls.win.close()
        cls.win.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        QApplication.processEvents()

    def test_theme_switch_opens_like_an_iris_and_leaves_nothing_behind(self):
        win = self.win
        set_motion_enabled(True)
        start = win.ui_theme
        other = "kiseki" if start != "kiseki" else "dark"
        try:
            win.apply_ui_theme(other)
            self.assertEqual(win.ui_theme, other, "切り替えそのものは、すぐに済んでいる")
            ov = getattr(win, "_motion_iris_overlay", None)
            self.assertIsNotNone(ov, "前の画面の写しが重なっている")
            self.assertEqual(ov.size(), win.size())
            QTest.qWait(MOTION.duration_iris_ms + 250)
            QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
            self.assertIsNone(getattr(win, "_motion_iris_overlay", None))
            # 同じテーマをもう一度選んでも、何も重ねない
            win.apply_ui_theme(other)
            self.assertIsNone(getattr(win, "_motion_iris_overlay", None))
        finally:
            win.apply_ui_theme(start)
            QTest.qWait(MOTION.duration_iris_ms + 250)

    def test_theme_switch_without_motion_is_instant(self):
        win = self.win
        start = win.ui_theme
        other = "forest" if start != "forest" else "dark"
        set_motion_enabled(False)
        try:
            win.apply_ui_theme(other)
            self.assertEqual(win.ui_theme, other)
            self.assertIsNone(getattr(win, "_motion_iris_overlay", None))
        finally:
            win.apply_ui_theme(start)
            set_motion_enabled(True)

    def test_brand_rises_once_and_settles(self):
        card = self.win.empty_state_card
        self.assertTrue(card._brand_shown, "最初に出たときに動かしてある")
        QTest.qWait(MOTION.duration_rise_ms + MOTION.rise_step_ms * 2 + 250)
        home = [w.pos() for w in (card.brand_icon, card.brand_name, card.brand_tagline)]
        self.assertEqual(card.play_brand_entrance(), 3)
        QTest.qWait(MOTION.duration_rise_ms + MOTION.rise_step_ms * 2 + 250)
        for w, p in zip((card.brand_icon, card.brand_name, card.brand_tagline), home):
            self.assertEqual(w.pos(), p)
            self.assertIsNone(w.graphicsEffect())
        # 隠して出し直しても、もう動かさない（毎回動くと、うるさい）
        card.hide()
        card.show()
        QApplication.processEvents()
        self.assertIsNone(card.brand_name.graphicsEffect())


if __name__ == "__main__":
    unittest.main()
