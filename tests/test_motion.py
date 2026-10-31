"""
UI モーション（gui/motion.py）のテスト

- 押下フィードバックのイベントフィルタがマウス押下→離すで例外なく動き、終了後に描画効果が外れること
- フェードイン／アウト、フェード＋スライド、カスケードが表示・非表示を正しく切り替え、位置を元に戻すこと
- シェイク／バウンスが終了時に必ず元の位置へ戻ること（レイアウトを乱さない）
- 数値ティッカーが最終的に正確な文字列で終わり、桁幅固定を解除すること
- フォーカスパルスのオーバーレイが終了後に消えること、アコーディオン展開が高さ制限を解除すること
- アニメーション OFF（set_motion_enabled(False)）では全て即時反映になること
- モーションの長さが「短い」範囲（反応 ≤ 100ms、出入り ≤ 300ms、遊び心 ≤ 450ms）に収まっていること
"""

import sys
import unittest

import shiboken6
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget

from gui.motion import (
    MOTION,
    ButtonPressMotion,
    bounce,
    cancel_fade_out,
    cascade_in,
    expand_animated,
    fade_in,
    fade_out_then_hide,
    fade_slide_in,
    focus_pulse,
    install_button_motion,
    install_number_ticker,
    motion_enabled,
    set_motion_enabled,
    shake,
)


class TestMotion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def setUp(self):
        set_motion_enabled(True)

    def tearDown(self):
        set_motion_enabled(True)

    # ------------------------------------------------------------------ 長さ
    def test_durations_are_short(self):
        self.assertLessEqual(MOTION.duration_fast_ms, 100)
        self.assertLessEqual(MOTION.duration_base_ms, 200)
        self.assertLessEqual(MOTION.duration_slow_ms, 300)
        self.assertLessEqual(MOTION.duration_expand_ms, 300)
        for d in (MOTION.duration_shake_ms, MOTION.duration_bounce_ms, MOTION.duration_pulse_ms, MOTION.duration_ticker_ms):
            self.assertLessEqual(d, 450)
        self.assertTrue(0.4 <= MOTION.press_opacity < 1.0)
        self.assertLessEqual(MOTION.cascade_step_ms * MOTION.cascade_max_steps + MOTION.duration_base_ms, 320)

    # ------------------------------------------------------------------ 押下
    def test_button_press_release_cycle(self):
        flt = install_button_motion(self.app)
        self.assertIsInstance(flt, ButtonPressMotion)
        host = QWidget()
        btn = QPushButton("押す", host)
        host.show()
        QTest.qWait(50)
        pos = btn.pos()
        QTest.mousePress(btn, Qt.LeftButton)
        self.assertIsNotNone(btn.graphicsEffect(), "押下時に不透明度の効果が付くこと")
        QTest.mouseRelease(btn, Qt.LeftButton)
        QTest.qWait(MOTION.duration_base_ms + 150)
        self.assertIsNone(btn.graphicsEffect(), "復帰後は効果が外れ、通常描画に戻ること")
        self.assertEqual(btn.pos(), pos)
        host.close()

    # ------------------------------------------------------------------ 出現・退場
    def test_fade_in_and_out(self):
        host = QWidget()
        lbl = QLabel("toast", host)
        host.show()
        lbl.hide()
        fade_in(lbl, 60)
        self.assertTrue(lbl.isVisible())
        QTest.qWait(200)
        self.assertIsNone(lbl.graphicsEffect())
        fade_out_then_hide(lbl, 60)
        QTest.qWait(200)
        self.assertFalse(lbl.isVisible())
        host.close()

    def test_cancel_fade_out_keeps_widget_visible(self):
        host = QWidget()
        lbl = QLabel("toast", host)
        host.show()
        QTest.qWait(30)
        self.assertFalse(cancel_fade_out(lbl), "退場中でなければ何もしないこと")
        fade_out_then_hide(lbl, 100)
        QTest.qWait(30)
        self.assertTrue(cancel_fade_out(lbl))
        self.assertIsNone(lbl.graphicsEffect())
        QTest.qWait(200)
        self.assertTrue(lbl.isVisible(), "取り消した後に隠れないこと")
        fade_out_then_hide(lbl, 40)  # 取り消した後も通常どおり退場できる
        QTest.qWait(150)
        self.assertFalse(lbl.isVisible())
        self.assertFalse(cancel_fade_out(lbl))
        host.close()

    def test_fade_slide_in_restores_position(self):
        host = QWidget()
        lbl = QLabel("toast", host)
        lbl.move(20, 30)
        host.show()
        lbl.hide()
        fade_slide_in(lbl, dy=8, duration_ms=60, overshoot=True)
        self.assertTrue(lbl.isVisible())
        QTest.qWait(250)
        self.assertEqual(lbl.pos(), QPoint(20, 30))
        self.assertIsNone(lbl.graphicsEffect())
        host.close()

    def test_cascade_in_leaves_widgets_visible_and_clean(self):
        host = QWidget()
        lay = QVBoxLayout(host)
        labels = [QLabel(f"箱 {i}") for i in range(4)]
        for l in labels:
            lay.addWidget(l)
        host.show()
        QTest.qWait(30)
        positions = [l.pos() for l in labels]
        cascade_in(labels, each_ms=60)
        QTest.qWait(MOTION.cascade_step_ms * 3 + 60 + 250)
        for l, p in zip(labels, positions):
            self.assertTrue(l.isVisible())
            self.assertIsNone(l.graphicsEffect(), "終了後は効果が外れること")
            self.assertEqual(l.pos(), p, "位置が元に戻ること")
        host.close()

    def test_cascade_skips_hidden_and_cleans_effect(self):
        host = QWidget()
        lay = QVBoxLayout(host)
        a, b = QLabel("a"), QLabel("b")
        lay.addWidget(a)
        lay.addWidget(b)
        host.show()
        QTest.qWait(30)
        cascade_in([a, b], each_ms=60)
        b.hide()  # 自分の番が来る前に隠された
        QTest.qWait(300)
        self.assertIsNone(b.graphicsEffect(), "隠されたウィジェットに不透明度 0 の効果が残らないこと")
        host.close()

    # ------------------------------------------------------------------ 遊び心
    def test_shake_and_bounce_restore_position(self):
        host = QWidget()
        lbl = QLabel("警告", host)
        lbl.move(40, 50)
        host.show()
        QTest.qWait(30)
        shake(lbl, duration_ms=80)
        QTest.qWait(20)
        bounce(lbl, duration_ms=80)  # 連続して始めても前の分は位置を戻してから止まる
        QTest.qWait(300)
        self.assertEqual(lbl.pos(), QPoint(40, 50))
        host.close()

    def test_offset_animation_follows_layout_moves(self):
        """アニメーション中にレイアウトが位置を変えても、最終位置はレイアウトの位置になること"""
        host = QWidget()
        lbl = QLabel("badge", host)
        lbl.move(10, 10)
        host.show()
        QTest.qWait(30)
        bounce(lbl, duration_ms=120)
        QTest.qWait(30)
        lbl.move(100, 100)  # レイアウトが動かしたことに相当
        QTest.qWait(300)
        self.assertEqual(lbl.pos(), QPoint(100, 100))
        host.close()

    def test_focus_pulse_overlay_is_removed(self):
        host = QWidget()
        btn = QPushButton("実行", host)
        btn.move(10, 10)
        host.show()
        QTest.qWait(30)
        overlay = focus_pulse(btn, "#38bdf8", duration_ms=60)
        self.assertIsNotNone(overlay)
        self.assertTrue(overlay.testAttribute(Qt.WA_TransparentForMouseEvents))
        QTest.qWait(300)
        self.assertFalse(shiboken6.isValid(overlay) and overlay.isVisible(), "終了後はオーバーレイが消えること")
        host.close()

    def test_focus_pulse_on_hidden_widget_does_nothing(self):
        host = QWidget()
        btn = QPushButton("実行", host)
        self.assertIsNone(focus_pulse(btn))

    def test_expand_animated_releases_height_limit(self):
        host = QWidget()
        lay = QVBoxLayout(host)
        content = QWidget()
        clay = QVBoxLayout(content)
        for i in range(5):
            clay.addWidget(QLabel(f"行 {i}"))
        lay.addWidget(content)
        host.show()
        QTest.qWait(30)
        expand_animated(content, duration_ms=60)
        self.assertLess(content.maximumHeight(), content.sizeHint().height())
        QTest.qWait(300)
        self.assertEqual(content.maximumHeight(), 16777215)
        self.assertIsNone(content.graphicsEffect())
        host.close()

    # ------------------------------------------------------------------ 数値ティッカー
    def test_number_ticker_ends_exact_and_restores_width(self):
        host = QWidget()
        lbl = QLabel("点数: 0 点", host)
        host.show()
        install_number_ticker(lbl, duration_ms=80)
        lbl.setText("点数: 1,234,567 点")
        self.assertGreater(lbl.minimumWidth(), 0, "数え上げ中は桁幅を固定すること")
        QTest.qWait(300)
        self.assertEqual(lbl.text(), "点数: 1,234,567 点")
        self.assertEqual(lbl.minimumWidth(), 0)
        # 複数の数値も同時に数える（形が変わる最初の 1 回は即時、以後は同じ形なので数え上げ）
        lbl.setText("有効: 1,234,567 点 | ノイズ: 0 点")
        self.assertEqual(lbl.text(), "有効: 1,234,567 点 | ノイズ: 0 点")
        lbl.setText("有効: 1,000,000 点 | ノイズ: 234,567 点")
        QTest.qWait(10)
        self.assertNotEqual(lbl.text(), "有効: 1,000,000 点 | ノイズ: 234,567 点")  # まだ数え上げ中
        QTest.qWait(300)
        self.assertEqual(lbl.text(), "有効: 1,000,000 点 | ノイズ: 234,567 点")
        host.close()

    def test_number_ticker_sets_immediately_when_shape_differs(self):
        host = QWidget()
        lbl = QLabel("点数: 0 点", host)
        host.show()
        install_number_ticker(lbl, duration_ms=80)
        lbl.setText("有効: 10 点 | ノイズ: 5 点")  # 数値の個数が違う → 即時
        self.assertEqual(lbl.text(), "有効: 10 点 | ノイズ: 5 点")
        lbl.setText("読み込み中")  # 数値なし → 即時
        self.assertEqual(lbl.text(), "読み込み中")
        host.close()

    def test_number_ticker_chains_existing_wrapper(self):
        """絵文字除去など既存の setText 包みを引き継ぐこと"""
        host = QWidget()
        lbl = QLabel("0 点", host)
        host.show()
        lbl.setText = lambda s, _l=lbl: QLabel.setText(_l, s.replace("✨", "").strip())  # type: ignore[assignment]
        install_number_ticker(lbl, duration_ms=50)
        lbl.setText("✨ 42 点")
        QTest.qWait(250)
        self.assertEqual(lbl.text(), "42 点")
        host.close()

    # ------------------------------------------------------------------ OFF
    def test_disabled_motion_is_immediate(self):
        set_motion_enabled(False)
        self.assertFalse(motion_enabled())
        host = QWidget()
        lbl = QLabel("点数: 0 点", host)
        lbl.move(5, 5)
        host.show()
        QTest.qWait(20)
        lbl.hide()
        fade_slide_in(lbl)
        self.assertTrue(lbl.isVisible())
        self.assertIsNone(lbl.graphicsEffect())
        shake(lbl)
        bounce(lbl)
        self.assertEqual(lbl.pos(), QPoint(5, 5))
        self.assertIsNone(focus_pulse(lbl))
        install_number_ticker(lbl)
        lbl.setText("点数: 99 点")
        self.assertEqual(lbl.text(), "点数: 99 点")
        fade_out_then_hide(lbl)
        self.assertFalse(lbl.isVisible())
        host.close()


if __name__ == "__main__":
    unittest.main()
