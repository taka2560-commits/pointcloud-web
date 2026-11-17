# -*- coding: utf-8 -*-
"""
ツールの名前とアイコン（gui/app_identity.py・tools/make_app_icon.py・assets/app/）

2026-11-16、利用者の依頼「ツール名を付けたい。ツールを開いた所にタイトルを。アイコン画像もあればよし」から:
- 名前は 1 か所（TOOL_NAME）で決め、ウィンドウの題名・起動した所のカード・バージョン情報・通知の題名に出る
- アイコンは、ウィンドウ・タスクバー・通知・カード・配布版の EXE に同じ絵が出る
- 絵のファイルが無くても、アプリは今までどおり動く（アイコンのせいで起動を止めない）
"""
import os
import struct
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.settings_sandbox import use_sandbox_settings  # noqa: E402

use_sandbox_settings()  # この PC の本番アプリの保存設定を、テストで書き換えない

from PySide6.QtCore import QCoreApplication, QEvent  # noqa: E402
from PySide6.QtGui import QIcon, QImage  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from gui import app_identity as AI  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(name: str) -> str:
    with open(os.path.join(ROOT, name), "r", encoding="utf-8") as f:
        return f.read()


class TestIconFiles(unittest.TestCase):
    """書き出してある絵（assets/app/）が、そろっていて、読める形をしている"""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def test_every_size_exists_and_has_that_size(self):
        for s in AI.ICON_SIZES:
            img = QImage(AI.app_icon_path(s))
            self.assertFalse(img.isNull(), f"{s}px の絵がありません")
            self.assertEqual((img.width(), img.height()), (s, s))

    def test_icon_has_round_corners_and_a_solid_middle(self):
        for s in (16, 48, 256):
            img = QImage(AI.app_icon_path(s))
            self.assertEqual(img.pixelColor(0, 0).alpha(), 0, f"{s}px: 角は透明（角の丸い四角）")
            self.assertEqual(img.pixelColor(s // 2, int(s * 0.9)).alpha(), 255, f"{s}px: 地は塗ってある")

    def test_small_icon_still_shows_the_noise_color(self):
        """16px でも、水色の点（空中のノイズ）が見える（大きな絵を縮めただけだと、つぶれて消える）"""
        img = QImage(AI.app_icon_path(16))
        bluish = 0
        for y in range(16):
            for x in range(16):
                c = img.pixelColor(x, y)
                if c.alpha() == 255 and c.blue() > 170 and c.blue() > c.red() + 40:
                    bluish += 1
        self.assertGreaterEqual(bluish, 2)

    def test_ico_holds_every_size(self):
        """EXE に埋め込む ICO: 入っている絵の数と大きさ、それぞれのデータが、ファイルの中に収まっている"""
        path = os.path.join(AI.app_icon_dir(), "app_icon.ico")
        with open(path, "rb") as f:
            data = f.read()
        reserved, kind, count = struct.unpack_from("<HHH", data, 0)
        self.assertEqual((reserved, kind), (0, 1))
        self.assertEqual(count, len(AI.ICON_SIZES))
        sizes = []
        for i in range(count):
            w, h, _colors, _res, planes, bpp, size, offset = struct.unpack_from("<BBBBHHII", data, 6 + 16 * i)
            sizes.append(w or 256)
            self.assertEqual(w, h)
            self.assertEqual((planes, bpp), (1, 32))
            self.assertLessEqual(offset + size, len(data))
            head = data[offset:offset + 8]
            if (w or 256) >= 256:
                self.assertEqual(head, b"\x89PNG\r\n\x1a\n", "256px は PNG のまま入れる")
            else:
                self.assertEqual(struct.unpack_from("<I", head)[0], 40, "小さい絵は BMP で入れる")
        self.assertEqual(tuple(sizes), AI.ICON_SIZES)

    def test_ico_opens_as_an_icon(self):
        icon = QIcon(os.path.join(AI.app_icon_dir(), "app_icon.ico"))
        self.assertFalse(icon.isNull())
        self.assertFalse(icon.pixmap(32, 32).isNull())

    def test_generator_draws_what_is_committed(self):
        """tools/make_app_icon.py が描く絵と、書き出してある絵は、同じ大きさ・同じ構図（絵だけ変えて書き出し忘れない）"""
        sys.path.insert(0, os.path.join(ROOT, "tools"))
        try:
            import make_app_icon as M
        finally:
            sys.path.pop(0)
        self.assertEqual(tuple(M.SIZES), AI.ICON_SIZES)
        for s in (16, 64):
            drawn = M.draw_icon(s)
            saved = QImage(AI.app_icon_path(s)).convertToFormat(QImage.Format_ARGB32)
            diff = 0
            for y in range(s):
                for x in range(s):
                    a, b = drawn.pixelColor(x, y), saved.pixelColor(x, y)
                    if max(abs(a.red() - b.red()), abs(a.green() - b.green()), abs(a.blue() - b.blue()),
                           abs(a.alpha() - b.alpha())) > 48:
                        diff += 1
            self.assertLessEqual(diff, s * s * 0.02, f"{s}px: 描く絵と、書き出してある絵が違います（書き出し直してください）")


class TestIdentity(unittest.TestCase):
    """名前が 1 か所で決まり、アイコンがアプリ全体に付く"""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def setUp(self):
        self._icon_before = QApplication.windowIcon()

    def tearDown(self):
        QApplication.setWindowIcon(self._icon_before)
        AI._icon = None

    def test_name_is_set_and_plain(self):
        self.assertTrue(AI.TOOL_NAME.strip())
        self.assertEqual(AI.TOOL_NAME, AI.TOOL_NAME.strip())
        self.assertLessEqual(len(AI.TOOL_NAME), 16, "タスクバーや通知で切れない長さ")

    def test_every_place_reads_the_one_name(self):
        from gui import system_tools as ST
        from gui import task_notify as TN

        self.assertTrue(ST.APP_TITLE.startswith(AI.TOOL_NAME))
        self.assertTrue(ST.window_title_with_build().startswith(AI.TOOL_NAME))
        self.assertTrue(ST.about_text().splitlines()[0].startswith(AI.TOOL_NAME))
        self.assertEqual(TN.NOTIFY_TITLE, AI.TOOL_NAME)

    def test_apply_sets_the_icon_for_the_whole_app(self):
        QApplication.setWindowIcon(QIcon())
        self.assertTrue(AI.apply_app_identity(self.app))
        icon = QApplication.windowIcon()
        self.assertFalse(icon.isNull())
        got = {(s.width(), s.height()) for s in icon.availableSizes()}
        for s in (16, 32, 48, 256):
            self.assertIn((s, s), got, "小さい絵から大きい絵まで持つ（縮めた絵で代用しない）")

    def test_notification_uses_the_app_icon(self):
        from gui import task_notify as TN

        AI.apply_app_identity(self.app)
        self.assertEqual(TN._app_icon().cacheKey(), QApplication.windowIcon().cacheKey())

    def test_missing_files_do_not_stop_the_app(self):
        """絵のフォルダが無くても、例外を出さずに「付けられなかった」と返す"""
        QApplication.setWindowIcon(QIcon())
        AI._icon = None
        with tempfile.TemporaryDirectory() as d, mock.patch.object(AI, "assets_dir", lambda: d):
            self.assertFalse(AI.apply_app_identity(self.app))
            self.assertTrue(AI.app_icon_pixmap(44).isNull())
        self.assertTrue(QApplication.windowIcon().isNull())

    def test_apply_never_raises(self):
        class Broken:
            def setWindowIcon(self, _icon):
                raise RuntimeError("boom")

        self.assertFalse(AI.apply_app_identity(Broken()))

    def test_taskbar_id_only_when_run_from_source(self):
        """配布版（EXE）では、タスクバーの指定をしない（ピン留めしたボタンの扱いを変えない）"""
        with mock.patch.object(sys, "frozen", True, create=True):
            self.assertFalse(AI._set_taskbar_app_id())


class TestWindowShowsTheName(unittest.TestCase):
    """ツールを開いた所に、名前とアイコンが出ている"""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)
        cls._icon_before = QApplication.windowIcon()
        AI._icon = None
        AI.apply_app_identity(cls.app)
        from gui.main_window import MainWindow

        cls.win = MainWindow()
        cls.win._reset_app_settings()
        cls.win.resize(1450, 900)
        cls.win.show()
        cls._pump()

    @classmethod
    def tearDownClass(cls):
        cls.win._set_dirty(False)
        cls.win.close()
        cls.win.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        QApplication.processEvents()
        QApplication.setWindowIcon(cls._icon_before)
        AI._icon = None

    @staticmethod
    def _pump(n=8):
        for _ in range(n):
            QApplication.processEvents()

    def test_title_bar(self):
        win = self.win
        self.assertTrue(win.windowTitle().startswith(AI.TOOL_NAME))
        self.assertFalse(win.windowIcon().isNull(), "ウィンドウの左上とタスクバーに絵が出る")
        win._set_dirty(True)
        try:
            self.assertTrue(win.windowTitle().startswith("* " + AI.TOOL_NAME), "未保存の印が付いても、名前は残る")
        finally:
            win._set_dirty(False)

    def test_start_card_shows_name_and_icon(self):
        card = self.win.empty_state_card
        self.assertTrue(card.isVisible())
        self.assertEqual(card.brand_name.text(), AI.TOOL_NAME)
        self.assertEqual(card.brand_tagline.text(), AI.TOOL_TAGLINE)
        self.assertTrue(card.brand_icon.isVisible())
        self.assertFalse(card.brand_icon.pixmap().isNull())
        # 名前は、カードの中でいちばん上・いちばん大きい字
        self.assertLess(card.brand_name.mapTo(card, card.brand_name.rect().topLeft()).y(),
                        card.btn_open.mapTo(card, card.btn_open.rect().topLeft()).y())
        self.assertGreater(card.brand_name.fontInfo().pixelSize(), card.brand_tagline.fontInfo().pixelSize())

    def test_start_card_fits_in_both_themes(self):
        win, card = self.win, self.win.empty_state_card
        try:
            for theme in ("kiseki", "abyss", "forest", "dark"):
                win.apply_ui_theme(theme)
                self._pump(12)
                v = win.viewer
                self.assertLessEqual(card.width(), v.width(), theme)
                self.assertLessEqual(card.height(), v.height(), theme)
                self.assertLessEqual(abs((card.x() + card.width() / 2) - v.width() / 2), 2, f"{theme}: カードは画面の中央")
                for w in (card.brand_name, card.brand_tagline):
                    need = w.fontMetrics().horizontalAdvance(w.text())
                    self.assertGreaterEqual(w.width(), need, f"{theme}: 字が切れている: {w.text()}")
        finally:
            win.apply_ui_theme("dark")
            self._pump(6)

    def test_every_startup_tip_fits_in_the_card(self):
        """
        カードの下のヒントは、3 行になる文でも下が切れない。
        2026-11-16 に見つけた: カードの高さを sizeHint だけで決めていて、明るいテーマでは 20 件のうち 2 件の下が切れていた
        """
        from gui.hints import TIPS

        win, card = self.win, self.win.empty_state_card
        lbl = win.ui_extras.startup_tip_label
        before = lbl.text()
        try:
            for theme in ("kiseki", "dark"):
                win.apply_ui_theme(theme)
                self._pump(12)
                for tip in TIPS:
                    lbl.setText(f"ヒント: {tip.text}")
                    card._recenter()
                    self._pump(3)
                    self.assertGreaterEqual(lbl.height(), lbl.heightForWidth(lbl.width()), f"{theme}: 下が切れる: {tip.text}")
                    self.assertLessEqual(card.height(), win.viewer.height(), theme)
        finally:
            lbl.setText(before)
            win.apply_ui_theme("dark")
            card._recenter()
            self._pump(6)

    def test_card_without_icon_shows_the_name_only(self):
        from gui.modern_ui import EmptyStateCard

        AI._icon = None
        with tempfile.TemporaryDirectory() as d, mock.patch.object(AI, "assets_dir", lambda: d):
            card = EmptyStateCard(self.win)
            try:
                card.show()
                self._pump()
                self.assertFalse(card.brand_icon.isVisible())
                self.assertEqual(card.brand_name.text(), AI.TOOL_NAME)
            finally:
                self.win.viewer.removeEventFilter(card)
                card.hide()
                card.deleteLater()
        AI._icon = None
        AI.app_icon()


class TestWiring(unittest.TestCase):
    """起動とビルドが、名前とアイコンを使うようにつないである"""

    def test_main_applies_identity_right_after_creating_the_app(self):
        src = _read("main.py")
        i_app = src.index("app = QApplication(sys.argv)")
        i_apply = src.index("apply_app_identity(app)")
        i_win = src.index("window = MainWindow()")
        self.assertLess(i_app, i_apply)
        self.assertLess(i_apply, i_win, "ウィンドウを作る前にアイコンを決める")

    def test_spec_embeds_the_icon_and_ships_the_pngs(self):
        spec = _read("E57_PointCloud_Tool.spec")
        self.assertIn("icon=os.path.join('assets', 'app', 'app_icon.ico')", spec)
        self.assertIn("('assets', 'assets')", spec, "画面で使う PNG は assets ごと同梱する")
        self.assertIn("name='E57_PointCloud_Tool'", spec, "EXE の名前は変えない（ショートカットと手順がそのまま使える）")
        self.assertTrue(os.path.isfile(os.path.join(ROOT, "assets", "app", "app_icon.ico")))


if __name__ == "__main__":
    unittest.main()
