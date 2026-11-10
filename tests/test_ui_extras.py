"""
新UIの操作まわりの追加機能（gui/ui_extras.py）のテスト
- コマンド表: ID の重複なし、絵文字なし、使えない理由、実行で既存の操作が動くこと
- コマンドランチャー: ひらがな・カタカナ・英字で探せる、Enter で実行、使えない機能は開いたまま、キー操作、最近使った
- 右クリックのクイックパレット: 選択の有無で中身が変わる、タイルで実行、使えないタイルは押せない
- パネルの小窓: 出す → タブを切り替えても隠れない → 戻すと元の並び順、ドックは左パネル格納時だけ
- モード別の操作ガイド: ツールに応じた案内、通常時は出さない、設定で消せる
- 操作説明書: 設定に「説明書」タブ、検索、リンクで機能を実行、別ウィンドウ、本文が参照するコマンドは全て実在
- 従来UIには入らないこと、テーマ切替で壊れないこと
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PySide6.QtCore import QPoint, QSettings, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QTabWidget

from gui.commands import GROUP_ORDER, build_commands, match_score, normalize
from gui.manual import TOPICS, command_ids_in_manual, render_topic, topic_ids
from gui.mode_hints import current_hint
from gui.modern_ui import has_emoji
from gui.popout_panels import PANELS

_SETTING_KEYS = ("launcher_recent", "mode_hints_enabled", "hints_enabled", "hints_never")


def _points(n=4000):
    rng = np.random.default_rng(5)
    return rng.random((n, 3)) * [10.0, 8.0, 3.0]


class TestSearchNormalization(unittest.TestCase):
    def test_normalize_and_match(self):
        self.assertEqual(normalize("ダンメン"), "だんめん")
        self.assertEqual(normalize("Ｃｔｒｌ＋Ｋ  "), "ctrl+k")
        cmds = {c.id: c for c in build_commands(None)}
        self.assertGreater(match_score(cmds["section.z"], "だんめん"), 0)
        self.assertGreater(match_score(cmds["section.z"], "ダンメン たかさ"), 0)
        self.assertEqual(match_score(cmds["section.z"], "だんめん ぶらし"), 0, "空白区切りの語はすべて含むこと")
        self.assertGreater(match_score(cmds["mode.brush"], "brush"), 0)
        self.assertGreater(match_score(cmds["file.save"], "ほぞん"), 0)
        # 題名が一致するものを優先する
        self.assertGreater(match_score(cmds["mode.rect"], "矩形"), match_score(cmds["mode.polygon"], "らっそ"))

    def test_registry_is_consistent(self):
        cmds = build_commands(None)
        ids = [c.id for c in cmds]
        self.assertEqual(len(ids), len(set(ids)), "コマンド ID が重複しています")
        for c in cmds:
            self.assertIn(c.group, GROUP_ORDER, c.id)
            self.assertTrue(c.title.strip(), c.id)
            self.assertFalse(has_emoji(c.title) or has_emoji(c.desc), f"絵文字が入っています: {c.id}")
            self.assertIsNotNone(c.run, c.id)

    def test_manual_refers_only_to_existing_commands(self):
        from gui.ui_extras import UiExtras  # noqa: F401  （追加ぶんの ID は下で足す）

        ids = {c.id for c in build_commands(None)} | {"help.manual", "help.keys", "help.mode_hints", "help.tips", "help.hints", "panel.restore_all"}
        ids |= {f"panel.{k}" for k, _a, _s, _t in PANELS}
        missing = sorted(set(command_ids_in_manual()) - ids)
        self.assertFalse(missing, f"説明書が存在しないコマンドを参照しています: {missing}")
        self.assertEqual(len(topic_ids()), len(set(topic_ids())))
        for required in ("start", "quick", "open", "nav", "auto", "select", "layer", "section", "detail", "measure", "save", "big", "keys", "trouble"):
            self.assertIn(required, topic_ids())


class _WinCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def setUp(self):
        from gui.main_window import MainWindow

        s = QSettings("E57PointCloudTool", "Settings")
        self._backup = {k: s.value(k, None) for k in _SETTING_KEYS}
        for k in _SETTING_KEYS:
            s.remove(k)
        self.win = MainWindow()
        self.win._reset_app_settings()
        self.win.resize(1400, 900)
        self.win.show()
        QApplication.processEvents()
        self.ex = self.win.ui_extras

    def tearDown(self):
        self.win._set_dirty(False)
        self.win.close()
        self.win.deleteLater()
        QApplication.processEvents()
        s = QSettings("E57PointCloudTool", "Settings")
        for k, v in self._backup.items():
            if v is None:
                s.remove(k)
            else:
                s.setValue(k, v)

    def _load(self):
        self.win.viewer.set_point_cloud(_points())
        QApplication.processEvents()

    def _cmd(self, cmd_id):
        return {c.id: c for c in self.ex.commands()}[cmd_id]


class TestCommandsOnWindow(_WinCase):
    def test_availability_and_execution(self):
        win = self.win
        self.assertIsNone(self._cmd("file.open").unavailable_reason(win))
        for cid in ("file.save", "mode.brush", "section.z", "edit.delete", "view.fit"):
            self.assertTrue(self._cmd(cid).unavailable_reason(win), f"点群なしでは使えないはず: {cid}")
        self.assertFalse(self._cmd("mode.brush").execute(win), "使えないときは実行しない")

        self._load()
        self.assertIsNone(self._cmd("mode.rect").unavailable_reason(win))
        self.assertTrue(self._cmd("mode.rect").execute(win))
        self.assertTrue(win.viewer.selection_mode)
        self.assertEqual(win.viewer.selection_shape, win.viewer.SELECT_RECT)
        self._cmd("mode.orbit").execute(win)
        self.assertFalse(win.viewer.selection_mode)

        self.assertFalse(win.viewer.clip_z_enabled)
        self._cmd("section.z").execute(win)
        self.assertTrue(win.viewer.clip_z_enabled)
        self._cmd("section.thick_0.5").execute(win)
        self.assertAlmostEqual(win.spin_thick_z.value(), 0.5)
        self.assertIsNone(self._cmd("section.crop").unavailable_reason(win))
        self._cmd("section.reset").execute(win)
        self.assertFalse(win.viewer.clip_z_enabled)
        self.assertTrue(self._cmd("section.reload").unavailable_reason(win), "断面なしでは読み直せない")

        self._cmd("view.top").execute(win)
        QTest.qWait(400)
        self.assertGreater(win.viewer.pitch, 80.0)
        self.assertTrue(self._cmd("edit.delete").unavailable_reason(win), "選択なしでは使えない")
        win.viewer.selected_indices = np.arange(50, dtype=np.int64)
        self.assertIsNone(self._cmd("edit.delete").unavailable_reason(win))
        self._cmd("edit.clear").execute(win)
        self.assertEqual(len(win.viewer.selected_indices), 0)

        # 配色とパネル切替
        self._cmd("view.color_height").execute(win)
        self.assertTrue(win.color_combo.currentText().startswith("高さ"))
        self._cmd("section.panel").execute(win)
        self.assertTrue(win.sidebar_tab_btns["section"].isChecked())
        win.viewer.validate_invariants()


class TestLauncher(_WinCase):
    def test_open_search_run_and_keys(self):
        win, la = self.win, self.ex.launcher
        self.assertFalse(la.is_open())
        la.toggle()
        self.assertTrue(la.is_open())
        all_ids = la.visible_command_ids()
        self.assertIn("file.open", all_ids)
        self.assertIn("help.manual", all_ids)
        self.assertIn("panel.section", all_ids)
        self.assertEqual(la.current_command_id(), all_ids[0])

        la.search.setText("だんめん")
        ids = la.visible_command_ids()
        self.assertIn("section.z", ids)
        self.assertTrue(all(i.startswith(("section.", "panel.section")) for i in ids), ids)
        la.search.setText("ぶらし")
        self.assertEqual(la.visible_command_ids()[0], "mode.brush")
        la.search.setText("zzzzzz")
        self.assertEqual(la.visible_command_ids(), [])

        # 点群なし: 使えない機能は実行されず、開いたまま
        la.search.setText("ブラシ")
        self.assertFalse(la.run_current())
        self.assertTrue(la.is_open())
        self.assertFalse(win.viewer.selection_mode)

        # キー操作: ↓ で次へ、Esc で閉じる（ウィンドウのショートカットに取られない）
        la.search.setText("")
        first = la.current_command_id()
        QTest.keyClick(la.search, Qt.Key_Down)
        self.assertNotEqual(la.current_command_id(), first)
        QTest.keyClick(la.search, Qt.Key_Up)
        self.assertEqual(la.current_command_id(), first)
        QTest.keyClick(la.search, Qt.Key_Escape)
        self.assertFalse(la.is_open())

        # 点群あり: 検索して Enter で実行 → 閉じて、矩形選択になる。最近使ったに入る
        self._load()
        la.open_launcher()
        la.search.setText("矩形")
        self.assertEqual(la.current_command_id(), "mode.rect")
        QTest.keyClick(la.search, Qt.Key_Return)
        QTest.qWait(50)
        self.assertFalse(la.is_open())
        self.assertTrue(win.viewer.selection_mode)
        self.assertEqual(win.viewer.selection_shape, win.viewer.SELECT_RECT)
        la.open_launcher()
        self.assertEqual(la.visible_command_ids()[0], "mode.rect", "最近使った機能が先頭に出る")
        # 外側をクリックすると閉じる
        QTest.mouseClick(win.viewer, Qt.LeftButton, Qt.NoModifier, QPoint(5, win.viewer.height() - 5))
        self.assertFalse(la.is_open())

    def test_entry_points(self):
        ex = self.ex
        self.assertIsNotNone(ex.nav_launcher_btn)
        self.assertFalse(has_emoji(ex.nav_launcher_btn.text()))
        ex.nav_launcher_btn.click()
        self.assertTrue(ex.launcher.is_open())
        ex.nav_launcher_btn.click()
        self.assertFalse(ex.launcher.is_open())
        titles = [a.text() for m in self.win.menuBar().actions() if m.menu() for a in m.menu().actions()]
        self.assertTrue(any(t.startswith("機能を検索") for t in titles))
        self.assertTrue(any(t.startswith("操作説明書") for t in titles))


class TestQuickPalette(_WinCase):
    def test_palette_follows_selection_state(self):
        win, pal = self.win, self.ex.palette
        pos = win.viewer.mapToGlobal(QPoint(300, 300))
        # 点群なし: ツールのタイルは押せない
        win.viewer.context_menu_requested.emit(pos)
        self.assertTrue(pal.is_open())
        self.assertFalse(pal.tile("mode.brush").isEnabled())
        pal.close_popup()

        self._load()
        win.viewer.context_menu_requested.emit(pos)
        ids = pal.tile_command_ids()
        self.assertIn("mode.rect", ids)
        self.assertIn("view.top", ids)
        self.assertNotIn("edit.delete", ids)
        self.assertTrue(pal.geometry().left() >= 0 and pal.geometry().right() <= win.viewer.width())
        pal.tile("mode.rect").click()
        self.assertFalse(pal.is_open())
        self.assertEqual(win.viewer.selection_shape, win.viewer.SELECT_RECT)

        # 選択あり: 選択に対する操作になる
        win.viewer.selected_indices = np.arange(200, dtype=np.int64)
        win.viewer.context_menu_requested.emit(pos)
        ids = pal.tile_command_ids()
        self.assertEqual(ids[0], "edit.delete")
        self.assertNotIn("mode.rect", ids)
        self.assertTrue(pal.tile("edit.delete").isEnabled())
        pal.tile("edit.clear").click()
        self.assertEqual(len(win.viewer.selected_indices), 0)
        self.assertFalse(pal.is_open())
        # Esc で閉じる
        win.viewer.context_menu_requested.emit(pos)
        QTest.keyClick(win.viewer, Qt.Key_Escape)
        self.assertFalse(pal.is_open())
        for b in pal.tiles:
            self.assertFalse(has_emoji(b.text()))


class TestPopouts(_WinCase):
    def _order(self):
        lay = self.win.sidebar_widget.layout()
        return [lay.indexOf(getattr(self.win, a)) for _k, a, _s, _t in PANELS] + [lay.indexOf(self.win.save_group)]

    def test_pop_out_keeps_working_and_returns_in_order(self):
        win, po = self.win, self.ex.popouts
        before = self._order()
        self.assertEqual(before, sorted(before))
        self._load()
        w = po.pop_out("section")
        QApplication.processEvents()
        self.assertTrue(po.is_popped("section"))
        self.assertTrue(w.isVisible())
        self.assertTrue(win.section_group.isVisibleTo(w))
        self.assertEqual(win.sidebar_widget.layout().indexOf(win.section_group), -1)
        # 小窓の中の操作はそのまま効く
        win.chk_sec_z.setChecked(True)
        self.assertTrue(win.viewer.clip_z_enabled)
        # 左パネルのタブを切り替えても、小窓のグループは隠れない
        for tab in ("filter", "edit", "layer", "view", "settings", "all"):
            win._set_sidebar_workflow_tab(tab)
            self.assertTrue(win.section_group.isVisibleTo(w), tab)
        # もう 1 つ出して、まとめて戻す
        po.pop_out("view")
        self.assertEqual(sorted(po.popped_keys()), ["section", "view"])
        po.restore_all()
        QApplication.processEvents()
        self.assertEqual(po.popped_keys(), [])
        after = self._order()
        self.assertTrue(all(i >= 0 for i in after))
        self.assertEqual(after, sorted(after), "戻したあとも作業順の並びが保たれる")
        self.assertTrue(win.viewer.clip_z_enabled, "状態はそのまま")
        # 小窓を × で閉じても戻る
        w2 = po.pop_out("filter")
        w2.close()
        QApplication.processEvents()
        self.assertFalse(po.is_popped("filter"))
        self.assertGreaterEqual(win.sidebar_widget.layout().indexOf(win.auto_group), 0)
        self.assertEqual(self._order(), sorted(self._order()))

    def test_header_button_and_dock(self):
        win, po, dock = self.win, self.ex.popouts, self.ex.dock
        self.assertEqual(sorted(po._header_buttons.keys()), sorted(k for k, _a, _s, _t in PANELS))
        self.assertTrue(dock.isHidden(), "左パネルが開いている間はドックを出さない")
        btn = po._header_buttons["layer"]
        self.assertEqual(btn.text(), "別窓")
        btn.click()
        self.assertTrue(po.is_popped("layer"))
        self.assertEqual(btn.text(), "戻す")
        btn.click()
        self.assertFalse(po.is_popped("layer"))
        # 左パネルを格納するとドックが出る。ボタンで小窓を出し入れできる
        win.btn_toggle_sidebar.click()
        QApplication.processEvents()
        self.assertTrue(win.sidebar_scroll.isHidden())
        self.assertFalse(dock.isHidden())
        dock.buttons["section"].click()
        self.assertTrue(po.is_popped("section"))
        self.assertTrue(dock.buttons["section"].isChecked())
        dock.buttons["section"].click()
        self.assertFalse(po.is_popped("section"))
        dock.buttons["launcher"].click()
        self.assertTrue(self.ex.launcher.is_open())
        self.ex.launcher.close_popup()
        win.btn_toggle_sidebar.click()
        QApplication.processEvents()
        self.assertTrue(dock.isHidden())
        for b in list(dock.buttons.values()) + list(po._header_buttons.values()):
            self.assertFalse(has_emoji(b.text()))


class TestModeHints(_WinCase):
    def test_hints_follow_mode(self):
        win, hints = self.win, self.ex.hints
        self.assertIsNone(current_hint(win), "点群なしでは出さない")
        self._load()
        self.assertIsNone(current_hint(win), "回転モードで未選択なら出さない")
        hints.refresh()
        self.assertTrue(hints.isHidden())

        win._set_interaction_mode("rect")
        title, lines = current_hint(win)
        self.assertEqual(title, "矩形選択")
        self.assertTrue(any("Del" in ln for ln in lines))
        hints.refresh()
        self.assertFalse(hints.isHidden())
        self.assertEqual(hints.title_label.text(), "矩形選択")
        self.assertLessEqual(hints.geometry().right(), win.viewer.width())

        win._set_interaction_mode("polygon")
        self.assertEqual(current_hint(win)[0], "多角形選択")
        win._set_interaction_mode("brush")
        self.assertEqual(current_hint(win)[0], "ブラシ選択")
        win._set_interaction_mode("orbit")
        win._toggle_measure_mode(True)
        self.assertEqual(current_hint(win)[0], "計測")
        win._toggle_measure_mode(False)
        win.viewer.selected_indices = np.arange(30, dtype=np.int64)
        self.assertEqual(current_hint(win)[0], "選択中 30 点")
        win._clear_selection()
        self.assertIsNone(current_hint(win))

        # 設定で消せる（説明書タブのチェックと連動し、保存される）
        win._set_interaction_mode("rect")
        self.ex.manual_tab.chk_hints.setChecked(False)
        hints.refresh()
        self.assertTrue(hints.isHidden())
        self.assertFalse(QSettings("E57PointCloudTool", "Settings").value("mode_hints_enabled", True, type=bool))
        self.ex.run_command("help.mode_hints")
        self.assertTrue(hints.is_enabled())
        self.assertTrue(self.ex.manual_tab.chk_hints.isChecked())
        hints.refresh()
        self.assertFalse(hints.isHidden())


class TestManual(_WinCase):
    def test_manual_tab_search_links_and_window(self):
        win, ex = self.win, self.ex
        tabs = win.settings_group.findChild(QTabWidget)
        labels = [tabs.tabText(i) for i in range(tabs.count())]
        self.assertIn("説明書", labels)
        br = ex.manual_tab.browser
        self.assertEqual(br.current_topic_id(), "start")
        self.assertIn("基本の流れ", br.view.toPlainText())

        br.search.setText("ほぞん")
        self.assertIn("save", br.visible_topic_ids())
        self.assertLess(len(br.visible_topic_ids()), len(TOPICS))
        br.search.setText("ぜったいにないことば")
        self.assertEqual(br.visible_topic_ids(), [])
        br.search.setText("")
        self.assertEqual(len(br.visible_topic_ids()), len(TOPICS))

        br.follow_link("topic:section")
        self.assertEqual(br.current_topic_id(), "section")
        self.assertIn("斜めの壁", br.view.toPlainText())
        # リンクから機能を実行できる
        self._load()
        br.follow_link("cmd:mode.rect")
        self.assertEqual(win.viewer.selection_shape, win.viewer.SELECT_RECT)
        self.assertTrue(win.viewer.selection_mode)

        # 全項目が描けて、記法が残らない。ショートカット一覧はコマンド表から作られる
        cmds = ex.commands()
        for t in TOPICS:
            html_text = render_topic(t, cmds)
            self.assertNotIn("[[", html_text, t.id)
            self.assertGreater(len(html_text), 100, t.id)
        keys_html = render_topic(next(t for t in TOPICS if t.id == "keys"), cmds)
        for needle in ("Ctrl+K", "Ctrl+Z", "Ctrl+S", "F1", "Del"):
            self.assertIn(needle, keys_html)

        # 別ウィンドウ（F1）
        ex.manual_tab.btn_open_window.click()
        mw = ex.manual_window()
        self.assertIsNotNone(mw)
        self.assertTrue(mw.isVisible())
        self.assertEqual(mw.browser.current_topic_id(), "section", "タブで見ていた項目から開く")
        self.assertEqual(mw.browser.topic_list.count(), len(TOPICS))
        ex.run_command("help.keys")
        self.assertEqual(mw.browser.current_topic_id(), "keys")
        mw.close()

    def test_theme_switch_keeps_extras_working(self):
        win, ex = self.win, self.ex
        other = "dark" if getattr(win, "ui_theme", "dark") != "dark" else "kiseki"
        win.apply_ui_theme(other)
        QApplication.processEvents()
        ex.launcher.open_launcher()
        self.assertTrue(ex.launcher.is_open())
        ex.launcher.close_popup()
        ex.open_manual("start").close()
        self.assertIn("基本の流れ", ex.manual_tab.browser.view.toPlainText())


class TestClassicUiHasNoExtras(unittest.TestCase):
    def test_classic_ui_is_untouched(self):
        from gui.main_window import MainWindow
        from gui.ui_mode import reset_ui_mode_cache

        QApplication.instance() or QApplication(sys.argv)
        os.environ["E57_UI_MODE"] = "classic"
        reset_ui_mode_cache()
        try:
            win = MainWindow()
            win.show()
            QApplication.processEvents()
            self.assertEqual(win.ui_mode, "classic")
            self.assertFalse(hasattr(win, "ui_extras"))
            tabs = win.settings_group.findChild(QTabWidget)
            self.assertNotIn("説明書", [tabs.tabText(i) for i in range(tabs.count())])
            win.close()
            win.deleteLater()
        finally:
            os.environ.pop("E57_UI_MODE", None)
            reset_ui_mode_cache()
        QApplication.processEvents()


if __name__ == "__main__":
    unittest.main()
