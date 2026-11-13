# -*- coding: utf-8 -*-
"""
システムの道具（gui/system_tools.py）: 作業用データの管理・書く前の空きの確認・版の表示

2026-11-13、利用者の PC の確認から:
- 作業用データ（1 つ約 17GB）が見えず、消せず、置き場所も変えられなかった（C ドライブの空きは約 20GB）
- 保存の前に空きを確かめていなかった
- 画面にもログにも版が出ず、どの版で起きたことか区別できなかった
"""
import os
import shutil
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.settings_sandbox import use_sandbox_settings  # noqa: E402

use_sandbox_settings()  # この PC の本番アプリの保存設定を、テストで書き換えない

import numpy as np  # noqa: E402
from PySide6.QtCore import QCoreApplication, QEvent, QTimer  # noqa: E402
from PySide6.QtGui import QShowEvent  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from core import section_cache as SC  # noqa: E402
from core.build_info import get_build_info  # noqa: E402
from core.sysinfo import GB, format_bytes  # noqa: E402
from core.workdata import FreeCheck, REASON_NO_SOURCE, REASON_OTHER_PLACE, WorkDataNeed  # noqa: E402
from gui import system_tools as ST  # noqa: E402
from tests.test_workdata import make_cache, make_source  # noqa: E402


class _WinCase(unittest.TestCase):
    """ウィンドウはクラスごとに 1 つ。テストごとに、開いているファイル・置き場所・確認の窓を元へ戻す"""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)
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

    @staticmethod
    def _pump(n=5):
        for _ in range(n):
            QApplication.processEvents()

    def setUp(self):
        win = self.win
        self.tmp = tempfile.mkdtemp(prefix="e57_systools_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self._saved = {
            "scans": win.current_scans, "ask_delete": ST.HOOKS.ask_delete, "ask_low_space": ST.HOOKS.ask_low_space,
            "question": QMessageBox.question, "information": QMessageBox.information, "warning": QMessageBox.warning,
        }
        self.addCleanup(self._restore)
        self.shown = []  # 出た案内（種類, 文）
        QMessageBox.information = staticmethod(lambda *a, **k: self.shown.append(("information", a[2] if len(a) > 2 else "")) or QMessageBox.Ok)
        QMessageBox.warning = staticmethod(lambda *a, **k: self.shown.append(("warning", a[2] if len(a) > 2 else "")) or QMessageBox.Ok)
        ST.HOOKS.ask_delete = lambda *a: self.fail("確認の窓を出す場面ではありません")
        ST.HOOKS.ask_low_space = lambda *a: self.fail("空きの確認を出す場面ではありません")
        self._set_input("")

    def _restore(self):
        win = self.win
        win.current_scans = self._saved["scans"]
        ST.HOOKS.ask_delete = self._saved["ask_delete"]
        ST.HOOKS.ask_low_space = self._saved["ask_low_space"]
        QMessageBox.question = self._saved["question"]
        QMessageBox.information = self._saved["information"]
        QMessageBox.warning = self._saved["warning"]
        ST.set_saved_workdata_dir("")
        SC.set_default_cache_root(None)
        win.__dict__.pop("_is_background_busy", None)
        if win.chk_lod.isChecked():
            win.chk_lod.setChecked(False)
        self._set_input("")

    def _set_input(self, text: str) -> None:
        for edit in (self.win.input_edit, self.win.output_edit):
            edit.blockSignals(True)
            edit.setText(text if edit is self.win.input_edit else "")
            edit.blockSignals(False)

    def _cache(self, name: str, folder: str = "a", **kw):
        src = make_source(os.path.join(self.tmp, folder), name)
        size = make_cache(src + ".cache", src, **kw)
        return src, size

    def _open(self, src: str) -> None:
        """src を「今開いているファイル」にする（読み込みはしない）"""
        self.win.current_scans = [SimpleNamespace(source_file=src, source_scan_index=0, point_count=10, original_point_count=10)]

    def _dialog(self) -> "ST.WorkDataDialog":
        dlg = ST.WorkDataDialog(self.win)
        self.addCleanup(dlg.deleteLater)
        dlg.refresh(sync=True)
        return dlg


class TestSystemTab(_WinCase):
    def test_rows_are_in_the_system_tab(self):
        win = self.win
        for name in ("lbl_workdata_summary", "btn_workdata_manage", "lbl_build_info"):
            w = getattr(win, name)
            self.assertTrue(win.tab_system.isAncestorOf(w), name)
        self.assertEqual(win.btn_workdata_manage.property("role"), "soft", "使う回数の少ない操作は淡い塗り")
        self.assertTrue(win.btn_workdata_manage.toolTip())
        self.assertEqual(win.lbl_build_info.text(), f"版: {get_build_info().describe()}")

    def test_title_carries_the_version(self):
        win = self.win
        tag = get_build_info().title_tag()
        base = win.windowTitle()
        self.assertTrue(base.startswith(ST.APP_TITLE))
        self.assertIn(tag, base, "画面の写しだけで、どの版かが分かる")
        win._set_dirty(True)
        self.assertTrue(win.windowTitle().startswith("* "))
        self.assertIn(tag, win.windowTitle())
        win._set_dirty(False)
        self.assertEqual(win.windowTitle(), base)

    def test_summary_shows_size_and_free_space(self):
        src, size = self._cache("現場.e57")
        self._set_input(src)
        ST.refresh_workdata_summary(self.win, sync=True)
        text = self.win.lbl_workdata_summary.text()
        self.assertIn(f"作業用データ: {format_bytes(size)}", text)
        self.assertRegex(text, r"空き [0-9.]+(GB|MB)")
        self.assertIn("1 件", self.win.lbl_workdata_summary.toolTip())
        # 何も無いとき
        self._set_input("")
        ST.refresh_workdata_summary(self.win, sync=True)
        self.assertIn("作業用データ: なし", self.win.lbl_workdata_summary.text())

    def test_summary_is_refreshed_when_the_tab_is_shown(self):
        """設定 → システムを開いたときに調べる（別スレッド。画面を止めない）"""
        win = self.win
        src, size = self._cache("現場.e57")
        self._set_input(src)
        win.lbl_workdata_summary.setText("古い表示")
        win._workdata_show_watcher._last = 0.0
        win._workdata_show_watcher.eventFilter(win.tab_system, QShowEvent())
        deadline = time.monotonic() + 5.0
        while win.lbl_workdata_summary.text() == "古い表示" and time.monotonic() < deadline:
            QApplication.processEvents()
            time.sleep(0.01)
        self.assertIn(format_bytes(size), win.lbl_workdata_summary.text())

    def test_about_text_can_be_pasted_into_a_report(self):
        text = ST.about_text()
        self.assertIn(f"版: {get_build_info().describe()}", text)
        self.assertIn("ログの場所: ", text)
        self.assertIn("作業用データの置き場所: 元ファイルの横（標準）", text)
        QTimer.singleShot(50, lambda: self.win._about_dialog.btn_copy.click())
        QTimer.singleShot(150, lambda: self.win._about_dialog.accept())
        ST.show_about_dialog(self.win)
        self.assertEqual(QApplication.clipboard().text(), ST.about_text())


class TestWorkDataDialog(_WinCase):
    def test_lists_entries_with_their_state(self):
        src_a, size_a = self._cache("開いている.e57", xyz_bytes=9000)
        src_b, size_b = self._cache("ほかの.e57", xyz_bytes=5000)
        src_c, size_c = self._cache("消した.e57", xyz_bytes=2000)
        os.remove(src_c)
        self._open(src_a)
        dlg = self._dialog()
        self.assertEqual([c.lbl_name.full_text() for c in dlg.cards], ["開いている.e57", "ほかの.e57", "消した.e57"], "大きい順")
        self.assertEqual([c.lbl_size.text() for c in dlg.cards], [format_bytes(s) for s in (size_a, size_b, size_c)])
        self.assertEqual([c.state_text for c in dlg.cards], ["今開いているファイル", "", REASON_NO_SOURCE])
        self.assertEqual([c.lbl_state.isHidden() for c in dlg.cards], [False, True, False], "状態は文字で出す")
        self.assertIn("読み直し用 + 高精細表示用", dlg.cards[0].lbl_detail.text())
        self.assertEqual(dlg.lbl_total.text(), f"3 件・合計 {format_bytes(size_a + size_b + size_c)}")
        self.assertRegex(dlg.lbl_drive.text(), r"の空き [0-9.]+(GB|MB) / 全体 [0-9.]+(GB|MB)")
        self.assertTrue(dlg.lbl_empty.isHidden())
        self.assertTrue(dlg.btn_delete_unused.isEnabled())
        self.assertIn(f"（1 件・{format_bytes(size_c)}）", dlg.btn_delete_unused.text())
        for c in dlg.cards:
            self.assertEqual(c.btn_delete.property("role"), "danger", "削除は赤 + ごみ箱")
            self.assertEqual(c.btn_delete.property("icon_name"), "trash")
        self.assertEqual(dlg.lbl_location.full_text(), "元ファイルの横（標準）")
        self.assertFalse(dlg.btn_location_reset.isEnabled())
        self.assertTrue(dlg.btn_location_reset.property("disabled_reason"), "無効のボタンには理由を添える")

    def test_empty_state_tells_how_data_gets_created(self):
        dlg = self._dialog()
        self.assertEqual(dlg.cards, [])
        self.assertFalse(dlg.lbl_empty.isHidden())
        self.assertIn("高精細表示", dlg.lbl_empty.text())
        self.assertEqual(dlg.lbl_total.text(), "作業用データなし")
        self.assertFalse(dlg.btn_delete_unused.isEnabled())
        self.assertTrue(dlg.btn_delete_unused.property("disabled_reason"))

    def test_delete_asks_first_and_keeps_the_data_on_cancel(self):
        src, _ = self._cache("現場.e57")
        self._set_input(src)
        dlg = self._dialog()
        asked = []
        ST.HOOKS.ask_delete = lambda parent, title, text, info: asked.append((text, info)) or False
        self.assertFalse(dlg.delete_entry(dlg.entries[0]))
        self.assertTrue(os.path.isdir(src + ".cache"))
        self.assertEqual(len(dlg.cards), 1)
        self.assertIn("現場.e57", asked[0][0])
        self.assertIn(dlg.cards[0].lbl_size.text(), asked[0][0], "消す大きさを見せる")
        self.assertIn("変わりません", asked[0][1], "元の E57 と保存した結果は変わらないと伝える")

    def test_delete_removes_the_data_and_reports_the_freed_space(self):
        src, size = self._cache("現場.e57")
        self._set_input(src)
        dlg = self._dialog()
        ST.HOOKS.ask_delete = lambda *a: True
        self.assertTrue(dlg.delete_entry(dlg.entries[0]))
        self.assertFalse(os.path.exists(src + ".cache"))
        self.assertTrue(os.path.isfile(src), "元ファイルには触らない")
        self.assertEqual(dlg.cards, [])
        self.assertFalse(dlg.lbl_empty.isHidden())
        if hasattr(self.win, "toast_label"):
            self.assertIn("空けました", self.win.toast_label.text())
        self.assertEqual(self.shown, [], "うまく消せたときは、注意の窓を出さない")

    def test_a_late_background_result_does_not_bring_back_deleted_data(self):
        """窓を開いたときに別スレッドで調べ始めた結果が、削除のあとに届いても、消したものを一覧へ戻さない"""
        src, _ = self._cache("現場.e57")
        self._set_input(src)
        dlg = ST.WorkDataDialog(self.win)      # ここで別スレッドの調べが始まる
        self.addCleanup(dlg.deleteLater)
        dlg.refresh(sync=True)
        self.assertEqual(len(dlg.cards), 1)
        ST.HOOKS.ask_delete = lambda *a: True
        self.assertTrue(dlg.delete_entry(dlg.entries[0]))
        deadline = time.monotonic() + 0.4
        while time.monotonic() < deadline:
            QApplication.processEvents()
            time.sleep(0.01)
        self.assertEqual(dlg.cards, [])
        self.assertEqual(dlg.lbl_total.text(), "作業用データなし")

    def test_data_deleted_elsewhere_just_disappears(self):
        """窓を開いている間にエクスプローラーで消されていたら、注意を出さずに一覧から外す"""
        src, _ = self._cache("現場.e57")
        self._set_input(src)
        dlg = self._dialog()
        shutil.rmtree(src + ".cache")
        ST.HOOKS.ask_delete = lambda *a: True
        self.assertTrue(dlg.delete_entry(dlg.entries[0]))
        self.assertEqual(dlg.cards, [])
        self.assertEqual(self.shown, [])

    def test_deleting_the_open_files_data_turns_lod_off(self):
        src, _ = self._cache("開いている.e57")
        self._open(src)
        win = self.win
        win.chk_lod.setChecked(True)  # 点群の無い間はチェックだけ覚える
        self.assertTrue(win.chk_lod.isChecked())
        dlg = self._dialog()
        asked = []
        ST.HOOKS.ask_delete = lambda parent, title, text, info: asked.append(info) or True
        self.assertTrue(dlg.delete_entry(dlg.entries[0]))
        self.assertIn("高精細表示を切ります", asked[0])
        self.assertFalse(win.chk_lod.isChecked())
        self.assertFalse(os.path.exists(src + ".cache"))

    def test_open_files_data_cannot_be_deleted_while_working(self):
        src_open, _ = self._cache("開いている.e57", xyz_bytes=9000)
        src_other, _ = self._cache("ほかの.e57", xyz_bytes=1000)
        self._open(src_open)
        dlg = self._dialog()
        self.win._is_background_busy = lambda: True
        ST.HOOKS.ask_delete = lambda *a: True
        self.assertFalse(dlg.delete_entry(dlg.entries[0]), "作っている最中かもしれないデータは消さない")
        self.assertTrue(os.path.isdir(src_open + ".cache"))
        self.assertEqual(self.shown[-1][0], "information")
        self.assertIn("終わってから", self.shown[-1][1])
        # 開いていないファイルのデータは、処理中でも消せる
        self.assertTrue(dlg.delete_entry(dlg.entries[1]))
        self.assertFalse(os.path.exists(src_other + ".cache"))
        self.assertFalse(dlg.change_location(self.tmp), "置き場所も、処理が終わってから")

    def test_delete_unused_in_one_go(self):
        keep_src, _ = self._cache("使う.e57")
        gone_src, size_gone = self._cache("消した.e57", xyz_bytes=3000)
        os.remove(gone_src)
        old_src, size_old = self._cache("古い.e57", version=SC.CACHE_VERSION - 1, lod=False, xyz_bytes=2000)
        self._set_input(keep_src)
        dlg = self._dialog()
        asked = []
        ST.HOOKS.ask_delete = lambda parent, title, text, info: asked.append((text, info)) or True
        self.assertTrue(dlg.delete_unused())
        self.assertIn(f"2 件（{format_bytes(size_gone + size_old)}）", asked[0][0])
        self.assertIn("消した.e57", asked[0][1])
        self.assertIn("古い.e57", asked[0][1])
        self.assertTrue(os.path.isdir(keep_src + ".cache"), "使われるものは残す")
        self.assertFalse(os.path.exists(gone_src + ".cache"))
        self.assertFalse(os.path.exists(old_src + ".cache"))
        self.assertEqual([c.lbl_name.full_text() for c in dlg.cards], ["使う.e57"])
        self.assertFalse(dlg.btn_delete_unused.isEnabled())
        self.assertFalse(dlg.delete_unused(), "もう無ければ何もしない")

    def test_a_folder_with_other_files_is_reported_not_destroyed(self):
        src, _ = self._cache("現場.e57")
        with open(os.path.join(src + ".cache", "大事なメモ.txt"), "w", encoding="utf-8") as f:
            f.write("消さないで")
        self._set_input(src)
        dlg = self._dialog()
        ST.HOOKS.ask_delete = lambda *a: True
        dlg.delete_entry(dlg.entries[0])
        self.assertTrue(os.path.isfile(os.path.join(src + ".cache", "大事なメモ.txt")))
        self.assertEqual(self.shown[-1][0], "warning")
        self.assertIn("大事なメモ.txt", self.shown[-1][1])

    def test_change_location_and_back(self):
        src, size = self._cache("現場.e57")
        self._set_input(src)
        root = os.path.join(self.tmp, "作業用")
        os.makedirs(root)
        win = self.win
        win.chk_lod.setChecked(True)
        dlg = self._dialog()
        self.assertTrue(dlg.change_location(root))
        self.assertEqual(os.path.normcase(ST.get_saved_workdata_dir()), os.path.normcase(os.path.abspath(root)), "その場で保存する")
        self.assertEqual(os.path.normcase(SC.get_default_cache_root()), os.path.normcase(os.path.abspath(root)))
        self.assertEqual(os.path.normcase(dlg.lbl_location.full_text()), os.path.normcase(os.path.abspath(root)))
        self.assertTrue(dlg.btn_location_reset.isEnabled())
        self.assertFalse(win.chk_lod.isChecked(), "置き場所が変わると今のデータは使えないので、高精細表示を切る")
        # 元ファイルの横にあったものは、もう使われない（まとめて消せる）
        self.assertEqual(dlg.cards[0].state_text, REASON_OTHER_PLACE)
        self.assertIn(f"（1 件・{format_bytes(size)}）", dlg.btn_delete_unused.text())
        self.assertIn(os.path.abspath(root), ST.about_text())
        self.assertFalse(dlg.change_location(root), "同じ場所なら何もしない")
        # 元に戻す
        self.assertTrue(dlg.change_location(""))
        self.assertEqual(ST.get_saved_workdata_dir(), "")
        self.assertIsNone(SC.get_default_cache_root())
        self.assertEqual(dlg.cards[0].state_text, "")
        self.assertEqual(dlg.lbl_location.full_text(), "元ファイルの横（標準）")

    def test_location_must_be_writable(self):
        dlg = self._dialog()
        self.assertFalse(dlg.change_location(os.path.join(self.tmp, "無いフォルダ")))
        self.assertEqual(self.shown[-1][0], "warning")
        self.assertIn("書き込めません", self.shown[-1][1])
        self.assertEqual(ST.get_saved_workdata_dir(), "")
        self.assertIsNone(SC.get_default_cache_root())

    def test_saved_location_is_applied_at_startup(self):
        root = os.path.join(self.tmp, "作業用")
        os.makedirs(root)
        ST.set_saved_workdata_dir(root)
        self.assertEqual(ST.apply_workdata_dir_setting(), root)
        self.assertEqual(os.path.normcase(SC.get_default_cache_root()), os.path.normcase(os.path.abspath(root)))
        # 外付けドライブを外しているだけかもしれないので、見つからなくても設定は消さない
        shutil.rmtree(root)
        self.assertEqual(ST.apply_workdata_dir_setting(), root)
        self.assertEqual(ST.get_saved_workdata_dir(), root)
        dlg = self._dialog()
        self.assertIn("見つかりません", dlg.lbl_location.full_text())
        ST.refresh_workdata_summary(self.win, sync=True)
        self.assertIn("置き場所が見つかりません", self.win.lbl_workdata_summary.text(), "設定 → システムの行でも知らせる")
        ST.set_saved_workdata_dir("")
        self.assertEqual(ST.apply_workdata_dir_setting(), "")
        self.assertIsNone(SC.get_default_cache_root())


class TestSpaceConfirm(_WinCase):
    def setUp(self):
        super().setUp()
        self._orig = {name: getattr(ST, name) for name in ("check_free", "estimate_save_bytes", "open_workdata_dialog", "workdata_need_for_scans")}
        self.addCleanup(lambda: [setattr(ST, k, v) for k, v in self._orig.items()])
        self.out = os.path.join(self.tmp, "出力.e57")
        self.mask = np.array([True, True, False, True])
        self.scans = [SimpleNamespace(source_file=None, source_scan_index=0, point_count=4, original_point_count=4)]

    def _low(self, ok_after: int = 10 ** 9):
        """check_free を「足りない」にする。ok_after 回目からは足りる"""
        calls = []

        def fake(path, need, keep_free=0):
            calls.append((path, need))
            return FreeCheck(ok=len(calls) > ok_after, need_bytes=int(need), free_bytes=3 * GB, drive="C:")

        ST.check_free = fake
        return calls

    def test_save_goes_ahead_without_asking_when_there_is_room(self):
        self.assertTrue(ST.confirm_save_free_space(self.win, self.out, self.scans, self.mask, None, False))

    def test_save_low_space_cancel_and_proceed(self):
        self._low()
        ST.estimate_save_bytes = lambda scans, n_out, fullres: 16 * GB
        asked = []
        ST.HOOKS.ask_low_space = lambda parent, title, text, info, allow: asked.append((text, info, allow)) or "cancel"
        self.assertFalse(ST.confirm_save_free_space(self.win, self.out, self.scans, self.mask, None, True))
        text, info, allow = asked[0]
        self.assertIn("C:", text)
        self.assertIn("約 16.0GB", info)
        self.assertIn("空き: 3.0GB", info)
        self.assertIn("今あるファイルは変わりません", info)
        self.assertTrue(allow, "見込みなので、このまま保存も選べる")
        ST.HOOKS.ask_low_space = lambda *a: "proceed"
        self.assertTrue(ST.confirm_save_free_space(self.win, self.out, self.scans, self.mask, None, True))

    def test_save_low_space_then_room_after_tidying(self):
        calls = self._low(ok_after=1)
        opened = []
        ST.open_workdata_dialog = lambda win: opened.append(win)
        ST.HOOKS.ask_low_space = lambda *a: "manage"
        self.assertTrue(ST.confirm_save_free_space(self.win, self.out, self.scans, self.mask, None, False))
        self.assertEqual(opened, [self.win], "作業用データの管理を開く")
        self.assertEqual(len(calls), 2, "整理のあとで、もう一度確かめる")

    def test_save_counts_the_points_to_write(self):
        seen = []
        ST.estimate_save_bytes = lambda scans, n_out, fullres: seen.append((n_out, fullres)) or 1
        ST.confirm_save_free_space(self.win, self.out, self.scans, self.mask, None, True)
        split = {"床": np.array([True, False, False, False]), "壁": np.array([False, True, True, False])}
        ST.confirm_save_free_space(self.win, self.out, self.scans, self.mask, split, False)
        self.assertEqual(seen, [(3, True), (3, False)], "分割保存は、レイヤーごとの点数の合計")

    def test_save_is_not_blocked_when_the_size_cannot_be_estimated(self):
        def boom(*a):
            raise ValueError("見積もれない")

        ST.estimate_save_bytes = boom
        self.assertTrue(ST.confirm_save_free_space(self.win, self.out, self.scans, self.mask, None, False))

    def test_lod_build_shows_size_place_and_free_space(self):
        src = make_source(os.path.join(self.tmp, "a"), "現場.e57")
        scans = [SimpleNamespace(source_file=src, source_scan_index=0, point_count=100, original_point_count=2_000_000)]
        asked = []

        def answer(value):
            def q(*a, **k):
                asked.append(a[2])
                return value
            return staticmethod(q)

        QMessageBox.question = answer(QMessageBox.Yes)
        self.assertTrue(ST.confirm_lod_build(self.win, scans))
        text = asked[0]
        self.assertIn("2,000,000 点", text)
        self.assertRegex(text, r"大きさ: 約 [0-9.]+MB（読み直し用 [0-9.]+MB \+ 高精細表示用 [0-9.]+MB）")
        self.assertIn(os.path.dirname(os.path.abspath(src)), text)
        self.assertIn("（元ファイルの横）", text)
        self.assertRegex(text, r"の空き: [0-9.]+(GB|MB)")
        self.assertIn("作業用データ", text, "あとで消せる場所を伝える")
        QMessageBox.question = answer(QMessageBox.No)
        self.assertFalse(ST.confirm_lod_build(self.win, scans))
        # 読み直し用がすでにあれば、そのぶんは数えない
        make_cache(src + ".cache", src, lod=False)
        QMessageBox.question = answer(QMessageBox.Yes)
        ST.confirm_lod_build(self.win, scans)
        self.assertIn("読み直し用は作成済み", asked[-1])

    def test_lod_build_does_not_start_without_room(self):
        low = WorkDataNeed(n_points=538_377_435, reload_bytes=9 * GB, lod_bytes=8 * GB, need_free_bytes=20 * GB,
                           directory=os.path.join(self.tmp, "x.e57.cache"), free_bytes=3 * GB, drive="C:")
        ok = WorkDataNeed(n_points=538_377_435, reload_bytes=9 * GB, lod_bytes=8 * GB, need_free_bytes=20 * GB,
                          directory=os.path.join(self.tmp, "x.e57.cache"), free_bytes=40 * GB, drive="C:")
        answers = [low]
        ST.workdata_need_for_scans = lambda scans: answers[0]
        asked = []
        ST.HOOKS.ask_low_space = lambda parent, title, text, info, allow: asked.append((text, info, allow)) or "cancel"
        QMessageBox.question = staticmethod(lambda *a, **k: self.fail("空きが足りないのに、作成を尋ねてはいけません"))
        self.assertFalse(ST.confirm_lod_build(self.win, self.scans))
        text, info, allow = asked[0]
        self.assertIn("空きが足りません", text)
        self.assertIn("約 17.0GB", info)
        self.assertIn("約 20.0GB", info)
        self.assertIn("空き: 3.0GB", info)
        self.assertFalse(allow, "足りないまま始める選択肢は出さない（数分かけてから失敗するため）")
        # 整理して空きができたら、作成を尋ねる
        opened = []

        def tidy(win):
            opened.append(win)
            answers[0] = ok

        ST.open_workdata_dialog = tidy
        ST.HOOKS.ask_low_space = lambda *a: "manage"
        QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
        self.assertTrue(ST.confirm_lod_build(self.win, self.scans))
        self.assertEqual(opened, [self.win])


class TestWiring(_WinCase):
    def test_save_checks_free_space_before_writing(self):
        """保存は、書き始める前に空きの確認を通る（足りなくてやめたら、何も始めない）"""
        import gui.main_window as MW

        win = self.win
        rng = np.random.default_rng(5)
        win.viewer.set_point_cloud(rng.random((500, 3)) * 10.0)
        self._pump()
        scans = [SimpleNamespace(source_file=None, source_scan_index=0, point_count=500, original_point_count=500, is_subsampled=False)]
        win.current_scans = scans
        out = os.path.join(self.tmp, "出力.e57")
        win.output_edit.blockSignals(True)
        win.output_edit.setText(out)
        win.output_edit.blockSignals(False)
        seen = []
        orig = MW.confirm_save_free_space
        MW.confirm_save_free_space = lambda *a: seen.append(a) or False
        try:
            win._save_current_scans()
        finally:
            MW.confirm_save_free_space = orig
            win.viewer.set_point_cloud(np.zeros((0, 3)))
            win._set_dirty(False)
        self.assertEqual(len(seen), 1)
        got_win, got_out, got_scans, got_mask, got_split, got_fullres = seen[0]
        self.assertIs(got_win, win)
        self.assertEqual(got_out, out)
        self.assertIs(got_scans, scans)
        self.assertEqual(int(np.count_nonzero(got_mask)), 500)
        self.assertIsNone(got_split)
        self.assertFalse(got_fullres)
        self.assertFalse(win.save_worker is not None and win.save_worker.isRunning(), "やめたら保存を始めない")
        self.assertTrue(win.save_manual_btn.isEnabled())
        self.assertFalse(os.path.exists(out))

    def test_commands_and_menu_reach_the_tools(self):
        from gui.commands import build_commands

        ids = {c.id for c in build_commands(None)}
        self.assertIn("help.workdata", ids)
        self.assertIn("help.about", ids)
        win = self.win
        if getattr(win, "ui_mode", "") != "modern":
            self.skipTest("メニューは新UIだけ")
        texts = []
        for act in win.menuBar().actions():
            if act.menu() is not None:
                texts += [a.text() for a in act.menu().actions()]
        self.assertIn("作業用データの管理...", texts)
        self.assertIn("バージョン情報", texts)

    def test_system_tab_hint_no_longer_points_to_the_app_folder(self):
        """配布版のログは EXE のフォルダの外へ移った。「logs/ フォルダ」と書いたままにしない"""
        from PySide6.QtWidgets import QLabel

        texts = [lbl.text() for lbl in self.win.tab_system.findChildren(QLabel)]
        self.assertTrue(any("ログのフォルダ" in t for t in texts))
        self.assertFalse(any("logs/" in t for t in texts))


if __name__ == "__main__":
    unittest.main()
