# -*- coding: utf-8 -*-
"""
長い処理の完了の知らせ（gui/task_notify.py）のテスト

- 進捗の窓が出てから消えるまでを測り、長くかかった処理の終わりを、アプリが手前に無いときだけ知らせること
- 短い処理・アプリを見ているとき・設定が切のときは知らせないこと
- 続けて出る進捗の窓は、ひと続きの処理として 1 回だけ知らせること
- 知らせの文は、完了やエラーの窓 → 画面の中の通知 → ステータスの順で選ぶこと
- 本物の裏の処理（_run_in_background）と、本物の読み込みでも働くこと
- Windows の通知の出し入れ（通知領域の絵は、利用者がアプリへ戻ったら片付ける）
- テストのあいだは、本物の通知を出さないこと
"""
import os
import shutil
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.settings_sandbox import use_sandbox_settings  # noqa: E402

use_sandbox_settings()  # この PC の本番アプリの保存設定を、テストで書き換えない

from PySide6.QtCore import QCoreApplication, QEvent, Qt  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox, QSystemTrayIcon  # noqa: E402

from gui import task_notify as TN  # noqa: E402
from gui.progress_dialog import ModernProgressDialog  # noqa: E402
from tests.test_memguard import LAYOUT_FLS, write_synthetic_e57  # noqa: E402


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _WinCase(unittest.TestCase):
    """ウィンドウはクラスごとに 1 つ。知らせの出口・時計・アプリが手前かどうかを、テストごとに差し替える"""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)
        from gui.main_window import MainWindow

        cls.win = MainWindow()
        cls.win._reset_app_settings()
        cls.win.resize(1450, 900)
        cls.win.show()
        cls._pump(50)

    @classmethod
    def tearDownClass(cls):
        cls.win._set_dirty(False)
        cls.win.close()
        cls.win.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        QApplication.processEvents()

    @staticmethod
    def _pump(ms: int = 30):
        end = time.monotonic() + ms / 1000.0
        while time.monotonic() < end:
            QApplication.processEvents()
            time.sleep(0.002)

    def setUp(self):
        self.clock = FakeClock()
        self.sent = []      # (題名, 文, 種類)
        self.active = False  # アプリが手前にあるか
        self._saved = (TN.HOOKS.notify, TN.HOOKS.is_app_active, TN.HOOKS.clock, TN.get_notify_enabled())
        TN.HOOKS.notify = lambda win, title, body, kind: self.sent.append((title, body, kind))
        TN.HOOKS.is_app_active = lambda: self.active
        TN.HOOKS.clock = self.clock
        TN.set_notify_enabled(True)
        self.addCleanup(self._restore)
        self.win.status_label.setText("")
        self.win._toast_timer.stop()
        self.win.toast_label.setVisible(False)
        host = getattr(self.win, "toast_host", None)
        if host is not None:
            host.setVisible(False)

    def _restore(self):
        TN.HOOKS.notify, TN.HOOKS.is_app_active, TN.HOOKS.clock = self._saved[:3]
        TN.set_notify_enabled(self._saved[3])
        dlg = self.win.progress_dialog
        if dlg is not None:
            dlg.close()
            self.win.progress_dialog = None
        self._settle()

    def _settle(self):
        """進捗の窓が消えたあとの待ち（SETTLE_MS）が明けるまで回す"""
        self._pump(TN.SETTLE_MS + 150)

    def _progress(self, title: str = "点群データ読み込み中") -> ModernProgressDialog:
        dlg = ModernProgressDialog(title, self.win)
        self.win.progress_dialog = dlg
        dlg.show()
        self._pump(20)
        return dlg

    def _finish(self, dlg: ModernProgressDialog, status: str = "") -> None:
        dlg.close()
        self.win.progress_dialog = None
        if status:
            self.win.status_label.setText(status)


class TestWhenToNotify(_WinCase):
    def test_long_task_finished_while_away_is_notified(self):
        dlg = self._progress()
        self.clock.advance(192.0)
        self._finish(dlg, "読み込み完了: 1 スキャン / 計 59,819,715 点")
        self.assertEqual(self.sent, [], "窓が消えた直後には、まだ知らせない（続きの窓が出るかもしれない）")
        self._settle()
        self.assertEqual(len(self.sent), 1)
        title, body, kind = self.sent[0]
        self.assertEqual(title, TN.NOTIFY_TITLE)
        self.assertEqual(body, "読み込み完了: 1 スキャン / 計 59,819,715 点（3 分 12 秒）")
        self.assertEqual(kind, "success")
        self.assertEqual(self.win._long_task_watch.last_notice[0], body)

    def test_short_task_is_not_notified(self):
        dlg = self._progress()
        self.clock.advance(TN.NOTIFY_MIN_SECONDS - 1.0)
        self._finish(dlg, "処理完了")
        self._settle()
        self.assertEqual(self.sent, [])

    def test_not_notified_while_the_user_is_watching(self):
        self.active = True
        dlg = self._progress()
        self.clock.advance(600.0)
        self._finish(dlg, "保存完了")
        self._settle()
        self.assertEqual(self.sent, [])

    def test_not_notified_when_switched_off(self):
        TN.set_notify_enabled(False)
        dlg = self._progress()
        self.clock.advance(600.0)
        self._finish(dlg, "保存完了")
        self._settle()
        self.assertEqual(self.sent, [])

    def test_back_to_back_dialogs_are_one_task(self):
        first = self._progress("作業用データの作成")
        self.clock.advance(15.0)
        first.close()
        second = self._progress("高精細表示 (LOD) データの作成")  # すぐに続きの窓が出る
        self._settle()
        self.assertEqual(self.sent, [], "続きがあるあいだは知らせない")
        self.clock.advance(15.0)
        self._finish(second, "高精細表示の準備完了")
        self._settle()
        self.assertEqual(len(self.sent), 1)
        self.assertIn("（30 秒）", self.sent[0][1])  # 2 つの窓を通した時間

    def test_waits_while_the_worker_is_still_finishing(self):
        busy = {"on": True}
        self.win._is_background_busy = lambda: busy["on"]
        self.addCleanup(lambda: self.win.__dict__.pop("_is_background_busy", None))
        dlg = self._progress()
        self.clock.advance(60.0)
        self._finish(dlg, "処理完了")
        self._settle()
        self.assertEqual(self.sent, [], "裏の処理が終わるまで待つ")
        busy["on"] = False
        self._settle()
        self.assertEqual(len(self.sent), 1)

    def test_a_new_task_starts_a_new_measurement(self):
        dlg = self._progress()
        self.clock.advance(100.0)
        self._finish(dlg, "読み込み完了")
        self._settle()
        self.clock.advance(5000.0)  # 利用者が長く作業したあと
        dlg = self._progress("自動ノイズフィルタ処理中")
        self.clock.advance(5.0)
        self._finish(dlg, "処理完了")
        self._settle()
        self.assertEqual(len(self.sent), 1, "2 つ目は 5 秒なので知らせない（前の処理からの時間で測らない）")

    def test_dialog_of_another_window_is_ignored(self):
        other = QDialog()
        self.addCleanup(other.deleteLater)
        dlg = ModernProgressDialog("別の窓の処理", other)
        dlg.show()
        self._pump(20)
        self.clock.advance(300.0)
        dlg.close()
        self._settle()
        self.assertEqual(self.sent, [])


class TestOutcomeMessage(_WinCase):
    def test_error_box_wins_and_is_an_error(self):
        dlg = self._progress("E57点群データ保存中")
        self.clock.advance(400.0)
        self._finish(dlg, "保存失敗")
        box = QMessageBox(QMessageBox.Critical, "保存エラー", "保存中にエラーが発生しました:\nディスクの空きが足りません", QMessageBox.Ok, self.win)
        box.setModal(True)
        box.show()
        self.addCleanup(box.deleteLater)
        try:
            self._settle()
        finally:
            box.close()
        self.assertEqual(len(self.sent), 1)
        self.assertEqual(self.sent[0][1], "保存中にエラーが発生しました: ディスクの空きが足りません（6 分 40 秒）")
        self.assertEqual(self.sent[0][2], "error")

    def test_completion_box_text_is_used(self):
        dlg = self._progress("E57点群データ保存中（全点転写）")
        self.clock.advance(573.0)
        self._finish(dlg, "保存完了")
        box = QMessageBox(QMessageBox.Information, "保存完了", "点群データをE57ファイルに保存しました！\n\n・出力点数: 538,339,020 点\n・除外点数: 38,415 点",
                          QMessageBox.Ok, self.win)
        box.setModal(True)
        box.show()
        self.addCleanup(box.deleteLater)
        try:
            self._settle()
        finally:
            box.close()
        self.assertEqual(self.sent[0][1], "点群データをE57ファイルに保存しました！ ・出力点数: 538,339,020 点（9 分 33 秒）")
        self.assertEqual(self.sent[0][2], "success")

    def test_toast_is_preferred_over_the_status_line(self):
        dlg = self._progress()
        self.clock.advance(45.0)
        self._finish(dlg, "大容量保護読み込み完了: 1 スキャン")
        self.win.show_toast("⚡ 大容量保護: 59,819,715 点を高速表示 (メモリ圧迫防止)")
        self._settle()
        self.assertEqual(self.sent[0][1], "大容量保護: 59,819,715 点を高速表示 (メモリ圧迫防止)（45 秒）")

    def test_warning_words_make_it_a_warning(self):
        dlg = self._progress()
        self.clock.advance(45.0)
        self._finish(dlg, "点群が変わったため結果を破棄しました")
        self._settle()
        self.assertEqual(self.sent[0][2], "warning")

    def test_fallback_text_when_nothing_is_shown(self):
        dlg = self._progress()
        self.clock.advance(45.0)
        self._finish(dlg)
        self._settle()
        self.assertEqual(self.sent[0][1], "処理が終わりました（45 秒）")
        self.assertEqual(self.sent[0][2], "info")

    def test_long_text_is_cut(self):
        dlg = self._progress()
        self.clock.advance(45.0)
        self._finish(dlg, "保存先: " + "とても長いフォルダ名/" * 40)
        self._settle()
        body = self.sent[0][1]
        self.assertLessEqual(len(body), TN.MAX_BODY_CHARS + len("（45 秒）"))
        self.assertIn("…（45 秒）", body)


class TestRealTasks(_WinCase):
    """本物の裏の処理と、本物の読み込みで確かめる（時計だけを進める）"""

    def test_run_in_background_task(self):
        win = self.win

        def work(seconds: float):
            time.sleep(seconds)
            return 42

        def done(result):
            win.status_label.setText(f"解析が終わりました: {result}")

        self.assertTrue(win._run_in_background("テスト用の解析", "計算中...", work, 0.15, on_success=done))
        self.clock.advance(75.0)
        self.assertTrue(win._wait_for_background(20000))
        self._pump(300)
        self._settle()
        self.assertEqual(len(self.sent), 1, self.sent)
        self.assertEqual(self.sent[0][1], "解析が終わりました: 42（1 分 15 秒）")

    def test_real_file_load(self):
        tmp = tempfile.mkdtemp(prefix="e57_notify_")
        self.addCleanup(shutil.rmtree, tmp, True)
        path = os.path.join(tmp, "現場.e57")
        write_synthetic_e57(path, LAYOUT_FLS, (3000,), pose=None)
        win = self.win
        self.assertIsNot(win._load_file_to_viewer(path), False)
        self.clock.advance(130.0)
        end = time.monotonic() + 30
        while (win.load_worker.isRunning() or win.progress_dialog is not None) and time.monotonic() < end:
            self._pump(20)
        self._pump(300)
        self._settle()
        self.assertEqual(len(win.viewer.points_raw), 3000)
        self.assertEqual(len(self.sent), 1, self.sent)
        body, kind = self.sent[0][1], self.sent[0][2]
        self.assertIn("3,000 点", body)
        self.assertIn("（2 分 10 秒）", body)
        self.assertNotEqual(kind, "error")
        # 後始末: 点群を空に戻す
        win.viewer.set_point_cloud(None)
        win.current_scans = []


class TestSettingCheckbox(_WinCase):
    def test_checkbox_is_in_the_system_tab_and_saves_at_once(self):
        win = self.win
        chk = win.chk_notify_long_task
        self.assertIs(chk.parentWidget(), win.tab_system)
        self.assertTrue(chk.isChecked())
        self.assertIn(str(int(TN.NOTIFY_MIN_SECONDS)), chk.toolTip())
        chk.setChecked(False)
        self.assertFalse(TN.get_notify_enabled())
        chk.setChecked(True)
        self.assertTrue(TN.get_notify_enabled())


class FakeSignal:
    def __init__(self):
        self.slots = []

    def connect(self, fn):
        self.slots.append(fn)

    def emit(self, *a):
        for fn in list(self.slots):
            fn(*a)


class FakeTray:
    def __init__(self):
        self.visible = False
        self.messages = []
        self.messageClicked = FakeSignal()
        self.activated = FakeSignal()

    def show(self):
        self.visible = True

    def hide(self):
        self.visible = False

    def showMessage(self, title, body, icon, msecs):
        assert self.visible, "通知領域に絵が出ていないと、通知は出ない"
        self.messages.append((title, body, icon, msecs))


class TestOsNotifier(_WinCase):
    def test_notification_goes_through_the_tray_and_is_cleared_on_return(self):
        tray = FakeTray()
        notifier = TN.OsNotifier(self.win, tray_factory=lambda: tray)
        self.addCleanup(notifier.deleteLater)
        self.assertTrue(notifier.notify("題名", "保存完了（9 分 33 秒）", "success"))
        self.assertTrue(tray.visible)
        self.assertEqual(tray.messages[0][:3], ("題名", "保存完了（9 分 33 秒）", QSystemTrayIcon.Information))
        notifier.notify("題名", "保存失敗（1 分 0 秒）", "error")
        self.assertEqual(tray.messages[1][2], QSystemTrayIcon.Critical)
        notifier.notify("題名", "中止しました（1 分 0 秒）", "warning")
        self.assertEqual(tray.messages[2][2], QSystemTrayIcon.Warning)
        # 利用者がアプリへ戻ったら、通知領域の絵を片付ける
        notifier._on_app_state(Qt.ApplicationInactive)
        self.assertTrue(tray.visible)
        notifier._on_app_state(Qt.ApplicationActive)
        self.assertFalse(tray.visible)

    def test_clicking_the_notification_brings_the_window_to_front(self):
        tray = FakeTray()
        notifier = TN.OsNotifier(self.win, tray_factory=lambda: tray)
        self.addCleanup(notifier.deleteLater)
        notifier.notify("題名", "文", "info")
        self.win.showMinimized()
        self._pump(50)
        tray.messageClicked.emit()
        self._pump(50)
        self.assertFalse(self.win.isMinimized())
        self.assertFalse(tray.visible)

    def test_without_a_tray_only_the_taskbar_is_used(self):
        notifier = TN.OsNotifier(self.win, tray_factory=lambda: None)
        self.addCleanup(notifier.deleteLater)
        self.assertFalse(notifier.notify("題名", "文", "info"))
        notifier.clear()  # 絵が無くても困らない

    def test_no_real_notification_during_tests(self):
        self.assertTrue(os.environ.get(TN.NO_OS_NOTIFY_ENV), "tests/settings_sandbox.py が入れるはず")
        self.win.__dict__.pop("_os_notifier", None)
        TN._notify_os(self.win, "題名", "文", "info")
        self.assertIsNone(getattr(self.win, "_os_notifier", None))


class TestFormatElapsed(unittest.TestCase):
    def test_format(self):
        self.assertEqual(TN.format_elapsed(0), "0 秒")
        self.assertEqual(TN.format_elapsed(59.4), "59 秒")
        self.assertEqual(TN.format_elapsed(59.6), "1 分 0 秒")
        self.assertEqual(TN.format_elapsed(192), "3 分 12 秒")
        self.assertEqual(TN.format_elapsed(3900), "1 時間 5 分")
        self.assertEqual(TN.format_elapsed(-5), "0 秒")


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
