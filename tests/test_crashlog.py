# -*- coding: utf-8 -*-
"""
異常終了の記録（core/crashlog.py）と、その画面側（gui/diagnostics.py）のテスト

- 起動中の印を置き、正常終了で消すこと
- 残った印から、正常に終了しなかった起動を見つけて記録（crash_*.txt）に残すこと
  （まだ動いているアプリの印には触らない。正常終了の書き込みがある印は黙って消す）
- 本当に落ちたプロセス（アクセス違反）・外から止められたプロセス・正常に終わったプロセスを、別プロセスで実際に作って確かめる
- 古い記録の片付け、印が大きくなりすぎたときの停止
- スレッドの中の例外がログへ残ること
- Qt の警告がログへ残ること（同じ文は 3 回まで）
- 前回の異常終了を、起動した画面で知らせること
"""
import faulthandler
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from datetime import date, datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.settings_sandbox import use_sandbox_settings  # noqa: E402

use_sandbox_settings()  # この PC の本番アプリの保存設定を、テストで書き換えない

from core import crashlog as CL  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 別プロセスで走らせる小さな台本。最後の引数で終わり方を選ぶ
CHILD_SCRIPT = r"""
import ctypes, os, sys, time
sys.path.insert(0, sys.argv[1])
from core import crashlog
if os.name == "nt":
    # 落ちたときに Windows の「動作を停止しました」の窓を出さない
    ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)
session = crashlog.begin_session(sys.argv[2], build="test build")
print("READY", session.path, flush=True)
how = sys.argv[3]
if how == "segv":
    import faulthandler
    faulthandler._sigsegv()
elif how == "hang":
    time.sleep(60)
elif how == "exit_without_end":
    os._exit(7)
elif how == "crash_during_exit":
    session.mark_clean_exit()   # アプリの終了処理は始まった
    import faulthandler
    faulthandler._sigsegv()     # そのあとの後始末で落ちた
else:
    session.end()
"""


def read_text(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def run_child(logs_dir: str, how: str, kill_after_ready: bool = False) -> int:
    proc = subprocess.Popen(
        [sys.executable, "-X", "utf8", "-c", CHILD_SCRIPT, ROOT, logs_dir, how],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace",
    )
    try:
        line = proc.stdout.readline()
        assert line.startswith("READY"), f"子プロセスが始まりませんでした: {line!r}"
        if kill_after_ready:
            proc.kill()
        return proc.wait(timeout=60)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)
        proc.stdout.close()


def wait_released(logs_dir: str, timeout: float = 5.0) -> None:
    """子プロセスが終わった直後は、Windows がファイルをまだ放していないことがある。印の名前を変えられるまで待つ"""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        busy = False
        for name in os.listdir(logs_dir):
            if name.startswith(CL.SESSION_PREFIX):
                p = os.path.join(logs_dir, name)
                try:
                    os.replace(p, p + ".probe")
                    os.replace(p + ".probe", p)
                except OSError:
                    busy = True
        if not busy:
            return
        time.sleep(0.05)


class _DirCase(unittest.TestCase):
    def setUp(self):
        self.logs = tempfile.mkdtemp(prefix="e57_crashlog_")
        self.addCleanup(shutil.rmtree, self.logs, True)

    def names(self, prefix: str = ""):
        return sorted(n for n in os.listdir(self.logs) if n.startswith(prefix))

    def leftover(self, pid: int = 4242, stamp: str = "20261114-100211", body: str = "", ended: bool = False) -> str:
        """前の起動が残した印を作る（開いているプロセスは無い）"""
        path = os.path.join(self.logs, f"{CL.SESSION_PREFIX}{pid}_{stamp}.txt")
        text = (
            "E57 点群ノイズ処理ツール: 起動中の印（正常に終了すると消えます）\n"
            "started: 2026-11-14 10:02:11\n"
            f"pid: {pid}\n"
            "build: 配布版 2026-11-14 00:50 作成（63a9d9e）\n"
            "exe: C:\\tool\\E57_PointCloud_Tool.exe\n"
            f"{CL.FAULT_HEADER}\n"
            f"{body}"
        )
        if ended:
            text += f"\n{CL.END_MARK}\n"
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return path


class TestSessionLifecycle(_DirCase):
    def test_begin_writes_a_mark_and_end_removes_it(self):
        s = CL.begin_session(self.logs, build="開発版 abc1234")
        try:
            self.assertTrue(s.active)
            self.assertEqual(self.names(CL.SESSION_PREFIX), [os.path.basename(s.path)])
            self.assertTrue(faulthandler.is_enabled())
            text = read_text(s.path)
            self.assertIn(f"pid: {os.getpid()}", text)
            self.assertIn("build: 開発版 abc1234", text)
            self.assertIn(CL.FAULT_HEADER, text)
            self.assertEqual(s.fault_bytes(), 0)
            self.assertEqual(s.abandoned, [])
        finally:
            s.end()
        self.assertEqual(self.names(), [])
        self.assertFalse(faulthandler.is_enabled())
        s.end()  # 2 回呼んでも困らない

    def test_clean_exit_mark_is_written_once(self):
        s = CL.begin_session(self.logs)
        try:
            s.mark_clean_exit()
            s.mark_clean_exit()
            text = read_text(s.path)
            self.assertEqual(text.count(CL.END_MARK), 1)
            self.assertEqual(s.fault_bytes(), 0)
        finally:
            s.end()

    def test_unwritable_folder_does_not_stop_startup(self):
        blocker = os.path.join(self.logs, "not_a_folder")
        with open(blocker, "w") as f:
            f.write("x")
        s = CL.begin_session(os.path.join(blocker, "logs"))
        self.assertFalse(s.active)
        self.assertEqual(s.abandoned, [])
        s.mark_clean_exit()
        s.end()

    def test_old_records_are_dropped_when_the_mark_grows_too_large(self):
        """大きくなった記録は捨てるが、記録そのものは止めない（止めると、そのあとの本当の異常終了を書けない）"""
        s = CL.begin_session(self.logs, build="b")
        try:
            self.assertFalse(s.check_size())
            with open(s.path, "ab") as f:
                f.write(b"Windows fatal exception: code 0x8001010d\n" * (CL.SESSION_MAX_BYTES // 40 + 10))
            self.assertGreater(os.path.getsize(s.path), CL.SESSION_MAX_BYTES)
            self.assertTrue(s.check_size())
            self.assertEqual(s.trimmed, 1)
            self.assertTrue(s.active)
            self.assertTrue(faulthandler.is_enabled())
            self.assertLess(os.path.getsize(s.path), 2000)
            text = read_text(s.path)
            self.assertIn("pid:", text)            # 見出しは残る
            self.assertIn("を捨てました", text)
            self.assertFalse(s.check_size())       # 小さくなったので、次は何もしない
            # 捨てたあとに書かれた記録は、ちゃんと残る
            with open(s.path, "ab") as f:
                f.write(b"Windows fatal exception: access violation\n")
            self.assertIn("access violation", read_text(s.path))
        finally:
            s.end()
        self.assertEqual(self.names(), [])

    def test_a_failed_header_write_leaves_no_mark(self):
        """見出しを書けなかった印を残すと、次の起動が「前回は正常に終了していません」と誤って知らせる"""
        original = CL._session_header
        self.addCleanup(lambda: setattr(CL, "_session_header", original))

        def broken(build, now):
            raise OSError("ディスクがいっぱい")

        CL._session_header = broken
        s = CL.begin_session(self.logs)
        self.assertFalse(s.active)
        self.assertEqual(self.names(), [])
        s.end()
        CL._session_header = original
        self.assertEqual(CL.collect_abandoned(self.logs), [])


class TestCollectAbandoned(_DirCase):
    def test_a_leftover_mark_becomes_a_report(self):
        fault = "Windows fatal exception: access violation\n\nCurrent thread 0x00002110 (most recent call first):\n  File \"gui\\gl_viewer.py\", line 490 in paintGL\n"
        self.leftover(body=fault)
        with open(os.path.join(self.logs, "system_history_2026-11-14.log"), "w", encoding="utf-8") as f:
            f.write("\n".join(f"[2026-11-14 10:0{i % 10}:00.000] [INFO ] 行 {i}" for i in range(60)) + "\n")
        found = CL.collect_abandoned(self.logs, now=datetime(2026, 11, 14, 11, 0, 0))
        self.assertEqual(len(found), 1)
        a = found[0]
        self.assertEqual((a.pid, a.started, a.build), (4242, "2026-11-14 10:02:11", "配布版 2026-11-14 00:50 作成（63a9d9e）"))
        self.assertTrue(a.has_fault_text)
        self.assertEqual(a.fault_line, "Windows fatal exception: access violation")
        self.assertIn("2026-11-14 10:02:11 に起動した回は、正常に終了していません", a.summary())
        self.assertIn("access violation", a.summary())
        self.assertEqual(self.names(CL.SESSION_PREFIX), [])
        self.assertEqual(self.names(CL.CRASH_PREFIX), ["crash_2026-11-14_100211.txt"])
        report = read_text(a.report_path)
        for part in ("正常に終了しなかった起動の記録", "見つけた日時: 2026-11-14 11:00:00", "pid: 4242", "paintGL",
                     "ログの末尾（system_history_2026-11-14.log", "行 59", "行 20"):
            self.assertIn(part, report)
        self.assertNotIn("行 19\n", report)  # 末尾の 40 行だけ

    def test_no_fault_text_means_killed_or_power_loss(self):
        self.leftover()
        a = CL.collect_abandoned(self.logs)[0]
        self.assertFalse(a.has_fault_text)
        self.assertIn("強制終了や電源断", a.summary())
        self.assertIn("記録なし", read_text(a.report_path))

    def test_a_mark_with_the_clean_exit_line_is_removed_silently(self):
        self.leftover(ended=True)
        self.leftover(pid=5, body="Windows fatal exception: code 0x8001010d\n", ended=True)  # 動いているあいだの、続行できた例外
        self.assertEqual(CL.collect_abandoned(self.logs), [])
        self.assertEqual(self.names(), [])

    def test_a_mark_without_a_header_is_removed_silently(self):
        for pid, content in ((11, ""), (12, "ゴミ")):
            with open(os.path.join(self.logs, f"{CL.SESSION_PREFIX}{pid}_20261114-100211.txt"), "w", encoding="utf-8") as f:
                f.write(content)
        self.assertEqual(CL.collect_abandoned(self.logs), [])
        self.assertEqual(self.names(), [])

    def test_summary_shows_the_last_fatal_line_not_the_first(self):
        """続行できた例外が先に書かれていても、要約には終了の原因（いちばん下）を出す"""
        body = (
            "Windows fatal exception: code 0x8001010d\n\nThread 0x00001 (most recent call first):\n  File \"a.py\", line 1 in f\n\n"
            "Windows fatal exception: access violation\n\nCurrent thread 0x00002 (most recent call first):\n  File \"b.py\", line 2 in g\n"
        )
        self.leftover(body=body)
        a = CL.collect_abandoned(self.logs)[0]
        self.assertEqual(a.fault_line, "Windows fatal exception: access violation")
        self.assertFalse(a.during_exit)
        self.assertEqual(CL.main_fault_line("ただの文\n次の行"), "ただの文")
        self.assertEqual(CL.main_fault_line(""), "")

    def test_a_crash_during_exit_is_kept_but_marked(self):
        """終了処理が始まったあとの記録があれば、消さずに残す（利用者には知らせず、ログにだけ出す）"""
        path = self.leftover(body="", ended=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write("Windows fatal exception: access violation\n\nCurrent thread 0x1 (most recent call first):\n  File \"gl.py\", line 9 in cleanup\n")
        found = CL.collect_abandoned(self.logs)
        self.assertEqual(len(found), 1)
        a = found[0]
        self.assertTrue(a.during_exit)
        self.assertIn("終了処理の途中で異常終了", a.summary())
        self.assertIn("access violation", a.summary())
        report = read_text(a.report_path)
        self.assertIn("終わり方: 終了処理の途中で落ちた", report)
        self.assertIn("cleanup", report)

    def test_a_huge_mark_keeps_the_end(self):
        """印が大きく膨らんでいても、末尾（落ちた瞬間の記録）を読み落とさない"""
        noise = "Windows fatal exception: code 0x8001010d\n\nThread 0x1 (most recent call first):\n  File \"x.py\", line 1 in noisy\n\n"
        body = noise * 90_000 + "Windows fatal exception: access violation\n\nCurrent thread 0x2 (most recent call first):\n  File \"real.py\", line 7 in the_real_cause\n"
        path = self.leftover(body=body)
        self.assertGreater(os.path.getsize(path), 9 * 1024 * 1024)
        a = CL.collect_abandoned(self.logs)[0]
        self.assertEqual(a.fault_line, "Windows fatal exception: access violation")
        self.assertEqual(a.started, "2026-11-14 10:02:11")
        report = read_text(a.report_path)
        self.assertIn("the_real_cause", report)
        self.assertLess(len(report), 200_000)

    @unittest.skipUnless(os.name == "nt", "開いているファイルの名前を変えさせないのは Windows の性質")
    def test_a_running_apps_mark_is_left_alone(self):
        path = self.leftover(pid=777)
        with open(path, "ab", buffering=0):  # ほかの起動が、まだ開いている
            self.assertEqual(CL.collect_abandoned(self.logs), [])
            self.assertEqual(self.names(), [os.path.basename(path)])
        self.assertEqual(len(CL.collect_abandoned(self.logs)), 1)  # 閉じられたあとは、残り物として見つかる

    def test_other_files_are_not_touched(self):
        for name in ("session_notes.txt", "system_history_2026-11-14.log", "session_12_bad.txt", "crash_memo.txt"):
            with open(os.path.join(self.logs, name), "w") as f:
                f.write("keep")
        self.assertEqual(CL.collect_abandoned(self.logs), [])
        self.assertEqual(len(self.names()), 4)
        self.assertEqual(CL.collect_abandoned(os.path.join(self.logs, "missing")), [])

    def test_two_leftovers_with_the_same_second_get_different_reports(self):
        self.leftover(pid=1)
        self.leftover(pid=2)
        found = CL.collect_abandoned(self.logs)
        self.assertEqual(len(found), 2)
        self.assertEqual(len(set(a.report_path for a in found)), 2)
        self.assertEqual(len(self.names(CL.CRASH_PREFIX)), 2)

    def test_a_long_fault_text_is_shortened_keeping_the_end(self):
        body = "HEAD-MARK\n" + ("x" * 100 + "\n") * 3000 + "TAIL-MARK\n"
        self.leftover(body=body)
        a = CL.collect_abandoned(self.logs)[0]
        report = read_text(a.report_path)
        self.assertIn("HEAD-MARK", report)
        self.assertIn("TAIL-MARK", report)
        self.assertIn("文字を省略", report)
        self.assertLess(len(report), 120_000)

    def test_begin_session_reports_the_previous_crash(self):
        self.leftover(body="Fatal Python error: Aborted\n")
        s = CL.begin_session(self.logs)
        try:
            self.assertEqual(len(s.abandoned), 1)
            lines = CL.describe_for_log(s)
            self.assertEqual(len(lines), 1)
            self.assertIn("正常に終了していません", lines[0])
            self.assertIn(s.abandoned[0].report_path, lines[0])
            self.assertEqual(len(self.names(CL.SESSION_PREFIX)), 1)  # 今回の印だけ
        finally:
            s.end()

    def test_old_reports_are_cleaned_up_by_the_date_in_the_name(self):
        for name in ("crash_2026-01-01_000000.txt", "crash_2026-11-10_120000.txt", "crash_2026-11-10_120000_2.txt",
                     "crash_memo.txt", "system_history_2026-01-01.log"):
            with open(os.path.join(self.logs, name), "w") as f:
                f.write("x")
        removed = CL.cleanup_old_reports(self.logs, keep_days=90, today=date(2026, 11, 14))
        self.assertEqual(removed, ["crash_2026-01-01_000000.txt"])
        self.assertEqual(len(self.names()), 4)
        self.assertEqual(CL.cleanup_old_reports(self.logs, keep_days=0, today=date(2026, 11, 14)), [])


class TestRealProcesses(_DirCase):
    """別プロセスを実際に落として、次の起動で見つかることを確かめる"""

    def test_access_violation_leaves_the_python_stack(self):
        code = run_child(self.logs, "segv")
        self.assertNotEqual(code, 0)
        wait_released(self.logs)
        found = CL.collect_abandoned(self.logs)
        self.assertEqual(len(found), 1)
        a = found[0]
        self.assertTrue(a.has_fault_text)
        self.assertEqual(a.build, "test build")
        report = read_text(a.report_path)
        self.assertTrue("access violation" in report or "Segmentation fault" in report, report[:600])
        self.assertIn("most recent call first", report)

    def test_killed_from_outside_leaves_no_fault_text(self):
        run_child(self.logs, "hang", kill_after_ready=True)
        wait_released(self.logs)
        found = CL.collect_abandoned(self.logs)
        self.assertEqual(len(found), 1)
        self.assertFalse(found[0].has_fault_text)

    def test_exit_without_cleanup_is_also_found(self):
        self.assertEqual(run_child(self.logs, "exit_without_end"), 7)
        wait_released(self.logs)
        self.assertEqual(len(CL.collect_abandoned(self.logs)), 1)

    def test_normal_exit_leaves_nothing(self):
        self.assertEqual(run_child(self.logs, "normal"), 0)
        self.assertEqual(self.names(), [])
        self.assertEqual(CL.collect_abandoned(self.logs), [])

    def test_crash_during_exit_is_found_and_marked(self):
        self.assertNotEqual(run_child(self.logs, "crash_during_exit"), 0)
        wait_released(self.logs)
        found = CL.collect_abandoned(self.logs)
        self.assertEqual(len(found), 1)
        self.assertTrue(found[0].during_exit)
        self.assertTrue(found[0].has_fault_text)


class TestExceptionHooks(unittest.TestCase):
    def setUp(self):
        self._saved = (threading.excepthook, sys.unraisablehook)
        self.addCleanup(self._restore)

    def _restore(self):
        threading.excepthook, sys.unraisablehook = self._saved

    def test_thread_exception_is_logged_with_the_thread_name(self):
        crit, err = [], []
        CL.install_exception_hooks(crit.append, err.append)

        def boom():
            raise ValueError("スレッドの中の失敗")

        t = threading.Thread(target=boom, name="作業スレッド")
        t.start()
        t.join()
        self.assertEqual(len(crit), 1)
        self.assertIn("作業スレッド", crit[0])
        self.assertIn("ValueError: スレッドの中の失敗", crit[0])
        self.assertEqual(err, [])

    def test_a_failing_logger_does_not_raise(self):
        def bad(_msg):
            raise RuntimeError("ログに書けない")

        CL.install_exception_hooks(bad, bad)
        t = threading.Thread(target=lambda: 1 / 0)
        t.start()
        t.join()  # 例外がここまで伝わらなければよい


class _ListHandler(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.records = []

    def emit(self, record):
        self.records.append((record.levelname, record.getMessage()))


class TestQtMessageLog(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication(sys.argv)

    def setUp(self):
        from core.logger import get_logger
        from gui import diagnostics as DG

        self.DG = DG
        self.handler = _ListHandler()
        self.logger = get_logger()
        self.logger.addHandler(self.handler)
        self.addCleanup(self.logger.removeHandler, self.handler)
        self.addCleanup(DG.uninstall_qt_message_log)

    def qt_lines(self):
        return [(lv, msg) for lv, msg in self.handler.records if msg.startswith("[Qt]")]

    def test_levels(self):
        from PySide6.QtCore import QtMsgType

        DG = self.DG
        self.assertEqual(DG.qt_message_level(QtMsgType.QtWarningMsg), "warning")
        self.assertEqual(DG.qt_message_level(QtMsgType.QtCriticalMsg), "error")
        self.assertEqual(DG.qt_message_level(QtMsgType.QtFatalMsg), "critical")
        self.assertIsNone(DG.qt_message_level(QtMsgType.QtDebugMsg))
        self.assertIsNone(DG.qt_message_level(QtMsgType.QtInfoMsg))
        # 害が無いと分かっている Qt の内部の警告は、警告として数えない（起動のたびに 1 回出るため）
        known = "QFont::setPointSize: Point size <= 0 (-1), must be greater than 0"
        self.assertEqual(DG.qt_message_level(QtMsgType.QtWarningMsg, known), "debug")
        self.assertEqual(DG.qt_message_level(QtMsgType.QtWarningMsg, "QFont::setPointSize: Point size <= 0 (0)"), "warning")

    def test_qt_warnings_reach_the_log_and_repeats_are_capped(self):
        from PySide6.QtCore import qCritical, qDebug, qWarning

        self.DG.install_qt_message_log()
        self.DG.install_qt_message_log()  # 2 回呼んでも二重にならない
        for _ in range(6):
            qWarning("同じ警告")
        qCritical("重いエラー")
        qDebug("デバッグ用の文は残さない")
        qWarning("QFont::setPointSize: Point size <= 0 (-1), must be greater than 0")
        lines = self.qt_lines()
        self.assertIn(("DEBUG", "[Qt] QFont::setPointSize: Point size <= 0 (-1), must be greater than 0"), lines)
        same = [msg for lv, msg in lines if "同じ警告" in msg]
        self.assertEqual(len(same), self.DG.QT_MAX_REPEAT)
        self.assertIn("これ以降は省きます", same[-1])
        self.assertIn(("ERROR", "[Qt] 重いエラー"), lines)
        self.assertFalse(any("デバッグ用" in msg for _, msg in lines))
        self.assertTrue(all(lv == "WARNING" for lv, msg in lines if "同じ警告" in msg))

    def test_after_uninstall_nothing_is_logged(self):
        from PySide6.QtCore import qInstallMessageHandler, qWarning

        self.DG.install_qt_message_log()
        self.DG.uninstall_qt_message_log()
        seen = []
        qInstallMessageHandler(lambda mode, ctx, msg: seen.append(msg))  # 標準エラー出力へ流さないための受け皿
        try:
            qWarning("外したあとの警告")
        finally:
            qInstallMessageHandler(None)
        self.assertEqual(seen, ["外したあとの警告"])
        self.assertEqual(self.qt_lines(), [])


class TestPreviousCrashNotice(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication(sys.argv)
        from gui.main_window import MainWindow

        cls.win = MainWindow()
        cls.win._reset_app_settings()
        cls.win.resize(1450, 900)
        cls.win.show()
        cls.app.processEvents()

    @classmethod
    def tearDownClass(cls):
        from PySide6.QtCore import QCoreApplication, QEvent

        cls.win._set_dirty(False)
        cls.win.close()
        cls.win.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        cls.app.processEvents()

    def _pump(self, ms: int = 120):
        end = time.monotonic() + ms / 1000.0
        while time.monotonic() < end:
            self.app.processEvents()
            time.sleep(0.005)

    def test_nothing_is_shown_when_the_last_run_ended_normally(self):
        from gui import diagnostics as DG

        self.assertEqual(DG.previous_crash_text([]), "")
        before = self.win.log_text.toPlainText()
        self.assertFalse(DG.show_previous_crash_notice(self.win, [], delay_ms=0))
        self._pump(60)
        self.assertEqual(self.win.log_text.toPlainText(), before)

    def test_notice_is_shown_in_the_log_pane_and_as_a_toast(self):
        from gui import diagnostics as DG

        a = CL.AbandonedSession(report_path=r"C:\logs\crash_2026-11-14_100211.txt", started="2026-11-14 10:02:11",
                                pid=1, build="b", has_fault_text=True,
                                fault_line="Windows fatal exception: access violation")
        self.assertTrue(DG.show_previous_crash_notice(self.win, [a], delay_ms=0))
        self._pump(150)
        log = self.win.log_text.toPlainText()
        self.assertIn("前回は正常に終了しませんでした", log)
        self.assertIn("2026-11-14 10:02:11 に起動した回", log)
        self.assertTrue(self.win.toast_label.isVisible())
        self.assertIn("前回は正常に終了しませんでした", self.win.toast_label.text())
        host = getattr(self.win, "toast_host", None)
        if host is not None:  # 新UI: 次の一手のボタンが付く
            self.assertTrue(host.has_action())
            self.assertEqual(host.action_btn.text(), "ログを開く")

    def test_several_crashes_are_counted(self):
        from gui import diagnostics as DG

        items = [CL.AbandonedSession(report_path=f"c{i}.txt") for i in range(3)]
        self.assertIn("3 回", DG.previous_crash_text(items))

    def test_a_crash_during_exit_is_not_shown_to_the_user(self):
        """利用者が自分で閉じたあとの異常は、画面では知らせない（ログには残る）"""
        from gui import diagnostics as DG

        during_exit = CL.AbandonedSession(report_path="c.txt", during_exit=True, fault_line="Windows fatal exception: access violation")
        self.assertEqual(DG.previous_crash_text([during_exit]), "")
        before = self.win.log_text.toPlainText()
        self.assertFalse(DG.show_previous_crash_notice(self.win, [during_exit], delay_ms=0))
        self._pump(60)
        self.assertEqual(self.win.log_text.toPlainText(), before)
        running = CL.AbandonedSession(report_path="d.txt")
        self.assertNotIn("回ありました", DG.previous_crash_text([during_exit, running]))  # 数えるのは 1 回だけ


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
