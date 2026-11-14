# -*- coding: utf-8 -*-
"""
読み込み前のメモリの確認（gui/load_guard.py）のテスト

- 収まる見込みなら何も聞かず、これまでと同じ上限で読み込むこと
- 「無制限」や点数の指定で収まらないときは、始める前に聞くこと（おすすめの点数で読む / このまま読む / やめる）
- 「自動最適化」で収まらない PC では、聞かずに収まる点数まで下げ、ログの欄に理由を書くこと
- やめたときは読み込みを始めず、今の点群をそのまま残すこと
- ファイルの情報を読めないときは止めないこと
- メモリ不足で読み込みが止まったときは、次の一手の分かる文を出すこと
メモリの量は差し替える（小さな合成ファイルで、収まらない状況を作る）。
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

from PySide6.QtCore import QCoreApplication, QEvent  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from core.memguard import AUTO_TIER_SMALL  # noqa: E402
from core.sysinfo import GB, MB  # noqa: E402
from gui import load_guard as LG  # noqa: E402
from gui import worker_thread as WT  # noqa: E402
from tests.test_memguard import LAYOUT_SCALED_NORMALS, POSE_TILT, write_synthetic_e57  # noqa: E402

N_POINTS = 200_000
# 20 万点・全点表示の見込みは 約 38.6MB（1 点 202.4 バイト）。載っているメモリを 48MB と見せると、使えるのは 24MB で収まらない
TINY_PC = (48 * MB, 20 * MB)
BIG_PC = (int(31.1 * GB), int(15 * GB))


class _WinCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)
        from gui.main_window import MainWindow

        cls.tmp = tempfile.mkdtemp(prefix="e57_loadguard_")
        cls.path = os.path.join(cls.tmp, "大きな現場.e57")
        write_synthetic_e57(cls.path, LAYOUT_SCALED_NORMALS, (N_POINTS,), pose=POSE_TILT, seed=5)
        cls.small = os.path.join(cls.tmp, "小さな現場.e57")
        write_synthetic_e57(cls.small, LAYOUT_SCALED_NORMALS, (4000,), pose=POSE_TILT, seed=6)
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
        shutil.rmtree(cls.tmp, True)

    @staticmethod
    def _pump(ms: int = 30):
        end = time.monotonic() + ms / 1000.0
        while time.monotonic() < end:
            QApplication.processEvents()
            time.sleep(0.002)

    def setUp(self):
        win = self.win
        self.asked = []   # 確認の窓に出た (題名, 文, 内訳, おすすめのボタン)
        self.answer = None
        self._saved = (LG.HOOKS.ask_load_memory, LG.HOOKS.memory_status, QMessageBox.critical)
        self.addCleanup(self._restore)
        self.errors = []
        QMessageBox.critical = staticmethod(lambda *a, **k: self.errors.append(a[2] if len(a) > 2 else "") or QMessageBox.Ok)

        def ask(parent, title, text, info, recommend_label):
            self.asked.append((title, text, info, recommend_label))
            if self.answer is None:
                self.fail("確認の窓を出す場面ではありません")
            return self.answer

        LG.HOOKS.ask_load_memory = ask
        LG.HOOKS.memory_status = lambda: BIG_PC
        self._set_load_setting(-1)
        win.budget_combo.setCurrentIndex(win.budget_combo.findData(0))  # 全点表示
        win.log_text.clear()

    def _restore(self):
        LG.HOOKS.ask_load_memory, LG.HOOKS.memory_status, QMessageBox.critical = self._saved
        self._wait_load()
        self.win.viewer.set_point_cloud(None)
        self.win.current_scans = []
        self.win._set_dirty(False)
        self.win._last_loaded_input_path = ""
        self.win.input_edit.setText("")
        self.win.output_edit.setText("")

    def _set_load_setting(self, data: int) -> None:
        combo = self.win.combo_load_budget
        idx = combo.findData(data)
        self.assertGreaterEqual(idx, 0)
        combo.setCurrentIndex(idx)

    def _wait_load(self, timeout: float = 30.0) -> None:
        win = self.win
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            worker = win.load_worker
            if (worker is None or not worker.isRunning()) and win.progress_dialog is None:
                break
            self._pump(20)
        self._pump(100)

    def _n_loaded(self) -> int:
        pts = self.win.viewer.points_raw
        return 0 if pts is None else len(pts)


class TestResolveLoadLimit(_WinCase):
    def test_fits_no_question_and_the_old_limit(self):
        for setting, expected in ((-1, 60_000_000), (30_000_000, 30_000_000), (100_000_000, 100_000_000), (0, None)):
            with self.subTest(setting=setting):
                self._set_load_setting(setting)
                lim = LG.resolve_load_limit(self.win, self.path)
                self.assertEqual((lim.max_points, lim.cancelled, lim.note), (expected, False, ""))
        self.assertEqual(self.asked, [])
        self.assertIsNone(QApplication.overrideCursor(), "待ちのカーソルを戻す")

    def test_unlimited_that_does_not_fit_asks_first(self):
        LG.HOOKS.memory_status = lambda: TINY_PC
        self._set_load_setting(0)
        self.answer = "cancel"
        lim = LG.resolve_load_limit(self.win, self.path)
        self.assertTrue(lim.cancelled)
        self.assertEqual(len(self.asked), 1)
        title, text, info, label = self.asked[0]
        self.assertEqual(title, LG.DIALOG_TITLE)
        self.assertIn("メモリに収まりません", text)
        for part in ("ファイルの点数: 20 万点", "読み込む点数: 20 万点（全点）", "無制限", "要るメモリの見込み: 約 39MB",
                     "この PC のメモリ: 48MB", "約 24MB まで", "全点で保存", "自動最適化"):
            self.assertIn(part, info)
        self.assertEqual(label, "10 万点で読み込む（おすすめ）")  # 2 点に 1 点
        self.assertIsNone(QApplication.overrideCursor())

    def test_recommend_reduces_the_limit_and_explains(self):
        LG.HOOKS.memory_status = lambda: TINY_PC
        self._set_load_setting(0)
        self.answer = "recommend"
        lim = LG.resolve_load_limit(self.win, self.path)
        self.assertFalse(lim.cancelled)
        self.assertIsNotNone(lim.max_points)
        self.assertLess(lim.max_points, N_POINTS)
        self.assertIn("10 万点", lim.note)
        self.assertIn("10 万点", self.win.log_text.toPlainText())

    def test_proceed_keeps_the_setting(self):
        LG.HOOKS.memory_status = lambda: TINY_PC
        self._set_load_setting(0)
        self.answer = "proceed"
        lim = LG.resolve_load_limit(self.win, self.path)
        self.assertEqual((lim.max_points, lim.cancelled), (None, False))

    def test_auto_is_lowered_without_asking(self):
        LG.HOOKS.memory_status = lambda: TINY_PC
        self._set_load_setting(-1)
        lim = LG.resolve_load_limit(self.win, self.path)
        self.assertEqual(self.asked, [])
        self.assertFalse(lim.cancelled)
        self.assertLess(lim.max_points, AUTO_TIER_SMALL)
        self.assertLess(lim.max_points, N_POINTS)
        log = self.win.log_text.toPlainText()
        self.assertIn("この PC のメモリ（48MB）に合わせて、読み込む点数を 10 万点に下げました", log)
        self.assertEqual(lim.note, log.strip())

    def test_small_file_never_asks(self):
        LG.HOOKS.memory_status = lambda: TINY_PC
        for setting in (-1, 0, 30_000_000):
            self._set_load_setting(setting)
            lim = LG.resolve_load_limit(self.win, self.small)
            self.assertFalse(lim.cancelled)
        self.assertEqual(self.asked, [])

    def test_unreadable_file_does_not_block(self):
        bad = os.path.join(self.tmp, "壊れている.e57")
        with open(bad, "wb") as f:
            f.write(b"this is not an e57 file")
        LG.HOOKS.memory_status = lambda: TINY_PC
        self._set_load_setting(0)
        lim = LG.resolve_load_limit(self.win, bad)
        self.assertEqual((lim.max_points, lim.cancelled), (None, False))
        self.assertEqual(self.asked, [])
        self.assertIsNone(QApplication.overrideCursor())
        # ファイルが無い・E57 の無いフォルダも止めない（読み込みの側が、いつもの案内を出す）
        self.assertFalse(LG.resolve_load_limit(self.win, os.path.join(self.tmp, "無い.e57")).cancelled)
        empty = os.path.join(self.tmp, "空のフォルダ")
        os.makedirs(empty, exist_ok=True)
        self.assertFalse(LG.resolve_load_limit(self.win, empty).cancelled)

    def test_unknown_memory_does_not_block(self):
        LG.HOOKS.memory_status = lambda: None
        self._set_load_setting(0)
        lim = LG.resolve_load_limit(self.win, self.path)
        self.assertEqual((lim.max_points, lim.cancelled), (None, False))
        self._set_load_setting(-1)
        self.assertEqual(LG.resolve_load_limit(self.win, self.path).max_points, AUTO_TIER_SMALL)  # 16GB とみなす（以前と同じ）

    def test_folder_is_checked_as_a_whole(self):
        folder = os.path.join(self.tmp, "フォルダ読み込み")
        os.makedirs(folder, exist_ok=True)
        for i in range(3):
            write_synthetic_e57(os.path.join(folder, f"s{i}.e57"), LAYOUT_SCALED_NORMALS, (70_000,), pose=POSE_TILT, seed=i)
        LG.HOOKS.memory_status = lambda: TINY_PC
        self._set_load_setting(0)
        self.answer = "cancel"
        self.assertTrue(LG.resolve_load_limit(self.win, folder).cancelled)
        self.assertIn("ファイルの点数: 21 万点", self.asked[0][2])


class TestFormatPoints(unittest.TestCase):
    def test_format(self):
        self.assertEqual(LG.format_points(8200), "8,200 点")
        self.assertEqual(LG.format_points(59_819_715), "5,982 万点")
        self.assertEqual(LG.format_points(100_000_000), "1 億点")
        self.assertEqual(LG.format_points(538_377_435), "5.38 億点")
        self.assertEqual(LG.format_points(107_675_487), "1.08 億点")
        self.assertEqual(LG.format_points(200_000), "20 万点")


class TestLoadThroughTheWindow(_WinCase):
    """本体の読み込みの入口（_load_file_to_viewer）を通して確かめる"""

    def test_recommended_load_reads_fewer_points(self):
        LG.HOOKS.memory_status = lambda: TINY_PC
        self._set_load_setting(0)
        self.answer = "recommend"
        self.assertIsNot(self.win._load_file_to_viewer(self.path), False)
        self._wait_load()
        self.assertEqual(self._n_loaded(), N_POINTS // 2)
        self.assertTrue(self.win.current_scans[0].is_subsampled)
        self.assertEqual(self.win.current_scans[0].original_point_count, N_POINTS)
        self.assertEqual(self.errors, [])

    def test_proceed_reads_everything(self):
        LG.HOOKS.memory_status = lambda: TINY_PC
        self._set_load_setting(0)
        self.answer = "proceed"
        self.assertIsNot(self.win._load_file_to_viewer(self.path), False)
        self._wait_load()
        self.assertEqual(self._n_loaded(), N_POINTS)

    def test_cancel_keeps_the_current_cloud(self):
        win = self.win
        self.assertIsNot(win._load_file_to_viewer(self.small), False)
        self._wait_load()
        self.assertEqual(self._n_loaded(), 4000)
        before = win.viewer.points_raw
        LG.HOOKS.memory_status = lambda: TINY_PC
        self._set_load_setting(0)
        self.answer = "cancel"
        self.assertIs(win._load_file_to_viewer(self.path), False)
        self._pump(100)
        self.assertIsNone(win.progress_dialog, "進捗の窓を出さない")
        self.assertIs(win.viewer.points_raw, before)
        self.assertEqual(os.path.normcase(win._last_loaded_input_path), os.path.normcase(self.small),
                         "やめたファイルを「読み込み済み」として覚えない（もう一度 Enter で、また確かめられる）")

    def test_cancel_puts_the_path_fields_back(self):
        """
        「やめる」を選んだら、入力欄と出力欄を今の点群のものへ戻す。
        新しいファイルの名前が残っていると、そのまま保存したときに、今の点群が新しいファイル用の出力先へ書かれてしまう。
        """
        win = self.win
        win.input_edit.setText(self.small)
        win._on_input_file_changed(self.small)
        self._wait_load()
        self.assertEqual(self._n_loaded(), 4000)
        out_before = win.output_edit.text()
        self.assertTrue(out_before.endswith("小さな現場_cleaned.e57"), out_before)

        LG.HOOKS.memory_status = lambda: TINY_PC
        self._set_load_setting(0)
        self.answer = "cancel"
        win.input_edit.setText(self.path)
        win._on_input_file_changed(self.path)
        self._pump(100)
        self.assertEqual(len(self.asked), 1)
        self.assertEqual(self._n_loaded(), 4000, "画面の点群はそのまま")
        self.assertEqual(os.path.normcase(win.input_edit.text()), os.path.normcase(self.small))
        self.assertEqual(win.output_edit.text(), out_before)

        # おすすめの点数で読むと、欄は新しいファイルのものになる
        self.answer = "recommend"
        win.input_edit.setText(self.path)
        win._on_input_file_changed(self.path)
        self._wait_load()
        self.assertEqual(self._n_loaded(), N_POINTS // 2)
        self.assertEqual(os.path.normcase(win.input_edit.text()), os.path.normcase(self.path))
        self.assertTrue(win.output_edit.text().endswith("大きな現場_cleaned.e57"), win.output_edit.text())

    def test_auto_setting_on_a_small_pc_loads_what_fits(self):
        LG.HOOKS.memory_status = lambda: TINY_PC
        self._set_load_setting(-1)
        self.assertIsNot(self.win._load_file_to_viewer(self.path), False)
        self._wait_load()
        self.assertEqual(self._n_loaded(), N_POINTS // 2)
        self.assertEqual(self.asked, [])

    def test_usual_pc_loads_as_before(self):
        self._set_load_setting(-1)
        self.assertIsNot(self.win._load_file_to_viewer(self.path), False)
        self._wait_load()
        self.assertEqual(self._n_loaded(), N_POINTS)
        self.assertEqual(self.asked, [])
        self.assertNotIn("下げました", self.win.log_text.toPlainText())


class TestOutOfMemoryMessage(unittest.TestCase):
    def test_memory_error_becomes_a_message_with_the_next_step(self):
        tmp = tempfile.mkdtemp(prefix="e57_oom_")
        self.addCleanup(shutil.rmtree, tmp, True)
        path = os.path.join(tmp, "a.e57")
        write_synthetic_e57(path, LAYOUT_SCALED_NORMALS, (100,), pose=None)
        original = WT.read_e57_scans
        self.addCleanup(lambda: setattr(WT, "read_e57_scans", original))

        def no_memory(*a, **k):
            raise MemoryError("Unable to allocate 96.3 GiB for an array with shape (538377435, 3) and data type float64")

        WT.read_e57_scans = no_memory
        worker = WT.FileLoadWorker(path)
        got, errors = [], []
        worker.finished_success.connect(got.append)
        worker.finished_error.connect(errors.append)
        worker.run()
        self.assertEqual(got, [])
        self.assertEqual(errors, [WT.LOAD_OUT_OF_MEMORY_TEXT])
        self.assertIn("大容量点群読込上限", errors[0])
        self.assertNotIn("Unable to allocate", errors[0])

    def test_other_errors_keep_their_own_text(self):
        worker = WT.FileLoadWorker(os.path.join(tempfile.gettempdir(), "no_such_folder_e57", "x.e57"))
        errors = []
        worker.finished_error.connect(errors.append)
        worker.run()
        self.assertEqual(len(errors), 1)
        self.assertIn("見つかりません", errors[0])


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
