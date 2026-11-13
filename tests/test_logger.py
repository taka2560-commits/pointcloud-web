# -*- coding: utf-8 -*-
"""
ログの置き場所・日付ごとのファイル・古いログの片付け（core/logger.py）

2026-11-13 に見つけた問題:
- 配布版のログは EXE と同じフォルダにあり、配布版を作り直すとフォルダごと置き換わって消えていた
  （利用者が 18:10 ごろに行った作業の記録が、その日の作り直しで残っていなかった）
- 書き込めないフォルダだと起動できなかった / 日をまたぐと前の日のファイルへ書き続けた / 古いログが消えなかった
"""
import datetime
import logging
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import logger as L  # noqa: E402


class _TmpCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="e57_logtest_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        saved = L._logs_dir_cache
        self.addCleanup(lambda: setattr(L, "_logs_dir_cache", saved))


class TestLogLocation(_TmpCase):
    def test_distribution_writes_outside_the_app_folder(self):
        """配布版は利用者ごとのフォルダが先（作り直しても消えない）。ソースからの起動はこれまでどおりアプリのフォルダ"""
        app = os.path.join(self.tmp, "app")
        user = os.path.join(self.tmp, "user")
        frozen = L.candidate_log_dirs(True, app, "", user_dir=user, temp_dir=self.tmp)
        self.assertEqual(frozen[0], os.path.join(user, "logs"))
        self.assertEqual(frozen[1], os.path.join(app, "logs"))
        self.assertTrue(frozen[-1].startswith(self.tmp))
        source = L.candidate_log_dirs(False, app, "", user_dir=user, temp_dir=self.tmp)
        self.assertEqual(source[0], os.path.join(app, "logs"))
        self.assertEqual(source[1], os.path.join(user, "logs"))

    def test_environment_variable_comes_first_and_no_duplicates(self):
        app = os.path.join(self.tmp, "app")
        env = os.path.join(self.tmp, "custom")
        dirs = L.candidate_log_dirs(True, app, env, user_dir=os.path.join(self.tmp, "user"), temp_dir=self.tmp)
        self.assertEqual(dirs[0], os.path.abspath(env))
        same = L.candidate_log_dirs(False, app, os.path.join(app, "logs"), user_dir=os.path.join(self.tmp, "u"), temp_dir=self.tmp)
        self.assertEqual(len(same), len({os.path.normcase(d) for d in same}), "同じフォルダを 2 回試さない")

    def test_user_data_dir_follows_localappdata(self):
        with mock.patch.dict(os.environ, {"LOCALAPPDATA": self.tmp}):
            self.assertEqual(L.get_user_data_dir(), os.path.join(self.tmp, L.APP_DATA_DIRNAME))

    def test_get_logs_dir_uses_the_environment_variable(self):
        target = os.path.join(self.tmp, "from_env")
        with mock.patch.dict(os.environ, {L.LOG_DIR_ENV: target}):
            got = L.get_logs_dir()
            self.assertEqual(os.path.normcase(got), os.path.normcase(os.path.abspath(target)))
            self.assertTrue(os.path.isdir(target), "無ければ作る")
            self.assertEqual(os.path.dirname(L.get_today_log_path()), got)
            self.assertRegex(os.path.basename(L.get_today_log_path()), r"^system_history_\d{4}-\d{2}-\d{2}\.log$")
            self.assertEqual(os.listdir(target), [], "書き込みの確認に使ったファイルを残さない")
        # 環境変数を外したら、元の場所へ戻る（前の結果を使い回さない）
        with mock.patch.dict(os.environ):
            os.environ.pop(L.LOG_DIR_ENV, None)
            self.assertNotEqual(os.path.normcase(L.get_logs_dir()), os.path.normcase(os.path.abspath(target)))

    def test_falls_back_when_the_folder_cannot_be_written(self):
        """書き込めない場所を指定されても起動できる（次の候補へ替える）"""
        blocker = os.path.join(self.tmp, "file.txt")
        with open(blocker, "w", encoding="utf-8") as f:
            f.write("x")
        bad = os.path.join(blocker, "logs")  # ファイルの下にはフォルダを作れない
        self.assertFalse(L._can_write_dir(bad))
        with mock.patch.dict(os.environ, {L.LOG_DIR_ENV: bad}):
            got = L.get_logs_dir()
        self.assertNotEqual(os.path.normcase(got), os.path.normcase(bad))
        self.assertTrue(os.path.isdir(got))

    def test_distribution_picks_the_user_folder(self):
        """配布版として動いているとき、EXE のフォルダではなく利用者ごとのフォルダを選ぶ"""
        fake_exe = os.path.join(self.tmp, "dist", "E57_PointCloud_Tool", "E57_PointCloud_Tool.exe")
        os.makedirs(os.path.dirname(fake_exe))
        local = os.path.join(self.tmp, "LocalAppData")
        with mock.patch.dict(os.environ, {"LOCALAPPDATA": local}), \
                mock.patch.object(sys, "frozen", True, create=True), mock.patch.object(sys, "executable", fake_exe):
            os.environ.pop(L.LOG_DIR_ENV, None)
            got = L.get_logs_dir()
        self.assertEqual(os.path.normcase(got), os.path.normcase(os.path.join(local, L.APP_DATA_DIRNAME, "logs")))
        self.assertFalse(os.path.exists(os.path.join(os.path.dirname(fake_exe), "logs")), "EXE のフォルダには作らない")


class TestDailyFile(_TmpCase):
    def _record(self, when: datetime.datetime, msg: str) -> logging.LogRecord:
        rec = logging.LogRecord("t", logging.INFO, __file__, 1, msg, None, None)
        rec.created = when.timestamp()
        return rec

    def test_switches_file_when_the_date_changes(self):
        h = L.DailyFileHandler(self.tmp)
        h.setFormatter(logging.Formatter("%(message)s"))
        try:
            h.handle(self._record(datetime.datetime(2026, 11, 13, 23, 59, 59), "13 日の最後の行 🚀"))
            h.handle(self._record(datetime.datetime(2026, 11, 14, 0, 0, 1), "14 日の最初の行"))
            h.handle(self._record(datetime.datetime(2026, 11, 14, 0, 0, 2), "14 日の 2 行目"))
            self.assertEqual(os.path.basename(h.current_path), "system_history_2026-11-14.log")
        finally:
            h.close()
        with open(os.path.join(self.tmp, "system_history_2026-11-13.log"), encoding="utf-8") as f:
            self.assertEqual(f.read(), "13 日の最後の行 🚀\n")
        with open(os.path.join(self.tmp, "system_history_2026-11-14.log"), encoding="utf-8") as f:
            self.assertEqual(f.read(), "14 日の最初の行\n14 日の 2 行目\n")

    def test_appends_and_survives_a_missing_folder(self):
        sub = os.path.join(self.tmp, "not_yet", "logs")
        when = datetime.datetime(2026, 11, 13, 12, 0, 0)
        for text in ("1 回目", "2 回目"):
            h = L.DailyFileHandler(sub)
            h.setFormatter(logging.Formatter("%(message)s"))
            h.handle(self._record(when, text))
            h.close()
        with open(os.path.join(sub, "system_history_2026-11-13.log"), encoding="utf-8") as f:
            self.assertEqual(f.read(), "1 回目\n2 回目\n", "起動し直しても同じ日のファイルへ足す")

    def test_closing_twice_is_harmless(self):
        h = L.DailyFileHandler(self.tmp)
        h.close()
        h.close()


class TestCleanup(_TmpCase):
    def _touch(self, name: str) -> str:
        path = os.path.join(self.tmp, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write("x")
        return path

    def test_removes_only_old_log_files(self):
        today = datetime.date(2026, 11, 13)
        old = self._touch("system_history_2026-08-01.log")        # 104 日前
        edge = self._touch("system_history_2026-08-15.log")       # ちょうど 90 日前 → 残す
        recent = self._touch("system_history_2026-11-12.log")
        other = self._touch("メモ.txt")
        odd = self._touch("system_history_old.log")
        bad_date = self._touch("system_history_2026-13-45.log")
        folder = os.path.join(self.tmp, "system_history_2020-01-01.log")
        os.mkdir(folder)  # 名前が同じ形のフォルダには触らない
        removed = L.cleanup_old_logs(self.tmp, keep_days=90, today=today)
        self.assertEqual(removed, ["system_history_2026-08-01.log"])
        self.assertFalse(os.path.exists(old))
        for keep in (edge, recent, other, odd, bad_date, folder):
            self.assertTrue(os.path.exists(keep), keep)

    def test_zero_days_and_missing_folder(self):
        self._touch("system_history_2000-01-01.log")
        self.assertEqual(L.cleanup_old_logs(self.tmp, keep_days=0), [], "0 は「消さない」")
        self.assertEqual(L.cleanup_old_logs(os.path.join(self.tmp, "none")), [])

    def test_default_keeps_three_months(self):
        self.assertGreaterEqual(L.LOG_KEEP_DAYS, 60)


class TestEnvironmentLines(unittest.TestCase):
    def test_startup_lines_tell_version_disk_and_memory(self):
        """起動時のログに、版・ディスクの空き・メモリを残す（不具合を調べるとき、どの版で・どれだけ余裕があったかが分かる）"""
        text = "\n".join(L._environment_lines())
        self.assertIn("・版: ", text)
        self.assertIn("・ディスク: ", text)
        self.assertRegex(text, r"空き [0-9.]+(GB|MB|KB|B) / 全体 [0-9.]+(GB|MB|KB|B)")
        if sys.platform == "win32":
            self.assertIn("・メモリ: 搭載 ", text)


if __name__ == "__main__":
    unittest.main()
