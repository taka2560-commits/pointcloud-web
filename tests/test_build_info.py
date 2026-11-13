# -*- coding: utf-8 -*-
"""
版の表示（core/build_info.py と tools/write_build_info.py、ビルド定義とのつながり）

2026-11-13: 配布版を 1 日に 3 回作り直したが、画面にもログにも版が出ず、どの版で起きたことか区別できなかった。
配布版は作った日時とコミットを同梱し、タイトル・バージョン情報・ログに出す。
"""
import datetime
import io
import json
import os
import shutil
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core.build_info import (  # noqa: E402
    BUILD_INFO_FILE, BuildInfo, get_build_info, load_build_file, make_build_record, read_git_info, resolve_build_info,
)


def _fake_git(head="35e7885|2026-11-13 22:08", status=""):
    calls = []

    def run(args, cwd):
        calls.append((tuple(args), cwd))
        if args[0] == "log":
            return head
        if args[0] == "status":
            return status
        return None

    run.calls = calls
    return run


class TestGitInfo(unittest.TestCase):
    def test_reads_commit_date_and_dirty(self):
        info = read_git_info("repo", _fake_git())
        self.assertEqual(info, {"commit": "35e7885", "commit_date": "2026-11-13 22:08", "dirty": False})
        dirty = read_git_info("repo", _fake_git(status=" M gui/main_window.py"))
        self.assertTrue(dirty["dirty"])

    def test_asks_git_in_the_given_folder_with_local_time(self):
        run = _fake_git()
        read_git_info("C:/somewhere", run)
        self.assertEqual(run.calls[0][1], "C:/somewhere")
        self.assertIn("--date=format-local:%Y-%m-%d %H:%M", run.calls[0][0])

    def test_never_fails_without_git(self):
        empty = {"commit": "", "commit_date": "", "dirty": False}
        self.assertEqual(read_git_info("repo", lambda a, c: None), empty, "git が無い・リポジトリでない")
        self.assertEqual(read_git_info("repo", _fake_git(head="fatal: not a git repository")), empty)
        self.assertEqual(read_git_info("repo", _fake_git(head="zzzz|2026-11-13 22:08")), empty, "コミットの番号に見えない")
        self.assertEqual(read_git_info("repo", _fake_git(head="|x")), empty)
        # 状態が聞けなかっただけなら、コミットは出す
        self.assertEqual(read_git_info("repo", _fake_git(status=None))["commit"], "35e7885")

    def test_real_git_call_does_not_raise(self):
        info = read_git_info(ROOT)
        self.assertEqual(set(info), {"commit", "commit_date", "dirty"})
        if info["commit"]:
            self.assertRegex(str(info["commit"]), r"^[0-9a-f]{7,40}$")
            self.assertRegex(str(info["commit_date"]), r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}$")
        not_a_repo = tempfile.mkdtemp(prefix="e57_nogit_")
        try:
            self.assertIn("commit", read_git_info(not_a_repo))
        finally:
            shutil.rmtree(not_a_repo, True)


class TestBuildRecord(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="e57_buildinfo_")
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def _write(self, data) -> str:
        path = os.path.join(self.tmp, BUILD_INFO_FILE)
        with open(path, "w", encoding="utf-8") as f:
            f.write(data if isinstance(data, str) else json.dumps(data, ensure_ascii=False))
        return path

    def test_record_has_build_time_and_commit(self):
        rec = make_build_record("repo", now=datetime.datetime(2026, 11, 13, 22, 16, 40), run=_fake_git())
        self.assertEqual(rec, {"commit": "35e7885", "commit_date": "2026-11-13 22:08", "dirty": False, "built_at": "2026-11-13 22:16"})

    def test_distribution_shows_build_time(self):
        path = self._write({"built_at": "2026-11-13 22:16", "commit": "35e7885", "commit_date": "2026-11-13 22:08", "dirty": False})
        info = resolve_build_info(True, candidates=[os.path.join(self.tmp, "none.json"), path])
        self.assertTrue(info.frozen)
        self.assertEqual(info.title_tag(), "版 2026-11-13 22:16")
        self.assertEqual(info.describe(), "配布版 2026-11-13 22:16 作成（35e7885）")

    def test_distribution_without_a_record_says_so(self):
        info = resolve_build_info(True, candidates=[os.path.join(self.tmp, "none.json")])
        self.assertEqual(info.title_tag(), "版の記録なし")
        self.assertEqual(info.describe(), "配布版（作成日時の記録なし）")
        for broken in ("{", "[]", json.dumps({"commit": "35e7885"}), json.dumps({"built_at": "  "})):
            self.assertIsNone(load_build_file(self._write(broken)), broken)

    def test_dirty_build_is_marked(self):
        path = self._write({"built_at": "2026-11-13 22:16", "commit": "35e7885", "dirty": True})
        self.assertIn("未コミットの変更", resolve_build_info(True, candidates=[path]).describe())
        no_commit = self._write({"built_at": "2026-11-13 22:16", "commit": ""})
        self.assertEqual(resolve_build_info(True, candidates=[no_commit]).describe(), "配布版 2026-11-13 22:16 作成")

    def test_source_run_shows_the_commit_and_ignores_a_stale_file(self):
        """ソースから起動したときは git に聞く。リポジトリに残っている前回の build_info.json は読まない"""
        self._write({"built_at": "2020-01-01 00:00", "commit": "0000000"})
        info = resolve_build_info(False, repo_dir=self.tmp, run=_fake_git(status=" M x.py"))
        self.assertFalse(info.frozen)
        self.assertEqual(info.built_at, "")
        self.assertEqual(info.title_tag(), "開発版 35e7885＋")
        self.assertEqual(info.describe(), "開発版 35e7885（2026-11-13 22:08 のコミット＋未コミットの変更）")
        clean = resolve_build_info(False, repo_dir=self.tmp, run=_fake_git())
        self.assertEqual(clean.title_tag(), "開発版 35e7885")
        self.assertEqual(clean.describe(), "開発版 35e7885（2026-11-13 22:08 のコミット）")
        no_git = resolve_build_info(False, repo_dir=self.tmp, run=lambda a, c: None)
        self.assertEqual(no_git.title_tag(), "開発版")
        self.assertEqual(no_git.describe(), "開発版（ソースから起動）")

    def test_this_process_has_a_version(self):
        info = get_build_info()
        self.assertIs(info, get_build_info(), "1 回だけ調べて覚えておく")
        self.assertIsInstance(info, BuildInfo)
        self.assertTrue(info.title_tag())
        self.assertTrue(info.describe())

    def test_writer_tool_writes_a_fresh_record(self):
        sys.path.insert(0, os.path.join(ROOT, "tools"))
        try:
            import write_build_info
        finally:
            sys.path.pop(0)
        self._write({"built_at": "2020-01-01 00:00", "commit": "0000000"})  # 前回の記録
        out = io.StringIO()
        old = sys.stdout
        sys.stdout = out
        try:
            self.assertEqual(write_build_info.main(self.tmp), 0)
        finally:
            sys.stdout = old
        data = load_build_file(os.path.join(self.tmp, BUILD_INFO_FILE))
        self.assertIsNotNone(data)
        self.assertNotEqual(data["built_at"], "2020-01-01 00:00", "毎回書き直す")
        self.assertRegex(data["built_at"], r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}$")
        self.assertTrue(out.getvalue().isascii(), ".bat から呼ぶので、画面への出力は英数字だけ")


class TestBuildWiring(unittest.TestCase):
    """配布版を作る手順が、版の記録を書いて同梱するようになっているか"""

    def _read(self, name: str) -> str:
        with open(os.path.join(ROOT, name), "rb") as f:
            return f.read().decode("utf-8")

    def test_batch_writes_the_stamp_before_building(self):
        bat = self._read("EXE配布パッケージ作成.bat")
        i_del = bat.index('del /q "build_info.json"')
        i_write = bat.index("tools\\write_build_info.py")
        i_build = bat.index("-m PyInstaller --noconfirm E57_PointCloud_Tool.spec")
        self.assertLess(i_del, i_write, "古い記録を先に消す（書けなかったときに前回の版を同梱しない）")
        self.assertLess(i_write, i_build, "ビルドの前に書く")
        self.assertTrue(bat.isascii(), ".bat は英数字だけ")

    def test_spec_bundles_the_stamp_and_git_ignores_it(self):
        spec = self._read("E57_PointCloud_Tool.spec")
        self.assertIn("datas.append(('build_info.json', '.'))", spec)
        self.assertIn("os.path.exists('build_info.json')", spec, "無くてもビルドは通す")
        ignore = self._read(".gitignore").splitlines()
        self.assertIn(BUILD_INFO_FILE, ignore)


if __name__ == "__main__":
    unittest.main()
