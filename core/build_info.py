# -*- coding: utf-8 -*-
"""
版の表示（どの版で動いているかを、画面とログで分かるようにする）。Qt に依存しない。

なぜ要るか（2026-11-13）
    配布版は 1 日に何度も作り直すことがある（この日は 3 回）。画面にもログにも版が出ていなかったので、
    画面の写しやログが「どの版で起きたことか」を区別できなかった。

仕組み
    - 配布版（EXE）: 作るときに tools/write_build_info.py が build_info.json（作った日時とコミット）を書き、
      PyInstaller が同梱する（E57_PointCloud_Tool.spec）。実行時はそれを読む。
    - ソースから起動: git に聞く（1 プロセスで 1 回だけ。git が無い・遅いときは「開発版」とだけ出す）。
      リポジトリに残っている build_info.json は前回のビルドのものなので読まない。

出す場所: ウィンドウのタイトル、バージョン情報、設定 → システム、ログの先頭。
"""

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, List, Optional

BUILD_INFO_FILE = "build_info.json"
_GIT_TIMEOUT_SEC = 3.0
_TIME_FORMAT = "%Y-%m-%d %H:%M"


@dataclass(frozen=True)
class BuildInfo:
    built_at: str = ""      # 配布版を作った日時（"2026-11-13 22:16"）。ソースから起動したときは ""
    commit: str = ""        # コミットの短い番号（分からなければ ""）
    commit_date: str = ""   # そのコミットの日時
    dirty: bool = False     # コミットしていない変更を含むか
    frozen: bool = False    # 配布版（EXE）か

    def title_tag(self) -> str:
        """ウィンドウのタイトルに添える短い表示"""
        if self.frozen:
            return f"版 {self.built_at}" if self.built_at else "版の記録なし"
        if self.commit:
            return f"開発版 {self.commit}" + ("＋" if self.dirty else "")
        return "開発版"

    def describe(self) -> str:
        """バージョン情報・設定・ログに出す 1 行"""
        if self.frozen:
            if not self.built_at:
                return "配布版（作成日時の記録なし）"
            text = f"配布版 {self.built_at} 作成"
            if self.commit:
                text += f"（{self.commit}" + ("＋未コミットの変更" if self.dirty else "") + "）"
            return text
        if not self.commit:
            return "開発版（ソースから起動）"
        text = f"開発版 {self.commit}"
        if self.commit_date:
            text += f"（{self.commit_date} のコミット"
            text += "＋未コミットの変更）" if self.dirty else "）"
        elif self.dirty:
            text += "（未コミットの変更あり）"
        return text


def _run_git(args: List[str], cwd: str) -> Optional[str]:
    """git を 1 回呼ぶ。失敗・時間切れ・git が無いときは None（例外は出さない）"""
    kwargs = {}
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)  # 黒い窓を出さない
    try:
        out = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=_GIT_TIMEOUT_SEC, stdin=subprocess.DEVNULL, **kwargs,
        )
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return (out.stdout or "").strip()


def read_git_info(repo_dir: str, run: Optional[Callable[[List[str], str], Optional[str]]] = None) -> Dict[str, object]:
    """リポジトリの今のコミット（短い番号・日時・未コミットの変更の有無）。分からなければ空の値"""
    run = run or _run_git
    empty: Dict[str, object] = {"commit": "", "commit_date": "", "dirty": False}
    head = run(["log", "-1", "--format=%h|%cd", f"--date=format-local:{_TIME_FORMAT}"], repo_dir)
    if not head or "|" not in head:
        return empty
    commit, commit_date = head.split("|", 1)
    commit = commit.strip()
    if not commit or len(commit) > 40 or not all(c in "0123456789abcdefABCDEF" for c in commit):
        return empty
    status = run(["status", "--porcelain", "--untracked-files=no"], repo_dir)
    return {"commit": commit, "commit_date": commit_date.strip(), "dirty": bool(status)}


def make_build_record(repo_dir: str, now: Optional[datetime] = None, run=None) -> Dict[str, object]:
    """配布版に埋め込む記録（tools/write_build_info.py が build_info.json に書く）"""
    record = read_git_info(repo_dir, run)
    record["built_at"] = (now or datetime.now()).strftime(_TIME_FORMAT)
    return record


def load_build_file(path: str) -> Optional[Dict[str, object]]:
    """build_info.json を読む。無い・壊れているときは None"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not str(data.get("built_at", "")).strip():
        return None
    return data


def _frozen_candidates() -> List[str]:
    out = []
    meipass = getattr(sys, "_MEIPASS", "")
    if meipass:
        out.append(os.path.join(meipass, BUILD_INFO_FILE))  # PyInstaller 6 は同梱ファイルを _internal に置く
    out.append(os.path.join(os.path.dirname(os.path.abspath(sys.executable)), BUILD_INFO_FILE))
    return out


def resolve_build_info(frozen: bool, candidates: Optional[List[str]] = None, repo_dir: Optional[str] = None, run=None) -> BuildInfo:
    """版の情報を求める（引数で状況を渡せる形。テストはこれを使う）"""
    if frozen:
        for path in (candidates if candidates is not None else _frozen_candidates()):
            data = load_build_file(path)
            if data is not None:
                return BuildInfo(
                    built_at=str(data.get("built_at", "")).strip(),
                    commit=str(data.get("commit", "") or "").strip(),
                    commit_date=str(data.get("commit_date", "") or "").strip(),
                    dirty=bool(data.get("dirty", False)),
                    frozen=True,
                )
        return BuildInfo(frozen=True)
    if repo_dir is None:
        repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    info = read_git_info(repo_dir, run)
    return BuildInfo(commit=str(info["commit"]), commit_date=str(info["commit_date"]), dirty=bool(info["dirty"]), frozen=False)


_cached: Optional[BuildInfo] = None


def get_build_info() -> BuildInfo:
    """このプロセスの版の情報（1 回だけ調べて覚えておく）"""
    global _cached
    if _cached is None:
        try:
            _cached = resolve_build_info(bool(getattr(sys, "frozen", False)))
        except Exception:
            _cached = BuildInfo(frozen=bool(getattr(sys, "frozen", False)))
    return _cached
