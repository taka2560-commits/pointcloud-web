# -*- coding: utf-8 -*-
"""
異常終了の記録: アプリが突然閉じたとき、何が起きたかを次の起動で分かるようにする。Qt に依存しない。

なぜ要るか（2026-11-14）
    Python の例外はログに残るが、次の終わり方は何も残らなかった。
      - 描画ドライバや E57 の読み書きライブラリ（C++）の中で落ちる（アクセス違反など）
      - 応答が無くなって、タスクマネージャーから終了させる
      - メモリ不足で Windows に止められる、電源が切れる
    配布版は窓だけのアプリで、標準エラー出力も無い。利用者から「突然閉じた」と聞いても、手がかりが無かった。

仕組み
    1. 起動のたびに、ログのフォルダへ「起動中の印」（session_<pid>_<日時>.txt）を置き、開いたままにする。
       Python の faulthandler をこのファイルへ向ける。落ちる瞬間に、Python が全スレッドの呼び出し位置を書き足す。
    2. アプリの終了処理が始まったら「ended: normal」と書く。Python が終わるとき（atexit）に、印を消す。
    3. 次の起動で、消えずに残っている印を探す。開いているプロセスが無ければ、前回は正常に終了していない。
         - 「ended: normal」が無い            → 動いている途中で終わった（利用者に知らせる）
         - 「ended: normal」のあとに記録がある → 終了処理の途中で落ちた（ログにだけ残す。利用者は自分で閉じている）
         - 「ended: normal」だけ              → 正常終了。印を消しそこねただけなので、黙って消す
       印は crash_<日時>.txt へ名前を変えて残し、ログの末尾を添える。

    印が「まだ動いているアプリのもの」かどうかは、名前を変えられるかで見分ける（Windows は、開かれているファイルの
    名前を変えさせない）。プロセス番号は使い回されるので、番号だけでは見分けない。
    同時に 2 つ起動していても、それぞれ別の印を持つ。

faulthandler の記録について
    Windows では、ドライバや Windows の部品が自分で処理して続行できる例外でも「Windows fatal exception」と書かれることがある。
    その場合もアプリは動き続ける。なので、記録があるかどうかではなく、印が残っているかどうかで異常終了を判定する。
    終了の原因は、記録のいちばん下のまとまり。要約にも、いちばん下の「… fatal …」の行を出す。
    続行できる例外が大量に出て印が大きくなり続けないよう、SESSION_MAX_BYTES を超えたら、それまでの記録を捨てて
    見出しだけに戻す（check_size。記録そのものは続ける）。
"""

import faulthandler
import os
import re
import sys
import threading
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, List, Optional, Tuple

SESSION_PREFIX = "session_"
CRASH_PREFIX = "crash_"
END_MARK = "ended: normal"
FAULT_HEADER = "--- 以下は、異常終了のときに Python が書き足す記録（faulthandler） ---"
CRASH_KEEP_DAYS = 90
SESSION_MAX_BYTES = 2 * 1024 * 1024     # 印がこれより大きくなったら、それまでの記録を捨てる
READ_HEAD_BYTES = 64 * 1024             # 大きな印を読むときの量（先頭。見出しが入る）
READ_TAIL_BYTES = 512 * 1024            # 　　　　　〃　　　　　（末尾。落ちた瞬間の記録は末尾にある）
REPORT_FAULT_HEAD_BYTES = 8 * 1024      # 報告に写す記録の量（先頭）
REPORT_FAULT_TAIL_BYTES = 64 * 1024     # 　　　　　〃　　　（末尾）
LOG_TAIL_LINES = 40
FATAL_LINE_PREFIXES = ("Windows fatal exception", "Fatal Python error")

_SESSION_RE = re.compile(r"^session_(\d+)_(\d{8})-(\d{6})\.txt$")
_CRASH_RE = re.compile(r"^crash_(\d{4})-(\d{2})-(\d{2})_\d{6}(?:_\d+)?\.txt$")
_LOG_RE = re.compile(r"^system_history_\d{4}-\d{2}-\d{2}\.log$")


@dataclass
class AbandonedSession:
    """正常に終了しなかった、前の起動"""
    report_path: str            # 残した記録（crash_*.txt）
    started: str = ""           # 起動した日時
    pid: int = 0
    build: str = ""
    has_fault_text: bool = False  # faulthandler の記録があるか
    fault_line: str = ""          # 記録のうち、いちばん下の「… fatal …」の行（「Windows fatal exception: access violation」など）
    during_exit: bool = False     # 終了処理が始まったあとで落ちた（利用者は自分で閉じている）

    def summary(self) -> str:
        when = f"{self.started} に起動した回" if self.started else "前の起動"
        if self.during_exit:
            return f"{when}は、終了処理の途中で異常終了していました（記録: {self.fault_line or '内容は記録のファイルを参照'}）"
        what = f"（記録: {self.fault_line}）" if self.fault_line else "（落ちた瞬間の記録は無し。強制終了や電源断など）"
        return f"{when}は、正常に終了していません{what}"


def _posix_pid_alive(pid: int) -> bool:
    """POSIX 専用（Windows の os.kill は、合図 0 でも相手を終了させてしまうので使わない）"""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except OSError:
        return True


def _read_text(path: str, max_bytes: int = 4 * 1024 * 1024) -> str:
    try:
        with open(path, "rb") as f:
            return f.read(max_bytes).decode("utf-8", errors="replace")
    except OSError:
        return ""


def _read_mark(path: str) -> str:
    """
    印を読む。大きいときは先頭（見出し）と末尾（落ちた瞬間の記録）だけを読む。
    先頭から決まった量だけ読むと、続行できた例外で印が膨らんでいたときに、肝心の末尾を読み落とす。
    """
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            if size <= READ_HEAD_BYTES + READ_TAIL_BYTES:
                data = f.read()
            else:
                head = f.read(READ_HEAD_BYTES)
                f.seek(size - READ_TAIL_BYTES)
                tail = f.read(READ_TAIL_BYTES)
                skipped = size - READ_HEAD_BYTES - READ_TAIL_BYTES
                data = head + f"\n\n…（途中 {skipped:,} バイトを省略）…\n\n".encode("utf-8") + tail
        return data.decode("utf-8", errors="replace")
    except OSError:
        return ""


def _header_value(text: str, key: str) -> str:
    m = re.search(rf"^{re.escape(key)}: (.*)$", text, flags=re.MULTILINE)
    return m.group(1).strip() if m else ""


def split_mark(text: str) -> Tuple[str, str, bool, str]:
    """
    印の本文を (見出し, 動いているあいだの記録, 終了処理が始まったか, 終了処理が始まったあとの記録) に分ける。
    「ended: normal」は、1 行まるごとがその文のときだけ数える。
    """
    idx = text.find(FAULT_HEADER)
    if idx < 0:
        return text.strip(), "", False, ""
    header = text[:idx].strip()
    body = text[idx + len(FAULT_HEADER):]
    m = re.search(rf"^{re.escape(END_MARK)}\s*$", body, flags=re.MULTILINE)
    if not m:
        return header, body.strip(), False, ""
    return header, body[:m.start()].strip(), True, body[m.end():].strip()


def main_fault_line(fault: str) -> str:
    """記録のうち、終了の原因を表す 1 行: いちばん下の「… fatal …」の行。無ければ最初の行"""
    lines = [ln.strip() for ln in fault.splitlines() if ln.strip()]
    fatal = [ln for ln in lines if ln.startswith(FATAL_LINE_PREFIXES)]
    if fatal:
        return fatal[-1][:200]
    return lines[0][:200] if lines else ""


def _newest_log_tail(logs_dir: str, n_lines: int = LOG_TAIL_LINES) -> List[str]:
    """いちばん新しいログファイルの末尾（落ちる直前に何をしていたかの手がかり）。先頭の要素はファイル名"""
    try:
        names = sorted(n for n in os.listdir(logs_dir) if _LOG_RE.match(n))
    except OSError:
        return []
    if not names:
        return []
    name = names[-1]
    path = os.path.join(logs_dir, name)
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            if size > 1024 * 1024:
                f.seek(size - 1024 * 1024)
            lines = f.read().decode("utf-8", errors="replace").splitlines()
    except OSError:
        return [name]
    return [name] + lines[-n_lines:]


def _shorten(text: str, head: int, tail: int) -> str:
    if len(text) <= head + tail:
        return text
    return text[:head] + f"\n\n…（途中 {len(text) - head - tail:,} 文字を省略）…\n\n" + text[-tail:]


def _unique_path(path: str) -> str:
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    for i in range(2, 1000):
        cand = f"{base}_{i}{ext}"
        if not os.path.exists(cand):
            return cand
    return path


def _remove_quietly(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def collect_abandoned(logs_dir: str, now: Optional[datetime] = None) -> List[AbandonedSession]:
    """
    残っている「起動中の印」を調べる。正常に終了しなかったものは crash_*.txt として残し、一覧を返す。
    まだ動いているアプリの印（名前を変えられない）には触らない。
    見出しの無い印（書き込みに失敗した残り）と、正常終了の書き込みだけがある印は、黙って消す。
    """
    out: List[AbandonedSession] = []
    try:
        names = sorted(os.listdir(logs_dir))
    except OSError:
        return out
    tail = None
    for name in names:
        m = _SESSION_RE.match(name)
        if not m:
            continue
        src = os.path.join(logs_dir, name)
        if os.name != "nt" and _posix_pid_alive(int(m.group(1))):
            continue
        stamp = f"{m.group(2)[:4]}-{m.group(2)[4:6]}-{m.group(2)[6:]}_{m.group(3)}"
        dest = _unique_path(os.path.join(logs_dir, f"{CRASH_PREFIX}{stamp}.txt"))
        try:
            os.replace(src, dest)  # 開いているプロセスがあれば、Windows はここで断る
        except OSError:
            continue
        text = _read_mark(dest)
        header, running_fault, ended, exit_fault = split_mark(text)
        if not _header_value(header, "started"):
            _remove_quietly(dest)  # 見出しを書けなかった印（ディスクがいっぱい、など）。起動していたとは言えない
            continue
        if ended and not exit_fault:
            _remove_quietly(dest)  # 正常に終了したが、印を消しそこねただけ
            continue
        fault = exit_fault if ended else running_fault
        item = AbandonedSession(
            report_path=dest,
            started=_header_value(header, "started"),
            pid=int(m.group(1)),
            build=_header_value(header, "build"),
            has_fault_text=bool(fault),
            fault_line=main_fault_line(fault),
            during_exit=ended,
        )
        if tail is None:
            tail = _newest_log_tail(logs_dir)
        _write_report(dest, header, fault, ended, tail, now or datetime.now())
        out.append(item)
    return out


def _write_report(path: str, header: str, fault: str, during_exit: bool, log_tail: List[str], now: datetime) -> None:
    """残した印を、読める形の記録に書き直す（見出し・起動時の情報・落ちた瞬間の記録・ログの末尾）"""
    lines = [
        "E57 点群ノイズ処理ツール: 正常に終了しなかった起動の記録",
        f"見つけた日時: {now.strftime('%Y-%m-%d %H:%M:%S')}（次の起動のとき）",
        "終わり方: " + ("終了処理の途中で落ちた（アプリを閉じる操作のあと）" if during_exit else "動いている途中で終わった"),
        "",
        "■ その起動の情報",
        header,
        "",
        "■ 落ちた瞬間の記録（Python が書き足したもの）",
    ]
    if fault:
        lines += [
            "※ 英語の記録です。「Current thread」や「Thread 0x…」の下が、そのとき実行していた場所です。",
            "※ 続行できた例外も同じ形で書かれます。終了の原因は、いちばん下のまとまりです。",
            "",
            _shorten(fault, REPORT_FAULT_HEAD_BYTES, REPORT_FAULT_TAIL_BYTES),
        ]
    else:
        lines.append("（記録なし。強制終了・電源断・Windows による停止など、Python が何も書けない終わり方でした）")
    lines += ["", f"■ ログの末尾（{log_tail[0] if log_tail else 'ログなし'}。別の起動の行が混ざることがあります）"]
    lines += log_tail[1:] if log_tail else []
    try:
        with open(path, "w", encoding="utf-8", errors="replace") as f:
            f.write("\n".join(lines) + "\n")
    except OSError:
        pass


def cleanup_old_reports(logs_dir: str, keep_days: int = CRASH_KEEP_DAYS, today=None) -> List[str]:
    """keep_days 日より古い crash_*.txt を消す（名前の日付で判断する。ほかのファイルには触らない）"""
    removed: List[str] = []
    if keep_days <= 0:
        return removed
    limit = (today or datetime.now().date()) - timedelta(days=int(keep_days))
    try:
        names = sorted(os.listdir(logs_dir))
    except OSError:
        return removed
    for name in names:
        m = _CRASH_RE.match(name)
        if not m:
            continue
        try:
            day = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3))).date()
        except ValueError:
            continue
        if day >= limit:
            continue
        try:
            os.remove(os.path.join(logs_dir, name))
            removed.append(name)
        except OSError:
            pass
    return removed


@dataclass
class Session:
    """いまの起動の「起動中の印」"""
    logs_dir: str
    path: str = ""
    abandoned: List[AbandonedSession] = field(default_factory=list)
    error: str = ""             # 前回までの印を調べるときに起きた、想定外の失敗（ログへ残すための文）
    trimmed: int = 0            # 大きくなった記録を捨てた回数
    _fh: Optional[object] = None
    _header_len: int = 0
    _fault_enabled: bool = False
    _ended: bool = False
    _closed: bool = False

    @property
    def active(self) -> bool:
        """印を置けていて、落ちた瞬間の記録が有効か"""
        return self._fh is not None and self._fault_enabled

    def mark_clean_exit(self) -> None:
        """終了処理が始まった、と書く（アプリの aboutToQuit で呼ぶ。Windows の終了・ログオフでも呼ばれる）"""
        if self._fh is None or self._ended:
            return
        self._ended = True
        try:
            self._fh.write(("\n" + END_MARK + "\n").encode("utf-8"))
            self._fh.flush()
        except (OSError, ValueError):
            pass

    def fault_bytes(self) -> int:
        """faulthandler が書き足した量（続行できた例外の記録も含む）"""
        if not self.path:
            return 0
        _, running, _, after = split_mark(_read_mark(self.path))
        return len((running + after).encode("utf-8"))

    def check_size(self) -> bool:
        """
        印が大きくなりすぎていたら、それまでの記録を捨てて見出しだけに戻す。捨てたら True。
        記録そのものは止めない（止めると、そのあとの本当の異常終了を書けなくなる）。
        """
        if self._fh is None or self._ended or not self.path:
            return False
        try:
            size = os.path.getsize(self.path)
        except OSError:
            return False
        if size <= SESSION_MAX_BYTES:
            return False
        try:
            self._fh.truncate(self._header_len)
            note = f"（{datetime.now().strftime('%H:%M:%S')} に、続行できた例外の記録 {size - self._header_len:,} バイトを捨てました）\n"
            self._fh.write(note.encode("utf-8"))
            self._fh.flush()
        except (OSError, ValueError):
            return False
        self.trimmed += 1
        return True

    def end(self) -> None:
        """正常終了: 記録を止め、印を消す（Python が終わるときに atexit から呼ぶ。2 回呼んでも困らない）"""
        if self._closed:
            return
        self._closed = True
        self.mark_clean_exit()
        if self._fault_enabled:
            try:
                faulthandler.disable()
            except Exception:
                pass
            self._fault_enabled = False
        if self._fh is not None:
            try:
                self._fh.close()
            except OSError:
                pass
            self._fh = None
        if self.path:
            _remove_quietly(self.path)


def _session_header(build: str, now: datetime) -> str:
    return "\n".join(
        [
            "E57 点群ノイズ処理ツール: 起動中の印（正常に終了すると消えます）",
            f"started: {now.strftime('%Y-%m-%d %H:%M:%S')}",
            f"pid: {os.getpid()}",
            f"build: {build}",
            f"exe: {sys.executable}",
            f"python: {sys.version.split()[0]}",
            FAULT_HEADER,
            "",
        ]
    )


def begin_session(logs_dir: str, build: str = "", now: Optional[datetime] = None) -> Session:
    """
    起動時に 1 回呼ぶ。前回までの異常終了を調べ（Session.abandoned）、今回の印を置いて faulthandler を有効にする。
    何が起きても例外は外へ出さない（調べる仕掛けのせいで、アプリが起動しなくなってはいけない）。
    印を置けなかったときは Session.active が False になるだけ。
    """
    now = now or datetime.now()
    session = Session(logs_dir=logs_dir)
    try:
        session.abandoned = collect_abandoned(logs_dir, now)
    except Exception as e:
        session.abandoned = []
        session.error = f"前回までの起動の印を調べられませんでした: {e!r}"
    try:
        cleanup_old_reports(logs_dir)
    except Exception:
        pass
    path = os.path.join(logs_dir, f"{SESSION_PREFIX}{os.getpid()}_{now.strftime('%Y%m%d-%H%M%S')}.txt")
    fh = None
    try:
        os.makedirs(logs_dir, exist_ok=True)
        fh = open(path, "ab", buffering=0)  # 開いたままにする（開いている間は、ほかの起動が名前を変えられない）
        header = _session_header(build, now).encode("utf-8")
        fh.write(header)
        session.path = path
        session._fh = fh
        session._header_len = len(header)
    except Exception:
        # 見出しを書けなかった印を残すと、次の起動が「前回は正常に終了していません」と誤って知らせる
        if fh is not None:
            try:
                fh.close()
            except OSError:
                pass
            _remove_quietly(path)
        return session
    try:
        faulthandler.enable(file=session._fh, all_threads=True)
        session._fault_enabled = True
    except Exception:
        session._fault_enabled = False
    return session


def install_exception_hooks(log_critical: Callable[[str], None], log_error: Callable[[str], None]) -> None:
    """
    メインスレッド以外で起きた例外と、後始末の中で起きた例外をログへ残す。
    （メインスレッドの未捕捉例外は core/logger.py の sys.excepthook が残す。スレッドの中の例外は、
    標準エラー出力へ出るだけで、窓だけの配布版ではどこにも残らなかった）
    """

    def thread_hook(args):
        if args.exc_type is SystemExit:
            return
        name = args.thread.name if args.thread is not None else "不明"
        tb = "".join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback))
        try:
            log_critical(f"【スレッド「{name}」の未捕捉例外】\n{tb}")
        except Exception:
            pass

    def unraisable_hook(unraisable):
        try:
            tb = "".join(
                traceback.format_exception(unraisable.exc_type, unraisable.exc_value, unraisable.exc_traceback)
            )
            where = unraisable.err_msg or "後始末の中で起きた例外"
            log_error(f"【{where}】{unraisable.object!r}\n{tb}")
        except Exception:
            pass

    threading.excepthook = thread_hook
    sys.unraisablehook = unraisable_hook


def describe_for_log(session: Session) -> List[str]:
    """起動時のログに書く行（前回までの異常終了）"""
    lines: List[str] = []
    if session.error:
        lines.append(session.error)
    for a in session.abandoned:
        lines.append(f"{a.summary()}。記録: {a.report_path}")
    return lines
