"""
システム履歴ログ記録モジュール
ツール起動時の環境情報、ユーザー操作、点群処理、およびエラー例外を恒久的なテキストファイルとして記録します。

ログの置き場所（2026-11-13 に変更）
    - 配布版（EXE）: 利用者ごとのフォルダ（Windows は %LOCALAPPDATA%\\E57PointCloudTool\\logs）。
      以前は EXE と同じフォルダの logs だった。配布版を作り直すとフォルダごと置き換わり、それまでの記録が消えていた
      （2026-11-13、利用者が 18:10 ごろに行った作業の記録が、その日の作り直しで残っていなかった）。
    - ソースから起動: これまでどおり、リポジトリの logs フォルダ。
    - 環境変数 E57_LOG_DIR があれば、そのフォルダ（起動確認やテストで、ふだんの記録と分けたいときに使う）。
    - 書き込めないときは、利用者ごとのフォルダ → 一時フォルダの順に替える（起動は止めない）。
ファイルは 1 日 1 つ（system_history_YYYY-MM-DD.log）。日付が変わったら次の 1 行から新しい日のファイルへ書く。
LOG_KEEP_DAYS 日より古いファイルは、起動時に消す。
"""

import os
import re
import sys
import logging
import platform
import tempfile
import traceback
from datetime import date, datetime, timedelta
from typing import List, Optional, Tuple

APP_DATA_DIRNAME = "E57PointCloudTool"
LOG_DIR_ENV = "E57_LOG_DIR"
LOG_KEEP_DAYS = 90
_LOG_NAME_RE = re.compile(r"^system_history_(\d{4})-(\d{2})-(\d{2})\.log$")


def get_app_dir() -> str:
    """アプリケーションの実行ルートディレクトリを取得 (EXE実行時とPython実行時の両方に対応)"""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(os.path.join(__file__, "..")))


def get_user_data_dir() -> str:
    """
    利用者ごとのデータ置き場（Windows: %LOCALAPPDATA%\\E57PointCloudTool）。
    配布版を作り直しても、ZIP を別の場所に解凍しても変わらない。
    """
    base = os.environ.get("LOCALAPPDATA", "").strip()
    if not base:
        home = os.path.expanduser("~")
        base = os.path.join(home, "AppData", "Local") if os.name == "nt" else os.path.join(home, ".local", "share")
    return os.path.join(base, APP_DATA_DIRNAME)


def candidate_log_dirs(frozen: bool, app_dir: str, env_dir: str = "", user_dir: Optional[str] = None,
                       temp_dir: Optional[str] = None) -> List[str]:
    """ログの置き場所の候補（先頭から順に、書き込める最初のものを使う）"""
    user_logs = os.path.join(user_dir if user_dir is not None else get_user_data_dir(), "logs")
    app_logs = os.path.join(app_dir, "logs")
    temp_logs = os.path.join(temp_dir if temp_dir is not None else tempfile.gettempdir(), APP_DATA_DIRNAME, "logs")
    out: List[str] = []
    if env_dir:
        out.append(os.path.abspath(env_dir))
    out += [user_logs, app_logs] if frozen else [app_logs, user_logs]
    out.append(temp_logs)
    seen = set()
    return [d for d in out if not (os.path.normcase(d) in seen or seen.add(os.path.normcase(d)))]


def _can_write_dir(path: str) -> bool:
    """フォルダを（無ければ作って）実際に書き込めるか確かめる"""
    probe = os.path.join(path, f".write_test_{os.getpid()}")
    try:
        os.makedirs(path, exist_ok=True)
        with open(probe, "w", encoding="utf-8"):
            pass
        os.remove(probe)
        return True
    except OSError:
        try:
            if os.path.exists(probe):
                os.remove(probe)
        except OSError:
            pass
        return False


_logs_dir_cache: Optional[Tuple[Tuple[str, bool, str], str]] = None


def get_logs_dir() -> str:
    """ログのフォルダの絶対パス（無ければ作る）。書き込める最初の候補を選び、同じ条件のあいだは覚えておく"""
    global _logs_dir_cache
    env_dir = os.environ.get(LOG_DIR_ENV, "").strip()
    frozen = bool(getattr(sys, "frozen", False))
    app_dir = get_app_dir()
    key = (env_dir, frozen, app_dir)
    if _logs_dir_cache is not None and _logs_dir_cache[0] == key and os.path.isdir(_logs_dir_cache[1]):
        return _logs_dir_cache[1]
    candidates = candidate_log_dirs(frozen, app_dir, env_dir)
    for d in candidates:
        if _can_write_dir(d):
            _logs_dir_cache = (key, d)
            return d
    return candidates[-1]  # どこにも書けない（ファイルへは残せないが、起動は止めない）


def get_today_log_path() -> str:
    """当日のログファイルパスを取得"""
    today_str = datetime.now().strftime("%Y-%m-%d")
    return os.path.join(get_logs_dir(), f"system_history_{today_str}.log")


class UTF8FileHandler(logging.FileHandler):
    """Windows環境でも文字化けしないUTF-8ファイルハンドラ（1 つのファイルへ書き続ける。日付で分けるのは DailyFileHandler）"""
    def __init__(self, filename, mode="a", encoding="utf-8", delay=False):
        super().__init__(filename, mode=mode, encoding=encoding, delay=delay)


class DailyFileHandler(logging.Handler):
    """
    日付ごとのファイル（system_history_YYYY-MM-DD.log、UTF-8）へ書く。
    日付が変わったら、次の 1 行から新しい日のファイルへ切り替える
    （以前は起動した日のファイルへ書き続け、日をまたいだ作業の記録が前の日のファイルに入っていた）。
    """

    def __init__(self, logs_dir: str, encoding: str = "utf-8"):
        super().__init__()
        self.logs_dir = logs_dir
        self.encoding = encoding
        self._day = ""
        self._stream = None

    def path_for(self, day: str) -> str:
        return os.path.join(self.logs_dir, f"system_history_{day}.log")

    @property
    def current_path(self) -> str:
        return self.path_for(self._day or datetime.now().strftime("%Y-%m-%d"))

    def _switch(self, day: str) -> None:
        if self._stream is not None:
            try:
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        os.makedirs(self.logs_dir, exist_ok=True)
        self._stream = open(self.path_for(day), "a", encoding=self.encoding, errors="replace")
        self._day = day

    def emit(self, record):
        try:
            day = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d")
            if self._stream is None or day != self._day:
                self._switch(day)
            self._stream.write(self.format(record) + "\n")
            self._stream.flush()
        except Exception:
            self.handleError(record)

    def close(self):
        self.acquire()
        try:
            if self._stream is not None:
                try:
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None
        finally:
            self.release()
        super().close()


class SafeStreamHandler(logging.StreamHandler):
    """Windowsのcp932コンソールでも絵文字で落ちない安全なストリームハンドラ"""
    def emit(self, record):
        try:
            msg = self.format(record)
            enc = getattr(self.stream, "encoding", "cp932") or "cp932"
            safe_msg = msg.encode(enc, errors="replace").decode(enc)
            self.stream.write(safe_msg + self.terminator)
            self.flush()
        except Exception:
            pass


def cleanup_old_logs(logs_dir: str, keep_days: int = LOG_KEEP_DAYS, today: Optional[date] = None) -> List[str]:
    """
    keep_days 日より古いログファイルを消す。消したファイル名を返す。
    消すのは、名前が system_history_YYYY-MM-DD.log の形で、その日付が古いものだけ（ほかのファイルには触らない）。
    """
    if keep_days <= 0:
        return []
    limit = (today or date.today()) - timedelta(days=int(keep_days))
    removed: List[str] = []
    try:
        names = os.listdir(logs_dir)
    except OSError:
        return removed
    for name in sorted(names):
        m = _LOG_NAME_RE.match(name)
        if not m:
            continue
        try:
            day = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            continue
        if day >= limit:
            continue
        path = os.path.join(logs_dir, name)
        try:
            if os.path.isfile(path):
                os.remove(path)
                removed.append(name)
        except OSError:
            pass
    return removed


_logger: Optional[logging.Logger] = None


def setup_logging() -> logging.Logger:
    """ロギングシステムを初期化してロガーを返す"""
    global _logger
    if _logger is not None:
        return _logger

    logs_dir = get_logs_dir()

    logger = logging.getLogger("E57PointCloudTool")
    logger.setLevel(logging.DEBUG)

    # 既存のハンドラをクリア
    if logger.hasHandlers():
        for h in list(logger.handlers):
            try:
                h.close()
            except Exception:
                pass
        logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s.%(msecs)03d] [%(levelname)-5s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 1. ファイルハンドラ (UTF-8、1 日 1 ファイル)
    file_handler = DailyFileHandler(logs_dir)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 2. コンソールハンドラ (標準出力・文字コード安全)。窓だけの配布版には標準出力が無い
    if sys.stdout is not None:
        console_handler = SafeStreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    _logger = logger

    # 未捕捉例外のグローバルフックを設定
    def handle_unhandled_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        tb_lines = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        logger.critical(f"【致命的システムエラー / 未捕捉例外】\n{tb_lines}")

    sys.excepthook = handle_unhandled_exception

    # 3. 古いログの片付け（失敗しても起動は続ける）
    try:
        removed = cleanup_old_logs(logs_dir)
        if removed:
            logger.info(f"{LOG_KEEP_DAYS} 日より古いログを {len(removed)} 個消しました（{removed[0]} 〜 {removed[-1]}）")
    except Exception:
        pass

    return logger


def get_logger() -> logging.Logger:
    """現在のロガーを取得 (未初期化の場合は自動初期化)"""
    global _logger
    if _logger is None:
        return setup_logging()
    return _logger


def log_info(msg: str):
    """情報ログを記録"""
    get_logger().info(msg)


def log_warning(msg: str):
    """警告ログを記録"""
    get_logger().warning(msg)


def log_error(msg: str):
    """エラーログを記録"""
    get_logger().error(msg)


def log_exception(msg: str):
    """例外トレースバック付きエラーログを記録"""
    get_logger().exception(msg)


def _environment_lines() -> List[str]:
    """版・ディスク・メモリ（不具合を調べるとき、どの版で・どれだけ余裕があったかが分かるようにする）"""
    lines: List[str] = []
    try:
        from core.build_info import get_build_info

        lines.append(f"・版: {get_build_info().describe()}")
    except Exception:
        pass
    try:
        from core.sysinfo import disk_status, drive_name, format_bytes, memory_status

        logs_dir = get_logs_dir()
        disk = disk_status(logs_dir)
        if disk is not None:
            lines.append(f"・ディスク: {drive_name(logs_dir)} 空き {format_bytes(disk[1])} / 全体 {format_bytes(disk[0])}")
        mem = memory_status()
        if mem is not None:
            lines.append(f"・メモリ: 搭載 {format_bytes(mem[0])} / 空き {format_bytes(mem[1])}")
    except Exception:
        pass
    return lines


def log_system_environment():
    """ツール起動時のシステム環境情報をログに出力"""
    logger = get_logger()
    logger.info("=" * 70)
    logger.info("🚀 E57点群ノイズ処理ツール 起動セッション開始")
    logger.info(f"・実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    for line in _environment_lines():
        logger.info(line)
    logger.info(f"・OS: {platform.system()} {platform.release()} ({platform.version()})")
    logger.info(f"・マシンアーキテクチャ: {platform.machine()} / プロセッサ: {platform.processor()}")
    logger.info(f"・Pythonバージョン: {sys.version.split()[0]}")
    logger.info(f"・実行形態: {'スタンドアロンEXE (Frozen)' if getattr(sys, 'frozen', False) else 'Pythonスクリプト'}")
    logger.info(f"・実行パス: {sys.executable}")
    logger.info(f"・作業ディレクトリ: {os.getcwd()}")
    logger.info(f"・ログ保存先: {get_today_log_path()}")
    logger.info("=" * 70)


def open_log_folder() -> bool:
    """ログフォルダをOSのエクスプローラーで開く"""
    logs_dir = get_logs_dir()
    try:
        if sys.platform == "win32":
            os.startfile(logs_dir)
            return True
        elif sys.platform == "darwin":
            import subprocess
            subprocess.Popen(["open", logs_dir])
            return True
        else:
            import subprocess
            subprocess.Popen(["xdg-open", logs_dir])
            return True
    except Exception as e:
        log_error(f"ログフォルダを開く際にエラーが発生しました: {e}")
        return False
