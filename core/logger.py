"""
システム履歴ログ記録モジュール
ツール起動時の環境情報、ユーザー操作、点群処理、およびエラー例外を恒久的なテキストファイルとして記録します。
"""

import os
import sys
import logging
import platform
import traceback
from datetime import datetime
from typing import Optional


def get_app_dir() -> str:
    """アプリケーションの実行ルートディレクトリを取得 (EXE実行時とPython実行時の両方に対応)"""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(os.path.join(__file__, "..")))


def get_logs_dir() -> str:
    """logs フォルダの絶対パスを取得し、存在しない場合は自動作成"""
    app_dir = get_app_dir()
    logs_dir = os.path.join(app_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    return logs_dir


def get_today_log_path() -> str:
    """当日のログファイルパスを取得"""
    today_str = datetime.now().strftime("%Y-%m-%d")
    return os.path.join(get_logs_dir(), f"system_history_{today_str}.log")


class UTF8FileHandler(logging.FileHandler):
    """Windows環境でも文字化けしないUTF-8ファイルハンドラ"""
    def __init__(self, filename, mode="a", encoding="utf-8", delay=False):
        super().__init__(filename, mode=mode, encoding=encoding, delay=delay)


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


_logger: Optional[logging.Logger] = None


def setup_logging() -> logging.Logger:
    """ロギングシステムを初期化してロガーを返す"""
    global _logger
    if _logger is not None:
        return _logger

    log_path = get_today_log_path()

    logger = logging.getLogger("E57PointCloudTool")
    logger.setLevel(logging.DEBUG)

    # 既存のハンドラをクリア
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s.%(msecs)03d] [%(levelname)-5s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 1. ファイルハンドラ (UTF-8)
    file_handler = UTF8FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 2. コンソールハンドラ (標準出力・文字コード安全)
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


def log_system_environment():
    """ツール起動時のシステム環境情報をログに出力"""
    logger = get_logger()
    logger.info("=" * 70)
    logger.info("🚀 E57点群ノイズ処理ツール 起動セッション開始")
    logger.info(f"・実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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
