# -*- coding: utf-8 -*-
"""
不具合を調べるための記録（画面側）: Qt が出す警告をログへ残す、前回の異常終了を起動時に知らせる

なぜ要るか（2026-11-14）
    - Qt（画面の部品と OpenGL）の警告や致命的なエラーは、標準エラー出力へ出るだけだった。配布版は窓だけのアプリで
      標準エラー出力が無いので、「OpenGL の準備に失敗した」のような Qt の言い分が、どこにも残らなかった。
    - アプリが突然閉じたことは core/crashlog.py が次の起動で見つける。見つけたことを、利用者にも一言知らせる。

main.py から呼ぶ。テストや品質ゲートは main.py を通らないので、ここの仕掛けは入らない。
"""

from typing import Dict, List, Optional

from PySide6.QtCore import QTimer, QtMsgType, qInstallMessageHandler

from core.logger import get_logger, get_logs_dir, open_log_folder

QT_MAX_REPEAT = 3            # 同じ文は、この回数まで残す（描画のたびに出る警告でログを埋めない）
QT_MAX_DISTINCT = 500        # 覚えておく文の種類の上限
PREVIOUS_CRASH_TOAST_MS = 12000

# Qt の内部から出る、害が無いと分かっている警告（ログには DEBUG で残すが、警告としては数えない）
QT_KNOWN_HARMLESS = (
    # メニュー付きのボタン（保存ボタンの右の「▼」）を Windows 11 の見た目で最初に描くとき、Qt が文字の大きさを
    # ポイントで取ろうとする。このアプリは大きさをピクセルで決めているので -1 が返り、Qt 自身がこの警告を出す
    # （Qt 6.11 の qwindows11style。2026-11-14 に、起動のたびに 1 回出ることを確かめた。描画は正しい）
    "QFont::setPointSize: Point size <= 0 (-1)",
)

_installed = False
_seen: Dict[str, int] = {}


def qt_message_level(mode, message: str = "") -> Optional[str]:
    """Qt の知らせの種類を、ログの重さにする（デバッグ用とお知らせは残さない）"""
    if mode == QtMsgType.QtWarningMsg:
        if str(message).strip().startswith(QT_KNOWN_HARMLESS):
            return "debug"
        return "warning"
    if mode == QtMsgType.QtCriticalMsg:
        return "error"
    if mode == QtMsgType.QtFatalMsg:
        return "critical"
    return None


def format_qt_message(mode, context, message: str) -> str:
    where = ""
    try:
        category = getattr(context, "category", "") or ""
        if category and category != "default":
            where = f"（{category}）"
    except Exception:
        pass
    return f"[Qt]{where} {str(message).strip()}"


def _handle_qt_message(mode, context, message) -> None:
    try:
        level = qt_message_level(mode, message)
        if level is None:
            return
        text = format_qt_message(mode, context, message)
        count = _seen.get(text, 0) + 1
        if text in _seen or len(_seen) < QT_MAX_DISTINCT:
            _seen[text] = count
        if count > QT_MAX_REPEAT and level != "critical":
            return
        if count == QT_MAX_REPEAT and level != "critical":
            text += "（同じ文は、これ以降は省きます）"
        logger = get_logger()
        getattr(logger, level)(text)
    except Exception:
        pass  # 記録の失敗で、アプリを止めない


def install_qt_message_log() -> None:
    """Qt の警告・エラー・致命的なエラーを、システム履歴ログへ残す"""
    global _installed
    if _installed:
        return
    qInstallMessageHandler(_handle_qt_message)
    _installed = True


def uninstall_qt_message_log() -> None:
    """終了のときに外す（Python の後始末が始まってから、Qt が Python の関数を呼ばないようにする）"""
    global _installed
    if not _installed:
        return
    qInstallMessageHandler(None)
    _installed = False
    _seen.clear()


def previous_crash_text(abandoned: List) -> str:
    """
    前回の異常終了を知らせる文（利用者向け。原因は断定しない）。
    終了処理の途中で落ちた回（during_exit）は数えない。利用者は自分で閉じていて、失ったものが無いので、ログにだけ残す。
    """
    abandoned = [a for a in abandoned if not getattr(a, "during_exit", False)]
    if not abandoned:
        return ""
    if len(abandoned) == 1:
        return "前回は正常に終了しませんでした。記録をログのフォルダに残しました"
    return f"正常に終了しなかった起動が {len(abandoned)} 回ありました。記録をログのフォルダに残しました"


def show_previous_crash_notice(win, abandoned: List, delay_ms: int = 1500) -> bool:
    """
    前回の異常終了を、起動した画面で一言知らせる（押して閉じる窓は出さない）。
    ログの欄には記録の場所を書き、画面の中の通知には「ログを開く」を付ける。知らせたら True。
    """
    text = previous_crash_text(abandoned)
    if not text:
        return False
    try:
        log_text = getattr(win, "log_text", None)
        if log_text is not None:
            log_text.append(f"{text}（{get_logs_dir()}）")
            for a in [x for x in abandoned if not getattr(x, "during_exit", False)][:5]:
                log_text.append(f"　・{a.summary()}")
    except RuntimeError:
        return False

    def show() -> None:
        try:
            toast = getattr(win, "show_toast", None)
            if not callable(toast):
                return
            try:
                toast(text, PREVIOUS_CRASH_TOAST_MS, action=("ログを開く", open_log_folder, "記録のあるフォルダを開きます"))
            except TypeError:
                toast(text, PREVIOUS_CRASH_TOAST_MS)  # 従来 UI の通知は、ボタンを付けられない
        except RuntimeError:
            pass  # 待っているあいだに画面が閉じられた

    QTimer.singleShot(max(0, int(delay_ms)), win, show)
    return True
