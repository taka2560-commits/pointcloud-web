# -*- coding: utf-8 -*-
"""
長い処理が終わったことを、ほかのアプリを使っている利用者へ知らせる

なぜ要るか（2026-11-14）
    5.4 億点のファイルでは、読み込みに数分、高精細表示のデータ作成に約 5 分、全点で保存に約 10 分かかる。
    待つあいだは別のアプリで作業するのがふつうだが、終わったかどうかは、このアプリの窓へ戻って見るまで分からなかった。

何をするか
    待たせる処理（読み込み・保存・自動ノイズ処理・解析・作業用データの作成）は、どれも進捗の窓
    （gui/progress_dialog.py: ModernProgressDialog）を出す。その窓が出てから消えるまでを 1 つの処理として時間を測る。
    NOTIFY_MIN_SECONDS より長くかかり、終わったときにこのアプリが手前に無ければ、
      - タスクバーのボタンを光らせる（窓へ戻るまで）
      - Windows の通知を出す（通知を押すと、このアプリが手前に出る）
    アプリを見ているときは何もしない（画面の中の通知で足りる）。エラーで止まったときも同じように知らせる。
    設定 → システムのチェックで切れる（QSettings の notify/long_task。既定は入）。

ホットファイル（gui/main_window.py）は変えない。進捗の窓の側に「出た・消えた」の知らせを足し、ここで受ける。
"""

import os
import time
from typing import Callable, Optional, Tuple

from PySide6.QtCore import QObject, QSettings, Qt, QTimer
from PySide6.QtGui import QColor, QGuiApplication, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import QApplication, QCheckBox, QMessageBox, QSystemTrayIcon

from core.logger import log_info
from gui.progress_dialog import add_progress_listener
from gui.status_feedback import classify_message

_SETTINGS_ORG = "E57PointCloudTool"
_SETTINGS_APP = "Settings"
NOTIFY_KEY = "notify/long_task"

NOTIFY_MIN_SECONDS = 20.0   # これより短い処理は知らせない
SETTLE_MS = 400             # 進捗の窓が消えてから、続きの窓が出ないかを待つ時間
MAX_BUSY_WAITS = 25         # 窓が消えたあとも裏の処理が動いているとき、待ち直す回数（× SETTLE_MS）
NOTIFY_TITLE = "E57 点群ノイズ処理ツール"
MAX_BODY_CHARS = 160
NO_OS_NOTIFY_ENV = "E57_NO_OS_NOTIFY"  # set by tests: never raise real Windows notifications from automated runs


def get_notify_enabled() -> bool:
    return bool(QSettings(_SETTINGS_ORG, _SETTINGS_APP).value(NOTIFY_KEY, True, type=bool))


def set_notify_enabled(on: bool) -> None:
    QSettings(_SETTINGS_ORG, _SETTINGS_APP).setValue(NOTIFY_KEY, bool(on))


def format_elapsed(seconds: float) -> str:
    """かかった時間を「45 秒」「3 分 12 秒」「1 時間 5 分」の形にする"""
    s = int(round(max(0.0, float(seconds))))
    if s < 60:
        return f"{s} 秒"
    if s < 3600:
        return f"{s // 60} 分 {s % 60} 秒"
    return f"{s // 3600} 時間 {(s % 3600) // 60} 分"


def _plain(text: str) -> str:
    """通知に出す 1 行の文（絵文字を外し、長すぎれば切る）"""
    from gui.modern_ui import strip_emoji  # ここで読む（gui.modern_ui は gui.system_tools を読むので、先頭では読めない）

    t = strip_emoji(" ".join(str(text or "").split()))
    return t if len(t) <= MAX_BODY_CHARS else t[: MAX_BODY_CHARS - 1] + "…"


def outcome_message(win) -> Tuple[str, str]:
    """
    いま画面に出ている「結果」の文と種類（success / info / warning / error）。
    完了やエラーの窓が出ていればその文、画面の中の通知が出ていればその文、どちらも無ければステータスの文。
    """
    box = QApplication.activeModalWidget()
    if isinstance(box, QMessageBox):
        lines = [ln.strip() for ln in box.text().splitlines() if ln.strip()]
        text = " ".join(lines[:2]) if lines else box.windowTitle()
        icon = box.icon()
        kind = "error" if icon == QMessageBox.Critical else "warning" if icon == QMessageBox.Warning else "success"
        return _plain(text), kind
    toast = getattr(win, "toast_label", None)
    try:
        if toast is not None and toast.isVisible() and toast.text().strip():
            return _plain(toast.text()), classify_message(toast.text())
    except RuntimeError:
        pass
    status = getattr(win, "status_label", None)
    try:
        if status is not None and status.text().strip():
            return _plain(status.text()), classify_message(status.text())
    except RuntimeError:
        pass
    return "処理が終わりました", "info"


def _app_icon() -> QIcon:
    """通知に添える小さな絵（このアプリは専用のアイコンを持たないので、点の並びを描く）"""
    icon = QApplication.windowIcon()
    if not icon.isNull():
        return icon
    pm = QPixmap(64, 64)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing, True)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor("#0f172a"))
    p.drawRoundedRect(2, 2, 60, 60, 12, 12)
    p.setBrush(QColor("#22bdd6"))
    for row in range(3):
        for col in range(3):
            if (row, col) == (0, 2):
                continue
            p.drawEllipse(14 + col * 14, 14 + row * 14, 8, 8)
    p.end()
    return QIcon(pm)


class OsNotifier(QObject):
    """タスクバーのボタンを光らせ、Windows の通知を出す。利用者がこのアプリへ戻ったら、通知を片付ける"""

    def __init__(self, win, tray_factory: Optional[Callable] = None):
        super().__init__(win)
        self.win = win
        self.tray = None
        self._tray_factory = tray_factory
        app = QGuiApplication.instance()
        if app is not None:
            app.applicationStateChanged.connect(self._on_app_state)

    def _make_tray(self):
        if self._tray_factory is not None:
            return self._tray_factory()
        if not QSystemTrayIcon.isSystemTrayAvailable() or not QSystemTrayIcon.supportsMessages():
            return None
        tray = QSystemTrayIcon(_app_icon(), self)
        tray.setToolTip(NOTIFY_TITLE)
        return tray

    def notify(self, title: str, body: str, kind: str) -> bool:
        """知らせる。Windows の通知まで出せたら True（タスクバーを光らせるだけなら False）"""
        QApplication.alert(self.win, 0)  # 窓へ戻るまで、タスクバーのボタンを光らせる
        if self.tray is None:
            self.tray = self._make_tray()
            if self.tray is None:
                return False
            self.tray.messageClicked.connect(self.bring_to_front)
            self.tray.activated.connect(lambda _reason=None: self.bring_to_front())
        icon = (
            QSystemTrayIcon.Critical if kind == "error"
            else QSystemTrayIcon.Warning if kind == "warning"
            else QSystemTrayIcon.Information
        )
        self.tray.show()  # 通知は、通知領域に絵が出ているあいだだけ出せる
        self.tray.showMessage(title, body, icon, 15000)
        return True

    def bring_to_front(self) -> None:
        win = self.win
        if win.isMinimized():
            win.showNormal()
        win.raise_()
        win.activateWindow()
        self.clear()

    def clear(self) -> None:
        """通知領域の絵を片付ける（出したままにしない）"""
        if self.tray is not None:
            self.tray.hide()

    def _on_app_state(self, state) -> None:
        if state == Qt.ApplicationActive:
            self.clear()


def _notify_os(win, title: str, body: str, kind: str) -> None:
    if os.environ.get(NO_OS_NOTIFY_ENV):
        return
    notifier = getattr(win, "_os_notifier", None)
    if notifier is None:
        notifier = OsNotifier(win)
        win._os_notifier = notifier
    notifier.notify(title, body, kind)


def _app_is_active() -> bool:
    return QGuiApplication.applicationState() == Qt.ApplicationActive


class _Hooks:
    """外の世界との接点。テストは、ここを差し替える"""
    notify = staticmethod(_notify_os)
    is_app_active = staticmethod(_app_is_active)
    clock = staticmethod(time.monotonic)


HOOKS = _Hooks()


class LongTaskWatch(QObject):
    """進捗の窓が出てから消えるまでを測り、長くかかった処理の終わりを知らせる"""

    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self._started: Optional[float] = None
        self._finished: float = 0.0
        self._busy_waits = 0
        self.last_notice: Optional[Tuple[str, str, float]] = None  # （文, 種類, かかった秒数）最後に知らせたもの
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(SETTLE_MS)
        self._timer.timeout.connect(self._settle)
        add_progress_listener(self)

    def _is_mine(self, dialog) -> bool:
        return dialog is not None and dialog.parent() is self.win

    def progress_shown(self, dialog) -> None:
        if not self._is_mine(dialog):
            return
        # 続きの窓がすぐに出たときは、ひと続きの処理として測り続ける（作業用データの作成 → 高精細表示の準備 など）
        self._timer.stop()
        if self._started is None:
            self._started = HOOKS.clock()

    def progress_hidden(self, dialog) -> None:
        if not self._is_mine(dialog) or self._started is None:
            return
        self._finished = HOOKS.clock()
        self._busy_waits = 0
        self._timer.start()

    def _still_working(self) -> bool:
        win = self.win
        dlg = getattr(win, "progress_dialog", None)
        if dlg is not None and dlg.isVisible():
            return True
        busy = getattr(win, "_is_background_busy", None)
        return bool(busy()) if callable(busy) else False

    def _settle(self) -> None:
        if self._started is None:
            return
        try:
            if self._still_working() and self._busy_waits < MAX_BUSY_WAITS:
                self._busy_waits += 1
                dlg = getattr(self.win, "progress_dialog", None)
                if not (dlg is not None and dlg.isVisible()):
                    self._timer.start()  # 窓は消えたが、裏の処理の後始末がまだ → もう少し待つ
                return  # 続きの窓が出ている → その窓が消えたときに、また呼ばれる
            elapsed = self._finished - self._started
            self._started = None
            if elapsed < NOTIFY_MIN_SECONDS or not get_notify_enabled() or HOOKS.is_app_active():
                return
            text, kind = outcome_message(self.win)
            body = f"{text}（{format_elapsed(elapsed)}）"
            self.last_notice = (body, kind, elapsed)
            log_info(f"完了の知らせ: {body}")
            HOOKS.notify(self.win, NOTIFY_TITLE, body, kind)
        except RuntimeError:
            self._started = None  # ウィンドウが先に閉じられた


def install_long_task_notify(win) -> LongTaskWatch:
    """MainWindow に 1 回取り付ける（gui/system_tools.py: install_system_tools から呼ぶ）"""
    watch = LongTaskWatch(win)
    win._long_task_watch = watch
    return watch


def make_notify_checkbox(win, parent) -> QCheckBox:
    """設定 → システムに置くチェック（変えたその場で保存する）"""
    chk = QCheckBox("長い処理が終わったら知らせる", parent)
    chk.setObjectName("chk_notify_long_task")
    chk.setChecked(get_notify_enabled())
    chk.setToolTip(
        f"読み込み・保存・解析などが {int(NOTIFY_MIN_SECONDS)} 秒より長くかかり、終わったときにほかのアプリを使っていたら、\n"
        "タスクバーのボタンを光らせ、Windows の通知を出します。\n"
        "このアプリを見ているときは出しません。エラーで止まったときも同じように知らせます。"
    )
    chk.toggled.connect(lambda on: set_notify_enabled(bool(on)))
    win.chk_notify_long_task = chk
    return chk
