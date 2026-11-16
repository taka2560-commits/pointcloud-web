# -*- coding: utf-8 -*-
"""
ツールの名前とアイコン（このアプリが「何という道具か」を、1 か所で決める）

名前を出す所は、ここの定数を読む。名前を変えるときは TOOL_NAME の 1 行だけを変える:
    ウィンドウの題名（gui/system_tools.py の APP_TITLE）・起動した所のカード（gui/modern_ui.py の EmptyStateCard）・
    バージョン情報（gui/system_tools.py の APP_NAME）・Windows の通知の題名（gui/task_notify.py）
変えない物（名前を変えても、今までの設定・ショートカット・手順がそのまま使えるように）:
    EXE とフォルダの名前（E57_PointCloud_Tool）・保存設定の置き場所（QSettings の "E57PointCloudTool"）・ログの中の文

アイコンは assets/app/ の絵（tools/make_app_icon.py が描いて書き出した物）。
    ウィンドウの左上・タスクバー・Alt+Tab・通知・起動した所のカードに出る。配布版の EXE のアイコンは、
    E57_PointCloud_Tool.spec の icon= が同じフォルダの app_icon.ico を埋め込む。
絵のファイルが無くても、アプリは今までどおり動く（アイコン無しになるだけ。起動を止めない）。
"""

import os
import sys
from typing import Optional

from PySide6.QtCore import QSize
from PySide6.QtGui import QGuiApplication, QIcon, QPixmap

from core.logger import log_warning
from gui.design_tokens import assets_dir

TOOL_NAME = "ポイントクリーナー"                                  # ツールの名前
TOOL_KIND = "E57 点群ノイズ処理ツール"                          # 何をする道具か（題名で、名前の後ろに添える）
TOOL_KIND_LONG = "E57 点群ノイズ処理・断面スライサー ツール"     # バージョン情報に出す、少し詳しい言い方
TOOL_TAGLINE = "E57 点群のノイズ処理・断面スライサー"            # 起動した所のカードで、名前の下に添える

ICON_SIZES = (16, 20, 24, 32, 40, 48, 64, 128, 256)
# タスクバーが「どのアプリの窓か」を見分ける名前。ソースから起動したとき（python.exe）に、Python の絵でなくこのアプリの絵を出すために要る
TASKBAR_APP_ID = "E57PointCloudTool.Desktop"

_icon: Optional[QIcon] = None


def app_icon_dir() -> str:
    return os.path.join(assets_dir(), "app")


def app_icon_path(size: int) -> str:
    return os.path.join(app_icon_dir(), f"app_icon_{int(size)}.png")


def app_icon() -> QIcon:
    """このアプリのアイコン。絵のファイルが 1 枚も無ければ、空の QIcon（呼ぶ側は isNull() を見る）"""
    global _icon
    if _icon is None:
        icon = QIcon()
        for s in ICON_SIZES:
            p = app_icon_path(s)
            if os.path.isfile(p):
                icon.addFile(p, QSize(s, s))
        if icon.isNull():
            log_warning(f"アプリのアイコンが見つかりません（アイコン無しで続けます）: {app_icon_dir()}")
        _icon = icon
    return _icon


def app_icon_pixmap(size: int) -> QPixmap:
    """画面に置くための絵（画面の拡大率に合わせた細かさで返す）。絵が無ければ空の QPixmap"""
    icon = app_icon()
    if icon.isNull():
        return QPixmap()
    screen = QGuiApplication.primaryScreen()
    try:
        ratio = max(1.0, float(screen.devicePixelRatio())) if screen is not None else 1.0
    except Exception:
        ratio = 1.0
    return icon.pixmap(QSize(int(size), int(size)), ratio)


def _set_taskbar_app_id() -> bool:
    """
    ソースから起動したときに、タスクバーのボタンをこのアプリの絵にする（何もしないと、python.exe の絵になる）。
    このプロセスの中だけの指定で、Windows の設定やレジストリは書き換えない。
    配布版は EXE に絵が埋め込んであるので、何もしない（タスクバーにピン留めしたボタンの扱いを変えないため）。
    """
    if sys.platform != "win32" or getattr(sys, "frozen", False):
        return False
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(TASKBAR_APP_ID)
        return True
    except Exception as e:
        log_warning(f"タスクバーの絵の指定ができませんでした（アプリはそのまま使えます）: {e!r}")
        return False


def apply_app_identity(app) -> bool:
    """
    アプリ全体のアイコンを決める（どのウィンドウ・確認の窓・通知にも同じ絵が出る）。起動のときに 1 回呼ぶ。
    アイコンを付けられたら True。何が起きても例外は出さない（アイコンのせいで起動を止めない）。
    """
    try:
        _set_taskbar_app_id()
        icon = app_icon()
        if icon.isNull():
            return False
        app.setWindowIcon(icon)
        return True
    except Exception as e:
        log_warning(f"アプリのアイコンを付けられませんでした（アプリはそのまま使えます）: {e!r}")
        return False
