# -*- coding: utf-8 -*-
"""
新UIのポップアップ（コマンドランチャー・右クリックのクイックパレット）の共通の土台

3D ビューアの子ウィジェットとして重ねて表示する（別ウィンドウにしないので、フォーカスや重なり順で迷子にならない）。
開いている間だけアプリ全体のイベントを見て、外側のクリックと Esc で閉じる。

Esc / Enter / 矢印キーにはウィンドウ全体のショートカット（選択解除・多角形確定など）が割り当てられているので、
開いている間は ShortcutOverride を受け取って、キー入力をポップアップ側へ回す。
見た目は gui/design_tokens.py の QSS（objectName で指定）で描く。インライン QSS は書かない。
"""

from typing import Tuple

from PySide6.QtCore import QEvent, QPoint, Qt, Signal
from PySide6.QtWidgets import QApplication, QFrame, QWidget


class PopupFrame(QFrame):
    closed = Signal()

    CAPTURE_KEYS: Tuple[int, ...] = (Qt.Key_Escape,)

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setFocusPolicy(Qt.NoFocus)
        self._filter_installed = False
        self.hide()

    # --- 開閉 ---
    def open_popup(self) -> None:
        self.show()
        self.raise_()
        if not self._filter_installed:
            app = QApplication.instance()
            if app is not None:
                app.installEventFilter(self)
                self._filter_installed = True
        try:
            from gui.motion import fade_slide_in, motion_enabled

            if motion_enabled():
                fade_slide_in(self, dy=6, duration_ms=140)
        except Exception:
            pass

    def close_popup(self) -> None:
        was_visible = self.isVisible()
        if self._filter_installed:
            app = QApplication.instance()
            if app is not None:
                app.removeEventFilter(self)
            self._filter_installed = False
        self.hide()
        if was_visible:
            self.closed.emit()

    def is_open(self) -> bool:
        return self.isVisible()

    # --- サブクラスが上書きする ---
    def popup_key_press(self, event) -> bool:
        """CAPTURE_KEYS のキーが押されたとき。処理したら True"""
        if event.key() == Qt.Key_Escape:
            self.close_popup()
            return True
        return False

    # --- 位置合わせ ---
    def move_clamped(self, pos: QPoint, margin: int = 8) -> None:
        """親の中に収まるように位置を決める"""
        parent = self.parentWidget()
        if parent is None:
            self.move(pos)
            return
        x = max(margin, min(pos.x(), parent.width() - self.width() - margin))
        y = max(margin, min(pos.y(), parent.height() - self.height() - margin))
        self.move(x, y)

    # --- アプリ全体のイベント（開いている間だけ） ---
    def eventFilter(self, obj, event) -> bool:  # type: ignore[override]
        if not self.isVisible():
            return False
        t = event.type()
        if t == QEvent.MouseButtonPress:
            try:
                gp = event.globalPosition().toPoint()
            except AttributeError:
                return False
            if not self.rect().contains(self.mapFromGlobal(gp)):
                self.close_popup()  # 外側をクリックしたら閉じる（クリック自体はそのまま通す）
            return False
        if t == QEvent.ShortcutOverride:
            if event.key() in self.CAPTURE_KEYS:
                event.accept()  # ウィンドウのショートカットに取られないようにする
                return True
            return False
        if t == QEvent.KeyPress:
            if event.key() in self.CAPTURE_KEYS:
                return bool(self.popup_key_press(event))
        return False
