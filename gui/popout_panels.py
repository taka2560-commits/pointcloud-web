# -*- coding: utf-8 -*-
"""
パネルのポップアップ化: 左パネルの各グループ（①自動・②手動・③確認・断面・表示・設定）を小窓として浮かせる

- 各グループの見出しの「別窓」ボタン、または左パネルを格納したときに 3D 画面の左端へ出る「パネルドック」から開く。
- 小窓は既存のグループ（ウィジェットそのもの）を左パネルから移すだけなので、操作も状態もそのまま（二重に作らない）。
- 小窓を閉じる（または「戻す」）と、左パネルの元の位置へ戻る。
- 小窓に出ている間は、左パネルのタブ切替でそのグループが隠れないようにする。
左パネルを格納したまま、必要な小窓だけ出して広い 3D 画面で作業できる。
"""

from typing import Dict, List, Optional, Tuple

from PySide6.QtCore import QEvent, QObject, QPoint, Qt, Signal
from PySide6.QtWidgets import QApplication, QDialog, QFrame, QPushButton, QScrollArea, QVBoxLayout, QWidget

# (キー = 左パネルのタブ名, MainWindow の属性名, 短い名前, 小窓の題名)
PANELS: List[Tuple[str, str, str, str]] = [
    ("filter", "auto_group", "①自動", "① 自動ノイズ除去"),
    ("edit", "tools_group", "②手動", "② 手動修正"),
    ("layer", "layer_group", "③確認", "③ 確認・レイヤー"),
    ("section", "section_group", "断面", "断面"),
    ("view", "view_group", "表示", "表示・カメラ"),
    ("settings", "settings_group", "設定", "設定・説明書"),
]


class PopoutWindow(QDialog):
    """グループ 1 つを入れる小窓（ツールウィンドウ。メインウィンドウの上に浮く）"""

    closing = Signal(str)

    def __init__(self, win, key: str, title: str):
        super().__init__(win, Qt.Tool | Qt.WindowTitleHint | Qt.WindowCloseButtonHint)
        self.key = key
        self.setObjectName("popout_window")
        self.setWindowTitle(title)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.scroll = QScrollArea(self)
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.host = QWidget()
        self.host.setObjectName("popout_host")
        self.host_layout = QVBoxLayout(self.host)
        self.host_layout.setContentsMargins(6, 6, 6, 6)
        self.host_layout.setSpacing(0)
        self.scroll.setWidget(self.host)
        lay.addWidget(self.scroll)

    def closeEvent(self, event):  # type: ignore[override]
        self.closing.emit(self.key)
        super().closeEvent(event)


class PanelDock(QFrame):
    """左パネルを格納している間、3D 画面の左端に出すボタンの列（小窓の出し入れ）"""

    def __init__(self, win, popouts: "PanelPopouts", open_launcher=None):
        super().__init__(win.viewer)
        self._win = win
        self._popouts = popouts
        self.setObjectName("panel_dock")
        self.setAttribute(Qt.WA_StyledBackground, True)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(2)
        self.buttons: Dict[str, QPushButton] = {}
        if open_launcher is not None:
            b = QPushButton("検索", self)
            b.setProperty("role", "dock")
            b.setToolTip("機能を検索して実行する (Ctrl+K)")
            b.clicked.connect(lambda _=False: open_launcher())  # clicked の引数（bool）を渡さない
            lay.addWidget(b)
            self.buttons["launcher"] = b
        for key, _attr, short, title in PANELS:
            if popouts.group(key) is None:
                continue
            b = QPushButton(short, self)
            b.setProperty("role", "dock")
            b.setCheckable(True)
            b.setToolTip(f"{title} を小窓で開く / 閉じる")
            b.clicked.connect(lambda _=False, k=key: self._on_clicked(k))
            lay.addWidget(b)
            self.buttons[key] = b
        self.adjustSize()
        win.viewer.installEventFilter(self)
        self.hide()

    def _on_clicked(self, key: str) -> None:
        anchor = self.mapToGlobal(QPoint(self.width() + 8, 0))
        self._popouts.toggle(key, anchor)

    def sync(self) -> None:
        """左パネルが格納されているときだけ表示し、開いている小窓のボタンを押した状態にする"""
        scroll = getattr(self._win, "sidebar_scroll", None)
        # isHidden = 明示的に隠した状態（ウィンドウ表示前の「まだ見えていない」と区別する）
        collapsed = scroll is not None and scroll.isHidden()
        for key, b in self.buttons.items():
            if key != "launcher":
                b.setChecked(self._popouts.is_popped(key))
        self.setVisible(bool(collapsed))
        if collapsed:
            self._reposition()

    def _reposition(self) -> None:
        parent = self.parentWidget()
        if parent is None:
            return
        self.adjustSize()
        self.move(8, max(8, (parent.height() - self.height()) // 2))
        self.raise_()

    def eventFilter(self, obj, event) -> bool:  # type: ignore[override]
        if event.type() == QEvent.Resize and self.isVisible():
            self._reposition()
        return False


class PanelPopouts(QObject):
    changed = Signal()

    def __init__(self, win):
        super().__init__(win)
        self._win = win
        self._windows: Dict[str, PopoutWindow] = {}
        self._header_buttons: Dict[str, QPushButton] = {}
        self.dock: Optional[PanelDock] = None
        self._install_header_buttons()
        self._wrap_tab_switch()
        self._wrap_sidebar_toggle()

    # --- 参照 ---
    def group(self, key: str):
        for k, attr, _short, _title in PANELS:
            if k == key:
                return getattr(self._win, attr, None)
        return None

    def title(self, key: str) -> str:
        for k, _attr, _short, title in PANELS:
            if k == key:
                return title
        return key

    def is_popped(self, key: str) -> bool:
        return key in self._windows

    def popped_keys(self) -> List[str]:
        return list(self._windows.keys())

    def window(self, key: str) -> Optional[PopoutWindow]:
        return self._windows.get(key)

    # --- 準備 ---
    def _install_header_buttons(self) -> None:
        for key, _attr, _short, title in PANELS:
            grp = self.group(key)
            header = getattr(grp, "_header_widget", None) if grp is not None else None
            lay = header.layout() if header is not None else None
            if lay is None:
                continue
            btn = QPushButton("別窓", header)
            btn.setObjectName("collapsible_popout")
            btn.setCursor(Qt.PointingHandCursor)
            btn.setToolTip(f"{title} を小窓として浮かせる（閉じると左パネルへ戻る）")
            btn.clicked.connect(lambda _=False, k=key: self.toggle(k))
            lay.insertWidget(max(0, lay.count() - 1), btn)
            self._header_buttons[key] = btn

    def _wrap_tab_switch(self) -> None:
        win = self._win
        orig = win._set_sidebar_workflow_tab

        def set_tab(tab_key: str, *args, **kwargs):
            result = orig(tab_key, *args, **kwargs)
            alias = {"select": "edit", "auto": "filter"}
            norm = alias.get(tab_key, tab_key)
            for key in self.popped_keys():
                grp = self.group(key)
                if grp is not None:
                    grp.setVisible(True)  # 小窓に出ているグループはタブ切替で隠さない
            w = self._windows.get(norm)
            if w is not None:
                w.raise_()
                w.activateWindow()
            return result

        win._set_sidebar_workflow_tab = set_tab  # type: ignore[assignment]

    def _wrap_sidebar_toggle(self) -> None:
        win = self._win
        orig = win._toggle_sidebar

        def toggle_sidebar(*args, **kwargs):
            result = orig(*args, **kwargs)
            if self.dock is not None:
                self.dock.sync()
            return result

        win._toggle_sidebar = toggle_sidebar  # type: ignore[assignment]
        btn = getattr(win, "btn_toggle_sidebar", None)
        if btn is not None:
            try:
                btn.clicked.disconnect()
            except Exception:
                pass
            btn.clicked.connect(lambda: win._toggle_sidebar())

    def install_dock(self, open_launcher=None) -> PanelDock:
        self.dock = PanelDock(self._win, self, open_launcher)
        self.dock.sync()
        return self.dock

    # --- 出し入れ ---
    def toggle(self, key: str, anchor_global: Optional[QPoint] = None) -> None:
        if self.is_popped(key):
            self.restore(key)
        else:
            self.pop_out(key, anchor_global)

    def pop_out(self, key: str, anchor_global: Optional[QPoint] = None) -> Optional[PopoutWindow]:
        if self.is_popped(key):
            w = self._windows[key]
            w.raise_()
            w.activateWindow()
            return w
        grp = self.group(key)
        win = self._win
        if grp is None:
            return None
        side_lay = win.sidebar_widget.layout()
        if side_lay is not None:
            side_lay.removeWidget(grp)
        w = PopoutWindow(win, key, self.title(key))
        w.host_layout.addWidget(grp)
        w.host_layout.addStretch(1)
        if hasattr(grp, "set_expanded"):
            try:
                grp.set_expanded(True)
            except Exception:
                pass
        grp.setVisible(True)
        self._windows[key] = w
        w.closing.connect(self._on_window_closing)

        hint = grp.sizeHint()
        screen = win.screen() or QApplication.primaryScreen()
        avail = screen.availableGeometry() if screen is not None else None
        width = max(380, min(hint.width() + 36, 560))
        height = hint.height() + 24
        if avail is not None:
            height = min(height, int(avail.height() * 0.8))
        w.resize(width, max(160, height))
        if anchor_global is None:
            anchor_global = win.viewer.mapToGlobal(QPoint(24 + 28 * (len(self._windows) - 1), 24 + 28 * (len(self._windows) - 1)))
        pos = QPoint(anchor_global)
        if avail is not None:
            pos.setX(max(avail.left(), min(pos.x(), avail.right() - width)))
            pos.setY(max(avail.top(), min(pos.y(), avail.bottom() - w.height())))
        w.move(pos)
        w.show()
        w.raise_()
        self._after_change(key)
        return w

    def _on_window_closing(self, key: str) -> None:
        self.restore(key, close_window=False)

    def restore(self, key: str, close_window: bool = True) -> None:
        w = self._windows.pop(key, None)
        if w is None:
            return
        grp = self.group(key)
        win = self._win
        if grp is not None:
            w.host_layout.removeWidget(grp)
            side_lay = win.sidebar_widget.layout()
            if side_lay is not None:
                side_lay.insertWidget(self._sidebar_index(key), grp)
            grp.setParent(win.sidebar_widget)
        if close_window:
            try:
                w.closing.disconnect(self._on_window_closing)
            except Exception:
                pass
            w.close()
        w.deleteLater()
        # 今のタブに合わせて表示・非表示を戻す
        current = "all"
        for tab_key, btn in getattr(win, "sidebar_tab_btns", {}).items():
            if btn.isChecked():
                current = tab_key
                break
        try:
            win._set_sidebar_workflow_tab(current)
        except Exception:
            if grp is not None:
                grp.setVisible(True)
        self._after_change(key)

    def restore_all(self) -> None:
        for key in self.popped_keys():
            self.restore(key)

    def _sidebar_index(self, key: str) -> int:
        """左パネルへ戻すときの挿入位置（決まった並び順で、次に居るグループの手前）"""
        win = self._win
        lay = win.sidebar_widget.layout()
        keys = [k for k, _a, _s, _t in PANELS]
        for nxt in keys[keys.index(key) + 1 :]:
            g = self.group(nxt)
            if g is not None:
                idx = lay.indexOf(g)
                if idx >= 0:
                    return idx
        for attr in ("save_group", "log_group"):
            g = getattr(win, attr, None)
            if g is not None:
                idx = lay.indexOf(g)
                if idx >= 0:
                    return idx
        return lay.count()

    def _after_change(self, key: str) -> None:
        btn = self._header_buttons.get(key)
        if btn is not None:
            popped = self.is_popped(key)
            btn.setText("戻す" if popped else "別窓")
            btn.setToolTip("左パネルへ戻す" if popped else f"{self.title(key)} を小窓として浮かせる（閉じると左パネルへ戻る）")
        if self.dock is not None:
            self.dock.sync()
        self.changed.emit()
