# -*- coding: utf-8 -*-
"""
モード別の操作ガイド: 選んだツールに合わせて「次に何をすればよいか」を 3D 画面の右上に 1〜3 行で出す

ブラシ・矩形・円形・多角形・パン・計測・斜め壁の指定のときと、点を選択しているときに表示する。
回転モードで何も選んでいないとき（ふだんの状態）は出さない。
ビューアの状態は短い間隔のタイマーで見る（キー操作での切替も拾える。マウス処理には手を入れない）。
設定「説明書 → モード別の操作ガイドを表示する」で消せる（QSettings: mode_hints_enabled）。
"""

from typing import List, Optional, Tuple

from PySide6.QtCore import QEvent, QSettings, Qt, QTimer
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout

SETTINGS_KEY = "mode_hints_enabled"

SELECT_TOOLS = {
    "brush": ("ブラシ選択", "なぞった所の点を選ぶ → Del でノイズへ", "Shift 追加・Ctrl 除外・[ ] で太さ・Esc 解除"),
    "rect": ("矩形選択", "ドラッグで囲んだ点を選ぶ → Del でノイズへ", "Shift 追加・Ctrl 除外・I 反転・Esc 解除"),
    "circle": ("円形選択", "中心からドラッグした円の中を選ぶ → Del でノイズへ", "Shift 追加・Ctrl 除外・I 反転・Esc 解除"),
    "polygon": ("多角形選択", "クリックで頂点を置いて囲む → Enter で確定", "Shift 追加・Ctrl 除外・Esc 取消"),
}


def hints_enabled() -> bool:
    return bool(QSettings("E57PointCloudTool", "Settings").value(SETTINGS_KEY, True, type=bool))


def set_hints_enabled(enabled: bool) -> None:
    QSettings("E57PointCloudTool", "Settings").setValue(SETTINGS_KEY, bool(enabled))


def current_hint(win) -> Optional[Tuple[str, List[str]]]:
    """今のビューアの状態に合うガイド (題名, 本文の行)。出すものが無ければ None"""
    v = getattr(win, "viewer", None)
    if v is None or v.points_raw is None or len(v.points_raw) == 0:
        return None
    sel = v.selected_indices
    n_sel = len(sel) if sel is not None else 0
    if getattr(v, "wall_align_mode", False):
        return ("斜め壁の指定", ["壁の上の 2 点を順にクリックする", "指定後は 横・縦・正面・真横 の視点と、壁に沿った断面が使える", "U でやめる"])
    if getattr(v, "measure_mode", False):
        return ("計測", ["点をクリック → 座標を表示。2 点目で距離・高低差", "Ctrl+C でコピー・右クリックでクリア・M で終了"])
    title = None
    lines: List[str] = []
    if getattr(v, "selection_mode", False):
        shape = getattr(v, "selection_shape", "")
        key = {getattr(v, "SELECT_RECT", "rect"): "rect", getattr(v, "SELECT_CIRCLE", "circle"): "circle",
               getattr(v, "SELECT_POLYGON", "polygon"): "polygon", getattr(v, "SELECT_BRUSH", "brush"): "brush"}.get(shape, str(shape))
        info = SELECT_TOOLS.get(key)
        if info is not None:
            title = info[0]
            lines = [info[1], info[2]]
            if getattr(v, "instant_delete_mode", False):
                lines.append("即削除 ON: 囲んだ瞬間にノイズへ移る（Ctrl+Z で戻せる）")
            lines.append("Alt + 左ドラッグで、このまま視点を回せる")
    elif getattr(v, "pan_mode", False):
        title = "パン"
        lines = ["左ドラッグで画面を平行移動する", "V で回転に戻る（Space 長押しでも一時的にパン）"]
    if n_sel > 0:
        if title is None:
            title = f"選択中 {n_sel:,} 点"
            lines = ["Del ノイズへ / G 連結塊を選択 / 右クリックで操作一覧", "Esc で解除・Ctrl+Z で元に戻す"]
        else:
            lines = [f"選択中 {n_sel:,} 点: Del ノイズへ / G 連結塊 / 右クリックで操作一覧"] + lines
    if title is None:
        return None
    return (title, lines)


class ModeHintOverlay(QFrame):
    def __init__(self, win):
        super().__init__(win.viewer)
        self._win = win
        self.setObjectName("mode_hint")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)  # 3D 画面の操作を邪魔しない
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 8, 12, 8)
        lay.setSpacing(2)
        self.title_label = QLabel("", self)
        self.title_label.setObjectName("mode_hint_title")
        lay.addWidget(self.title_label)
        self.body_label = QLabel("", self)
        self.body_label.setObjectName("mode_hint_body")
        lay.addWidget(self.body_label)
        self._enabled = hints_enabled()
        self._shown_key = None
        self.hide()
        win.viewer.installEventFilter(self)
        self._timer = QTimer(self)
        self._timer.setInterval(250)
        self._timer.timeout.connect(self.refresh)
        self._timer.start()

    def is_enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = bool(enabled)
        set_hints_enabled(self._enabled)
        self._shown_key = None
        self.refresh()

    def refresh(self) -> None:
        hint = current_hint(self._win) if self._enabled else None
        if hint is None:
            if self.isVisible():
                self.hide()
            self._shown_key = None
            return
        key = (hint[0], tuple(hint[1]))
        if key != self._shown_key:
            self._shown_key = key
            self.title_label.setText(hint[0])
            self.body_label.setText("\n".join(hint[1]))
            self.adjustSize()
            self._reposition()
        if not self.isVisible():
            self.show()
            self._reposition()

    def _reposition(self) -> None:
        parent = self.parentWidget()
        if parent is None:
            return
        self.adjustSize()
        self.move(max(8, parent.width() - self.width() - 12), 12)
        self.raise_()

    def eventFilter(self, obj, event) -> bool:  # type: ignore[override]
        if event.type() == QEvent.Resize and self.isVisible():
            self._reposition()
        return False
