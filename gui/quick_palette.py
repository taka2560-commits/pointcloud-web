# -*- coding: utf-8 -*-
"""
右クリックのクイックパレット: カーソルの横に「その場面で使う操作」だけをタイルで出す

- 点を選択しているとき: ノイズへ・連結塊・範囲内の精密除去・レイヤー移動・復元・反転・解除
- 選択していないとき  : ツール（回転・パン・ブラシ・矩形・円形・多角形・計測・2点壁）と視点（上面・正面・側面・等角・フィット・投影）
- どちらも下に「機能を検索 (Ctrl+K)」と「その他のメニュー」（従来の右クリックメニュー）を置く。
タイルの中身はコマンド表（gui/commands.py）から作るので、使えない操作は自動で押せなくなる。
"""

from typing import Callable, Dict, List, Optional, Sequence, Tuple

from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QGridLayout, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

from gui.commands import Command
from gui.popup_base import PopupFrame

# (コマンド ID, タイルの短い名前)
SELECTION_TILES: Sequence[Tuple[str, str]] = (
    ("edit.delete", "ノイズへ"),
    ("edit.wand", "連結塊を選択"),
    ("edit.local_clean", "範囲内を精密除去"),
    ("layer.move", "レイヤーへ移動"),
    ("layer.restore_selected", "ノイズを復元"),
    ("edit.invert", "反転"),
    ("edit.clear", "解除"),
)
TOOL_TILES: Sequence[Tuple[str, str]] = (
    ("mode.orbit", "回転"),
    ("mode.pan", "パン"),
    ("mode.brush", "ブラシ"),
    ("mode.rect", "矩形"),
    ("mode.circle", "円形"),
    ("mode.polygon", "多角形"),
    ("mode.measure", "計測"),
    ("section.wall_pick", "2点壁"),
)
VIEW_TILES: Sequence[Tuple[str, str]] = (
    ("view.top", "上面"),
    ("view.front", "正面"),
    ("view.side", "側面"),
    ("view.iso", "等角"),
    ("view.fit", "フィット"),
    ("view.projection", "投影切替"),
)


class QuickPalette(PopupFrame):
    def __init__(
        self,
        win,
        commands_provider: Callable[[], Dict[str, Command]],
        open_launcher: Optional[Callable[[], None]] = None,
        open_legacy_menu: Optional[Callable[[QPoint], None]] = None,
    ):
        super().__init__(win.viewer)
        self._win = win
        self._provider = commands_provider
        self._open_launcher = open_launcher
        self._open_legacy_menu = open_legacy_menu
        self._last_global = QPoint()
        self.setObjectName("quick_palette")
        self._root = QVBoxLayout(self)
        self._root.setContentsMargins(10, 10, 10, 10)
        self._root.setSpacing(8)
        self.tiles: List[QPushButton] = []

    # --- 組み立て ---
    def _clear(self) -> None:
        self.tiles = []
        def _discard(widget) -> None:
            if widget is not None:
                widget.hide()
                widget.deleteLater()

        while self._root.count():
            item = self._root.takeAt(0)
            _discard(item.widget())
            lay = item.layout()
            if lay is not None:
                while lay.count():
                    _discard(lay.takeAt(0).widget())

    def _add_title(self, text: str) -> None:
        lbl = QLabel(text, self)
        lbl.setObjectName("palette_title")
        self._root.addWidget(lbl)

    def _add_tiles(self, specs: Sequence[Tuple[str, str]], columns: int, by_id: Dict[str, Command]) -> None:
        grid = QGridLayout()
        grid.setSpacing(6)
        n = 0
        for cmd_id, label in specs:
            cmd = by_id.get(cmd_id)
            if cmd is None:
                continue
            text = f"{label}\n{cmd.shortcut}" if cmd.shortcut else label
            btn = QPushButton(text, self)
            btn.setProperty("role", "tile")
            if cmd_id == "edit.delete":
                btn.setProperty("danger", True)
            btn.setProperty("command_id", cmd_id)
            reason = cmd.unavailable_reason(self._win)
            btn.setEnabled(not reason)
            btn.setToolTip(f"今は使えません: {reason}" if reason else (cmd.desc or cmd.title))
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(lambda _=False, c=cmd: self._run(c))
            grid.addWidget(btn, n // columns, n % columns)
            self.tiles.append(btn)
            n += 1
        self._root.addLayout(grid)

    def _add_footer(self) -> None:
        row = QHBoxLayout()
        row.setSpacing(6)
        if self._open_launcher is not None:
            b = QPushButton("機能を検索 (Ctrl+K)", self)
            b.setProperty("role", "ghost")
            b.clicked.connect(self._on_launcher)
            row.addWidget(b)
        if self._open_legacy_menu is not None:
            b2 = QPushButton("その他のメニュー", self)
            b2.setProperty("role", "ghost")
            b2.clicked.connect(self._on_legacy)
            row.addWidget(b2)
        row.addStretch(1)
        self._root.addLayout(row)

    def rebuild(self) -> None:
        self._clear()
        by_id = self._provider()
        sel = getattr(self._win.viewer, "selected_indices", None)
        n_sel = len(sel) if sel is not None else 0
        if n_sel > 0:
            self._add_title(f"選択中 {n_sel:,} 点")
            self._add_tiles(SELECTION_TILES, 4, by_id)
        else:
            self._add_title("ツール")
            self._add_tiles(TOOL_TILES, 4, by_id)
            self._add_title("視点")
            self._add_tiles(VIEW_TILES, 3, by_id)
        self._add_footer()
        self.adjustSize()

    def tile_command_ids(self) -> List[str]:
        return [str(b.property("command_id")) for b in self.tiles]

    def tile(self, cmd_id: str) -> Optional[QPushButton]:
        for b in self.tiles:
            if b.property("command_id") == cmd_id:
                return b
        return None

    # --- 開く ---
    def show_at(self, global_pos: QPoint) -> None:
        self._last_global = QPoint(global_pos)
        self.rebuild()
        self.resize(self.sizeHint())
        parent = self.parentWidget()
        local = parent.mapFromGlobal(global_pos) + QPoint(10, 10)
        self.move_clamped(local)
        self.open_popup()

    # --- 実行 ---
    def _run(self, cmd: Command) -> None:
        self.close_popup()
        cmd.execute(self._win)

    def _on_launcher(self) -> None:
        self.close_popup()
        if self._open_launcher is not None:
            self._open_launcher()

    def _on_legacy(self) -> None:
        pos = QPoint(self._last_global)
        self.close_popup()
        if self._open_legacy_menu is not None:
            self._open_legacy_menu(pos)
