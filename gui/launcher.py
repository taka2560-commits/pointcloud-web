# -*- coding: utf-8 -*-
"""
コマンドランチャー（Ctrl+K）: 機能名を打って Enter で実行する検索窓

- 何も打たなければ「最近使った」と、作業順（開く → ①自動 → ②手動 → ③確認 → 断面 → 表示 → 設定）の全機能を並べる。
- 打つと絞り込む。ひらがな・カタカナ・英字のどれでも探せる（gui/commands.py の keywords）。
- 今は使えない機能（点群が無い・選択が無いなど）は薄く表示し、理由を添える。Enter を押すと理由を通知する。
- ↑↓ で選ぶ、Enter で実行、Esc か外側クリックで閉じる。
"""

from typing import Callable, List, Optional

from PySide6.QtCore import QRect, QSettings, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QPainter
from PySide6.QtWidgets import QLabel, QLineEdit, QListWidget, QListWidgetItem, QStyle, QStyledItemDelegate, QVBoxLayout

from gui.commands import GROUP_ORDER, Command, match_score
from gui.design_tokens import get_tokens
from gui.popup_base import PopupFrame

ROLE_KIND = Qt.UserRole + 1      # "header" / "command"
ROLE_ID = Qt.UserRole + 2
ROLE_DESC = Qt.UserRole + 3
ROLE_SHORTCUT = Qt.UserRole + 4
ROLE_DISABLED = Qt.UserRole + 5
RECENT_KEY = "launcher_recent"
RECENT_MAX = 6
ROW_H = 44
HEADER_H = 26


def load_recent_ids() -> List[str]:
    val = QSettings("E57PointCloudTool", "Settings").value(RECENT_KEY, [])
    if isinstance(val, str):
        val = [val] if val else []
    return [str(v) for v in (val or [])][:RECENT_MAX]


def push_recent_id(cmd_id: str) -> None:
    ids = [i for i in load_recent_ids() if i != cmd_id]
    ids.insert(0, cmd_id)
    QSettings("E57PointCloudTool", "Settings").setValue(RECENT_KEY, ids[:RECENT_MAX])


class _RowDelegate(QStyledItemDelegate):
    """1 行 = 題名（太字）＋説明（小さく）＋右端にショートカット。色はテーマのトークンから"""

    def __init__(self, theme_getter: Callable[[], str], parent=None):
        super().__init__(parent)
        self._theme = theme_getter

    def sizeHint(self, option, index) -> QSize:  # type: ignore[override]
        return QSize(0, HEADER_H if index.data(ROLE_KIND) == "header" else ROW_H)

    def paint(self, painter: QPainter, option, index) -> None:  # type: ignore[override]
        t = get_tokens(self._theme())
        r: QRect = option.rect
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        if index.data(ROLE_KIND) == "header":
            f = QFont(painter.font())
            f.setPixelSize(t.size_xs)
            f.setBold(True)
            painter.setFont(f)
            painter.setPen(QColor(t.text_secondary))
            painter.drawText(r.adjusted(14, 0, -14, 0), Qt.AlignVCenter | Qt.AlignLeft, str(index.data(Qt.DisplayRole)))
            painter.restore()
            return

        disabled = bool(index.data(ROLE_DISABLED))
        if option.state & QStyle.State_Selected:
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(t.hover_bg))
            painter.drawRoundedRect(r.adjusted(6, 2, -6, -2), t.radius_md, t.radius_md)
            painter.setBrush(QColor(t.accent_base))
            painter.drawRoundedRect(QRect(r.left() + 6, r.top() + 8, 3, r.height() - 16), 1, 1)

        shortcut = str(index.data(ROLE_SHORTCUT) or "")
        right = r.right() - 16
        if shortcut:
            f = QFont(painter.font())
            f.setPixelSize(t.size_xs)
            painter.setFont(f)
            fm = painter.fontMetrics()
            w = fm.horizontalAdvance(shortcut) + 14
            chip = QRect(right - w, r.center().y() - 10, w, 20)
            painter.setPen(QColor(t.border_strong))
            painter.setBrush(QColor(t.bg_surface))
            painter.drawRoundedRect(chip, t.radius_sm, t.radius_sm)
            painter.setPen(QColor(t.text_disabled if disabled else t.text_secondary))
            painter.drawText(chip, Qt.AlignCenter, shortcut)
            right = chip.left() - 10

        text_rect = QRect(r.left() + 18, r.top() + 4, max(10, right - r.left() - 18), r.height() - 8)
        f = QFont(painter.font())
        f.setPixelSize(t.size_sm)
        f.setBold(True)
        painter.setFont(f)
        painter.setPen(QColor(t.text_disabled if disabled else t.text_primary))
        title = painter.fontMetrics().elidedText(str(index.data(Qt.DisplayRole)), Qt.ElideRight, text_rect.width())
        painter.drawText(QRect(text_rect.left(), text_rect.top(), text_rect.width(), 20), Qt.AlignVCenter | Qt.AlignLeft, title)

        desc = str(index.data(ROLE_DESC) or "")
        if desc:
            f = QFont(painter.font())
            f.setPixelSize(t.size_xs)
            f.setBold(False)
            painter.setFont(f)
            painter.setPen(QColor(t.text_disabled if disabled else t.text_secondary))
            desc = painter.fontMetrics().elidedText(desc, Qt.ElideRight, text_rect.width())
            painter.drawText(QRect(text_rect.left(), text_rect.top() + 19, text_rect.width(), 17), Qt.AlignVCenter | Qt.AlignLeft, desc)
        painter.restore()


class CommandLauncher(PopupFrame):
    command_executed = Signal(str)

    CAPTURE_KEYS = (Qt.Key_Escape, Qt.Key_Return, Qt.Key_Enter, Qt.Key_Up, Qt.Key_Down, Qt.Key_PageUp, Qt.Key_PageDown)

    def __init__(self, win, commands_provider: Callable[[], List[Command]]):
        super().__init__(win.viewer)
        self._win = win
        self._provider = commands_provider
        self._commands: List[Command] = []
        self.setObjectName("launcher")

        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 8)
        lay.setSpacing(8)
        self.search = QLineEdit(self)
        self.search.setObjectName("launcher_search")
        self.search.setPlaceholderText("機能を検索（例: だんめん、ブラシ、保存、計測）")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._rebuild)
        lay.addWidget(self.search)

        self.list = QListWidget(self)
        self.list.setObjectName("launcher_list")
        self.list.setFocusPolicy(Qt.NoFocus)
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.list.setItemDelegate(_RowDelegate(lambda: getattr(win, "ui_theme", "dark"), self.list))
        self.list.itemClicked.connect(self._on_item_clicked)
        lay.addWidget(self.list, 1)

        self.footer = QLabel("↑↓ 選ぶ　　Enter 実行　　Esc 閉じる", self)
        self.footer.setObjectName("launcher_footer")
        lay.addWidget(self.footer)

    # --- 開く ---
    def open_launcher(self, initial_text: str = "") -> None:
        self._commands = list(self._provider())
        parent = self.parentWidget()
        w = max(360, min(680, parent.width() - 40))
        h = max(260, min(520, parent.height() - 80))
        self.resize(w, h)
        self.move(max(8, (parent.width() - w) // 2), max(8, int(parent.height() * 0.10)))
        self.search.blockSignals(True)
        self.search.setText(initial_text)
        self.search.blockSignals(False)
        self._rebuild()
        self.open_popup()
        self.search.setFocus(Qt.ShortcutFocusReason)
        self.search.selectAll()

    def toggle(self) -> None:
        if self.is_open():
            self.close_popup()
        else:
            self.open_launcher()

    # --- 一覧 ---
    def _add_header(self, text: str) -> None:
        it = QListWidgetItem(text)
        it.setData(ROLE_KIND, "header")
        it.setFlags(Qt.NoItemFlags)
        self.list.addItem(it)

    def _add_command(self, cmd: Command) -> None:
        reason = cmd.unavailable_reason(self._win)
        it = QListWidgetItem(cmd.title)
        it.setData(ROLE_KIND, "command")
        it.setData(ROLE_ID, cmd.id)
        it.setData(ROLE_DESC, f"今は使えません: {reason}" if reason else cmd.desc)
        it.setData(ROLE_SHORTCUT, cmd.shortcut)
        it.setData(ROLE_DISABLED, bool(reason))
        it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
        self.list.addItem(it)

    def _rebuild(self, *_args) -> None:
        query = self.search.text().strip()
        self.list.clear()
        cmds = self._commands
        if not query:
            by_id = {c.id: c for c in cmds}
            recent = [by_id[i] for i in load_recent_ids() if i in by_id]
            if recent:
                self._add_header("最近使った")
                for c in recent:
                    self._add_command(c)
            for group in GROUP_ORDER:
                members = [c for c in cmds if c.group == group]
                if not members:
                    continue
                self._add_header(group)
                for c in members:
                    self._add_command(c)
        else:
            scored = [(match_score(c, query), i, c) for i, c in enumerate(cmds)]
            hits = [(s, i, c) for s, i, c in scored if s > 0]
            hits.sort(key=lambda x: (-x[0], x[1]))
            if hits:
                self._add_header(f"検索結果 {len(hits)} 件")
                for _s, _i, c in hits:
                    self._add_command(c)
            else:
                self._add_header("見つかりません（別の言葉で試してください。例: けす、きる、みる）")
        self._select_first()

    def _command_rows(self) -> List[int]:
        return [r for r in range(self.list.count()) if self.list.item(r).data(ROLE_KIND) == "command"]

    def _select_first(self) -> None:
        rows = self._command_rows()
        if rows:
            self.list.setCurrentRow(rows[0])
            self.list.scrollToTop()

    def _move_selection(self, delta: int) -> None:
        rows = self._command_rows()
        if not rows:
            return
        cur = self.list.currentRow()
        if cur not in rows:
            target = rows[0]
        else:
            target = rows[max(0, min(len(rows) - 1, rows.index(cur) + delta))]
        self.list.setCurrentRow(target)
        self.list.scrollToItem(self.list.item(target))

    def current_command_id(self) -> Optional[str]:
        it = self.list.currentItem()
        if it is None or it.data(ROLE_KIND) != "command":
            return None
        return str(it.data(ROLE_ID))

    def visible_command_ids(self) -> List[str]:
        return [str(self.list.item(r).data(ROLE_ID)) for r in self._command_rows()]

    # --- 実行 ---
    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        if item.data(ROLE_KIND) == "command":
            self.list.setCurrentItem(item)
            self.run_current()

    def run_current(self) -> bool:
        cmd_id = self.current_command_id()
        if cmd_id is None:
            return False
        cmd = next((c for c in self._commands if c.id == cmd_id), None)
        if cmd is None:
            return False
        if cmd.unavailable_reason(self._win):
            cmd.execute(self._win)  # 理由をトーストで知らせる。ランチャーは開いたまま
            return False
        push_recent_id(cmd.id)
        self.close_popup()
        # 閉じてから実行する（ファイル選択などのダイアログが、閉じかけのランチャーと重ならないように）
        QTimer.singleShot(0, lambda: self._execute(cmd))
        return True

    def _execute(self, cmd: Command) -> None:
        cmd.execute(self._win)
        self.command_executed.emit(cmd.id)

    def popup_key_press(self, event) -> bool:
        key = event.key()
        if key == Qt.Key_Escape:
            self.close_popup()
        elif key in (Qt.Key_Return, Qt.Key_Enter):
            self.run_current()
        elif key == Qt.Key_Down:
            self._move_selection(+1)
        elif key == Qt.Key_Up:
            self._move_selection(-1)
        elif key == Qt.Key_PageDown:
            self._move_selection(+8)
        elif key == Qt.Key_PageUp:
            self._move_selection(-8)
        else:
            return False
        return True
