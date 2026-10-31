# -*- coding: utf-8 -*-
"""
折り返しレイアウト (FlowLayout)

横幅が足りないときにボタンを次の行へ折り返す QLayout。
上部ツールバー・下部アクションバーが狭い画面（1366×768 @125% など）でボタンの途中で切れる問題への対策。
Qt 公式サンプル (flowlayout) の PySide6 移植。
"""

from typing import List, Optional

from PySide6.QtCore import QPoint, QRect, QSize, Qt
from PySide6.QtWidgets import QLayout, QLayoutItem, QSizePolicy, QStyle, QWidget


class FlowLayout(QLayout):
    def __init__(self, parent: Optional[QWidget] = None, margin: int = 0, h_spacing: int = 4, v_spacing: int = 4):
        super().__init__(parent)
        self._items: List[QLayoutItem] = []
        self._h_space = h_spacing
        self._v_space = v_spacing
        if parent is not None:
            self.setContentsMargins(margin, margin, margin, margin)

    # --- QLayout 必須実装 ---
    def addItem(self, item: QLayoutItem) -> None:  # type: ignore[override]
        self._items.append(item)

    def count(self) -> int:  # type: ignore[override]
        return len(self._items)

    def itemAt(self, index: int):  # type: ignore[override]
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def takeAt(self, index: int):  # type: ignore[override]
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    def expandingDirections(self):  # type: ignore[override]
        return Qt.Orientations(Qt.Orientation(0))

    def hasHeightForWidth(self) -> bool:  # type: ignore[override]
        return True

    def heightForWidth(self, width: int) -> int:  # type: ignore[override]
        return self._do_layout(QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect: QRect) -> None:  # type: ignore[override]
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)

    def sizeHint(self) -> QSize:  # type: ignore[override]
        return self.minimumSize()

    def minimumSize(self) -> QSize:  # type: ignore[override]
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        m = self.contentsMargins()
        size += QSize(m.left() + m.right(), m.top() + m.bottom())
        return size

    # --- 内部 ---
    def _spacing(self, horizontal: bool) -> int:
        explicit = self._h_space if horizontal else self._v_space
        if explicit >= 0:
            return explicit
        parent = self.parent()
        if isinstance(parent, QWidget):
            pm = QStyle.PM_LayoutHorizontalSpacing if horizontal else QStyle.PM_LayoutVerticalSpacing
            return parent.style().pixelMetric(pm, None, parent)
        return 4

    def _do_layout(self, rect: QRect, test_only: bool) -> int:
        m = self.contentsMargins()
        effective = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        x = effective.x()
        y = effective.y()
        line_height = 0
        h_space = self._spacing(True)
        v_space = self._spacing(False)

        for item in self._items:
            w = item.widget()
            if w is not None and not w.isVisibleTo(w.parentWidget()) and w.isHidden():
                continue
            hint = item.sizeHint()
            next_x = x + hint.width() + h_space
            if next_x - h_space > effective.right() + 1 and line_height > 0:
                x = effective.x()
                y = y + line_height + v_space
                next_x = x + hint.width() + h_space
                line_height = 0
            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), hint))
            x = next_x
            line_height = max(line_height, hint.height())
        return y + line_height - rect.y() + m.bottom()


def replace_layout_with_flow(container: QWidget, h_spacing: int = 4, v_spacing: int = 4) -> FlowLayout:
    """
    既存の QBoxLayout を持つコンテナのレイアウトを FlowLayout に差し替える。
    子ウィジェットはそのままの順序で移し替える（ストレッチ等の非ウィジェット項目は捨てる）。
    """
    old = container.layout()
    widgets: List[QWidget] = []
    margins = None
    if old is not None:
        margins = old.contentsMargins()
        while old.count():
            item = old.takeAt(0)
            w = item.widget()
            if w is not None:
                widgets.append(w)
        # Qt は setLayout の上書きを許さないため、旧レイアウトを即座に削除してから新レイアウトを付ける
        try:
            import shiboken6

            shiboken6.delete(old)
        except Exception:
            old.setParent(None)
    flow = FlowLayout(container, 0, h_spacing, v_spacing)
    if margins is not None:
        flow.setContentsMargins(margins)
    for w in widgets:
        flow.addWidget(w)
    container.setLayout(flow)
    return flow
