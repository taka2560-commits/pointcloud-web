# -*- coding: utf-8 -*-
"""
状態と次の一手を伝える部品（新UI専用。gui/modern_ui.py の apply_modern_ui から呼ばれる）

考え方: 画面が黙ると迷う。状態は色だけでなく文字と印で伝え、次にできることを添える（「UIの状態表示20選」を参考にした）。
  - 通知（下部バーの右）  : 印（成功・注意・失敗・お知らせ）+ 文 + 次の一手のボタン。点を動かした直後は「元に戻す」
  - 未保存の変更          : ステータスバーに「未保存の変更あり」。押すと保存
  - 入力の誤り            : 直す場所を赤い枠で、理由を文で示す（入力 E57 / 出力 E57）
処理中・完了・無効の理由はボタン側の部品（gui/button_kit.py）が受け持つ。
通信・ログイン・権限・承認待ちなどの状態は、手元のファイルだけを扱うこのツールには無いので作らない。
"""

import os
import re
from typing import Callable, Dict, List, Optional, Tuple

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtWidgets import QFileDialog, QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QWidget

from core.logger import log_warning
from gui.design_tokens import Tokens, get_tokens
from gui.icons import icon_pixmap, icon_size, themed_icon

KIND_ICONS: Dict[str, str] = {"success": "check-circle", "warning": "alert", "error": "error", "info": "info"}
TOAST_ICON_PX = 14
ACTION_TOAST_MS = 6000      # 次の一手のボタンが付く通知は、押す時間が取れるように長めに出す
UNDO_ACTION_TEXT = "元に戻す"

_WARNING_WORDS = ("⚠", "中止", "中断", "破棄", "見つかりません", "できません", "ありません", "使えません", "読み直せません")
_ERROR_WORDS = ("エラー", "失敗", "開けませんでした")
_SUCCESS_WORDS = ("完了", "保存しました", "✅", "復元", "除去しました", "移動しました", "消去", "合格", "追加しました", "コピーしました")
_POINT_COUNT_RE = re.compile(r"\d\s*点")
_NOT_UNDOABLE_WORDS = ("元に戻しました", "やり直しました", "選択しました", "読み込みました")


def classify_message(message: str) -> str:
    """通知の文から種類（success / warning / error / info）を決める"""
    m = message or ""
    if any(k in m for k in _ERROR_WORDS):
        return "error"
    if any(k in m for k in _WARNING_WORDS):
        return "warning"
    if any(k in m for k in _SUCCESS_WORDS):
        return "success"
    return "info"


def is_edit_report(message: str) -> bool:
    """点を動かした結果を知らせる文か（「○○ 点を…しました」。取り消し・やり直し・選択・読み込みの報告は除く）"""
    m = message or ""
    return bool(_POINT_COUNT_RE.search(m)) and not any(k in m for k in _NOT_UNDOABLE_WORDS)


def _tokens_of(win) -> Tokens:
    return get_tokens(getattr(win, "ui_theme", "dark"))


def _repolish(w: QWidget) -> None:
    st = w.style()
    st.unpolish(w)
    st.polish(w)
    w.update()


# ----------------------------------------------------------------------------- 通知
class ToastHost(QFrame):
    """通知の入れ物: [印] [文] [次の一手のボタン]。枠の色と印で種類を示す"""

    def __init__(self, label: QLabel, tokens_provider: Callable[[], Tokens], parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._tokens = tokens_provider
        self._kind = "info"
        self._action: Optional[Callable[[], None]] = None
        self.setObjectName("toast_host")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setProperty("kind", "info")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 1, 6, 1)
        lay.setSpacing(6)

        self.icon_label = QLabel(self)
        self.icon_label.setObjectName("toast_icon")
        self.icon_label.setFixedSize(icon_size(TOAST_ICON_PX))
        lay.addWidget(self.icon_label)

        self.label = label
        label.setParent(self)
        label.setVisible(True)
        lay.addWidget(label, 1)

        self.action_btn = QPushButton("", self)
        self.action_btn.setObjectName("toast_action")
        self.action_btn.setCursor(Qt.PointingHandCursor)
        self.action_btn.setFocusPolicy(Qt.NoFocus)
        self.action_btn.clicked.connect(self._run_action)
        self.action_btn.setVisible(False)
        lay.addWidget(self.action_btn)
        self.set_kind("info")

    def kind(self) -> str:
        return self._kind

    def set_kind(self, kind: str) -> None:
        self._kind = kind if kind in KIND_ICONS else "info"
        self.setProperty("kind", self._kind)
        _repolish(self)
        self.refresh_theme()

    def refresh_theme(self) -> None:
        t = self._tokens()
        color = {"success": t.toast_success, "warning": t.toast_warning, "error": t.toast_error}.get(self._kind, t.toast_info)
        self.icon_label.setPixmap(icon_pixmap(KIND_ICONS[self._kind], color, TOAST_ICON_PX))

    def set_action(self, text: str, callback: Callable[[], None], tip: str = "") -> None:
        self._action = callback
        self.action_btn.setText(text)
        self.action_btn.setToolTip(tip)
        self.action_btn.setVisible(True)

    def clear_action(self) -> None:
        self._action = None
        self.action_btn.setVisible(False)

    def has_action(self) -> bool:
        return self._action is not None and not self.action_btn.isHidden()

    def _run_action(self) -> None:
        fn = self._action
        self.clear_action()
        if fn is not None:
            fn()


    # --- 3D 画面の右下に浮かせる（下部バーの中に置くと、出るたびにバーが 2 段になって画面が動くため） ---
    MARGIN = 12

    def place(self) -> None:
        """文の長さに合わせて大きさを決め、親（3D 画面）の右下に置く。長い文は折り返す"""
        parent = self.parentWidget()
        if parent is None:
            return
        lay = self.layout()
        m = lay.contentsMargins()
        extra = m.left() + m.right() + self.icon_label.width() + lay.spacing() + 2 * self.frameWidth() + 6
        if not self.action_btn.isHidden():
            extra += self.action_btn.sizeHint().width() + lay.spacing()
        avail = max(160, parent.width() - 2 * self.MARGIN - extra)
        text_w = self.label.fontMetrics().horizontalAdvance(self.label.text()) + 8
        self.label.setWordWrap(text_w > avail)
        self.label.setFixedWidth(min(text_w, avail))
        lay.invalidate()
        self.adjustSize()
        x = max(self.MARGIN, parent.width() - self.width() - self.MARGIN)
        y = max(self.MARGIN, parent.height() - self.height() - self.MARGIN)
        self.move(x, y)
        self.raise_()

    def eventFilter(self, obj: QObject, event) -> bool:  # type: ignore[override]
        if obj is self.parentWidget() and event.type() == QEvent.Resize and not self.isHidden():
            self.place()
        return False


def install_toast_host(win) -> Optional[ToastHost]:
    """下部バーの通知ラベルを ToastHost で包み、3D 画面の右下に浮かせる"""
    label = getattr(win, "toast_label", None)
    viewer = getattr(win, "viewer", None)
    if label is None or viewer is None:
        return None
    bar = getattr(win, "bottom_overlay_bar", None)
    lay = bar.layout() if bar is not None else None
    if lay is not None and lay.indexOf(label) >= 0:
        lay.removeWidget(label)
    host = ToastHost(label, lambda: _tokens_of(win), viewer)
    viewer.installEventFilter(host)
    host.setVisible(False)
    win.toast_host = host
    return host


# ----------------------------------------------------------------------------- 未保存の変更
class UnsavedPill(QPushButton):
    """ステータスバーの「未保存の変更あり」。押すと保存する（保存していない編集があることを、色と印と文で知らせる）"""

    def __init__(self, win, parent: Optional[QWidget] = None):
        super().__init__("未保存の変更あり", parent)
        self._win = win
        self.setObjectName("unsaved_pill")
        self.setCursor(Qt.PointingHandCursor)
        self.setFocusPolicy(Qt.NoFocus)
        self.setToolTip("保存していない編集があります。押すと保存します (Ctrl+S)")
        self.clicked.connect(lambda _checked=False: win._save_current_scans())
        self.setVisible(False)
        self.refresh_theme()

    def refresh_theme(self) -> None:
        t = _tokens_of(self._win)
        self.setIcon(themed_icon("alert", t.badge_text, t.badge_text, 13))
        self.setIconSize(icon_size(13))


def install_unsaved_pill(win) -> Optional[UnsavedPill]:
    sb = getattr(win, "modern_status_bar", None)
    if sb is None:
        return None
    pill = UnsavedPill(win, sb)
    sb.insertPermanentWidget(0, pill)
    pill.setVisible(bool(getattr(win, "_dirty", False)))
    win.unsaved_pill = pill
    return pill


# ----------------------------------------------------------------------------- 入力の誤り
def validate_input_path(text: str) -> Optional[str]:
    """入力 E57 の欄。問題があれば理由の文、無ければ None"""
    path = (text or "").strip().strip('"')
    if not path:
        return None
    if not os.path.exists(path):
        return "入力 E57: ファイルまたはフォルダが見つかりません。場所と名前を確認してください"
    if os.path.isfile(path) and not path.lower().endswith(".e57"):
        return "入力 E57: E57 ファイル（拡張子 .e57）を指定してください"
    return None


def validate_output_path(text: str) -> Optional[str]:
    """出力 E57 の欄。問題があれば理由の文、無ければ None"""
    path = (text or "").strip().strip('"')
    if not path:
        return None
    folder = os.path.dirname(path)
    if folder and not os.path.isdir(folder):
        return "出力 E57: 保存先のフォルダがありません。「参照」で選び直してください"
    if os.path.isdir(path):
        return "出力 E57: フォルダではなく、ファイル名（○○.e57）まで指定してください"
    return None


class FieldErrors(QObject):
    """入力欄の誤りを、欄の赤い枠（直す場所）と下の 1 行（理由）で示す。確定したときに調べ、直ったらすぐ消す"""

    def __init__(self, win, row_label: QLabel, row_icon: QLabel, row: QWidget):
        super().__init__(win)
        self._win = win
        self.label = row_label
        self.icon = row_icon
        self.row = row
        self._fields: List[Tuple[QLineEdit, Callable[[str], Optional[str]]]] = []
        self._errors: Dict[int, str] = {}

    def watch(self, edit: QLineEdit, validator: Callable[[str], Optional[str]]) -> None:
        self._fields.append((edit, validator))
        edit.editingFinished.connect(lambda e=edit, v=validator: self.check(e, v))
        edit.textChanged.connect(lambda _t, e=edit, v=validator: self._recheck_if_flagged(e, v))

    def check(self, edit: QLineEdit, validator: Callable[[str], Optional[str]]) -> Optional[str]:
        msg = validator(edit.text())
        self._set(edit, msg)
        return msg

    def check_all(self) -> List[str]:
        return [m for m in (self.check(e, v) for e, v in self._fields) if m]

    def _recheck_if_flagged(self, edit: QLineEdit, validator: Callable[[str], Optional[str]]) -> None:
        # 打っている途中の文字で誤りを出さない。すでに誤りが出ている欄だけ、直った瞬間に消す
        if id(edit) in self._errors:
            self._set(edit, validator(edit.text()))

    def _set(self, edit: QLineEdit, msg: Optional[str]) -> None:
        if msg:
            self._errors[id(edit)] = msg
        else:
            self._errors.pop(id(edit), None)
        flagged = bool(msg)
        if bool(edit.property("invalid")) != flagged:
            edit.setProperty("invalid", flagged)
            _repolish(edit)
        self._render()

    def _render(self) -> None:
        msgs = list(self._errors.values())
        self.label.setText(" / ".join(msgs))
        self.row.setVisible(bool(msgs))

    def messages(self) -> List[str]:
        return list(self._errors.values())

    def refresh_theme(self) -> None:
        t = _tokens_of(self._win)
        self.icon.setPixmap(icon_pixmap("error", t.danger_text, 13))


def install_field_errors(win) -> Optional[FieldErrors]:
    """ファイル入出力の箱の下に、誤りの理由を出す 1 行を足す"""
    grp = getattr(win, "file_group", None)
    lay = grp.content_widget.layout() if grp is not None else None
    if not isinstance(lay, QGridLayout) or getattr(win, "input_edit", None) is None:
        return None
    row = QWidget(grp.content_widget)
    h = QHBoxLayout(row)
    h.setContentsMargins(0, 0, 0, 0)
    h.setSpacing(5)
    icon = QLabel(row)
    icon.setObjectName("field_error_icon")
    icon.setFixedSize(icon_size(13))
    h.addWidget(icon, 0, Qt.AlignTop)
    label = QLabel("", row)
    label.setObjectName("field_error")
    label.setWordWrap(True)
    h.addWidget(label, 1)
    lay.addWidget(row, lay.rowCount(), 0, 1, max(1, lay.columnCount()))
    row.setVisible(False)
    errors = FieldErrors(win, label, icon, row)
    errors.refresh_theme()
    errors.watch(win.input_edit, validate_input_path)
    if getattr(win, "output_edit", None) is not None:
        errors.watch(win.output_edit, validate_output_path)
    win.field_errors = errors
    return errors


# ----------------------------------------------------------------------------- 保存の関連操作（分割ボタンのメニュー）
def save_as(win) -> bool:
    """保存先を選んでから保存する。取りやめたら何もしない"""
    path, _ = QFileDialog.getSaveFileName(
        win, "保存先の E57 ファイルを選ぶ", win.output_edit.text(), "E57 点群ファイル (*.e57);;すべてのファイル (*.*)"
    )
    if not path:
        return False
    win.output_edit.setText(path)
    win._save_current_scans()
    return True


# ----------------------------------------------------------------------------- まとめ
class StatusFeedback(QObject):
    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self.toast_host: Optional[ToastHost] = None
        self.unsaved_pill: Optional[UnsavedPill] = None
        self.field_errors: Optional[FieldErrors] = None
        for name, fn in (("toast_host", install_toast_host), ("unsaved_pill", install_unsaved_pill),
                         ("field_errors", install_field_errors)):
            try:
                setattr(self, name, fn(win))
            except Exception as exc:  # 表示の補助なので、失敗しても本体は動かす
                log_warning(f"新UI: 状態表示（{name}）の組み込みに失敗しました: {exc}")

    def refresh_theme(self) -> None:
        for part in (self.toast_host, self.unsaved_pill, self.field_errors):
            if part is not None:
                part.refresh_theme()


def install_status_feedback(win) -> StatusFeedback:
    fb = StatusFeedback(win)
    win.status_feedback = fb
    return fb
