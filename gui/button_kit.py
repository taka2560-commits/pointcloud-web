# -*- coding: utf-8 -*-
"""
ボタンの役割と状態をそろえる部品（新UI専用。gui/modern_ui.py の apply_modern_ui から呼ばれる）

考え方: ボタンは色と形だけでなく「役割」と「状態」で選ぶ（「UIボタン20選」を参考にした）。
  役割 … ROLE_GUIDE のとおり。見た目は gui/design_tokens.py の QSS が role プロパティで描く
  状態 … hover / 押下 / 無効 / 選択中は QSS。ここでは QSS だけでは出せないものを足す
         - アイコン+文字（保存・削除・戻すなど。意味を素早く伝える。削除は赤 + ごみ箱）
         - 処理中（「保存中…」と回る印）と完了（「保存済み」とチェック）
         - 無効の理由（押せないボタンにマウスを置くと「なぜ押せないか」を出す）
         - キーボードで移動中の現在位置（Tab で移ったボタンだけ太い枠。マウスで押しただけでは出さない）
         - オン/オフの印（選択中はチェック、そうでなければ丸。色だけに頼らない）
         - 分割ボタン（主操作 + 関連する操作のメニュー）
処理そのものは持たない。既存のボタンに後付けするだけで、従来UIには入れない。
"""

import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from PySide6.QtCore import QEvent, QObject, Qt, QTimer
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QAbstractButton, QApplication, QHBoxLayout, QMenu, QPushButton, QSizePolicy, QWidget,
)

from gui.design_tokens import Tokens, get_tokens
from gui.icons import icon_pixmap, icon_size, themed_icon

# 役割の一覧: (role, 見た目, 使いどころ)。説明書の「ボタンと表示の見方」とテストが参照する
ROLE_GUIDE: Tuple[Tuple[str, str, str], ...] = (
    ("primary", "塗りつぶし", "その場でいちばん大事な操作。1 つの場所に 1 つだけ（保存・設定を保存・ファイルを開く）"),
    ("outline", "枠だけ", "主操作を支える操作（戻す・視点の切り替え・参照）"),
    ("soft", "淡い塗り", "使う回数の少ない操作（ログのフォルダを開く・作業用データの管理）"),
    ("ghost", "文字だけ", "いちばん軽い操作（全て展開・閉じる・最近使ったファイル）"),
    ("danger", "赤 + ごみ箱", "点を消す・切り落とすなど、結果に注意が要る操作"),
    ("segment", "並んだ切り替え", "道具やタブをどれか 1 つ選ぶ（回転・矩形・①自動）"),
    ("pill", "丸い小さなボタン", "検索や、その場で値を選ぶ小さな操作（機能を検索・断面の厚み）"),
)

# 属性名 → アイコン名（アイコン+文字にするボタン）
BUTTON_ICONS: Dict[str, str] = {
    "save_manual_btn": "save",
    "btn_save_settings": "save",
    "btn_save_layer_proj": "save",
    "btn_load_layer_proj": "open-file",
    "delete_pts_btn": "trash",
    "floating_del_btn": "trash",
    "btn_crop_section": "trash",
    "undo_btn": "undo",
    "auto_undo_btn": "undo",
    "redo_btn": "redo",
    "auto_redo_btn": "redo",
    "auto_exec_btn": "play",
    "btn_restore_noise": "restore",
    "btn_restore_selected_noise": "restore",
    "floating_restore_btn": "restore",
    "floating_clear_btn": "close",
    "nav_btn_screenshot": "camera",
}

# アイコンを付けたので外す、文字の矢印（属性名 → 新しい文字）
BUTTON_TEXTS: Dict[str, str] = {
    "undo_btn": "戻す",
    "redo_btn": "やり直し",
    "auto_undo_btn": "1つ戻る",
    "auto_redo_btn": "1つ進む",
}

# 文字から外したショートカットは、ツールチップに書く（属性名 → 説明）
BUTTON_TIPS: Dict[str, str] = {
    "undo_btn": "直前の編集を取り消します (Ctrl+Z)",
    "redo_btn": "取り消した編集をもう一度行います (Ctrl+Y)",
}

# 属性名 → 無効のときに出す理由
_NEED_SELECTION = "点を選ぶと使えます（矩形・ブラシなどで囲む）"
DISABLED_REASONS: Dict[str, str] = {
    "floating_del_btn": _NEED_SELECTION,
    "floating_wand_btn": _NEED_SELECTION,
    "floating_local_clean_btn": _NEED_SELECTION,
    "floating_move_btn": _NEED_SELECTION,
    "floating_restore_btn": _NEED_SELECTION,
    "floating_clear_btn": "選んでいる点がありません",
    "undo_btn": "戻せる操作がまだありません",
    "auto_undo_btn": "戻せる操作がまだありません",
    "redo_btn": "やり直せる操作がありません（「戻す」のあとで使えます）",
    "auto_redo_btn": "やり直せる操作がありません（「1つ戻る」のあとで使えます）",
    "btn_poly_confirm": "多角形の道具で 3 点以上を打つと確定できます",
    "save_manual_btn": "保存しています。終わるまでお待ちください",
    "auto_exec_btn": "処理しています。終わるまでお待ちください",
}

# オン/オフの印を付けるボタン（選択中はチェック、そうでなければ丸）
#   表示パネルの「凹凸陰影」「距離連動サイズ」には付けない: 3 つ並びの行で、印のぶん広がると
#   明るいテーマで左パネルの最小幅を超える（Gate4 が検査）。この 2 つは塗りつぶしと太字でオンを示す
TOGGLE_MARK_BUTTONS: Tuple[str, ...] = ("btn_instant_del", "btn_clean_mode", "btn_sec_wall_pick")

ICON_PX = 14
_BAND_PARENTS = ("nav_bar", "bottom_bar", "panel_dock")


def tokens_for(win) -> Tokens:
    return get_tokens(getattr(win, "ui_theme", "dark"))


def repolish(w: QWidget) -> None:
    st = w.style()
    st.unpolish(w)
    st.polish(w)
    w.update()


def in_band(btn: QWidget) -> bool:
    """上部バー・下部バーなど「帯」の上にあるか（帯は地の色が違うので、文字とアイコンの色も帯用にする）"""
    p = btn.parentWidget()
    while p is not None:
        if p.objectName() in _BAND_PARENTS:
            return True
        p = p.parentWidget()
    return False


def icon_colors_for(btn: QAbstractButton, t: Tokens) -> Tuple[str, str]:
    """ボタンの役割から、アイコンの色（通常, 無効）を決める。文字と同じ色にする"""
    role = str(btn.property("role") or "")
    tone = str(btn.property("tone") or "")
    band = in_band(btn)
    disabled = t.band_muted if band else t.text_disabled
    if btn.objectName() == "execute_button":
        return t.success_text, disabled
    if tone in t.tones:
        return t.tones[tone][1], disabled
    if role == "primary":
        return t.primary_text, disabled
    if role == "danger":
        return t.text_on_accent, disabled
    if role == "ghost":
        return t.accent_soft, disabled
    if band:
        return t.band_text, disabled
    return t.text_primary, disabled


def set_button_icon(btn: QAbstractButton, name: str, t: Tokens, size: int = ICON_PX) -> None:
    """ボタンにアイコンを付ける（名前はプロパティに残し、テーマ切り替えで塗り直せるようにする）"""
    color, disabled = icon_colors_for(btn, t)
    btn.setProperty("icon_name", name)
    btn.setIcon(themed_icon(name, color, disabled, size))
    btn.setIconSize(icon_size(size))


def refresh_button_icon(btn: QAbstractButton, t: Tokens) -> None:
    name = btn.property("icon_name")
    if name:
        set_button_icon(btn, str(name), t)


# ----------------------------------------------------------------------------- キーボードの現在位置
class KeyboardFocusRing(QObject):
    """
    Tab / Shift+Tab / ショートカットでボタンへ移ったときだけ kbfocus="true" を付ける（QSS が太い枠を描く）。
    マウスで押したあとに枠が残り続けると画面がうるさくなるので、キーボードで移動中だけに限る。
    """

    _REASONS = (Qt.TabFocusReason, Qt.BacktabFocusReason, Qt.ShortcutFocusReason)

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # type: ignore[override]
        et = event.type()
        if et == QEvent.FocusIn:
            if isinstance(obj, QAbstractButton) and event.reason() in self._REASONS:
                obj.setProperty("kbfocus", True)
                repolish(obj)
        elif et == QEvent.FocusOut:
            if isinstance(obj, QAbstractButton) and obj.property("kbfocus"):
                obj.setProperty("kbfocus", False)
                repolish(obj)
        return False


def install_keyboard_focus_ring(app: Optional[QApplication] = None) -> Optional[KeyboardFocusRing]:
    app = app or QApplication.instance()
    if app is None:
        return None
    ring = getattr(app, "_e57_kbfocus_ring", None)
    if ring is None:
        ring = KeyboardFocusRing(app)
        app.installEventFilter(ring)
        app._e57_kbfocus_ring = ring  # type: ignore[attr-defined]
    return ring


# ----------------------------------------------------------------------------- 無効の理由
class DisabledReason(QObject):
    """無効のあいだだけ、ツールチップを「なぜ押せないか」に差し替える（有効に戻ったら元の説明へ）"""

    def __init__(self, btn: QAbstractButton, reason: str):
        super().__init__(btn)
        self.btn = btn
        self.reason = reason
        self._normal_tip = btn.toolTip()
        btn.setProperty("disabled_reason", reason)
        btn.installEventFilter(self)
        self._sync()

    def _sync(self) -> None:
        if self.btn.isEnabled():
            self.btn.setToolTip(self._normal_tip)
        else:
            self.btn.setToolTip(self.reason)

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # type: ignore[override]
        if obj is self.btn:
            et = event.type()
            if et == QEvent.EnabledChange:
                self._sync()
            elif et == QEvent.ToolTipChange and self.btn.isEnabled():
                self._normal_tip = self.btn.toolTip()
        return False


def set_disabled_reason(btn: QAbstractButton, reason: str) -> DisabledReason:
    old = getattr(btn, "_disabled_reason_helper", None)
    if old is not None:
        old.reason = reason
        btn.setProperty("disabled_reason", reason)
        old._sync()
        return old
    helper = DisabledReason(btn, reason)
    btn._disabled_reason_helper = helper  # type: ignore[attr-defined]
    return helper


def disabled_reason(btn: QAbstractButton) -> str:
    return str(btn.property("disabled_reason") or "")


# ----------------------------------------------------------------------------- 処理中・完了
class BusyBinding(QObject):
    """
    処理のあいだ無効になるボタンに「処理中」と「完了」の状態を付ける。

    本体のコードは「開始で setEnabled(False)、終了で setEnabled(True)」とするだけなので、その切り替わりを見て
      無効になった → 文字を busy_text（例: 保存中…）に替え、回る印を出す
      有効に戻った → 元の文字とアイコンへ戻す
    成功したことが分かったら flash_done() を呼ぶと、しばらく「保存済み」とチェックを出してから元に戻る。
    """

    def __init__(self, btn: QAbstractButton, busy_text: str, tokens_provider: Callable[[], Tokens], done_text: str = "完了"):
        super().__init__(btn)
        self.btn = btn
        self.busy_text = busy_text
        self.done_text = done_text
        self.followers: List[QWidget] = []
        self._tokens = tokens_provider
        self._state = "idle"
        self._normal_text = btn.text()
        self._normal_icon = QIcon(btn.icon())
        self._normal_min_w = btn.minimumWidth()
        self._left_busy_at = 0.0
        self._angle = 0
        self._hold_after_modal = False
        self._hold_ms = 2200
        self._spin = QTimer(self)
        self._spin.setInterval(90)
        self._spin.timeout.connect(self._tick)
        self._done_timer = QTimer(self)
        self._done_timer.setSingleShot(True)
        self._done_timer.timeout.connect(self._end_done)
        btn.installEventFilter(self)
        if not btn.isEnabled():
            self._enter_busy()

    # --- 状態 ---
    def state(self) -> str:
        return self._state

    def just_finished(self, within_sec: float = 0.5) -> bool:
        """直前（within_sec 秒以内）に処理中から戻ったか"""
        return self._state == "idle" and (time.monotonic() - self._left_busy_at) <= within_sec

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # type: ignore[override]
        if obj is self.btn and event.type() == QEvent.EnabledChange:
            if not self.btn.isEnabled():
                self._enter_busy()
            elif self._state == "busy":
                self._leave_busy()
        return False

    def _remember_normal(self) -> None:
        self._normal_text = self.btn.text()
        self._normal_icon = QIcon(self.btn.icon())
        self._normal_min_w = self.btn.minimumWidth()

    def _restore_normal(self) -> None:
        self._spin.stop()
        self._done_timer.stop()
        self.btn.setProperty("busy", False)
        self.btn.setProperty("done", False)
        self.btn.setText(self._normal_text)
        if self.btn.property("icon_name"):
            # 処理中・完了のあいだにテーマが替わっていても、今のテーマの色で描き直す
            refresh_button_icon(self.btn, self._tokens())
        else:
            self.btn.setIcon(self._normal_icon)
        self.btn.setMinimumWidth(self._normal_min_w)
        self._state = "idle"
        repolish(self.btn)

    def _enter_busy(self) -> None:
        if self._state == "done":
            self._restore_normal()
        if self._state == "idle":
            self._remember_normal()
            self.btn.setMinimumWidth(max(self._normal_min_w, self.btn.width()))  # 文字が替わっても幅を動かさない
        self._state = "busy"
        self.btn.setProperty("done", False)
        self.btn.setProperty("busy", True)
        self.btn.setText(self.busy_text)
        self._angle = 0
        self._tick()
        from gui.motion import motion_enabled

        if motion_enabled():
            self._spin.start()
        for w in self.followers:
            w.setEnabled(False)
        repolish(self.btn)

    def _leave_busy(self) -> None:
        self._restore_normal()
        self._left_busy_at = time.monotonic()
        for w in self.followers:
            w.setEnabled(True)

    def _tick(self) -> None:
        t = self._tokens()
        pm = icon_pixmap("spinner", t.text_primary, ICON_PX, rotate_deg=self._angle)
        icon = QIcon()
        for mode in (QIcon.Normal, QIcon.Disabled, QIcon.Active):
            icon.addPixmap(pm, mode, QIcon.Off)
        self.btn.setIcon(icon)
        self.btn.setIconSize(icon_size(ICON_PX))
        self._angle = (self._angle + 30) % 360

    def flash_done(self, text: Optional[str] = None, hold_ms: int = 2200) -> None:
        """「保存済み」のような完了の表示を出し、しばらくして元に戻す（確認のダイアログが出ているあいだは保つ）"""
        if self._state == "busy":
            return
        if self._state == "idle":
            self._remember_normal()
            self.btn.setMinimumWidth(max(self._normal_min_w, self.btn.width()))
        t = self._tokens()
        self._state = "done"
        self._hold_ms = int(hold_ms)
        self._hold_after_modal = False
        self.btn.setProperty("busy", False)
        self.btn.setProperty("done", True)
        self.btn.setText(text or self.done_text)
        self.btn.setIcon(themed_icon("check", t.success_text, t.success_text, ICON_PX))
        self.btn.setIconSize(icon_size(ICON_PX))
        repolish(self.btn)
        self._done_timer.start(self._hold_ms)

    def clear_done(self) -> None:
        """「保存済み」などの完了の表示を、待たずに元へ戻す（完了の直後にまた編集が入ったとき）"""
        if self._state == "done":
            self._restore_normal()

    def _end_done(self) -> None:
        if self._state != "done":
            return
        if QApplication.activeModalWidget() is not None:
            self._hold_after_modal = True
            self._done_timer.start(300)
            return
        if self._hold_after_modal:
            self._hold_after_modal = False
            self._done_timer.start(self._hold_ms)
            return
        self._restore_normal()

    def refresh_theme(self) -> None:
        if self._state == "idle":
            self._normal_icon = QIcon(self.btn.icon())


def bind_busy(btn: QAbstractButton, busy_text: str, tokens_provider: Callable[[], Tokens], done_text: str = "完了") -> BusyBinding:
    old = getattr(btn, "_busy_binding", None)
    if old is not None:
        return old
    binding = BusyBinding(btn, busy_text, tokens_provider, done_text)
    btn._busy_binding = binding  # type: ignore[attr-defined]
    return binding


# ----------------------------------------------------------------------------- オン/オフの印
class ToggleMark(QObject):
    """
    オン/オフのボタンに印を付ける: 選択中はチェック、そうでなければ丸（印の有無で幅は変わらない）。
    1 つの QIcon にオン用とオフ用の絵を登録し、どちらを描くかは Qt がボタンの状態から決める。
    toggled シグナルで差し替える方式にしないこと: 本体は blockSignals の中で setChecked することがあり、
    そのとき印だけが古い状態のまま残る（N キーのノイズ専用、Esc でのピック終了などで起きた）。
    """

    def __init__(self, btn: QAbstractButton, tokens_provider: Callable[[], Tokens]):
        super().__init__(btn)
        self.btn = btn
        self._tokens = tokens_provider
        self.refresh()

    def _colors(self, t: Tokens) -> Tuple[str, str]:
        """(選択中の色, そうでないときの色)"""
        role = str(self.btn.property("role") or "")
        tone = str(self.btn.property("tone") or "")
        band = in_band(self.btn)
        if tone in t.tones:
            return "#ffffff", t.tones[tone][1]
        on = t.text_on_accent if role == "danger-toggle" else t.on_text
        off = t.band_muted if band else t.text_secondary
        return on, off

    def refresh(self) -> None:
        """テーマの色でオン用・オフ用の絵を作り直す（テーマ切り替えのときに呼ぶ）"""
        t = self._tokens()
        on, off = self._colors(t)
        pm_on = icon_pixmap("check", on, ICON_PX)
        pm_off = icon_pixmap("circle", off, ICON_PX)
        icon = QIcon()
        for mode in (QIcon.Normal, QIcon.Active, QIcon.Disabled, QIcon.Selected):
            icon.addPixmap(pm_off, mode, QIcon.Off)
            icon.addPixmap(pm_on, mode, QIcon.On)
        self.btn.setIcon(icon)
        self.btn.setIconSize(icon_size(ICON_PX))

    def mark(self) -> str:
        """いま出ている印（"on" = チェック / "off" = 丸）。ボタンの状態そのものから決まる"""
        return "on" if self.btn.isChecked() else "off"


def install_toggle_mark(btn: QAbstractButton, tokens_provider: Callable[[], Tokens]) -> ToggleMark:
    old = getattr(btn, "_toggle_mark", None)
    if old is not None:
        old.refresh()
        return old
    mark = ToggleMark(btn, tokens_provider)
    btn._toggle_mark = mark  # type: ignore[attr-defined]
    return mark


# ----------------------------------------------------------------------------- 分割ボタン
class SplitButton(QWidget):
    """
    主操作のボタンの右に「関連する操作」のメニューを開く小さなボタンを付ける（分割ボタン）。
    左はそのまま主操作、右の矢印でメニュー。entries は (表示名, 実行する関数) の並びで、None は区切り線。
    """

    def __init__(self, main_btn: QPushButton, entries: Sequence[Optional[Tuple[str, Callable[[], None]]]],
                 tokens_provider: Callable[[], Tokens], arrow_tip: str = "関連する操作", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.main_btn = main_btn
        self._tokens = tokens_provider
        self.setObjectName("split_button")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        main_btn.setParent(self)
        main_btn.setProperty("split", "main")
        lay.addWidget(main_btn, 1)

        self.arrow_btn = QPushButton("", self)
        self.arrow_btn.setProperty("role", str(main_btn.property("role") or "primary"))
        self.arrow_btn.setProperty("split", "arrow")
        self.arrow_btn.setToolTip(arrow_tip)
        self.arrow_btn.setAccessibleName(arrow_tip)
        self.arrow_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        self.arrow_btn.setFixedWidth(28)
        self.menu = QMenu(self.arrow_btn)
        self.actions_by_title: Dict[str, object] = {}
        for entry in entries:
            if entry is None:
                self.menu.addSeparator()
                continue
            title, fn = entry
            act = self.menu.addAction(title)
            act.triggered.connect(lambda _checked=False, f=fn: f())
            self.actions_by_title[title] = act
        self.arrow_btn.setMenu(self.menu)
        # 矢印を押すとメニューが開き、マウスを離す合図はメニューへ行く。そのままだと押下の見た目（沈み・半透明）が
        # 残るので、メニューが閉じたら「離れた」合図を送って元に戻す
        self.menu.aboutToHide.connect(self._release_arrow)
        lay.addWidget(self.arrow_btn, 0)
        self.refresh_theme()
        main_btn.show()

    def _release_arrow(self) -> None:
        try:
            QApplication.sendEvent(self.arrow_btn, QEvent(QEvent.Leave))
            self.arrow_btn.setDown(False)
        except RuntimeError:
            pass  # ウィンドウを閉じている最中

    def refresh_theme(self) -> None:
        set_button_icon(self.arrow_btn, "chevron-down", self._tokens(), 12)


# ----------------------------------------------------------------------------- まとめて適用
class ButtonKit(QObject):
    """ウィンドウのボタンにアイコン・無効の理由・印・処理中/完了を付けて、テーマ切り替えで塗り直す"""

    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self.busy: Dict[str, BusyBinding] = {}
        self.marks: List[ToggleMark] = []
        self.splits: List[SplitButton] = []
        self.focus_ring = install_keyboard_focus_ring()
        self._apply()

    def tokens(self) -> Tokens:
        return tokens_for(self.win)

    def _btn(self, name: str) -> Optional[QAbstractButton]:
        b = getattr(self.win, name, None)
        return b if isinstance(b, QAbstractButton) else None

    def _apply(self) -> None:
        t = self.tokens()
        for name, text in BUTTON_TEXTS.items():
            b = self._btn(name)
            if b is not None:
                b.setText(text)
        for name, tip in BUTTON_TIPS.items():
            b = self._btn(name)
            if b is not None:
                b.setToolTip(tip)
        for name, icon in BUTTON_ICONS.items():
            b = self._btn(name)
            if b is not None:
                set_button_icon(b, icon, t)
        for name in TOGGLE_MARK_BUTTONS:
            b = self._btn(name)
            if b is not None and b.isCheckable():
                self.marks.append(install_toggle_mark(b, self.tokens))
        # 処理中/完了（無効の理由より先に付ける。処理中の文字に替わってから理由のツールチップが出る）
        for name, busy_text, done_text in (
            ("save_manual_btn", "保存中…", "保存済み"),
            ("auto_exec_btn", "実行中…", "完了"),
        ):
            b = self._btn(name)
            if b is not None:
                self.busy[name] = bind_busy(b, busy_text, self.tokens, done_text)
        for name, reason in DISABLED_REASONS.items():
            b = self._btn(name)
            if b is not None:
                set_disabled_reason(b, reason)

    def add_split(self, split: SplitButton) -> None:
        self.splits.append(split)

    def set_icon(self, btn: QAbstractButton, name: str) -> None:
        set_button_icon(btn, name, self.tokens())

    def flash_done(self, name: str, text: Optional[str] = None) -> None:
        binding = self.busy.get(name)
        if binding is not None:
            binding.flash_done(text)

    def refresh_theme(self) -> None:
        """テーマが替わったら、アイコンをその色で描き直す"""
        t = self.tokens()
        for b in self.win.findChildren(QAbstractButton):
            if b.property("icon_name") and getattr(b, "_toggle_mark", None) is None:
                binding = getattr(b, "_busy_binding", None)
                if binding is not None and binding.state() != "idle":
                    continue
                refresh_button_icon(b, t)
        for m in self.marks:
            m.refresh()
        for s in self.splits:
            s.refresh_theme()
        for binding in self.busy.values():
            binding.refresh_theme()


def apply_button_kit(win) -> ButtonKit:
    kit = ButtonKit(win)
    win.button_kit = kit
    return kit
