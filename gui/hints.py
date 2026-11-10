# -*- coding: utf-8 -*-
"""
ヒント機能（新UI）

1. 状況に応じたヒント（HintEngine / HintCard）
   画面の状態を見て「次にやるとよいこと」を 3D 画面の右上にカードで 1 つだけ出す。
   例: 開いた直後 →「まず自動ノイズ除去」、自動処理の後 →「消した点を赤で表示して確認」、
       間引き読込で寄っている →「高精細表示で元の密度に」、編集が残っている →「保存」。
   カードのボタンでその場で実行できる。「閉じる」はこの起動中だけ、「今後表示しない」は以後ずっと出さない。
   ツールを使っている最中（選択ツール・計測・点を選択中）は出さない（モード別の操作ガイドが出る場面）。
   どのヒントを出すかは next_suggestion()（状態 → ヒント の純粋関数）で決める。

2. ヒント集（TIPS / TipsDialog）
   知っていると便利な操作の一覧。起動直後の画面（空状態カード）に 1 つ出し、ヘルプ・検索から一覧を開ける。
   「試す」でその機能を実行、「説明書で読む」で該当の項目を開く。

設定は QSettings（hints_enabled / hints_never）。表示の入切は設定の「説明書」タブ。
"""

import random
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Set

import numpy as np
from PySide6.QtCore import QEvent, QObject, QSettings, Qt, QTimer, Signal
from PySide6.QtWidgets import QDialog, QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

ENABLED_KEY = "hints_enabled"
NEVER_KEY = "hints_never"
HEAVY_POINTS = 8_000_000   # 描画点数「無制限」でこの点数を超えていたら、重さのヒントを出す
ZOOM_RATIO = 0.2           # カメラ距離が点群の大きさのこの割合より近ければ「寄っている」とみなす
STABLE_MS = 1200           # 同じ状態がこれだけ続いてからカードを出す（操作の途中でちらつかせない）


# ----------------------------------------------------------------------------- ヒント集
@dataclass(frozen=True)
class Tip:
    id: str
    text: str
    command_id: str = ""   # 「試す」で実行するコマンド（無ければボタンを出さない）
    topic_id: str = ""     # 「説明書で読む」で開く項目


TIPS: List[Tip] = [
    Tip("search", "Ctrl+K を押すと、機能を名前から探して実行できます。ひらがなで打てます（例: だんめん）。", "", "quick"),
    Tip("right_click", "3D 画面で右クリックすると、その場面で使う操作がカーソルの横に出ます。", "", "quick"),
    Tip("alt_rotate", "選択ツールを使っている最中でも、Alt を押しながら左ドラッグで視点を回せます。", "", "nav"),
    Tip("add_remove", "Shift を押しながら囲むと選択に追加、Ctrl を押しながら囲むと選択から外せます。", "", "select"),
    Tip("wand", "塊の一部を少し選んでから G を押すと、つながっている塊全体を選べます。", "", "tools"),
    Tip("invert", "I で選択を反転できます。残したい所を囲んでから反転すると、周りをまとめて消せます。", "", "select"),
    Tip("section_select", "断面を入れている間は、選択も断面の中の点だけが対象になります。手前や奥を巻き込まずに消せます。", "section.panel", "section"),
    Tip("instant", "「即削除」を入れると、囲んだ瞬間にノイズへ移ります。たくさん消すときに速い方法です。", "mode.instant", "select"),
    Tip("restore", "消した点はノイズレイヤーに残っています。赤で表示して、囲んで復元できます。", "layer.toggle_noise", "layer"),
    Tip("fit", "点群を見失ったら、F で全体表示に戻れます。", "view.fit", "nav"),
    Tip("ortho", "寸法や通りを見るときは、P で正射影に切り替えると遠近の歪みがなくなります。", "view.projection", "nav"),
    Tip("measure_copy", "計測した座標は Ctrl+C でコピーして、Excel や CAD にそのまま貼れます。", "mode.measure", "measure"),
    Tip("dock", "左パネルを格納すると、3D 画面の左端にボタンの列が出ます。必要なパネルだけを小窓で開けます。", "view.sidebar", "quick"),
    Tip("local_clean", "全体は「大まか」で取り、残った所だけを囲んで「範囲内だけ精密除去」をかけると、面を削らずに仕上がります。", "", "tools"),
    Tip("full_save", "間引いて読み込んだファイルは、保存のとき「全点で保存」を選ぶと元の密度で書き出せます。", "", "save"),
    Tip("lod", "高精細表示 (LOD) を入れると、寄って手を止めた所が元ファイルの密度で描かれます。", "section.lod", "detail"),
    Tip("wall", "斜めの壁は、U を押して壁の上の 2 点をクリックすると、壁に沿った視点と断面が使えます。", "section.wall_pick", "section"),
    Tip("brush_size", "ブラシの太さは [ と ] で変えられます。", "mode.brush", "select"),
    Tip("ground", "地面や大きな壁は自動で別レイヤーに分けられます。ロックしておけば、誤って消すことがありません。", "layer.panel", "layer"),
    Tip("manual", "F1 で、使い方を作業順にまとめた操作説明書を開けます。本文のリンクを押すとその機能が動きます。", "help.manual", "start"),
]


def tip_ids() -> List[str]:
    return [t.id for t in TIPS]


# ----------------------------------------------------------------------------- 状況に応じたヒント
@dataclass
class HintState:
    """ヒントの判定に使う画面の状態（collect_state で MainWindow から作る）"""

    has_points: bool = False
    n_points: int = 0
    noise_count: int = 0
    noise_visible: bool = False
    auto_done: bool = False       # 自動ノイズ処理を実行済み
    dirty: bool = False           # 保存していない編集がある
    subsampled: bool = False      # 間引いて読み込んでいる
    subsample_ratio: int = 1      # 何点に 1 点か
    section_active: bool = False
    has_overlay: bool = False     # 重ね描き（読み直し・LOD）が出ている
    lod_on: bool = False
    zoomed_in: bool = False
    budget_unlimited: bool = False
    n_selected: int = 0
    default_mode: bool = True     # 回転モード（選択ツール・パン・計測・壁の指定ではない）


@dataclass(frozen=True)
class Suggestion:
    id: str
    text: str
    action_label: str
    command_id: str


def _suggestions(s: HintState) -> List[Suggestion]:
    """今の状態に当てはまるヒントを、優先する順に返す"""
    out: List[Suggestion] = []
    if not s.auto_done and s.noise_count == 0 and not s.dirty:
        out.append(Suggestion(
            "auto_first", "まず自動ノイズ除去を試しましょう。プリセットは「大まか」から始めるのが安全です。",
            "①自動 を開く", "auto.panel",
        ))
    if s.noise_count > 0 and not s.noise_visible:
        out.append(Suggestion(
            "check_noise", f"{s.noise_count:,} 点をノイズへ移しました。赤で表示すると、消しすぎが無いか確かめられます。",
            "ノイズを表示", "layer.toggle_noise",
        ))
    if s.section_active and s.subsampled and not s.has_overlay and not s.lod_on:
        out.append(Suggestion(
            "section_reload", f"このファイルは {s.subsample_ratio} 点に 1 点で読み込んでいます。断面の箱の中を、元ファイルの密度で重ね描きできます。",
            "全点で読み直す", "section.reload",
        ))
    if s.subsampled and not s.lod_on and s.zoomed_in and not s.section_active:
        out.append(Suggestion(
            "detail_lod", f"このファイルは {s.subsample_ratio} 点に 1 点で読み込んでいます。高精細表示を入れると、寄った所が元ファイルの密度で描かれます。",
            "高精細表示を入れる", "section.lod",
        ))
    if s.budget_unlimited and s.n_points > HEAVY_POINTS and not s.section_active:
        out.append(Suggestion(
            "heavy_unlimited", "描画点数が「無制限」です。回転が重いときは 200 万〜500 万点に下げ、細部は断面や高精細表示で見るのがおすすめです。",
            "設定を開く", "help.settings",
        ))
    if s.noise_count > 0 and s.noise_visible:
        out.append(Suggestion(
            "manual_fix", "残ったノイズは、矩形やブラシで囲んで Del で消せます。消しすぎた赤い点は、囲んで復元できます。",
            "矩形選択にする", "mode.rect",
        ))
    if s.dirty and s.noise_count > 0:
        out.append(Suggestion("save", "保存していない編集があります。", "保存する", "file.save"))
    return out


def next_suggestion(state: HintState, hidden: Optional[Set[str]] = None) -> Optional[Suggestion]:
    """今出すヒントを 1 つ返す（無ければ None）。hidden は閉じた・今後表示しないにしたヒントの ID"""
    if not state.has_points or not state.default_mode or state.n_selected > 0:
        return None
    hidden = hidden or set()
    for sug in _suggestions(state):
        if sug.id not in hidden:
            return sug
    return None


def collect_state(win, noise_count: int = 0) -> HintState:
    """MainWindow とビューアから状態を集める（重い計算はしない。ノイズ点数は呼び出し側が覚えている値）"""
    v = getattr(win, "viewer", None)
    st = HintState()
    if v is None or v.points_raw is None or len(v.points_raw) == 0:
        return st
    st.has_points = True
    st.n_points = int(len(v.points_raw))
    st.noise_count = int(noise_count)
    chk = getattr(win, "chk_layer_noise", None)
    st.noise_visible = bool(chk.isChecked()) if chk is not None else False
    undo_btn = getattr(win, "auto_undo_btn", None)
    st.auto_done = bool(undo_btn is not None and undo_btn.isEnabled()) or st.noise_count > 0
    st.dirty = bool(getattr(win, "_dirty", False))
    scans = getattr(win, "current_scans", None) or []
    st.subsampled = any(getattr(s, "is_subsampled", False) for s in scans)
    if st.subsampled:
        orig = sum(int(getattr(s, "original_point_count", len(s.points))) for s in scans)
        st.subsample_ratio = max(2, int(round(orig / max(1, st.n_points))))
    st.section_active = bool(v.is_section_active())
    st.has_overlay = bool(getattr(v, "has_section_overlay", False))
    lod = getattr(win, "lod", None)
    st.lod_on = bool(lod is not None and lod.is_active)
    try:
        span = float(np.linalg.norm(np.asarray(v.bounds_max, dtype=np.float64) - np.asarray(v.bounds_min, dtype=np.float64)))
        st.zoomed_in = bool(span > 0 and float(v.camera_distance) < ZOOM_RATIO * span)
    except Exception:
        st.zoomed_in = False
    st.budget_unlimited = int(getattr(v, "point_budget", 1)) <= 0
    sel = v.selected_indices
    st.n_selected = int(len(sel)) if sel is not None else 0
    st.default_mode = not (
        getattr(v, "selection_mode", False) or getattr(v, "pan_mode", False)
        or getattr(v, "measure_mode", False) or getattr(v, "wall_align_mode", False)
    )
    return st


def load_never() -> Set[str]:
    val = QSettings("E57PointCloudTool", "Settings").value(NEVER_KEY, [])
    if isinstance(val, str):
        val = [val] if val else []
    return {str(v) for v in (val or [])}


def save_never(ids: Set[str]) -> None:
    QSettings("E57PointCloudTool", "Settings").setValue(NEVER_KEY, sorted(ids))


def hints_enabled() -> bool:
    return bool(QSettings("E57PointCloudTool", "Settings").value(ENABLED_KEY, True, type=bool))


class HintCard(QFrame):
    """3D 画面の右上に出すヒントのカード（実行・閉じる・今後表示しない）"""

    action_clicked = Signal()
    close_clicked = Signal()
    never_clicked = Signal()

    def __init__(self, viewer):
        super().__init__(viewer)
        self.setObjectName("hint_card")
        self.setAttribute(Qt.WA_StyledBackground, True)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(6)
        self.title_label = QLabel("ヒント", self)
        self.title_label.setObjectName("hint_card_title")
        lay.addWidget(self.title_label)
        self.text_label = QLabel("", self)
        self.text_label.setObjectName("hint_card_text")
        self.text_label.setWordWrap(True)
        self.text_label.setFixedWidth(340)
        lay.addWidget(self.text_label)
        row = QHBoxLayout()
        row.setSpacing(6)
        self.btn_action = QPushButton("", self)
        self.btn_action.setProperty("role", "primary")
        self.btn_action.clicked.connect(lambda _=False: self.action_clicked.emit())
        row.addWidget(self.btn_action)
        self.btn_close = QPushButton("閉じる", self)
        self.btn_close.setProperty("role", "ghost")
        self.btn_close.setToolTip("このヒントを閉じる（次に起動したときはまた出ます）")
        self.btn_close.clicked.connect(lambda _=False: self.close_clicked.emit())
        row.addWidget(self.btn_close)
        self.btn_never = QPushButton("今後表示しない", self)
        self.btn_never.setProperty("role", "ghost")
        self.btn_never.setToolTip("このヒントを今後出さない（設定 → 説明書 で元に戻せます）")
        self.btn_never.clicked.connect(lambda _=False: self.never_clicked.emit())
        row.addWidget(self.btn_never)
        row.addStretch(1)
        lay.addLayout(row)
        self.hide()
        viewer.installEventFilter(self)

    def set_suggestion(self, sug: Suggestion) -> None:
        self.text_label.setText(sug.text)
        self.btn_action.setText(sug.action_label)
        self.adjustSize()
        self.reposition()

    def reposition(self) -> None:
        parent = self.parentWidget()
        if parent is None:
            return
        self.adjustSize()
        self.move(max(8, parent.width() - self.width() - 12), 12)
        self.raise_()

    def eventFilter(self, obj, event) -> bool:  # type: ignore[override]
        if event.type() == QEvent.Resize and self.isVisible():
            self.reposition()
        return False


class HintEngine(QObject):
    """状態を見てヒントのカードを出し入れする"""

    shown = Signal(str)  # 出したヒントの ID

    def __init__(self, win, run_command: Callable[[str], bool]):
        super().__init__(win)
        self._win = win
        self._run_command = run_command
        self._enabled = hints_enabled()
        self._snoozed: Set[str] = set()       # この起動中は出さない（閉じた・実行した）
        self._never: Set[str] = load_never()  # 今後ずっと出さない
        self._noise_count = 0
        self._candidate: Optional[str] = None
        self._candidate_since = 0.0
        self.current: Optional[Suggestion] = None
        self.card = HintCard(win.viewer)
        self.card.action_clicked.connect(self._on_action)
        self.card.close_clicked.connect(self._on_close)
        self.card.never_clicked.connect(self._on_never)
        win.viewer.layers_updated.connect(self._on_layers_updated)
        self._timer = QTimer(self)
        self._timer.setInterval(500)
        self._timer.timeout.connect(self.refresh)
        self._timer.start()

    # --- 設定 ---
    def is_enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = bool(enabled)
        QSettings("E57PointCloudTool", "Settings").setValue(ENABLED_KEY, self._enabled)
        self.refresh(force=True)

    def reset_hidden(self) -> None:
        """閉じたヒント・今後表示しないにしたヒントを、もう一度出るようにする"""
        self._snoozed.clear()
        self._never.clear()
        save_never(self._never)
        self.refresh(force=True)

    def hidden_ids(self) -> Set[str]:
        return set(self._snoozed) | set(self._never)

    # --- 状態 ---
    def _on_layers_updated(self, _normal: int, noise: int) -> None:
        self._noise_count = int(noise)

    def state(self) -> HintState:
        return collect_state(self._win, self._noise_count)

    def refresh(self, force: bool = False) -> None:
        """状態を見直してカードを出し入れする。force=True は「落ち着くまで待つ」を省く（テスト・設定変更用）"""
        sug = next_suggestion(self.state(), self.hidden_ids()) if self._enabled else None
        if sug is None:
            self._candidate = None
            self.current = None
            if not self.card.isHidden():
                self.card.hide()
            return
        now = time.monotonic()
        if sug.id != self._candidate:
            self._candidate = sug.id
            self._candidate_since = now
        if not force and (now - self._candidate_since) * 1000.0 < STABLE_MS and (self.current is None or self.current.id != sug.id):
            if self.current is not None and not self.card.isHidden():
                self.card.hide()  # 状態が変わった。新しいヒントは落ち着いてから出す
                self.current = None
            return
        changed = self.current is None or self.current.id != sug.id or self.current.text != sug.text
        self.current = sug
        if changed:
            self.card.set_suggestion(sug)
        if self.card.isHidden():
            self.card.show()
            self.card.reposition()
            self.shown.emit(sug.id)

    # --- カードのボタン ---
    def _dismiss_current(self, never: bool) -> Optional[Suggestion]:
        sug = self.current
        if sug is None:
            return None
        self._snoozed.add(sug.id)
        if never:
            self._never.add(sug.id)
            save_never(self._never)
        self.current = None
        self._candidate = None
        self.card.hide()
        return sug

    def _on_action(self) -> None:
        sug = self._dismiss_current(never=False)
        if sug is not None:
            self._run_command(sug.command_id)

    def _on_close(self) -> None:
        self._dismiss_current(never=False)

    def _on_never(self) -> None:
        sug = self._dismiss_current(never=True)
        if sug is not None and hasattr(self._win, "show_toast"):
            self._win.show_toast("このヒントは今後表示しません（設定 → 説明書 で元に戻せます）")


class TipsDialog(QDialog):
    """ヒント集（前へ・次へ・試す・説明書で読む）"""

    def __init__(self, win, run_command: Callable[[str], bool], open_manual: Callable[[str], object],
                 command_title: Callable[[str], str]):
        super().__init__(win)
        self.setObjectName("tips_dialog")
        self.setWindowTitle("ヒント集")
        self.setModal(False)
        self._run_command = run_command
        self._open_manual = open_manual
        self._command_title = command_title
        self._index = 0
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 14, 16, 12)
        lay.setSpacing(10)
        self.counter_label = QLabel("", self)
        self.counter_label.setObjectName("tips_counter")
        lay.addWidget(self.counter_label)
        self.text_label = QLabel("", self)
        self.text_label.setObjectName("tips_text")
        self.text_label.setWordWrap(True)
        self.text_label.setMinimumHeight(72)
        self.text_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        lay.addWidget(self.text_label, 1)
        row = QHBoxLayout()
        row.setSpacing(6)
        self.btn_prev = QPushButton("前へ", self)
        self.btn_prev.setProperty("role", "outline")
        self.btn_prev.clicked.connect(lambda _=False: self.step(-1))
        row.addWidget(self.btn_prev)
        self.btn_next = QPushButton("次へ", self)
        self.btn_next.setProperty("role", "outline")
        self.btn_next.clicked.connect(lambda _=False: self.step(+1))
        row.addWidget(self.btn_next)
        row.addStretch(1)
        self.btn_try = QPushButton("試す", self)
        self.btn_try.setProperty("role", "primary")
        self.btn_try.clicked.connect(lambda _=False: self._on_try())
        row.addWidget(self.btn_try)
        self.btn_manual = QPushButton("説明書で読む", self)
        self.btn_manual.setProperty("role", "outline")
        self.btn_manual.clicked.connect(lambda _=False: self._on_manual())
        row.addWidget(self.btn_manual)
        self.btn_close = QPushButton("閉じる", self)
        self.btn_close.setProperty("role", "ghost")
        self.btn_close.clicked.connect(self.close)
        row.addWidget(self.btn_close)
        lay.addLayout(row)
        self.resize(620, 210)
        self.show_tip(random.randrange(len(TIPS)))

    def current_tip(self) -> Tip:
        return TIPS[self._index]

    def show_tip(self, index: int) -> None:
        self._index = index % len(TIPS)
        tip = TIPS[self._index]
        self.counter_label.setText(f"ヒント {self._index + 1} / {len(TIPS)}")
        self.text_label.setText(tip.text)
        self.btn_try.setVisible(bool(tip.command_id))
        if tip.command_id:
            self.btn_try.setText(f"試す: {self._command_title(tip.command_id)}")
        self.btn_manual.setVisible(bool(tip.topic_id))

    def show_tip_id(self, tip_id: str) -> None:
        for i, t in enumerate(TIPS):
            if t.id == tip_id:
                self.show_tip(i)
                return

    def step(self, delta: int) -> None:
        self.show_tip(self._index + delta)

    def _on_try(self) -> None:
        tip = self.current_tip()
        if tip.command_id:
            self._run_command(tip.command_id)

    def _on_manual(self) -> None:
        tip = self.current_tip()
        if tip.topic_id:
            self._open_manual(tip.topic_id)
