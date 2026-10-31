# -*- coding: utf-8 -*-
"""
新UI（modern）アダプタ

従来の MainWindow._init_ui() が組み立てた画面に対し、起動後に「後付け」で以下を行う。
従来UIのコードは一切変更・削除せず、UI モードが classic のときは本モジュールは呼ばれない
（gui/ui_mode.py）。そのため設定・環境変数・起動引数でいつでも改修前の UI に戻せる。

行うこと（docs/DESIGN_STYLE_PLAN.md の P2〜P5 に相当）:
  1. インライン QSS の無効化（トークン生成 QSS を効かせる。色付きボタンの hover/disabled が効かない問題も解消）
  2. 絵文字の除去（ボタン・見出し・タブ・コンボ・ステータス・トースト）
  3. 役割プロパティの付与（主ボタン / 危険トグル / セグメント）と状態表現の :checked 化
  4. 重複操作の非表示（サイドバーのモードボタン群・形状コンボ・側の計測ボタン等。上部バーに一本化）
  5. サイドバーを作業順に並べ替え（①自動除去 → ②手動修正 → ③レイヤー確認 → 断面 → 表示 → 設定）、保存グループを新設
  6. メニューバー（ファイル / 編集 / 表示 / ヘルプ）、Ctrl+O / Ctrl+S / Ctrl+Q、最近使ったファイル
  7. ステータスバー（状態文・進捗・点数バッジ）を新設し、サイドバー最下部から移動
  8. 3Dビュー中央の空状態カード（ドロップ／開く／最近使ったファイル）
  9. 上部・下部バーを折り返しレイアウトにして狭い画面でもボタンが切れないようにする
"""

import os
import re
from typing import Callable, Iterable, List, Optional

from PySide6.QtCore import QEvent, QObject, QSettings, Qt, QTimer, QUrl
from PySide6.QtGui import QAction, QDesktopServices, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QStatusBar,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.logger import get_app_dir, log_warning, open_log_folder
from gui.flow_layout import replace_layout_with_flow
from gui.ui_mode import UI_MODE_CLASSIC, set_saved_ui_mode

_SETTINGS_ORG = "E57PointCloudTool"
_SETTINGS_APP = "Settings"
RECENT_FILES_KEY = "recent_files"
MAX_RECENT = 6

# 絵文字・絵文字的記号（◀▼▶ などの幾何記号、━┃ の罫線、①②③、⇄ などの矢印は UI 記号として残す）
_EMOJI_RE = re.compile(
    "["
    "\U0001F000-\U0001FAFF"  # 絵文字全般
    "\U0001F1E6-\U0001F1FF"  # 国旗
    "☀-➿"          # 記号・装飾 (⚡ ✂ ✨ ★ ♻ ✖ ⚙ ...)
    "⬀-⯿"          # ⬜ ⭕ ⬠ ⬅ ...
    "⌀-⏿"          # ⌨ ⏱ ...
    "️‍⃣"     # 異体字セレクタ・ZWJ・囲み
    "]"
)
_SPACE_RE = re.compile(r"[ 　]{2,}")


def strip_emoji(text: str) -> str:
    """UI 文字列から絵文字を取り除き、余った空白を整える"""
    if not text:
        return text
    cleaned = _EMOJI_RE.sub("", str(text))
    cleaned = _SPACE_RE.sub(" ", cleaned)
    # 先頭に残った区切り記号や空白を除去（例: "⚡ 即削除: ON" → "即削除: ON"）
    cleaned = cleaned.lstrip(" 　・:：")
    return cleaned.rstrip()


def has_emoji(text: str) -> bool:
    return bool(_EMOJI_RE.search(text or ""))


# ----------------------------------------------------------------------------- 低レベル補助
def _repolish(w: QWidget) -> None:
    st = w.style()
    st.unpolish(w)
    st.polish(w)
    w.update()


def _disable_inline_stylesheet(w: QWidget) -> None:
    """
    以後そのウィジェットへの setStyleSheet() を無効化する（Python 側のインスタンス属性で上書き）。
    従来コードの「状態をインライン QSS の差し替えで表現する」箇所を、新UIでは :checked / role で表現するため。
    """
    try:
        w.setStyleSheet("")
    except Exception:
        pass
    try:
        w.setStyleSheet = lambda *_args, **_kw: None  # type: ignore[assignment]
    except Exception:
        pass


def _install_text_stripper(w: QWidget) -> None:
    """以後の setText() で絵文字を自動除去する（ボタン・ラベル・チェックボックス）"""
    cls = type(w)
    base = None
    for c in (QPushButton, QLabel, QCheckBox, QRadioButton):
        if isinstance(w, c):
            base = c
            break
    if base is None:
        return
    try:
        w.setText(strip_emoji(w.text()))
        w.setText = lambda s, _w=w, _b=base: _b.setText(_w, strip_emoji(s))  # type: ignore[assignment]
    except Exception:
        pass


def _name_collapsible(grp) -> None:
    """CollapsibleGroupBox の見出し・トグルに objectName を付け、インライン QSS を外す（トークン QSS で描くため）"""
    hdr = getattr(grp, "_header_widget", None)
    ttl = getattr(grp, "_title_label", None)
    tgl = getattr(grp, "_toggle_btn", None)
    for w in (grp, hdr, ttl, tgl):
        if w is not None and w.styleSheet():
            _disable_inline_stylesheet(w)
    if hdr is not None:
        hdr.setObjectName("collapsible_header")
    if ttl is not None:
        ttl.setObjectName("collapsible_title")
    if tgl is not None:
        tgl.setObjectName("collapsible_toggle")


def _find_label(parent: QWidget, text: str) -> Optional[QLabel]:
    for lbl in parent.findChildren(QLabel):
        if lbl.text().strip() == text:
            return lbl
    return None


# ----------------------------------------------------------------------------- 最近使ったファイル
def load_recent_files() -> List[str]:
    s = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
    val = s.value(RECENT_FILES_KEY, [])
    if isinstance(val, str):
        val = [val]
    out: List[str] = []
    for p in val or []:
        p = str(p)
        if p and p not in out and os.path.exists(p):
            out.append(p)
    return out[:MAX_RECENT]


def add_recent_file(path: str) -> None:
    if not path:
        return
    path = os.path.normpath(path)
    items = [p for p in load_recent_files() if os.path.normpath(p) != path]
    items.insert(0, path)
    s = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
    s.setValue(RECENT_FILES_KEY, items[:MAX_RECENT])


# ----------------------------------------------------------------------------- 空状態カード
class EmptyStateCard(QFrame):
    """3Dビュー中央に表示する「E57 を開く」導線（点群が無いときだけ表示）"""

    def __init__(self, win):
        super().__init__(win.viewer)
        self._win = win
        self.setObjectName("empty_state_card")
        self.setAttribute(Qt.WA_StyledBackground, True)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(28, 24, 28, 24)
        lay.setSpacing(10)

        title = QLabel("E57 点群を開く")
        title.setObjectName("empty_state_title")
        title.setAlignment(Qt.AlignCenter)
        lay.addWidget(title)

        hint = QLabel("ここにファイルやフォルダをドラッグ＆ドロップできます（Ctrl+O で選択）")
        hint.setObjectName("empty_state_hint")
        hint.setAlignment(Qt.AlignCenter)
        hint.setWordWrap(True)
        lay.addWidget(hint)

        row = QHBoxLayout()
        row.setSpacing(8)
        btn_open = QPushButton("ファイルを開く")
        btn_open.setProperty("role", "primary")
        btn_open.clicked.connect(win._browse_input_file)
        row.addWidget(btn_open)
        btn_folder = QPushButton("フォルダを一括で開く")
        btn_folder.setProperty("role", "outline")
        btn_folder.clicked.connect(win._browse_input_folder)
        row.addWidget(btn_folder)
        lay.addLayout(row)

        self._recent_title = QLabel("最近使ったファイル")
        self._recent_title.setObjectName("empty_state_recent_title")
        lay.addWidget(self._recent_title)
        self._recent_box = QVBoxLayout()
        self._recent_box.setSpacing(4)
        lay.addLayout(self._recent_box)

        self.refresh_recent()
        win.viewer.installEventFilter(self)
        self._recenter()

    def refresh_recent(self) -> None:
        while self._recent_box.count():
            item = self._recent_box.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        recent = load_recent_files()
        self._recent_title.setVisible(bool(recent))
        for p in recent:
            b = QPushButton(os.path.basename(p) or p)
            b.setProperty("role", "ghost")
            b.setToolTip(p)
            b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(lambda _=False, path=p: self._open_recent(path))
            self._recent_box.addWidget(b)
        self.adjustSize()
        self._recenter()

    def _open_recent(self, path: str) -> None:
        win = self._win
        win.input_edit.setText(path)
        win._on_input_file_changed(path)

    def _recenter(self) -> None:
        parent = self.parentWidget()
        if parent is None:
            return
        self.adjustSize()
        w = max(self.sizeHint().width(), 360)
        h = self.sizeHint().height()
        self.resize(w, h)
        self.move(max(0, (parent.width() - w) // 2), max(0, (parent.height() - h) // 2))
        self.raise_()

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # type: ignore[override]
        if event.type() == QEvent.Resize:
            self._recenter()
        return False


# ----------------------------------------------------------------------------- 各ステップ
def _neutralize_inline_styles(win) -> None:
    """インライン QSS を外し、以後の差し替えも無効化する（新UIはトークン生成 QSS のみで描く）"""
    keep_names = {"collapsible_content"}  # 自分自身に限定済みの安全な QSS
    for w in win.findChildren(QWidget):
        name = w.objectName()
        if name in keep_names:
            continue
        if w.styleSheet():
            _disable_inline_stylesheet(w)
    # 実行時にインライン QSS を差し替える既知のウィジェット（初期状態では空でも後から設定される）
    for attr in (
        "btn_measure", "select_mode_btn", "btn_instant_del", "chk_instant_del",
        "btn_clean_mode", "btn_clean_mode_side", "floating_sel_badge", "toast_label",
    ):
        w = getattr(win, attr, None)
        if w is not None:
            _disable_inline_stylesheet(w)
    # ウィンドウ自身（従来テーマが設定されている場合）
    try:
        win.setStyleSheet("")
    except Exception:
        pass


def _strip_all_emoji(win) -> None:
    for w in win.findChildren(QWidget):
        if isinstance(w, (QPushButton, QLabel, QCheckBox, QRadioButton)):
            _install_text_stripper(w)
        elif isinstance(w, QComboBox):
            for i in range(w.count()):
                w.setItemText(i, strip_emoji(w.itemText(i)))
        elif isinstance(w, QTabWidget):
            for i in range(w.count()):
                w.setTabText(i, strip_emoji(w.tabText(i)))
        elif isinstance(w, QGroupBox) and w.title():
            w.setTitle(strip_emoji(w.title()))
    win.setWindowTitle(strip_emoji(win.windowTitle()))


def _assign_roles_and_names(win) -> None:
    g = lambda n: getattr(win, n, None)  # noqa: E731

    for name in ("btn_instant_del", "chk_instant_del"):
        b = g(name)
        if b is not None:
            b.setProperty("role", "danger-toggle")
    for name in ("btn_clean_mode", "btn_clean_mode_side", "select_mode_btn", "btn_measure"):
        b = g(name)
        if b is not None:
            b.setProperty("role", "segment")
    for name in ("nav_btn_screenshot",):
        b = g(name)
        if b is not None:
            b.setProperty("role", "outline")

    # 機能別の控えめな色分け（tone）: 「どれが何か」を色でも見分けられるようにする（意味は design_tokens.TONES 参照）
    tone_map = {
        "violet": ("btn_magic_wand", "floating_wand_btn", "btn_wall_pick"),
        "teal": ("btn_local_clean", "floating_local_clean_btn", "btn_quick_edge_clean", "btn_add_layer"),
        "amber": ("btn_keep_main_cluster", "btn_auto_ground", "btn_preset_fine"),
        "green": ("btn_auto_plane", "btn_restore_selected_noise", "btn_restore_noise", "floating_restore_btn", "btn_preset_rough"),
        "blue": ("btn_preset_std", "btn_move_to_layer", "btn_sec_from_current"),
    }
    for tone, names in tone_map.items():
        for name in names:
            b = g(name)
            if b is not None:
                b.setProperty("tone", tone)
                # tone は role より後に定義された QSS で描く。outline/primary の指定は tone に置き換える
                if b.property("role") in ("outline", "secondary"):
                    b.setProperty("role", "")
    # 断面外クロップ・選択削除は危険寄りの一括操作
    for name in ("btn_crop_section", "delete_pts_btn"):
        b = g(name)
        if b is not None:
            b.setProperty("role", "danger")

    # プリセット名は短く（詳細はツールチップ）。サイドバー幅 320px でも 3 つ並ぶようにする
    for name, label, tip in (
        ("btn_preset_rough", "大まか", "大まか（粗取り）: 浮遊塊除去 + 標準 SOR。まず全体のゴミを落とすとき"),
        ("btn_preset_std", "標準", "標準バランス: SOR + 孤立クラスタ。迷ったらこれ"),
        ("btn_preset_fine", "細かい", "細かい（精密）: エッジ飛散・ヒゲ除去 + 高感度 SOR。仕上げ用"),
    ):
        b = g(name)
        if b is not None:
            b.setText(label)
            b.setToolTip(tip)

    if g("point_count_label") is not None:
        win.point_count_label.setObjectName("point_count_pill")
    if g("floating_sel_badge") is not None:
        win.floating_sel_badge.setObjectName("floating_sel_badge")
        win.floating_sel_badge.setProperty("active", False)
    if g("toast_label") is not None:
        win.toast_label.setObjectName("toast_label")
        win.toast_label.setProperty("kind", "info")
    if g("bottom_overlay_bar") is not None:
        win.bottom_overlay_bar.setObjectName("bottom_bar")
        win.bottom_overlay_bar.setAttribute(Qt.WA_StyledBackground, True)
    nav_bar = win.btn_toggle_sidebar.parentWidget() if g("btn_toggle_sidebar") is not None else None
    if nav_bar is not None:
        nav_bar.setObjectName("nav_bar")
        nav_bar.setAttribute(Qt.WA_StyledBackground, True)
        win._nav_bar_widget = nav_bar
    if g("sidebar_widget") is not None:
        win.sidebar_widget.setObjectName("sidebar_widget")

    # 折りたたみグループの見出し・トグルに名前を付けてトークン QSS で描く
    for grp in win.findChildren(QGroupBox):
        if grp.objectName() == "collapsible_group":
            _name_collapsible(grp)

    # 高さ色分け枠（QFrame）に名前を付ける
    for fr in win.findChildren(QFrame):
        if fr.objectName() == "" and getattr(win, "height_range_frame", None) is fr:
            fr.setObjectName("height_range_frame")

    # 注記ラベル（※ で始まる）は補足スタイルに
    for lbl in win.findChildren(QLabel):
        t = lbl.text().strip()
        if t.startswith("※") or t.startswith("💡") or t.startswith("配色範囲"):
            lbl.setProperty("hint", True)


def _hide_duplicate_controls(win) -> None:
    """上部バーと重複するサイドバー内のモード切替を隠す（操作経路を1本化）"""
    for name in (
        "side_btn_orbit", "side_btn_pan", "side_btn_brush", "side_btn_rect", "side_btn_circle", "side_btn_poly",
        "btn_clean_mode_side", "chk_instant_del", "select_shape_combo", "select_mode_btn", "btn_measure",
    ):
        w = getattr(win, name, None)
        if w is not None:
            w.hide()
    tools = getattr(win, "tools_group", None)
    if tools is not None:
        lbl = _find_label(tools.content_widget, "形状:")
        if lbl is not None:
            lbl.hide()


def _reorganize_sidebar(win) -> None:
    """サイドバーを作業順に並べ替え、保存グループを新設し、タブの並び・名称を揃える"""
    lay = win.sidebar_widget.layout()
    if lay is None:
        return

    groups = [
        getattr(win, "auto_group", None),
        getattr(win, "tools_group", None),
        getattr(win, "layer_group", None),
        getattr(win, "section_group", None),
        getattr(win, "view_group", None),
        getattr(win, "settings_group", None),
    ]
    groups = [g for g in groups if g is not None]
    base = lay.indexOf(win.sidebar_nav_bar) + 2  # ナビバーと「全て展開/折りたたみ」行の直後
    for grp in groups:
        lay.removeWidget(grp)
    for i, grp in enumerate(groups):
        lay.insertWidget(base + i, grp)

    # 保存グループ（⑤）: ファイル入出力から「保存対象」コンボと保存ボタンを移す
    group_cls = type(win.file_group)
    save_group = group_cls("保存")
    _name_collapsible(save_group)  # 後から作る箱なので、インライン QSS 無効化と命名をここで行う
    save_group.toggled_collapse.connect(win._on_panel_collapse_toggled)
    sl = QHBoxLayout(save_group.content_widget)
    sl.setContentsMargins(6, 4, 6, 4)
    sl.setSpacing(6)
    file_layout = win.file_group.content_widget.layout()
    lbl_target = _find_label(win.file_group.content_widget, "保存対象:")
    for w in (lbl_target, win.export_mode_combo, win.save_manual_btn):
        if w is None:
            continue
        if file_layout is not None:
            file_layout.removeWidget(w)
        w.setParent(save_group.content_widget)
        w.show()
    if lbl_target is not None:
        lbl_target.setMinimumWidth(lbl_target.sizeHint().width())  # フォントが広いテーマでも切れない
        sl.addWidget(lbl_target)
    sl.addWidget(win.export_mode_combo, 1)
    win.save_manual_btn.setMaximumWidth(16777215)
    win.save_manual_btn.setMinimumWidth(96)
    win.save_manual_btn.setText("保存 (Ctrl+S)")
    sl.addWidget(win.save_manual_btn)
    win.save_group = save_group
    lay.insertWidget(lay.indexOf(win.log_group), save_group)

    # タブバー: 作業順（全て / ①自動 / ②手動 / ③確認 / 断面 / 表示 / 設定）
    nav_layout = win.sidebar_nav_bar.layout()
    order = [
        ("all", "全て"),
        ("filter", "①自動"),
        ("edit", "②手動"),
        ("layer", "③確認"),
        ("section", "断面"),
        ("view", "表示"),
        ("settings", "設定"),
    ]
    if nav_layout is not None:
        for key, _ in order:
            b = win.sidebar_tab_btns.get(key)
            if b is not None:
                nav_layout.removeWidget(b)
        for key, label in order:
            b = win.sidebar_tab_btns.get(key)
            if b is not None:
                b.setText(label)
                nav_layout.addWidget(b, 1)
    tips = {
        "filter": "① 自動ノイズ除去: レシピを選んで実行",
        "edit": "② 手動修正: 選んで消す・塊を選ぶ・局所精密除去",
        "layer": "③ 確認: レイヤー表示・ロック・復元・自動分類",
    }
    for key, tip in tips.items():
        b = win.sidebar_tab_btns.get(key)
        if b is not None:
            b.setToolTip(tip)

    # 情報量の削減はタブ（①自動 / ②手動 / ③確認 …）で行い、グループ自体は展開したままにする
    # （タブで開いたグループが畳まれていると、もう一度クリックが必要になり操作性が落ちるため）

    # サイドバーの既定幅と最小幅はテーマのトークンから（dark: 400 / 320、kiseki はフォントが広いので 430 / 320）
    from gui.design_tokens import get_tokens

    _t = get_tokens(getattr(win, "ui_theme", "dark"))
    win.sidebar_scroll.setMinimumWidth(_t.sidebar_min)
    win._sidebar_min_width = _t.sidebar_min
    win._sidebar_user_width = _t.sidebar_default
    try:
        sizes = win.splitter.sizes()
        total = sum(sizes) if sizes else 0
        if total > 0:
            win.splitter.setSizes([_t.sidebar_default, max(200, total - _t.sidebar_default)])
    except Exception:
        pass


def _install_status_bar(win) -> None:
    sb = QStatusBar(win)
    sb.setSizeGripEnabled(False)
    win.setStatusBar(sb)
    win.status_label.setWordWrap(False)
    # 長い状態文がウィンドウの最小幅を押し広げないようにする（はみ出した分は切れる）
    win.status_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
    win.status_label.setMinimumWidth(80)
    win.status_label.setParent(sb)
    sb.addWidget(win.status_label, 1)
    win.progress_bar.setParent(sb)
    win.progress_bar.setFixedWidth(180)
    sb.addPermanentWidget(win.progress_bar)
    win.point_count_label.setParent(sb)
    sb.addPermanentWidget(win.point_count_label)
    win.modern_status_bar = sb


def _install_menu_bar(win) -> None:
    mb = win.menuBar()
    mb.clear()

    # --- ファイル ---
    m_file = mb.addMenu("ファイル(&F)")
    act_open = QAction("E57 ファイルを開く...", win)
    act_open.setShortcut(QKeySequence("Ctrl+O"))
    act_open.triggered.connect(win._browse_input_file)
    m_file.addAction(act_open)
    act_folder = QAction("フォルダ内の E57 を一括で開く...", win)
    act_folder.setShortcut(QKeySequence("Ctrl+Shift+O"))
    act_folder.triggered.connect(win._browse_input_folder)
    m_file.addAction(act_folder)
    m_recent = m_file.addMenu("最近使ったファイル")

    def _rebuild_recent():
        m_recent.clear()
        recent = load_recent_files()
        if not recent:
            a = m_recent.addAction("（履歴なし）")
            a.setEnabled(False)
            return
        for p in recent:
            a = m_recent.addAction(p)
            a.triggered.connect(lambda _=False, path=p: (win.input_edit.setText(path), win._on_input_file_changed(path)))

    m_recent.aboutToShow.connect(_rebuild_recent)
    m_file.addSeparator()
    act_save = QAction("E57 を保存...", win)
    act_save.setShortcut(QKeySequence("Ctrl+S"))
    act_save.triggered.connect(win._save_current_scans)
    m_file.addAction(act_save)
    m_file.addSeparator()
    act_quit = QAction("終了", win)
    act_quit.setShortcut(QKeySequence("Ctrl+Q"))
    act_quit.triggered.connect(win.close)
    m_file.addAction(act_quit)

    # --- 編集 ---（既存の QShortcut と二重登録にならないよう、ここではショートカットを設定せずラベルに併記）
    m_edit = mb.addMenu("編集(&E)")
    m_edit.addAction("元に戻す\tCtrl+Z", win._undo_action)
    m_edit.addAction("やり直し\tCtrl+Y", win._redo_action)
    m_edit.addSeparator()
    m_edit.addAction("選択を反転\tI", win._invert_selection)
    m_edit.addAction("選択を解除\tEsc", win._clear_selection)
    m_edit.addAction("計測した座標をコピー\tCtrl+C (計測中)", lambda: win.viewer.copy_measure_to_clipboard())
    m_edit.addAction("選択点をノイズへ移動\tDel", win._delete_selected_points)
    m_edit.addSeparator()
    m_edit.addAction("全ノイズを通常レイヤーへ復元", win._restore_all_noise)

    # --- 表示 ---
    m_view = mb.addMenu("表示(&V)")
    m_view.addAction("全体フィット\tF", lambda: win.viewer.fit_to_screen())
    m_view.addAction("透視 / 正射影 切替\tP", lambda: win.viewer.toggle_projection())
    m_preset = m_view.addMenu("視点")
    for label, attr in (("等角", "btn_view_iso"), ("上面 (XY)", "btn_view_top"), ("正面 (XZ)", "btn_view_front"), ("側面 (YZ)", "btn_view_side")):
        b = getattr(win, attr, None)
        if b is not None:
            m_preset.addAction(label, b.click)
    m_view.addSeparator()
    m_view.addAction("左パネルの表示 / 格納", lambda: win.btn_toggle_sidebar.click())
    m_view.addAction("3D 画面を撮影...", win._on_capture_screenshot)

    # --- ヘルプ ---
    m_help = mb.addMenu("ヘルプ(&H)")
    m_help.addAction("使い方マニュアルを開く", lambda: _open_manual(win))
    m_help.addAction("ショートカット一覧...", lambda: _show_shortcut_sheet(win))
    m_help.addAction("システム履歴ログフォルダを開く", open_log_folder)
    m_help.addSeparator()
    m_help.addAction("従来UIに戻す（次回起動から）", lambda: _switch_to_classic(win))
    m_help.addAction("バージョン情報", lambda: QMessageBox.information(
        win, "バージョン情報",
        "E57 点群ノイズ処理・断面スライサー ツール\n新UI（2026-10 デザイン改修）\n\n"
        "従来UIに戻すには: ヘルプ → 従来UIに戻す、または起動引数 --classic-ui、環境変数 E57_UI_MODE=classic",
    ))


def _open_manual(win) -> None:
    app_dir = get_app_dir()
    candidates = []
    for root in (app_dir, os.getcwd()):
        try:
            for f in os.listdir(root):
                if f.endswith(".html") and "マニュアル" in f:
                    candidates.append(os.path.join(root, f))
        except Exception:
            pass
    if not candidates:
        QMessageBox.information(win, "マニュアル", "マニュアル（ノイズフィルター設定ガイド …マニュアル.html）が見つかりませんでした。")
        return
    QDesktopServices.openUrl(QUrl.fromLocalFile(candidates[0]))


def _shortcut_sheet_text(win) -> str:
    settings = getattr(win, "settings_group", None)
    if settings is not None:
        for lbl in settings.findChildren(QLabel):
            if "Ctrl+Z" in lbl.text() and len(lbl.text()) > 80:
                return lbl.text()
    return (
        "Ctrl+O 開く / Ctrl+S 保存 / Ctrl+Z 元に戻す / Ctrl+Y やり直し\n"
        "V 回転⇄パン / H パン / B ブラシ / R 矩形 / C 円形 / L 多角形 / M 計測 / U 2点壁 / N ノイズ専用\n"
        "Del ノイズへ移動 / I 選択反転 / Esc 解除 / F フィット / P 投影切替 / Shift+B 断面枠の表示切替 / [ ] ブラシ径\n"
        "計測中: クリックで座標表示 / 2点目で距離 / Ctrl+C 座標をコピー / 右クリック クリア"
    )


def _show_shortcut_sheet(win) -> None:
    dlg = QDialog(win)
    dlg.setWindowTitle("ショートカット一覧")
    dlg.resize(560, 480)
    lay = QVBoxLayout(dlg)
    txt = QTextEdit(dlg)
    txt.setReadOnly(True)
    txt.setPlainText(strip_emoji(_shortcut_sheet_text(win)))
    lay.addWidget(txt)
    bb = QDialogButtonBox(QDialogButtonBox.Close, dlg)
    bb.rejected.connect(dlg.reject)
    bb.accepted.connect(dlg.accept)
    lay.addWidget(bb)
    dlg.exec()


def _switch_to_classic(win) -> None:
    ret = QMessageBox.question(
        win, "従来UIに戻す",
        "次回起動から改修前の従来UIで起動します。よろしいですか？\n（設定 → システム → UIスタイル でいつでも切り替えられます）",
        QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
    )
    if ret == QMessageBox.Yes:
        set_saved_ui_mode(UI_MODE_CLASSIC)
        win.show_toast("次回起動から従来UIになります")


def _convert_bars_to_flow(win) -> None:
    nav_bar = getattr(win, "_nav_bar_widget", None)
    if nav_bar is not None:
        nav_bar.setMinimumHeight(40)
        nav_bar.setMaximumHeight(16777215)
        replace_layout_with_flow(nav_bar, h_spacing=4, v_spacing=4)
        nav_bar.layout().setContentsMargins(8, 4, 8, 4)
    bar = getattr(win, "bottom_overlay_bar", None)
    if bar is not None:
        bar.setMinimumHeight(36)
        bar.setMaximumHeight(16777215)
        replace_layout_with_flow(bar, h_spacing=6, v_spacing=4)
        bar.layout().setContentsMargins(10, 4, 10, 4)


def _install_behaviors(win) -> None:
    """トーストの種類付け・選択バッジの状態・最近使ったファイル・空状態カードの表示制御・モーション"""
    from gui.design_tokens import get_tokens
    from gui.motion import (
        bounce, cascade_in, expand_animated, fade_in, fade_out_then_hide, fade_slide_in, focus_pulse,
        install_button_motion, install_number_ticker, set_motion_enabled, shake,
    )
    from gui.ui_mode import get_saved_motion_enabled

    # モーション ON/OFF（設定「アニメーション」）と押下フィードバック（アプリ全体の QAbstractButton）
    set_motion_enabled(get_saved_motion_enabled())
    install_button_motion(QApplication.instance())

    def pulse_color() -> str:
        # テーマはその場で切り替わることがあるので、使う瞬間のトークンから色を決める
        t = get_tokens(getattr(win, "ui_theme", "dark"))
        return t.accent_base if t.is_light else t.accent_soft

    # トースト: 内容から種類を判定し、:kind プロパティで色分け。絵文字は除去。
    #   出現 = フェード＋スライド（オーバーシュート）、警告 = エラーシェイク、退出 = フェード
    orig_toast = win.show_toast
    try:
        win._toast_timer.timeout.disconnect()
    except Exception:
        pass
    win._toast_timer.timeout.connect(lambda: fade_out_then_hide(win.toast_label))

    def classify(msg: str) -> str:
        m = msg or ""
        if any(k in m for k in ("エラー", "失敗", "⚠", "中止", "破棄", "見つかりません", "できません")):
            return "warning"
        if any(k in m for k in ("完了", "保存しました", "✅", "復元", "除去しました", "移動しました", "消去", "合格")):
            return "success"
        return "info"

    def show_toast(message: str, duration_ms: int = 2600):
        kind = classify(message)
        win.toast_label.setProperty("kind", kind)
        _repolish(win.toast_label)
        was_visible = win.toast_label.isVisible()
        orig_toast(strip_emoji(message), duration_ms)
        if kind == "warning":
            if not was_visible:
                fade_in(win.toast_label)
            shake(win.toast_label)
        elif not was_visible:
            fade_slide_in(win.toast_label, overshoot=True)

    win.show_toast = show_toast  # type: ignore[assignment]

    # 選択バッジ: 選択中は active プロパティで強調。選択が始まった瞬間に 1 回だけ跳ねる
    orig_update_bar = win._update_floating_selection_bar

    def update_bar(count: int):
        badge = win.floating_sel_badge
        was_active = bool(badge.property("active"))
        orig_update_bar(count)
        badge.setProperty("active", bool(count > 0))
        _repolish(badge)
        if count > 0 and not was_active:
            bounce(badge)

    win._update_floating_selection_bar = update_bar  # type: ignore[assignment]

    # 点数ピル: 数値ティッカー（前の値から新しい値へ数え上げ。桁幅固定）
    if getattr(win, "point_count_label", None) is not None:
        install_number_ticker(win.point_count_label)

    # 最近使ったファイル: 読み込み要求時に記録
    orig_load = win._load_file_to_viewer

    def load_with_recent(path: str, *args, **kwargs):
        result = orig_load(path, *args, **kwargs)
        if result is not False:
            add_recent_file(path)
            card = getattr(win, "empty_state_card", None)
            if card is not None:
                card.refresh_recent()
        return result

    win._load_file_to_viewer = load_with_recent  # type: ignore[assignment]

    # 空状態カード: 最初の点群が入ったらフェードで退場し、次に押す「実行」ボタンをフォーカスパルスで示す
    card = EmptyStateCard(win)
    win.empty_state_card = card
    has_points = win.viewer.points_raw is not None and len(win.viewer.points_raw) > 0
    card.setVisible(not has_points)
    exec_btn = getattr(win, "auto_exec_btn", None)

    def on_first_points(*_):
        if not card.isVisible():
            return
        fade_out_then_hide(card)
        if exec_btn is not None:
            QTimer.singleShot(350, lambda: focus_pulse(exec_btn, pulse_color()))

    win.viewer.bounds_ready.connect(on_first_points)

    # タブ切替: 表示される箱を上から順に時間差で出す（カスケード表示）
    orig_set_tab = win._set_sidebar_workflow_tab
    tab_state = {"key": None}
    alias = {"select": "edit", "auto": "filter"}

    def set_tab(tab_key: str, *args, **kwargs):
        norm = alias.get(tab_key, tab_key)
        changed = tab_state["key"] != norm
        tab_state["key"] = norm
        result = orig_set_tab(tab_key, *args, **kwargs)
        if changed:
            groups = [getattr(win, n, None) for n in
                      ("auto_group", "tools_group", "layer_group", "section_group", "view_group", "settings_group")]
            cascade_in([g for g in groups if g is not None])
        return result

    win._set_sidebar_workflow_tab = set_tab  # type: ignore[assignment]

    # 折りたたみ箱: 展開はアコーディオン（高さ＋透明度）。畳むのは即時
    for grp in win.findChildren(QGroupBox):
        if grp.objectName() != "collapsible_group" or not hasattr(grp, "set_expanded"):
            continue
        _wrap_collapsible(grp, expand_animated)


def _wrap_collapsible(grp, expand_animated) -> None:
    orig_set_expanded = grp.set_expanded

    def set_expanded(expanded: bool, _g=grp, _orig=orig_set_expanded):
        was = bool(_g.is_expanded())
        _orig(expanded)
        if expanded and not was and _g.isVisible():
            expand_animated(_g.content_widget)

    grp.set_expanded = set_expanded  # type: ignore[assignment]


# ----------------------------------------------------------------------------- 公開 API
def on_theme_changed(win, theme: str) -> None:
    """テーマのその場切替の後処理: サイドバー幅の既定値をテーマのトークンに合わせ、全ウィジェットを再描画する"""
    from gui.design_tokens import get_tokens

    t = get_tokens(theme)
    try:
        win.sidebar_scroll.setMinimumWidth(t.sidebar_min)
        win._sidebar_min_width = t.sidebar_min
        win._sidebar_user_width = max(getattr(win, "_sidebar_user_width", 0) or 0, t.sidebar_default)
        win._lock_sidebar_width()
    except Exception as exc:
        log_warning(f"テーマ切替: サイドバー幅の更新に失敗しました: {exc}")
    for w in win.findChildren(QWidget):
        _repolish(w)
    win.update()


def apply_modern_ui(win) -> None:
    """MainWindow に新UIを後付けする（UI モードが modern のときに __init__ の最後で呼ばれる）"""
    _neutralize_inline_styles(win)
    _assign_roles_and_names(win)
    _strip_all_emoji(win)
    _hide_duplicate_controls(win)
    _reorganize_sidebar(win)
    _install_status_bar(win)
    _install_menu_bar(win)
    _convert_bars_to_flow(win)
    _install_behaviors(win)

    # 既定タブは「①自動」（初めての人が最初に行う作業）
    try:
        win._set_sidebar_workflow_tab("filter")
    except Exception as exc:
        log_warning(f"新UI: 既定タブの設定に失敗しました: {exc}")

    # プロパティ変更を反映
    for w in win.findChildren(QWidget):
        _repolish(w)


def restyle_progress_dialog(dlg) -> None:
    """ModernProgressDialog を新UIのトークン QSS で描く（インライン QSS を外し、名前を付ける）"""
    for w in dlg.findChildren(QWidget):
        if w.styleSheet():
            w.setStyleSheet("")
    names = {
        "label_tag": "progressTag",
        "label_percent": "progressPercent",
        "label_title": "progressTitle",
        "label_sub": "progressSub",
        "btn_cancel": "progressCancel",
        "bar": "progressBarModern",
    }
    for attr, name in names.items():
        w = getattr(dlg, attr, None)
        if w is not None:
            w.setObjectName(name)
    tag = getattr(dlg, "label_tag", None)
    if tag is not None:
        tag.setText("処理中")
    for w in dlg.findChildren(QWidget):
        _repolish(w)
