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
        self.btn_open = btn_open
        btn_folder = QPushButton("フォルダを一括で開く")
        btn_folder.setProperty("role", "outline")
        btn_folder.clicked.connect(win._browse_input_folder)
        row.addWidget(btn_folder)
        self.btn_folder = btn_folder
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

    # 役割の見直し（「役割」と「状態」で選ぶ。一覧は gui/button_kit.py の ROLE_GUIDE）
    #   下部バー: 選んだ点への主操作は「ノイズ削除」。レイヤー移動は補助にして、塗りつぶしを 1 つにする
    #   全て展開 / 全て折りたたみ: いちばん軽い操作なので文字だけ
    #   断面のピック: オン/オフの切り替え
    for name, role in (
        ("floating_move_btn", "outline"),
        ("btn_expand_all", "ghost"),
        ("btn_collapse_all", "ghost"),
        ("btn_sec_wall_pick", "segment"),
        ("btn_poly_confirm", "outline"),
    ):
        b = g(name)
        if b is not None:
            b.setProperty("role", role)
    for b in win.findChildren(QPushButton):
        if b.property("role") or b.property("tone") or b.objectName():
            continue
        text = b.text()
        if "ログフォルダ" in text:
            b.setProperty("role", "soft")  # 使う回数の少ない操作は淡い塗り
    # 断面の厚みの候補（0.1m / 0.5m / 1m …）は、その場で値を選ぶ小さなボタンなのでピル型
    sec = g("section_group")
    if sec is not None:
        for b in sec.findChildren(QPushButton):
            if re.fullmatch(r"\d+(\.\d+)?m", b.text().strip()):
                b.setProperty("role", "pill")

    # 断面枠の切替: 幅 52px に「切替(Shift+B)」は収まらず文字が切れる。短くしてショートカットはツールチップへ
    b = g("btn_toggle_sec_guides")
    if b is not None:
        b.setText("切替")

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
    # 1 段目: 保存対象、2 段目: 保存ボタン（全幅。作業の最後の主導線をはっきり見せる）
    vl = QVBoxLayout(save_group.content_widget)
    vl.setContentsMargins(6, 4, 6, 6)
    vl.setSpacing(6)
    sl = QHBoxLayout()
    sl.setSpacing(6)
    vl.addLayout(sl)
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
    try:
        from gui.button_kit import SplitButton, tokens_for
        from gui.status_feedback import save_as

        split = SplitButton(
            win.save_manual_btn,
            [
                ("E57 を保存\tCtrl+S", lambda: win._save_current_scans()),
                ("保存先を選んで保存...", lambda: save_as(win)),
                None,
                ("作業の状態を保存 (.npz)...", lambda: win._on_save_layer_project()),
            ],
            lambda: tokens_for(win),
            arrow_tip="保存に関連する操作（保存先を選んで保存・作業の状態を保存）",
            parent=save_group.content_widget,
        )
        vl.addWidget(split)
        win.save_split = split
    except Exception as exc:  # 分割ボタンを作れなくても、保存ボタンそのものは置く
        log_warning(f"新UI: 保存の分割ボタンを作れませんでした: {exc}")
        vl.addWidget(win.save_manual_btn)
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
        bounce, cancel_fade_out, cascade_in, expand_animated, fade_in, fade_out_then_hide, fade_slide_in, focus_pulse,
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
    #   印（成功・注意・失敗・お知らせ）と枠の色で種類を示す。色だけに頼らない
    #   点を動かした直後の通知には「元に戻す」を付け、押す時間が取れるように長めに出す
    from gui.status_feedback import ACTION_TOAST_MS, UNDO_ACTION_TEXT, classify_message, is_edit_report

    orig_toast = win.show_toast
    toast_host = getattr(win, "toast_host", None)
    toast_box = toast_host if toast_host is not None else win.toast_label  # 出し入れ・動きを付ける相手
    try:
        win._toast_timer.timeout.disconnect()
    except Exception:
        pass
    toast_hold = {"after_modal": False}

    def on_toast_timeout() -> None:
        # 次の一手のボタン（「元に戻す」）が付いた通知は、モーダルのダイアログが開いているあいだは消さない。
        # 自動ノイズ処理は通知の直後に完了ダイアログを出すので、裏で時間が過ぎると押す前に消えてしまう。
        # 閉じたあと、あらためて押す時間を取ってから消す
        has_action = toast_host is not None and toast_host.has_action()
        if has_action and QApplication.activeModalWidget() is not None:
            toast_hold["after_modal"] = True
            win._toast_timer.start(300)
            return
        if has_action and toast_hold["after_modal"]:
            toast_hold["after_modal"] = False
            win._toast_timer.start(ACTION_TOAST_MS)
            return
        toast_hold["after_modal"] = False
        fade_out_then_hide(toast_box)

    win._toast_timer.timeout.connect(on_toast_timeout)

    # 編集のたびに数える（「前の通知のあとで点が動いたか」を知るため）
    edit_state = {"edits": 0, "seen": 0}
    orig_push_undo = win._push_layer_undo_state

    def expire_edit_mark(mark: int) -> None:
        if edit_state["seen"] < mark:
            edit_state["seen"] = mark

    def push_undo(*args, **kwargs):
        edit_state["edits"] += 1
        # 「編集の直後」として扱うのは、この編集と同じ処理の中で出た通知だけ。
        # 通知を出さない編集（断面外のクロップなど）の印が残ると、あとの無関係な通知に「元に戻す」が付いてしまうので、
        # いまの処理が終わったら（イベントループへ戻ったら）印を下ろす
        QTimer.singleShot(0, win, lambda m=edit_state["edits"]: expire_edit_mark(m))
        return orig_push_undo(*args, **kwargs)

    win._push_layer_undo_state = push_undo  # type: ignore[assignment]

    def can_undo() -> bool:
        try:
            return bool(win.layer_history.can_undo())
        except Exception:
            return False

    def show_toast(message: str, duration_ms: int = 2600, action=None):
        cancel_fade_out(toast_box)  # 前の通知の退場中でも、新しい通知を巻き込んで消さない
        toast_hold["after_modal"] = False
        text = strip_emoji(message)
        kind = classify_message(message)
        fresh_edit = edit_state["edits"] != edit_state["seen"]
        edit_state["seen"] = edit_state["edits"]
        if action is None and fresh_edit and is_edit_report(text) and can_undo():
            action = (UNDO_ACTION_TEXT, lambda: win._undo_action(), "直前の操作を取り消します (Ctrl+Z)")
        if action is not None:
            duration_ms = max(int(duration_ms), ACTION_TOAST_MS)
        win.toast_label.setProperty("kind", kind)
        _repolish(win.toast_label)
        if toast_host is not None:
            toast_host.set_kind(kind)
            if action is not None:
                toast_host.set_action(action[0], action[1], action[2] if len(action) > 2 else "")
            else:
                toast_host.clear_action()
        was_visible = toast_box.isVisible()
        orig_toast(text, duration_ms)
        if toast_host is not None:
            toast_host.setVisible(True)
            toast_host.place()
        if kind in ("warning", "error"):
            if not was_visible:
                fade_in(toast_box)
            shake(toast_box)
        elif not was_visible:
            fade_slide_in(toast_box, overshoot=True)

    win.show_toast = show_toast  # type: ignore[assignment]

    # 未保存の変更: タイトルの「*」に加えて、ステータスバーに「未保存の変更あり」を出す（押すと保存）
    #   保存が成功した直後（保存ボタンが処理中から戻ってすぐ）なら、ボタンを「保存済み」にする
    orig_set_dirty = win._set_dirty

    def set_dirty(dirty: bool) -> None:
        orig_set_dirty(dirty)
        pill = getattr(win, "unsaved_pill", None)
        if pill is not None:
            pill.setVisible(bool(win._dirty))
        kit = getattr(win, "button_kit", None)
        binding = kit.busy.get("save_manual_btn") if kit is not None else None
        if binding is None:
            return
        if not dirty and binding.just_finished():
            binding.flash_done("保存済み")
        elif dirty:
            binding.clear_done()  # 「保存済み」と「未保存の変更あり」を同時に出さない

    win._set_dirty = set_dirty  # type: ignore[assignment]

    # 選択バッジ: 選択中は active プロパティで強調。選択が始まった瞬間に 1 回だけ跳ねる
    orig_update_bar = win._update_floating_selection_bar

    #   未選択のときは「なぜ下のボタンが押せないか」を短く書く（長い文だとバーが 2 段になる）
    idle_badge_text = "未選択: 点を囲むと右の操作が使えます"

    def update_bar(count: int):
        badge = win.floating_sel_badge
        was_active = bool(badge.property("active"))
        orig_update_bar(count)
        if count <= 0:
            badge.setText(idle_badge_text)
        badge.setProperty("active", bool(count > 0))
        _repolish(badge)
        if count > 0 and not was_active:
            bounce(badge)

    win._update_floating_selection_bar = update_bar  # type: ignore[assignment]
    if not bool(win.floating_sel_badge.property("active")):
        win.floating_sel_badge.setText(idle_badge_text)

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
            QTimer.singleShot(350, exec_btn, lambda: focus_pulse(exec_btn, pulse_color()))  # ボタンが消えたら発火しない

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
    extras = getattr(win, "ui_extras", None)
    if extras is not None:
        extras.refresh_theme()  # 操作説明書の本文の色をテーマに合わせ直す
    for attr in ("button_kit", "status_feedback"):
        part = getattr(win, attr, None)
        if part is not None:
            part.refresh_theme()  # アイコン・印をテーマの色で描き直す
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
    # 状態表示（通知の入れ物・未保存の印・入力の誤り）。下部バーを流し込みレイアウトへ替える前に入れる
    from gui.status_feedback import install_status_feedback

    install_status_feedback(win)
    _convert_bars_to_flow(win)
    _install_behaviors(win)

    # ボタンの役割と状態（アイコン+文字・処理中/完了・無効の理由・キーボードの現在位置・オン/オフの印）
    kit = None
    try:
        from gui.button_kit import apply_button_kit, bind_busy

        kit = apply_button_kit(win)
        split = getattr(win, "save_split", None)
        if split is not None:
            kit.add_split(split)
            if "save_manual_btn" in kit.busy:
                kit.busy["save_manual_btn"].followers.append(split.arrow_btn)
        # 設定を保存: 押したら「保存済み」を少しのあいだ出す
        btn_set = getattr(win, "btn_save_settings", None)
        if btn_set is not None:
            set_binding = bind_busy(btn_set, "保存中…", kit.tokens, "保存済み")
            kit.busy["btn_save_settings"] = set_binding
            btn_set.clicked.connect(lambda _checked=False, b=set_binding: b.flash_done())
        card = getattr(win, "empty_state_card", None)
        if card is not None:
            kit.set_icon(card.btn_open, "open-file")
            kit.set_icon(card.btn_folder, "folder")
    except Exception as exc:  # 見た目の補助なので、失敗しても本体は動かす
        log_warning(f"新UI: ボタンの役割と状態の組み込みに失敗しました: {exc}")

    # 操作まわりの追加機能（コマンドランチャー・右クリックのパレット・パネルの小窓・操作ガイド・操作説明書）
    from gui.ui_extras import install_ui_extras

    extras = install_ui_extras(win)
    if kit is not None and extras is not None and extras.nav_launcher_btn is not None:
        kit.set_icon(extras.nav_launcher_btn, "search")

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
