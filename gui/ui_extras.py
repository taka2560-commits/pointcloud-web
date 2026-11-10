# -*- coding: utf-8 -*-
"""
新UIの操作まわりの追加機能をまとめて組み込む（gui/modern_ui.py の apply_modern_ui から呼ばれる）

- コマンドランチャー（Ctrl+K、上のバーの「機能を検索」）      … gui/launcher.py
- 右クリックのクイックパレット（従来のメニューは「その他のメニュー」から）… gui/quick_palette.py
- パネルの小窓化（見出しの「別窓」、左パネル格納時のドック）      … gui/popout_panels.py
- モード別の操作ガイド（3D 画面の右上）                        … gui/mode_hints.py
- 操作説明書（設定の「説明書」タブ、F1 の別ウィンドウ）          … gui/manual.py
どれも既存のボタンやメソッドを呼ぶだけで、処理そのものは持たない。従来UIには入れない。
"""

import random
from typing import Dict, List, Optional

from PySide6.QtCore import QObject, QPoint, Qt
from PySide6.QtGui import QAction, QKeySequence, QShortcut
from PySide6.QtWidgets import QLabel, QPushButton, QTabWidget

from core.logger import log_warning
from gui.commands import GROUP_HELP, GROUP_VIEW, Command, build_commands, commands_by_id
from gui.hints import TIPS, HintEngine, TipsDialog
from gui.launcher import CommandLauncher
from gui.manual import ManualTab, ManualWindow
from gui.mode_hints import ModeHintOverlay
from gui.popout_panels import PANELS, PanelPopouts
from gui.quick_palette import QuickPalette


class UiExtras(QObject):
    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self._manual_window: Optional[ManualWindow] = None
        self._shortcuts: List[QShortcut] = []
        self.manual_tab: Optional[ManualTab] = None
        self.nav_launcher_btn: Optional[QPushButton] = None

        self.popouts = PanelPopouts(win)
        self.launcher = CommandLauncher(win, self.commands)
        self.palette = QuickPalette(
            win,
            lambda: commands_by_id(self.commands()),
            open_launcher=lambda: self.launcher.open_launcher(),
            open_legacy_menu=self._open_legacy_menu,
        )
        self.hints = ModeHintOverlay(win)
        self.hint_engine = HintEngine(win, self.run_command)
        self._tips_dialog: Optional[TipsDialog] = None
        self.dock = self.popouts.install_dock(lambda: self.launcher.open_launcher())

        self._install_shortcuts()
        self._install_nav_button()
        self._rewire_context_menu()
        self._install_manual_tab()
        self._install_menu_entries()
        self._install_startup_tip()

    # --- コマンド表（基本 + この追加機能のぶん） ---
    def commands(self) -> List[Command]:
        cmds = build_commands(self.win)
        for key, _attr, _short, title in PANELS:
            if self.popouts.group(key) is None:
                continue
            cmds.append(Command(
                f"panel.{key}", GROUP_VIEW, f"{title} のパネルを小窓で開く / 戻す",
                "左パネルのグループを小窓として浮かせる。閉じると元の場所へ戻る", "", "こまど べつまど ぽっぷあっぷ うかせる",
                lambda w, k=key: self.popouts.toggle(k),
            ))
        cmds.append(Command(
            "panel.restore_all", GROUP_VIEW, "小窓をすべて左パネルへ戻す", "", "", "こまど もどす",
            lambda w: self.popouts.restore_all(),
            lambda w: None if self.popouts.popped_keys() else "開いている小窓がありません",
        ))
        cmds.append(Command(
            "help.manual", GROUP_HELP, "操作説明書を開く", "使い方を作業順にまとめた説明書を別ウィンドウで開く", "F1",
            "せつめいしょ まにゅある つかいかた へるぷ help", lambda w: self.open_manual(),
        ))
        cmds.append(Command(
            "help.keys", GROUP_HELP, "ショートカット一覧を見る", "キーボードとマウスの操作の一覧", "", "しょーとかっと きー いちらん",
            lambda w: self.open_manual("keys"),
        ))
        cmds.append(Command(
            "help.mode_hints", GROUP_HELP, "操作ガイドの表示 入/切", "ツールを選んだときに 3D 画面の右上へ出る案内", "", "がいど あんない ひんと",
            lambda w: self.set_hints_enabled(not self.hints.is_enabled()),
        ))
        cmds.append(Command(
            "help.tips", GROUP_HELP, "ヒント集を開く", "知っていると便利な操作を順に見る。「試す」でその機能が動く", "", "ひんと こつ ちっぷす tips",
            lambda w: self.open_tips(),
        ))
        cmds.append(Command(
            "help.hints", GROUP_HELP, "状況に応じたヒントの表示 入/切", "次にやるとよいことを 3D 画面の右上に出すカード", "", "ひんと つぎ あんない",
            lambda w: self.set_tips_enabled(not self.hint_engine.is_enabled()),
        ))
        return cmds

    def run_command(self, cmd_id: str) -> bool:
        cmd = commands_by_id(self.commands()).get(cmd_id)
        if cmd is None:
            return False
        return cmd.execute(self.win)

    # --- 組み込み ---
    def _install_shortcuts(self) -> None:
        sc = QShortcut(QKeySequence("Ctrl+K"), self.win)
        sc.activated.connect(self.launcher.toggle)
        self._shortcuts.append(sc)
        f1 = QShortcut(QKeySequence("F1"), self.win)
        f1.activated.connect(lambda: self.open_manual())
        self._shortcuts.append(f1)

    def _install_nav_button(self) -> None:
        """「機能を検索」の入口をメニューバーの右端に置く（上部バーはツールで一杯なので段を増やさない）"""
        mb = self.win.menuBar()
        if mb is None:
            return
        btn = QPushButton("機能を検索 (Ctrl+K)", mb)
        btn.setObjectName("menu_launcher_button")
        btn.setToolTip("やりたいことの名前を打って実行する（例: だんめん、けいそく、ほぞん）")
        btn.clicked.connect(self.launcher.toggle)
        mb.setCornerWidget(btn, Qt.TopRightCorner)
        btn.show()
        self.nav_launcher_btn = btn

    def _rewire_context_menu(self) -> None:
        sig = self.win.viewer.context_menu_requested
        try:
            sig.disconnect(self.win._show_viewer_context_menu)
        except Exception:
            try:
                sig.disconnect()
            except Exception:
                pass
        sig.connect(self.palette.show_at)

    def _open_legacy_menu(self, global_pos: QPoint) -> None:
        self.win._show_viewer_context_menu(global_pos)

    def _install_manual_tab(self) -> None:
        grp = getattr(self.win, "settings_group", None)
        tabs = grp.findChild(QTabWidget) if grp is not None else None
        if tabs is None:
            return
        tab = ManualTab(
            self.win, self.commands, hints_enabled=self.hints.is_enabled(), tips_enabled=self.hint_engine.is_enabled()
        )
        tab.browser.command_requested.connect(self.run_command)
        tab.open_window_requested.connect(lambda: self.open_manual(tab.browser.current_topic_id()))
        tab.hints_toggled.connect(self.hints.set_enabled)
        tab.tips_toggled.connect(self.hint_engine.set_enabled)
        tab.open_tips_requested.connect(lambda: self.open_tips())
        tab.reset_hints_requested.connect(self._reset_hints)
        tabs.addTab(tab, "説明書")
        self.manual_tab = tab
        self.manual_tabs = tabs

    def _install_menu_entries(self) -> None:
        mb = self.win.menuBar()
        if mb is None:
            return
        for act in mb.actions():
            menu = act.menu()
            if menu is None:
                continue
            title = act.text()
            first = menu.actions()[0] if menu.actions() else None
            if title.startswith("ヘルプ"):
                a = QAction("操作説明書\tF1", self.win)
                a.triggered.connect(lambda: self.open_manual())
                menu.insertAction(first, a)
                t = QAction("ヒント集...", self.win)
                t.triggered.connect(lambda _=False: self.open_tips())
                menu.insertAction(first, t)
                for sub in menu.actions():
                    if sub.text().startswith("使い方マニュアル"):
                        sub.setText("ノイズフィルター設定ガイドを開く")
            elif title.startswith("表示"):
                a = QAction("機能を検索...\tCtrl+K", self.win)
                a.triggered.connect(lambda _=False: self.launcher.open_launcher())
                menu.insertAction(first, a)

    # --- 説明書 ---
    def open_manual(self, topic_id: Optional[str] = None) -> ManualWindow:
        if self._manual_window is None:
            self._manual_window = ManualWindow(self.win, self.commands)
            self._manual_window.browser.command_requested.connect(self.run_command)
        w = self._manual_window
        if topic_id:
            w.browser.show_topic(topic_id)
        w.show()
        w.raise_()
        w.activateWindow()
        return w

    def manual_window(self) -> Optional[ManualWindow]:
        return self._manual_window

    def set_hints_enabled(self, enabled: bool) -> None:
        self.hints.set_enabled(enabled)
        if self.manual_tab is not None:
            self.manual_tab.chk_hints.blockSignals(True)
            self.manual_tab.chk_hints.setChecked(bool(enabled))
            self.manual_tab.chk_hints.blockSignals(False)
        if hasattr(self.win, "show_toast"):
            self.win.show_toast(f"操作ガイドの表示: {'ON' if enabled else 'OFF'}")

    # --- ヒント ---
    def _command_title(self, cmd_id: str) -> str:
        cmd = commands_by_id(self.commands()).get(cmd_id)
        return cmd.title if cmd is not None else cmd_id

    def open_tips(self, tip_id: Optional[str] = None) -> TipsDialog:
        if self._tips_dialog is None:
            self._tips_dialog = TipsDialog(self.win, self.run_command, self.open_manual, self._command_title)
        d = self._tips_dialog
        if tip_id:
            d.show_tip_id(tip_id)
        d.show()
        d.raise_()
        d.activateWindow()
        return d

    def tips_dialog(self) -> Optional[TipsDialog]:
        return self._tips_dialog

    def set_tips_enabled(self, enabled: bool) -> None:
        self.hint_engine.set_enabled(enabled)
        if self.manual_tab is not None:
            self.manual_tab.chk_tips.blockSignals(True)
            self.manual_tab.chk_tips.setChecked(bool(enabled))
            self.manual_tab.chk_tips.blockSignals(False)
        if hasattr(self.win, "show_toast"):
            self.win.show_toast(f"状況に応じたヒントの表示: {'ON' if enabled else 'OFF'}")

    def _reset_hints(self) -> None:
        self.hint_engine.reset_hidden()
        if hasattr(self.win, "show_toast"):
            self.win.show_toast("閉じたヒントを、もう一度表示するようにしました")

    def _install_startup_tip(self) -> None:
        """起動直後の画面（空状態カード）に、ヒント集から 1 つ出す"""
        card = getattr(self.win, "empty_state_card", None)
        lay = card.layout() if card is not None else None
        if lay is None:
            return
        tip = random.choice(TIPS)
        lbl = QLabel(f"ヒント: {tip.text}", card)
        lbl.setObjectName("empty_state_tip")
        lbl.setWordWrap(True)
        lay.addWidget(lbl)
        btn = QPushButton("ほかのヒントを見る", card)
        btn.setProperty("role", "ghost")
        btn.clicked.connect(lambda _=False: self.open_tips())
        lay.addWidget(btn)
        self.startup_tip_label = lbl
        try:
            card.adjustSize()
            card._recenter()
        except Exception:
            pass

    def refresh_theme(self) -> None:
        if self.manual_tab is not None:
            self.manual_tab.browser.refresh_theme()
        if self._manual_window is not None:
            self._manual_window.browser.refresh_theme()


def install_ui_extras(win) -> Optional[UiExtras]:
    """新UIに追加機能を組み込む。失敗しても本体は動くよう、警告を残して None を返す"""
    try:
        extras = UiExtras(win)
        win.ui_extras = extras
        return extras
    except Exception as exc:  # 追加機能の不具合で起動できなくならないようにする
        log_warning(f"新UI: 追加機能（ランチャー・パレット・小窓・説明書）の組み込みに失敗しました: {exc}")
        return None
