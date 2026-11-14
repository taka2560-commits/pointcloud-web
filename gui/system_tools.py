# -*- coding: utf-8 -*-
"""
システムの道具（点群処理そのもの以外の裏方）: 作業用データの管理・書く前の空きの確認・版の表示

なぜ要るか（2026-11-13、利用者の PC の確認から）
    - 作業用データ（断面の読み直しと高精細表示のための、元ファイルの写し）は 5.4 億点で 1 つ約 17GB。
      デスクトップに 2 つあり、C ドライブの空きは約 20GB だった。アプリからは大きさが見えず、消せず、置き場所も変えられなかった。
    - 「全点で保存」は約 16GB 書くのに、始める前に空きを確かめていなかった。
    - 配布版を 1 日に 3 回作り直したが、画面にもログにも版が出ず、どの版で起きたことか区別できなかった。

ホットファイル（gui/main_window.py）から呼ぶのは次の 4 つだけ。中身はすべてここと core/ にある。
    install_system_tools(win)       … __init__ で 1 回。置き場所の設定を core へ渡し、設定 → システムに行を足す。
                                       長い処理の完了の知らせ（gui/task_notify.py）もここで取り付ける
    window_title_with_build()       … タイトルに版を添える
    confirm_save_free_space(...)    … 保存を始める前
    confirm_lod_build(win, scans)   … 高精細表示のデータを作る前

置き場所の設定は QSettings の workdata/dir（空なら元ファイルの横）。変えたその場で保存する。
「初期設定」ボタンでは戻さない（戻すと、すでに作ったデータが使われなくなるため）。
"""

import os
import sys
import threading
import time
from typing import Callable, List, Optional, Sequence

import numpy as np
from PySide6.QtCore import QEvent, QObject, QSettings, Qt, QTimer, Signal
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QDialog, QFileDialog, QFrame, QGridLayout, QHBoxLayout, QLabel, QMessageBox, QPushButton, QScrollArea, QSizePolicy,
    QVBoxLayout, QWidget,
)

from core.build_info import get_build_info
from core.logger import get_logs_dir, log_info, log_warning
from core.section_cache import cache_dir_for, get_default_cache_root, set_default_cache_root
from core.sysinfo import GB, can_write_dir, disk_status, drive_name, format_bytes
from core.workdata import (
    WorkData, check_free, estimate_save_bytes, find_workdata, remove_workdata, summarize, workdata_need_for_scans,
)
from gui.button_kit import set_button_icon, set_disabled_reason, tokens_for
from gui.gpu_select import install_gpu_row
from gui.task_notify import install_long_task_notify, make_notify_checkbox

_SETTINGS_ORG = "E57PointCloudTool"
_SETTINGS_APP = "Settings"
WORKDATA_DIR_KEY = "workdata/dir"
_RECENT_FILES_KEY = "recent_files"   # gui/modern_ui.py の RECENT_FILES_KEY と同じ（循環 import を避けて直接読む）

APP_TITLE = "E57 点群ノイズ処理ツール - 3Dエディタ"
APP_NAME = "E57 点群ノイズ処理・断面スライサー ツール"

# 空きが「少ない」と知らせる目安（ドライブの 10% 未満か、10GB 未満）
LOW_FREE_RATIO = 0.10
LOW_FREE_BYTES = 10 * GB


# ----------------------------------------------------------------------------- 版
def window_title_with_build(base: str = APP_TITLE) -> str:
    """ウィンドウのタイトルに版を添える（画面の写しだけで、どの版かが分かる）"""
    tag = get_build_info().title_tag()
    return f"{base}  ［{tag}］" if tag else base


def about_text() -> str:
    """バージョン情報に出す文（そのままコピーして、問い合わせに貼れる）"""
    info = get_build_info()
    try:
        import PySide6

        qt_ver = PySide6.__version__
    except Exception:
        qt_ver = "不明"
    root = get_default_cache_root()
    lines = [
        APP_NAME,
        "",
        f"版: {info.describe()}",
        f"起動のしかた: {'配布版 (EXE)' if info.frozen else 'ソースから起動'}",
        f"Python {sys.version.split()[0]} / PySide6 {qt_ver}",
        f"ログの場所: {get_logs_dir()}",
        f"作業用データの置き場所: {root if root else '元ファイルの横（標準）'}",
    ]
    return "\n".join(lines)


def show_about_dialog(win) -> None:
    dlg = QDialog(win)
    dlg.setObjectName("about_dialog")
    dlg.setWindowTitle("バージョン情報")
    lay = QVBoxLayout(dlg)
    lay.setContentsMargins(16, 14, 16, 12)
    lay.setSpacing(10)
    lbl = QLabel(about_text(), dlg)
    lbl.setObjectName("about_text")
    lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
    lbl.setWordWrap(True)
    lay.addWidget(lbl, 1)
    row = QHBoxLayout()
    row.setSpacing(6)
    btn_copy = QPushButton("コピー", dlg)
    btn_copy.setProperty("role", "outline")
    btn_copy.setToolTip("この内容をクリップボードへコピーします（問い合わせに貼れます）")

    def _copy():
        QGuiApplication.clipboard().setText(about_text())
        btn_copy.setText("コピーしました")
        QTimer.singleShot(1800, btn_copy, lambda: btn_copy.setText("コピー"))

    btn_copy.clicked.connect(lambda _=False: _copy())
    row.addWidget(btn_copy)
    row.addStretch(1)
    btn_close = QPushButton("閉じる", dlg)
    btn_close.setProperty("role", "ghost")
    btn_close.clicked.connect(dlg.accept)
    row.addWidget(btn_close)
    lay.addLayout(row)
    dlg.resize(560, 230)
    dlg.btn_copy = btn_copy  # type: ignore[attr-defined]
    win._about_dialog = dlg
    dlg.exec()


# ----------------------------------------------------------------------------- 置き場所の設定
def get_saved_workdata_dir() -> str:
    val = QSettings(_SETTINGS_ORG, _SETTINGS_APP).value(WORKDATA_DIR_KEY, "", type=str)
    return str(val or "").strip()


def set_saved_workdata_dir(path: str) -> None:
    QSettings(_SETTINGS_ORG, _SETTINGS_APP).setValue(WORKDATA_DIR_KEY, str(path or ""))


def apply_workdata_dir_setting() -> str:
    """
    保存してある置き場所を core へ渡す（起動時）。
    フォルダが見つからなくても設定は変えない（外付けドライブを外しているだけかもしれない）。
    """
    path = get_saved_workdata_dir()
    set_default_cache_root(path or None)
    if path and not os.path.isdir(path):
        log_warning(f"作業用データの置き場所が見つかりません（作業用データなしで動きます）: {path}")
    return path


# ----------------------------------------------------------------------------- 探す場所
def known_source_files(win) -> List[str]:
    """作業用データを探す手がかり: 今開いているファイル、入力欄・出力欄のパス、最近使ったファイル"""
    out: List[str] = []

    def add(p) -> None:
        p = str(p or "").strip()
        if p and all(os.path.normcase(p) != os.path.normcase(q) for q in out):
            out.append(p)

    for s in getattr(win, "current_scans", None) or []:
        add(getattr(s, "source_file", None))
    for attr in ("input_edit", "output_edit"):
        edit = getattr(win, attr, None)
        if edit is not None:
            try:
                add(edit.text())
            except Exception:
                pass
    try:
        val = QSettings(_SETTINGS_ORG, _SETTINGS_APP).value(_RECENT_FILES_KEY, [])
        if isinstance(val, str):
            val = [val]
        for p in val or []:
            add(p)
    except Exception:
        pass
    return out


def current_workdata_dirs(win) -> List[str]:
    """今開いているファイルの作業用データのフォルダ（正規化済み）"""
    out: List[str] = []
    for s in getattr(win, "current_scans", None) or []:
        src = getattr(s, "source_file", None)
        if src:
            key = os.path.normcase(os.path.abspath(cache_dir_for(src)))
            if key not in out:
                out.append(key)
    return out


def _is_low(disk) -> bool:
    return disk is not None and (disk[1] < LOW_FREE_BYTES or disk[1] < disk[0] * LOW_FREE_RATIO)


def _space_probe_path(win, entries: Sequence[WorkData]) -> str:
    """空きを見るドライブ: 置き場所の設定 → 今開いているファイル → いちばん大きい作業用データ → ホーム"""
    root = get_default_cache_root()
    if root:
        return root
    for s in getattr(win, "current_scans", None) or []:
        src = getattr(s, "source_file", None)
        if src:
            return os.path.dirname(os.path.abspath(src))
    if entries:
        return entries[0].directory
    return os.path.expanduser("~")


class WorkDataScanner(QObject):
    """
    作業用データを別スレッドで探す（最近使ったファイルが、つながっていないネットワークの先にあると、
    フォルダを調べるだけで数秒止まることがある。画面を止めない）。
    あとから頼んだ結果だけを渡す。待っている間に画面が閉じられたら、結果は捨てる。
    """
    finished = Signal(int, object)

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._seq = 0
        self._callback: Optional[Callable[[List[WorkData]], None]] = None
        self.finished.connect(self._deliver)

    def start(self, sources: Sequence[str], callback: Callable[[List[WorkData]], None]) -> None:
        self._seq += 1
        seq = self._seq
        self._callback = callback
        sources = list(sources)

        def work():
            try:
                entries = find_workdata(sources)
            except Exception:
                entries = []
            try:
                self.finished.emit(seq, entries)
            except RuntimeError:
                pass  # 画面が先に閉じられた

        threading.Thread(target=work, name="workdata-scan", daemon=True).start()

    def cancel(self) -> None:
        """頼んである結果を捨てる（その場で調べ直したあとに、古い結果で上書きしないため）"""
        self._seq += 1
        self._callback = None

    def _deliver(self, seq: int, entries) -> None:
        if seq != self._seq or self._callback is None:
            return
        cb, self._callback = self._callback, None
        cb(list(entries))


# ----------------------------------------------------------------------------- 確認の窓（テストで差し替えられる）
def _ask_low_space(parent, title: str, text: str, info: str, allow_proceed: bool) -> str:
    """空きが足りないときの確認。戻り値は manage（作業用データを整理する）/ proceed（このまま続ける）/ cancel のどれか"""
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Warning)
    box.setWindowTitle(title)
    box.setText(text)
    box.setInformativeText(info)
    btn_manage = box.addButton("作業用データを整理する", QMessageBox.ActionRole)
    btn_go = box.addButton("このまま保存する", QMessageBox.DestructiveRole) if allow_proceed else None
    btn_cancel = box.addButton("やめる", QMessageBox.RejectRole)
    box.setDefaultButton(btn_cancel)
    box.setEscapeButton(btn_cancel)
    box.exec()
    clicked = box.clickedButton()
    if clicked is btn_manage:
        return "manage"
    if btn_go is not None and clicked is btn_go:
        return "proceed"
    return "cancel"


def _ask_delete(parent, title: str, text: str, info: str) -> bool:
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Warning)
    box.setWindowTitle(title)
    box.setText(text)
    box.setInformativeText(info)
    btn_del = box.addButton("削除する", QMessageBox.DestructiveRole)
    btn_cancel = box.addButton("やめる", QMessageBox.RejectRole)
    box.setDefaultButton(btn_cancel)
    box.setEscapeButton(btn_cancel)
    box.exec()
    return box.clickedButton() is btn_del


class _Hooks:
    """確認の窓。テストは、ここの関数を差し替えて答えを決める"""
    ask_low_space = staticmethod(_ask_low_space)
    ask_delete = staticmethod(_ask_delete)


HOOKS = _Hooks()


# ----------------------------------------------------------------------------- 管理の窓
class _ElideLabel(QLabel):
    """長いパスやファイル名を、幅に合わせて途中を「…」にして出す（全文はツールチップ）"""

    def __init__(self, text: str = "", parent: Optional[QWidget] = None, mode=Qt.ElideMiddle):
        super().__init__(parent)
        self._full = ""
        self._mode = mode
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.setMinimumWidth(60)
        self.set_full_text(text)

    def set_full_text(self, text: str) -> None:
        self._full = str(text or "")
        self.setToolTip(self._full)
        self._apply()

    def full_text(self) -> str:
        return self._full

    def resizeEvent(self, event):  # type: ignore[override]
        super().resizeEvent(event)
        self._apply()

    def _apply(self) -> None:
        super().setText(self.fontMetrics().elidedText(self._full, self._mode, max(10, self.width() - 2)))


def _date_text(ts: float) -> str:
    if not ts:
        return "日付不明"
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))


def _dialog_qss(t) -> str:
    return f"""
QFrame#wd_card {{ background: {t.bg_surface}; border: 1px solid {t.border_subtle}; border-radius: 6px; }}
QLabel#wd_name, QLabel#wd_size {{ font-weight: bold; }}
QLabel#wd_muted {{ color: {t.text_secondary}; }}
QLabel#wd_chip {{ color: {t.text_primary}; background: {t.accent_tint}; border: 1px solid {t.border_strong};
                  border-radius: 8px; padding: 1px 7px; }}
QLabel#wd_badge {{ color: {t.badge_text}; background: {t.badge_bg}; border: 1px solid {t.badge_border};
                   border-radius: 8px; padding: 1px 7px; }}
QScrollArea#wd_scroll {{ border: none; }}
"""


class _EntryCard(QFrame):
    """作業用データ 1 つぶんの行"""

    def __init__(self, entry: WorkData, in_use: bool, tokens, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.entry = entry
        self.in_use = in_use
        self.setObjectName("wd_card")
        g = QGridLayout(self)
        g.setContentsMargins(10, 8, 10, 8)
        g.setHorizontalSpacing(10)
        g.setVerticalSpacing(3)

        self.lbl_name = _ElideLabel(entry.source_name, self)
        self.lbl_name.setObjectName("wd_name")
        g.addWidget(self.lbl_name, 0, 0)

        self.lbl_size = QLabel(format_bytes(entry.size_bytes), self)
        self.lbl_size.setObjectName("wd_size")
        self.lbl_size.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        g.addWidget(self.lbl_size, 0, 1)

        self.btn_delete = QPushButton("削除", self)
        self.btn_delete.setProperty("role", "danger")
        self.btn_delete.setToolTip("この作業用データを消して空きを作ります（元の E57 と保存した結果は変わりません）")
        set_button_icon(self.btn_delete, "trash", tokens)
        g.addWidget(self.btn_delete, 0, 2, 2, 1, Qt.AlignVCenter)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        self.lbl_detail = QLabel(f"{entry.contents_text}・{_date_text(entry.modified)} 作成", self)
        self.lbl_detail.setObjectName("wd_muted")
        row.addWidget(self.lbl_detail)
        self.state_text = "今開いているファイル" if in_use else entry.reason
        self.lbl_state = QLabel(self.state_text, self)
        self.lbl_state.setObjectName("wd_chip" if in_use else "wd_badge")
        self.lbl_state.setVisible(bool(self.state_text))
        row.addWidget(self.lbl_state)
        row.addStretch(1)
        g.addLayout(row, 1, 0, 1, 2)

        self.lbl_path = _ElideLabel(os.path.dirname(entry.directory), self)
        self.lbl_path.setObjectName("wd_muted")
        g.addWidget(self.lbl_path, 2, 0, 1, 3)
        g.setColumnStretch(0, 1)


class WorkDataDialog(QDialog):
    """作業用データの管理: 大きさを見る・消す・置き場所を変える"""

    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self.entries: List[WorkData] = []
        self.cards: List[_EntryCard] = []
        self._scanner = WorkDataScanner(self)
        self._tokens = tokens_for(win)
        self.setObjectName("workdata_dialog")
        self.setWindowTitle("作業用データの管理")
        self.setModal(True)
        self.setStyleSheet(_dialog_qss(self._tokens))

        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 14, 16, 12)
        lay.setSpacing(8)

        intro = QLabel(
            "断面の読み直しと高精細表示を速くするための、元ファイルの写しです。"
            "削除しても、元の E57 と保存した結果は変わりません。必要になれば作り直されます。", self,
        )
        intro.setObjectName("wd_muted")
        intro.setWordWrap(True)
        lay.addWidget(intro)

        head = QHBoxLayout()
        head.setSpacing(8)
        self.lbl_total = QLabel("調べています…", self)
        self.lbl_total.setObjectName("wd_name")
        head.addWidget(self.lbl_total)
        head.addStretch(1)
        self.lbl_drive = QLabel("", self)
        head.addWidget(self.lbl_drive)
        self.lbl_drive_state = QLabel("少なくなっています", self)
        self.lbl_drive_state.setObjectName("wd_badge")
        self.lbl_drive_state.setVisible(False)
        head.addWidget(self.lbl_drive_state)
        lay.addLayout(head)

        self.scroll = QScrollArea(self)
        self.scroll.setObjectName("wd_scroll")
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._list_host = QWidget(self.scroll)
        self._list_layout = QVBoxLayout(self._list_host)
        self._list_layout.setContentsMargins(0, 0, 0, 0)
        self._list_layout.setSpacing(6)
        self.lbl_empty = QLabel(
            "作業用データはまだありません。\n断面の「元ファイルから全点で読み直す」か「高精細表示 (LOD)」を使うと作られます。", self._list_host,
        )
        self.lbl_empty.setObjectName("wd_muted")
        self.lbl_empty.setAlignment(Qt.AlignCenter)
        self.lbl_empty.setWordWrap(True)
        self._list_layout.addWidget(self.lbl_empty)
        self._list_layout.addStretch(1)
        self.scroll.setWidget(self._list_host)
        lay.addWidget(self.scroll, 1)

        loc = QHBoxLayout()
        loc.setSpacing(6)
        loc.addWidget(QLabel("置き場所:", self))
        self.lbl_location = _ElideLabel("", self)
        loc.addWidget(self.lbl_location, 1)
        self.btn_location = QPushButton("変更...", self)
        self.btn_location.setProperty("role", "outline")
        self.btn_location.setToolTip(
            "これから作る作業用データの置き場所を、空きの多いドライブのフォルダへ変えます。\n"
            "すでにある作業用データは移しません（新しい場所で作り直されます）。"
        )
        self.btn_location.clicked.connect(lambda _=False: self._choose_location())
        loc.addWidget(self.btn_location)
        self.btn_location_reset = QPushButton("元ファイルの横に戻す", self)
        self.btn_location_reset.setProperty("role", "ghost")
        self.btn_location_reset.clicked.connect(lambda _=False: self.change_location(""))
        set_disabled_reason(self.btn_location_reset, "置き場所は、すでに元ファイルの横です")
        loc.addWidget(self.btn_location_reset)
        lay.addLayout(loc)

        bottom = QHBoxLayout()
        bottom.setSpacing(6)
        self.btn_delete_unused = QPushButton("使われないものをまとめて削除", self)
        self.btn_delete_unused.setProperty("role", "danger")
        set_button_icon(self.btn_delete_unused, "trash", self._tokens)
        self.btn_delete_unused.setToolTip("元ファイルが無い・変わったなどで、もう使われない作業用データをまとめて消します")
        self.btn_delete_unused.clicked.connect(lambda _=False: self.delete_unused())
        set_disabled_reason(self.btn_delete_unused, "使われない作業用データはありません")
        bottom.addWidget(self.btn_delete_unused)
        bottom.addStretch(1)
        self.btn_close = QPushButton("閉じる", self)
        self.btn_close.setProperty("role", "ghost")
        self.btn_close.clicked.connect(self.accept)
        bottom.addWidget(self.btn_close)
        lay.addLayout(bottom)

        self.resize(680, 440)
        self._show_entries([], scanned=False)
        self.refresh()

    # --- 表示 ---
    def refresh(self, sync: bool = False) -> None:
        """作業用データを探し直す。sync=True はその場で（テスト用）"""
        sources = known_source_files(self.win)
        self._scanner.cancel()
        if sync:
            self._show_entries(find_workdata(sources), scanned=True)
        else:
            self.lbl_total.setText("調べています…")
            self._scanner.start(sources, lambda entries: self._show_entries(entries, scanned=True))

    def _busy(self) -> bool:
        fn = getattr(self.win, "_is_background_busy", None)
        try:
            return bool(fn()) if callable(fn) else False
        except Exception:
            return False

    def _show_entries(self, entries: List[WorkData], scanned: bool) -> None:
        self.entries = list(entries)
        for card in self.cards:
            self._list_layout.removeWidget(card)
            card.setParent(None)
            card.deleteLater()
        self.cards = []
        in_use_dirs = current_workdata_dirs(self.win)
        for i, entry in enumerate(self.entries):
            in_use = os.path.normcase(os.path.abspath(entry.directory)) in in_use_dirs
            card = _EntryCard(entry, in_use, self._tokens, self._list_host)
            card.btn_delete.clicked.connect(lambda _=False, e=entry: self.delete_entry(e))
            self._list_layout.insertWidget(i, card)
            self.cards.append(card)
        self.lbl_empty.setVisible(scanned and not self.entries)

        count, total, n_unused, unused_bytes = summarize(self.entries)
        if not scanned:
            self.lbl_total.setText("調べています…")
        elif count:
            self.lbl_total.setText(f"{count} 件・合計 {format_bytes(total)}")
        else:
            self.lbl_total.setText("作業用データなし")

        probe = _space_probe_path(self.win, self.entries)
        disk = disk_status(probe)
        if disk is not None:
            self.lbl_drive.setText(f"{drive_name(probe)} の空き {format_bytes(disk[1])} / 全体 {format_bytes(disk[0])}")
        else:
            self.lbl_drive.setText("空きは調べられませんでした")
        self.lbl_drive_state.setVisible(_is_low(disk))

        root = get_default_cache_root()
        if not root:
            self.lbl_location.set_full_text("元ファイルの横（標準）")
        elif os.path.isdir(root):
            self.lbl_location.set_full_text(root)
        else:
            self.lbl_location.set_full_text(f"{root}（見つかりません。ドライブがつながっているか確かめてください）")
        self.btn_location_reset.setEnabled(bool(root))

        self.btn_delete_unused.setEnabled(n_unused > 0)
        self.btn_delete_unused.setText(
            f"使われないものをまとめて削除（{n_unused} 件・{format_bytes(unused_bytes)}）" if n_unused else "使われないものをまとめて削除"
        )

    # --- 削除 ---
    def _is_in_use(self, entry: WorkData) -> bool:
        return os.path.normcase(os.path.abspath(entry.directory)) in current_workdata_dirs(self.win)

    def _remove(self, entry: WorkData) -> int:
        """1 つ消して、空いたバイト数を返す（消せなかったものは知らせる）"""
        if not os.path.isdir(entry.directory):
            return 0  # 窓を開いている間に、ほかで消されていた
        if self._is_in_use(entry):
            _release_current_workdata(self.win)
        ok, freed, kept = remove_workdata(entry.directory)
        log_info(
            f"作業用データを削除: {entry.directory}（{format_bytes(freed)} を空けました"
            + ("" if ok else f"、残したもの: {', '.join(kept) if kept else 'フォルダ'}") + "）"
        )
        if not ok:
            QMessageBox.warning(
                self, "作業用データの削除",
                f"「{entry.source_name}」の作業用データを消しきれませんでした。\n\n"
                + (f"残したもの: {', '.join(kept[:6])}\n" if kept else "")
                + "ほかのアプリが開いているか、このアプリのものではないファイルが入っています。\n"
                f"場所: {entry.directory}",
            )
        return freed

    def delete_entry(self, entry: WorkData) -> bool:
        in_use = self._is_in_use(entry)
        if in_use and self._busy():
            QMessageBox.information(self, "作業用データの削除", "処理の途中です。終わってから削除してください。")
            return False
        info = "元の E57 と保存した結果は変わりません。断面の読み直しや高精細表示をもう一度使うと、作り直されます。"
        if in_use:
            info += "\n\n今開いているファイルのデータです。削除すると、高精細表示を切ります。"
        if not HOOKS.ask_delete(
            self, "作業用データの削除", f"「{entry.source_name}」の作業用データ（{format_bytes(entry.size_bytes)}）を削除しますか？", info,
        ):
            return False
        freed = self._remove(entry)
        self._after_delete(freed)
        return True

    def delete_unused(self) -> bool:
        targets = [e for e in self.entries if not e.usable]
        if not targets:
            return False
        names = "\n".join(f"・{e.source_name}（{format_bytes(e.size_bytes)}）{e.reason}" for e in targets[:6])
        if len(targets) > 6:
            names += f"\n…ほか {len(targets) - 6} 件"
        total = sum(e.size_bytes for e in targets)
        if not HOOKS.ask_delete(
            self, "作業用データの削除", f"使われない作業用データ {len(targets)} 件（{format_bytes(total)}）を削除しますか？", names,
        ):
            return False
        freed = sum(self._remove(e) for e in targets)
        self._after_delete(freed)
        return True

    def _after_delete(self, freed: int) -> None:
        toast = getattr(self.win, "show_toast", None)
        if callable(toast) and freed > 0:
            toast(f"作業用データを削除しました（{format_bytes(freed)} を空けました）", 4000)
        self.refresh(sync=True)

    # --- 置き場所 ---
    def _choose_location(self) -> None:
        start = get_default_cache_root() or os.path.expanduser("~")
        chosen = QFileDialog.getExistingDirectory(self, "作業用データの置き場所", start)
        if chosen:
            self.change_location(chosen)

    def change_location(self, new_dir: str) -> bool:
        """置き場所を変える（空文字で元ファイルの横へ戻す）。変えたら True"""
        if self._busy():
            QMessageBox.information(self, "作業用データの置き場所", "処理の途中です。終わってから変更してください。")
            return False
        new_dir = os.path.abspath(new_dir) if new_dir else ""
        old = get_default_cache_root() or ""
        if os.path.normcase(new_dir) == os.path.normcase(old):
            return False
        if new_dir and not can_write_dir(new_dir):
            QMessageBox.warning(self, "作業用データの置き場所", f"このフォルダには書き込めません。別のフォルダを選んでください。\n\n{new_dir}")
            return False
        lod_was_on = _release_current_workdata(self.win)
        set_saved_workdata_dir(new_dir)
        set_default_cache_root(new_dir or None)
        log_info(f"作業用データの置き場所を変更: {new_dir if new_dir else '元ファイルの横（標準）'}")
        toast = getattr(self.win, "show_toast", None)
        if callable(toast):
            msg = "作業用データの置き場所を変えました"
            if lod_was_on:
                msg += "（高精細表示を切りました。入れ直すと、新しい場所のデータを使います）"
            toast(msg, 5000)
        self.refresh(sync=True)
        return True


def _release_current_workdata(win) -> bool:
    """今開いているファイルの作業用データを手放す（高精細表示を切る）。切ったら True"""
    chk = getattr(win, "chk_lod", None)
    try:
        if chk is not None and chk.isChecked():
            chk.setChecked(False)  # → MainWindow._on_lod_toggled(False) がストアを閉じる
            return True
        lod = getattr(win, "lod", None)
        if lod is not None:
            lod.set_enabled(False)
            lod.detach()
    except Exception as e:
        log_warning(f"高精細表示を切れませんでした: {e}")
    return False


def open_workdata_dialog(win) -> None:
    dlg = WorkDataDialog(win)
    win._workdata_dialog = dlg
    dlg.exec()
    refresh_workdata_summary(win)


# ----------------------------------------------------------------------------- 書く前の空きの確認
def confirm_save_free_space(win, out_path: str, scans: Sequence, layer_mask, split_masks, use_fullres: bool) -> bool:
    """
    保存を始める前に、保存先の空きを確かめる。続けてよければ True。
    書く大きさは見込みなので、足りないと見えたときも「このまま保存する」を選べる
    （保存は一時ファイルに書いてから置き換えるので、途中で止まっても今あるファイルは変わらない）。
    見積もれないときは止めない。
    """
    try:
        if split_masks:
            n_out = int(sum(int(np.count_nonzero(m)) for m in split_masks.values()))
        else:
            n_out = int(np.count_nonzero(layer_mask))
        need = estimate_save_bytes(scans, n_out, bool(use_fullres))
    except Exception as e:
        log_warning(f"保存前の空きの確認を飛ばしました（大きさを見積もれません）: {e}")
        return True
    while True:
        chk = check_free(out_path, need)
        if chk.ok:
            return True
        choice = HOOKS.ask_low_space(
            win, "保存先の空きの確認",
            f"保存先（{chk.drive}）の空きが足りない見込みです。",
            f"・書き出す大きさの見込み: 約 {format_bytes(need)}\n"
            f"・{chk.drive} の空き: {format_bytes(chk.free_bytes)}\n\n"
            "このまま保存すると、途中で止まることがあります（その場合も、今あるファイルは変わりません）。\n"
            "作業用データを消すと、空きを作れます。",
            True,
        )
        if choice == "proceed":
            log_warning(f"空きが足りない見込みのまま保存を始めます（必要 約 {format_bytes(need)} / 空き {format_bytes(chk.free_bytes)}）: {out_path}")
            return True
        if choice != "manage":
            return False
        open_workdata_dialog(win)


def confirm_lod_build(win, scans: Sequence) -> bool:
    """
    高精細表示のデータを作る前の確認。作ってよければ True。
    空きが足りないときは始めない（core/lod_store.py も同じ式で止めるが、そちらは読み直し用を数分かけて書いたあとになる）。
    """
    title = "高精細表示 (LOD) データの作成"
    need = workdata_need_for_scans(scans)
    while not need.ok:
        choice = HOOKS.ask_low_space(
            win, title,
            f"作業用データを作る空きが足りません（{need.drive}）。",
            f"・作る大きさ: 約 {format_bytes(need.total_bytes)}（作成中は一時的に約 {format_bytes(need.need_free_bytes)} 使います）\n"
            f"・{need.drive} の空き: {format_bytes(need.free_bytes)}\n\n"
            "ほかの作業用データを消すか、置き場所を空きの多いドライブへ変えると作れます。",
            False,
        )
        if choice != "manage":
            return False
        open_workdata_dialog(win)
        need = workdata_need_for_scans(scans)

    if need.reload_bytes > 0:
        size_txt = (
            f"約 {format_bytes(need.total_bytes)}（読み直し用 {format_bytes(need.reload_bytes)} + "
            f"高精細表示用 {format_bytes(need.lod_bytes)}）"
        )
    else:
        size_txt = f"約 {format_bytes(need.lod_bytes)}（読み直し用は作成済み）"
    place = os.path.dirname(need.directory) if need.directory else ""
    place_txt = f"{place}（{'設定した置き場所' if get_default_cache_root() else '元ファイルの横'}）"
    free_txt = (
        f"・{need.drive} の空き: {format_bytes(need.free_bytes)}（作成中は一時的に約 {format_bytes(need.need_free_bytes)} 使います）\n"
        if need.free_bytes is not None else ""
    )
    reply = QMessageBox.question(
        win, title,
        f"このファイルの LOD データがまだありません。作成しますか？\n\n"
        f"・元ファイル {need.n_points:,} 点を空間順に並べ替えたデータを作ります\n"
        f"・大きさ: {size_txt}\n"
        f"・置き場所: {place_txt}\n"
        f"{free_txt}"
        f"・時間: 1 億点あたり 1〜2 分（読み直し用が未作成なら、元ファイルを読むぶんが加わります）\n\n"
        f"作成は 1 回だけで、次回からはすぐ使えます。途中で中止できます。\n"
        f"あとで「設定 → システム → 作業用データ」から削除できます。",
        QMessageBox.Yes | QMessageBox.No,
        QMessageBox.Yes,
    )
    return reply == QMessageBox.Yes


# ----------------------------------------------------------------------------- 設定 → システムの行
def refresh_workdata_summary(win, sync: bool = False) -> None:
    """設定 → システムの「作業用データ」の行を更新する"""
    lbl = getattr(win, "lbl_workdata_summary", None)
    scanner = getattr(win, "_workdata_scanner", None)
    if lbl is None or scanner is None:
        return

    def apply(entries: List[WorkData]) -> None:
        try:
            count, total, n_unused, unused_bytes = summarize(entries)
            probe = _space_probe_path(win, entries)
            disk = disk_status(probe)
            text = f"作業用データ: {format_bytes(total) if count else 'なし'}"
            if disk is not None:
                text += f" ／ 空き {format_bytes(disk[1])}" + ("（少ない）" if _is_low(disk) else "")
            root = get_default_cache_root()
            if root and not os.path.isdir(root):
                text += "\n置き場所が見つかりません（「管理...」で確かめてください）"
            lbl.setText(text)
            tip = [f"作業用データ {count} 件・合計 {format_bytes(total)}" if count else "作業用データはまだありません"]
            if n_unused:
                tip.append(f"うち使われないもの {n_unused} 件・{format_bytes(unused_bytes)}（「管理...」からまとめて消せます）")
            if disk is not None:
                tip.append(f"{drive_name(probe)} の空き {format_bytes(disk[1])} / 全体 {format_bytes(disk[0])}")
            tip.append("断面の読み直しと高精細表示のための、元ファイルの写しです。「管理...」で大きさの確認・削除・置き場所の変更ができます。")
            lbl.setToolTip("\n".join(tip))
            win._workdata_entries = list(entries)
        except RuntimeError:
            pass  # 待っている間に画面が閉じられた

    try:
        sources = known_source_files(win)
        scanner.cancel()
        if sync:
            apply(find_workdata(sources))
        else:
            scanner.start(sources, apply)
    except RuntimeError:
        pass


class _ShowWatcher(QObject):
    """設定 → システムのタブが開かれたら、作業用データの行を更新する（続けて何度も調べない）"""

    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self._last = 0.0

    def eventFilter(self, obj, event):  # type: ignore[override]
        if event.type() == QEvent.Show:
            now = time.monotonic()
            if now - self._last > 2.0:
                self._last = now
                QTimer.singleShot(0, self.win, lambda: refresh_workdata_summary(self.win))
        return False


def install_system_tools(win) -> None:
    """
    MainWindow の __init__ から 1 回呼ぶ（新UI・従来UI の両方）。
    置き場所の設定を core へ渡し、設定 → システムに「描画の GPU」「完了の知らせ」「作業用データ」「版」の行を足す。
    長い処理の完了の知らせ（gui/task_notify.py）も、ここで取り付ける。
    """
    apply_workdata_dir_setting()
    install_long_task_notify(win)
    tab = getattr(win, "tab_system", None)
    lay = tab.layout() if tab is not None else None
    if not isinstance(lay, QGridLayout):
        return
    row = lay.rowCount()
    row = install_gpu_row(win, tab, lay, row)      # 描画の GPU（gui/gpu_select.py）
    lay.addWidget(make_notify_checkbox(win, tab), row, 0, 1, 2)
    row += 1

    box = QWidget(tab)
    h = QHBoxLayout(box)
    h.setContentsMargins(0, 0, 0, 0)
    h.setSpacing(6)
    win.lbl_workdata_summary = QLabel("作業用データ: 調べています…", box)
    win.lbl_workdata_summary.setWordWrap(True)
    h.addWidget(win.lbl_workdata_summary, 1)
    win.btn_workdata_manage = QPushButton("管理...", box)
    win.btn_workdata_manage.setProperty("role", "soft")
    win.btn_workdata_manage.setToolTip("作業用データ（断面の読み直しと高精細表示のための写し）の大きさを見る・消す・置き場所を変える")
    win.btn_workdata_manage.clicked.connect(lambda _=False: open_workdata_dialog(win))
    h.addWidget(win.btn_workdata_manage)
    lay.addWidget(box, row, 0, 1, 2)

    win.lbl_build_info = QLabel(f"版: {get_build_info().describe()}", tab)
    win.lbl_build_info.setWordWrap(True)
    win.lbl_build_info.setTextInteractionFlags(Qt.TextSelectableByMouse)
    win.lbl_build_info.setToolTip("今動いているアプリの版です。不具合を伝えるときに、この行を添えると調べやすくなります")
    lay.addWidget(win.lbl_build_info, row + 1, 0, 1, 2)

    win._workdata_scanner = WorkDataScanner(win)
    win._workdata_show_watcher = _ShowWatcher(win)
    tab.installEventFilter(win._workdata_show_watcher)
    # 起動時には調べない（この行が見えるのはタブを開いたときだけ。開かれたときに _ShowWatcher が調べる）
