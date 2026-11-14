# -*- coding: utf-8 -*-
"""
読み込みの前に、PC のメモリに収まるかを確かめる（見積もりと判定は core/memguard.py。ここは画面側）

なぜ要るか（2026-11-14、利用者の PC の確認から）
    設定 → システムの「大容量点群読込上限」を「無制限」にして 5.4 億点のファイルを開くと、約 100GB のメモリが要る。
    32GB の PC では、足りない分を Windows がディスクで補う（ページファイル）。利用者の PC ではページファイルが 90GB まで
    膨らみ、C ドライブの空きが 20GB まで減っていた。読み込みは何も言わずに始まり、PC 全体が遅くなるだけだった。

ふるまい
    - 収まる見込みなら、これまでと同じ（何も聞かない）。利用者のいつもの設定（自動最適化）は収まる。
    - 「自動最適化」で収まらない PC（メモリの少ない PC）では、収まる点数まで自動で下げ、ログの欄に理由を書く。
    - 点数や「無制限」を自分で選んでいて収まらないときは、始める前に聞く:
        おすすめの点数で読み込む（既定）／ このまま読み込む ／ やめる
      点数を減らして読み込んでも、保存のときに「全点で保存」を選べば、元ファイルの全点へ判定を写して出力できる。
    - ヘッダを読めない・メモリの量が分からないときは、止めない（読み込みそのものが、いつものエラーを出す）。

ホットファイル（gui/main_window.py）から呼ぶのは resolve_load_limit(win, file_path) だけ。
"""

from dataclasses import dataclass
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMessageBox

from core.logger import log_info, log_warning
from core.memguard import LoadCheck, auto_load_limit, check_load, read_file_costs
from core.sysinfo import format_bytes, memory_status
from gui.worker_thread import collect_e57_files

LOAD_SETTING_AUTO = -1       # 「自動最適化」
DEFAULT_DISPLAY_BUDGET = 2_000_000
DIALOG_TITLE = "読み込みのメモリの確認"


@dataclass(frozen=True)
class LoadLimit:
    """読み込みに使う上限"""
    max_points: Optional[int]   # FileLoadWorker へ渡す読み込み上限（None = 全点）
    cancelled: bool = False     # 利用者が「やめる」を選んだ
    note: str = ""              # 上限を設定から変えたときの説明


def format_points(n: int) -> str:
    """点数を「5,982 万点」「5.38 億点」「8,200 点」の形にする"""
    n = int(n)
    if n >= 100_000_000:
        return f"{n / 1e8:.2f}".rstrip("0").rstrip(".") + " 億点"
    if n >= 10_000:
        return f"{round(n / 1e4):,} 万点"
    return f"{n:,} 点"


def _ask_load_memory(parent, title: str, text: str, info: str, recommend_label: str) -> str:
    """メモリに収まらないときの確認。戻り値は recommend（おすすめの点数で読む）/ proceed（このまま読む）/ cancel のどれか"""
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Warning)
    box.setWindowTitle(title)
    box.setText(text)
    box.setInformativeText(info)
    btn_rec = box.addButton(recommend_label, QMessageBox.AcceptRole)
    btn_go = box.addButton("このまま読み込む", QMessageBox.DestructiveRole)
    btn_cancel = box.addButton("やめる", QMessageBox.RejectRole)
    box.setDefaultButton(btn_rec)
    box.setEscapeButton(btn_cancel)
    box.exec()
    clicked = box.clickedButton()
    if clicked is btn_rec:
        return "recommend"
    if clicked is btn_go:
        return "proceed"
    return "cancel"


class _Hooks:
    """確認の窓。テストは、ここの関数を差し替えて答えを決める"""
    ask_load_memory = staticmethod(_ask_load_memory)
    memory_status = staticmethod(memory_status)


HOOKS = _Hooks()


def _combo_data(win, name: str, default: int) -> int:
    combo = getattr(win, name, None)
    if combo is None:
        return default
    data = combo.currentData()
    return default if data is None else int(data)


def _setting_text(win) -> str:
    combo = getattr(win, "combo_load_budget", None)
    return combo.currentText() if combo is not None else "自動最適化"


def describe_check(chk: LoadCheck, setting_text: str) -> str:
    """確認の窓に出す内訳"""
    plan = chk.plan
    if plan.subsampled:
        loaded = f"{format_points(plan.n_loaded)}（{plan.max_step} 点に 1 点）"
    else:
        loaded = f"{format_points(plan.n_loaded)}（全点）"
    lines = [
        f"・ファイルの点数: {format_points(plan.n_total)}" + ("（大きさからの見込みを含む）" if plan.estimated else ""),
        f"・読み込む点数: {loaded} … 設定「{setting_text}」",
        f"・要るメモリの見込み: 約 {format_bytes(plan.need_bytes)}",
        f"・この PC のメモリ: {format_bytes(chk.total_bytes)}（読み込みに使えるのは 約 {format_bytes(chk.limit_bytes)} まで）",
        "",
        "収まらないまま読み込むと、Windows がディスクをメモリの代わりに使い、PC 全体がとても遅くなります。"
        "ディスクの空きも減ります。",
        "",
        "点数を減らして読み込んでも、保存のときに「全点で保存」を選べば、元ファイルの全点へ判定を写して出力できます。",
        "この確認を毎回出さないようにするには、設定 → システムの「大容量点群読込上限」を「自動最適化」にしてください。",
    ]
    return "\n".join(lines)


def resolve_load_limit(win, file_path: str) -> LoadLimit:
    """
    読み込みを始める前に呼ぶ。設定とメモリから読み込み上限を決める。
    収まらない見込みで、利用者が「やめる」を選んだら cancelled=True を返す（呼び出し側は読み込みを始めない）。
    """
    setting = _combo_data(win, "combo_load_budget", LOAD_SETTING_AUTO)
    display_budget = _combo_data(win, "budget_combo", DEFAULT_DISPLAY_BUDGET)
    mem = HOOKS.memory_status()
    tier = auto_load_limit(mem[0] if mem else None)
    if setting == LOAD_SETTING_AUTO:
        requested: Optional[int] = tier
    else:
        requested = setting if setting > 0 else None

    if mem is None:
        return LoadLimit(requested)  # メモリの量が分からない環境では、確かめようがないので止めない

    chk: Optional[LoadCheck] = None
    QApplication.setOverrideCursor(Qt.WaitCursor)
    try:
        paths = collect_e57_files(file_path)
        if paths:
            chk = check_load(read_file_costs(paths), requested, display_budget, memory=mem)
    except Exception as e:
        log_warning(f"読み込み前のメモリの確認を飛ばしました（ファイルの情報を読めません）: {e}")
    finally:
        QApplication.restoreOverrideCursor()
    if chk is None:
        return LoadLimit(requested)

    plan = chk.plan
    summary = (
        f"要るメモリの見込み 約 {format_bytes(plan.need_bytes)}（{plan.n_loaded:,} 点"
        + (f"、{plan.max_step} 点に 1 点" if plan.subsampled else "、全点")
        + f"）／ 使える量 約 {format_bytes(chk.limit_bytes)}"
    )
    if chk.ok:
        log_info(f"読み込み前のメモリの確認: {summary}")
        return LoadLimit(requested)

    rec = int(chk.recommended_points or 0)
    rec_plan = chk.recommended_plan
    if rec <= 0 or rec_plan is None:
        return LoadLimit(requested)
    rec_txt = format_points(rec_plan.n_loaded)

    if setting == LOAD_SETTING_AUTO:
        note = (
            f"この PC のメモリ（{format_bytes(chk.total_bytes)}）に合わせて、読み込む点数を {rec_txt}に下げました"
            f"（{format_points(plan.n_loaded)}では 約 {format_bytes(plan.need_bytes)} が要り、使える 約 {format_bytes(chk.limit_bytes)} を超えるため）"
        )
        log_info(f"読み込み前のメモリの確認: {note}")
        _append_log(win, note)
        return LoadLimit(rec, note=note)

    log_warning(f"読み込み前のメモリの確認: 収まらない見込みです。{summary}")
    choice = HOOKS.ask_load_memory(
        win, DIALOG_TITLE,
        "この設定で読み込むと、この PC のメモリに収まりません。",
        describe_check(chk, _setting_text(win)),
        f"{rec_txt}で読み込む（おすすめ）",
    )
    if choice == "recommend":
        note = f"メモリに収まるよう、読み込む点数を {rec_txt}にしました（設定は「{_setting_text(win)}」のまま）"
        log_info(f"読み込み前のメモリの確認: {note}")
        _append_log(win, note)
        return LoadLimit(rec, note=note)
    if choice == "proceed":
        log_warning(f"メモリに収まらない見込みのまま読み込みます（{summary}）: {file_path}")
        return LoadLimit(requested)
    log_info("読み込み前のメモリの確認: 利用者が読み込みをやめました")
    return LoadLimit(requested, cancelled=True)


def _append_log(win, text: str) -> None:
    log_text = getattr(win, "log_text", None)
    try:
        if log_text is not None:
            log_text.append(text)
    except RuntimeError:
        pass
