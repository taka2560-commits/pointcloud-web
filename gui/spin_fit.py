# -*- coding: utf-8 -*-
"""
数値欄（QDoubleSpinBox）の幅を、入る値の桁数に合わせる。

断面シークの数値欄は幅 78px の固定だった。平面直角座標のような桁の多い座標（例: -148769.06 m）は
「-1487」までしか見えず、位置を読むことも打ち直すこともできなかった（2026-11-13、利用者の画面で確認）。
範囲の両端の値が欠けずに入る幅を、見た目の指定（枠・余白・上下ボタン）から求めて当てる。
"""
from typing import Iterable, Optional, Sequence

from PySide6.QtCore import QRect
from PySide6.QtWidgets import QAbstractSpinBox, QStyle, QStyleOptionSpinBox

# 文字のほかに空けておく幅（カーソルと、欄の内側の左右の遊び）
TEXT_SLACK = 8
MIN_WIDTH = 78
MAX_WIDTH = 150
_PROBE_WIDTH = 300


def spin_overhead(spin: QAbstractSpinBox) -> int:
    """数値欄のうち、文字の入らない部分の幅 (px)。隠れている欄（まだ配置されていない欄）でも求められる"""
    spin.ensurePolished()
    opt = QStyleOptionSpinBox()
    spin.initStyleOption(opt)
    opt.rect = QRect(0, 0, _PROBE_WIDTH, max(20, spin.sizeHint().height()))
    field = spin.style().subControlRect(QStyle.CC_SpinBox, opt, QStyle.SC_SpinBoxEditField, spin)
    w = field.width()
    if w <= 0 or w > _PROBE_WIDTH:
        return 40
    return _PROBE_WIDTH - w


def needed_spin_width(spin: QAbstractSpinBox, texts: Optional[Sequence[str]] = None) -> int:
    """texts（省略時は範囲の両端の値）が欠けずに入る幅 (px)"""
    spin.ensurePolished()
    fm = spin.lineEdit().fontMetrics()
    if texts is None:
        texts = [spin.textFromValue(spin.minimum()), spin.textFromValue(spin.maximum())]
    suffix = spin.suffix() if hasattr(spin, "suffix") else ""
    prefix = spin.prefix() if hasattr(spin, "prefix") else ""
    text_w = max((fm.horizontalAdvance(prefix + t + suffix) for t in texts), default=0)
    return text_w + spin_overhead(spin) + TEXT_SLACK


def fit_spin_group(spins: Iterable[QAbstractSpinBox], widths: Iterable[int], lo: int = MIN_WIDTH, hi: int = MAX_WIDTH) -> int:
    """同じ列に並ぶ数値欄を、いちばん広い必要幅にそろえる（そろえると、横のスライダーの長さもそろう）"""
    spins = [sp for sp in spins if sp is not None]
    width = max([lo] + [int(w) for w in widths])
    width = max(lo, min(hi, width))
    for sp in spins:
        sp.setMaximumWidth(width)
    return width


def fit_section_spins(win) -> None:
    """断面シークと厚みの数値欄の幅を、今の点群の座標の桁数に合わせる"""
    pos = [getattr(win, f"spin_pos_{a}", None) for a in "xyz"]
    thick = [getattr(win, f"spin_thick_{a}", None) for a in "xyz"]
    if all(sp is not None for sp in pos + thick):
        # 厚みは 5000m まで入るが、ふつうは 2 桁。めったに使わない桁数のために欄を広げない（広げるとスライダーが短くなる）
        widths = [needed_spin_width(sp) for sp in pos] + [needed_spin_width(sp, ["99.99"]) for sp in thick]
        fit_spin_group(pos + thick, widths)
    wall_pos = getattr(win, "spin_pos_wall", None)
    wall_thick = getattr(win, "spin_thick_wall", None)
    if wall_pos is not None and wall_thick is not None:
        widths = [needed_spin_width(wall_pos), needed_spin_width(wall_thick, ["99.99"])]
        fit_spin_group([wall_pos, wall_thick], widths)
