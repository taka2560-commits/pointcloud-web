# -*- coding: utf-8 -*-
"""
断面シークのスライダーの目盛りを決める（Qt に依存しない計算だけ）。

スライダーは整数しか持てないので、「位置 (m) = スライダーの値 ÷ 倍率」で対応させる。
倍率は設定の「断面シーク刻み」の逆数（1cm なら 100）。**倍率とスライダーの範囲は必ず一緒に決めること。**
片方だけ変えると、スライダーの左端・右端が点群の範囲とずれる（2026-11-13 の不具合。
読み込み後に刻みを 1cm → 5cm に変えると、左端が 7.20m ではなく 720 ÷ 20 = 36.00m になり、地面まで下げられなかった）。
"""
import math
from typing import Sequence, Tuple

# QSlider の値は 32bit 整数。余裕を見てこの大きさまでに収める
SLIDER_INT_LIMIT = 2_000_000_000

DEFAULT_STEP = 0.01


def seek_scale(step: float, world_min: Sequence[float], world_max: Sequence[float]) -> float:
    """
    スライダーの倍率（位置 = 値 ÷ 倍率）を返す。ふつうは刻み幅の逆数。
    座標が大きすぎて 32bit 整数に収まらないときだけ、収まるまで 10 分の 1 ずつ粗くする。
    """
    try:
        step = float(step)
    except (TypeError, ValueError):
        step = DEFAULT_STEP
    if not math.isfinite(step) or step <= 0.0:
        step = DEFAULT_STEP
    scale = 1.0 / step
    max_abs = 1.0
    for v in list(world_min) + list(world_max):
        v = float(v)
        if math.isfinite(v):
            max_abs = max(max_abs, abs(v))
    while scale * max_abs > SLIDER_INT_LIMIT and scale > 1e-6:
        scale /= 10.0
    return scale


def seek_slider_range(lo: float, hi: float, scale: float) -> Tuple[int, int]:
    """
    点群の範囲 lo〜hi (m) を覆うスライダーの範囲（整数）を返す。
    端は外側へ丸めるので、粗い刻みでも最小・最大の位置まで必ず届く。
    """
    lo, hi = float(lo), float(hi)
    if hi < lo:
        lo, hi = hi, lo
    a = int(math.floor(lo * scale + 1e-9))
    b = int(math.ceil(hi * scale - 1e-9))
    if b <= a:
        b = a + 1
    return a, b


def seek_slider_value(pos: float, scale: float) -> int:
    """位置 (m) に対応するスライダーの値"""
    return int(round(float(pos) * scale))


def spin_decimals_for(step: float) -> int:
    """刻み幅を表せる小数の桁数（1cm 以上は 2 桁、5mm・1mm は 3 桁）"""
    try:
        step = float(step)
    except (TypeError, ValueError):
        return 2
    return 3 if 0.0 < step < 0.01 - 1e-12 else 2
