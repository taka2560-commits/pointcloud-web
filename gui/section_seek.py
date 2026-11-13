# -*- coding: utf-8 -*-
"""
断面シークのスライダーの目盛りを決める（Qt に依存しない計算だけ）。

スライダーは整数しか持てないので、「位置 (m) = スライダーの値 ÷ 倍率」で対応させる。
倍率は設定の「断面シーク刻み」の逆数（1cm なら 100）。**倍率とスライダーの範囲は必ず一緒に決めること。**
片方だけ変えると、スライダーの左端・右端が点群の範囲とずれる（2026-11-13 の不具合。
読み込み後に刻みを 1cm → 5cm に変えると、左端が 7.20m ではなく 720 ÷ 20 = 36.00m になり、地面まで下げられなかった）。

どちらのスライダーも「真ん中が起点」（2026-11-13、利用者の提案）:
- シーク: 基点（読み込み直後は点群の中央、「現在位置/選択点を基点に」を押したらその位置）がバーの真ん中。
  左右どちらへも動かせ、点群の最小・最大までは必ず届く（基点が中央でないときは、片側が点群の外まで伸びる）。
- 厚み幅: 真ん中が 1m。左へ薄く（左端 1cm）、右へ厚く（右端 100m）。位置は厚みの桁に比例する（0.1m は左から 1/4、10m は 3/4）。
  以前は 0.1m〜敷地の大きさを等間隔に割り当てていて、よく使う 0.1〜5m が左端の数 px に固まっていた。
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


def centered_seek_span(center: float, lo: float, hi: float) -> Tuple[float, float]:
    """
    基点 center を真ん中にして、点群の範囲 lo〜hi を全部覆う範囲 (a, b) を返す。
    基点が点群の中央なら (lo, hi) そのもの。基点が端に寄っているほど、反対側が点群の外まで伸びる。
    """
    lo, hi = float(lo), float(hi)
    if hi < lo:
        lo, hi = hi, lo
    try:
        c = float(center)
    except (TypeError, ValueError):
        c = 0.5 * (lo + hi)
    if not math.isfinite(c):
        c = 0.5 * (lo + hi)
    c = min(max(c, lo), hi)
    r = max(c - lo, hi - c)
    if r <= 0.0:
        r = 0.5
    return c - r, c + r


# ---- 厚み幅のスライダー（真ん中が 1m） ----
THICK_SLIDER_MAX = 400      # 目盛りは 0〜400（厚みが 10 倍になるごとに 100 目盛り）
THICK_CENTER = 1.0          # 真ん中（目盛り 200）の厚み (m)
THICK_MIN = 0.01            # 左端 (m)
THICK_MAX = 100.0           # 右端 (m)
THICK_SINGLE_STEP = 2       # 矢印キー 1 回（約 5%）
THICK_PAGE_STEP = 20        # バーの空いた所をクリック（約 1.6 倍）
_THICK_STEPS_PER_DECADE = 100.0


def _nice_thickness(t: float) -> float:
    """切りのよい厚みに丸める（1m 未満は 1cm、10m 未満は 5cm、それ以上は 50cm 刻み）"""
    if t < 1.0:
        q = 0.01
    elif t < 10.0:
        q = 0.05
    else:
        q = 0.5
    return min(THICK_MAX, max(THICK_MIN, round(round(t / q) * q, 2)))


def thick_from_slider(value: int) -> float:
    """厚み幅のスライダーの値（0〜THICK_SLIDER_MAX）を厚み (m) にする"""
    v = min(max(int(value), 0), THICK_SLIDER_MAX)
    half = THICK_SLIDER_MAX / 2.0
    return _nice_thickness(THICK_CENTER * 10.0 ** ((v - half) / _THICK_STEPS_PER_DECADE))


def thick_to_slider(thick: float) -> int:
    """厚み (m) に対応する厚み幅のスライダーの値。範囲の外は端に寄せる（数値欄にはもっと大きい値も入る）"""
    try:
        t = float(thick)
    except (TypeError, ValueError):
        t = THICK_CENTER
    if not math.isfinite(t) or t <= 0.0:
        t = THICK_MIN
    t = min(max(t, THICK_MIN), THICK_MAX)
    half = THICK_SLIDER_MAX / 2.0
    return int(round(half + _THICK_STEPS_PER_DECADE * math.log10(t / THICK_CENTER)))
