# -*- coding: utf-8 -*-
"""
空中に浮いたノイズを探すための、反射強度の配色（Qt に依存しない。numpy だけ）

なぜ要るか（2026-11-14、利用者の依頼「反射強度の色で空中に浮遊したノイズ処理しやすくなる色パターンを研究し実装してほしい」）
    これまでの反射強度の配色（白黒・Heat・Plasma・Turbo・Viridis）は、どれも「弱い反射 = 暗い色」だった。
    3D 画面の背景は暗い。弱い反射の点は、背景に溶けて見えなくなる。

調べて分かったこと（利用者の実データ 3 現場。1 億 3,564 万点・2,253 万点・167 万点）
    「周りに点がほとんど無い点」（30cm の立方体の中に 6 点未満 = 空中に浮いた点）を数え、反射強度の順位ごとの出やすさを見た。
      - いちばん弱い 10% の帯に、空中の点の 33〜39% が集まる（出やすさは全体の 3〜4 倍。いちばん弱い 1% では最大 9〜18 倍）
      - いちばん強い 20% の帯には、空中の点の 2〜11% しか無い（出やすさは全体の 0.1〜0.6 倍）
      - 残りの半分前後は、中くらいの強さにある（反射強度だけでは見分けられない）
    利用者が実際に消した点（元 1 億 3,564 万点 → 掃除後 1 億 980 万点）で、空中にぶら下がっていたものを横から描くと、
    白黒の配色ではほぼ黒（背景に溶ける）で、弱い反射を明るくした配色ではっきり見えた。
    レーザースキャナーの浮遊ノイズの主な原因（物の縁でレーザーが手前と奥にまたがる混合画素、ほこり・雨・霧、
    ガラス越しの写り込み）は、どれも戻ってくる光が弱い、という文献の説明と合う。
    数え方と数字は QA_REPORT_2026-10-31.md（2026-11-14 の行）と AI_HANDOVER.md §5(3)。

配色の考え方
    1. 弱い反射ほど、背景からいちばん離れた明るさにする（暗い背景なら、白に近い水色）。これまでと明暗が逆になる。
    2. どの色も、背景とのコントラスト比 3 以上を保つ（空中の点の半分は中くらいの強さにある。強い反射も真っ暗にはしない）。
    3. 強い反射（ほぼ確実に面）は、暗めの無彩色に落として、手前の点の邪魔をしない。
    4. 色は、強度の値そのものではなく「その点群の中での順位」で決める。スキャナーや書き出しの設定で値の幅が違っても、
       「いちばん弱い 10% が水色より明るい」という意味が変わらない。
    5. 色相は、白 → 水色 → 青 → 紫 → 灰。選択中の点の黄色・ノイズレイヤーの赤と紛れない色だけを使う
       （黄〜橙〜赤の並びは、いちばん目立つが、選択した点と見分けがつかなくなるので使わない）。
    6. 凹凸の陰影は掛けない（陰影は色を最大 3 割暗くするので、2. の保証が崩れる）。
    7. 背景が白のときは、明暗を反転させた表を使う（弱い反射 = 濃い色）。

配色は 2 つ
    MODE_NOISE_GLOW … 順位ぜんたいを色で分ける（白 → 水色 → 青 → 紫 → 灰）。空中の点を探すときの標準
    MODE_NOISE_SPOT … 灰色の濃淡を基本に、いちばん弱い 10% だけを水色にする。画面が落ち着くので、長い作業向け
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

MODE_NOISE_GLOW = "反射強度 (ノイズ探索 / 弱い反射を光らせる)"
MODE_NOISE_SPOT = "反射強度 (ノイズ探索・淡色 / 弱い 10% だけ着色)"
NOISE_MODES: Tuple[str, ...] = (MODE_NOISE_GLOW, MODE_NOISE_SPOT)

LUT_SIZE = 1024
TABLE_SIZE = 4096             # 「正規化済みの強度 → 色」の表の細かさ（3D 画面が使う。順位の変換と色の表を 1 つにまとめたもの）
RANK_KNOTS = 513              # 順位の表の細かさ（0.2% 刻み）
RANK_SAMPLE_MAX = 4_000_000   # 順位の表を作るときに見る点数の上限（それより多ければ等間隔に抜き出す）
HIGHLIGHT_RANK = 0.10         # 「いちばん弱い帯」の上端（調べた 3 現場で、空中の点の 33〜39% がここに入る）
LIGHT_BACKGROUND_LUMA = 0.5   # 背景の明るさがこれより上なら、白い背景用の表を使う
MIN_CONTRAST = 3.0            # どの色も、背景とのコントラスト比をこれ以上に保つ（図として見分けられる目安）

# 3D 画面の背景色の選択肢（gui/main_window.py の bg_combo と同じ値。配色の検査に使う）
BACKGROUND_DARK = (20 / 255.0, 20 / 255.0, 26 / 255.0)
BACKGROUND_BLACK = (0.0, 0.0, 0.0)
BACKGROUND_WHITE = (245 / 255.0, 245 / 255.0, 247 / 255.0)
# 点に使われている決まった色（紛れてはいけない相手）: 選択中の点、ノイズレイヤー
SELECTION_COLOR = (1.0, 0.9, 0.1)
NOISE_LAYER_COLOR = (0.95, 0.25, 0.25)

Stops = Sequence[Tuple[float, Tuple[float, float, float]]]

# 順位（0 = いちばん弱い反射、1 = いちばん強い反射）→ 色。暗い背景用（順位が上がるほど暗くなる）
_GLOW_DARK: Stops = (
    (0.00, (0.92, 1.00, 0.96)),
    (0.02, (0.35, 1.00, 0.75)),
    (0.10, (0.25, 0.90, 1.00)),
    (0.25, (0.40, 0.70, 1.00)),
    (0.45, (0.58, 0.55, 0.95)),
    (0.65, (0.56, 0.50, 0.66)),
    (0.85, (0.44, 0.44, 0.50)),
    (1.00, (0.38, 0.39, 0.44)),
)
# 白い背景用（明暗を反転。弱い反射 = 濃い色。順位が上がるほど明るくなる）
_GLOW_LIGHT: Stops = (
    (0.00, (0.00, 0.16, 0.20)),
    (0.02, (0.00, 0.30, 0.36)),
    (0.10, (0.00, 0.42, 0.55)),
    (0.25, (0.16, 0.40, 0.72)),
    (0.45, (0.42, 0.38, 0.70)),
    (0.65, (0.50, 0.46, 0.60)),
    (0.85, (0.53, 0.54, 0.59)),
    (1.00, (0.54, 0.55, 0.59)),
)
_SPOT_DARK: Stops = (
    (0.00, (0.92, 1.00, 0.96)),
    (0.02, (0.35, 1.00, 0.75)),
    (HIGHLIGHT_RANK, (0.25, 0.90, 1.00)),
    (HIGHLIGHT_RANK + 0.03, (0.72, 0.72, 0.74)),
    (0.50, (0.55, 0.55, 0.58)),
    (1.00, (0.39, 0.39, 0.42)),
)
_SPOT_LIGHT: Stops = (
    (0.00, (0.00, 0.16, 0.20)),
    (0.02, (0.00, 0.30, 0.36)),
    (HIGHLIGHT_RANK, (0.00, 0.42, 0.55)),
    (HIGHLIGHT_RANK + 0.03, (0.36, 0.36, 0.39)),
    (0.50, (0.46, 0.46, 0.49)),
    (1.00, (0.55, 0.55, 0.58)),
)
_STOPS: Dict[Tuple[str, bool], Stops] = {
    (MODE_NOISE_GLOW, False): _GLOW_DARK,
    (MODE_NOISE_GLOW, True): _GLOW_LIGHT,
    (MODE_NOISE_SPOT, False): _SPOT_DARK,
    (MODE_NOISE_SPOT, True): _SPOT_LIGHT,
}
_LUTS: Dict[Tuple[str, bool], np.ndarray] = {}

# 配色の読み方（説明書と、切り替えたときの案内に使う）
LEGEND_GLOW = "白〜水色 = いちばん弱い反射（空中の点が多い帯）／青〜紫 = 中くらい／暗い灰 = 強い反射（ほぼ面）"
LEGEND_SPOT = "白〜水色 = いちばん弱い 10% の反射（空中の点が多い帯）／灰色 = それ以外（明るいほど弱い）"


def is_noise_mode(mode: str) -> bool:
    return mode in NOISE_MODES


def legend_text(mode: str) -> str:
    """配色の読み方を 1 行で返す（ノイズ探索の配色でなければ空文字）"""
    if mode == MODE_NOISE_GLOW:
        return LEGEND_GLOW
    if mode == MODE_NOISE_SPOT:
        return LEGEND_SPOT
    return ""


def is_light_background(r: float, g: float, b: float) -> bool:
    """背景色（0〜1）が明るいか。明るい背景では、明暗を反転させた表を使う"""
    return (0.2126 * float(r) + 0.7152 * float(g) + 0.0722 * float(b)) > LIGHT_BACKGROUND_LUMA


def noise_lut(mode: str, light_background: bool = False) -> np.ndarray:
    """順位 0〜1 を 1024 段に分けた色の表 (1024, 3) float32"""
    key = (mode if mode in NOISE_MODES else MODE_NOISE_GLOW, bool(light_background))
    lut = _LUTS.get(key)
    if lut is None:
        stops = _STOPS[key]
        x = np.linspace(0.0, 1.0, LUT_SIZE)
        xp = [s[0] for s in stops]
        lut = np.column_stack([np.interp(x, xp, [s[1][c] for s in stops]) for c in range(3)]).astype(np.float32)
        _LUTS[key] = lut
    return lut


def rank_curve(values: Optional[np.ndarray]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    値 → 順位（0〜1）の表 (値の並び, 順位の並び) を作る。values は反射強度（正規化の前後どちらでもよい）。
    同じ値がたくさんある所（正規化で 0 や 1 に張り付いた点など）は、その値に、その帯の真ん中の順位を割り当てる。
    点が無い・値が 1 種類しか無いときは None（呼び出し側は、値をそのまま順位として使う）。
    """
    if values is None:
        return None
    v = np.asarray(values)
    if v.size == 0:
        return None
    if v.size > RANK_SAMPLE_MAX:
        v = v[:: int(np.ceil(v.size / RANK_SAMPLE_MAX))]
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None
    levels = np.linspace(0.0, 1.0, RANK_KNOTS)
    knots = np.quantile(v, levels)
    xs, first = np.unique(knots, return_index=True)
    if len(xs) < 2:
        return None
    last = np.append(first[1:], len(knots)) - 1
    ys = (levels[first] + levels[last]) * 0.5
    return xs.astype(np.float64), ys.astype(np.float64)


def to_rank(values: np.ndarray, curve: Optional[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """値を順位 0〜1 にする。表が無ければ、値を 0〜1 に切り詰めて順位の代わりにする"""
    v = np.asarray(values, dtype=np.float32)
    if curve is None:
        return np.clip(np.nan_to_num(v, nan=0.0), 0.0, 1.0)
    xs, ys = curve
    return np.interp(np.nan_to_num(v, nan=float(xs[0])), xs, ys).astype(np.float32)


def colorize(
    values: np.ndarray,
    mode: str,
    curve: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    light_background: bool = False,
) -> np.ndarray:
    """反射強度の値（正規化済みでもよい）を、ノイズ探索の配色で (N, 3) float32 の色にする"""
    if len(values) == 0:
        return np.empty((0, 3), dtype=np.float32)
    rank = to_rank(values, curve)
    idx = np.clip((rank * (LUT_SIZE - 1)).astype(np.int32), 0, LUT_SIZE - 1)
    return noise_lut(mode, light_background)[idx].copy()


def color_table(
    mode: str,
    curve: Optional[Tuple[np.ndarray, np.ndarray]],
    light_background: bool = False,
    size: int = TABLE_SIZE,
) -> np.ndarray:
    """
    正規化済みの反射強度（0〜1）を size 段に分けた「強度 → 色」の表 (size, 3) float32。
    順位の変換（curve）と色の表を、先に 1 つへまとめておく。数千万点に色を付けるとき、点ごとに順位を補間せずに済み、
    ほかの配色と同じ速さ（表を引くだけ）になる（4,521 万点で、切り替えが 2.3 秒 → 1.1 秒）。
    """
    t = np.linspace(0.0, 1.0, int(size), dtype=np.float32)
    return colorize(t, mode, curve, light_background)


def apply_table(t: np.ndarray, table: np.ndarray) -> np.ndarray:
    """正規化済みの反射強度 t（0〜1）を、color_table() の表で (N, 3) float32 の色にする"""
    if len(t) == 0:
        return np.empty((0, 3), dtype=np.float32)
    last = len(table) - 1
    idx = np.clip((np.asarray(t, dtype=np.float32) * last + 0.5).astype(np.int32), 0, last)
    return table[idx]


def relative_luminance(rgb: np.ndarray) -> np.ndarray:
    """sRGB（0〜1）の色の相対輝度（WCAG の定義）。配色の見えやすさの検査に使う"""
    c = np.asarray(rgb, dtype=np.float64)
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return 0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2]


def contrast_ratio(rgb: np.ndarray, background: Sequence[float]) -> np.ndarray:
    """色と背景色のコントラスト比（WCAG。1 = 同じ、3 以上で図として見分けられる目安）"""
    la = relative_luminance(rgb)
    lb = float(relative_luminance(np.asarray(background, dtype=np.float64)))
    hi = np.maximum(la, lb)
    lo = np.minimum(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def stop_ranks(mode: str, light_background: bool = False) -> List[float]:
    """色の節目の順位（テストと説明用）"""
    key = (mode if mode in NOISE_MODES else MODE_NOISE_GLOW, bool(light_background))
    return [float(s[0]) for s in _STOPS[key]]
