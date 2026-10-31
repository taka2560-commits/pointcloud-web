# -*- coding: utf-8 -*-
"""
計測ツール（M）の座標読み取り: 表示・ステータス・クリップボード用テキストの生成

GUI に依存しない純粋関数だけを置く（tests/test_measure_text.py で検証）。
- 座標は「表示系」（pose と、読み込み時の座標系変換を適用した後の値）で出す
- 座標系変換（測量 X/Y 入替など）をして読み込んだときは「ファイル系」（pose 適用後で、軸の入替をしていない値）も併記する
- E57 に保存されているスキャンのローカル座標（pose 適用前）は表示しない
- 小数 3 桁（mm 単位）。クリップボードはタブ区切りで Excel / CAD にそのまま貼れる
"""

from typing import List, Optional, Sequence

import numpy as np

from core.processor import inverse_transform_coordinate_system, is_identity_coord_mode

COORD_MODE_LABELS = {
    "cartesian": "数学座標系",
    "survey_swap": "測量座標系 (X/Y入替)",
    "survey_swap_inv": "測量座標系 (X/Y入替・Y反転)",
    "y_up_to_z_up": "Y-up → Z-up",
    "z_invert": "Z反転",
}


def file_coords(world_pt: Sequence[float], coord_mode: Optional[str]) -> np.ndarray:
    """表示系の 1 点から取り込み時の座標系変換を外し、ファイル系（pose 適用後）へ戻す"""
    pt = np.asarray(world_pt, dtype=np.float64).reshape(1, 3)
    if is_identity_coord_mode(coord_mode):
        return pt[0].copy()
    return inverse_transform_coordinate_system(pt.copy(), coord_mode)[0]


def fmt_xyz(pt: Sequence[float], decimals: int = 3) -> str:
    return f"X={pt[0]:.{decimals}f}  Y={pt[1]:.{decimals}f}  Z={pt[2]:.{decimals}f}"


def _delta(points_world: Sequence[Sequence[float]]) -> np.ndarray:
    p1 = np.asarray(points_world[0], dtype=np.float64)
    p2 = np.asarray(points_world[1], dtype=np.float64)
    return p2 - p1


def measure_distance_text(points_world: Sequence[Sequence[float]], decimals: int = 3) -> str:
    """2 点間の距離行（従来の表示と同じ形）。点が 2 つ未満なら空文字"""
    if len(points_world) < 2:
        return ""
    d = _delta(points_world)
    return (
        f"直線距離: {np.linalg.norm(d):.{decimals}f}m | 水平: {np.linalg.norm(d[:2]):.{decimals}f}m"
        f" | 高低差: {abs(d[2]):.{decimals}f}m"
    )


def measure_hud_lines(points_world: Sequence[Sequence[float]], coord_mode: Optional[str], decimals: int = 3) -> List[str]:
    """
    3D ビュー左上の HUD に出す行。
    P1 / P2 の座標（表示系。座標系変換して読み込んだときは「   ファイル系 …」の行も続ける）、
    2 点そろったら距離行と ΔX/ΔY/ΔZ 行。
    """
    lines: List[str] = []
    ident = is_identity_coord_mode(coord_mode)
    for i, p in enumerate(list(points_world)[:2], start=1):
        lines.append(f"P{i} {fmt_xyz(p, decimals)}")
        if not ident:
            lines.append(f"   ファイル系 {fmt_xyz(file_coords(p, coord_mode), decimals)}")
    if len(points_world) >= 2:
        d = _delta(points_world)
        lines.append(measure_distance_text(points_world, decimals))
        lines.append(f"ΔX={d[0]:+.{decimals}f}  ΔY={d[1]:+.{decimals}f}  ΔZ={d[2]:+.{decimals}f}")
    return lines


def measure_status_text(points_world: Sequence[Sequence[float]], coord_mode: Optional[str], decimals: int = 3) -> str:
    """ステータスバー用の 1 行"""
    n = len(points_world)
    if n == 0:
        return "計測モード: 3D画面の点をクリックすると座標を表示します（2点目で距離、Ctrl+C で座標をコピー）"
    if n == 1:
        return f"P1 {fmt_xyz(points_world[0], decimals)}（2点目をクリックで距離 / Ctrl+C で座標をコピー）"
    return measure_distance_text(points_world, decimals) + "（Ctrl+C で座標と距離をコピー）"


def measure_clipboard_text(points_world: Sequence[Sequence[float]], coord_mode: Optional[str], decimals: int = 3) -> str:
    """
    Excel / CAD に貼れるタブ区切り。
    1 行目はヘッダー、以下 P1 / P2 の座標（表示系。必要ならファイル系の 3 列を追加）、
    2 点そろっていれば距離の行。
    """
    ident = is_identity_coord_mode(coord_mode)
    header = ["点", "X", "Y", "Z"]
    if not ident:
        header += ["ファイルX", "ファイルY", "ファイルZ"]
    rows: List[List[str]] = [header]
    for i, p in enumerate(list(points_world)[:2], start=1):
        row = [f"P{i}"] + [f"{float(v):.{decimals}f}" for v in np.asarray(p, dtype=np.float64)]
        if not ident:
            row += [f"{float(v):.{decimals}f}" for v in file_coords(p, coord_mode)]
        rows.append(row)
    if len(points_world) >= 2:
        d = _delta(points_world)
        rows.append([
            "直線距離", f"{np.linalg.norm(d):.{decimals}f}",
            "水平距離", f"{np.linalg.norm(d[:2]):.{decimals}f}",
            "高低差", f"{abs(d[2]):.{decimals}f}",
        ])
        rows.append(["ΔX", f"{d[0]:.{decimals}f}", "ΔY", f"{d[1]:.{decimals}f}", "ΔZ", f"{d[2]:.{decimals}f}"])
    return "\n".join("\t".join(r) for r in rows) + "\n"
