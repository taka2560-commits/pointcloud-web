# -*- coding: utf-8 -*-
"""
読み込みに要るメモリの見積もりと、PC のメモリに収まるかの判定。Qt に依存しない。

なぜ要るか（2026-11-14、利用者の PC の確認から）
    読み込み上限を「無制限」にして 5.4 億点のファイルを開くと、約 100GB のメモリが要る（PC は 32GB）。
    足りない分は Windows がディスクをメモリの代わりに使う（ページファイル）。利用者の PC では、ページファイルが
    90GB まで膨らみ、C ドライブの空きが 20GB まで減っていた。
    点数と項目の並びはファイルのヘッダに書いてあり、0.01 秒で読める。始める前に分かることなので、始める前に確かめる。

見積もりの式（1 点あたりのバイト数。core/e57_io.py の読み込みと gui/worker_thread.py: FileLoadWorker が作る配列を数えたもの）
    スキャンが持つ分   … 項目ごとの生の値（ScaledInteger は int64 の 8、Float は 4 か 8、Integer は範囲に合う最小の型）
                         + 倍精度の座標 (N, 3) の 24。倍精度の XYZ は座標の列を見るだけなので、別には数えない
    画面が持つ分       … 表示用の座標 24（スキャンが複数か、pose があるとき）、スキャン番号 4、色 12、反射強度 4、
                         中心を引いた単精度の座標 12、レイヤー番号と有効フラグ 5
    読み込み中の一時分 … 中心を引くときの倍精度の一時配列 24
    描画に使う分       … 描く点 1 つにつき、表示用の写し 24 + 点の番号 8 + GPU へ送る分 24
    見込み = max(読み込み中の山, 読み込み後に使う量) × SAFETY_FACTOR
    式が読み込みの実装とずれていないかは、tests/test_memguard.py が実際の読み込み結果の配列の大きさと突き合わせる。

判定は「PC に載っているメモリ」に対して行う（いま空いている量ではない）。
    空いている量は、開いているアプリや、いま読み込んでいる点群で刻々と変わる。同じファイルで日によって結果が変わると
    使いにくいので、載っているメモリの 3 分の 2（少なくとも 4GB は Windows とほかのアプリに残す）までを使ってよい量とする。
"""

import math
import os
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from pye57 import libe57

from core.e57_io import FieldSchema, _optimal_integer_dtype, e57_scan_layouts, subsample_step, subsampled_count
from core.processor import scan_pose_transform
from core.sysinfo import GB, memory_status

# --- 「自動最適化」の読み込み上限（載っているメモリの段階ごと。2026-11-14 まで gui/main_window.py にあった値と同じ）
AUTO_TIERS: Tuple[Tuple[float, int], ...] = ((60.0, 120_000_000), (28.0, 60_000_000))
AUTO_TIER_SMALL = 30_000_000
FALLBACK_TOTAL_BYTES = 16 * GB  # メモリの量を調べられないときの仮の値（以前と同じ 16GB）

# --- 使ってよいメモリ
RESERVE_RATIO = 1.0 / 3.0   # Windows とほかのアプリに残す割合
RESERVE_MIN_BYTES = 4 * GB  # 少なくともこれだけは残す
SAFETY_FACTOR = 1.10        # 見込みに掛ける余裕

# --- 1 点あたりのバイト数（上の説明の内訳）
XYZ_DOUBLE_BYTES = 24
DISPLAY_XYZ_BYTES = 24
SCAN_ID_BYTES = 4
COLOR_BYTES = 12
INTENSITY_BYTES = 4
CENTERED_BYTES = 12
LAYER_BYTES = 5
CENTERING_TEMP_BYTES = 24
DRAW_BYTES = 24 + 8 + 24

HEADER_TIME_LIMIT_SEC = 2.0  # フォルダ読み込みで、ヘッダを読むのに使ってよい時間（超えたら残りは大きさから推定）
_CARTESIAN = ("cartesianX", "cartesianY", "cartesianZ")
_COLOR = ("colorRed", "colorGreen", "colorBlue")


def auto_load_limit(total_bytes: Optional[int]) -> int:
    """「自動最適化」のときの読み込み上限（点数）。載っているメモリの量で決める"""
    total_gb = float(total_bytes if total_bytes else FALLBACK_TOTAL_BYTES) / GB
    for min_gb, points in AUTO_TIERS:
        if total_gb >= min_gb:
            return points
    return AUTO_TIER_SMALL


def memory_limit_bytes(total_bytes: int) -> int:
    """読み込みに使ってよいメモリ（載っている量の 3 分の 2。少なくとも 4GB は残すが、半分は必ず使える）"""
    total_bytes = max(0, int(total_bytes))
    reserve = max(RESERVE_MIN_BYTES, int(total_bytes * RESERVE_RATIO))
    return max(total_bytes // 2, total_bytes - reserve)


def scan_bytes_per_point(schemas: Dict[str, FieldSchema]) -> int:
    """スキャンが持つ配列（E57ScanData の points / raw_fields / raw_unscaled_fields）の 1 点あたりのバイト数"""
    has_xyz = all(k in schemas for k in _CARTESIAN)
    total = 0
    for name, sc in schemas.items():
        is_xyz = has_xyz and name in _CARTESIAN
        if sc.node_type == "Float":
            if sc.precision == libe57.E57_SINGLE:
                total += 4
            elif not is_xyz:
                total += 8
        elif sc.node_type == "ScaledInteger":
            total += 8
            if not is_xyz:
                total += 8
        elif sc.node_type == "Integer":
            total += int(np.dtype(_optimal_integer_dtype(int(sc.minimum), int(sc.maximum))).itemsize)
    if has_xyz or "sphericalRange" in schemas:
        total += XYZ_DOUBLE_BYTES
    return total


@dataclass(frozen=True)
class ScanCost:
    """1 スキャンの、読み込みに要るメモリの材料"""
    n_points: int          # ファイルにある点数
    bytes_per_point: int   # スキャンが持つ配列の 1 点あたり
    has_color: bool
    has_intensity: bool
    pose_identity: bool    # pose が無いか、恒等変換（表示用の座標を別に持たない）


@dataclass(frozen=True)
class FileCost:
    path: str
    scans: Tuple[ScanCost, ...]
    estimated: bool = False  # ヘッダを読まず、ファイルの大きさから推定した

    @property
    def n_points(self) -> int:
        return sum(s.n_points for s in self.scans)


def read_file_cost(path: str) -> FileCost:
    """ファイルのヘッダだけを読んで、メモリの見積もりの材料を返す（点は読まない）"""
    scans: List[ScanCost] = []
    for lay in e57_scan_layouts(path):
        pose = scan_pose_transform(SimpleNamespace(rotation=lay.rotation, translation=lay.translation))
        scans.append(
            ScanCost(
                n_points=int(lay.n_points),
                bytes_per_point=scan_bytes_per_point(lay.schemas),
                has_color=all(k in lay.schemas for k in _COLOR),
                has_intensity="intensity" in lay.schemas,
                pose_identity=pose is None,
            )
        )
    return FileCost(path=path, scans=tuple(scans))


def read_file_costs(
    paths: Sequence[str],
    time_limit: float = HEADER_TIME_LIMIT_SEC,
    clock: Callable[[], float] = time.monotonic,
    reader: Callable[[str], FileCost] = read_file_cost,
) -> List[FileCost]:
    """
    複数のファイルのヘッダを読む。time_limit 秒を超えたら、残りは読めた分の平均から推定する
    （ネットワークの先のフォルダに数百ファイルあっても、画面を長く止めない）。
    1 つも読めないうちに時間が切れることは無い（最初の 1 つは必ず読む）。
    読めた分がどれも 0 点で推定の材料が無いときは、時間が切れていても残りを読む。
    """
    out: List[FileCost] = []
    start = clock()
    rest: List[str] = []
    for i, p in enumerate(paths):
        if i > 0 and clock() - start > time_limit:
            rest = list(paths[i:])
            break
        out.append(reader(p))
    if not rest:
        return out

    read_points = sum(f.n_points for f in out)
    read_bytes = 0
    for f in out:
        try:
            read_bytes += os.path.getsize(f.path)
        except OSError:
            pass
    known = [s for f in out for s in f.scans if s.n_points > 0]
    if not known or read_points <= 0 or read_bytes <= 0:
        # 読めた分がどれも 0 点で、大きさからの推定ができない。残りを見積もりから落とすと少なく見積もってしまうので、
        # 時間はかかっても残りのヘッダを読む
        return out + [reader(p) for p in rest]
    points_per_byte = read_points / read_bytes
    avg_bpp = int(math.ceil(sum(s.bytes_per_point * s.n_points for s in known) / read_points))
    has_color = any(s.has_color for s in known)
    has_intensity = any(s.has_intensity for s in known)
    for p in rest:
        try:
            n = int(os.path.getsize(p) * points_per_byte)
        except OSError:
            n = 0
        out.append(FileCost(p, (ScanCost(n, avg_bpp, has_color, has_intensity, False),), estimated=True))
    return out


@dataclass(frozen=True)
class LoadPlan:
    """ある読み込み上限で読んだときの見込み"""
    n_files: int
    n_total: int        # ファイルにある点数の合計
    n_loaded: int       # 読み込む点数
    n_drawn: int        # そのうち描く点数（描画点数の設定で決まる）
    max_step: int       # いちばん粗い間引きの間隔（1 = 全点）
    need_bytes: int     # 要るメモリの見込み（余裕を含む）
    base_bytes: int     # 読み込みが作って持ち続ける配列の合計（描画とレイヤーの分を除く。テストが実物と突き合わせる）
    peak_bytes: int     # 読み込み中の山（余裕を含まない）
    estimated: bool     # 一部のファイルは、大きさから推定した

    @property
    def subsampled(self) -> bool:
        return self.max_step > 1


def plan_load(files: Sequence[FileCost], max_load_points: Optional[int], display_budget: int) -> LoadPlan:
    """
    読み込み上限 max_load_points（None か 0 以下で全点）と描画点数 display_budget（0 以下で全点表示）で
    読み込んだときの、点数とメモリの見込み。間引きの決め方は FileLoadWorker と同じ
    （上限をファイル数で割り、ファイルごとに間隔を決める）。
    """
    n_files = len(files)
    limited = max_load_points is not None and int(max_load_points) > 0
    per_file = max(1, int(max_load_points) // max(1, n_files)) if limited else None

    loaded: List[Tuple[int, ScanCost]] = []
    max_step = 1
    for f in files:
        step = subsample_step(f.n_points, per_file)
        max_step = max(max_step, step)
        for s in f.scans:
            n = subsampled_count(s.n_points, step)
            if n > 0:
                loaded.append((n, s))

    n_loaded = sum(n for n, _ in loaded)
    n_total = sum(f.n_points for f in files)
    estimated = any(f.estimated for f in files)
    if n_loaded == 0:
        return LoadPlan(n_files, n_total, 0, 0, max_step, 0, 0, 0, estimated)

    scan_bytes = sum(n * s.bytes_per_point for n, s in loaded)
    # 表示用の座標: スキャンが 1 つで pose が恒等なら、スキャンの座標をそのまま使う（別には持たない）
    separate_xyz = len(loaded) > 1 or not loaded[0][1].pose_identity
    per_point = SCAN_ID_BYTES + COLOR_BYTES + CENTERED_BYTES
    if separate_xyz:
        per_point += DISPLAY_XYZ_BYTES
    if any(s.has_intensity for _, s in loaded):
        per_point += INTENSITY_BYTES
    base = scan_bytes + n_loaded * per_point

    if display_budget is None or display_budget <= 0 or n_loaded <= display_budget:
        n_drawn = n_loaded
    else:
        n_drawn = subsampled_count(n_loaded, int(math.ceil(n_loaded / display_budget)))

    peak = base + n_loaded * CENTERING_TEMP_BYTES
    working = base + n_loaded * LAYER_BYTES + n_drawn * DRAW_BYTES
    need = int(max(peak, working) * SAFETY_FACTOR)
    return LoadPlan(n_files, n_total, n_loaded, n_drawn, max_step, need, base, peak, estimated)


def nice_floor_points(n: int) -> int:
    """点数を、読みやすい区切りへ切り下げる（1 億以上は 1,000 万点、1,000 万以上は 100 万点、100 万以上は 10 万点の単位）"""
    n = max(0, int(n))
    for threshold, unit in ((100_000_000, 10_000_000), (10_000_000, 1_000_000), (1_000_000, 100_000)):
        if n >= threshold:
            return (n // unit) * unit
    return n


def fit_max_points(files: Sequence[FileCost], display_budget: int, limit_bytes: int) -> int:
    """
    見込みが limit_bytes 以下に収まる、いちばん大きい読み込み上限（読みやすい区切りへ切り下げる。最低 1 点）。
    上限を大きくすると読み込む点数は増えるか変わらないので、二分探索で求める。
    """
    hi = max(1, sum(f.n_points for f in files))
    if plan_load(files, hi, display_budget).need_bytes <= limit_bytes:
        return hi
    lo = 1  # lo は「収まる」側（1 点も収まらないことは実際には無い）
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if plan_load(files, mid, display_budget).need_bytes <= limit_bytes:
            lo = mid
        else:
            hi = mid
    return max(1, nice_floor_points(lo))


@dataclass(frozen=True)
class LoadCheck:
    """読み込む前の判定の結果"""
    ok: bool
    plan: LoadPlan                      # 頼まれた設定での見込み
    limit_bytes: int                    # 読み込みに使ってよいメモリ
    total_bytes: int                    # 載っているメモリ
    avail_bytes: Optional[int]          # いま空いているメモリ（知らせるためだけに使う。判定には使わない）
    recommended_points: Optional[int]   # 収まらないとき: おすすめの読み込み上限
    recommended_plan: Optional[LoadPlan]


def check_load(
    files: Sequence[FileCost],
    max_load_points: Optional[int],
    display_budget: int,
    memory: Optional[Tuple[int, int]] = None,
) -> Optional[LoadCheck]:
    """
    読み込みがメモリに収まるかを判定する。メモリの量を調べられないときは None（止めない）。
    収まらないときのおすすめは、「自動最適化」の上限と、収まる上限の小さいほう。
    """
    mem = memory if memory is not None else memory_status()
    if mem is None or mem[0] <= 0:
        return None
    total, avail = int(mem[0]), int(mem[1])
    limit = memory_limit_bytes(total)
    plan = plan_load(files, max_load_points, display_budget)
    if plan.need_bytes <= limit:
        return LoadCheck(True, plan, limit, total, avail, None, None)
    rec = min(auto_load_limit(total), fit_max_points(files, display_budget, limit))
    return LoadCheck(False, plan, limit, total, avail, rec, plan_load(files, rec, display_budget))
