# -*- coding: utf-8 -*-
"""
連結塊の追跡（マジックワンド）: 選んだ点から、つながっている塊ぜんたいを取り出す。Qt に依存しない。

なぜ作り直したか（2026-11-14、利用者の報告「連結塊の機能を使うと途中でフリーズした」）
    以前の作り（core/filters.py: extract_connected_cluster_from_seeds の中身）は、選べる点の全部に対して
      1. 地面の抽出  2. 近傍探索の木（KD 木）の構築  3. 点を 1 つずつたどる幅優先探索（Python のループ）
    を行っていた。利用者の実データ（5,490 万点）で測ると、1 と 2 だけで約 51 秒かかり、そのあいだ進み具合は出ず、
    中止も効かなかった。3 は、大きな構造物につながる所を選ぶと 92 秒で 174 万点しか進まず、打ち切りの上限は
    全体の 45%（2,470 万点）だった。数十分かかる計算で、利用者にはフリーズに見えた。

答えは以前と同じ（連結距離 eps 以内の点を次々にたどった範囲）。変えたのは、求め方だけ。
    1. 窓を切る: 選んだ点のまわり（最初は ±4m か ±30×連結距離の大きいほう）に入る点だけを相手にする。
       通行人・車・看板・浮遊物など、ふつうの塊はこの中で終わる。塊が窓の端に届いたら、窓を 2 倍に広げてやり直す。
    2. 格子に入れる: 1 辺が eps/1.75 の格子（ボクセル）に点を入れる。対角線が eps より短いので、
       同じ格子の点は、必ずたがいに eps 以内にある（= 格子の中は、調べなくても全部つながっている）。
       eps 以内にある 2 点は、同じ格子か、2 つ隣までの格子にある。
    3. 格子の単位で広げる（確かな近道）: 隣の格子の代表点（その格子の点のどれか 1 つ）どうしが eps 以内なら、
       2 つの格子はつながっている。1 段ぶんを numpy でまとめて処理する（Python の点ごとのループは無い）。
       密な面や物体は、ほとんどこれだけで広がる。
    4. 行き止まりで確かめる: 近道でつながらなかった隣の格子は、保留にしておく。広がりが止まったら、
       保留の格子の点と、塊の側の点との距離を、その場の小さな KD 木で正確に調べ、eps 以内ならつなぐ。
       つながった格子から、また 3 を続ける。保留が無くなるまでくり返す。
       3 は「つながっている」としか言わず、4 は正確なので、答えは点を 1 つずつたどる方法と同じになる。
    5. 塊の点数が上限（選べる点の max_cluster_ratio 倍）を超えたら打ち切り、capped=True を返す
       （建物や地面までつながってしまった合図。呼び出し側は選択を変えず、利用者に知らせる）。
    進み具合は progress_callback で知らせ、is_cancelled が True になったら InterruptedError で止まる。

    地面の判定（地面を伝って建物ぜんたいへ広がるのを防ぐ）は、窓の XY の範囲に入る点を、上下で切らずに使う。
    空中の高い所の塊を選んだとき、窓の中のいちばん低い点（塊の底）を地面と見まちがえないため。
    窓の XY は地面の格子にそろえて切る（格子の途中で切ると、その格子の最低点が変わってしまう）。

    格子だけでつなぐ方法（CloudCompare の連結成分の抽出と同じ考え方）も試したが、採らなかった。
    速いが、eps 〜 2.6×eps 離れた点までつなぐことがあり、実データで「以前は 8 点 → 990 点」のように結果が変わった。
    壁のそばのノイズを選んだときに壁まで巻き込むおそれがあるので、答えを変えない作りにした。

    実データ（5,490 万点）での実測は QA_REPORT_2026-10-31.md（2026-11-14 の行）。
"""

import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

GROUND_GRID_MIN = 0.8             # 地面の抽出の格子（m）。以前の作りと同じ値
GROUND_GRID_EPS = 4.0             # 地面の抽出の格子（連結距離の何倍か）。以前の作りと同じ値
GROUND_HEIGHT = 0.12              # 局所の最低点からこの高さまでを地面とみなす（m）。以前の作りと同じ値
GROUND_MIN_POINTS = 50            # これ以下の点数では、地面の判定をしない。以前の作りと同じ値
GROUND_DENSE_CELLS = 16_000_000   # 地面の格子がこの数までなら、格子の表を直接持つ（速い）。超えたら並べ替えで求める
INITIAL_MARGIN_MIN = 4.0          # 最初の窓の余白（m）
INITIAL_MARGIN_EPS = 30.0         # 最初の窓の余白（連結距離の何倍か）
MARGIN_GROWTH = 2.0               # 塊が窓の端に届いたとき、余白を何倍にするか
CELL_PER_EPS = 1.75               # 連結距離を何個の格子に分けるか（√3 = 1.732 より大きいこと: 同じ格子の点が必ず eps 以内になる）
NEIGHBOR_CELLS = 2                # eps 以内の点がありうるのは、何個隣の格子までか（= CELL_PER_EPS の切り上げ）
SCAN_CHUNK = 8_000_000            # 窓に入る点を探すときの、1 回に見る点数
PROGRESS_INTERVAL = 0.15          # 進み具合を知らせる間隔（秒）
PARALLEL_QUERY_POINTS = 200_000   # 行き止まりで確かめる点がこの数を超えたら、KD 木の問い合わせを並列にする
_BITS = 21                        # 格子の番号 1 軸ぶんのビット数（3 軸で 63 ビット）
_PAD = NEIGHBOR_CELLS + 1         # 格子の番号に足す下駄（隣の番号を引いても負にならないようにする）
_MAX_INDEX = (1 << _BITS) - 2 * _PAD

CANCEL_TEXT = "処理がユーザーによって中止されました。"

# 作りの前提（崩すと、答えが変わる）: 同じ格子の点は必ず eps 以内にあり、eps 以内の点は必ず NEIGHBOR_CELLS 個隣までの格子にある
assert CELL_PER_EPS > 3.0 ** 0.5, "格子の対角線が eps を超えると、離れた点までつないでしまう"
assert NEIGHBOR_CELLS >= CELL_PER_EPS, "eps 以内の点が、調べる範囲の外の格子に入ってしまう"


@dataclass
class ClusterGrowth:
    """連結塊の追跡の結果"""
    indices: np.ndarray        # 塊に入る点の番号（渡した points の番号。昇順）
    capped: bool = False       # 点数の上限を超えて打ち切った（建物や地面までつながっている）
    n_seed: int = 0            # 起点にした点数（選べない点を除いたあと）
    limit: int = 0             # 点数の上限
    rounds: int = 0            # 窓を広げた回数
    window_points: int = 0     # 最後の窓に入っていた点数
    margin: float = 0.0        # 最後の窓の余白（m）
    seconds: float = 0.0       # かかった時間
    checks: int = 0            # 行き止まりで確かめた回数（最後の窓）


class _Ticker:
    """進み具合の知らせと、中止の確認（知らせるのは PROGRESS_INTERVAL 秒に 1 回まで）"""

    def __init__(self, progress_callback, is_cancelled, max_pts: int):
        self._progress = progress_callback
        self._cancelled = is_cancelled
        self._max_pts = max(1, int(max_pts))
        self._last = 0.0
        self._best = 0.0
        self.rounds = 0
        self.margin = 0.0
        self.checks = 0

    def check(self) -> None:
        if self._cancelled is not None and self._cancelled():
            raise InterruptedError(CANCEL_TEXT)

    def say(self, ratio: float, message: str, force: bool = False) -> None:
        self.check()
        if self._progress is None:
            return
        now = time.perf_counter()
        if force or now - self._last >= PROGRESS_INTERVAL:
            self._last = now
            self._best = max(self._best, max(0.0, min(0.97, ratio)))   # 窓を広げてやり直しても、進み具合の値は戻さない
            self._progress(self._best, message)

    def stage(self, text: str) -> None:
        """段取りの切り替わり（点を集める・地面を見分ける・格子に分ける）を、必ず知らせる"""
        self.say(0.12 + 0.03 * min(self.rounds, 10), f"{text}（まわり ±{self.margin:g}m）", force=True)

    def growing(self, total: int) -> None:
        # 窓を 3 回広げても収まらないのは、大きな物（建物・地面）につながっているとき。文は進捗の窓の幅に収まる長さにする
        text = "大きな範囲をたどっています..." if self.rounds >= 3 else "つながりをたどっています..."
        self.say(
            0.15 + 0.03 * min(self.rounds, 10) + 0.5 * min(1.0, total / self._max_pts),
            f"{text} {total:,} 点（まわり ±{self.margin:g}m）",
        )


def ground_point_mask(points: np.ndarray, grid_size: float, height_threshold: float) -> np.ndarray:
    """
    局所最低標高グリッド法の判定を、真偽の配列で返す（core/filters.py: segment_ground_points の中身）。
    XY を grid_size の格子に分け、格子ごとの最低の Z から height_threshold 以内の点を地面とする。
    """
    n = 0 if points is None else len(points)
    if n == 0:
        return np.zeros(0, dtype=bool)
    grid_size = max(0.05, float(grid_size))
    height_threshold = max(0.001, float(height_threshold))
    z = points[:, 2]
    cx = np.floor(points[:, 0] / grid_size).astype(np.int64)
    cy = np.floor(points[:, 1] / grid_size).astype(np.int64)
    nx = int(cx.max()) - int(cx.min()) + 1
    ny = int(cy.max()) - int(cy.min()) + 1
    if nx <= 0 or ny <= 0 or nx * ny >= (1 << 62):
        # 座標が壊れている（極端に大きい・数でない）ときは、番号をまとめずに、行ごとに見分ける（遅いが、必ず動く）
        _, inv = np.unique(np.stack([cx, cy], axis=1), axis=0, return_inverse=True)
        inv = np.asarray(inv).ravel()
        n_cells = int(inv.max()) + 1
    else:
        cx -= cx.min()
        cy -= cy.min()
        cx *= ny
        cx += cy
        inv, n_cells = cx, nx * ny
        if n_cells > GROUND_DENSE_CELLS:
            _, inv = np.unique(inv, return_inverse=True)
            inv = np.asarray(inv).ravel()
            n_cells = int(inv.max()) + 1
    del cx, cy
    cell_min = np.full(n_cells, np.inf, dtype=np.float64)
    np.minimum.at(cell_min, inv, z)
    return (z - cell_min[inv]) <= height_threshold


def _grid_edge(k: float, grid: float) -> float:
    """
    floor(x / grid) >= k になる、いちばん小さい x。
    k * grid のままだと、丸めのせいで格子の番号と食い違うことがある（0.8m の格子で floor(13.6 / 0.8) は 17 なのに、
    17 * 0.8 は 13.600000000000001）。窓に入るかどうかと、地面の格子の番号を、必ず一致させるために使う。
    """
    x = float(k) * grid
    while np.floor(x / grid) >= k:
        x = float(np.nextafter(x, -np.inf))
    while np.floor(x / grid) < k:
        x = float(np.nextafter(x, np.inf))
    return x


def _column_indices(points: np.ndarray, lo: np.ndarray, hi: np.ndarray, selectable: Optional[np.ndarray],
                    ticker: _Ticker) -> np.ndarray:
    """
    窓の XY の範囲（lo 以上・hi 未満）に入る、選べる点の番号（昇順。上下では切らない）。
    座標が数でない点（NaN）は入れない（高さが NaN の点が 1 つあると、その格子の「最低の高さ」が NaN になり、地面の判定が狂う）。
    """
    parts = []
    n = len(points)
    for a in range(0, n, SCAN_CHUNK):
        ticker.check()
        p = points[a:a + SCAN_CHUNK]
        m = p[:, 0] >= lo[0]
        m &= p[:, 0] < hi[0]
        m &= p[:, 1] >= lo[1]
        m &= p[:, 1] < hi[1]
        m &= np.isfinite(p[:, 2])
        if selectable is not None:
            m &= selectable[a:a + SCAN_CHUNK]
        idx = np.flatnonzero(m)
        if len(idx):
            parts.append(idx + a)
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)


def _voxel_keys(pts: np.ndarray, k0: np.ndarray, cell: float) -> np.ndarray:
    """
    点ごとの、格子の番号をまとめた値（x・y・z の番号を 21 ビットずつ並べる）。
    格子は座標の原点にそろえて切る（floor(座標 / cell)）。k0 = 窓の下の端がある格子の番号（番号を小さくするために引く）。
    """
    key = np.floor(pts[:, 0] / cell).astype(np.int64)
    key += _PAD - int(k0[0])
    key <<= _BITS
    q = np.floor(pts[:, 1] / cell).astype(np.int64)
    q += _PAD - int(k0[1])
    key |= q
    key <<= _BITS
    q = np.floor(pts[:, 2] / cell).astype(np.int64)
    q += _PAD - int(k0[2])
    key |= q
    return key


class _Cells:
    """
    窓の中の点を、格子ごとにまとめたもの。
    uk = 点のある格子の番号（昇順）、order = 点を格子の順に並べたときの元の位置、
    starts / counts = 格子ごとの、order の中での始まりと点数、rep = 格子ごとの代表点（その格子の点のどれか 1 つ）。
    """

    def __init__(self, pts: np.ndarray, keys: np.ndarray):
        self.pts = pts
        order = np.argsort(keys)
        ks = keys[order]
        n = len(ks)
        new = np.empty(n, dtype=bool)
        new[0] = True
        np.not_equal(ks[1:], ks[:-1], out=new[1:])
        self.order = order
        self.starts = np.flatnonzero(new)
        self.uk = ks[self.starts]
        self.counts = np.diff(np.append(self.starts, n))
        self.rep = order[self.starts]
        self.n = len(self.uk)
        r = range(-NEIGHBOR_CELLS, NEIGHBOR_CELLS + 1)
        # 柱 (x+dx, y+dy) の、高さ z-2 の格子の番号との差（25 本ぶん）
        self.column_offsets = [(dx << (2 * _BITS)) + (dy << _BITS) - NEIGHBOR_CELLS for dx in r for dy in r]
        self.span = 2 * NEIGHBOR_CELLS

    def neighbors(self, cells: np.ndarray):
        """
        cells（格子の位置の配列）それぞれの、まわり 5×5×5 にある「点のある格子」を、まとめて順に返す。
        戻り値は (src, nb) の組の列: src = cells の中での番号、nb = 隣の格子の位置（昇順）。自分自身も含む。

        uk は (x, y, z) の順に並んでいるので、同じ (x, y) の柱の中では z の順に続いている。
        そこで、まわりの 25 本の柱ごとに「z-2 以上の最初の格子」を二分探索で 1 回だけ探し、
        そこから続く最大 5 個（z-2 〜 z+2）を順に見る。隣の 124 個を 1 つずつ探すより、ずっと速い。
        """
        uk, n_vox, span = self.uk, self.n, self.span
        keys = uk[cells]
        src0 = np.arange(len(cells))
        for off in self.column_offsets:
            base = keys + off
            p = np.searchsorted(uk, base)
            top = base + span
            src = src0
            for _ in range(span + 1):
                inside = p < n_vox
                if not inside.all():
                    p, top, src = p[inside], top[inside], src[inside]
                if len(p) == 0:
                    break
                hit = uk[p] <= top
                if not hit.all():
                    p, top, src = p[hit], top[hit], src[hit]
                if len(p) == 0:
                    break
                yield src, p
                p = p + 1

    def point_rows(self, cells: np.ndarray):
        """cells に入る点の座標と、点ごとの「cells の中での番号」"""
        c = self.counts[cells]
        total = int(c.sum())
        owner = np.repeat(np.arange(len(cells)), c)
        first = np.cumsum(c) - c
        pos = np.arange(total) - np.repeat(first, c) + np.repeat(self.starts[cells], c)
        return self.pts[self.order[pos]], owner

    def member_mask(self, visited: np.ndarray) -> np.ndarray:
        """たどり着いた格子（visited）に入る点の、pts の中での真偽の配列"""
        member = np.empty(len(self.pts), dtype=bool)
        member[self.order] = np.repeat(visited, self.counts)
        return member


def _resolve_pending(cells: _Cells, visited: np.ndarray, pending: np.ndarray, eps: float, ticker: _Ticker) -> np.ndarray:
    """
    行き止まりで確かめる: 保留の格子（近道ではつながらなかった、まだたどり着いていない隣の格子）のうち、
    塊の側の点から eps 以内に点があるものを返す。点どうしの距離を、その場の小さな KD 木で正確に調べる。
    """
    cand = np.flatnonzero(pending)
    if len(cand) == 0:
        return cand
    pending[cand] = False
    cand = cand[~visited[cand]]
    if len(cand) == 0:
        return cand
    ticker.check()
    near_parts = [nb[visited[nb]] for _src, nb in cells.neighbors(cand)]
    near_parts = [p for p in near_parts if len(p)]
    if not near_parts:
        return cand[:0]
    cluster_side = np.unique(np.concatenate(near_parts))
    from scipy.spatial import cKDTree

    side_pts, _ = cells.point_rows(cluster_side)
    cand_pts, owner = cells.point_rows(cand)
    ticker.check()
    ticker.checks += 1
    tree = cKDTree(side_pts)
    workers = -1 if len(cand_pts) > PARALLEL_QUERY_POINTS else 1
    dist, nearest = tree.query(cand_pts, k=1, distance_upper_bound=eps * 1.000001, workers=workers)
    hit = np.flatnonzero(np.isfinite(dist))
    # eps ちょうどの境目は、近道と同じ式（差の 2 乗の和 <= eps の 2 乗）で決める（式が違うと、境目の 1 点で答えがずれる）
    d = cand_pts[hit] - side_pts[nearest[hit]]
    hit = hit[np.einsum("ij,ij->i", d, d) <= eps * eps]
    ok = np.zeros(len(cand), dtype=bool)
    ok[owner[hit]] = True
    return cand[ok]


def _flood(cells: _Cells, seed_vox: np.ndarray, eps: float, max_pts: int, ticker: _Ticker):
    """
    起点の格子（seed_vox）からつながる格子を求める。
    戻り値: (たどり着いた格子の真偽の配列, 点数の上限を超えて打ち切ったか)
    """
    pts, rep, counts = cells.pts, cells.rep, cells.counts
    eps2 = eps * eps
    visited = np.zeros(cells.n, dtype=bool)
    pending = np.zeros(cells.n, dtype=bool)
    frontier = np.unique(seed_vox)
    visited[frontier] = True
    total = int(counts[frontier].sum())
    while True:
        # 確かな近道: 代表点どうしが eps 以内なら、2 つの格子はつながっている
        while len(frontier) and total <= max_pts:
            ticker.growing(total)
            f_pts = pts[rep[frontier]]
            f_single = counts[frontier] == 1
            parts = []
            for src, nb in cells.neighbors(frontier):
                fresh = ~visited[nb]
                if not fresh.any():
                    continue
                b, a = nb[fresh], src[fresh]
                d = f_pts[a] - pts[rep[b]]
                near = np.einsum("ij,ij->i", d, d) <= eps2
                reached = b[near]
                if len(reached):
                    visited[reached] = True
                    parts.append(reached)
                far = ~near
                if far.any():
                    # どちらも 1 点だけの格子なら、いまの判定で確定している（あとで調べ直さない）
                    far &= ~(f_single[a] & (counts[b] == 1))
                    pending[b[far]] = True
            frontier = np.unique(np.concatenate(parts)) if parts else frontier[:0]
            total += int(counts[frontier].sum())
        if total > max_pts:
            return visited, True
        # 行き止まり: 保留の格子を、点どうしの距離で確かめる
        frontier = _resolve_pending(cells, visited, pending, eps, ticker)
        if len(frontier) == 0:
            return visited, False
        visited[frontier] = True
        total += int(counts[frontier].sum())


def _fill_window(pts: np.ndarray, seed_pos: np.ndarray, eps: float, max_pts: int,
                 lo: np.ndarray, hi: np.ndarray, check_axes, ticker: _Ticker):
    """
    窓の中の、たどってよい点（pts）で、起点（seed_pos = pts の中での位置）からつながる点を求める。
    check_axes = 窓の外に点が残っているかもしれない軸（x, y, z の真偽）。
    戻り値: (塊に入る点の、pts の中での真偽の配列, 上限で打ち切ったか, 窓の端に届いたか)
    """
    cell = eps / CELL_PER_EPS
    k0 = np.floor(lo / cell).astype(np.int64)                    # 窓の下の端がある格子の番号
    q_hi = np.floor(hi / cell).astype(np.int64) - k0 + _PAD      # 窓の上の端がある格子の番号（下駄つき）
    if int(q_hi.max()) >= _MAX_INDEX:
        raise ValueError("範囲が広すぎて、連結塊を追跡できません（起点を近くの点だけにするか、連結距離を大きくしてください）")
    ticker.stage(f"{len(pts):,} 点を格子に分けています...")
    keys = _voxel_keys(pts, k0, cell)
    seed_keys = keys[seed_pos]
    ticker.check()
    cells = _Cells(pts, keys)
    del keys
    ticker.check()
    ticker.checks = 0
    visited, capped = _flood(cells, np.searchsorted(cells.uk, seed_keys), eps, max_pts, ticker)

    # 窓の端に届いたか（届いていれば、窓の外にも続いているかもしれない）。
    # 窓の外の点は、下の側では番号 _PAD 以下、上の側では q_hi 以上の格子にある。そこへ 2 個隣で届く格子が塊にあれば「届いた」
    touches = False
    if not capped and any(check_axes):
        mask = (1 << _BITS) - 1
        vk = cells.uk[visited]
        for axis, shift in enumerate((2 * _BITS, _BITS, 0)):
            if not check_axes[axis]:
                continue
            v = (vk >> shift) & mask
            if int(v.min()) <= _PAD + NEIGHBOR_CELLS or int(v.max()) >= int(q_hi[axis]) - NEIGHBOR_CELLS:
                touches = True
                break
    return cells.member_mask(visited), capped, touches


def grow_connected_cluster(
    points: np.ndarray,
    seed_indices: np.ndarray,
    eps: float = 0.15,
    selectable_mask: Optional[np.ndarray] = None,
    exclude_ground: bool = True,
    max_cluster_ratio: float = 0.45,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    is_cancelled: Optional[Callable[[], bool]] = None,
    initial_margin: Optional[float] = None,
) -> ClusterGrowth:
    """
    起点（seed_indices）から、3D 空間でつながっている塊ぜんたいの点の番号を返す。
    「つながっている」= 連結距離 eps 以内の点を、次々にたどって行き着けること。

    :param points: (N, 3) の座標。写しを作らずに読むだけなので、数千万点でもそのまま渡してよい
    :param seed_indices: 起点の点の番号
    :param eps: 連結距離（m）
    :param selectable_mask: 相手にしてよい点（表示中・未ロック・断面の中）。None なら全部。
                            ここに無い点は、起点にもしないし、橋渡しにもならないし、結果にも入らない
    :param exclude_ground: 起点の半分以上が地面の上でなければ、地面を伝っては広げない
    :param max_cluster_ratio: 塊の点数の上限（選べる点の何割か）。超えたら打ち切って capped=True（1.0 なら打ち切らない）
    :param progress_callback: 進み具合の知らせ (0〜1, 文)。呼び出し側が、この中で中止の例外を出してもよい
    :param is_cancelled: True を返したら InterruptedError で止まる
    :param initial_margin: 最初の窓の余白（m）。None なら自動
    """
    t0 = time.perf_counter()
    empty = np.empty(0, dtype=np.int64)
    n = 0 if points is None else len(points)
    if n == 0 or seed_indices is None or len(seed_indices) == 0:
        return ClusterGrowth(empty)
    seeds = np.unique(np.asarray(seed_indices, dtype=np.int64))
    seeds = seeds[(seeds >= 0) & (seeds < n)]
    selectable = None
    if selectable_mask is not None:
        selectable = np.asarray(selectable_mask, dtype=bool)
        if len(selectable) != n:
            raise ValueError("selectable_mask の長さが点数と合いません")
        seeds = seeds[selectable[seeds]]
        if selectable.all():
            selectable = None
    if len(seeds):
        seeds = seeds[np.isfinite(np.asarray(points[seeds], dtype=np.float64)).all(axis=1)]   # 座標が数でない点は、起点にしない
    if len(seeds) == 0:
        return ClusterGrowth(empty)
    eps = max(0.01, float(eps))
    n_cand = n if selectable is None else int(np.count_nonzero(selectable))
    max_pts = max(len(seeds), int(n_cand * max(0.05, min(1.0, float(max_cluster_ratio)))))
    ticker = _Ticker(progress_callback, is_cancelled, max_pts)

    seed_pts = np.asarray(points[seeds], dtype=np.float64)
    lo0, hi0 = seed_pts.min(axis=0), seed_pts.max(axis=0)
    margin = float(initial_margin) if initial_margin else max(INITIAL_MARGIN_MIN, INITIAL_MARGIN_EPS * eps)
    ground_grid = max(GROUND_GRID_MIN, eps * GROUND_GRID_EPS)
    rounds = 0
    while True:
        ticker.rounds, ticker.margin = rounds, margin
        ticker.stage("まわりの点を集めています...")
        lo, hi = lo0 - margin, hi0 + margin
        # 窓の XY は、地面の格子の区切りにぴったり合わせる（格子の途中で切ると、その格子の最低点が変わってしまう）
        lo[:2] = [_grid_edge(np.floor(v / ground_grid), ground_grid) for v in lo[:2]]
        hi[:2] = [_grid_edge(np.floor(v / ground_grid) + 1.0, ground_grid) for v in hi[:2]]

        # 1. 窓の XY の範囲に入る点（上下では切らない。地面の判定に使う）
        col_idx = _column_indices(points, lo, hi, selectable, ticker)
        whole = len(col_idx) == n
        col_pts = points if whole else points[col_idx]
        keep = (col_pts[:, 2] >= lo[2]) & (col_pts[:, 2] <= hi[2])       # 窓の上下に入る点
        z_all = bool(keep.all())
        xy_all = len(col_idx) >= n_cand
        n_window = int(np.count_nonzero(keep))

        # 2. 地面を伝って建物ぜんたいへ広がるのを防ぐ。起点の半分以上が地面の上なら、地面も塊として扱う（以前の作りと同じ決まり）
        seed_in_col = seeds if whole else np.searchsorted(col_idx, seeds)
        if exclude_ground and n_cand > GROUND_MIN_POINTS:   # 以前の作りと同じく、選べる点の全部の点数で決める（窓の中の点数ではない）
            ticker.stage("地面を見分けています...")
            ground = ground_point_mask(col_pts, ground_grid, GROUND_HEIGHT)
            if float(np.mean(ground[seed_in_col])) < 0.5:
                ground[seed_in_col] = False          # 起点は、地面の上にあっても相手にする
                keep &= ~ground
            del ground
        ticker.check()

        # 3. 窓の中の、たどってよい点で、つながりをたどる
        if keep.all():
            pts, pts_idx, seed_pos = col_pts, col_idx, seed_in_col
        else:
            pts = col_pts[keep]
            pts_idx = col_idx[keep]
            seed_pos = np.searchsorted(pts_idx, seeds)
        del col_pts, col_idx, keep
        pts = np.asarray(pts, dtype=np.float64)
        check_axes = (not xy_all, not xy_all, not z_all)
        member, capped, touches = _fill_window(pts, seed_pos, eps, max_pts, lo, hi, check_axes, ticker)
        del pts
        if touches and not capped:
            del member, pts_idx
            margin *= MARGIN_GROWTH
            rounds += 1
            continue
        return ClusterGrowth(
            indices=np.asarray(pts_idx[member], dtype=np.int64),
            capped=bool(capped),
            n_seed=int(len(seeds)),
            limit=int(max_pts),
            rounds=rounds,
            window_points=n_window,
            margin=float(margin),
            seconds=time.perf_counter() - t0,
            checks=int(ticker.checks),
        )
