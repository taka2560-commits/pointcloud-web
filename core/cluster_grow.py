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

    塊そのものの底（2026-11-15、利用者の依頼）: 地面の判定は、下に床の無い所（空中のノイズ・現場の外・床のレイヤーを
    隠した所）では、塊の底を地面とみなしてしまい、底の 12cm が選ばれなかった。たどり終わったあとに、支えている面
    （塊の足もとの外まで続いている面）でない「地面扱いの点」を塊に足す（_own_bottom）。ここだけは、以前と答えが変わる。
    即削除モードで確かめずに消してよい大きさは too_big_for_instant_delete が決める（画面は、大きな塊を選択にとどめる）。

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
BOTTOM_SUPPORT_REACH = 0.5        # 支えている面をたどる距離（地面の格子の何倍か。0.8m の格子で 0.4m。連結距離より広く取る）
BOTTOM_Z_REACH = 1000.0           # 塊の底・足もとの面として相手にする高さの範囲（塊の上下から何 m まで）
BOTTOM_MAX_CELLS = 25_000_000     # 塊の真下を調べる格子の表の上限（これを超える広さの塊では、底を足さない）
INSTANT_DELETE_MAX_POINTS = 1_000_000   # 即削除モードで、確かめずに消してよい塊の点数の上限
INSTANT_DELETE_MAX_RATIO = 0.10         # 同じく、選べる点に対する割合の上限
INSTANT_DELETE_RATIO_FLOOR = 20_000     # 割合の上限は、この点数を超える塊にだけ当てはめる（小さな点群で、小さな塊まで止めないため）
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
    n_selectable: int = 0      # 相手にしてよい点の数（選べる点の全部）
    n_bottom: int = 0          # 「塊そのものの底」として足した点数（地面とみなされていた点）


def too_big_for_instant_delete(n_cluster: int, n_selectable: int) -> bool:
    """
    即削除モードで、確かめずに消すには大きすぎる塊か。
    100 万点を超える、または、選べる点の 1 割を超える（2 万点を超える塊に限る）とき True。
    呼び出し側は、消さずに選択にとどめて、利用者に知らせる（Del で消せる）。
    """
    n_cluster = int(n_cluster)
    if n_cluster > INSTANT_DELETE_MAX_POINTS:
        return True
    return n_cluster > INSTANT_DELETE_RATIO_FLOOR and n_cluster > int(n_selectable) * INSTANT_DELETE_MAX_RATIO


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
        self.quiet = False     # True の間は、たどった点数の文を出さない（塊の底の確かめ。数えているのが塊の点ではないため）

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
        if self.quiet:
            self.check()
            return
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


def _grid_numbers(values: np.ndarray, grid: float) -> np.ndarray:
    """地面の格子の番号（ground_point_mask と同じ式。同じ型のまま割る）"""
    return np.floor(values / grid).astype(np.int64)


def _dilate(mask: np.ndarray) -> np.ndarray:
    """真偽の表を、まわり 1 格子ぶん（8 方向）広げる"""
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    side = out.copy()
    out[:, 1:] |= side[:, :-1]
    out[:, :-1] |= side[:, 1:]
    return out


def _own_bottom(points: np.ndarray, cluster_idx: np.ndarray, band_idx: np.ndarray, eps: float, grid: float,
                window_cells, ticker: _Ticker):
    """
    塊そのものの底を見つける。

    地面の判定（格子ごとの最低点から GROUND_HEIGHT まで）は、下に床の無い所では、塊の底を「地面」とみなしてしまう
    （空中のノイズ・現場の外・床のレイヤーを隠した / ロックした所）。そこで、塊をたどり終わったあとに、次の決まりで底を足す。

      1. 足もと = 塊の点がある地面の格子。まわり = 足もと + 8 方向の隣。その外側の 1 列を「外の輪」と呼ぶ
      2. 支えている面: 「外の輪」にある地面扱いの点から、地面扱いの点だけを次々にたどって行き着ける点。
         床・地面・屋根のように、塊の足もとの外まで続いている面は、ここに入る（今までどおり、塊に足さない）。
         たどる距離は BOTTOM_SUPPORT_REACH × 格子（0.4m。eps より広い）。床の点がまばらな所や斜面でも、面を面として扱うため
      3. 塊の底: 「まわり」にある地面扱いの点のうち、支えている面に入らず、塊から eps 以内でたどれる点。これを塊に足す
         （足した点から、さらに別の物へは広げない）

    床に立つ人・車・屋根の上の物は、足もとの面が外まで続いているので、今までと同じ結果になる。
    薄く広がった塊（厚さ 12cm 以下の部分が、足もとから 1 格子より外まで続く）は、その部分が「支えている面」と区別できないので、
    今までどおり足さない（起点の半分以上がその部分にあれば、地面の決まりそのものが外れて、丸ごと選ばれる）。

    points = 全点の座標、cluster_idx = 塊の点の番号（昇順）、band_idx = 窓の柱の中で地面とみなした点の番号（昇順）、
    window_cells = 窓に入っている地面の格子の範囲 (x0, x1, y0, y1)。窓が選べる点の全部を含むときは None。
    戻り値: (足す点の番号（昇順）, 窓が足りないか)。窓が足りない = 判定に要る格子が窓の外にある（窓を広げてやり直す）。
    """
    empty = np.empty(0, dtype=np.int64)
    if len(cluster_idx) == 0 or len(band_idx) == 0:
        return empty, False
    ticker.check()
    xy = points[cluster_idx, :2]               # 列を 1 つずつ取り出すより、行ごとにまとめて取り出すほうが速い
    cx = _grid_numbers(xy[:, 0], grid)
    cy = _grid_numbers(xy[:, 1], grid)
    del xy
    x0, x1 = int(cx.min()) - 3, int(cx.max()) + 3
    y0, y1 = int(cy.min()) - 3, int(cy.max()) + 3
    nx, ny = x1 - x0 + 1, y1 - y0 + 1
    if nx * ny > BOTTOM_MAX_CELLS:
        return empty, False
    foot = np.zeros((nx, ny), dtype=bool)
    foot[cx - x0, cy - y0] = True
    near = _dilate(foot)                       # 足もと + まわり 1 格子
    wide = _dilate(near)                       # さらに外側の 1 列（外の輪）まで
    xy = points[band_idx, :2]
    bx = _grid_numbers(xy[:, 0], grid)
    by = _grid_numbers(xy[:, 1], grid)
    del xy
    inside = (bx >= x0) & (bx <= x1) & (by >= y0) & (by <= y1)
    inside[inside] = wide[bx[inside] - x0, by[inside] - y0]
    if inside.any():
        # 塊の高さから大きく外れた点（壊れた座標の外れ値）は相手にしない。塊の底にも、塊の足もとの面にもなりえず、
        # 残すと、たどる範囲が広すぎて格子に分けられなくなる
        cz = points[cluster_idx, 2]
        bz = points[band_idx[inside], 2]
        inside[inside] = (bz >= float(cz.min()) - BOTTOM_Z_REACH) & (bz <= float(cz.max()) + BOTTOM_Z_REACH)
        del cz, bz
    g_idx = band_idx[inside]                   # 「まわり」と「外の輪」にある、地面扱いの点
    if len(g_idx) == 0:
        return empty, False
    gx, gy = bx[inside] - x0, by[inside] - y0
    del bx, by, inside
    in_near = near[gx, gy]
    pos = np.minimum(np.searchsorted(cluster_idx, g_idx), len(cluster_idx) - 1)
    cand = in_near & (cluster_idx[pos] != g_idx)        # 足す候補（すでに塊に入っている起点は除く）
    if not cand.any():
        return empty, False
    ticker.check()

    # 支えている面: 外の輪から、地面扱いの点だけを伝って行き着ける点
    reach = max(eps, grid * BOTTOM_SUPPORT_REACH)

    pts_g = []

    def reached_from(start: np.ndarray) -> np.ndarray:
        if not pts_g:
            pts_g.append(np.asarray(points[g_idx], dtype=np.float64))
        p = pts_g[0]
        out, _capped, _touches = _fill_window(p, start, reach, len(p) + 1, p.min(axis=0) - reach, p.max(axis=0) + reach,
                                              (False, False, False), ticker)
        return out

    if not in_near.all():
        cand &= ~reached_from(np.flatnonzero(~in_near))
        if not cand.any():
            return empty, False
    extra = g_idx[cand]

    # 塊の底: 候補のうち、塊から eps 以内でたどれる点（eps は格子の 1/4 以下なので、相手は同じ格子か隣の格子にいる）
    has = np.zeros((nx, ny), dtype=bool)
    has[gx[cand], gy[cand]] = True
    local = cluster_idx[_dilate(has)[cx - x0, cy - y0]]
    if len(local) == 0:
        return empty, False
    pts = np.concatenate([np.asarray(points[local], dtype=np.float64), np.asarray(points[extra], dtype=np.float64)])
    member, _capped, _touches = _fill_window(pts, np.arange(len(local)), eps, len(pts) + 1,
                                             pts.min(axis=0) - eps, pts.max(axis=0) + eps, (False, False, False), ticker)
    del pts
    got = extra[member[len(local):]]           # 昇順のまま
    if len(got) == 0:
        return empty, False

    # 窓の外の格子の点は見えていない。足す点が、窓の外と隣り合う格子の「地面扱いの点」とつながっていたら、
    # その先に面が続いているかどうかが分からない（窓を広げてやり直す）
    if window_cells is not None:
        wx0, wx1, wy0, wy1 = window_cells
        unknown = np.ones((nx, ny), dtype=bool)
        a0, a1, b0, b1 = max(x0, wx0), min(x1, wx1), max(y0, wy0), min(y1, wy1)
        if a0 <= a1 and b0 <= b1:
            unknown[a0 - x0:a1 - x0 + 1, b0 - y0:b1 - y0 + 1] = False
        if unknown.any():
            at_edge = np.flatnonzero(_dilate(unknown)[gx, gy])
            if len(at_edge):
                maybe_support = reached_from(at_edge)
                if maybe_support[np.searchsorted(g_idx, got)].any():
                    return empty, True
    return got, False


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
    keep_own_bottom: bool = True,
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
    :param keep_own_bottom: 地面を伝わないとき、塊そのものの底（下に床が無い所で「地面」とみなされた点）を塊に足す
                            （_own_bottom の決まり）。False なら、2026-11-15 までの動き（底の 12cm が欠ける）
    """
    t0 = time.perf_counter()
    empty = np.empty(0, dtype=np.int64)
    n = 0 if points is None else len(points)
    if n == 0 or seed_indices is None or len(seed_indices) == 0:
        return ClusterGrowth(empty)
    if points.dtype != np.float64:
        # 地面の格子の番号（座標 / 格子）と、窓の端（float64）を、同じ型で計算する。float32 のまま割ると、格子の境目のすぐ下の点が
        # 「番号は隣の格子・窓の判定はこちら側」になり、窓の大きさで地面の分類が変わる（2026-11-15 の独立レビュー）。
        # 画面は float64 の points_raw を渡すので、写しは作られない
        points = np.asarray(points, dtype=np.float64)
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
        band_idx = None                                      # 窓の柱の中で、地面とみなして外した点（塊の底を足すときに使う）
        if exclude_ground and n_cand > GROUND_MIN_POINTS:   # 以前の作りと同じく、選べる点の全部の点数で決める（窓の中の点数ではない）
            ticker.stage("地面を見分けています...")
            ground = ground_point_mask(col_pts, ground_grid, GROUND_HEIGHT)
            if float(np.mean(ground[seed_in_col])) < 0.5:
                if keep_own_bottom:
                    band_idx = col_idx[ground]
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
            del member, pts_idx, band_idx
            margin *= MARGIN_GROWTH
            rounds += 1
            continue
        indices = np.asarray(pts_idx[member], dtype=np.int64)
        del member, pts_idx
        checks = int(ticker.checks)

        # 4. 塊そのものの底（下に床が無い所で、地面とみなして外した点）を足す
        n_bottom = 0
        if band_idx is not None and not capped and len(indices):
            window_cells = None if xy_all else (
                int(np.floor(lo[0] / ground_grid)), int(np.floor(hi[0] / ground_grid)) - 1,
                int(np.floor(lo[1] / ground_grid)), int(np.floor(hi[1] / ground_grid)) - 1,
            )
            ticker.stage("塊の底を確かめています...")
            ticker.quiet = True
            try:
                bottom, too_narrow = _own_bottom(points, indices, band_idx, eps, ground_grid, window_cells, ticker)
            except ValueError:
                # 範囲が広すぎて格子に分けられない（座標の壊れた点が混じっている）。底は足さず、ここまでの塊を返す
                bottom, too_narrow = np.empty(0, dtype=np.int64), False
            finally:
                ticker.quiet = False
            if too_narrow:
                del indices, band_idx, bottom
                margin *= MARGIN_GROWTH
                rounds += 1
                continue
            if len(bottom):
                n_bottom = int(len(bottom))
                indices = np.insert(indices, np.searchsorted(indices, bottom), bottom)   # どちらも昇順で、重なりは無い
                capped = len(indices) > max_pts
        return ClusterGrowth(
            indices=indices,
            capped=bool(capped),
            n_seed=int(len(seeds)),
            limit=int(max_pts),
            rounds=rounds,
            window_points=n_window,
            margin=float(margin),
            seconds=time.perf_counter() - t0,
            checks=checks,
            n_selectable=int(n_cand),
            n_bottom=n_bottom,
        )
