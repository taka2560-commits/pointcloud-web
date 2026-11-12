# -*- coding: utf-8 -*-
"""
全点転写（間引き読込で決めたノイズ判定を、元ファイルの全点へ写す）の純粋関数

1 億点超の E57 はメモリ保護のため「k 点に 1 点」の間引きで読み込まれ、ユーザーはその間引き点群で
ノイズ判定（自動フィルタ・手動削除・レイヤー分け）を行う。保存時に元ファイルを全点で読み直し、
判定済みの点（判定点）から各点へラベルを写すのがここの役割。GUI にも E57 にも依存しない。

転写の規則（スキャンのローカル座標で行う。pose・座標系変換は両側に掛かっていないので不要）
- 全点の各点 p について、最も近い判定点 q を探す（scipy cKDTree、全コア並列）。
- 判定点 q ごとに「他の判定点までの最短距離」s(q) を求めておき、
  p と q の距離 d が  max(factor × s(q), min_radius)  以内なら p は q のラベルを引き継ぐ。
- それより遠い p は「孤立点」。判定点のどれとも離れていて、間引き前の点群でも周囲に点がないことを
  意味するので、ノイズとして除外する（ラベル ORPHAN_LABEL）。
- factor は間引き率 k に応じて max(3, k + 1)。スキャナの格子状の点群では、行方向に k 点おきに
  判定点があるため、同じ面上の未判定点は必ず (k/2) × 点間隔以内に判定点を持ち、面の点が孤立点に
  なることはない。面から factor × 点間隔より離れた浮遊点だけが孤立点になる。
"""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

ORPHAN_LABEL = -1
DEFAULT_MIN_RADIUS = 0.01  # m。判定点が重なっている（間隔 0）場合の下限
DEFAULT_BASE_FACTOR = 3.0


@dataclass
class TransferStats:
    """1 スキャン分の転写結果の集計"""
    n_full: int = 0        # 元ファイルの点数
    n_judged: int = 0      # 判定点の点数
    n_inherited: int = 0   # 判定点からラベルを引き継いだ点数
    n_orphan: int = 0      # 孤立点（除外）の点数
    stride: int = 1        # 推定した間引き率 k
    factor: float = DEFAULT_BASE_FACTOR
    max_inherit_distance: float = 0.0  # 引き継いだ点のうち最も遠かった距離 (m)


def estimate_stride(n_full: int, n_judged: int) -> int:
    """元の点数と判定点数から間引き率 k を推定する（NaN 除外などで割り切れなくても丸める）"""
    if n_judged <= 0:
        return 1
    return max(1, int(round(n_full / float(n_judged))))


def radius_factor(stride: int, base_factor: float = DEFAULT_BASE_FACTOR) -> float:
    """間引き率 k に応じた許容半径の倍率。k が大きいほど判定点が粗くなるので広げる"""
    return float(max(base_factor, stride + 1))


def judged_spacing(judged_local: np.ndarray, tree: Optional[cKDTree] = None, workers: int = -1) -> np.ndarray:
    """各判定点から最も近い他の判定点までの距離 (N_j,)。判定点が 1 つだけなら無限大"""
    n = len(judged_local)
    if n <= 1:
        return np.full(n, np.inf, dtype=np.float64)
    if tree is None:
        tree = cKDTree(judged_local)
    d, _ = tree.query(judged_local, k=2, workers=workers)
    spacing = d[:, 1].astype(np.float64)
    # 同一座標の重複点があると 0 になる。そのまま半径 0 にせず min_radius 側で下限を掛ける
    return spacing


class LabelTransfer:
    """
    判定点から全点へのラベルの転写を、全点を少しずつ渡して行う（全点をメモリに載せずに保存するため）。
    木と許容半径は最初に 1 回だけ作り、labels_for() を何回呼んでも、全点を一度に渡したときと同じラベルになる。
    """

    def __init__(
        self,
        judged_local: np.ndarray,
        judged_labels: np.ndarray,
        stride: int,
        base_factor: float = DEFAULT_BASE_FACTOR,
        min_radius: float = DEFAULT_MIN_RADIUS,
        workers: int = -1,
    ):
        judged_local = np.ascontiguousarray(judged_local, dtype=np.float64).reshape(-1, 3)
        judged_labels = np.asarray(judged_labels).reshape(-1)
        if len(judged_labels) != len(judged_local):
            raise ValueError(f"判定点の座標 {len(judged_local):,} 点とラベル {len(judged_labels):,} 個の数が一致しません。")
        self.workers = workers
        self.stats = TransferStats(n_judged=len(judged_local))
        self.stats.stride = max(1, int(stride))
        self.stats.factor = radius_factor(self.stats.stride, base_factor)
        self._labels_j = judged_labels.astype(np.int32, copy=False)
        self._tree: Optional[cKDTree] = None
        self._radius: Optional[np.ndarray] = None
        if len(judged_local) > 0:
            self._tree = cKDTree(judged_local, balanced_tree=False, compact_nodes=False)
            spacing = judged_spacing(judged_local, tree=self._tree, workers=workers)
            self._radius = np.maximum(spacing * self.stats.factor, float(min_radius))

    def labels_for(self, full_local: np.ndarray) -> np.ndarray:
        """全点の一部 (n, 3) のラベル (n,) int32 を返し、集計（stats）に足す。孤立点は ORPHAN_LABEL"""
        full_local = np.ascontiguousarray(full_local, dtype=np.float64).reshape(-1, 3)
        n = len(full_local)
        labels = np.full(n, ORPHAN_LABEL, dtype=np.int32)
        st = self.stats
        st.n_full += n
        if n == 0:
            return labels
        if self._tree is None:
            st.n_orphan += n
            return labels
        d, idx = self._tree.query(full_local, k=1, workers=self.workers)
        inherit = d <= self._radius[idx]
        labels[inherit] = self._labels_j[idx[inherit]]
        n_inherit = int(np.count_nonzero(inherit))
        if n_inherit:
            st.max_inherit_distance = max(st.max_inherit_distance, float(np.max(d[inherit])))
        st.n_inherited += n_inherit
        st.n_orphan += n - n_inherit
        return labels


def transfer_labels(
    judged_local: np.ndarray,
    judged_labels: np.ndarray,
    full_local: np.ndarray,
    stride: Optional[int] = None,
    base_factor: float = DEFAULT_BASE_FACTOR,
    min_radius: float = DEFAULT_MIN_RADIUS,
    chunk_size: int = 2_000_000,
    workers: int = -1,
    is_cancelled: Optional[Callable[[], bool]] = None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> Tuple[np.ndarray, TransferStats]:
    """
    判定点 (judged_local, judged_labels) から全点 full_local へラベルを写す（全点を一度に渡す版）。

    :param judged_local: (N_j, 3) 判定点のローカル座標
    :param judged_labels: (N_j,) 整数ラベル（例: 1=保存, 0=除外。分割保存ではレイヤー番号）
    :param full_local: (N, 3) 元ファイルの全点のローカル座標
    :param stride: 間引き率 k（None なら点数比から推定）
    :return: (labels_full (N,) int32、TransferStats)。孤立点は ORPHAN_LABEL
    """
    judged_local = np.ascontiguousarray(judged_local, dtype=np.float64).reshape(-1, 3)
    full_local = np.ascontiguousarray(full_local, dtype=np.float64).reshape(-1, 3)
    n_full = len(full_local)
    n_judged = len(judged_local)
    k = int(stride) if stride is not None and stride >= 1 else estimate_stride(n_full, n_judged)
    lt = LabelTransfer(judged_local, judged_labels, k, base_factor=base_factor, min_radius=min_radius, workers=workers)
    labels_full = np.full(n_full, ORPHAN_LABEL, dtype=np.int32)
    chunk_size = max(1, int(chunk_size))
    for start in range(0, n_full, chunk_size):
        if is_cancelled is not None and is_cancelled():
            raise InterruptedError("全点転写がユーザーによって中止されました。")
        end = min(n_full, start + chunk_size)
        labels_full[start:end] = lt.labels_for(full_local[start:end])
        if progress is not None:
            progress(end, n_full)
    return labels_full, lt.stats


def labels_from_masks(masks, n_points: int) -> np.ndarray:
    """
    レイヤー別マスクのリスト（互いに素）を 1 本の整数ラベル (N,) にする。
    どのマスクにも入らない点は ORPHAN_LABEL（保存しない）。
    """
    labels = np.full(int(n_points), ORPHAN_LABEL, dtype=np.int32)
    for i, m in enumerate(masks):
        m = np.asarray(m, dtype=bool).reshape(-1)
        if len(m) != n_points:
            raise ValueError(f"マスク {i} の長さ {len(m):,} が点数 {n_points:,} と一致しません。")
        labels[m] = i
    return labels
