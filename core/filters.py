"""
点群のノイズ除去およびフィルタリングアルゴリズムを提供するモジュール
"""

from typing import Optional, Tuple
import numpy as np
import open3d as o3d


def filter_statistical_outlier(
    points: np.ndarray,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
) -> np.ndarray:
    """
    統計的外れ値除去 (SOR: Statistical Outlier Removal)
    各点から近傍k点までの平均距離を算出し、全体平均 + (std_ratio * 標準偏差) を超える点をノイズとして除去する。

    :param points: (N, 3) 形状の点群座標配列
    :param nb_neighbors: 探索する近傍点の数（デフォルト: 20）
    :param std_ratio: 標準偏差の倍率閾値（値が小さいほど厳格にノイズを除去、デフォルト: 2.0）
    :return: 残存する点のインデックス配列 (np.ndarray)
    """
    if len(points) == 0:
        return np.array([], dtype=np.int64)

    # 点数が近傍数より少ない場合はそのまま返す
    if len(points) <= nb_neighbors:
        return np.arange(len(points), dtype=np.int64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    _, inlier_indices = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio,
    )

    return np.asarray(inlier_indices, dtype=np.int64)


def filter_radius_outlier(
    points: np.ndarray,
    nb_points: int = 16,
    radius: float = 0.05,
) -> np.ndarray:
    """
    半径ベース外れ値除去 (ROR: Radius Outlier Removal)
    指定半径以内に存在する隣接点の数が閾値未満である孤立点をノイズとして除去する。

    :param points: (N, 3) 形状の点群座標配列
    :param nb_points: 半径内に必要な最小近傍点数（デフォルト: 16）
    :param radius: 探索半径（メートル単位、デフォルト: 0.05）
    :return: 残存する点のインデックス配列 (np.ndarray)
    """
    if len(points) == 0:
        return np.array([], dtype=np.int64)

    if len(points) <= nb_points:
        return np.arange(len(points), dtype=np.int64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    _, inlier_indices = pcd.remove_radius_outlier(
        nb_points=nb_points,
        radius=radius,
    )

    return np.asarray(inlier_indices, dtype=np.int64)


def filter_distance(
    points: np.ndarray,
    min_dist: Optional[float] = None,
    max_dist: Optional[float] = None,
) -> np.ndarray:
    """
    スキャン原点 (0,0,0) からの距離に基づくフィルタリング
    近すぎるスキャナー周辺ノイズや遠すぎる不正確な点を除去する。

    :param points: (N, 3) 形状の点群座標配列
    :param min_dist: 最小許容距離 (m)
    :param max_dist: 最大許容距離 (m)
    :return: 残存する点のインデックス配列 (np.ndarray)
    """
    if len(points) == 0:
        return np.array([], dtype=np.int64)

    # 各点のユークリッド距離を計算
    dists = np.linalg.norm(points, axis=1)
    mask = np.ones(len(points), dtype=bool)

    if min_dist is not None:
        mask &= (dists >= min_dist)
    if max_dist is not None:
        mask &= (dists <= max_dist)

    return np.where(mask)[0]


def filter_intensity(
    intensities: np.ndarray,
    min_intensity: Optional[float] = None,
    max_intensity: Optional[float] = None,
) -> np.ndarray:
    """
    反射強度 (Intensity) に基づくフィルタリング
    極端に反射強度の低いノイズ点等を除去する。

    :param intensities: 強度配列
    :param min_intensity: 最小反射強度
    :param max_intensity: 最大反射強度
    :return: 残存する点のインデックス配列 (np.ndarray)
    """
    if len(intensities) == 0:
        return np.array([], dtype=np.int64)

    mask = np.ones(len(intensities), dtype=bool)

    if min_intensity is not None:
        mask &= (intensities >= min_intensity)
    if max_intensity is not None:
        mask &= (intensities <= max_intensity)

    return np.where(mask)[0]


def voxel_downsample_indices(
    points: np.ndarray,
    voxel_size: float = 0.02,
) -> np.ndarray:
    """
    ボクセルグリッド間引きフィルタ
    指定したボクセルサイズで点群を均一化し、代表点（各ボクセル内の最初の点）のインデックスを返す。
    属性配列（RGBやIntensity）との同期を保ちながら高速に間引くことが可能。

    :param points: (N, 3) 形状の点群座標配列
    :param voxel_size: ボクセルのサイズ（メートル単位、例: 0.02 = 2cm）
    :return: 残存する点のインデックス配列 (np.ndarray)
    """
    if len(points) == 0 or voxel_size <= 0:
        return np.arange(len(points), dtype=np.int64)

    # 各点のボクセルグリッドインデックスを算出
    voxel_coords = np.floor(points / voxel_size).astype(np.int64)

    # 一意なボクセルの最初のインデックスを抽出
    _, unique_indices = np.unique(voxel_coords, axis=0, return_index=True)
    return np.sort(unique_indices)
