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


def filter_isolated_clusters(
    points: np.ndarray,
    eps: float = 0.15,
    min_cluster_points: int = 30,
) -> np.ndarray:
    """
    孤立クラスタ（浮遊する小塊ノイズ）除去フィルタ
    DBSCANクラスタリングにより点群を連結成分（塊）に分割し、
    構成点数が min_cluster_points 未満の小規模な浮遊塊（通行人・飛散物・枝葉など）を一括除去する。
    大量点群（1万点超）では適応型ボクセル代表クラスタリングにより、メモリ枯渇やフリーズを
    物理的に防止しながら超高速・省メモリに処理する。

    :param points: (N, 3) 形状の点群座標配列
    :param eps: 同一クラスタとみなす近傍探索距離（メートル単位、デフォルト: 0.15）
    :param min_cluster_points: 保持するクラスタの最小構成点数（デフォルト: 30）
    :return: 残存する点のインデックス配列 (np.ndarray)
    """
    if points is None or len(points) == 0:
        return np.array([], dtype=np.int64)

    n = len(points)
    if n < min_cluster_points or eps <= 0:
        return np.arange(n, dtype=np.int64)

    # 1. 小規模点群 (10,000点以下): 既存の直接DBSCANで完全な結果整合性を保証
    if n <= 10000:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        core_min_pts = max(2, min(15, min_cluster_points // 3))
        labels = np.asarray(
            pcd.cluster_dbscan(eps=float(eps), min_points=int(core_min_pts), print_progress=False),
            dtype=np.int64,
        )

        valid_mask = labels >= 0
        if not np.any(valid_mask):
            return np.array([], dtype=np.int64)

        valid_labels = labels[valid_mask]
        counts = np.bincount(valid_labels)
        large_clusters = np.where(counts >= min_cluster_points)[0]

        if len(large_clusters) == 0:
            return np.array([], dtype=np.int64)

        keep_mask = np.isin(labels, large_clusters)
        return np.where(keep_mask)[0]

    # 2. 大規模点群 (10,000点超): 適応型ボクセル代表DBSCANによる超高速・安全処理
    # ボクセル代表点により数百万〜数千万点を安全な点数に圧縮し、メモリ枯渇とPCフリーズを防止
    init_voxel_size = max(0.015, float(eps) * 0.45)
    coords = np.floor(points / init_voxel_size).astype(np.int64)
    unique_coords, inverse_idx, counts = np.unique(
        coords, axis=0, return_inverse=True, return_counts=True
    )

    # 代表点数が25万点を超える場合は、代表点上限25万点以下になるようボクセルサイズを適応拡大（安全リミッター）
    max_safe_reps = 250000
    voxel_size = init_voxel_size
    if len(unique_coords) > max_safe_reps:
        scale = (len(unique_coords) / max_safe_reps) ** (1.0 / 3.0) * 1.05
        voxel_size = init_voxel_size * scale
        coords = np.floor(points / voxel_size).astype(np.int64)
        unique_coords, inverse_idx, counts = np.unique(
            coords, axis=0, return_inverse=True, return_counts=True
        )

    rep_points = (unique_coords.astype(np.float64) + 0.5) * voxel_size

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(rep_points)

    # 探索距離はボクセルサイズも考慮
    eff_eps = max(float(eps), voxel_size * 1.5)
    rep_labels = np.asarray(
        pcd.cluster_dbscan(eps=eff_eps, min_points=1, print_progress=False),
        dtype=np.int64,
    )

    valid_mask = rep_labels >= 0
    if not np.any(valid_mask):
        return np.array([], dtype=np.int64)

    # 元点群の正確な点数を各クラスタごとに重み付き集計
    cluster_point_counts = np.bincount(
        rep_labels[valid_mask],
        weights=counts[valid_mask],
    ).astype(np.int64)

    large_clusters = np.where(cluster_point_counts >= min_cluster_points)[0]
    if len(large_clusters) == 0:
        return np.array([], dtype=np.int64)

    rep_keep_mask = np.isin(rep_labels, large_clusters)
    point_keep_mask = rep_keep_mask[inverse_idx]
    return np.where(point_keep_mask)[0]


def segment_ground_points(
    points: np.ndarray,
    grid_size: float = 1.0,
    height_threshold: float = 0.25,
) -> np.ndarray:
    """
    局所最低標高グリッド法による地面（グラウンド / 床面）点群の高速自動抽出
    XY平面を grid_size 四方のセルに分割し、各セル内の最低Z値（局所地面標高）から
    height_threshold 以内にある点のインデックスを返す。

    :param points: (N, 3) 形状の点群座標配列
    :param grid_size: XYグリッドのセルサイズ（メートル単位、デフォルト: 1.0m）
    :param height_threshold: 局所最低点からの許容高さ（メートル単位、デフォルト: 0.25m）
    :return: 地面と判定された点のインデックス配列 (np.ndarray)
    """
    if points is None or len(points) == 0:
        return np.array([], dtype=np.int64)

    grid_size = max(0.05, float(grid_size))
    height_threshold = max(0.001, float(height_threshold))

    xy_cells = np.floor(points[:, :2] / grid_size).astype(np.int64)
    _, inverse_idx = np.unique(xy_cells, axis=0, return_inverse=True)

    n_cells = int(np.max(inverse_idx)) + 1 if len(inverse_idx) > 0 else 0
    cell_min_z = np.full(n_cells, np.inf, dtype=np.float64)
    np.minimum.at(cell_min_z, inverse_idx, points[:, 2])

    local_min_z = cell_min_z[inverse_idx]
    ground_mask = (points[:, 2] - local_min_z) <= height_threshold
    return np.where(ground_mask)[0]


def segment_largest_plane(
    points: np.ndarray,
    distance_threshold: float = 0.05,
    ransac_n: int = 3,
    num_iterations: int = 400,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    RANSACアルゴリズムによる支配的平面（壁面・床面・天井・屋根等）の自動検出
    点群の中から最も多くの点を含む平面を検出し、その平面に属する点のインデックス配列と
    平面方程式 [a, b, c, d] (ax + by + cz + d = 0) を返す。

    :param points: (N, 3) 形状の点群座標配列
    :param distance_threshold: 平面からの許容距離（メートル単位、デフォルト: 0.05m）
    :param ransac_n: 平面推定に用いるサンプル点数（デフォルト: 3）
    :param num_iterations: RANSAC反復回数（デフォルト: 400）
    :return: (平面内の点インデックス配列, 平面モデル [a, b, c, d])
    """
    if points is None or len(points) < max(3, ransac_n):
        return np.array([], dtype=np.int64), np.zeros(4, dtype=np.float64)

    # 全点がほぼ同一座標の場合の保護
    pt_span = np.max(points, axis=0) - np.min(points, axis=0)
    if float(np.linalg.norm(pt_span)) < 1e-7:
        return np.arange(len(points), dtype=np.int64), np.array([0.0, 0.0, 1.0, -float(points[0, 2])], dtype=np.float64)

    try:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=max(1e-4, float(distance_threshold)),
            ransac_n=max(3, int(ransac_n)),
            num_iterations=max(10, int(num_iterations)),
        )
        return np.asarray(inliers, dtype=np.int64), np.asarray(plane_model, dtype=np.float64)
    except Exception:
        return np.array([], dtype=np.int64), np.zeros(4, dtype=np.float64)


def keep_largest_clusters(
    points: np.ndarray,
    eps: float = 0.30,
    top_k: int = 1,
    min_points: int = 10,
    min_keep_ratio: float = 0.0,
) -> np.ndarray:
    """
    最大連結クラスタ保持フィルタ（大まかなノイズ一括除去用）
    主要構造物（点数が上位 top_k 個の大きな塊）のみを残し、
    周囲に浮遊する不要塊・飛散ノイズ・外れ値を一括して除外する。
    大規模点群（1万点超）では適応型ボクセル代表クラスタリングにより、フリーズやメモリ枯渇を防止。

    :param points: (N, 3) 形状の点群座標配列
    :param eps: 同一の塊とみなす連結距離（メートル単位、デフォルト: 0.30m）
    :param top_k: 残す上位クラスタ数（デフォルト: 1 = 最大の塊のみ保持）
    :param min_points: DBSCANのコア点最小近傍点数（デフォルト: 10）
    :param min_keep_ratio: 全体の一定割合以上を占めるサブクラスタも保持する下限比率
    :return: 保持する（メイン構造物に属する）点のインデックス配列 (np.ndarray)
    """
    if points is None or len(points) == 0:
        return np.array([], dtype=np.int64)

    n = len(points)
    if n <= max(3, min_points) or eps <= 0:
        return np.arange(n, dtype=np.int64)

    # 1. 小規模点群 (10,000点以下): 直接DBSCANでテスト互換性と精度を保持
    if n <= 10000:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        core_pts = max(2, min(15, int(min_points)))
        labels = np.asarray(
            pcd.cluster_dbscan(eps=float(eps), min_points=core_pts, print_progress=False),
            dtype=np.int64,
        )

        valid_mask = labels >= 0
        if not np.any(valid_mask):
            return np.arange(n, dtype=np.int64)

        counts = np.bincount(labels[valid_mask])
        if len(counts) == 0:
            return np.arange(n, dtype=np.int64)

        k = max(1, min(int(top_k), len(counts)))
        top_cluster_ids = set(np.argsort(counts)[::-1][:k].tolist())
        if min_keep_ratio > 0.0:
            min_cnt = int(n * float(min_keep_ratio))
            for cid, cnt in enumerate(counts):
                if cnt >= min_cnt:
                    top_cluster_ids.add(cid)

        keep_mask = np.isin(labels, list(top_cluster_ids))
        return np.where(keep_mask)[0]

    # 2. 大規模点群 (10,000点超): 適応型ボクセル代表DBSCAN
    init_voxel_size = max(0.02, float(eps) * 0.45)
    coords = np.floor(points / init_voxel_size).astype(np.int64)
    unique_coords, inverse_idx, counts = np.unique(
        coords, axis=0, return_inverse=True, return_counts=True
    )

    max_safe_reps = 250000
    voxel_size = init_voxel_size
    if len(unique_coords) > max_safe_reps:
        scale = (len(unique_coords) / max_safe_reps) ** (1.0 / 3.0) * 1.05
        voxel_size = init_voxel_size * scale
        coords = np.floor(points / voxel_size).astype(np.int64)
        unique_coords, inverse_idx, counts = np.unique(
            coords, axis=0, return_inverse=True, return_counts=True
        )

    rep_points = (unique_coords.astype(np.float64) + 0.5) * voxel_size

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(rep_points)

    eff_eps = max(float(eps), voxel_size * 1.5)
    rep_labels = np.asarray(
        pcd.cluster_dbscan(eps=eff_eps, min_points=1, print_progress=False),
        dtype=np.int64,
    )

    valid_mask = rep_labels >= 0
    if not np.any(valid_mask):
        return np.arange(n, dtype=np.int64)

    cluster_point_counts = np.bincount(
        rep_labels[valid_mask],
        weights=counts[valid_mask],
    ).astype(np.int64)

    if len(cluster_point_counts) == 0:
        return np.arange(n, dtype=np.int64)

    k = max(1, min(int(top_k), len(cluster_point_counts)))
    top_cluster_ids = set(np.argsort(cluster_point_counts)[::-1][:k].tolist())
    if min_keep_ratio > 0.0:
        min_cnt = int(n * float(min_keep_ratio))
        for cid, cnt in enumerate(cluster_point_counts):
            if cnt >= min_cnt:
                top_cluster_ids.add(cid)

    rep_keep_mask = np.isin(rep_labels, list(top_cluster_ids))
    point_keep_mask = rep_keep_mask[inverse_idx]
    return np.where(point_keep_mask)[0]


def filter_edge_veil_noise(
    points: np.ndarray,
    nb_neighbors: int = 15,
    std_ratio: float = 1.3,
    linearity_threshold: float = 0.72,
    sor_std_ratio: float = -1.0,
) -> np.ndarray:
    """
    エッジ飛散・混合ピクセル（ベールノイズ / ヒゲノイズ）精密除去フィルタ（細かいノイズ処理用）
    物体輪郭部でレーザースキャナーの視線方向に引き伸ばされる飛散点（ベールノイズ）や、
    表面からわずかに浮き上がった微細ヒゲノイズを、局所共分散（主成分の異方性・視線平行度）と
    近傍距離勾配の複合判定によって精密に除去する。

    :param points: (N, 3) 形状の点群座標配列
    :param nb_neighbors: 局所解析に用いる近傍点数（デフォルト: 15）
    :param std_ratio: 許容するばらつき倍率（小さいほど細かいヒゲを厳格に除去、デフォルト: 1.3）
    :param linearity_threshold: 線状突出度（ヒゲ）とみなす閾値（0.5〜0.95、デフォルト: 0.72）
    :param sor_std_ratio: std_ratio の別名キーワード引数（指定時は優先）
    :return: 残存する点のインデックス配列 (np.ndarray)
    """
    if points is None or len(points) == 0:
        return np.array([], dtype=np.int64)

    eff_std = float(sor_std_ratio) if sor_std_ratio > 0 else float(std_ratio)
    n = len(points)
    k = max(5, min(int(nb_neighbors), n - 1))
    if n <= k + 1:
        return np.arange(n, dtype=np.int64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # 1. 高感度な局所統計的外れ値判定
    _, sor_inliers = pcd.remove_statistical_outlier(nb_neighbors=k, std_ratio=eff_std)
    sor_mask = np.zeros(n, dtype=bool)
    sor_mask[np.asarray(sor_inliers, dtype=np.int64)] = True

    # 2. 局所共分散行列から固有値を計算し、極端に散乱または視線方向に孤立突出したエッジ飛散点を検出
    try:
        pcd.estimate_covariances(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k))
        covs = np.asarray(pcd.covariances, dtype=np.float64)
        if len(covs) == n:
            eigvals = np.linalg.eigvalsh(covs)  # 昇順: [lambda_min, lambda_mid, lambda_max]
            eigvals = np.maximum(eigvals, 0.0)
            # 散乱度 (Sphericity) と 線状突出度 (Linearity)
            sphericity = eigvals[:, 0] / (eigvals[:, 2] + 1e-12)
            linearity = (eigvals[:, 2] - eigvals[:, 1]) / (eigvals[:, 2] + 1e-12)
            # 局所サイズ（最大固有値）が全体の92パーセンタイルを超えて大きく、かつ平面性が低い（エッジの飛び点・ヒゲ）ものを除外
            size_thresh = float(np.percentile(eigvals[:, 2], 92))
            if size_thresh > 1e-9:
                lin_th = max(0.3, min(0.98, float(linearity_threshold)))
                veil_outlier = (eigvals[:, 2] > size_thresh * max(0.5, eff_std)) & (
                    (sphericity > 0.12) | (linearity > lin_th)
                )
                sor_mask &= ~veil_outlier
    except Exception:
        pass

    return np.where(sor_mask)[0]


def extract_connected_cluster_from_seeds(
    points: np.ndarray,
    seed_indices: np.ndarray,
    radius: float = 0.25,
    exclude_ground: bool = True,
    eps: float = -1.0,
    max_cluster_ratio: float = 0.85,
) -> np.ndarray:
    """
    マジックワンド（連結塊のワンクリック自動選択）
    指定されたシード点（選択点）から半径 radius 以内で3D空間的に繋がっている同一の塊（クラスタ）を
    高速に探索してインデックス配列を返す。通行人・車両・看板・浮遊ノイズ塊のピンポイント一括選択に最適。

    :param points: (N, 3) 形状の点群座標配列
    :param seed_indices: 起点となる点インデックス配列
    :param radius: 同一オブジェクトとみなす連結探索半径 (m)
    :param exclude_ground: Trueの場合、地面を伝って建物全体へ広がるのを防ぐため局所最低面付近の伝播を抑制する
    :param eps: radius の別名キーワード引数（指定時は優先）
    :param max_cluster_ratio: 全体点数に対する最大膨張比率（建物全体への暴走防止）
    :return: 連結クラスタに属する点インデックス配列 (np.ndarray)
    """
    if points is None or len(points) == 0 or seed_indices is None or len(seed_indices) == 0:
        return np.array([], dtype=np.int64)

    n = len(points)
    seeds = np.asarray(seed_indices, dtype=np.int64)
    seeds = seeds[(seeds >= 0) & (seeds < n)]
    if len(seeds) == 0:
        return np.array([], dtype=np.int64)

    eff_radius = max(0.01, float(eps) if eps > 0 else float(radius))
    max_pts = max(len(seeds), int(n * max(0.05, min(1.0, float(max_cluster_ratio)))))

    # 地面伝播防止マスク（オプション）: シード点自体が地面にある場合は除外しないが、シードが高い位置にある場合は床面伝播を遮断
    allowed_mask = np.ones(n, dtype=bool)
    if exclude_ground and n > 50:
        ground_idx = segment_ground_points(points, grid_size=max(0.8, eff_radius * 4.0), height_threshold=0.12)
        if len(ground_idx) > 0 and len(ground_idx) < n:
            seed_in_ground = np.isin(seeds, ground_idx)
            # シードの過半数が地面以外のオブジェクト上にある場合のみ、地面への伝播をブロック
            if np.mean(seed_in_ground) < 0.5:
                allowed_mask[ground_idx] = False
                allowed_mask[seeds] = True

    try:
        from scipy.spatial import cKDTree

        valid_indices = np.where(allowed_mask)[0]
        if len(valid_indices) == 0:
            return np.unique(seeds)

        sub_pts = points[valid_indices]
        tree = cKDTree(sub_pts)

        # 全体インデックス -> valid_indices 内の局所インデックスへの逆引き
        global_to_local = np.full(n, -1, dtype=np.int64)
        global_to_local[valid_indices] = np.arange(len(valid_indices), dtype=np.int64)

        visited_local = np.zeros(len(valid_indices), dtype=bool)
        seed_locals = global_to_local[seeds]
        seed_locals = seed_locals[seed_locals >= 0]
        visited_local[seed_locals] = True
        total_visited = int(np.sum(visited_local))

        frontier = list(seed_locals)
        # 大規模点群でも暴走しないよう最大探索ステップを制御しつつ幅優先探索
        while frontier and total_visited < max_pts:
            neighbors_list = tree.query_ball_point(sub_pts[frontier], r=eff_radius)
            next_frontier = []
            for nbrs in neighbors_list:
                for nb in nbrs:
                    if not visited_local[nb]:
                        visited_local[nb] = True
                        next_frontier.append(nb)
                        total_visited += 1
                        if total_visited >= max_pts:
                            break
                if total_visited >= max_pts:
                    break
            frontier = next_frontier

        return np.sort(valid_indices[visited_local])
    except Exception:
        return np.unique(seeds)


