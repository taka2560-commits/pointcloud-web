"""
E57点群データに対するノイズ処理全体の実行管理パイプライン
"""

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Any
import numpy as np

from core.e57_io import E57ScanData, read_e57_scans, write_e57_scans
from core.filters import (
    filter_statistical_outlier,
    filter_radius_outlier,
    filter_distance,
    filter_intensity,
    voxel_downsample_indices,
    filter_isolated_clusters,
    filter_edge_veil_noise,
)


@dataclass
class FilterConfig:
    """フィルタ設定パラメータを保持するデータクラス"""

    # 統計的外れ値除去 (SOR)
    use_sor: bool = True
    sor_neighbors: int = 20
    sor_std_ratio: float = 2.0

    # 半径ベース外れ値除去 (ROR)
    use_ror: bool = False
    ror_radius: float = 0.05
    ror_min_points: int = 16

    # 孤立クラスタ（浮遊小塊ノイズ）除去
    use_cluster_filter: bool = False
    cluster_eps: float = 0.15
    cluster_min_points: int = 30

    # エッジ飛散・ヒゲ（ベールノイズ）精密除去
    use_edge_veil_filter: bool = False
    edge_neighbors: int = 16
    edge_linearity_thresh: float = 0.72
    edge_sor_std_ratio: float = 1.25

    # 距離フィルタ
    use_distance_filter: bool = False
    min_distance: Optional[float] = 0.5
    max_distance: Optional[float] = 100.0

    # 反射強度フィルタ
    use_intensity_filter: bool = False
    min_intensity: Optional[float] = 0.01
    max_intensity: Optional[float] = None

    # ボクセル間引き
    use_voxel_downsample: bool = False
    voxel_size: float = 0.02


@dataclass
class ScanProcessResult:
    """スキャンごとの処理結果サマリー"""

    scan_index: int
    initial_points: int
    final_points: int
    removed_points: int
    removal_ratio: float
    time_taken_sec: float


@dataclass
class ProcessSummary:
    """ファイル全体の処理結果サマリー"""

    input_file: str
    output_file: str
    scan_results: List[ScanProcessResult] = field(default_factory=list)
    total_initial_points: int = 0
    total_final_points: int = 0
    total_removed_points: int = 0
    total_removal_ratio: float = 0.0
    total_time_taken_sec: float = 0.0


class PointCloudProcessor:
    """E57点群のノイズ除去パイプラインを制御するクラス"""

    def __init__(self, config: Optional[FilterConfig] = None):
        self.config = config or FilterConfig()

    def apply_filters_to_scan(
        self,
        scan: E57ScanData,
        log_callback: Optional[Callable[[str], None]] = None,
    ) -> Tuple[E57ScanData, np.ndarray, np.ndarray]:
        """
        単一スキャンに対して設定されたフィルタを順次適用する。

        :param scan: 対象スキャンデータ
        :param log_callback: ログ出力コールバック
        :return: (フィルタ適用後スキャン, 保持インデックス, 除去インデックス)
        """
        points = scan.points
        total_points = len(points)
        keep_mask = np.ones(total_points, dtype=bool)

        def log(msg: str):
            if log_callback:
                log_callback(msg)

        # 1. 距離フィルタ
        if self.config.use_distance_filter:
            dist_indices = filter_distance(
                points,
                min_dist=self.config.min_distance,
                max_dist=self.config.max_distance,
            )
            dist_mask = np.zeros(total_points, dtype=bool)
            dist_mask[dist_indices] = True
            keep_mask &= dist_mask
            log(f"距離フィルタ適用: 残り {np.sum(keep_mask):,} 点")

        # 2. 反射強度フィルタ
        if self.config.use_intensity_filter and "intensity" in scan.raw_fields:
            intensities = np.asarray(scan.raw_fields["intensity"])
            int_indices = filter_intensity(
                intensities,
                min_intensity=self.config.min_intensity,
                max_intensity=self.config.max_intensity,
            )
            int_mask = np.zeros(total_points, dtype=bool)
            int_mask[int_indices] = True
            keep_mask &= int_mask
            log(f"反射強度フィルタ適用: 残り {np.sum(keep_mask):,} 点")

        # 3. ボクセル間引き
        if self.config.use_voxel_downsample and self.config.voxel_size > 0:
            current_indices = np.where(keep_mask)[0]
            if len(current_indices) > 0:
                voxel_local_indices = voxel_downsample_indices(
                    points[current_indices],
                    voxel_size=self.config.voxel_size,
                )
                voxel_mask = np.zeros(total_points, dtype=bool)
                voxel_mask[current_indices[voxel_local_indices]] = True
                keep_mask &= voxel_mask
                log(f"ボクセル間引き適用 ({self.config.voxel_size}m): 残り {np.sum(keep_mask):,} 点")

        # 4. 統計的外れ値除去 (SOR)
        if self.config.use_sor:
            current_indices = np.where(keep_mask)[0]
            if len(current_indices) > 0:
                sor_local_indices = filter_statistical_outlier(
                    points[current_indices],
                    nb_neighbors=self.config.sor_neighbors,
                    std_ratio=self.config.sor_std_ratio,
                )
                sor_mask = np.zeros(total_points, dtype=bool)
                sor_mask[current_indices[sor_local_indices]] = True
                keep_mask &= sor_mask
                log(f"SORノイズ除去適用 (近傍{self.config.sor_neighbors}, 倍率{self.config.sor_std_ratio}): 残り {np.sum(keep_mask):,} 点")

        # 5. 半径ベース外れ値除去 (ROR)
        if self.config.use_ror:
            current_indices = np.where(keep_mask)[0]
            if len(current_indices) > 0:
                ror_local_indices = filter_radius_outlier(
                    points[current_indices],
                    nb_points=self.config.ror_min_points,
                    radius=self.config.ror_radius,
                )
                ror_mask = np.zeros(total_points, dtype=bool)
                ror_mask[current_indices[ror_local_indices]] = True
                keep_mask &= ror_mask
                log(f"RORノイズ除去適用 (半径{self.config.ror_radius}m, 最小点数{self.config.ror_min_points}): 残り {np.sum(keep_mask):,} 点")

        # 6. 孤立クラスタ（浮遊小塊ノイズ）除去
        if self.config.use_cluster_filter:
            current_indices = np.where(keep_mask)[0]
            if len(current_indices) > 0:
                cluster_local_indices = filter_isolated_clusters(
                    points[current_indices],
                    eps=self.config.cluster_eps,
                    min_cluster_points=self.config.cluster_min_points,
                )
                cluster_mask = np.zeros(total_points, dtype=bool)
                cluster_mask[current_indices[cluster_local_indices]] = True
                keep_mask &= cluster_mask
                log(f"孤立クラスタ除去適用 (距離{self.config.cluster_eps}m, 最小塊{self.config.cluster_min_points}点): 残り {np.sum(keep_mask):,} 点")

        # 7. エッジ飛散・ヒゲ（ベールノイズ）精密除去
        if self.config.use_edge_veil_filter:
            current_indices = np.where(keep_mask)[0]
            if len(current_indices) > 0:
                edge_local_indices = filter_edge_veil_noise(
                    points[current_indices],
                    nb_neighbors=self.config.edge_neighbors,
                    linearity_threshold=self.config.edge_linearity_thresh,
                    sor_std_ratio=self.config.edge_sor_std_ratio,
                )
                edge_mask = np.zeros(total_points, dtype=bool)
                edge_mask[current_indices[edge_local_indices]] = True
                keep_mask &= edge_mask
                log(f"エッジ飛散・ヒゲ精密除去適用: 残り {np.sum(keep_mask):,} 点")

        keep_indices = np.where(keep_mask)[0]
        removed_indices = np.where(~keep_mask)[0]

        filtered_scan = scan.filter_by_indices(keep_indices)
        return filtered_scan, keep_indices, removed_indices

    def apply_filters_to_points(
        self,
        points: np.ndarray,
        log_callback: Optional[Callable[[str], None]] = None,
        intensities: Optional[np.ndarray] = None,
        distance_points: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        メモリ上の3D点群座標に対して各種フィルタを直接適用し、(keep_indices, removed_indices) を返す。
        手動削除後の通常レイヤー点群に対して実行でき、手動削除点が復活する問題を根本解決します。

        distance_points: 距離フィルタだけに使う「スキャン原点からの相対座標」。
        points が pose 適用後の座標のとき、原点からの距離を保つために渡す（省略時は points を使う）。
        """
        total = len(points)
        keep_mask = np.ones(total, dtype=bool)

        def log(msg: str):
            if log_callback:
                log_callback(msg)

        # 1. 距離フィルタ
        if self.config.use_distance_filter:
            current = np.where(keep_mask)[0]
            if len(current) > 0:
                d_idx = filter_distance(
                    (points if distance_points is None else distance_points)[current],
                    min_dist=self.config.min_distance,
                    max_dist=self.config.max_distance,
                )
                m = np.zeros(total, dtype=bool)
                m[current[d_idx]] = True
                keep_mask &= m
                log(f"距離フィルタ適用: 残り {np.sum(keep_mask):,} 点")

        # 1b. 反射強度フィルタ
        if self.config.use_intensity_filter and intensities is not None and len(intensities) == total:
            current = np.where(keep_mask)[0]
            if len(current) > 0:
                i_idx = filter_intensity(
                    intensities[current],
                    min_intensity=self.config.min_intensity,
                    max_intensity=self.config.max_intensity,
                )
                m = np.zeros(total, dtype=bool)
                m[current[i_idx]] = True
                keep_mask &= m
                log(f"反射強度フィルタ適用: 残り {np.sum(keep_mask):,} 点")

        # 2. ボクセル間引き
        if self.config.use_voxel_downsample and self.config.voxel_size > 0:
            current = np.where(keep_mask)[0]
            if len(current) > 0:
                v_idx = voxel_downsample_indices(
                    points[current],
                    voxel_size=self.config.voxel_size,
                )
                m = np.zeros(total, dtype=bool)
                m[current[v_idx]] = True
                keep_mask &= m
                log(f"ボクセル間引き適用: 残り {np.sum(keep_mask):,} 点")

        # 3. SORフィルタ
        if self.config.use_sor:
            current = np.where(keep_mask)[0]
            if len(current) > 0:
                s_idx = filter_statistical_outlier(
                    points[current],
                    nb_neighbors=self.config.sor_neighbors,
                    std_ratio=self.config.sor_std_ratio,
                )
                m = np.zeros(total, dtype=bool)
                m[current[s_idx]] = True
                keep_mask &= m
                log(f"SORノイズ除去適用: 残り {np.sum(keep_mask):,} 点")

        # 4. RORフィルタ
        if self.config.use_ror:
            current = np.where(keep_mask)[0]
            if len(current) > 0:
                r_idx = filter_radius_outlier(
                    points[current],
                    nb_points=self.config.ror_min_points,
                    radius=self.config.ror_radius,
                )
                m = np.zeros(total, dtype=bool)
                m[current[r_idx]] = True
                keep_mask &= m
                log(f"RORノイズ除去適用: 残り {np.sum(keep_mask):,} 点")

        # 5. 孤立クラスタ（浮遊小塊ノイズ）除去
        if self.config.use_cluster_filter:
            current = np.where(keep_mask)[0]
            if len(current) > 0:
                c_idx = filter_isolated_clusters(
                    points[current],
                    eps=self.config.cluster_eps,
                    min_cluster_points=self.config.cluster_min_points,
                )
                m = np.zeros(total, dtype=bool)
                m[current[c_idx]] = True
                keep_mask &= m
                log(f"孤立クラスタ除去適用: 残り {np.sum(keep_mask):,} 点")

        # 6. エッジ飛散・ヒゲ（ベールノイズ）精密除去
        if self.config.use_edge_veil_filter:
            current = np.where(keep_mask)[0]
            if len(current) > 0:
                e_idx = filter_edge_veil_noise(
                    points[current],
                    nb_neighbors=self.config.edge_neighbors,
                    linearity_threshold=self.config.edge_linearity_thresh,
                    sor_std_ratio=self.config.edge_sor_std_ratio,
                )
                m = np.zeros(total, dtype=bool)
                m[current[e_idx]] = True
                keep_mask &= m
                log(f"エッジ飛散・ヒゲ精密除去適用: 残り {np.sum(keep_mask):,} 点")

        keep_indices = np.where(keep_mask)[0]
        removed_indices = np.where(~keep_mask)[0]
        return keep_indices, removed_indices

    def process_file(
        self,
        input_path: str,
        output_path: str,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> ProcessSummary:
        """
        E57ファイルを読み込み、ノイズ処理を実行して新規E57に書き出す。

        :param input_path: 入力E57パス
        :param output_path: 出力E57パス
        :param progress_callback: 進捗率 (0.0〜1.0) とメッセージを受け取るコールバック
        :return: 処理サマリー (ProcessSummary)
        """
        start_time = time.time()

        def update_progress(ratio: float, message: str):
            if progress_callback:
                progress_callback(ratio, message)

        update_progress(0.05, f"ファイル読み込み中: {input_path}")
        scans = read_e57_scans(input_path)
        scan_count = len(scans)

        update_progress(0.15, f"スキャン数 {scan_count} 件のデータを解析開始")

        filtered_scans: List[E57ScanData] = []
        scan_results: List[ScanProcessResult] = []

        total_init = sum(s.point_count for s in scans)
        total_final = 0

        for idx, scan in enumerate(scans):
            scan_start = time.time()
            init_pts = scan.point_count
            update_progress(
                0.20 + 0.60 * (idx / scan_count),
                f"スキャン #{idx + 1}/{scan_count} 処理中 (元点数: {init_pts:,} 点)...",
            )

            filtered_scan, keep_idx, remove_idx = self.apply_filters_to_scan(scan)
            filtered_scans.append(filtered_scan)

            final_pts = filtered_scan.point_count
            removed_pts = init_pts - final_pts
            ratio = (removed_pts / init_pts * 100.0) if init_pts > 0 else 0.0
            scan_time = time.time() - scan_start

            scan_results.append(
                ScanProcessResult(
                    scan_index=idx,
                    initial_points=init_pts,
                    final_points=final_pts,
                    removed_points=removed_pts,
                    removal_ratio=ratio,
                    time_taken_sec=scan_time,
                )
            )
            total_final += final_pts

        update_progress(0.85, f"ノイズ処理完了。出力ファイル書き込み中: {output_path}")
        write_e57_scans(output_path, filtered_scans)

        total_time = time.time() - start_time
        total_removed = total_init - total_final
        total_ratio = (total_removed / total_init * 100.0) if total_init > 0 else 0.0

        update_progress(1.0, "すべての処理が正常に完了しました。")

        return ProcessSummary(
            input_file=input_path,
            output_file=output_path,
            scan_results=scan_results,
            total_initial_points=total_init,
            total_final_points=total_final,
            total_removed_points=total_removed,
            total_removal_ratio=total_ratio,
            total_time_taken_sec=total_time,
        )


# 座標系変換の定義表: 新しい軸 i の値 = sign[i] * 元の軸 src[i] の値
# （軸の入替と符号反転のみで構成されるため、浮動小数点でも完全に可逆・ロスレス）
#   - "cartesian":       数学座標系 (X, Y, Z そのまま)
#   - "survey_swap":     測量座標系 → 数学座標系 (X=Y, Y=X, Z=Z: 測量X=北, Y=東を数学X=東, Y=北へ入替)
#   - "survey_swap_inv": 測量系反転 (X=Y, Y=-X, Z=Z)
#   - "y_up_to_z_up":    Y-up → Z-up 変換 (X=X, Y=-Z, Z=Y)
#   - "z_invert":        Z軸反転 (X=X, Y=Y, Z=-Z)
_COORD_MODE_TABLE: Dict[str, Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = {
    "cartesian": ((0, 1, 2), (1, 1, 1)),
    "survey_swap": ((1, 0, 2), (1, 1, 1)),
    "survey_swap_inv": ((1, 0, 2), (1, -1, 1)),
    "y_up_to_z_up": ((0, 2, 1), (1, -1, 1)),
    "z_invert": ((0, 1, 2), (1, 1, -1)),
}

_CARTESIAN_FIELDS = ("cartesianX", "cartesianY", "cartesianZ")


def _coord_mode_axes(coord_mode: Optional[str]) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """座標系モード名から (src, sign) を返す。未知のモードは恒等変換として扱う。"""
    if coord_mode in ("", None):
        coord_mode = "cartesian"
    return _COORD_MODE_TABLE.get(coord_mode, _COORD_MODE_TABLE["cartesian"])


def is_identity_coord_mode(coord_mode: Optional[str]) -> bool:
    """座標を変化させないモード（cartesian / 未指定 / 未知）なら True"""
    return _coord_mode_axes(coord_mode) == _COORD_MODE_TABLE["cartesian"]


def inverse_coord_axes(coord_mode: Optional[str]) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """
    coord_mode の逆変換を (src, sign) 形式で返す。
    new[i] = sign[i] * old[src[i]]  ⇔  old[src[i]] = sign[i] * new[i]
    """
    src, sign = _coord_mode_axes(coord_mode)
    inv_src = [0, 0, 0]
    inv_sign = [1, 1, 1]
    for i in range(3):
        inv_src[src[i]] = i
        inv_sign[src[i]] = sign[i]
    return (inv_src[0], inv_src[1], inv_src[2]), (inv_sign[0], inv_sign[1], inv_sign[2])


def _apply_axes(points: np.ndarray, axes: Tuple[Tuple[int, int, int], Tuple[int, int, int]]) -> np.ndarray:
    """(src, sign) に従って (N, 3) 配列を変換した新しい配列を返す"""
    src, sign = axes
    new_pts = np.empty_like(points)
    for i in range(3):
        if sign[i] >= 0:
            new_pts[:, i] = points[:, src[i]]
        else:
            np.negative(points[:, src[i]], out=new_pts[:, i])
    return new_pts


def transform_coordinate_system(points: np.ndarray, coord_mode: str) -> np.ndarray:
    """
    点群の3次元座標配列を取り込み設定に応じて変換する（モード一覧は _COORD_MODE_TABLE を参照）。

    Parameters:
    - points: (N, 3) float64 座標配列
    - coord_mode: "cartesian" / "survey_swap" / "survey_swap_inv" / "y_up_to_z_up" / "z_invert"
    """
    if points is None or len(points) == 0 or is_identity_coord_mode(coord_mode):
        return points
    return _apply_axes(points, _coord_mode_axes(coord_mode))


def inverse_transform_coordinate_system(points: np.ndarray, coord_mode: str) -> np.ndarray:
    """transform_coordinate_system(points, coord_mode) の逆変換（元の座標系へ戻す）"""
    if points is None or len(points) == 0 or is_identity_coord_mode(coord_mode):
        return points
    return _apply_axes(points, inverse_coord_axes(coord_mode))


def scan_pose_transform(scan) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    スキャンの pose を (回転行列 R, 平行移動 t) で返す。world = R @ local + t。
    pose が無い、または恒等変換なら None（呼び出し側は座標をコピーせずに済む）。
    """
    rotation = getattr(scan, "rotation", None)
    translation = getattr(scan, "translation", None)
    t = np.zeros(3, dtype=np.float64) if translation is None else np.asarray(translation, dtype=np.float64)
    R = np.eye(3, dtype=np.float64)
    if rotation is not None:
        q = np.asarray(rotation, dtype=np.float64)
        norm = float(np.linalg.norm(q))
        if norm > 0.0:
            w, x, y, z = q / norm
            R = np.array(
                [
                    [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                    [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                    [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
                ],
                dtype=np.float64,
            )
    if np.array_equal(R, np.eye(3)) and not np.any(t):
        return None
    return R, t


def scan_display_points(scan) -> np.ndarray:
    """
    スキャンの表示用座標を返す: 生座標に pose を適用し、そのあと取り込み時の座標系変換を適用する。
    scan.points は保存に使う生座標のまま変更しない（ロスレス保存のため）。
    pose が恒等変換なら scan.points をそのまま返し、メモリを増やさない。
    """
    pose = scan_pose_transform(scan)
    if pose is None or len(scan.points) == 0:
        return scan.points
    R, t = pose
    mode = getattr(scan, "coord_mode", "cartesian")
    world = inverse_transform_coordinate_system(scan.points, mode).dot(R.T)
    world += t
    return transform_coordinate_system(world, mode)


def scan_display_origin(scan) -> np.ndarray:
    """スキャンのローカル原点（距離フィルタの基準）の表示用座標を返す"""
    pose = scan_pose_transform(scan)
    origin = np.zeros((1, 3), dtype=np.float64) if pose is None else pose[1].reshape(1, 3)
    return transform_coordinate_system(origin, getattr(scan, "coord_mode", "cartesian"))[0]


def _negate_schema(schema):
    """
    ScaledInteger / Float の FieldSchema を「値の符号を反転した軸」用に変換した複製を返す。
    v = raw * scale + offset  →  -v = (-raw) * scale + (-offset)
    （raw 整数は呼び出し側で符号反転する。scale は変えないため量子化幅は完全に保たれる）
    """
    import copy

    s = copy.copy(schema)
    if schema.node_type == "ScaledInteger":
        s.offset = -float(schema.offset)
        if schema.minimum is not None and schema.maximum is not None:
            s.minimum, s.maximum = -int(schema.maximum), -int(schema.minimum)
        if schema.default_value is not None:
            s.default_value = -int(schema.default_value)
    elif schema.node_type == "Float":
        if schema.minimum is not None and schema.maximum is not None:
            s.minimum, s.maximum = -float(schema.maximum), -float(schema.minimum)
    return s


def _apply_axes_to_scan(scan, axes: Tuple[Tuple[int, int, int], Tuple[int, int, int]]) -> None:
    """
    E57ScanData の座標 (points / raw_fields / raw_unscaled_fields / field_schemas の cartesianX,Y,Z) を
    (src, sign) に従って一貫してインプレース変換する。
    ScaledInteger の生整数とスケール定義も軸ごとに付け替えるため、変換後も
    「points == raw_unscaled * scale + offset」が厳密に成り立ち、ロスレス性が保たれる。
    """
    src, sign = axes
    scan.points = _apply_axes(scan.points, axes)

    raw_fields = getattr(scan, "raw_fields", None)
    if isinstance(raw_fields, dict) and all(k in raw_fields for k in _CARTESIAN_FIELDS):
        # float64 の座標フィールドは points の列ビューに差し替える（元実装と同じメモリ節約方針）。
        # E57_SINGLE (float32) は単精度配列のまま軸を付け替える。
        old_fields = {k: raw_fields[k] for k in _CARTESIAN_FIELDS}
        for i, key in enumerate(_CARTESIAN_FIELDS):
            old = old_fields[_CARTESIAN_FIELDS[src[i]]]
            if isinstance(old, np.ndarray) and old.dtype == np.float32:
                raw_fields[key] = (-old) if sign[i] < 0 else old
            else:
                raw_fields[key] = scan.points[:, i]

    raw_unscaled = getattr(scan, "raw_unscaled_fields", None)
    schemas = getattr(scan, "field_schemas", None)
    if isinstance(schemas, dict):
        old_schemas = {k: schemas.get(k) for k in _CARTESIAN_FIELDS}
        old_unscaled = {k: (raw_unscaled.get(k) if isinstance(raw_unscaled, dict) else None) for k in _CARTESIAN_FIELDS}
        for i, key in enumerate(_CARTESIAN_FIELDS):
            src_key = _CARTESIAN_FIELDS[src[i]]
            sch = old_schemas[src_key]
            if sch is not None:
                schemas[key] = _negate_schema(sch) if sign[i] < 0 else sch
            elif key in schemas:
                del schemas[key]
            if isinstance(raw_unscaled, dict):
                arr = old_unscaled[src_key]
                if arr is not None:
                    raw_unscaled[key] = (-arr) if sign[i] < 0 else arr
                elif key in raw_unscaled:
                    del raw_unscaled[key]


def transform_scan_coordinate_system(scan, coord_mode: str) -> None:
    """
    単一の E57ScanData の内部座標を指定の座標系へ一括インプレース変換する（取り込み時用）。
    変換後は scan.coord_mode に適用モードを記録し、保存時に restore_scan_original_coordinates() で
    元の座標系へ厳密に戻せるようにする。座標系変換は表示・編集用であり、E57 への保存は元の座標系で行う。
    """
    if scan is None or getattr(scan, "points", None) is None or len(scan.points) == 0:
        return
    if is_identity_coord_mode(coord_mode):
        return
    current = getattr(scan, "coord_mode", "cartesian")
    if not is_identity_coord_mode(current):
        # 二重変換を防ぐため、いったん元の座標系へ戻してから適用する
        restore_scan_original_coordinates(scan)

    _apply_axes_to_scan(scan, _coord_mode_axes(coord_mode))
    scan.coord_mode = coord_mode
    # 軸入替はシグネチャ（和・最小・最大）では検知できないため、変更済みであることを明示する
    if hasattr(scan, "mark_coordinates_modified"):
        scan.mark_coordinates_modified()


def restore_scan_original_coordinates(scan) -> str:
    """
    transform_scan_coordinate_system() で変換した E57ScanData を元の座標系へインプレースで戻す。
    戻り値は戻す前に適用されていた coord_mode（再変換に使う。未変換なら "cartesian"）。
    復元後は読み込み時の生データ（ScaledInteger の生整数・スキーマ）と完全に一致するため、
    E57 書き出しはビット完全一致のロスレスとなる。
    """
    if scan is None:
        return "cartesian"
    mode = getattr(scan, "coord_mode", "cartesian")
    if is_identity_coord_mode(mode):
        return "cartesian"
    if getattr(scan, "points", None) is not None and len(scan.points) > 0:
        _apply_axes_to_scan(scan, inverse_coord_axes(mode))
    scan.coord_mode = "cartesian"
    # 元の座標に戻ったので、読み込み時シグネチャとの比較を再び有効化する
    if hasattr(scan, "_initial_points_signature") and scan._initial_points_signature is None:
        from core.e57_io import _compute_points_signature

        scan._initial_points_signature = _compute_points_signature(scan.points)
    return mode

