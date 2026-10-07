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

        keep_indices = np.where(keep_mask)[0]
        removed_indices = np.where(~keep_mask)[0]

        filtered_scan = scan.filter_by_indices(keep_indices)
        return filtered_scan, keep_indices, removed_indices

    def apply_filters_to_points(
        self,
        points: np.ndarray,
        log_callback: Optional[Callable[[str], None]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        メモリ上の3D点群座標に対して各種フィルタを直接適用し、(keep_indices, removed_indices) を返す。
        手動削除後の通常レイヤー点群に対して実行でき、手動削除点が復活する問題を根本解決します。
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
                    points[current],
                    min_distance=self.config.min_distance,
                    max_distance=self.config.max_distance,
                )
                m = np.zeros(total, dtype=bool)
                m[current[d_idx]] = True
                keep_mask &= m
                log(f"距離フィルタ適用: 残り {np.sum(keep_mask):,} 点")

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
