"""
GUIの応答性を保つための非同期ワーカースレッドモジュール
巨大点群(2,700万点超)の読み込み、座標変換、前処理をバックグラウンドスレッドで
完全非同期実行し、UIのフリーズを100%防止します。
"""

import math
import os
import time
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from PySide6.QtCore import QThread, Signal
import numpy as np

from core.processor import FilterConfig, PointCloudProcessor, ProcessSummary
from core.e57_io import E57ScanData, read_e57_scans


@dataclass
class PreloadedData:
    """バックグラウンドで前処理完了済みの点群データオブジェクト"""
    scans: List[E57ScanData]
    points_raw: np.ndarray          # (N, 3) float64 原本
    points_centered: np.ndarray     # (N, 3) float32 中心オフセット済み
    center_offset: np.ndarray       # (3,) float64
    colors_raw: Optional[np.ndarray]
    intensities_raw: Optional[np.ndarray]
    active_colors: np.ndarray       # (N, 3) float32
    world_bounds_min: np.ndarray    # (3,) float64
    world_bounds_max: np.ndarray    # (3,) float64
    bounds_min: np.ndarray          # (3,) float32
    bounds_max: np.ndarray          # (3,) float32
    # 表示用サンプリング配列
    display_points: np.ndarray
    display_colors: np.ndarray
    display_orig_indices: Optional[np.ndarray]


class ProcessWorker(QThread):
    """E57ノイズ除去処理を実行するバックグラウンドワーカースレッド"""

    progress_changed = Signal(int, str)  # (進捗率0-100, メッセージ)
    log_message = Signal(str)           # ログメッセージ
    finished_success = Signal(object)    # ProcessSummary
    finished_error = Signal(str)         # エラーメッセージ

    def __init__(
        self,
        input_path: str,
        output_path: str,
        config: FilterConfig,
        parent=None,
    ):
        super().__init__(parent)
        self.input_path = input_path
        self.output_path = output_path
        self.config = config
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        try:
            processor = PointCloudProcessor(config=self.config)

            def progress_callback(ratio: float, message: str):
                if self._is_cancelled:
                    raise InterruptedError("処理がユーザーによって中止されました。")
                percent = int(ratio * 100)
                self.progress_changed.emit(percent, message)
                self.log_message.emit(f"[{percent:3d}%] {message}")

            summary = processor.process_file(
                self.input_path,
                self.output_path,
                progress_callback=progress_callback,
            )
            self.finished_success.emit(summary)
        except Exception as e:
            self.finished_error.emit(str(e))


class FileLoadWorker(QThread):
    """
    E57ファイルの読み込み＆全前処理をバックグラウンドスレッドで完結させるワーカード
    巨大データでもUIスレッドを1ミリ秒もフリーズさせません。
    """

    progress_changed = Signal(int, str, str)  # (0-100%, メインメッセージ, サブメッセージ)
    finished_success = Signal(object)         # PreloadedData
    finished_error = Signal(str)              # エラーメッセージ

    def __init__(self, file_path: str, point_budget: int = 2000000, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self.point_budget = point_budget
        self._is_cancelled = False

    def run(self):
        try:
            fname = os.path.basename(self.file_path)
            self.progress_changed.emit(5, "E57ヘッダー解析中...", f"ファイル: {fname}")

            # 1. E57スキャンの読み込み
            def on_e57_progress(ratio: float, msg: str):
                pct = int(5 + ratio * 60)  # 5% - 65%
                self.progress_changed.emit(pct, msg, f"ファイル: {fname}")

            scans = read_e57_scans(self.file_path, progress_callback=on_e57_progress)
            if not scans:
                raise ValueError("E57ファイル内にスキャンデータが見つかりませんでした。")

            total_pts = sum(s.point_count for s in scans)
            self.progress_changed.emit(70, "全スキャン点群の統合中...", f"総点数: {total_pts:,} 点")

            # 2. 点群・属性配列の結合 (NumPy高速処理)
            all_pts = []
            all_cols = []
            all_ints = []

            for s in scans:
                all_pts.append(s.points)
                if "colorRed" in s.raw_fields and "colorGreen" in s.raw_fields and "colorBlue" in s.raw_fields:
                    r = s.raw_fields["colorRed"]
                    g = s.raw_fields["colorGreen"]
                    b = s.raw_fields["colorBlue"]
                    all_cols.append(np.column_stack([r, g, b]))
                if "intensity" in s.raw_fields:
                    all_ints.append(s.raw_fields["intensity"])

            combined_pts = np.vstack(all_pts) if all_pts else np.empty((0, 3), dtype=np.float64)
            combined_cols = np.vstack(all_cols) if len(all_cols) == len(all_pts) else None
            combined_ints = np.concatenate(all_ints) if len(all_ints) == len(all_pts) else None

            self.progress_changed.emit(80, "中心オフセット & 座標系計算中...", "数学座標系(Z-up)へマッピング中")

            # 3. 中心オフセットとバウンディングボックスの計算
            min_b = np.min(combined_pts, axis=0)
            max_b = np.max(combined_pts, axis=0)
            world_bounds_min = min_b.copy()
            world_bounds_max = max_b.copy()

            center_offset = (min_b + max_b) / 2.0
            points_centered = np.ascontiguousarray((combined_pts - center_offset), dtype=np.float32)

            bounds_min = (min_b - center_offset).astype(np.float32)
            bounds_max = (max_b - center_offset).astype(np.float32)

            # カラー正規化
            n = len(combined_pts)
            if combined_cols is not None and len(combined_cols) == n:
                c = np.asarray(combined_cols, dtype=np.float32)
                if np.max(c) > 1.0:
                    c = c / 255.0
                colors_raw = np.clip(c, 0.0, 1.0)
                active_colors = colors_raw.copy()
            else:
                colors_raw = None
                active_colors = np.tile([0.3, 0.8, 0.95], (n, 1)).astype(np.float32)

            if combined_ints is not None and len(combined_ints) == n:
                intensities_raw = np.asarray(combined_ints, dtype=np.float32)
            else:
                intensities_raw = None

            self.progress_changed.emit(90, "高速描画用バジェット配列生成中...", f"表示上限: {self.point_budget:,} 点")

            # 4. 表示用サンプリング配列 (Point Budget)
            budget = self.point_budget
            if budget <= 0 or n <= budget:
                display_points = points_centered
                display_colors = active_colors
                display_orig_indices = None
            else:
                step = max(1, int(math.ceil(n / budget)))
                display_orig_indices = np.arange(0, n, step)
                display_points = np.ascontiguousarray(points_centered[display_orig_indices])
                display_colors = np.ascontiguousarray(active_colors[display_orig_indices])

            self.progress_changed.emit(100, "読み込み完了！", f"{total_pts:,} 点を即座に描画します")

            preloaded = PreloadedData(
                scans=scans,
                points_raw=combined_pts,
                points_centered=points_centered,
                center_offset=center_offset,
                colors_raw=colors_raw,
                intensities_raw=intensities_raw,
                active_colors=active_colors,
                world_bounds_min=world_bounds_min,
                world_bounds_max=world_bounds_max,
                bounds_min=bounds_min,
                bounds_max=bounds_max,
                display_points=display_points,
                display_colors=display_colors,
                display_orig_indices=display_orig_indices,
            )

            self.finished_success.emit(preloaded)
        except Exception as e:
            self.finished_error.emit(str(e))


class MemoryFilterWorker(QThread):
    """メモリ上の通常レイヤー点群に対して直接ノイズ除去を実行するバックグラウンドスレッド"""

    progress_changed = Signal(int, str, str)
    log_message = Signal(str)
    finished_success = Signal(object, object)  # (keep_indices, removed_indices)
    finished_error = Signal(str)

    def __init__(self, points: np.ndarray, config: FilterConfig, parent=None):
        super().__init__(parent)
        self.points = points
        self.config = config

    def run(self):
        try:
            total = len(self.points)
            self.progress_changed.emit(10, "自動ノイズ解析開始...", f"対象: {total:,} 点 (通常レイヤー)")
            self.log_message.emit(f"通常レイヤー点群の自動ノイズ処理を開始: {total:,} 点")

            processor = PointCloudProcessor(config=self.config)

            def log_cb(msg: str):
                self.log_message.emit(msg)
                self.progress_changed.emit(60, "フィルタ適用中...", msg)

            keep_idx, remove_idx = processor.apply_filters_to_points(self.points, log_callback=log_cb)

            self.progress_changed.emit(100, "ノイズ検出完了！", f"除去: {len(remove_idx):,} 点をノイズレイヤーへ移動")
            self.finished_success.emit(keep_idx, remove_idx)
        except Exception as e:
            self.finished_error.emit(str(e))

