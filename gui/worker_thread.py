"""
GUIの応答性を保つための非同期ワーカースレッドモジュール
"""

from typing import Optional
from PySide6.QtCore import QThread, Signal
import numpy as np

from core.processor import FilterConfig, PointCloudProcessor, ProcessSummary
from core.e57_io import read_e57_scans


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
        """処理のキャンセル要求"""
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


class PreviewWorker(QThread):
    """3Dプレビュー用に点群とノイズ分類を抽出するバックグラウンドワーカースレッド"""

    preview_ready = Signal(object, object)  # (inliers_pts, outliers_pts)
    preview_error = Signal(str)

    def __init__(
        self,
        input_path: str,
        config: FilterConfig,
        scan_index: int = 0,
        max_preview_points: int = 500000,
        parent=None,
    ):
        super().__init__(parent)
        self.input_path = input_path
        self.config = config
        self.scan_index = scan_index
        self.max_preview_points = max_preview_points

    def run(self):
        try:
            scans = read_e57_scans(self.input_path)
            if not scans:
                raise ValueError("E57ファイル内にスキャンが見つかりません。")

            target_scan = scans[min(self.scan_index, len(scans) - 1)]
            processor = PointCloudProcessor(config=self.config)

            # 点数が多すぎる場合はプレビュー用に間引いてから計算
            scan_to_process = target_scan
            pts = target_scan.points
            if len(pts) > self.max_preview_points:
                step = len(pts) // self.max_preview_points
                sub_indices = np.arange(0, len(pts), step)
                scan_to_process = target_scan.filter_by_indices(sub_indices)

            filtered_scan, keep_idx, remove_idx = processor.apply_filters_to_scan(scan_to_process)

            inliers = scan_to_process.points[keep_idx]
            outliers = scan_to_process.points[remove_idx]

            self.preview_ready.emit(inliers, outliers)
        except Exception as e:
            self.preview_error.emit(str(e))
