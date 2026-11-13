"""
GUIの応答性を保つための非同期ワーカースレッドモジュール
巨大点群(2,700万点超)の読み込み、座標変換、前処理をバックグラウンドスレッドで
完全非同期実行し、UIのフリーズを100%防止します。
"""

import math
import os
import time
import inspect
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Callable
from PySide6.QtCore import QThread, Signal
import numpy as np

from core.logger import log_exception, log_info, log_warning

from core.processor import (
    FilterConfig,
    PointCloudProcessor,
    ProcessSummary,
    transform_scan_coordinate_system,
    restore_scan_original_coordinates,
    inverse_transform_coordinate_system,
    is_identity_coord_mode,
    scan_display_points,
    scan_pose_transform,
)
from core.filters import keep_largest_clusters
from core.e57_io import E57ScanData, E57StreamWriter, file_signature, read_e57_scans, write_e57_scans
from core.fullres_transfer import ORPHAN_LABEL, LabelTransfer, TransferStats, estimate_stride, labels_from_masks
from core.e57_stream import (
    DEFAULT_STREAM_CHUNK,
    E57SchemaBoundsError,
    ScanStreamSink,
    iter_scan_stream_chunks,
    measure_field_ranges,
    scan_stream_template,
)


LOAD_OUT_OF_MEMORY_TEXT = (
    "メモリが足りず、読み込めませんでした。\n"
    "設定 → システムの「大容量点群読込上限」を下げるか、ほかのアプリを閉じてから、もう一度開いてください。"
)


def collect_e57_files(path: str) -> List[str]:
    """
    ファイルパスならそのまま、フォルダパスならフォルダ直下の *.e57 を名前順で返す。
    (サブフォルダは対象外)
    """
    if not path:
        return []
    if os.path.isdir(path):
        files = [
            os.path.join(path, f)
            for f in os.listdir(path)
            if f.lower().endswith(".e57") and os.path.isfile(os.path.join(path, f))
        ]
        return sorted(files, key=lambda p: os.path.basename(p).lower())
    return [path] if os.path.isfile(path) else []


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
    scan_ids: Optional[np.ndarray] = None  # (N,) int32 各点が属するスキャン番号
    is_downsampled: bool = False           # 大容量メモリ保護サンプリングが適用されたか
    original_total_pts: int = 0            # 元ファイルの総点数


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
            # ユーザーには要約を、ログにはトレースバック付きで記録する（原因調査のため）
            log_exception(f"{type(self).__name__} でエラー: {e}")
            self.finished_error.emit(str(e))


class FileLoadWorker(QThread):
    """
    E57ファイルの読み込み＆全前処理をバックグラウンドスレッドで完結させるワーカード
    巨大データでもUIスレッドを1ミリ秒もフリーズさせません。
    """

    progress_changed = Signal(int, str, str)  # (0-100%, メインメッセージ, サブメッセージ)
    finished_success = Signal(object)         # PreloadedData
    finished_error = Signal(str)              # エラーメッセージ

    def __init__(
        self,
        file_path: str,
        point_budget: int = 2000000,
        max_load_points: Optional[int] = None,
        coord_mode: str = "cartesian",
        parent=None,
    ):
        super().__init__(parent)
        self.file_path = file_path
        self.point_budget = point_budget
        # 実務でのデータ欠損を防ぐため、デフォルト(None)では全点ロスレス読み込みを行う
        # （point_budget は3D描画用の表示間引きのみに使用し、元データは1点も減らさない）
        self.max_load_points = max_load_points
        self.coord_mode = coord_mode
        self._is_cancelled = False

    def cancel(self):
        """読み込みの中断を要求（ファイル境界および E57 読み込みの進捗コールバックで反映される）"""
        self._is_cancelled = True

    def run(self):
        try:
            # 1. E57スキャンの読み込み (フォルダ指定時はフォルダ内の全E57を統合)
            file_list = collect_e57_files(self.file_path)
            if not file_list:
                raise ValueError(f"E57ファイルが見つかりませんでした: {self.file_path}")
            n_files = len(file_list)
            scans: List[E57ScanData] = []

            # 複数ファイル統合時かつ max_load_points 指定がある場合はファイル数で按分
            per_file_load_budget = (
                max(1, self.max_load_points // n_files)
                if (self.max_load_points is not None and self.max_load_points > 0)
                else None
            )

            for f_i, fpath in enumerate(file_list):
                if self._is_cancelled:
                    raise InterruptedError("読み込みがユーザーによって中止されました。")
                fname = os.path.basename(fpath)
                sub = f"ファイル {f_i + 1}/{n_files}: {fname}"
                self.progress_changed.emit(int(5 + 60 * f_i / n_files), "E57ヘッダー解析中...", sub)

                def on_e57_progress(ratio: float, msg: str, _fi=f_i, _sub=sub):
                    if self._is_cancelled:
                        raise InterruptedError("読み込みがユーザーによって中止されました。")
                    pct = int(5 + 60 * (_fi + ratio) / n_files)  # 5% - 65%
                    self.progress_changed.emit(pct, msg, _sub)

                file_scans = read_e57_scans(
                    fpath,
                    progress_callback=on_e57_progress,
                    point_budget=per_file_load_budget,
                )
                base = os.path.splitext(fname)[0]
                for s in file_scans:
                    if n_files > 1:
                        # 器械点一覧で元ファイルが分かるように「ファイル名 / スキャン名」とする
                        if len(file_scans) == 1 or not s.name:
                            s.name = base if len(file_scans) == 1 else f"{base} / Scan {s.scan_index}"
                        else:
                            s.name = f"{base} / {s.name}"
                    s.scan_index = len(scans)
                    if self.coord_mode and self.coord_mode != "cartesian":
                        transform_scan_coordinate_system(s, self.coord_mode)
                    scans.append(s)

            if not scans:
                raise ValueError("E57ファイル内にスキャンデータが見つかりませんでした。")

            total_pts = sum(s.point_count for s in scans)
            original_total_pts = sum(getattr(s, "original_point_count", s.point_count) for s in scans)
            is_downsampled = any(getattr(s, "is_subsampled", False) for s in scans)

            if total_pts == 0:
                raise ValueError("E57ファイル内の有効な点数が 0 点です。")

            self.progress_changed.emit(70, "全スキャン点群の統合中...", f"読み込み点数: {total_pts:,} 点")

            # 2. 点群・属性配列の結合（事前確保と単一スキャン時の参照共有によりピークメモリ倍増を完全防止）
            non_empty_scans = [(s_idx, s) for s_idx, s in enumerate(scans) if s.point_count > 0]
            has_any_color = any(
                "colorRed" in s.raw_fields and "colorGreen" in s.raw_fields and "colorBlue" in s.raw_fields
                for _, s in non_empty_scans
            )
            has_any_int = any("intensity" in s.raw_fields for _, s in non_empty_scans)

            # 表示・計測・断面には pose を適用した座標を使う。s.points は保存用の生座標として残す
            n_posed = sum(1 for _, s in non_empty_scans if scan_pose_transform(s) is not None)
            if n_posed:
                log_info(f"E57 の pose を表示座標へ適用します: {n_posed} / {len(non_empty_scans)} スキャン")

            if len(non_empty_scans) == 1:
                # 単一スキャンで pose が恒等なら s.points を参照共有し、倍精度座標配列の二重コピーをゼロにする
                only_idx, only_scan = non_empty_scans[0]
                combined_pts = scan_display_points(only_scan)
                combined_scan_ids = np.full(total_pts, only_idx, dtype=np.int32)
            else:
                combined_pts = np.empty((total_pts, 3), dtype=np.float64)
                combined_scan_ids = np.empty(total_pts, dtype=np.int32)
                offset_cursor = 0
                for s_idx, s in non_empty_scans:
                    n_cur = len(s.points)
                    combined_pts[offset_cursor : offset_cursor + n_cur] = scan_display_points(s)
                    combined_scan_ids[offset_cursor : offset_cursor + n_cur] = s_idx
                    offset_cursor += n_cur

            colors_raw: Optional[np.ndarray] = None
            if has_any_color:
                colors_raw = np.empty((total_pts, 3), dtype=np.float32)
                offset_cursor = 0
                for _, s in non_empty_scans:
                    n_cur = len(s.points)
                    end_c = offset_cursor + n_cur
                    if "colorRed" in s.raw_fields and "colorGreen" in s.raw_fields and "colorBlue" in s.raw_fields:
                        colors_raw[offset_cursor:end_c, 0] = s.raw_fields["colorRed"]
                        colors_raw[offset_cursor:end_c, 1] = s.raw_fields["colorGreen"]
                        colors_raw[offset_cursor:end_c, 2] = s.raw_fields["colorBlue"]
                        sub_c = colors_raw[offset_cursor:end_c]
                        max_c = float(np.max(sub_c)) if n_cur > 0 else 1.0
                        if max_c > 255.0:
                            sub_c /= 65535.0
                        elif max_c > 1.0:
                            sub_c /= 255.0
                        np.clip(sub_c, 0.0, 1.0, out=sub_c)
                    else:
                        colors_raw[offset_cursor:end_c] = (0.3, 0.8, 0.95)
                    offset_cursor = end_c

            intensities_raw: Optional[np.ndarray] = None
            if has_any_int:
                intensities_raw = np.empty(total_pts, dtype=np.float32)
                offset_cursor = 0
                for _, s in non_empty_scans:
                    n_cur = len(s.points)
                    end_c = offset_cursor + n_cur
                    if "intensity" in s.raw_fields:
                        intensities_raw[offset_cursor:end_c] = s.raw_fields["intensity"]
                        # 非有限の反射強度が含まれていた場合は 0.0 にサニタイズ
                        sub_i = intensities_raw[offset_cursor:end_c]
                        if not np.isfinite(sub_i).all():
                            np.nan_to_num(sub_i, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                    else:
                        intensities_raw[offset_cursor:end_c] = 0.0
                    offset_cursor = end_c

            import gc
            gc.collect()

            self.progress_changed.emit(80, "中心オフセット & 座標系計算中...", "数学座標系(Z-up)へマッピング中")

            # 3. 中心オフセットとバウンディングボックスの計算（非有限値に対する安全ガード付き）
            if not np.isfinite(combined_pts).all():
                np.nan_to_num(combined_pts, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            min_b = np.min(combined_pts, axis=0)
            max_b = np.max(combined_pts, axis=0)
            world_bounds_min = min_b.copy()
            world_bounds_max = max_b.copy()

            center_offset = (min_b + max_b) / 2.0
            points_centered = np.ascontiguousarray((combined_pts - center_offset), dtype=np.float32)

            bounds_min = (min_b - center_offset).astype(np.float32)
            bounds_max = (max_b - center_offset).astype(np.float32)

            # カラー正規化 (二重コピーを排除して参照共有)
            n = len(combined_pts)
            if colors_raw is not None and len(colors_raw) == n:
                active_colors = colors_raw
            else:
                colors_raw = None
                active_colors = np.tile([0.3, 0.8, 0.95], (n, 1)).astype(np.float32)

            self.progress_changed.emit(90, "高速描画用バジェット配列生成中...", f"表示点数: {n:,} 点")

            # 4. 表示用サンプリング配列 (Point Budget)
            budget = self.point_budget
            if n <= budget or budget <= 0:
                display_points = points_centered
                display_colors = active_colors
                display_orig_indices = np.arange(n, dtype=np.int64)
            else:
                step = max(1, int(math.ceil(n / budget)))
                display_orig_indices = np.arange(0, n, step, dtype=np.int64)
                display_points = np.ascontiguousarray(points_centered[display_orig_indices])
                display_colors = np.ascontiguousarray(active_colors[display_orig_indices])

            gc.collect()

            self.progress_changed.emit(100, "読み込み完了！", f"{n:,} 点を即座に描画します")

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
                scan_ids=combined_scan_ids,
                is_downsampled=is_downsampled,
                original_total_pts=original_total_pts,
            )

            self.finished_success.emit(preloaded)
        except MemoryError as e:
            # numpy や E57 のライブラリが配列を確保できなかった。英語の「Unable to allocate …」のままでは次の一手が分からない
            log_exception(f"{type(self).__name__} でメモリ不足: {e}")
            self.finished_error.emit(LOAD_OUT_OF_MEMORY_TEXT)
        except Exception as e:
            # ユーザーには要約を、ログにはトレースバック付きで記録する（原因調査のため）
            log_exception(f"{type(self).__name__} でエラー: {e}")
            self.finished_error.emit(str(e))


class MemoryFilterWorker(QThread):
    """メモリ上の通常レイヤー点群に対して直接ノイズ除去を実行するバックグラウンドスレッド"""

    progress_changed = Signal(int, str, str)
    log_message = Signal(str)
    finished_success = Signal(object, object)  # (keep_indices, removed_indices)
    finished_stages = Signal(object)  # [(フィルタ名, 除去インデックス(points内の局所index)), ...]
    finished_error = Signal(str)
    cancelled = Signal()

    # (表示名, 有効フラグ属性名) ※processor の適用順に合わせる
    STAGES = [
        ("距離", "use_distance_filter"),
        ("反射強度", "use_intensity_filter"),
        ("間引き", "use_voxel_downsample"),
        ("SOR", "use_sor"),
        ("ROR", "use_ror"),
        ("クラスタ", "use_cluster_filter"),
        ("エッジ飛散", "use_edge_veil_filter"),
    ]

    def __init__(
        self,
        points: np.ndarray,
        config: FilterConfig,
        intensities: Optional[np.ndarray] = None,
        parent=None,
        distance_points: Optional[np.ndarray] = None,
    ):
        super().__init__(parent)
        self.points = points
        self.config = config
        self.intensities = intensities
        self.distance_points = distance_points  # 距離フィルタ用のスキャン原点からの相対座標（省略可）
        self._is_cancelled = False

    def cancel(self):
        """処理の中断を要求"""
        self._is_cancelled = True

    def run(self):
        try:
            import dataclasses

            total = len(self.points)
            self.progress_changed.emit(10, "自動ノイズ解析開始...", f"対象: {total:,} 点 (通常レイヤー)")
            self.log_message.emit(f"通常レイヤー点群の自動ノイズ処理を開始: {total:,} 点")

            all_off = {attr: False for _, attr in self.STAGES}
            enabled = [(n, a) for n, a in self.STAGES if getattr(self.config, a, False)]
            remaining = np.arange(total, dtype=np.int64)
            stages = []
            all_removed = []

            for i, (name, attr) in enumerate(enabled):
                if self._is_cancelled:
                    self.log_message.emit("ユーザーによりノイズ処理が中断されました。")
                    self.cancelled.emit()
                    return
                if len(remaining) == 0:
                    break
                pct = min(90, 20 + int(70 * i / max(1, len(enabled))))
                self.progress_changed.emit(pct, f"{name}フィルタ適用中...", f"残り {len(remaining):,} 点")
                single_cfg = dataclasses.replace(self.config, **{**all_off, attr: True})
                processor = PointCloudProcessor(config=single_cfg)
                sub_int = self.intensities[remaining] if self.intensities is not None else None

                def on_filter_log(msg: str):
                    self.log_message.emit(msg)
                    self.progress_changed.emit(pct, f"{name}フィルタ処理中...", msg)

                keep_local, remove_local = processor.apply_filters_to_points(
                    self.points[remaining],
                    log_callback=on_filter_log,
                    intensities=sub_int,
                    distance_points=None if self.distance_points is None else self.distance_points[remaining],
                )
                removed_global = remaining[np.asarray(remove_local, dtype=np.int64)]
                if len(removed_global) > 0:
                    stages.append((name, removed_global))
                    all_removed.append(removed_global)
                remaining = remaining[np.asarray(keep_local, dtype=np.int64)]

            if self._is_cancelled:
                self.log_message.emit("ユーザーによりノイズ処理が中断されました。")
                self.cancelled.emit()
                return

            remove_idx = np.concatenate(all_removed) if all_removed else np.array([], dtype=np.int64)
            self.progress_changed.emit(100, "ノイズ検出完了！", f"除去: {len(remove_idx):,} 点をフィルタ別レイヤーへ移動")
            self.finished_stages.emit(stages)
            self.finished_success.emit(remaining, remove_idx)
        except Exception as e:
            # ユーザーには要約を、ログにはトレースバック付きで記録する（原因調査のため）
            log_exception(f"{type(self).__name__} でエラー: {e}")
            self.finished_error.emit(str(e))


class KeepMainClusterWorker(QThread):
    """メイン構造物（最大連結クラスタ）のみを残し、周囲の浮遊塊ノイズを一括分離するバックグラウンドスレッド"""
    progress_changed = Signal(int, str, str)
    finished_success = Signal(object)  # remove_indices (points内の局所インデックス)
    finished_error = Signal(str)
    cancelled = Signal()

    def __init__(
        self,
        points: np.ndarray,
        eps: float = 0.25,
        top_k: int = 1,
        min_keep_ratio: float = 0.05,
        parent=None,
    ):
        super().__init__(parent)
        self.points = points
        self.eps = eps
        self.top_k = top_k
        self.min_keep_ratio = min_keep_ratio
        self._is_cancelled = False

    def cancel(self):
        """処理の中断を要求"""
        self._is_cancelled = True

    def run(self):
        try:
            total = len(self.points)
            self.progress_changed.emit(20, "メイン構造物を解析中...", f"対象: {total:,} 点 (ボクセル量子化)")
            if self._is_cancelled:
                self.cancelled.emit()
                return

            keep_local = keep_largest_clusters(
                self.points,
                eps=self.eps,
                top_k=self.top_k,
                min_keep_ratio=self.min_keep_ratio,
            )

            if self._is_cancelled:
                self.cancelled.emit()
                return

            self.progress_changed.emit(80, "浮遊ノイズを分離中...", "クラスタ判定完了")
            keep_mask = np.zeros(total, dtype=bool)
            keep_mask[keep_local] = True
            remove_local = np.where(~keep_mask)[0]

            self.progress_changed.emit(100, "解析完了！", f"浮遊ノイズ: {len(remove_local):,} 点")
            self.finished_success.emit(remove_local)
        except Exception as e:
            # ユーザーには要約を、ログにはトレースバック付きで記録する（原因調査のため）
            log_exception(f"{type(self).__name__} でエラー: {e}")
            self.finished_error.emit(str(e))


class FileSaveWorker(QThread):
    """
    指定マスクまたは複数レイヤーごとの点群をバックグラウンドスレッドでE57へロスレス保存するワーカースレッド
    単一ファイル保存および全レイヤー別ファイル一括分割保存の両方に対応します。
    """

    progress_changed = Signal(int, str, str)  # (0-100%, メインメッセージ, サブメッセージ)
    finished_success = Signal(int, int, str)  # (保存総点数, 除外点数, 保存先パス説明)
    finished_error = Signal(str)

    def __init__(
        self,
        output_path: str,
        scans: List[E57ScanData],
        layer_mask: np.ndarray,
        split_masks: Optional[Dict[str, np.ndarray]] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.output_path = output_path
        self.scans = scans
        self.layer_mask = layer_mask.copy()
        self.split_masks = {k: v.copy() for k, v in split_masks.items()} if split_masks else None
        self._is_cancelled = False

    def cancel(self):
        """
        保存の中断を要求。単一ファイルは書き込み開始前、分割保存はファイル境界で反映される。
        書き込み中のファイルは一時ファイル (.tmp) 経由のアトミック置換なので、中断しても既存ファイルは壊れない。
        """
        self._is_cancelled = True

    def _check_cancelled(self):
        if self._is_cancelled:
            raise InterruptedError("保存がユーザーによって中止されました。")

    def _restore_original_coordinates(self) -> List[str]:
        """
        取り込み時に座標系変換（測量X/Y入替・Y-up・Z反転）を適用したスキャンを、保存のために
        元ファイルの座標系へインプレースで戻す。戻り値は各スキャンの変換前モード（再適用用）。
        E57 は常に元の座標系・元のスケールで書き出す（ユーザー決定: ロスレス往復を優先）。
        巨大点群のコピーを避けるためインプレースで行い、run() の finally で必ず再適用する。
        """
        modes: List[str] = []
        for scan in self.scans:
            modes.append(restore_scan_original_coordinates(scan))
        return modes

    def _reapply_coordinates(self, modes: List[str]) -> None:
        """_restore_original_coordinates() で戻した座標を表示用の座標系へ再適用する"""
        for scan, mode in zip(self.scans, modes):
            if not is_identity_coord_mode(mode):
                transform_scan_coordinate_system(scan, mode)

    def _extract_scans_by_mask(self, mask: np.ndarray) -> List[E57ScanData]:
        """全体マスクから各スキャンに対応する点群をロスレス抽出"""
        filtered_scans: List[E57ScanData] = []
        if len(self.scans) == 1:
            scan = self.scans[0]
            keep_idx = np.where(mask)[0]
            if len(keep_idx) > 0:
                filtered_scans.append(scan.filter_by_indices(keep_idx))
        else:
            current_offset = 0
            for scan in self.scans:
                n_s = len(scan.points)
                if n_s == 0:
                    continue
                s_mask = mask[current_offset : current_offset + n_s]
                keep_idx = np.where(s_mask)[0]
                if len(keep_idx) > 0:
                    filtered_scans.append(scan.filter_by_indices(keep_idx))
                current_offset += n_s
        return filtered_scans

    def run(self):
        applied_modes: List[str] = []
        try:
            # 保存は常に元ファイルの座標系で行う（取り込み時の座標系変換を一時的に戻す）
            applied_modes = self._restore_original_coordinates()
            self._run_save()
        except Exception as e:
            # ユーザーには要約を、ログにはトレースバック付きで記録する（原因調査のため）
            log_exception(f"{type(self).__name__} でエラー: {e}")
            self.finished_error.emit(str(e))
        finally:
            # 表示用の座標系へ必ず戻す（保存失敗時も含む）
            try:
                self._reapply_coordinates(applied_modes)
            except Exception as re_exc:  # 再適用失敗は致命的ではないが記録する
                from core.logger import log_error

                log_error(f"保存後の座標系再適用に失敗しました: {re_exc}")

    def _run_save(self):
        # 複数レイヤーの一括分割保存モード
        if self.split_masks:
            base_path, ext = os.path.splitext(self.output_path)
            if not ext:
                ext = ".e57"
            non_empty_items = [(name, m) for name, m in self.split_masks.items() if int(np.sum(m)) > 0]
            if not non_empty_items:
                raise ValueError("出力対象のレイヤーに有効な点がありません。")

            total_saved_pts = 0
            saved_files = []
            used_out_files = set()
            n_items = len(non_empty_items)

            for idx, (layer_name, mask) in enumerate(non_empty_items):
                self._check_cancelled()
                safe_name = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in layer_name).strip("_")
                if not safe_name:
                    safe_name = f"layer_{idx}"
                out_file = f"{base_path}_{safe_name}{ext}"
                if out_file in used_out_files:
                    out_file = f"{base_path}_{safe_name}_{idx}{ext}"
                used_out_files.add(out_file)
                pct = int(10 + (idx / max(1, n_items)) * 80)
                self.progress_changed.emit(
                    pct,
                    f"レイヤー「{layer_name}」をE57出力中 ({idx + 1}/{n_items})...",
                    f"出力先: {os.path.basename(out_file)}",
                )
                sub_scans = self._extract_scans_by_mask(mask)
                if sub_scans:
                    write_e57_scans(out_file, sub_scans)
                    pts_cnt = sum(s.point_count for s in sub_scans)
                    total_saved_pts += pts_cnt
                    saved_files.append(f"{os.path.basename(out_file)} ({pts_cnt:,}点)")

            total_excluded_pts = max(0, int(len(self.layer_mask) - total_saved_pts))
            self.progress_changed.emit(100, "全レイヤー分割保存完了！", f"{len(saved_files)} ファイルを出力")
            summary_path = f"{len(saved_files)} ファイル ({', '.join(saved_files)})"
            self.finished_success.emit(total_saved_pts, total_excluded_pts, summary_path)
            return

        # 単一ファイル保存モード
        fname = os.path.basename(self.output_path)
        self.progress_changed.emit(15, "保存対象レイヤーの点群を抽出中...", f"出力先: {fname}")

        self._check_cancelled()
        filtered_scans = self._extract_scans_by_mask(self.layer_mask)
        if not filtered_scans:
            raise ValueError("保存対象レイヤーの点数が 0 点のため出力できません。")
        self._check_cancelled()

        total_clean_pts = sum(s.point_count for s in filtered_scans)
        total_noise_pts = int(len(self.layer_mask) - total_clean_pts)

        self.progress_changed.emit(
            60,
            "E57バイナリ書き込み中 (ロスレス・アトミック保存)...",
            f"対象: {total_clean_pts:,} 点",
        )
        write_e57_scans(self.output_path, filtered_scans)

        self.progress_changed.emit(100, "保存完了！", f"保存点数: {total_clean_pts:,} 点")
        self.finished_success.emit(total_clean_pts, total_noise_pts, self.output_path)





def split_output_path(base_path: str, ext: str, layer_name: str, idx: int, used: set) -> str:
    """全レイヤー分割保存のファイル名（FileSaveWorker と FullResolutionSaveWorker で共通）"""
    safe_name = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in layer_name).strip("_")
    if not safe_name:
        safe_name = f"layer_{idx}"
    out_file = f"{base_path}_{safe_name}{ext}"
    if out_file in used:
        out_file = f"{base_path}_{safe_name}_{idx}{ext}"
    used.add(out_file)
    return out_file


class FullResolutionSaveWorker(QThread):
    """
    全点転写保存: 間引き読込（1 億点超のメモリ保護）で決めたノイズ判定を、元ファイルの全点へ写して E57 出力する。

    手順（スキャンごとに、100 万点ずつ流して行う。メモリは「読み込んでいる間引き点群 + チャンク数個」で済み、
    1 スキャンが 5 億点を超えても足りる。以前は 1 スキャンの全点を一度に読んでいたので、5.4 億点・1 スキャンの
    ファイルでは約 89GB が要り、32GB の PC では保存できなかった）
      1. 元ファイルからそのスキャンを 100 万点ずつ読む（core.e57_stream.iter_scan_stream_chunks）
      2. 読み込んでいる間引き点群（判定点）のラベルを、最近傍の判定点からその点へ写す（core.fullres_transfer.LabelTransfer）
      3. 保存対象の点だけを、出力のスキャンへそのまま追記する（core.e57_stream.ScanStreamSink。生の整数・スケール・
         pose・メタデータはそのまま。出力は以前の「全点を読んでから書く」方式と同じになる）
    元ファイルの値がプロトタイプの範囲を外れていたときは、範囲を測ってから最初から書き直す（まれ）。
    判定点の座標は取り込み時の座標系変換を外してから使う（元ファイルと同じローカル座標で照合する）。
    元ファイルが読込後に変更されていたら（サイズか更新時刻の不一致）保存しない。
    中止は転写のチャンク境界とスキャン境界で反映され、書きかけの一時ファイルは消される。
    """

    progress_changed = Signal(int, str, str)  # (0-100%, メインメッセージ, サブメッセージ)
    finished_success = Signal(int, int, str)  # (保存総点数, 除外点数, 保存先パス説明)
    finished_error = Signal(str)

    KEEP_LABEL = 1

    def __init__(
        self,
        output_path: str,
        scans: List[E57ScanData],
        layer_mask: np.ndarray,
        split_masks: Optional[Dict[str, np.ndarray]] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.output_path = output_path
        self.scans = list(scans)
        self.layer_mask = np.asarray(layer_mask, dtype=bool).copy()
        self.split_masks = (
            {k: np.asarray(v, dtype=bool).copy() for k, v in split_masks.items() if int(np.sum(v)) > 0}
            if split_masks
            else None
        )
        self._is_cancelled = False
        self.scan_stats: List[Any] = []  # (スキャン名, TransferStats) の一覧（検証・ログ用）
        self.total_orphans = 0
        self.total_full_points = 0

    def cancel(self):
        self._is_cancelled = True

    def _check_cancelled(self):
        if self._is_cancelled:
            raise InterruptedError("保存がユーザーによって中止されました。")

    def _validate_sources(self) -> None:
        n_all = sum(len(s.points) for s in self.scans)
        if len(self.layer_mask) != n_all:
            raise ValueError(f"レイヤーマスクの長さ {len(self.layer_mask):,} が読み込み点数 {n_all:,} と一致しません。")
        for s in self.scans:
            if not s.source_file or s.source_scan_index is None:
                raise ValueError(
                    f"スキャン「{s.name}」に読み込み元ファイルの記録がないため全点転写保存はできません。"
                    "「間引いたまま保存」を使ってください。"
                )
            if not os.path.isfile(s.source_file):
                raise FileNotFoundError(f"読み込み元ファイルが見つかりません: {s.source_file}")
            if s.source_signature is not None and file_signature(s.source_file) != tuple(s.source_signature):
                raise RuntimeError(
                    f"読み込み元ファイルが読込後に変更されています（サイズか更新時刻が異なります）: {s.source_file}\n"
                    "元ファイルを開き直してから保存してください。"
                )

    def run(self):
        try:
            self._run_save()
        except Exception as e:
            log_exception(f"{type(self).__name__} でエラー: {e}")
            self.finished_error.emit(str(e))

    def _run_save(self):
        try:
            self._run_save_once(measure_ranges=False)
        except E57SchemaBoundsError as exc:
            # 元ファイルの値がそのファイル自身のプロトタイプの範囲を外れていた（まれ）。範囲を測って最初から書き直す
            log_warning(f"全点で保存: {exc} 各スキャンの値の範囲を測ってから、最初から書き直します。")
            self._run_save_once(measure_ranges=True)

    def _run_save_once(self, measure_ranges: bool):
        t0 = time.perf_counter()
        self._validate_sources()
        self.scan_stats = []
        self.total_orphans = 0

        # 判定点のラベル: 単一ファイルは 1=保存 / 0=除外、分割保存はレイヤー番号（どれにも入らなければ除外）
        if self.split_masks:
            layer_names = list(self.split_masks.keys())
            labels_all = labels_from_masks([self.split_masks[k] for k in layer_names], len(self.layer_mask))
        else:
            layer_names = []
            labels_all = self.layer_mask.astype(np.int32)
            if int(np.sum(self.layer_mask)) == 0:
                raise ValueError("保存対象レイヤーの点数が 0 点のため出力できません。")

        base_path, ext = os.path.splitext(self.output_path)
        if not ext:
            ext = ".e57"
        writers: Dict[int, E57StreamWriter] = {}
        writer_paths: Dict[int, str] = {}
        used_out_files: set = set()
        open_sinks: List[ScanStreamSink] = []
        total_saved = 0
        total_excluded = 0
        self.total_full_points = sum(int(getattr(s, "original_point_count", len(s.points))) for s in self.scans)
        done_full = 0
        offset = 0
        chunk = int(getattr(self, "stream_chunk_size", DEFAULT_STREAM_CHUNK))

        def pct_of(done: int) -> int:
            return int(5 + 90 * (done / max(1, self.total_full_points)))

        def writer_for(li: int) -> E57StreamWriter:
            if li not in writers:
                if self.split_masks:
                    out_file = split_output_path(base_path, ext, layer_names[li], li, used_out_files)
                else:
                    out_file = self.output_path
                writer_paths[li] = out_file
                writers[li] = E57StreamWriter(out_file)
            return writers[li]

        try:
            for s in self.scans:
                n_j = len(s.points)
                labels_j = labels_all[offset : offset + n_j]
                offset += n_j
                self._check_cancelled()
                name = s.name or f"Scan {s.scan_index}"
                n_orig = int(getattr(s, "original_point_count", n_j))
                tpl = scan_stream_template(s.source_file, int(s.source_scan_index))

                if measure_ranges:
                    self.progress_changed.emit(
                        pct_of(done_full), f"スキャン「{name}」の値の範囲を確認中...", f"{n_orig:,} 点"
                    )
                    tpl.bounds_override = measure_field_ranges(tpl, chunk, is_cancelled=lambda: self._is_cancelled)

                if n_j == 0:
                    log_warning(f"全点転写: スキャン「{name}」は判定点が 0 点のため、{n_orig:,} 点すべてを除外します。")
                    judged_local = np.empty((0, 3), dtype=np.float64)
                else:
                    judged_local = (
                        s.points
                        if is_identity_coord_mode(getattr(s, "coord_mode", "cartesian"))
                        else inverse_transform_coordinate_system(s.points, s.coord_mode)
                    )
                lt = LabelTransfer(judged_local, labels_j, estimate_stride(n_orig, n_j))
                del judged_local

                sinks: Dict[int, ScanStreamSink] = {}
                n_full = 0
                self.progress_changed.emit(
                    pct_of(done_full), f"スキャン「{name}」を全点で読み直し、判定を転写中...", f"0 / {n_orig:,} 点"
                )
                for raw, xyz in iter_scan_stream_chunks(tpl, chunk, is_cancelled=lambda: self._is_cancelled):
                    self._check_cancelled()
                    labels = lt.labels_for(xyz)
                    targets = (
                        [int(v) for v in np.unique(labels) if v >= 0] if self.split_masks
                        else ([self.KEEP_LABEL] if np.any(labels == self.KEEP_LABEL) else [])
                    )
                    for li in targets:
                        idx = np.flatnonzero(labels == li)
                        if len(idx) == 0:
                            continue
                        key = li if self.split_masks else 0
                        sink = sinks.get(key)
                        if sink is None:
                            sink = ScanStreamSink(writer_for(key), tpl, name_override=s._name or None, capacity=chunk)
                            sinks[key] = sink
                            open_sinks.append(sink)
                        sink.write(raw, xyz, idx)
                    n_full += len(xyz)
                    del raw, xyz, labels
                    self.progress_changed.emit(
                        pct_of(done_full + n_full),
                        f"スキャン「{name}」を全点で読み直し、判定を転写中...",
                        f"{n_full:,} / {n_orig:,} 点",
                    )

                kept_in_scan = 0
                for sink in sinks.values():
                    kept_in_scan += sink.finish()
                    open_sinks.remove(sink)
                stats = lt.stats
                self.scan_stats.append((name, stats))
                self.total_orphans += int(stats.n_orphan)
                total_saved += kept_in_scan
                total_excluded += n_full - kept_in_scan
                done_full += n_full
                log_info(
                    f"全点転写: 「{name}」 元 {n_full:,} 点 / 判定点 {n_j:,} 点 (間引き k={stats.stride}, 許容半径 {stats.factor:.0f}×点間隔)"
                    f" → 保存 {kept_in_scan:,} 点, 除外 {n_full - kept_in_scan:,} 点 (うち孤立点 {stats.n_orphan:,} 点,"
                    f" 引き継ぎ最大距離 {stats.max_inherit_distance:.3f}m)"
                )
                del lt, sinks

            if not writers:
                raise ValueError("保存対象レイヤーの点数が 0 点のため出力できません。")

            self._check_cancelled()
            self.progress_changed.emit(96, "出力ファイルを検証中...", f"{len(writers)} ファイル")
            for li in list(writers.keys()):
                writers[li].close()
        except BaseException:
            for sink in open_sinks:
                sink.close_quietly()
            for w in writers.values():
                w.abort()
            raise

        elapsed = time.perf_counter() - t0
        log_info(
            f"全点転写保存 完了: 元 {self.total_full_points:,} 点 → 保存 {total_saved:,} 点, 除外 {total_excluded:,} 点"
            f" (孤立点 {self.total_orphans:,} 点), {len(writers)} ファイル, {elapsed:.1f} 秒"
        )
        if self.split_masks:
            parts = [f"{os.path.basename(writer_paths[li])} ({writers[li].point_count:,}点)" for li in writers]
            summary = f"{len(parts)} ファイル ({', '.join(parts)})"
            self.progress_changed.emit(100, "全レイヤー分割保存完了！", f"{len(parts)} ファイルを出力")
            self.finished_success.emit(total_saved, total_excluded, summary)
        else:
            self.progress_changed.emit(100, "保存完了！", f"保存点数: {total_saved:,} 点")
            self.finished_success.emit(total_saved, total_excluded, self.output_path)


class FunctionWorker(QThread):
    """
    任意の重い関数をバックグラウンドで実行する汎用ワーカー（設計憲章③: 重い処理をメインスレッドで実行しない）。

    使い方:
        worker = FunctionWorker(segment_ground_points, pts, grid_size=1.0)
        worker.finished_success.connect(on_done)   # fn の戻り値がそのまま渡る
        worker.finished_error.connect(on_error)
        worker.start()

    fn が `is_cancelled` キーワード引数を受け取れる場合は、中断判定用の callable を自動で渡す。
    fn が `progress_callback(ratio: float, message: str)` を受け取れる場合も同様に渡す。
    """

    progress_changed = Signal(int, str, str)  # (0-100%, メインメッセージ, サブメッセージ)
    finished_success = Signal(object)         # fn の戻り値
    finished_error = Signal(str)
    cancelled = Signal()

    def __init__(self, fn: Callable, *args, parent=None, **kwargs):
        super().__init__(parent)
        self._fn = fn
        self._args = args
        self._kwargs = kwargs
        self._is_cancelled = False

    def cancel(self):
        """処理の中断を要求（fn が is_cancelled を確認する場合のみ途中中断できる）"""
        self._is_cancelled = True

    @property
    def is_cancelled(self) -> bool:
        return self._is_cancelled

    def run(self):
        try:
            kwargs = dict(self._kwargs)
            try:
                params = inspect.signature(self._fn).parameters
            except (TypeError, ValueError):
                params = {}
            if "is_cancelled" in params:
                kwargs["is_cancelled"] = lambda: self._is_cancelled
            if "progress_callback" in params:
                def _cb(ratio: float, message: str = ""):
                    if self._is_cancelled:
                        raise InterruptedError("処理がユーザーによって中止されました。")
                    self.progress_changed.emit(int(max(0.0, min(1.0, ratio)) * 100), message, "")
                kwargs["progress_callback"] = _cb

            result = self._fn(*self._args, **kwargs)
            if self._is_cancelled:
                self.cancelled.emit()
                return
            self.finished_success.emit(result)
        except InterruptedError:
            self.cancelled.emit()
        except Exception as e:
            log_exception(f"FunctionWorker({getattr(self._fn, '__name__', 'fn')}) でエラー: {e}")
            self.finished_error.emit(str(e))
