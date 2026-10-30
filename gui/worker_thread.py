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
from core.e57_io import E57ScanData, read_e57_scans, write_e57_scans


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
        parent=None,
    ):
        super().__init__(parent)
        self.file_path = file_path
        self.point_budget = point_budget
        # 実務でのデータ欠損を防ぐため、デフォルト(None)では全点ロスレス読み込みを行う
        # （point_budget は3D描画用の表示間引きのみに使用し、元データは1点も減らさない）
        self.max_load_points = max_load_points
        self._is_cancelled = False

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
                fname = os.path.basename(fpath)
                sub = f"ファイル {f_i + 1}/{n_files}: {fname}"
                self.progress_changed.emit(int(5 + 60 * f_i / n_files), "E57ヘッダー解析中...", sub)

                def on_e57_progress(ratio: float, msg: str, _fi=f_i, _sub=sub):
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

            if len(non_empty_scans) == 1:
                # 単一スキャン時は s.points をそのまま参照共有し、倍精度座標配列の二重コピーをゼロにする
                only_idx, only_scan = non_empty_scans[0]
                combined_pts = only_scan.points
                combined_scan_ids = np.full(total_pts, only_idx, dtype=np.int32)
            else:
                combined_pts = np.empty((total_pts, 3), dtype=np.float64)
                combined_scan_ids = np.empty(total_pts, dtype=np.int32)
                offset_cursor = 0
                for s_idx, s in non_empty_scans:
                    n_cur = len(s.points)
                    combined_pts[offset_cursor : offset_cursor + n_cur] = s.points
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
        except Exception as e:
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
    ):
        super().__init__(parent)
        self.points = points
        self.config = config
        self.intensities = intensities
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
                keep_local, remove_local = processor.apply_filters_to_points(
                    self.points[remaining],
                    log_callback=self.log_message.emit,
                    intensities=sub_int,
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
        try:
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

            filtered_scans = self._extract_scans_by_mask(self.layer_mask)
            if not filtered_scans:
                raise ValueError("保存対象レイヤーの点数が 0 点のため出力できません。")

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
        except Exception as e:
            self.finished_error.emit(str(e))



