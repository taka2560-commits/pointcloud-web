# -*- coding: utf-8 -*-
"""
視点連動の高精細表示（LOD）のコントローラ

ビューアの視点が止まったら、画面に入っている範囲を LOD ストア（core/lod_store.py）から元ファイルの密度で読み、
表示専用の重ね描き（PointCloudViewer.set_view_overlay）として描く。近く・大きく見えている所ほど細かく、
全体で予算（既定 600 万点）に収める。動かしている間は直前の重ね描きをそのまま描き、止まったら読み直す。

- 読み出しはバックグラウンド（FunctionWorker）。視点・断面・レイヤーが変わったら、落ち着いてから作り直す
  （レイヤーの変更＝ノイズ化などは待たずにすぐ作り直す）。
- 読み込み済みの点のうち非表示のもの（ノイズレイヤーなど）だけがある格子に入る点は重ね描きから隠す
  （core.lod_store.hidden_voxel_sets。ノイズとして消した点の周りに、元ファイルの点が残って見えないようにする）。
  判定には、重ね描きの範囲（＋最も粗い格子ぶんの余白）に入る読み込み済みの点だけを使うので、寄っているほど速い。
- 断面が有効なら箱の中だけを読む。断面の「元ファイルから全点で読み直す」で作った重ね描きがある間は上書きしない。
- 重ね描きは表示専用（選択・ノイズ移動・計測の対象外）。編集は読み込み済みの点で行う。
ビューアの変更は最小限にするため、視点の変化はタイマーで見ている（マウス処理には手を入れない）。
"""

import time
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal

from core.lod_store import DEFAULT_LEAF_MAX_POINTS, HIDDEN_VOXELS, LodScene, hidden_voxel_sets, not_hidden_mask
from core.logger import log_exception, log_info
from core.section_reload import SectionBox
from gui.worker_thread import FunctionWorker

DEFAULT_BUDGET = 6_000_000
DEFAULT_PX_DENSITY = 1.0
RAW_FACTOR = 4  # 断面で絞る前に読む点数の上限（予算の何倍まで）


def compute_lod_overlay(
    scene: LodScene,
    mvp: np.ndarray,
    viewport: Tuple[int, int],
    budget: int,
    px_density: float,
    box: Optional[SectionBox],
    loaded: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    is_cancelled=None,
    refine: bool = True,
) -> Dict[str, Any]:
    """
    1 回ぶんの重ね描きを作る（GUI に触らない純粋な計算。ワーカースレッドから呼ぶ）。
    loaded=(読み込み済みの点 (N,3) 中心オフセット済み float32, 表示フラグ (N,) bool) を渡すと、
    非表示の点だけがある格子に入る重ね描きの点を隠す。非表示の点が無いときは None でよい。

    断面が有効なとき、葉の箱の広さで決めたレベルのままだと、箱で絞ったあとに点がわずかしか残らない
    （薄い断面は葉のごく一部しか使わない）。残った点が予算の 6 割に満たなければ、残った割合から逆算して
    目標密度を上げ、読み直す（refine。読む点数は予算の RAW_FACTOR 倍まで）。それでも予算を超えたら等間隔に間引く。
    """
    t0 = time.perf_counter()

    def _check():
        if is_cancelled is not None and is_cancelled():
            raise InterruptedError("高精細表示の更新を中止しました。")

    box_active = box is not None and box.is_active()
    levels, info = scene.select(mvp, viewport, budget, px_density, box)
    _check()
    pts, rgb, inten, n_selected = scene.gather_view(levels, box)
    if refine and box_active:
        raw_cap = int(budget) * RAW_FACTOR
        density = float(px_density)
        for _ in range(2):
            n_now = len(pts)
            full = scene.cum is not None and np.all((levels < 0) | (levels >= scene.max_levels - 1))
            if n_now >= 0.6 * budget or n_selected >= raw_cap or full or info.get("n_visible", 0) == 0:
                break
            boost = min(budget / max(1, n_now), raw_cap / max(1, n_selected))
            if boost < 1.3:
                break
            density *= boost
            _check()
            levels, info = scene.select(mvp, viewport, raw_cap, density, box)
            pts, rgb, inten, n_selected = scene.gather_view(levels, box)
    if len(pts) > budget:
        step = int(np.ceil(len(pts) / float(budget)))
        pts = pts[::step]
        rgb = rgb[::step] if rgb is not None else None
        inten = inten[::step] if inten is not None else None
        info = dict(info)
        info["scale"] = min(float(info.get("scale", 1.0)), 1.0 / step)
    n_hidden = 0
    if loaded is not None and len(pts) > 0:
        loaded_pts, visible = loaded
        # 重ね描きの範囲 + 最も粗い格子ぶんの余白に入る読み込み済みの点だけで判定する（格子が範囲の端で欠けないように）
        margin = np.float32(max(HIDDEN_VOXELS))
        lo = pts.min(axis=0) - margin
        hi = pts.max(axis=0) + margin
        near = (
            (loaded_pts[:, 0] >= lo[0]) & (loaded_pts[:, 0] <= hi[0])
            & (loaded_pts[:, 1] >= lo[1]) & (loaded_pts[:, 1] <= hi[1])
            & (loaded_pts[:, 2] >= lo[2]) & (loaded_pts[:, 2] <= hi[2])
        )
        idx = np.flatnonzero(near)
        if len(idx) > 0 and not visible[idx].all():
            zero = np.zeros(3)
            sets = hidden_voxel_sets(loaded_pts[idx], visible[idx], origin=zero)
            keep = not_hidden_mask(pts, sets, origin=zero)
            n_hidden = int(len(pts) - np.count_nonzero(keep))
            if n_hidden:
                kept = np.flatnonzero(keep)
                pts = pts[kept]
                rgb = rgb[kept] if rgb is not None else None
                inten = inten[kept] if inten is not None else None
    colors = (rgb.astype(np.float32) / np.float32(255.0)) if rgb is not None else None
    intensities = inten.astype(np.float32) if inten is not None else None
    return {
        "points": pts,
        "colors": colors,
        "intensities": intensities,
        "n_selected": int(n_selected),
        "n_hidden": n_hidden,
        "info": info,
        "elapsed": time.perf_counter() - t0,
    }


class LodController(QObject):
    status_changed = Signal(str)  # ステータスバー向けの一言
    refreshed = Signal(int)       # 重ね描きした点数

    POLL_MS = 200     # 視点の変化を見る間隔
    SETTLE_MS = 350   # 視点が止まってから読み直すまでの待ち

    def __init__(self, viewer, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.scene: Optional[LodScene] = None
        self.enabled = False
        self.budget = DEFAULT_BUDGET
        self.px_density = DEFAULT_PX_DENSITY
        self._worker: Optional[FunctionWorker] = None
        self._job_key = None
        self._applied_key = None
        self._seen_key = None
        self._seen_at = 0.0
        self._layer_rev = 0
        self._refresh_soon = False                      # レイヤーが変わった → 待たずに作り直す
        self._visible: Optional[np.ndarray] = None      # 読み込み済みの点の表示フラグ（レイヤーが変わるまで使い回す）
        self._visible_rev = -1
        self._has_hidden = False
        self._points_ref = None
        self.last_result_info: Dict[str, Any] = {}
        self._timer = QTimer(self)
        self._timer.setInterval(self.POLL_MS)
        self._timer.timeout.connect(self._on_poll)
        viewer.layers_updated.connect(self._on_layers_changed)
        viewer.layer_list_changed.connect(self._on_layers_changed)

    # --- 状態 ---
    @property
    def is_active(self) -> bool:
        return bool(self.enabled and self.scene is not None)

    @property
    def is_busy(self) -> bool:
        return self._worker is not None and self._worker.isRunning()

    def attach(self, scans: Sequence, cache_dir: Optional[str] = None, leaf_max_points: int = DEFAULT_LEAF_MAX_POINTS) -> bool:
        """読み込み中の点群に対応する LOD ストアを開く。無ければ False（作成は MainWindow が行う）"""
        self.detach()
        if not scans or self.viewer.points_raw is None:
            return False
        scene = LodScene.open_for_scans(scans, cache_dir=cache_dir, leaf_max_points=leaf_max_points)
        if scene is None:
            return False
        scene.prepare(self.viewer.center_offset)
        self.scene = scene
        self._points_ref = self.viewer.points_raw
        self._applied_key = None
        self._seen_key = None
        self._visible = None
        self._visible_rev = -1
        log_info(f"高精細表示 (LOD): ストアを開きました（{scene.n_points:,} 点, 葉 {scene.n_leaves:,} 個）")
        return True

    def detach(self) -> None:
        self._cancel_worker(wait_ms=2000)  # 読み出し中にストアを閉じない
        if self.scene is not None:
            self.scene.close()
        self.scene = None
        self._points_ref = None
        self._applied_key = None
        self._visible = None
        if self.viewer.overlay_kind == "lod":
            self.viewer.clear_section_overlay()
            self.viewer.update()

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        if self.enabled:
            self._applied_key = None
            self._seen_key = None
            self._timer.start()
        else:
            self._timer.stop()
            self._cancel_worker()
            if self.viewer.overlay_kind == "lod":
                self.viewer.clear_section_overlay()
                self.viewer.update()

    def set_budget(self, budget: int) -> None:
        self.budget = max(100_000, int(budget))
        self._applied_key = None

    def shutdown(self) -> None:
        self._timer.stop()
        self._cancel_worker(wait_ms=3000)
        if self.scene is not None:
            self.scene.close()
            self.scene = None

    def _cancel_worker(self, wait_ms: int = 0) -> None:
        w = self._worker
        if w is not None:
            try:
                w.cancel()
                if wait_ms and w.isRunning():
                    w.wait(wait_ms)
            except RuntimeError:
                pass
        self._job_key = None

    def _on_layers_changed(self, *_args) -> None:
        self._layer_rev += 1
        self._refresh_soon = True

    # --- 視点の監視 ---
    def _state_key(self):
        mvp, w, h = self.viewer.camera_mvp()
        return (
            np.round(mvp, 5).tobytes(),
            w,
            h,
            self.viewer._section_box_key(),
            self._layer_rev,
            self.budget,
        )

    def _cloud_replaced(self) -> bool:
        return self.viewer.points_raw is None or self.viewer.points_raw is not self._points_ref

    def _on_poll(self) -> None:
        if not self.is_active:
            return
        if self._cloud_replaced():
            self.detach()
            self.status_changed.emit("高精細表示 (LOD): 点群が変わったため停止しました")
            return
        key = self._state_key()
        now = time.monotonic()
        if key != self._seen_key:
            self._seen_key = key
            self._seen_at = now
            if not self._refresh_soon:
                return
        if key == self._applied_key or self.is_busy:
            return
        if not self._refresh_soon and (now - self._seen_at) * 1000.0 < self.SETTLE_MS:
            return
        if getattr(self.viewer, "is_rotating", False) or getattr(self.viewer, "is_panning", False):
            return
        if self.viewer.overlay_kind == "section":
            self._applied_key = key  # 断面の「全点で読み直す」で作った重ね描きは上書きしない
            return
        self._start_refresh(key)

    def _job_inputs(self) -> Dict[str, Any]:
        """ワーカーへ渡す入力（メインスレッドでビューアの状態を写し取る）"""
        mvp, w, h = self.viewer.camera_mvp()
        spec = self.viewer.section_box_spec()
        if self._visible_rev != self._layer_rev or self._visible is None:
            self._visible = self.viewer._get_layer_and_scan_visibility_mask()
            self._has_hidden = not bool(self._visible.all())
            self._visible_rev = self._layer_rev
        loaded = (self.viewer.points_centered, self._visible) if self._has_hidden else None
        return dict(
            scene=self.scene,
            mvp=mvp,
            viewport=(w, h),
            budget=int(self.budget),
            px_density=float(self.px_density),
            box=SectionBox(**spec) if spec is not None else None,
            loaded=loaded,
        )

    def _start_refresh(self, key) -> None:
        inputs = self._job_inputs()
        self._job_key = key
        self._refresh_soon = False
        self.status_changed.emit("高精細表示 (LOD): 見えている範囲を読み込み中...")
        worker = FunctionWorker(compute_lod_overlay, **inputs)
        worker.finished_success.connect(lambda result, _k=key, _w=worker: self._on_job_done(result, _k, _w))
        worker.finished_error.connect(lambda msg, _w=worker: self._on_job_error(msg, _w))
        worker.cancelled.connect(lambda _w=worker: self._on_job_cancelled(_w))
        self._worker = worker
        worker.start()

    def _on_job_done(self, result: Dict[str, Any], key, worker) -> None:
        if worker is not self._worker or key != self._job_key:
            return  # 取り消された古い結果
        self._apply(result, key)

    def _on_job_error(self, msg: str, worker) -> None:
        if worker is self._worker and self._job_key is not None:  # 取り消したあとの失敗は知らせない
            self._applied_key = self._job_key  # 同じ状態で失敗を繰り返さない
            self.status_changed.emit(f"高精細表示 (LOD): 読み込みに失敗しました（{msg}）")

    def _on_job_cancelled(self, worker) -> None:
        if worker is self._worker:
            self._job_key = None

    def _apply(self, result: Dict[str, Any], key) -> None:
        if not self.is_active or self._cloud_replaced():
            return
        if key[3] != self.viewer._section_box_key():
            return  # 読んでいる間に断面が変わった。次の監視で作り直す
        if self.viewer.overlay_kind == "section":
            self._applied_key = key
            return
        pts = result["points"]
        self.viewer.set_view_overlay(
            pts, result["colors"], result["intensities"], n_total=result["n_selected"], capped=result["info"].get("scale", 1.0) < 1.0
        )
        self._applied_key = key
        self.last_result_info = {
            "n_points": int(len(pts)),
            "n_selected": result["n_selected"],
            "n_hidden": result["n_hidden"],
            "elapsed": result["elapsed"],
            **{k: v for k, v in result["info"].items()},
        }
        hidden_txt = f"、非表示の周り {result['n_hidden']:,} 点を除外" if result["n_hidden"] else ""
        self.status_changed.emit(f"高精細表示 (LOD): {len(pts):,} 点{hidden_txt}（{result['elapsed']:.1f} 秒）")
        self.refreshed.emit(int(len(pts)))

    def refresh_now(self) -> int:
        """今の視点で同期的に作り直す（テスト・スクリプト用）。重ね描きした点数を返す"""
        if not self.is_active or self._cloud_replaced():
            return 0
        self._cancel_worker()
        key = self._state_key()
        try:
            result = compute_lod_overlay(**self._job_inputs())
        except Exception as e:
            log_exception(f"高精細表示 (LOD) の更新でエラー: {e}")
            raise
        self._seen_key = key
        self._refresh_soon = False
        self._apply(result, key)
        return int(len(result["points"])) if self.viewer.overlay_kind == "lod" else 0
