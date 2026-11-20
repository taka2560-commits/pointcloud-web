# -*- coding: utf-8 -*-
"""
全点表示（断面の「元ファイルから全点で読み直す」の重ね描き）の点を、選んで消せるようにする部品
gui/gl_viewer.py: PointCloudViewer に混ぜて使う（ビューア本体の変更を小さくするため、別のファイルに分けてある）。

なぜ要るか（2026-11-20、利用者の報告「断面の全点読み込みをすると、ブラシで消えないノイズがあった」）
    1 億点超のファイルは「k 点に 1 点」で読み込まれ、ブラシが選べるのは読み込んだ点だけだった。
    全点表示で見える残りの点（読み込みで飛ばされた点）は表示専用で、面のすぐそばのノイズは消す手段が無かった
    （保存のときは、近くの読み込み済みの点の判定を引き継ぐので、面の判定を引き継いで残る）。

いまの作り
    ・重ね描きの点は、元ファイルでの行の番号（core/section_reload.py の rows）と一緒に受け取る。
    ・読み込んだ点と同じ行（番号が間引きの間隔の倍数）は、重ね描きでは描かない・選ばない。その点は、点群の本体が描いていて、
      今までどおりレイヤーで扱う（同じ点を 2 つの仕組みで持たない。本体でノイズへ移せば、画面からも消える）。
    ・残りの点（読み込みで飛ばされた点）は、矩形・円・多角形・ブラシで、本体の点と同じ形の判定で選ばれる。
      消すと、行の番号を印として覚え（core/fullres_marks.py）、重ね描きから外す。
    ・印は、箱を動かして重ね描きを作り直しても残る。「全点で保存」が、その行をノイズとして扱う。
    ・元に戻す・やり直しは、レイヤーの履歴と同じ 1 手に入る（gui/layer_history.py の attach_extra）。

守っている決まり（tests/test_overlay_edit.py が見張る）
    ・同じ形で選ばれる点は、本体の選び方と同じ（形の判定は、本体と同じ式）。
    ・重ね描きの選択は、本体の選択（selected_indices）がその配列のままのあいだだけ有効。ほかの操作が本体の選択を
      入れ替えたら、重ね描きの選択は無かったことにする（古い選択が残って、あとで消されることが無いように）。
    ・行の番号が分からない点・間引きの間隔が分からないスキャンの点は、今までどおり表示専用（選べない）。
    ・視点連動の高精細表示（LOD）の重ね描きは、今までどおり表示専用。
"""

import math
import weakref
from typing import Optional, Sequence, Tuple

import numpy as np
from OpenGL import GL as gl
from PySide6.QtCore import QRect, Qt

from core.fullres_marks import FullResMarks, loaded_row_mask
from core.logger import log_warning
from gui.color_patch import SELECTION_COLOR
from gui.select_cull import CULL_SKIP_FRACTION, block_boxes_of, blocks_near_screen_box

OVERLAY_BLOCK = 4096            # 重ね描きの区画の大きさ（点数）。選ぶときに、形の近くの区画だけを調べる
OVERLAY_CULL_MIN_POINTS = 4 * OVERLAY_BLOCK
_EMPTY = np.empty(0, dtype=np.int64)


# ----------------------------------------------------------------------------- 形の判定（本体の選び方と同じ式）
def hit_rect(scr_x, scr_y, valid_z, left, right, top, bottom) -> np.ndarray:
    return valid_z & (scr_x >= left) & (scr_x <= right) & (scr_y >= top) & (scr_y <= bottom)


def hit_circle(scr_x, scr_y, valid_z, cx, cy, radius) -> np.ndarray:
    return valid_z & (((scr_x - cx) ** 2 + (scr_y - cy) ** 2) <= (radius ** 2))


def hit_polygon(scr_x, scr_y, valid_z, poly: np.ndarray) -> np.ndarray:
    n_poly = len(poly)
    inside = np.zeros(len(scr_x), dtype=bool)
    j = n_poly - 1
    for i in range(n_poly):
        xi, yi = poly[i]
        xj, yj = poly[j]
        cond = ((yi > scr_y) != (yj > scr_y)) & (
            scr_x < (xj - xi) * (scr_y - yi) / np.maximum(1e-6, np.abs(yj - yi)) * np.sign(yj - yi) + xi
        )
        inside ^= cond
        j = i
    return valid_z & inside


def hit_brush(scr_x, scr_y, valid_z, stroke_xy: np.ndarray, r: float) -> np.ndarray:
    r_sq = r * r
    min_x, max_x = float(np.min(stroke_xy[:, 0])) - r, float(np.max(stroke_xy[:, 0])) + r
    min_y, max_y = float(np.min(stroke_xy[:, 1])) - r, float(np.max(stroke_xy[:, 1])) + r
    out = np.zeros(len(scr_x), dtype=bool)
    sub = np.flatnonzero(valid_z & (scr_x >= min_x) & (scr_x <= max_x) & (scr_y >= min_y) & (scr_y <= max_y))
    if len(sub) == 0:
        return out
    px, py = scr_x[sub], scr_y[sub]
    if len(stroke_xy) == 1:
        hit = ((px - stroke_xy[0, 0]) ** 2 + (py - stroke_xy[0, 1]) ** 2) <= r_sq
    else:
        hit = np.zeros(len(sub), dtype=bool)
        for i in range(len(stroke_xy) - 1):
            x0, y0 = stroke_xy[i, 0], stroke_xy[i, 1]
            x1, y1 = stroke_xy[i + 1, 0], stroke_xy[i + 1, 1]
            dx_seg, dy_seg = x1 - x0, y1 - y0
            seg_len_sq = dx_seg * dx_seg + dy_seg * dy_seg
            if seg_len_sq < 1e-4:
                dist_sq = (px - x0) ** 2 + (py - y0) ** 2
            else:
                t = np.clip(((px - x0) * dx_seg + (py - y0) * dy_seg) / seg_len_sq, 0.0, 1.0)
                dist_sq = (px - (x0 + t * dx_seg)) ** 2 + (py - (y0 + t * dy_seg)) ** 2
            hit |= (dist_sq <= r_sq)
    out[sub[hit]] = True
    return out


def marks_label_for_export(viewer, export_mode: str, target_layer_id, keep_label: int, orphan_label: int) -> int:
    """
    「全点で保存」で、全点表示で消した点に付けるラベル。消した点は、ノイズのレイヤーの点として扱う:
    ノイズのレイヤーが保存の対象に入るときだけ保存し、それ以外は除く。レイヤーごとに分ける保存は、ノイズを出さないので除く
    """
    noise_id = viewer.LAYER_NOISE
    if export_mode == "split_all":
        return int(orphan_label)
    if export_mode == "noise_only":
        return int(keep_label)
    if export_mode == "single_layer":
        return int(keep_label) if (target_layer_id is not None and int(target_layer_id) == noise_id) else 0
    if export_mode == "visible":
        layer = viewer.layers.get(noise_id)
        return int(keep_label) if (layer is not None and layer.visible) else 0
    return 0


def apply_marks_to_save_worker(worker, viewer, export_mode: str, target_layer_id=None) -> int:
    """「全点で保存」のワーカーへ、全点表示で消した点の印を渡す。渡した印の数を返す"""
    from core.fullres_transfer import ORPHAN_LABEL

    marks = viewer.fullres_marks
    if marks.is_empty():
        return 0
    label = marks_label_for_export(viewer, export_mode, target_layer_id, int(worker.KEEP_LABEL), int(ORPHAN_LABEL))
    scan_labels = {}
    if export_mode == "visible":
        # 表示中の点だけを保存するとき、隠したスキャンの点は 1 点も出さない（消した点だけが出ることの無いように）
        scan_labels = {int(sid): 0 for sid, vis in viewer.scan_visibility.items() if not vis}
    worker.set_marks(marks, label, scan_labels)
    return marks.count()


def erase_message(moved: int, n_overlay: int) -> str:
    """消した点の数の言い方（読み込んだ点はノイズのレイヤーへ、全点表示の点は印を付けて画面から外す）"""
    parts = []
    if moved > 0:
        parts.append(f"{moved:,} 点をノイズレイヤーへ移動")
    if n_overlay > 0:
        parts.append(f"全点表示の {n_overlay:,} 点を削除")
    return "、".join(parts)


def restore_all_marks_command(win) -> None:
    """コマンド「全点表示で消した点をすべて戻す」（元に戻す 1 手として記録する）"""
    n = win.viewer.fullres_marks_count()
    if n == 0:
        win.show_toast("全点表示で消した点はありません")
        return
    win._push_layer_undo_state()
    win.viewer.restore_all_fullres_marks()
    win._sync_undo_buttons()
    win.log_text.append(f"全点表示で消した {n:,} 点を、すべて戻しました (Ctrl+Zで取消可)")
    win.show_toast(f"全点表示で消した {n:,} 点を戻しました")


class MarksTracker:
    """
    履歴（gui/layer_history.py の attach_extra）へ渡す、印の出し入れ口。
    いつもビューアの fullres_marks を通す（点群を入れ替えたあとに、前の点群の印を写してしまわないため）
    """

    def __init__(self, viewer):
        self._viewer = viewer

    def snapshot(self):
        return self._viewer.fullres_marks.snapshot()

    def diff(self, old, new):
        return FullResMarks.diff(old, new)

    def toggle(self, diff) -> None:
        self._viewer.fullres_marks.toggle(diff)

    def diff_nbytes(self, diff) -> int:
        return FullResMarks.diff_nbytes(diff)


# ----------------------------------------------------------------------------- ビューアに混ぜる部品
class OverlayEditMixin:
    """PointCloudViewer に混ぜる、全点表示の点の選択・削除"""

    _ov_rows: Optional[np.ndarray] = None       # (M,) int64 重ね描きの点の、元ファイルでの行の番号
    _ov_sids: Optional[np.ndarray] = None       # (M,) int32 スキャンの番号（読み込んだスキャンの並びの添字）
    _ov_loaded: Optional[np.ndarray] = None     # (M,) 読み込んだ点と同じ行（本体が描いているので、重ね描きでは描かない）
    _ov_editable: Optional[np.ndarray] = None   # (M,) 選んで消せる点（行が分かっていて、読み込んだ点ではない）
    _ov_visible: Optional[np.ndarray] = None    # (M,) 描く点（読み込んだ点と同じ行・消した点を除く）
    _ov_all_visible: bool = True
    _ov_boxes = None                            # 区画ごとの箱（最小, 最大）
    _ov_selected: np.ndarray = _EMPTY           # 選んでいる点（重ね描きの中での番号・昇順）
    _ov_sel_owner = None                        # その選択を作ったときの、本体の選択の配列（これが替わったら無効）
    _ov_pending_erase: Optional[np.ndarray] = None  # 即削除モードで、消すのを待っている点
    _ov_index_dirty: bool = True
    _ov_index_count: int = 0
    _vbo_overlay_index = None
    _ov_marks: Optional[FullResMarks] = None
    _ov_marks_owner = None                      # 印を付けたときの点群（points_raw）への弱い参照
    _ov_skip_refresh: bool = False
    _ov_hide_loaded: bool = True                # 読み込んだ点と同じ行を、重ね描きで描かないか（本体が間引かずに描いているときだけ）

    # ------------------------------------------------------------------ 印（消した点）
    @property
    def fullres_marks(self) -> FullResMarks:
        """全点表示で消した点の印。点群を入れ替えたら、空に戻る"""
        m = self._ov_marks
        if m is None:
            m = FullResMarks()
            m.add_listener(self._on_fullres_marks_changed)
            self._ov_marks = m
        pr = getattr(self, "points_raw", None)
        owner = self._ov_marks_owner() if self._ov_marks_owner is not None else None
        if owner is not pr:
            self._ov_marks_owner = weakref.ref(pr) if pr is not None else None
            if not m.is_empty():
                m.clear()
        return m

    def fullres_marks_count(self) -> int:
        return self.fullres_marks.count()

    def fullres_marks_tracker(self) -> MarksTracker:
        return MarksTracker(self)

    def _fullres_marks_reset(self) -> None:
        """
        点群を入れ替えるときに呼ぶ。印を、その場で空にする。
        あとで（次に印を見たときに）空にする作りだと、そのあいだに履歴が古い印を写し、
        新しい点群で「元に戻す」と、前の点群の印が生き返る（2026-11-20、独立レビューで再現）
        """
        m = self._ov_marks
        if m is not None and not m.is_empty():
            m.clear()
        self._ov_marks_owner = None

    def restore_all_fullres_marks(self) -> int:
        """全点表示で消した点を、全部戻す。戻した数を返す"""
        return self.fullres_marks.clear()

    # ------------------------------------------------------------------ 作業の状態の保存（.npz）
    def _fullres_marks_arrays(self) -> dict:
        """作業の状態のファイルへ入れる配列（印が無ければ空の辞書。古い版でも読める形のまま）"""
        m = self.fullres_marks
        scans = m.scans()
        if not scans:
            return {}
        return {
            "fullres_mark_scans": np.concatenate([np.full(len(m.rows_of(s)), s, dtype=np.int32) for s in scans]),
            "fullres_mark_rows": np.concatenate([m.rows_of(s) for s in scans]),
        }

    @staticmethod
    def _fullres_marks_read(data):
        """作業の状態のファイルから、印の配列を読む（当てるのは _fullres_marks_apply。ファイルに無ければ None）"""
        files = getattr(data, "files", None) or list(data.keys())
        if "fullres_mark_scans" not in files or "fullres_mark_rows" not in files:
            return None
        sids = np.asarray(data["fullres_mark_scans"], dtype=np.int32)
        rows = np.asarray(data["fullres_mark_rows"], dtype=np.int64)
        return (sids, rows) if (len(sids) == len(rows) and len(rows) > 0) else None

    def _fullres_marks_apply(self, marks_data) -> int:
        """読んだ印を当てる（None なら、印は空になる）。レイヤーの定義を確かめ終わってから呼ぶ"""
        m = self.fullres_marks
        m.clear()
        if marks_data is None:
            return 0
        return m.add(marks_data[0], marks_data[1])

    def _on_fullres_marks_changed(self) -> None:
        if self._ov_skip_refresh:      # 自分で消したときは、消した点だけを外してある（全部を調べ直さない）
            return
        self._overlay_refresh_visible()
        try:
            self.update()
        except RuntimeError:
            pass

    # ------------------------------------------------------------------ 重ね描きとつなぐ
    def _overlay_edit_attach(self, rows, sids, scan_steps: Optional[Sequence] = None) -> None:
        """重ね描きを作ったときに呼ぶ。rows が無い（LOD の重ね描き・古い呼び方）ときは、今までどおり表示専用"""
        self._ov_selected = _EMPTY
        self._ov_sel_owner = None
        self._ov_pending_erase = None
        self._ov_boxes = None
        self._ov_index_dirty = True
        src = getattr(self, "_overlay_source", None)
        n = len(src[0]) if src is not None else 0
        if rows is None or sids is None or n == 0 or len(rows) != n or len(sids) != n:
            self._ov_rows = self._ov_sids = self._ov_editable = self._ov_visible = None
            self._ov_all_visible = True
            return
        rows = np.ascontiguousarray(rows, dtype=np.int64)
        sids = np.ascontiguousarray(sids, dtype=np.int32)
        known = rows >= 0
        loaded = np.zeros(n, dtype=bool)
        editable = known.copy()
        steps = list(scan_steps) if scan_steps is not None else []
        for scan in np.unique(sids):
            sel = np.flatnonzero(sids == scan)
            step = steps[int(scan)] if 0 <= int(scan) < len(steps) else None
            if step is None or int(step) < 1:
                editable[sel] = False       # 読み込んだ点と見分けられないスキャンは、表示専用のまま
                continue
            loaded[sel] = loaded_row_mask(rows[sel], int(step)) & known[sel]
        editable &= ~loaded
        self._ov_rows, self._ov_sids = rows, sids
        self._ov_loaded = loaded
        self._ov_editable = editable
        self._ov_hide_loaded = self._overlay_main_draws_all()
        self._overlay_refresh_visible()

    def _overlay_edit_detach(self) -> None:
        self._ov_rows = self._ov_sids = self._ov_editable = self._ov_visible = self._ov_loaded = None
        self._ov_all_visible = True
        self._ov_boxes = None
        self._ov_selected = _EMPTY
        self._ov_sel_owner = None
        self._ov_pending_erase = None
        self._ov_index_dirty = True

    def overlay_editable_count(self) -> int:
        return int(np.count_nonzero(self._ov_editable)) if self._ov_editable is not None else 0

    def _overlay_refresh_visible(self) -> None:
        """描く点を決め直す（読み込んだ点と同じ行と、消した点を除く）"""
        if self._ov_rows is None:
            return
        marked = self.fullres_marks.contains(self._ov_sids, self._ov_rows) & self._ov_editable
        visible = ~(self._ov_loaded | marked) if self._ov_hide_loaded else ~marked
        self._ov_visible = visible
        self._ov_all_visible = bool(visible.all())
        self._ov_index_dirty = True
        sel = self._ov_selected
        if len(sel):
            self._ov_selected = sel[visible[sel]]

    def _overlay_main_draws_all(self) -> bool:
        """
        本体が、箱の中の読み込んだ点を間引かずに描いているか（_update_display_arrays と同じ決まり）。
        間引いて描いているあいだは、重ね描きが「読み込んだ点と同じ行」も描く（描かないと、どこにも描かれない点が出る）
        """
        budget = int(getattr(self, "point_budget", 0) or 0)
        if budget <= 0:
            return True
        if getattr(self, "section_full_detail", False):
            budget = max(budget, int(self.SECTION_DETAIL_MAX_POINTS))
        return int(getattr(self, "section_box_count", 0) or 0) <= budget

    def _overlay_pickable(self) -> np.ndarray:
        """
        いま選べる重ね描きの点（真偽の配列）。読み込みで飛ばされた点は、通常のレイヤーの点として扱う:
        通常のレイヤーが非表示かロック中なら選べない。隠したスキャンの点も選べない（本体の点と同じ決まり）
        """
        pickable = self._ov_editable & self._ov_visible
        layer = self.layers.get(self.LAYER_NORMAL)
        if layer is None or layer.locked or not layer.visible:
            return np.zeros(len(pickable), dtype=bool)
        hidden = [int(sid) for sid, vis in self.scan_visibility.items() if not vis]
        if hidden:
            pickable &= ~np.isin(self._ov_sids, hidden)
        return pickable

    def _overlay_shape_is_valid(self, label: str) -> bool:
        """いまの形が、選ぶのに足りる大きさか（本体の選び方と同じ下限）"""
        if label.startswith("ブラシ"):
            return len(self.brush_stroke_points) > 0
        if label.startswith("多角形"):
            return len(self.polygon_points) >= 3
        if label.startswith("範囲") and self.selection_shape == self.SELECT_RECT:
            rect = QRect(self.selection_start, self.selection_end).normalized()
            return rect.width() >= 4 and rect.height() >= 4
        if label.startswith("範囲") and self.selection_shape == self.SELECT_CIRCLE:
            return math.hypot(self.selection_end.x() - self.selection_start.x(),
                              self.selection_end.y() - self.selection_start.y()) >= 3
        return False

    def _overlay_select_without_main(self, label: str) -> None:
        """
        本体に選べる点が 1 つも無いとき（箱の中の読み込んだ点を、先に全部消したあと、など）に呼ぶ。
        重ね描きに選べる点があれば、本体は「0 点」として、いつもの選択の流れに乗せる
        （これが無いと、読み込んだ点を消したあとに残る「飛ばされた点」を、選べなくなる。2026-11-20、独立レビュー）
        """
        if self._ov_rows is None or getattr(self, "overlay_kind", "") != "section":
            return
        if not self._overlay_shape_is_valid(label) or not np.any(self._overlay_pickable()):
            return
        self._apply_selection_with_modifiers(np.empty(0, dtype=np.int64), label)

    # ------------------------------------------------------------------ 選ぶ
    def overlay_selection(self) -> np.ndarray:
        """重ね描きで選んでいる点。本体の選択が、選んだときの配列のままのあいだだけ有効"""
        if len(self._ov_selected) == 0:
            return _EMPTY
        if self._ov_rows is None or self._ov_sel_owner is not getattr(self, "selected_indices", None):
            self._ov_selected = _EMPTY
            self._ov_sel_owner = None
            return _EMPTY
        return self._ov_selected

    def overlay_selection_count(self) -> int:
        return int(len(self.overlay_selection()))

    def _overlay_candidates(self, min_x: float, max_x: float, min_y: float, max_y: float) -> np.ndarray:
        """画面上の四角の中に映りうる、選べる点の番号（昇順）。四角の中の選べる点は、必ず入っている"""
        src = self._overlay_source
        pts = src[0]
        n = len(pts)
        pickable = self._overlay_pickable()
        if n < OVERLAY_CULL_MIN_POINTS:
            return np.flatnonzero(pickable)
        if self._ov_boxes is None:
            self._ov_boxes = block_boxes_of(pts, OVERLAY_BLOCK)
        lo, hi = self._ov_boxes
        alive = blocks_near_screen_box(
            lo, hi, self._mvp_matrix(), float(self.width()), float(self.height()), min_x, max_x, min_y, max_y
        )
        if alive is None or np.count_nonzero(alive) > CULL_SKIP_FRACTION * len(alive):
            return np.flatnonzero(pickable)
        blocks = np.flatnonzero(alive)
        if len(blocks) == 0:
            return _EMPTY
        breaks = np.flatnonzero(np.diff(blocks) != 1) + 1
        idx = np.concatenate([
            np.arange(int(run[0]) * OVERLAY_BLOCK, min(n, (int(run[-1]) + 1) * OVERLAY_BLOCK), dtype=np.int64)
            for run in np.split(blocks, breaks)
        ])
        return idx[pickable[idx]]

    def _overlay_pick_for(self, label_prefix: str) -> np.ndarray:
        """いまの形（矩形・円・多角形・ブラシ）の中にある、重ね描きの選べる点の番号"""
        if (
            self._ov_rows is None or getattr(self, "overlay_kind", "") != "section"
            or getattr(self, "_overlay_source", None) is None or not np.any(self._ov_editable)
        ):
            return _EMPTY
        try:
            return self._overlay_pick_shape(str(label_prefix))
        except Exception as e:   # 重ね描きの選択のせいで、本体の選択まで止めない（原因はログに残す）
            log_warning(f"全点表示の点の選択に失敗しました（読み込んだ点の選択は、そのまま続けます）: {e!r}")
            return _EMPTY

    def _overlay_pick_shape(self, label: str) -> np.ndarray:
        if label.startswith("ブラシ"):
            if len(self.brush_stroke_points) == 0:
                return _EMPTY
            r = float(max(4, self.brush_radius))
            stroke = np.array([[p.x(), p.y()] for p in self.brush_stroke_points], dtype=np.float32)
            box = (float(stroke[:, 0].min()) - r, float(stroke[:, 0].max()) + r,
                   float(stroke[:, 1].min()) - r, float(stroke[:, 1].max()) + r)
            test = lambda x, y, z: hit_brush(x, y, z, stroke, r)  # noqa: E731
        elif label.startswith("多角形"):
            if len(self.polygon_points) < 3:
                return _EMPTY
            poly = np.array([[p.x(), p.y()] for p in self.polygon_points], dtype=np.float32)
            box = (float(poly[:, 0].min()), float(poly[:, 0].max()), float(poly[:, 1].min()), float(poly[:, 1].max()))
            test = lambda x, y, z: hit_polygon(x, y, z, poly)  # noqa: E731
        elif label.startswith("範囲") and self.selection_shape == self.SELECT_RECT:
            rect = QRect(self.selection_start, self.selection_end).normalized()
            if rect.width() < 4 or rect.height() < 4:
                return _EMPTY
            box = (rect.left(), rect.right(), rect.top(), rect.bottom())
            test = lambda x, y, z: hit_rect(x, y, z, *box)  # noqa: E731
        elif label.startswith("範囲") and self.selection_shape == self.SELECT_CIRCLE:
            cx, cy = self.selection_start.x(), self.selection_start.y()
            radius = math.hypot(self.selection_end.x() - cx, self.selection_end.y() - cy)
            if radius < 3:
                return _EMPTY
            box = (cx - radius, cx + radius, cy - radius, cy + radius)
            test = lambda x, y, z: hit_circle(x, y, z, cx, cy, radius)  # noqa: E731
        else:
            return _EMPTY
        cand = self._overlay_candidates(*box)
        if len(cand) == 0:
            return _EMPTY
        scr_x, scr_y, valid_z = self._project_to_screen(self._overlay_source[0][cand])
        return cand[test(scr_x, scr_y, valid_z)]

    def _overlay_set_selection(self, pick: np.ndarray, mods, had_main_selection: bool, prior: np.ndarray) -> None:
        """本体の選択を決めたあとに呼ぶ。Shift で足す・Ctrl で外す・それ以外は選び直す（本体と同じ決まり）"""
        pick = np.asarray(pick, dtype=np.int64)
        if mods & Qt.ShiftModifier:
            combined = np.union1d(prior, pick)
        elif (mods & Qt.ControlModifier) and (had_main_selection or len(prior) > 0):
            combined = np.setdiff1d(prior, pick)
        else:
            combined = pick
        self._ov_selected = combined.astype(np.int64)
        self._ov_sel_owner = self.selected_indices if len(combined) else None

    # ------------------------------------------------------------------ 消す
    def _overlay_erase(self, idx: np.ndarray) -> int:
        idx = np.asarray(idx, dtype=np.int64)
        if self._ov_rows is None or len(idx) == 0:
            return 0
        idx = idx[self._ov_editable[idx] & self._ov_visible[idx]]
        if len(idx) == 0:
            return 0
        marks = self.fullres_marks
        # 数百万点の重ね描きで、なぞるたびに全部の点を印と突き合わせると遅い。消した点だけを、描く点から外す
        self._ov_skip_refresh = True
        try:
            added = marks.add(self._ov_sids[idx], self._ov_rows[idx])
        finally:
            self._ov_skip_refresh = False
        self._ov_visible[idx] = False
        self._ov_all_visible = False
        self._ov_index_dirty = True
        sel = self._ov_selected
        if len(sel):
            self._ov_selected = sel[self._ov_visible[sel]]
        try:
            self.update()
        except RuntimeError:
            pass
        return added

    def overlay_pending_erase_count(self) -> int:
        p = self._ov_pending_erase
        return int(len(p)) if p is not None else 0

    def commit_overlay_erase(self) -> int:
        """即削除モードで選んだ重ね描きの点を消す。消した数を返す"""
        p = self._ov_pending_erase
        self._ov_pending_erase = None
        return self._overlay_erase(p) if p is not None else 0

    def erase_overlay_selection(self) -> int:
        """重ね描きで選んでいる点を消す。消した数を返す"""
        sel = self.overlay_selection()
        self._ov_selected = _EMPTY
        self._ov_sel_owner = None
        return self._overlay_erase(sel)

    # ------------------------------------------------------------------ 描く
    def _overlay_draw_points(self, render_count: int, use_vbo: bool = True) -> None:
        """
        重ね描きの点を描く（_draw_section_overlay の中から呼ぶ。頂点と色の配列は、呼ぶ側が指してある）。
        全部を描くなら今までどおり。描かない点があるときは、描く点の番号で描く。そのあと、選んでいる点を選択の色で重ねる
        """
        if self._ov_rows is not None:
            hide = self._overlay_main_draws_all()
            if hide != self._ov_hide_loaded:      # 本体の描き方（間引くか）が変わったら、描く点を決め直す
                self._ov_hide_loaded = hide
                self._overlay_refresh_visible()
        if self._ov_visible is None or self._ov_all_visible:
            gl.glDrawArrays(gl.GL_POINTS, 0, int(render_count))
        else:
            if self._ov_index_dirty or getattr(self, "_ov_index_array", None) is None:
                self._ov_index_array = np.ascontiguousarray(np.flatnonzero(self._ov_visible), dtype=np.uint32)
                self._ov_index_count = int(len(self._ov_index_array))
                self._ov_index_uploaded = False
                self._ov_index_dirty = False
            count = self._ov_index_count
            if (self.is_rotating or self.is_panning) and count > 800000:
                count = max(400000, count // 2)
            if count > 0:
                if use_vbo:
                    if self._vbo_overlay_index is None:
                        self._vbo_overlay_index = gl.glGenBuffers(1)
                        self._ov_index_uploaded = False
                    gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self._vbo_overlay_index)
                    if not self._ov_index_uploaded:
                        arr = self._ov_index_array
                        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, arr.nbytes, arr, gl.GL_DYNAMIC_DRAW)
                        self._ov_index_uploaded = True
                    gl.glDrawElements(gl.GL_POINTS, count, gl.GL_UNSIGNED_INT, None)
                    gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0)
                else:
                    gl.glDrawElements(gl.GL_POINTS, count, gl.GL_UNSIGNED_INT, self._ov_index_array)
        sel = self.overlay_selection()
        if len(sel):
            idx = np.ascontiguousarray(sel, dtype=np.uint32)
            gl.glDisableClientState(gl.GL_COLOR_ARRAY)
            gl.glColor3f(float(SELECTION_COLOR[0]), float(SELECTION_COLOR[1]), float(SELECTION_COLOR[2]))
            gl.glDrawElements(gl.GL_POINTS, len(idx), gl.GL_UNSIGNED_INT, idx)
            gl.glEnableClientState(gl.GL_COLOR_ARRAY)

    # ------------------------------------------------------------------ 画面に出す文のための数
    def overlay_edit_summary(self) -> Tuple[int, int, int]:
        """(重ね描きで選べる点の数, 選んでいる数, 消した点の印の数)"""
        return self.overlay_editable_count(), self.overlay_selection_count(), self.fullres_marks_count()
