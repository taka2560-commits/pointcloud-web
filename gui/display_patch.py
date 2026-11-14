# -*- coding: utf-8 -*-
"""
表示の部分更新: 点をレイヤーへ移した（消した・戻した）とき、表示用の配列を全点ぶん作り直さずに、移した点の所だけを書き換える。
gui/gl_viewer.py: PointCloudViewer に混ぜて使う（ビューア本体の変更を小さくするため、別のファイルに分けてある）。

なぜ要るか（2026-11-14、利用者の実データ 5,490 万点・全点表示で実測）
    点を 1 つ消すだけでも、次の 2 つを全点ぶん行っていた。
      1. 表示用の配列（点・色・番号）を作り直す（_update_display_arrays）   約 1.3 秒
      2. 点と色の VBO（合わせて 1.3GB）を、まるごと GPU へ送り直す           約 0.3 秒
    手作業のノイズ取りでは、消す・戻すを何百回もくり返す。

いまの作り
    ・消えた点（表示から外れる点）の枠は、詰めずに、**すぐ手前の枠の点の写し**で埋める（「重複の枠」）。
      配列の長さも並びも変わらないので、書き換えるのは消えた点の枠だけ。GPU へも、その枠を含む区画だけを送る。
      同じ点を同じ色で 2 回描くだけなので、絵は変わらない。
    ・戻った点（表示に入る点）は、番号の順が保てる所にある重複の枠へ入れる（自分が消えたときに空いた枠）。
      すき間の中のどの枠へ入れるかは、番号の比で決める（全点を表示していたなら、消える前と同じ枠に戻る）。
      こうしておくと、続けて消した所を「元に戻す」で順に戻すときに、あとから戻る点の枠が残る。
      入れる枠が足りなければ、全部を作り直す。
    ・重複の枠が表示の 1/4 を超えたら、全部を作り直して詰める。

守っている決まり（tests/test_display_patch.py が見張る）
    ・display_orig_indices は、いつも昇順（同じ番号が続くことはある）。
    ・表示している点の集まり（重複を除く）と、それぞれの座標・色は、全部を作り直したとき（_update_display_arrays）と同じ。
    ・display_points[i] は points_centered[display_orig_indices[i]]、display_colors[i] は active_colors[display_orig_indices[i]]。
    ・表示している点の数は shown_point_count()（len(display_points) は、重複の枠のぶん多い）。
      断面の下書き表示（_update_display_arrays_fast_preview）は、重複の枠を除いて作る（_drop_duplicate_slots）。
    ・表示する点を決める設定（断面・レイヤーとスキャンの表示・表示点数の上限）を、全部を作り直したときに控えておく
      （_display_rule_stamp）。部分更新の入口で、いまの設定が控えと違っていたら、全部を作り直す。
    ・表示点数の上限で間引いているとき（何点おきに描くかは、残った点の順番で決まる）は、いつも全部を作り直す。
    ・迷ったら全部を作り直す（下書き表示の途中・配列がほかと共有・先頭の枠が消える、など）。
"""

import numpy as np

from core.logger import log_warning
from gui.color_patch import FULL_REBUILD_FRACTION, expand_ranges, queue_blocks, upload_blocks

DUP_REBUILD_FRACTION = 4        # 重複の枠が表示の 1/4 を超えたら、全部を作り直して詰める
CHANGE_REBUILD_FRACTION = 4     # 移した点が表示の 1/4 を超えたら、全部を作り直す


class DisplayPatchMixin:
    """PointCloudViewer に混ぜる、表示用の配列の部分更新"""

    # 表示用の配列の中の、重複の枠の数（すぐ手前の枠と同じ点が入っている枠）。_update_display_arrays で 0 に戻る
    _display_dups = 0
    # 表示用の配列を全部作ったときの、表示する点を決める設定の控え。None = 部分更新してはいけない（間引き中・読み込みの直後など）
    _display_stamp = None
    # GPU へ送り直す点の区画: (そのときの表示用の配列の長さ, 区画の番号の配列)。無ければ None
    _point_patch = None
    _point_patch_warned = False

    # ---------------------------------------------------------------- 状態
    def shown_point_count(self) -> int:
        """いま描いている点の数（重複の枠を除く）"""
        if self.display_points is None:
            return 0
        n = len(self.display_points)
        if self.display_points is self._preview_base_points:
            n -= self._display_dups
        return max(0, n)

    def _drop_duplicate_slots(self, mask: np.ndarray) -> np.ndarray:
        """
        確定した表示の枠ごとの印（mask。断面の下書き表示が、どの枠を描くかを決めたもの）から、重複の枠を外す。
        下書きの配列には同じ点が 1 回だけ入るので、len() がそのまま描いている点の数になる。
        """
        shown = self.display_orig_indices
        if self._display_dups > 0 and shown is not None and len(shown) == len(mask) and len(shown) > 1:
            mask[1:] &= shown[1:] != shown[:-1]
        return mask

    def _display_rule_stamp(self):
        """表示する点を決める設定の控え（_update_display_arrays が読む設定）。読む設定を足したら、ここにも足すこと"""
        layers = tuple(sorted((int(lid), bool(info.visible)) for lid, info in self.layers.items()))
        scans = tuple(sorted((int(sid), bool(vis)) for sid, vis in self.scan_visibility.items()))
        return (
            len(self.points_centered) if self.points_centered is not None else 0,
            self.is_section_active(), self._section_box_key(), layers, scans,
            int(self.point_budget), bool(self.section_full_detail), int(self.SECTION_DETAIL_MAX_POINTS),
        )

    def _forget_display_state(self) -> None:
        """表示用の配列を入れ替える所で呼ぶ。前の配列についての覚えを捨てる"""
        self._display_dups = 0
        self._display_stamp = None
        self._point_patch = None

    def _note_display_rebuilt(self, n_surviving: int, budget: int) -> None:
        """_update_display_arrays が配列を作り終えたときに呼ぶ。間引いていなければ、部分更新してよい印（設定の控え）を付ける"""
        self._display_dups = 0
        self._point_patch = None
        self._display_stamp = self._display_rule_stamp() if (budget <= 0 or n_surviving <= budget) else None

    def _display_patchable(self) -> bool:
        """表示用の配列を、その場で一部だけ書き換えてよい状態か"""
        pc = self.points_centered
        shown, pts, cols = self.display_orig_indices, self.display_points, self.display_colors
        if pc is None or shown is None or pts is None or cols is None or self.active_colors is None:
            return False
        m = len(shown)
        if m == 0 or pts.shape != (m, 3) or cols.shape != (m, 3) or shown.ndim != 1:
            return False
        # 下書き表示（断面のスライダー・高さの範囲）の途中は、配列が確定した表示のものではない
        if pts is not self._preview_base_points or cols is not self._preview_base_colors:
            return False
        for arr, dtype in ((pts, np.float32), (cols, np.float32), (shown, np.int64)):
            if arr.dtype != dtype or not arr.flags.writeable or not arr.flags.c_contiguous:
                return False
        # 読み込みの直後は、表示用の配列が元の配列そのもの（gui/worker_thread.py: FileLoadWorker が、写しを作らずに渡す）
        if np.may_share_memory(pts, pc) or np.may_share_memory(cols, self.active_colors):
            return False
        if self.colors_raw is not None and np.may_share_memory(cols, self.colors_raw):
            return False
        return self._display_stamp is not None and self._display_stamp == self._display_rule_stamp()

    # ---------------------------------------------------------------- レイヤーを移したあとの更新
    def _update_after_layer_move(self, indices: np.ndarray) -> None:
        """
        点（indices）のレイヤーを書き換えて layer_mask を合わせたあとに呼ぶ。色と表示を、新しいレイヤーに合わせる。
        _update_active_colors() + _update_display_arrays() と同じ絵になる（速いほうの道を選ぶ）。
        """
        n = len(self.points_centered) if self.points_centered is not None else 0
        idx = np.asarray(indices, dtype=np.int64)
        if n == 0 or not self._colors_are_private() or len(idx) * FULL_REBUILD_FRACTION > n:
            if n:
                self._update_active_colors()
            self._update_display_arrays()
            return
        self._recolor_indices(idx)
        if not self._patch_display_membership(idx):
            self._update_display_arrays()

    def _patch_display_membership(self, indices: np.ndarray) -> bool:
        """
        レイヤーが変わった点（indices。色は作り直してあること）について、表示用の配列を書き換える。
        戻り値: 書き換えたか。False のときは、呼び出し側が全部を作り直す（途中まで書き換えていても、作り直せば正しくなる）。
        """
        if not self._display_patchable():
            return False
        n = len(self.points_centered)
        idx = np.asarray(indices, dtype=np.int64)
        idx = idx[(idx >= 0) & (idx < n)]
        if len(idx) == 0:
            return True
        if len(idx) > 1 and not np.all(idx[1:] > idx[:-1]):
            idx = np.unique(idx)
        shown, pts, cols = self.display_orig_indices, self.display_points, self.display_colors
        m = len(shown)
        if len(idx) * CHANGE_REBUILD_FRACTION > m:
            return False

        # 移した点は、いま表示に入っているか・これから入るべきか
        should = self._get_section_inlier_mask(idx) & self._get_layer_and_scan_visibility_mask(idx)
        left = np.searchsorted(shown, idx, side="left")
        present = shown[np.minimum(left, m - 1)] == idx
        gone = present & ~should             # 表示から外れる
        back = should & ~present             # 表示に入る
        n_gone = int(np.count_nonzero(gone))
        n_back = int(np.count_nonzero(back))
        dups_after = self._display_dups + n_gone - n_back
        real_after = m - dups_after
        # 表示点数の上限は、ここでは見なくてよい: 点が入るのは重複の枠だけなので、全部を作ったときの点数（上限以内）を超えない
        if real_after <= 0 or dups_after * DUP_REBUILD_FRACTION > m:
            return False

        touched = []
        if n_gone:
            slots = self._free_display_slots(shown, idx[gone], left[gone])
            if slots is None:
                return False
            touched.append(slots)
        if n_back:
            slots = self._fill_display_gaps(shown, idx[back])
            if slots is None:
                return False
            touched.append(slots)
        self._display_dups = dups_after

        section_active = self.is_section_active()
        self.section_box_count = real_after if section_active else 0
        self.section_detail_capped = False       # 間引いていない（上のとおり、上限に収まっている）

        if touched:
            positions = touched[0] if len(touched) == 1 else np.concatenate(touched)
            self._queue_point_upload(positions)
            self._queue_color_upload(positions)
        stay = present & should                  # 表示に残る点は、色だけが変わる（レイヤーの固有色など）
        if np.any(stay):
            self._patch_display_colors(idx[stay])
        return True

    def _free_display_slots(self, shown: np.ndarray, gone: np.ndarray, left: np.ndarray):
        """
        表示から外れる点（gone。昇順）の枠を、すぐ手前の「外れない枠」の点の写しで埋める。
        戻り値: 書き換えた枠の位置。先頭の枠が外れる（手前が無い）ときは None
        """
        if self._display_dups > 0:
            right = np.searchsorted(shown, gone, side="right")
            slots = expand_ranges(left, right)       # その点が入っている枠は、すべて
        else:
            slots = left
        if int(slots[0]) == 0:
            return None
        # 続いた枠（連なり）は、連なりのすぐ手前の枠から写す
        run_start = np.empty(len(slots), dtype=bool)
        run_start[0] = True
        np.not_equal(slots[1:], slots[:-1] + 1, out=run_start[1:])
        src = np.maximum.accumulate(np.where(run_start, slots - 1, -1))
        shown[slots] = shown[src]
        self.display_points[slots] = self.display_points[src]
        self.display_colors[slots] = self.display_colors[src]
        return slots

    def _fill_display_gaps(self, shown: np.ndarray, add: np.ndarray):
        """
        表示に入る点（add。昇順・いまは表示に無い）を、番号の順が保てる所にある重複の枠へ入れる。
        戻り値: 書き換えた枠の位置。枠が足りなければ None（何も書き換えない）

        すき間 = ある点 Y の枠（本物 1 つ + 重複）の並び。Y より大きく、次の点 Z より小さい番号の点は、ここにしか入れられない。
        入れたあとも昇順になるように、すき間の中を「Y の写し… 点 1・その写し… 点 2・その写し…」と並べ直す。
        """
        m = len(shown)
        n_total = len(self.points_centered)
        # stop = 番号が add より大きい最初の枠（= すき間の終わり）。同じ stop になる点は、同じすき間に入る
        stop = np.searchsorted(shown, add, side="left")
        first = np.empty(len(add), dtype=bool)
        first[0] = True
        np.not_equal(stop[1:], stop[:-1], out=first[1:])
        group_start = np.flatnonzero(first)
        group_size = np.diff(np.append(group_start, len(add)))
        gap_stop = stop[group_start]
        if int(gap_stop[0]) == 0:
            return None                              # 表示しているどの点よりも番号が小さい（手前に枠が無い）
        gap_index = shown[gap_stop - 1]              # すき間に入っている点 Y
        gap_first = np.searchsorted(shown, gap_index, side="left")     # Y の本物の枠
        if np.any(gap_stop - gap_first - 1 < group_size):
            return None                              # 重複の枠が足りない
        next_index = np.where(gap_stop < m, shown[np.minimum(gap_stop, m - 1)], n_total)   # 次の点 Z（無ければ、点の総数）

        # すき間の中の位置を、番号の比で決める（全点を表示していたなら、消える前と同じ枠）
        group_of = np.cumsum(first) - 1
        rank = np.arange(len(add), dtype=np.int64) - group_start[group_of]
        a, b, y, z = gap_first[group_of], gap_stop[group_of], gap_index[group_of], next_index[group_of]
        slot = a + ((add - y) * (b - a)) // (z - y)
        # 同じすき間の点が、重ならず、すき間からはみ出さないように直す（slot - rank が、a+1 以上・b-点数 以下・減らない）
        base = np.clip(slot - rank, a + 1, b - group_size[group_of])
        slot = np.maximum.accumulate(base) + rank    # すき間は左から順に並んでいるので、前のすき間の値が次へ及ぶことは無い

        # すき間（Y の本物の枠から、終わりまで）を、まるごと並べ直す
        region = expand_ranges(gap_first, gap_stop)
        offset = np.cumsum(gap_stop - gap_first) - (gap_stop - gap_first)      # region の中での、各すき間の始まり
        table = np.full(len(region), -1, dtype=np.int64)       # 枠に入る点の番号。-1 = まだ決まっていない（手前の点の写しになる）
        table[offset] = gap_index
        table[offset[group_of] + (slot - a)] = add
        filled = np.maximum.accumulate(np.where(table >= 0, np.arange(len(region)), -1))   # 手前の「決まっている枠」の位置
        new_index = table[filled]
        shown[region] = new_index
        self.display_points[region] = np.take(self.points_centered, new_index, axis=0)
        self.display_colors[region] = np.take(self.active_colors, new_index, axis=0)
        return region

    # ---------------------------------------------------------------- GPU への転送（点の VBO）
    def _queue_point_upload(self, positions: np.ndarray) -> None:
        """表示用の配列の中で点が変わった位置（positions）を含む区画を、次の描画で GPU へ送り直すように覚える"""
        if self._vbo_points_dirty:
            return                               # まるごと送り直す予定なら、覚えなくてよい
        self._point_patch = queue_blocks(self._point_patch, len(self.display_points), positions)
        if self._point_patch is None:
            self._vbo_points_dirty = True        # 大半が変わったなら、まるごと送るほうが速い

    def _upload_point_patches(self) -> None:
        """
        paintGL の中で、点の VBO をまるごと送る処理のすぐあとに呼ぶ。覚えておいた区画だけを GPU へ送り直す。
        まるごと送った直後に呼ばれたときは、同じ中身を送り直すだけなので害は無い。
        """
        pending = self._point_patch
        if pending is None:
            return
        self._point_patch = None
        pts = self.display_points
        if pts is None or pending[0] != len(pts) or getattr(self, "_vbo_points", None) is None:
            return                               # 表示用の配列が入れ替わった（そのときは、まるごと送り直されている）
        try:
            upload_blocks(self._vbo_points, pts, pending[1])
        except Exception as e:  # noqa: BLE001  ドライバによる失敗は、次の描画でまるごと送り直して立て直す
            self._vbo_points_dirty = True
            if not self._point_patch_warned:
                self._point_patch_warned = True
                log_warning(f"点の部分転送に失敗したので、まるごと送り直します: {e}")
            self.update()
