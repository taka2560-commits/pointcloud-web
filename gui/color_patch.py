# -*- coding: utf-8 -*-
"""
色の部分更新: 選択やレイヤーが変わったとき、変わった点の色だけを作り直し、GPU へも変わった所だけを送る。
gui/gl_viewer.py: PointCloudViewer に混ぜて使う（ビューア本体の変更を小さくするため、別のファイルに分けてある）。

なぜ要るか（2026-11-14、利用者の実データ 5,490 万点・全点表示で実測）
    以前は、選択が 1 点変わるだけでも、次の 3 つを全点ぶん行っていた。
      1. 全点の色の配列を作り直す（_update_active_colors）            約 0.75 秒
      2. 表示用の色の配列を、全点ぶん写し直す（_update_display_colors_only） 約 0.5 秒
      3. 色の VBO（660MB）を、まるごと GPU へ送り直す                    約 0.4 秒
    選ぶ・選択を解除する、のたびに約 1.7〜2 秒。手作業のノイズ取りでは、これを何百回もくり返す。

いまの作り
    ・選択が変わったとき（_sync_selection_colors）: 選択から外れた点は元の色に戻し、新しく選んだ点は黄色にする。
      触るのは、その点だけ。
    ・レイヤーが変わったとき（_recolor_indices）: 移した点の色だけを作り直す。
    ・表示用の色の配列も、その点の所だけを書き換える（_patch_display_colors）。
    ・GPU へは、変わった点を含む区画（PATCH_BLOCK 点ごと）だけを送り直す（_upload_color_patches。paintGL から呼ぶ）。

守っている決まり（tests/test_color_patch.py が見張る）
    ・部分更新をくり返した結果は、全部を作り直した結果（_update_active_colors）と 1 つも違わないこと。
      色の式はビューア本体の _base_colors() だけにあり、全点でも一部でも同じ式を通る。
    ・読み込みの直後は、active_colors が colors_raw と同じ配列を指している（メモリを節約するため）。
      そこへ黄色を書き込むと元の色が壊れるので、自前の配列になるまでは、全部を作り直す道を通る。
    ・色の式が読む設定（配色・パレット・陰影・高さの範囲・背景の明暗・レイヤーの色）を、全部を作り直したときに控えておく
      （_color_stamp）。部分更新の入口で、いまの設定が控えと違っていたら、全部を作り直す。
      色の配列を作り直さずに設定だけを変える経路（高さの範囲の下書き表示 preview_custom_height_range など）があり、
      以前は「次の選択で全部を作り直す」ことで直っていたので、その動きを保つため（2026-11-14 の独立レビューで判明）。
    ・変わる点が多いとき（全体の 1/4 を超える）は、全部を作り直すほうが速いので、そちらを通る。
    ・迷ったら全部を作り直す（表示用の配列の長さが合わない・並びが想定と違う、など）。
"""

import numpy as np
import OpenGL.GL as gl

from core.logger import log_warning
from gui.noise_palette import SELECTION_COLOR

PATCH_BLOCK = 65536             # GPU へ送り直すときの区画の大きさ（点数）。1 区画 = 768KB
FULL_REBUILD_FRACTION = 4       # 変わる点が全体の 1/4 を超えたら、全部を作り直す
FULL_UPLOAD_FRACTION = 2        # 送り直す区画が全体の 1/2 を超えたら、まるごと送る


class ColorPatchMixin:
    """PointCloudViewer に混ぜる、色の部分更新"""

    # active_colors の中で、いま黄色（選択中）に塗ってある点の番号。_update_active_colors と _sync_selection_colors だけが書き換える
    _colored_selection = np.empty(0, dtype=np.int64)
    # GPU へ送り直す区画: (そのときの表示点数, 区画の番号の配列)。無ければ None
    _color_patch = None
    _color_patch_warned = False
    # active_colors を全点ぶん作ったときの、色の式が読む設定の控え（_color_stamp）。None = 式で作った色ではない（読み込みの直後など）
    _colors_stamp = None

    # ---------------------------------------------------------------- 色の配列（全点ぶん）
    def _color_stamp(self):
        """色の式（_base_colors）が読む設定の控え。式が読む設定を足したら、ここにも足すこと"""
        layers = tuple(sorted(
            (int(lid), tuple(float(c) for c in info.color), bool(info.use_layer_color)) for lid, info in self.layers.items()
        ))
        custom = None
        if getattr(self, "custom_height_enabled", False):
            custom = (float(getattr(self, "custom_height_min", 0.0)), float(getattr(self, "custom_height_max", 0.0)))
        return (
            len(self.points_centered) if self.points_centered is not None else 0,
            self.color_mode, getattr(self, "color_palette", None),
            bool(getattr(self, "shading_enabled", False)), float(getattr(self, "shading_strength", 0.65)),
            custom, bool(self._bg_is_light()), layers,
        )

    def _forget_color_state(self) -> None:
        """点群を入れ替えたときに呼ぶ。前の点群についての覚え（黄色に塗った点・送り直す区画・設定の控え）を捨てる"""
        self._colored_selection = np.empty(0, dtype=np.int64)
        self._color_patch = None
        self._colors_stamp = None

    def _colors_are_private(self) -> bool:
        """active_colors が、その場で一部だけ書き換えてよい状態か（自前の配列で、いまの設定で作った色であること）"""
        colors = self.active_colors
        n = len(self.points_centered) if self.points_centered is not None else 0
        if colors is None or n == 0 or colors.shape != (n, 3) or colors.dtype != np.float32 or not colors.flags.writeable:
            return False
        # 読み込みの直後は colors_raw と同じ配列（gui/worker_thread.py: FileLoadWorker が、写しを作らずに渡す）
        if self.colors_raw is not None and np.may_share_memory(colors, self.colors_raw):
            return False
        # 色を作ったあとで設定が変わっていたら、active_colors は古い（全部を作り直す）
        return self._colors_stamp is not None and self._colors_stamp == self._color_stamp()

    def _valid_selection(self) -> np.ndarray:
        n = len(self.points_centered) if self.points_centered is not None else 0
        sel = np.asarray(self.selected_indices, dtype=np.int64)
        if len(sel) == 0:
            return np.empty(0, dtype=np.int64)
        return sel[(sel >= 0) & (sel < n)]

    def _sync_selection_colors(self) -> None:
        """
        選択（selected_indices）を変えたあとに呼ぶ。色の配列と表示を、新しい選択に合わせる。
        _update_active_colors() + _update_display_colors_only() と同じ結果になる。
        """
        n = len(self.points_centered) if self.points_centered is not None else 0
        if n == 0:
            return
        old = self._colored_selection
        old = old[old < n]
        new = self._valid_selection()
        if not self._colors_are_private() or (len(old) + len(new)) * FULL_REBUILD_FRACTION > n:
            self._update_active_colors()
            self._update_display_colors_only()
            return
        if len(old) == 0 and len(new) == 0:
            return
        if len(old):
            self.active_colors[old] = self._base_colors(old)
        if len(new):
            self.active_colors[new] = SELECTION_COLOR
        self._colored_selection = new
        self._patch_display_colors(np.concatenate([old, new]) if len(old) and len(new) else (old if len(old) else new))

    def _recolor_indices(self, indices: np.ndarray) -> None:
        """
        レイヤーを移した点（indices）の色を作り直す。表示用の配列は、呼び出し側が作り直す（_update_display_arrays）。
        _update_active_colors() と同じ結果になる。
        """
        n = len(self.points_centered) if self.points_centered is not None else 0
        if n == 0:
            return
        idx = np.asarray(indices, dtype=np.int64)
        if not self._colors_are_private() or len(idx) * FULL_REBUILD_FRACTION > n:
            self._update_active_colors()
            return
        if len(idx) == 0:
            return
        self.active_colors[idx] = self._base_colors(idx)
        painted = self._colored_selection[self._colored_selection < n]
        if len(painted):
            self.active_colors[painted] = SELECTION_COLOR   # 選択中の点は、黄色のまま

    # ---------------------------------------------------------------- 表示用の色の配列と、GPU への転送
    def _patch_display_colors(self, changed: np.ndarray) -> None:
        """色が変わった点（元の点の番号）の所だけ、表示用の色の配列を書き換え、GPU へ送り直す区画を覚える"""
        shown = self.display_orig_indices
        colors = self.display_colors
        if (
            shown is None or colors is None or self.display_points is None
            or len(shown) == 0 or len(colors) != len(shown) or len(self.display_points) != len(shown)
            or colors.dtype != np.float32 or not colors.flags.writeable or not colors.flags.c_contiguous
            or np.may_share_memory(colors, self.active_colors)
            or (self.colors_raw is not None and np.may_share_memory(colors, self.colors_raw))
        ):
            self._update_display_colors_only()    # 迷ったら、全部を写し直す
            return
        if len(changed) == 0:
            return
        # 表示している点は、元の点の番号の昇順に並んでいる（_update_display_arrays / FileLoadWorker）
        pos = np.minimum(np.searchsorted(shown, changed), len(shown) - 1)
        hit = shown[pos] == changed
        pos = pos[hit]
        if len(pos) == 0:
            return
        colors[pos] = self.active_colors[changed[hit]]
        self._queue_color_upload(pos)

    def _queue_color_upload(self, positions: np.ndarray) -> None:
        """表示用の配列の中で色が変わった位置（positions）を含む区画を、次の描画で GPU へ送り直すように覚える"""
        if self._vbo_colors_dirty:
            return                               # まるごと送り直す予定なら、覚えなくてよい
        n_shown = len(self.display_colors)
        blocks = np.unique(positions // PATCH_BLOCK)
        pending = self._color_patch
        if pending is not None and pending[0] == n_shown:
            blocks = np.union1d(pending[1], blocks)
        if len(blocks) * PATCH_BLOCK * FULL_UPLOAD_FRACTION >= n_shown:
            self._color_patch = None
            self._vbo_colors_dirty = True        # 大半が変わったなら、まるごと送るほうが速い
            return
        self._color_patch = (n_shown, blocks)

    def _upload_color_patches(self) -> None:
        """
        paintGL の中で、色の VBO をまるごと送る処理のすぐあとに呼ぶ。覚えておいた区画だけを GPU へ送り直す。
        まるごと送った直後に呼ばれたときは、同じ中身を送り直すだけなので害は無い。
        """
        pending = self._color_patch
        if pending is None:
            return
        self._color_patch = None
        colors = self.display_colors
        if colors is None or pending[0] != len(colors) or getattr(self, "_vbo_colors", None) is None:
            return                               # 表示用の配列が入れ替わった（そのときは、まるごと送り直されている）
        try:
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._vbo_colors)
            # 大きさは 32 ビットで返るので、2GB を超える VBO（約 1.8 億点の表示）では合わず、毎回まるごと送る（絵は正しい）
            if int(gl.glGetBufferParameteriv(gl.GL_ARRAY_BUFFER, gl.GL_BUFFER_SIZE)) != colors.nbytes:
                gl.glBufferData(gl.GL_ARRAY_BUFFER, colors.nbytes, colors, gl.GL_DYNAMIC_DRAW)   # 大きさが合わなければ、まるごと
                return
            blocks = pending[1]
            row_bytes = colors.strides[0]
            # 隣り合う区画は、1 回にまとめて送る
            breaks = np.flatnonzero(np.diff(blocks) != 1) + 1
            for run in np.split(blocks, breaks):
                start = int(run[0]) * PATCH_BLOCK
                stop = min(len(colors), (int(run[-1]) + 1) * PATCH_BLOCK)
                chunk = colors[start:stop]
                gl.glBufferSubData(gl.GL_ARRAY_BUFFER, start * row_bytes, chunk.nbytes, chunk)
        except Exception as e:  # noqa: BLE001  ドライバによる失敗は、次の描画でまるごと送り直して立て直す
            self._vbo_colors_dirty = True
            if not self._color_patch_warned:
                self._color_patch_warned = True
                log_warning(f"色の部分転送に失敗したので、まるごと送り直します: {e}")
            self.update()
