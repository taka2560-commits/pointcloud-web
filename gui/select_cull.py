# -*- coding: utf-8 -*-
"""
選択の候補のふるい: 矩形・円・多角形・ブラシで選ぶとき、画面の位置を計算する点を、形の近くにありうる点だけに絞る。
gui/gl_viewer.py: PointCloudViewer に混ぜて使う（ビューア本体の変更を小さくするため、別のファイルに分けてある）。

なぜ要るか（2026-11-14、利用者の実データ 5,490 万点・全点表示で実測）
    以前は、選ぶたびに、選べる点の全部について画面の位置を計算していた。
      選べる点の判定 0.19 秒 + 点の取り出し 0.51 秒 + 画面の位置の計算 1.36 秒 + 形の判定 0.1 秒 = 約 2.1 秒
    ブラシでなぞった所が画面のごく一部でも、同じだけかかる。

いまの作り
    ・点を、番号の順に CULL_BLOCK 点ずつの区画に分け、区画ごとに点を囲む箱（最小・最大）を 1 回だけ作っておく。
      点群のファイルは、近い点が近い番号に並んでいることが多い（スキャンの順・八分木の順）ので、区画の箱は小さい。
    ・選ぶ形を囲む画面上の四角（少し広げる）を、視点から伸びる 4 枚の面に直し、箱がどれかの面の外にすべて出ている区画を捨てる。
    ・残った区画の点だけを候補にして、あとは今までと同じ計算（_get_selectable_mask・_project_to_screen・形の判定）を通す。

守っている決まり（tests/test_select_cull.py が見張る）
    ・選ばれる点は、ふるわないときと同じ。ふるいは「形の中に入りうる点を 1 つも落とさない」側に倒してある:
      四角は CULL_PAD_PX だけ広げる。視点のすぐ近く（CULL_NEAR_W より手前）に点がありうる区画は、面では捨てない
      （視点に近い点は、画面の位置の計算の誤差が大きいため）。
    ・点の並びがばらばらで、区画の箱が効かないとき（残る区画が多いとき）は、今までどおり全部の点を候補にする。
    ・「選べる点が 1 つも無いときは、何もしない」という以前の動きを保つ（候補が空のときだけ、全点で確かめる）。
"""

import weakref

import numpy as np

CULL_BLOCK = 16384              # 区画の大きさ（点数）
CULL_MIN_POINTS = 4 * CULL_BLOCK  # これより少ない点群では、ふるわない（全部を計算しても速い）
CULL_PAD_PX = 2.0               # 画面上の四角を広げる幅 [px]（画面の位置の計算は float32 で、丸めの誤差がある）
CULL_NEAR_W = 0.5               # 視点からの奥行きがこれより小さい点がありうる区画は、面では捨てない
CULL_SKIP_FRACTION = 0.6        # 残る区画がこの割合より多ければ、ふるっても得が無いので、全部の点を候補にする


class SelectionCullMixin:
    """PointCloudViewer に混ぜる、選択の候補のふるい"""

    # 区画の箱: (points_centered への弱い参照, 最小 (区画数, 3), 最大 (区画数, 3))。点群が替われば作り直す
    _cull_boxes = None

    def _block_boxes(self):
        """区画ごとの箱（最小・最大。float64）。同じ点群なら、作ってあるものを返す"""
        pc = self.points_centered
        cache = self._cull_boxes
        if cache is not None and cache[0]() is pc:
            return cache[1], cache[2]
        n = len(pc)
        n_full = n // CULL_BLOCK
        n_blocks = (n + CULL_BLOCK - 1) // CULL_BLOCK
        lo = np.empty((n_blocks, 3), dtype=np.float64)
        hi = np.empty((n_blocks, 3), dtype=np.float64)
        if n_full:
            body = pc[: n_full * CULL_BLOCK].reshape(n_full, CULL_BLOCK, 3)
            lo[:n_full] = body.min(axis=1)
            hi[:n_full] = body.max(axis=1)
        if n_blocks > n_full:
            tail = pc[n_full * CULL_BLOCK:]
            lo[n_full] = tail.min(axis=0)
            hi[n_full] = tail.max(axis=0)
        self._cull_boxes = (weakref.ref(pc), lo, hi)
        return lo, hi

    def _blocks_near_screen_box(self, min_x: float, max_x: float, min_y: float, max_y: float):
        """
        画面上の四角（min_x〜max_x, min_y〜max_y [px]）の中に点が映りうる区画の印（真偽の配列）。
        ふるえないとき（画面の大きさが 0・行列が数でない、など）は None
        """
        width, height = float(self.width()), float(self.height())
        if width <= 0 or height <= 0:
            return None
        mvp = np.asarray(self._mvp_matrix(), dtype=np.float64)      # clip = [x, y, z, 1] @ mvp（_project_to_screen と同じ行列）
        corners = np.array([min_x, max_x, min_y, max_y], dtype=np.float64)
        if not (np.isfinite(mvp).all() and np.isfinite(corners).all()):
            return None
        # 画面の位置 → 正規化した座標（_project_to_screen の式の逆）
        ax0 = (min_x - CULL_PAD_PX) / width * 2.0 - 1.0
        ax1 = (max_x + CULL_PAD_PX) / width * 2.0 - 1.0
        ay0 = 1.0 - (max_y + CULL_PAD_PX) / height * 2.0
        ay1 = 1.0 - (min_y - CULL_PAD_PX) / height * 2.0
        clip_x, clip_y, clip_w = mvp[:, 0], mvp[:, 1], mvp[:, 3]
        # w > 0 の点について: 四角の中 ⇔ 4 つの式がどれも 0 以上（どれも点の座標の 1 次式）
        sides = (clip_x - ax0 * clip_w, ax1 * clip_w - clip_x, clip_y - ay0 * clip_w, ay1 * clip_w - clip_y)
        lo, hi = self._block_boxes()

        def largest(plane):      # 箱の中での、式の最大
            normal = plane[:3]
            return np.where(normal >= 0, hi, lo) @ normal + plane[3]

        def smallest(plane):
            normal = plane[:3]
            return np.where(normal >= 0, lo, hi) @ normal + plane[3]

        in_front = largest(clip_w) > 0.0            # 視点より前に点がありうる（後ろの点は、もともと選ばれない）
        near = smallest(clip_w) < CULL_NEAR_W       # 視点のすぐ近くに点がありうる区画は、面では捨てない
        inside = np.ones(len(lo), dtype=bool)
        for plane in sides:
            inside &= largest(plane) >= 0.0
        return in_front & (near | inside)

    def _selection_candidates(self, min_x: float, max_x: float, min_y: float, max_y: float):
        """
        画面上の四角の中に映りうる、選べる点の番号（昇順）。四角の中の選べる点は、必ず入っている（外の点も混じる）。
        選べる点が 1 つも無いときは None（呼び出し側は、何もせずに戻る。以前と同じ動き）
        """
        n = len(self.points_centered) if self.points_centered is not None else 0
        if n == 0:
            return None
        alive = self._blocks_near_screen_box(min_x, max_x, min_y, max_y) if n >= CULL_MIN_POINTS else None
        if alive is None or np.count_nonzero(alive) > CULL_SKIP_FRACTION * len(alive):
            cand = np.where(self._get_selectable_mask())[0]          # 今までどおり、全部の点から
            return cand if len(cand) else None
        blocks = np.flatnonzero(alive)
        if len(blocks):
            breaks = np.flatnonzero(np.diff(blocks) != 1) + 1
            idx = np.concatenate([
                np.arange(int(run[0]) * CULL_BLOCK, min(n, (int(run[-1]) + 1) * CULL_BLOCK), dtype=np.int64)
                for run in np.split(blocks, breaks)
            ])
            cand = idx[self._get_selectable_mask(idx)]
        else:
            cand = np.empty(0, dtype=np.int64)
        if len(cand) == 0 and not np.any(self._get_selectable_mask()):
            return None
        return cand
