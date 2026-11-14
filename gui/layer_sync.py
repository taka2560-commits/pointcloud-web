# -*- coding: utf-8 -*-
"""
レイヤーの整合の確認と、点数の数え直しを速くする。
gui/gl_viewer.py: PointCloudViewer に混ぜて使う（ビューア本体の変更を小さくするため、別のファイルに分けてある）。

なぜ要るか（2026-11-14、利用者の実データ 5,490 万点で実測）
    点を消す・戻す 1 回の操作で、次のことを全点ぶん何度も行っていた。
      1. layer_mask と layer_ids が合っているかの確認（_ensure_layer_sync）が 4〜8 回。1 回 0.06〜0.08 秒
      2. レイヤーごとの点数（get_layer_counts）とスキャンごとの点数（get_scan_counts）を、全点の並べ替えで 2 回ずつ数える
    さらに、自動フィルタで「自動除去:」のノイズレイヤーができていると、1 の確認と layer_mask の作り直しが
    1 回 0.23 秒になっていた（np.isin が、点ごとに表を引く遅い道に入るため）。即削除のブラシ 1 回で、合わせて 1〜3 秒。

いまの作り
    ・「この点はノイズ扱いのレイヤーか」は、レイヤーの番号ごとの比べ（layer_ids == 番号）を重ねて作る（_is_noise_point。
      ノイズ扱いのレイヤーが多いときだけ np.isin）。結果は np.isin と同じ。
    ・整合の確認は、合っていると確かめた（または layer_mask を layer_ids から作った）ときの状態を覚えておき、
      同じ状態なら、全点の比べを省く（_layers_known_synced）。覚えるのは、2 つの配列そのもの（弱い参照）・ノイズ扱いの
      レイヤーの番号・layer_mask の True の数。どれかが違えば、今までどおり全点を比べる。
    ・点数は、数えたときの「レイヤーの版」（_layer_version。layer_ids が変わるたびに進む）と一緒に覚えておき、
      版が同じなら数え直さない。数え方も、レイヤーが少ないときは番号ごとの比べにした（結果は np.unique と同じ）。
    ・スキャンごとの点数は、読み込みのあと変わらないので、scan_ids が同じ配列なら数え直さない。

守っている決まり（tests/test_layer_sync.py が見張る）
    ・決まりを守った操作のあとは、_ensure_layer_sync / _sync_mask_from_ids / get_layer_counts / get_scan_counts の結果が、以前の作りと同じ。
    ・layer_mask を配列ごと差し替えた・その場で書き換えて True の数が変わった・ノイズ扱いのレイヤーが変わった、のどれでも、
      次の _ensure_layer_sync が今までどおり全点を比べて layer_ids を合わせる。
    ・**layer_ids をその場で書き換えたら、必ず _sync_mask_from_ids() を呼ぶ**（ビューアの中と gui/layer_history.py は、そうしている）。
      呼ばないと、整合の確認も点数も「変わっていない」と見なす。保存の前など、確実に確かめたい所は _ensure_layer_sync(force=True)。
      書き換える所が増えていないかは、TestWritersAreKnown が、ソースの中の書き換えの数で見張る。
    ・書き換えの途中で失敗したとき（元に戻す処理の例外など）は、_forget_layer_sync() で覚えを捨てる（次の確認が、全点を比べる）。
"""

import weakref

import numpy as np

NOISE_COMPARE_MAX = 8       # ノイズ扱いのレイヤーがこの数までなら、番号ごとの比べで作る（1 つ 0.012 秒。np.isin は 0.22 秒）
COUNT_COMPARE_MAX = 5       # レイヤーがこの数までなら、点数を番号ごとの比べで数える（1 つ 0.013 秒。np.unique は 0.075 秒）


class LayerSyncMixin:
    """PointCloudViewer に混ぜる、レイヤーの整合の確認と点数の数え直し"""

    # layer_mask と layer_ids が合っていると確かめたときの状態: (mask の弱い参照, ids の弱い参照, ノイズ扱いの番号, True の数)
    _layer_sync_key = None
    # layer_ids の中身が変わるたびに進む番号（点数の覚えが古くないかを見る）
    _layer_version = 0
    # レイヤーごとの点数の覚え: (ids の弱い参照, 版, レイヤーの番号の並び, {番号: 点数})
    _layer_counts_cache = None
    # スキャンごとの点数の覚え: (scan_ids の弱い参照, {番号: 点数})
    _scan_counts_cache = None

    def _noise_layer_ids(self) -> list:
        """ノイズ扱いのレイヤーの番号（ノイズレイヤー 0 は必ず入る）"""
        noise_ids = [lid for lid in self.layers if self.is_noise_layer(lid)]
        if self.LAYER_NOISE not in noise_ids:
            noise_ids.append(self.LAYER_NOISE)
        return noise_ids

    def _is_noise_point(self, noise_ids) -> np.ndarray:
        """点ごとの「ノイズ扱いのレイヤーに入っているか」。np.isin(self.layer_ids, noise_ids) と同じ結果"""
        ids = self.layer_ids
        wanted = list(dict.fromkeys(int(a) for a in noise_ids))
        if not wanted or len(wanted) > NOISE_COMPARE_MAX:
            return np.isin(ids, wanted)
        out = ids == wanted[0]
        for a in wanted[1:]:
            out |= ids == a
        return out

    # ---------------------------------------------------------------- 整合の確認
    def _layers_known_synced(self) -> bool:
        """layer_mask と layer_ids が、合っていると確かめたときのままか（全点を比べずに分かる範囲で）"""
        key = self._layer_sync_key
        mask, ids = self.layer_mask, self.layer_ids
        if key is None or mask is None or ids is None or self.points_raw is None:
            return False
        if key[0]() is not mask or key[1]() is not ids:
            return False                                 # 配列ごと差し替えられた
        if len(ids) != len(self.points_raw) or len(mask) != len(ids):
            return False
        if key[2] != tuple(sorted(self._noise_layer_ids())):
            return False                                 # ノイズ扱いのレイヤーが変わった
        return int(np.count_nonzero(mask)) == key[3]     # layer_mask が、その場で書き換えられていないか（0.001 秒）

    def _forget_layer_sync(self) -> None:
        """覚えを捨てる。layer_ids の書き換えが途中で失敗したときなどに呼ぶ（次の _ensure_layer_sync が、全点を比べる）"""
        self._layer_sync_key = None
        self._layer_version += 1

    def _note_layers_synced(self) -> None:
        """layer_mask と layer_ids が合っている状態になったときに呼ぶ（確かめた・layer_ids から作った）"""
        self._layer_version += 1                         # layer_ids が変わったかもしれない（点数の覚えを使わせない）
        mask, ids = self.layer_mask, self.layer_ids
        if mask is None or ids is None:
            self._layer_sync_key = None
            return
        self._layer_sync_key = (
            weakref.ref(mask), weakref.ref(ids), tuple(sorted(self._noise_layer_ids())), int(np.count_nonzero(mask)),
        )

    # ---------------------------------------------------------------- 点数
    def _count_points_per_layer(self) -> dict:
        """レイヤー（self.layers にあるもの）ごとの点数。layer_ids が合っていること（_ensure_layer_sync のあと）"""
        ids = self.layer_ids
        layer_ids = tuple(self.layers)
        cache = self._layer_counts_cache
        if cache is not None and ids is not None and cache[0]() is ids and cache[1] == self._layer_version and cache[2] == layer_ids:
            return dict(cache[3])
        if len(layer_ids) <= COUNT_COMPARE_MAX:
            counts = {lid: int(np.count_nonzero(ids == lid)) for lid in layer_ids}
        else:
            unique_ids, unique_counts = np.unique(ids, return_counts=True)
            found = {int(k): int(c) for k, c in zip(unique_ids, unique_counts)}
            counts = {lid: found.get(lid, 0) for lid in layer_ids}
        self._layer_counts_cache = (weakref.ref(ids), self._layer_version, layer_ids, counts)
        return dict(counts)

    def _count_points_per_scan(self) -> dict:
        """スキャン（器械点）ごとの点数。scan_ids は読み込みのあと変わらないので、同じ配列なら数え直さない"""
        scan_ids = self.scan_ids
        cache = self._scan_counts_cache
        if cache is not None and scan_ids is not None and cache[0]() is scan_ids:
            return dict(cache[1])
        unique_ids, unique_counts = np.unique(scan_ids, return_counts=True)
        counts = {int(k): int(c) for k, c in zip(unique_ids, unique_counts)}
        self._scan_counts_cache = (weakref.ref(scan_ids), counts)
        return dict(counts)
