# -*- coding: utf-8 -*-
"""
全点表示で消した点の印（元ファイルでの点の番号で覚える）

1 億点超の E57 は「k 点に 1 点」で読み込まれ、ブラシや自動フィルタが扱うのは読み込んだ点だけ。
断面の「元ファイルから全点で読み直す」で見える残りの点（読み込みで飛ばされた点）は、以前は表示専用で消せなかった。
ここは、その点を利用者が消したことを「スキャンの番号 → 元ファイルでの点の番号（行）」の集合として覚える。
- 覚えるのは番号だけなので、断面の箱を動かして重ね描きを作り直しても、消した点は消えたままになる。
- 「全点で保存」（gui/worker_thread.py の FullResolutionSaveWorker）が、元ファイルを流し読みするときに、この番号の点を
  ノイズとして扱う（近くの読み込み済みの点から引き継いだ判定より優先する）。
- 読み込んだ点（行の番号が間引きの間隔の倍数）には印を付けない。読み込んだ点は、今までどおりレイヤーで扱う
  （同じ点を 2 つの仕組みで持つと、レイヤーで戻したのに印が残って保存で消える、という食い違いが起きる）。

行の番号: スキャンの points（CompressedVector）の中での 0 始まりの順番。読み込み・読み直し・保存の 3 つが同じ数え方をする
（読み込みは 0, k, 2k, … を取る。core/e57_io.py の iter_e57_scans）。
GUI にも E57 にも依存しない。配列は書き換えず、変えるたびに新しい配列に差し替える（写しを安く取れるようにするため）。
"""

from typing import Callable, Dict, List, Optional

import numpy as np

Snapshot = Dict[int, np.ndarray]


def _as_rows(rows) -> np.ndarray:
    arr = np.unique(np.asarray(rows, dtype=np.int64).ravel())
    return arr[arr >= 0]


def _frozen(arr: np.ndarray) -> np.ndarray:
    arr = np.ascontiguousarray(arr, dtype=np.int64)
    arr.setflags(write=False)
    return arr


def loaded_row_mask(rows: np.ndarray, step: int) -> np.ndarray:
    """読み込んだ点（間引きの間隔 step の倍数の行）かどうか。step が 1 以下なら、全部が読み込んだ点"""
    rows = np.asarray(rows, dtype=np.int64)
    if step is None or int(step) <= 1:
        return np.ones(len(rows), dtype=bool)
    return (rows % int(step)) == 0


class FullResMarks:
    """全点表示で消した点の印の集まり"""

    def __init__(self):
        self._rows: Dict[int, np.ndarray] = {}   # スキャンの番号 → 行の番号（昇順・重複なし・書き換え不可）
        self._listeners: List[Callable[[], None]] = []
        self.version = 0                          # 変わるたびに増える（画面側が、作り直しが要るかを見る）

    # ------------------------------------------------------------------ 読む
    def count(self) -> int:
        return int(sum(len(v) for v in self._rows.values()))

    def is_empty(self) -> bool:
        return self.count() == 0

    def scans(self) -> List[int]:
        return sorted(k for k, v in self._rows.items() if len(v))

    def rows_of(self, scan: int) -> np.ndarray:
        return self._rows.get(int(scan), np.empty(0, dtype=np.int64))

    def contains(self, scan_ids: np.ndarray, rows: np.ndarray) -> np.ndarray:
        """(スキャンの番号, 行の番号) の並びのうち、印の付いている物（真偽の配列）"""
        scan_ids = np.asarray(scan_ids)
        rows = np.asarray(rows, dtype=np.int64)
        out = np.zeros(len(rows), dtype=bool)
        if len(rows) == 0:
            return out
        for scan, marked in self._rows.items():
            if len(marked) == 0:
                continue
            sel = np.flatnonzero(scan_ids == scan)
            if len(sel) == 0:
                continue
            r = rows[sel]
            pos = np.searchsorted(marked, r)
            pos[pos >= len(marked)] = len(marked) - 1
            out[sel] = marked[pos] == r
        return out

    def mask_for_chunk(self, scan: int, row_start: int, n: int) -> Optional[np.ndarray]:
        """
        行 row_start 〜 row_start + n - 1 のうち、印の付いている物（真偽の配列）。1 つも無ければ None。
        保存が、元ファイルを少しずつ流し読みしながら使う
        """
        marked = self._rows.get(int(scan))
        if marked is None or len(marked) == 0 or n <= 0:
            return None
        a = int(np.searchsorted(marked, int(row_start), side="left"))
        b = int(np.searchsorted(marked, int(row_start) + int(n), side="left"))
        if b <= a:
            return None
        mask = np.zeros(int(n), dtype=bool)
        mask[marked[a:b] - int(row_start)] = True
        return mask

    def mask_for_rows(self, scan: int, rows: np.ndarray) -> Optional[np.ndarray]:
        """
        昇順に並んだ行の番号 rows のうち、印の付いている物（真偽の配列）。1 つも無ければ None。
        保存が使う。流し読みは「数でない座標の点」を除いて返すので、行の番号は続き番号とは限らない
        """
        marked = self._rows.get(int(scan))
        rows = np.asarray(rows, dtype=np.int64)
        if marked is None or len(marked) == 0 or len(rows) == 0:
            return None
        a = int(np.searchsorted(marked, int(rows[0]), side="left"))
        b = int(np.searchsorted(marked, int(rows[-1]), side="right"))
        if b <= a:
            return None
        want = marked[a:b]
        pos = np.searchsorted(rows, want)
        pos[pos >= len(rows)] = len(rows) - 1
        hit = pos[rows[pos] == want]
        if len(hit) == 0:
            return None
        mask = np.zeros(len(rows), dtype=bool)
        mask[hit] = True
        return mask

    def copy(self) -> "FullResMarks":
        """いまの印の写し（あとから元を変えても、写しは変わらない。保存のスレッドへ渡すのに使う）"""
        other = FullResMarks()
        other._rows = dict(self._rows)
        return other

    # ------------------------------------------------------------------ 変える
    def _changed(self) -> None:
        self.version += 1
        for fn in list(self._listeners):
            fn()

    def add_listener(self, fn: Callable[[], None]) -> None:
        if fn not in self._listeners:
            self._listeners.append(fn)

    def add(self, scan_ids: np.ndarray, rows: np.ndarray) -> int:
        """印を付ける。新しく付いた数を返す（すでに付いている物・番号が負の物は数えない）"""
        scan_ids = np.asarray(scan_ids)
        rows = np.asarray(rows, dtype=np.int64)
        added = 0
        for scan in np.unique(scan_ids):
            new = _as_rows(rows[scan_ids == scan])
            if len(new) == 0:
                continue
            old = self.rows_of(int(scan))
            merged = np.union1d(old, new)
            added += len(merged) - len(old)
            self._rows[int(scan)] = _frozen(merged)
        if added:
            self._changed()
        return int(added)

    def toggle(self, diff: Snapshot) -> None:
        """diff の行について、印の有る無しを入れ替える（元に戻す・やり直しは、同じ diff をもう一度当てる）"""
        if not diff:
            return
        for scan, rows in diff.items():
            flipped = np.setxor1d(self.rows_of(int(scan)), _as_rows(rows))
            if len(flipped):
                self._rows[int(scan)] = _frozen(flipped)
            else:
                self._rows.pop(int(scan), None)
        self._changed()

    def clear(self) -> int:
        """印を全部外す。外した数を返す"""
        n = self.count()
        self._rows = {}
        if n:
            self._changed()
        return n

    # ------------------------------------------------------------------ 履歴（元に戻す）のための写しと差分
    def snapshot(self) -> Snapshot:
        """いまの印の写し（配列は書き換えないので、辞書の浅い写しで足りる）"""
        return dict(self._rows)

    @staticmethod
    def diff(old: Optional[Snapshot], new: Optional[Snapshot]) -> Optional[Snapshot]:
        """2 つの写しのあいだで、印の有る無しが変わった行。変わっていなければ None"""
        old = old or {}
        new = new or {}
        out: Snapshot = {}
        empty = np.empty(0, dtype=np.int64)
        for scan in set(old) | set(new):
            a, b = old.get(scan, empty), new.get(scan, empty)
            if a is b:
                continue
            changed = np.setxor1d(a, b)
            if len(changed):
                out[int(scan)] = _frozen(changed)
        return out or None

    @staticmethod
    def diff_nbytes(diff: Optional[Snapshot]) -> int:
        return int(sum(v.nbytes for v in diff.values())) if diff else 0

    @staticmethod
    def diff_count(diff: Optional[Snapshot]) -> int:
        return int(sum(len(v) for v in diff.values())) if diff else 0
