"""
レイヤー割り当ての差分 Undo / Redo 履歴（メモリ節約版）

背景:
  従来は Undo 1 ステップごとに layer_ids (int32) と layer_mask (bool) を丸ごとコピーしていたため、
  6,000 万点では 1 ステップ約 300MB を消費し、履歴数を 5 に絞らざるを得なかった。
  本モジュールは「直前のプッシュ時点の layer_ids の写し（shadow）1 本」だけを保持し、
  各ステップは「変化した点のインデックス＋変化前の layer_id」だけを記録する。
  メモリは (変化点数 × 最大 8 バイト) + レイヤー定義辞書の小さな複製で済む。

使い方（MainWindow 側）:
  history = LayerUndoHistory()
  # 変更前に必ず呼ぶ（従来の _push_layer_undo_state と同じタイミング）
  history.push(LayerState(viewer.layer_ids, viewer.layers), max_steps)
  ... viewer.layer_ids をインプレースで変更 ...
  # 元に戻す
  new_state = history.undo(LayerState(viewer.layer_ids, viewer.layers))  # layer_ids はインプレースで戻る
  viewer.layers = new_state.layers

設計上の要点:
  - push() 時点では「これから何が変わるか」が分からないため、直前のプッシュからの差分は
    次の push() / undo() のときに shadow と現在値を比較して確定（commit）する。
  - layer_mask は layer_ids とレイヤー定義 (is_noise) から再計算できるため保存しない。
  - undo_stack / redo_stack は list 互換（len / bool / clear が使える）。外部から clear() されたときも
    shadow を安全にリセットするため、list のサブクラスで clear() をフックしている。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class LayerState:
    """履歴に渡す現在のレイヤー状態（layer_ids は参照のまま渡す。コピーは履歴側で必要分だけ行う）"""
    layer_ids: Optional[np.ndarray]
    layers: Dict[int, Any]


@dataclass
class LayerDiffEntry:
    """1 ステップ分の差分: idx の点の layer_id を values に戻し、レイヤー定義を layers に戻す"""
    idx: np.ndarray            # 変化した点のインデックス (int32 / int64)
    values: np.ndarray         # その点の「戻し先」の layer_id (int32)
    layers: Dict[int, Any]     # 戻し先のレイヤー定義（LayerInfo の複製）

    @property
    def nbytes(self) -> int:
        return int(self.idx.nbytes + self.values.nbytes)


def _copy_layers(layers: Dict[int, Any]) -> Dict[int, Any]:
    """LayerInfo（dataclass）辞書の浅い複製。LayerInfo 自体は不変扱いなので値ごと複製する"""
    import copy

    return {lid: copy.copy(info) for lid, info in (layers or {}).items()}


def _layers_equal(a: Dict[int, Any], b: Dict[int, Any]) -> bool:
    if a is b:
        return True
    if set(a.keys()) != set(b.keys()):
        return False
    for k in a:
        if a[k] != b[k]:  # LayerInfo は dataclass なのでフィールド単位で比較される
            return False
    return True


class _HistoryList(list):
    """外部から clear() されたときに履歴オブジェクトへ通知する list"""

    def __init__(self, owner: "LayerUndoHistory"):
        super().__init__()
        self._owner = owner

    def clear(self) -> None:  # type: ignore[override]
        super().clear()
        self._owner._on_external_clear(self)


class LayerUndoHistory:
    """差分ベースのレイヤー Undo / Redo 履歴"""

    def __init__(self):
        self.undo_stack: List[LayerDiffEntry] = _HistoryList(self)
        self.redo_stack: List[LayerDiffEntry] = _HistoryList(self)
        self._shadow_ids: Optional[np.ndarray] = None
        self._shadow_layers: Dict[int, Any] = {}
        # push() 後に（まだ確定していない）変更が起こり得る状態か
        self._pending_expected: bool = False

    # ------------------------------------------------------------------ 内部
    def _on_external_clear(self, which: list) -> None:
        if which is self.undo_stack:
            self._shadow_ids = None
            self._shadow_layers = {}
            self._pending_expected = False

    def _shadow_valid(self, state: LayerState) -> bool:
        return (
            self._shadow_ids is not None
            and state.layer_ids is not None
            and len(self._shadow_ids) == len(state.layer_ids)
        )

    def _take_shadow(self, state: LayerState) -> None:
        self._shadow_ids = state.layer_ids.copy() if state.layer_ids is not None else None
        self._shadow_layers = _copy_layers(state.layers)

    @staticmethod
    def _index_dtype(n: int):
        return np.int32 if n < 2**31 - 1 else np.int64

    def _commit_pending(self, state: LayerState) -> bool:
        """
        直前の push() 以降に起きた変更を shadow と比較して差分エントリとして undo_stack に確定する。
        戻り値: 差分を確定した（＝変更があった）か
        """
        if not self._shadow_valid(state):
            self._pending_expected = False
            return False
        changed = np.nonzero(self._shadow_ids != state.layer_ids)[0]
        layers_changed = not _layers_equal(self._shadow_layers, state.layers)
        self._pending_expected = False
        if changed.size == 0 and not layers_changed:
            return False
        idx = changed.astype(self._index_dtype(len(state.layer_ids)), copy=False)
        entry = LayerDiffEntry(
            idx=idx,
            values=self._shadow_ids[changed].astype(np.int32, copy=True),
            layers=self._shadow_layers,  # shadow は作り直すので所有権を移してよい
        )
        self.undo_stack.append(entry)
        return True

    # ------------------------------------------------------------------ 公開 API
    def reset(self) -> None:
        """履歴を全消去（新しい点群を読み込んだときなど）"""
        list.clear(self.undo_stack)
        list.clear(self.redo_stack)
        self._shadow_ids = None
        self._shadow_layers = {}
        self._pending_expected = False

    def push(self, state: LayerState, max_steps: int = 50) -> None:
        """
        レイヤーを変更する直前に呼ぶ。直前のプッシュからの差分を確定し、現在の状態を新しい基準 (shadow) にする。
        max_steps <= 0 は無制限。
        """
        if state.layer_ids is None:
            return
        self._commit_pending(state)
        self._take_shadow(state)
        self._pending_expected = True
        list.clear(self.redo_stack)
        if max_steps > 0:
            while len(self.undo_stack) > max_steps:
                self.undo_stack.pop(0)

    def can_undo(self) -> bool:
        return bool(self.undo_stack) or (self._pending_expected and self._shadow_ids is not None)

    def can_redo(self) -> bool:
        return bool(self.redo_stack)

    def undo(self, state: LayerState) -> Optional[LayerState]:
        """
        1 ステップ元に戻す。state.layer_ids はインプレースで書き換わる。
        戻り値: 戻した後の状態（layers は新しい辞書）。戻すものが無ければ None。
        """
        if state.layer_ids is None:
            return None
        self._commit_pending(state)
        if not self.undo_stack:
            return None
        entry = self.undo_stack.pop()
        # やり直し用: 現在値を記録してから戻す
        self.redo_stack.append(
            LayerDiffEntry(
                idx=entry.idx,
                values=state.layer_ids[entry.idx].astype(np.int32, copy=True),
                layers=_copy_layers(state.layers),
            )
        )
        state.layer_ids[entry.idx] = entry.values
        new_state = LayerState(state.layer_ids, _copy_layers(entry.layers))
        self._take_shadow(new_state)
        return new_state

    def redo(self, state: LayerState) -> Optional[LayerState]:
        """取り消した 1 ステップをやり直す。state.layer_ids はインプレースで書き換わる。"""
        if state.layer_ids is None or not self.redo_stack:
            return None
        entry = self.redo_stack.pop()
        self.undo_stack.append(
            LayerDiffEntry(
                idx=entry.idx,
                values=state.layer_ids[entry.idx].astype(np.int32, copy=True),
                layers=_copy_layers(state.layers),
            )
        )
        state.layer_ids[entry.idx] = entry.values
        new_state = LayerState(state.layer_ids, _copy_layers(entry.layers))
        self._take_shadow(new_state)
        self._pending_expected = False
        return new_state

    def memory_bytes(self) -> int:
        """履歴が保持している配列の概算バイト数（shadow を含む）"""
        total = sum(e.nbytes for e in self.undo_stack) + sum(e.nbytes for e in self.redo_stack)
        if self._shadow_ids is not None:
            total += int(self._shadow_ids.nbytes)
        return total
