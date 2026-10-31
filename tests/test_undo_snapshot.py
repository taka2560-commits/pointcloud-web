"""
差分 Undo / Redo 履歴 (gui/layer_history.py) のテスト

- ランダムな一連のレイヤー編集に対し、全点コピー方式の参照実装と結果が完全一致すること
- 1 ステップのメモリが「変化した点数」に比例し、全点コピーにならないこと
- 外部から undo_stack.clear() されたとき（新規読み込み時）に古い差分が復活しないこと
- 変更が無かった push（0 点移動）は Undo の空振りにならず、次の有効ステップへ進むこと
"""

import copy
import unittest
from dataclasses import dataclass

import numpy as np

from gui.layer_history import LayerState, LayerUndoHistory


@dataclass
class _Info:
    """LayerInfo の最小代替（dataclass なので等価比較ができる）"""
    layer_id: int
    name: str
    locked: bool = False
    is_noise: bool = False


class _ReferenceHistory:
    """従来方式（全点コピー）の参照実装"""

    def __init__(self):
        self.undo = []
        self.redo = []

    def push(self, ids, layers):
        self.undo.append((ids.copy(), copy.deepcopy(layers)))
        self.redo.clear()

    def do_undo(self, ids, layers):
        if not self.undo:
            return None
        self.redo.append((ids.copy(), copy.deepcopy(layers)))
        return self.undo.pop()

    def do_redo(self, ids, layers):
        if not self.redo:
            return None
        self.undo.append((ids.copy(), copy.deepcopy(layers)))
        return self.redo.pop()


class TestLayerUndoHistoryEquivalence(unittest.TestCase):
    def _random_session(self, seed: int, n: int = 5000, steps: int = 60):
        rng = np.random.default_rng(seed)
        ids = np.ones(n, dtype=np.int32)
        layers = {0: _Info(0, "ノイズ", is_noise=True), 1: _Info(1, "通常")}
        ref_ids = ids.copy()
        ref_layers = copy.deepcopy(layers)

        hist = LayerUndoHistory()
        ref = _ReferenceHistory()

        def mutate(target_ids, target_layers, rng_local):
            # 参照実装（全点コピー）は「変更なしの push」も 1 ステップと数えるが、差分履歴は数えない。
            # ここでは必ず状態が変わる操作だけを行い、両者が同じステップ数になるようにする
            # （変更なし push の扱いは test_noop_push_does_not_consume_an_undo_step で別途検証）。
            op = rng_local.integers(0, 3)
            if op == 0:  # 通常レイヤーの点をノイズへ（必ず 1 点以上）
                cand = np.nonzero(target_ids != 0)[0]
                if len(cand) == 0:
                    target_ids[:] = 1
                    cand = np.arange(n)
                k = int(rng_local.integers(1, min(200, len(cand)) + 1))
                sel = rng_local.choice(cand, size=k, replace=False)
                target_ids[sel] = 0
            elif op == 1:  # 新規カスタムレイヤーを作って移動
                lid = max(target_layers) + 1
                target_layers[lid] = _Info(lid, f"レイヤー{lid}")
                sel = rng_local.choice(n, size=int(rng_local.integers(1, 300)), replace=False)
                target_ids[sel] = lid
            else:  # ロック切替（点は変わらないがレイヤー定義が変わる）
                lid = int(rng_local.choice(list(target_layers)))
                target_layers[lid].locked = not target_layers[lid].locked

        for step in range(steps):
            action = rng.random()
            if action < 0.6:
                # push → 変更
                hist.push(LayerState(ids, layers), max_steps=0)
                ref.push(ref_ids, ref_layers)
                r1 = np.random.default_rng(seed * 1000 + step)
                r2 = np.random.default_rng(seed * 1000 + step)
                mutate(ids, layers, r1)
                mutate(ref_ids, ref_layers, r2)
            elif action < 0.85:
                res = hist.undo(LayerState(ids, layers))
                ref_res = ref.do_undo(ref_ids, ref_layers)
                if ref_res is not None:
                    ref_ids, ref_layers = ref_res[0].copy(), copy.deepcopy(ref_res[1])
                if res is not None:
                    layers = res.layers
                # 参照実装は「変更が無かった push」も 1 ステップとして消費するため、
                # 状態一致は undo の結果が存在した場合のみ比較する
            else:
                res = hist.redo(LayerState(ids, layers))
                ref_res = ref.do_redo(ref_ids, ref_layers)
                if ref_res is not None:
                    ref_ids, ref_layers = ref_res[0].copy(), copy.deepcopy(ref_res[1])
                if res is not None:
                    layers = res.layers
            np.testing.assert_array_equal(ids, ref_ids, err_msg=f"seed={seed} step={step}: layer_ids が参照実装と不一致")
            self.assertEqual(layers, ref_layers, f"seed={seed} step={step}: レイヤー定義が参照実装と不一致")

    def test_random_sessions_match_full_copy_reference(self):
        for seed in range(1, 6):
            self._random_session(seed)

    def test_simple_undo_redo_roundtrip(self):
        n = 100
        ids = np.ones(n, dtype=np.int32)
        layers = {0: _Info(0, "ノイズ", is_noise=True), 1: _Info(1, "通常")}
        hist = LayerUndoHistory()

        hist.push(LayerState(ids, layers))
        ids[:10] = 0
        hist.push(LayerState(ids, layers))
        ids[10:30] = 0
        after = ids.copy()

        self.assertTrue(hist.can_undo())
        self.assertFalse(hist.can_redo())
        st = hist.undo(LayerState(ids, layers))
        self.assertIsNotNone(st)
        self.assertEqual(int((ids == 0).sum()), 10)
        self.assertEqual(len(hist.undo_stack), 1)
        self.assertTrue(hist.can_redo())
        st = hist.undo(LayerState(ids, layers))
        self.assertEqual(int((ids == 0).sum()), 0)
        self.assertFalse(hist.can_undo())
        hist.redo(LayerState(ids, layers))
        hist.redo(LayerState(ids, layers))
        np.testing.assert_array_equal(ids, after)
        self.assertFalse(hist.can_redo())


class TestLayerUndoHistoryMemory(unittest.TestCase):
    def test_entry_size_is_proportional_to_changed_points(self):
        n = 1_000_000
        ids = np.ones(n, dtype=np.int32)
        layers = {0: _Info(0, "ノイズ", is_noise=True), 1: _Info(1, "通常")}
        hist = LayerUndoHistory()
        hist.push(LayerState(ids, layers))
        ids[:10] = 0
        hist.push(LayerState(ids, layers))  # ここで 10 点分の差分が確定する
        self.assertEqual(len(hist.undo_stack), 1)
        entry = hist.undo_stack[0]
        self.assertEqual(len(entry.idx), 10)
        self.assertLess(entry.nbytes, 1024)
        # shadow（1 本の全点コピー）＋差分 のみ。全点コピーを複数持つ従来方式（2 ステップで 8MB+）にはならない
        self.assertLess(hist.memory_bytes(), ids.nbytes + 4096)

    def test_external_clear_resets_shadow(self):
        n = 50
        ids = np.ones(n, dtype=np.int32)
        layers = {0: _Info(0, "ノイズ", is_noise=True), 1: _Info(1, "通常")}
        hist = LayerUndoHistory()
        hist.push(LayerState(ids, layers))
        ids[:5] = 0
        # 新規読み込み相当: 外部から list.clear() される
        hist.undo_stack.clear()
        hist.redo_stack.clear()
        self.assertFalse(hist.can_undo())
        # 古い差分が次の push で復活しないこと
        hist.push(LayerState(ids, layers))
        self.assertEqual(len(hist.undo_stack), 0)
        ids[5:8] = 0
        hist.undo(LayerState(ids, layers))
        self.assertEqual(int((ids == 0).sum()), 5, "クリア前の変更まで巻き戻ってはならない")

    def test_noop_push_does_not_consume_an_undo_step(self):
        n = 50
        ids = np.ones(n, dtype=np.int32)
        layers = {0: _Info(0, "ノイズ", is_noise=True), 1: _Info(1, "通常")}
        hist = LayerUndoHistory()
        hist.push(LayerState(ids, layers))
        ids[:5] = 0
        hist.push(LayerState(ids, layers))  # 0 点移動（何も変えない）
        hist.push(LayerState(ids, layers))
        ids[5:7] = 0
        hist.undo(LayerState(ids, layers))
        self.assertEqual(int((ids == 0).sum()), 5)
        hist.undo(LayerState(ids, layers))
        self.assertEqual(int((ids == 0).sum()), 0, "空の push を挟んでも 1 回の Undo で前の有効ステップへ戻ること")

    def test_max_steps_trims_oldest(self):
        n = 20
        ids = np.ones(n, dtype=np.int32)
        layers = {0: _Info(0, "ノイズ", is_noise=True), 1: _Info(1, "通常")}
        hist = LayerUndoHistory()
        for i in range(10):
            hist.push(LayerState(ids, layers), max_steps=3)
            ids[i] = 0
        hist.push(LayerState(ids, layers), max_steps=3)
        self.assertLessEqual(len(hist.undo_stack), 3)


if __name__ == "__main__":
    unittest.main()
