"""
大きな .npy を mmap せずに読み書きする部品（core/npy_file.py）のテスト

2026-11-13 に、読み直しキャッシュと LOD データの読み書きを mmap から普通のファイル読み書きへ替えた
（利用者の実データ 5.4 億点で、作業用データを作る間にメモリが 95〜97% まで埋まったため）。
- 順に書いたファイルが、以前の方式（np.lib.format.open_memmap で確保して前から書く）と 1 バイトも違わないこと
  （作成済みの作業用データをそのまま使えるように、形式を変えていない）
- 範囲を指定した読み出しが np.load と同じ値になること
- 開いたままのハンドルを持たないこと（Windows でも、使っている最中のファイルを消せる）
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from core.npy_file import NpyFileArray, NpySequentialWriter


class TestNpyFile(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.rng = np.random.default_rng(0)

    def tearDown(self):
        self.tmp.cleanup()

    def _p(self, name):
        return os.path.join(self.tmp.name, name)

    def test_sequential_writer_equals_open_memmap(self):
        for dtype, row in ((np.int32, (3,)), (np.float32, (3,)), (np.uint8, (3,)), (np.float16, ())):
            n_alloc, n_written = 1000, 870  # 宣言より少なく書く（NaN の点を除いたときなど）。残りは 0 で埋まる
            data = (self.rng.random((n_written,) + row) * 100).astype(dtype)
            ref = self._p("ref.npy")
            m = np.lib.format.open_memmap(ref, mode="w+", dtype=dtype, shape=(n_alloc,) + row)
            m[:n_written] = data
            m.flush()
            del m
            new = self._p("new.npy")
            w = NpySequentialWriter(new, dtype, (n_alloc,) + row)
            for a in range(0, n_written, 333):
                w.append(data[a : a + 333])
            self.assertEqual(w.n_written, n_written)
            w.close()
            with open(ref, "rb") as f1, open(new, "rb") as f2:
                self.assertEqual(f1.read(), f2.read(), f"{np.dtype(dtype)} {row}: 以前の方式と違うファイルになりました")
            np.testing.assert_array_equal(np.load(new)[:n_written], data)
            self.assertTrue(np.all(np.load(new)[n_written:] == 0))

    def test_writer_rejects_overflow_and_wrong_shape(self):
        w = NpySequentialWriter(self._p("a.npy"), np.float32, (10, 3))
        w.append(np.zeros((8, 3), dtype=np.float32))
        with self.assertRaises(ValueError):
            w.append(np.zeros((3, 3), dtype=np.float32))
        with self.assertRaises(ValueError):
            w.append(np.zeros((1, 2), dtype=np.float32))
        w.close(pad=False)
        os.remove(self._p("a.npy"))  # 閉じた後は消せる

    def test_file_array_reads_like_np_load(self):
        data = self.rng.integers(-1000, 1000, size=(5000, 3)).astype(np.int32)
        path = self._p("x.npy")
        np.save(path, data)
        arr = NpyFileArray(path)
        self.assertEqual((arr.shape, arr.dtype, arr.ndim, len(arr)), ((5000, 3), np.dtype(np.int32), 2, 5000))
        for a, b in ((0, 5000), (0, 1), (123, 4567), (4999, 5000), (10, 10), (4000, 9999)):
            np.testing.assert_array_equal(arr[a:b], data[a:b])
        np.testing.assert_array_equal(arr[:], data)
        with self.assertRaises(ValueError):
            arr[0:10:2]
        with self.assertRaises(TypeError):
            arr[5]

        flat = self.rng.random(3000).astype(np.float16)
        path1 = self._p("i.npy")
        np.save(path1, flat)
        arr1 = NpyFileArray(path1)
        np.testing.assert_array_equal(arr1[100:2000], flat[100:2000])

        starts, ends = [10, 500, 4000, 4990], [20, 900, 4005, 5000]
        total = sum(e - s for s, e in zip(starts, ends))
        out = np.empty((total, 3), dtype=np.int32)
        arr.read_ranges(starts, ends, out)
        np.testing.assert_array_equal(out, np.concatenate([data[s:e] for s, e in zip(starts, ends)]))
        with self.assertRaises(ValueError):
            arr.read_ranges(starts, ends, np.empty((total + 1, 3), dtype=np.int32))
        with self.assertRaises(ValueError):
            arr.read_ranges(starts, ends, np.empty((total, 3), dtype=np.float32))

    def test_no_open_handles_and_truncated_file(self):
        data = np.arange(3000, dtype=np.float32).reshape(1000, 3)
        path = self._p("h.npy")
        np.save(path, data)
        arr = NpyFileArray(path)
        _ = arr[0:500]
        os.remove(path)  # 読んだ直後でも、オブジェクトが生きていても消せる（mmap だと Windows では消せない）
        self.assertFalse(os.path.exists(path))
        self.assertIsNotNone(arr)

        np.save(path, data)
        with open(path, "r+b") as f:
            f.truncate(os.path.getsize(path) - 100)
        with self.assertRaises(ValueError):
            NpyFileArray(path)


if __name__ == "__main__":
    unittest.main()
