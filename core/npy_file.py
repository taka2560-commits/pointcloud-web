# -*- coding: utf-8 -*-
"""
大きな .npy ファイルを、メモリに割り付けず（mmap を使わず）に読み書きする部品

なぜ要るか（2026-11-13、利用者の実データ 5.4 億点での確認から）
    読み直しキャッシュと LOD データ（1 ファイル 6〜9GB）を np.lib.format.open_memmap / np.load(mmap_mode="r") で
    読み書きしていた。Windows では、mmap で触れたページはそのプロセスの使用メモリに数えられ、書いたページは
    ディスクへ書き終わるまで追い出せない。作業用データを作る間にメモリが 95〜97% まで埋まり、ディスクの応答が
    2.4 秒まで落ちた（LOD の並べ替えは 9GB のファイルへ飛び飛びに書くので特にひどい）。
    ここでは普通のファイル読み書きだけを使う。書くときは先頭から順に書き、一定量ごとにディスクへ流すので、
    使うメモリはファイルの大きさに依らない。

ファイルの形式は .npy のまま変えない（np.load で読めるし、以前に mmap で作ったファイルもそのまま読める）。
開いたままのハンドルを持たないので、Windows でもファイルやフォルダをいつでも消せる。
"""

import os
from typing import Sequence, Tuple

import numpy as np

FLUSH_BYTES = 256 * 1024 * 1024  # これだけ書くたびにディスクへ流す（書きかけのデータをメモリにためない）


def _read_header(f) -> Tuple[Tuple[int, ...], np.dtype, int]:
    version = np.lib.format.read_magic(f)
    if version == (1, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
    elif version == (2, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
    else:
        raise ValueError(f".npy の版 {version} には対応していません")
    if fortran:
        raise ValueError("列優先（Fortran 順）の .npy には対応していません")
    if dtype.hasobject:
        raise ValueError("オブジェクト型の .npy には対応していません")
    return tuple(int(v) for v in shape), dtype, f.tell()


def _read_exact(f, view: memoryview) -> None:
    """view がちょうど埋まるまで読む（1 回の read が途中で返ってきても続ける）"""
    pos = 0
    total = len(view)
    while pos < total:
        n = f.readinto(view[pos:])
        if not n:
            raise OSError(f"ファイルが途中で終わっています（{pos:,} / {total:,} バイト）")
        pos += n


class NpyFileArray:
    """
    .npy を読み取り専用で、範囲を指定したときだけファイルから読む。
    arr[a:b]（連続した範囲）と read_ranges(...) だけを持つ。戻り値は普通の ndarray（コピー）。
    """

    def __init__(self, path: str):
        self.path = path
        with open(path, "rb") as f:
            self.shape, self.dtype, self._offset = _read_header(f)
        if len(self.shape) < 1:
            raise ValueError(f"0 次元の .npy は扱えません: {path}")
        self.ndim = len(self.shape)
        self._row_shape = self.shape[1:]
        self._row_bytes = int(np.prod(self._row_shape, dtype=np.int64)) * self.dtype.itemsize if self._row_shape else self.dtype.itemsize
        need = self._offset + self.shape[0] * self._row_bytes
        size = os.path.getsize(path)
        if size < need:
            raise ValueError(f".npy が宣言より短いです（{size:,} < {need:,} バイト）: {path}")

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, key) -> np.ndarray:
        if not isinstance(key, slice):
            raise TypeError("NpyFileArray は arr[a:b] の形だけを読めます")
        a, b, step = key.indices(self.shape[0])
        if step != 1:
            raise ValueError("NpyFileArray は連続した範囲だけを読めます")
        n = max(0, b - a)
        out = np.empty((n,) + self._row_shape, dtype=self.dtype)
        if n:
            with open(self.path, "rb", buffering=0) as f:
                f.seek(self._offset + a * self._row_bytes)
                _read_exact(f, memoryview(out).cast("B"))
        return out

    def read_ranges(self, starts: Sequence[int], ends: Sequence[int], out: np.ndarray) -> np.ndarray:
        """
        [starts[i], ends[i]) の行を順に out へ詰めて読む（ファイルを開くのは 1 回）。
        out は合計の行数ぶんの大きさで、dtype と 1 行の形がこのファイルと同じであること。
        """
        if out.dtype != self.dtype or out.shape[1:] != self._row_shape:
            raise ValueError("read_ranges: out の型か形がファイルと合いません")
        view = memoryview(out).cast("B")
        pos = 0
        with open(self.path, "rb", buffering=0) as f:
            for a, b in zip(starts, ends):
                a, b = int(a), int(b)
                n = b - a
                if n <= 0:
                    continue
                nbytes = n * self._row_bytes
                f.seek(self._offset + a * self._row_bytes)
                _read_exact(f, view[pos : pos + nbytes])
                pos += nbytes
        if pos != len(view):
            raise ValueError("read_ranges: out の大きさと読んだ行数が合いません")
        return out


class NpySequentialWriter:
    """
    最終的な形 shape の .npy を、先頭から順に書く。close() で、書いていない残りの行は 0 で埋める
    （np.lib.format.open_memmap で確保して前から書いたときと同じファイルになる）。
    """

    def __init__(self, path: str, dtype, shape: Tuple[int, ...]):
        self.path = path
        self.dtype = np.dtype(dtype)
        self.shape = tuple(int(v) for v in shape)
        self._row_shape = self.shape[1:]
        self._row_bytes = int(np.prod(self._row_shape, dtype=np.int64)) * self.dtype.itemsize if self._row_shape else self.dtype.itemsize
        self.n_written = 0
        self._since_flush = 0
        self._f = open(path, "wb")
        np.lib.format.write_array_header_1_0(
            self._f, {"descr": np.lib.format.dtype_to_descr(self.dtype), "fortran_order": False, "shape": self.shape}
        )
        self._offset = self._f.tell()

    def append(self, arr: np.ndarray) -> None:
        a = np.ascontiguousarray(arr, dtype=self.dtype)
        if a.shape[1:] != self._row_shape:
            raise ValueError("NpySequentialWriter: 1 行の形が合いません")
        n = len(a)
        if self.n_written + n > self.shape[0]:
            raise ValueError(f"NpySequentialWriter: 宣言した行数 {self.shape[0]:,} を超えて書こうとしました")
        if n == 0:
            return
        self._f.write(memoryview(a).cast("B"))
        self.n_written += n
        self._since_flush += n * self._row_bytes
        if self._since_flush >= FLUSH_BYTES:
            self._sync()

    def _sync(self) -> None:
        self._f.flush()
        try:
            os.fsync(self._f.fileno())
        except OSError:
            pass
        self._since_flush = 0

    def close(self, pad: bool = True) -> None:
        """閉じる。pad=True なら、宣言した行数に満たない残りを 0 で埋めて、ファイルを宣言どおりの大きさにする"""
        if self._f is None:
            return
        try:
            if pad:
                self._f.flush()
                self._f.truncate(self._offset + self.shape[0] * self._row_bytes)
            self._sync()
        finally:
            self._f.close()
            self._f = None
