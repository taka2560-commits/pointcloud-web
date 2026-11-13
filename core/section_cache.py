# -*- coding: utf-8 -*-
"""
断面ボックスの読み直し用キャッシュ（元 E57 を 1 回デコードして、圧縮の無い自前形式でディスクに置く）

libe57 には読み飛ばし（seek）が無く、箱に掛からない部分も毎回デコードする必要がある（5.4 億点で約 100 秒）。
そこで初回の読み直しで全点を 1 回デコードしながら、スキャンごとに次を書き出す。
  <元ファイル>.cache/                （既定は元ファイルの横。cache_dir で別の場所も可）
    header.json                       … 版、元ファイルの (サイズ, 更新時刻ns)、スキャンごとの点数・座標の持ち方・有無
    scanNN_xyz.npy    (n, 3)          … 座標。ScaledInteger の元ファイルは生整数のまま（int32、収まらなければ int64）、
                                        Float の元ファイルは同じ精度の float32 / float64。復元は全点読込と同じ式
                                        （生整数 × scale + offset）なので、デコード結果・読み込み済みの点とビット単位で一致する
    scanNN_rgb.npy    uint8   (n, 3)  … 色（16bit 色は 8bit に落とす。表示専用）
    scanNN_int.npy    float16 (n,)    … 反射強度（表示のストレッチ用。無ければ作らない）
    scanNN_chunks.npy float64 (m, 8)  … チャンク表 [先頭位置, 点数, xmin, xmax, ymin, ymax, zmin, zmax]（ローカル座標）
2 回目からは、チャンク表の範囲を表示座標へ変換して箱に掛かるチャンクだけをファイルから読む（mmap は使わない。core/npy_file.py）。
1 点 17 バイト（int32 / float32 の場合。5.4 億点で約 9GB）。元ファイルが変わると（署名不一致）作り直す。
スキャン単位で追加できるので、箱に掛からず読まなかったスキャンは後で必要になったときに加わる。
書き込みに失敗しても（読み取り専用フォルダ・空き容量不足）読み直し自体は従来どおり続ける。
"""

import json
import os
import shutil
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.e57_io import file_signature
from core.npy_file import NpyFileArray, NpySequentialWriter
from core.logger import log_info, log_warning

CACHE_VERSION = 2
CACHE_SUFFIX = ".cache"
XYZ_KINDS = ("int32", "int64", "float32", "float64")
_XYZ_DTYPES = {"int32": np.int32, "int64": np.int64, "float32": np.float32, "float64": np.float64}
_XYZ_ITEM_BYTES = {"int32": 4, "int64": 8, "float32": 4, "float64": 8}
BYTES_PER_POINT = 4 * 3 + 3 + 2  # xyz (int32/float32) + rgb uint8 + intensity float16
FREE_SPACE_MARGIN = 1.1
_PART = ".part.npy"


def cache_dir_for(source_file: str, cache_dir: Optional[str] = None) -> str:
    """キャッシュフォルダのパス。既定は <元ファイル>.cache、cache_dir 指定時はその下に元ファイル名で作る"""
    src = os.path.abspath(source_file)
    if cache_dir:
        return os.path.join(cache_dir, os.path.basename(src) + CACHE_SUFFIX)
    return src + CACHE_SUFFIX


def estimate_cache_bytes(n_points: int, xyz_kind: str = "int32") -> int:
    per_point = _XYZ_ITEM_BYTES.get(xyz_kind, 4) * 3 + 3 + 2
    return int(n_points) * per_point + 64 * (int(n_points) // 1_000_000 + 2)


def local_from_stored(
    xyz: np.ndarray, kind: str, scale: Optional[Sequence[float]], offset: Optional[Sequence[float]]
) -> np.ndarray:
    """
    保存形式の座標 (n, 3) からローカル座標 float64 (n, 3) を復元する。
    生整数は core.e57_io の全点読込と同じ式（int → float64, × scale, + offset）で、ビット単位で同じ値になる。
    """
    xyz = np.asarray(xyz)
    if kind in ("int32", "int64"):
        out = np.empty((len(xyz), 3), dtype=np.float64)
        for i in range(3):
            np.multiply(xyz[:, i], float(scale[i]), out=out[:, i], casting="unsafe")
            if float(offset[i]) != 0.0:
                out[:, i] += float(offset[i])
        return out
    return np.asarray(xyz, dtype=np.float64) if xyz.dtype != np.float64 else np.array(xyz, dtype=np.float64, copy=True)


@dataclass
class CachedScan:
    """キャッシュ済みスキャン 1 つ分（配列は NpyFileArray = 範囲を指定したときだけファイルから読む。n_valid 点だけが有効）"""

    scan_index: int
    n_valid: int
    xyz_kind: str
    xyz_scale: Optional[List[float]]
    xyz_offset: Optional[List[float]]
    chunks: np.ndarray            # (m, 8) float64
    xyz: NpyFileArray               # (n, 3)（xyz_kind の dtype）
    rgb: Optional[NpyFileArray]     # (n, 3) uint8
    intensity: Optional[NpyFileArray]  # (n,) float16

    def chunk_slice(self, k: int) -> Tuple[int, int]:
        start = int(self.chunks[k, 0])
        count = int(self.chunks[k, 1])
        return start, start + count

    def chunk_corners_local(self, k: int) -> np.ndarray:
        """チャンク k の範囲の 8 頂点（ローカル座標）"""
        xmin, xmax, ymin, ymax, zmin, zmax = (float(v) for v in self.chunks[k, 2:8])
        return np.array(
            [[x, y, z] for x in (xmin, xmax) for y in (ymin, ymax) for z in (zmin, zmax)],
            dtype=np.float64,
        )

    def load_chunk_raw(self, k: int) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """チャンク k を (ローカル座標 float64 (n,3), 色 uint8 (n,3) or None, 強度 float16 (n,) or None) で返す（LOD 作成用）"""
        a, b = self.chunk_slice(k)
        local = local_from_stored(self.xyz[a:b], self.xyz_kind, self.xyz_scale, self.xyz_offset)
        rgb = np.asarray(self.rgb[a:b]) if self.rgb is not None else None
        ints = np.asarray(self.intensity[a:b]) if self.intensity is not None else None
        return local, rgb, ints

    def load_chunk(self, k: int) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """チャンク k を (ローカル座標 float64 (n,3), 色 float32 0〜1 or None, 強度 float32 or None) で返す"""
        a, b = self.chunk_slice(k)
        local = local_from_stored(self.xyz[a:b], self.xyz_kind, self.xyz_scale, self.xyz_offset)
        rgb = (np.asarray(self.rgb[a:b], dtype=np.float32) / 255.0) if self.rgb is not None else None
        ints = np.asarray(self.intensity[a:b], dtype=np.float32) if self.intensity is not None else None
        return local, rgb, ints


class ScanCacheWriter:
    """1 スキャン分をチャンクごとに書き込む。finish() でチャンク表と header を確定、abort() で部分ファイルを消す"""

    def __init__(
        self,
        cache: "SectionCache",
        scan_index: int,
        n_points: int,
        has_rgb: bool,
        has_int: bool,
        xyz_kind: str = "int32",
        xyz_scale: Optional[Sequence[float]] = None,
        xyz_offset: Optional[Sequence[float]] = None,
    ):
        if xyz_kind not in XYZ_KINDS:
            raise ValueError(f"座標の保存形式が不正です: {xyz_kind}")
        self.cache = cache
        self.scan_index = int(scan_index)
        self.n_points = int(n_points)
        self.has_rgb = has_rgb
        self.has_int = has_int
        self.xyz_kind = xyz_kind
        self.xyz_scale = [float(v) for v in xyz_scale] if xyz_scale is not None else None
        self.xyz_offset = [float(v) for v in xyz_offset] if xyz_offset is not None else None
        self.n_written = 0
        self.chunk_rows: List[List[float]] = []
        base = cache._scan_base(self.scan_index)
        self._paths = {
            "xyz": base + "_xyz" + _PART,
            "rgb": base + "_rgb" + _PART,
            "int": base + "_int" + _PART,
        }
        n_alloc = max(1, self.n_points)
        self._n_alloc = n_alloc
        # mmap ではなく、先頭から順に書く（mmap で書くと、書いた分がメモリに残り、5 億点でメモリが 95% まで埋まった）
        self._xyz = NpySequentialWriter(self._paths["xyz"], _XYZ_DTYPES[xyz_kind], (n_alloc, 3))
        self._rgb = NpySequentialWriter(self._paths["rgb"], np.uint8, (n_alloc, 3)) if has_rgb else None
        self._int = NpySequentialWriter(self._paths["int"], np.float16, (n_alloc,)) if has_int else None

    def add(self, xyz_store: np.ndarray, local_xyz: np.ndarray, rgb01: Optional[np.ndarray], ints: Optional[np.ndarray]) -> None:
        """
        保存形式の座標 (n,3)（生整数または float）、復元済みローカル座標 float64 (n,3)、色 0〜1 float32 (n,3) or None、
        強度 float32 (n,) or None を書く（非有限は呼び出し側で除いておく）。チャンク表の範囲は local_xyz から取る。
        """
        n = len(xyz_store)
        if n == 0:
            return
        if len(local_xyz) != n:
            raise ValueError("キャッシュ書き込み: 座標と復元座標の長さが一致しません")
        a = self.n_written
        b = a + n
        if b > self._n_alloc:
            raise ValueError(f"キャッシュの書き込みが宣言点数 {self.n_points:,} を超えました（スキャン #{self.scan_index}）")
        self._xyz.append(np.asarray(xyz_store, dtype=self._xyz.dtype))
        if self._rgb is not None:
            if rgb01 is not None:
                self._rgb.append(np.clip(np.rint(rgb01 * 255.0), 0, 255).astype(np.uint8))
            else:
                self._rgb.append(np.zeros((n, 3), dtype=np.uint8))
        if self._int is not None:
            if ints is not None:
                self._int.append(np.clip(ints, -65504.0, 65504.0).astype(np.float16))
            else:
                self._int.append(np.zeros(n, dtype=np.float16))
        mins = np.min(local_xyz, axis=0)
        maxs = np.max(local_xyz, axis=0)
        self.chunk_rows.append([float(a), float(n), float(mins[0]), float(maxs[0]), float(mins[1]), float(maxs[1]), float(mins[2]), float(maxs[2])])
        self.n_written = b

    def _close_maps(self, pad: bool = True) -> None:
        """書き込みを閉じる。pad=True なら宣言点数に満たない残りを 0 で埋める（以前の mmap 版と同じ大きさのファイルになる）"""
        for m in (self._xyz, self._rgb, self._int):
            if m is not None:
                try:
                    m.close(pad=pad)
                except Exception:
                    pass
        self._xyz = self._rgb = self._int = None

    def finish(self) -> None:
        """部分ファイルを正式名へ改名し、チャンク表と header.json を更新する"""
        self._close_maps()
        base = self.cache._scan_base(self.scan_index)
        chunks = np.asarray(self.chunk_rows, dtype=np.float64).reshape(-1, 8)
        np.save(base + "_chunks.npy", chunks)
        for key, final in (("xyz", base + "_xyz.npy"), ("rgb", base + "_rgb.npy"), ("int", base + "_int.npy")):
            part = self._paths[key]
            if os.path.exists(part):
                os.replace(part, final)
        self.cache._register_scan(
            self.scan_index, self.n_written, self.has_rgb, self.has_int, self.xyz_kind, self.xyz_scale, self.xyz_offset
        )

    def abort(self) -> None:
        self._close_maps(pad=False)
        for p in self._paths.values():
            try:
                if os.path.exists(p):
                    os.remove(p)
            except OSError:
                pass


class SectionCache:
    """元ファイル 1 つ分のキャッシュフォルダ。open() で読み（署名が合わなければ作り直し）、writer で追加する"""

    def __init__(self, source_file: str, directory: str, signature: Tuple[int, int]):
        self.source_file = os.path.abspath(source_file)
        self.directory = directory
        self.signature = (int(signature[0]), int(signature[1]))
        self.scans: Dict[int, dict] = {}
        self._loaded: Dict[int, CachedScan] = {}
        self.writable = True

    # --- パス ---
    def _scan_base(self, scan_index: int) -> str:
        return os.path.join(self.directory, f"scan{int(scan_index):02d}")

    @property
    def header_path(self) -> str:
        return os.path.join(self.directory, "header.json")

    # --- 開く / 作る ---
    @classmethod
    def open(cls, source_file: str, cache_dir: Optional[str] = None, create: bool = True) -> Optional["SectionCache"]:
        """
        キャッシュを開く。無ければ（create=True なら）空のキャッシュフォルダを用意して返す。
        署名（サイズ・更新時刻）か版が合わない既存キャッシュは捨てて作り直す。フォルダが作れなければ None。
        """
        directory = cache_dir_for(source_file, cache_dir)
        sig = file_signature(source_file)
        cache = cls(source_file, directory, sig)
        header = cache._read_header()
        if header is not None:
            if header.get("version") == CACHE_VERSION and tuple(header.get("signature", ())) == sig:
                for k, v in header.get("scans", {}).items():
                    cache.scans[int(k)] = v
                cache._cleanup_parts()
                return cache
            log_info(f"読み直しキャッシュが元ファイル（または版）と合わないため作り直します: {directory}")
            if not cache._remove_dir():
                return None
        elif os.path.isdir(directory):
            cache._cleanup_parts()
        if not create:
            return None
        try:
            os.makedirs(directory, exist_ok=True)
            cache._write_header()
        except OSError as e:
            log_warning(f"読み直しキャッシュのフォルダを作れないためキャッシュ無しで続けます: {directory} ({e})")
            return None
        return cache

    def _read_header(self) -> Optional[dict]:
        try:
            with open(self.header_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    def _write_header(self) -> None:
        data = {
            "version": CACHE_VERSION,
            "source": self.source_file,
            "signature": list(self.signature),
            "scans": {str(k): v for k, v in sorted(self.scans.items())},
        }
        tmp = self.header_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        os.replace(tmp, self.header_path)

    def _remove_dir(self) -> bool:
        try:
            shutil.rmtree(self.directory)
            return True
        except OSError as e:
            log_warning(f"古い読み直しキャッシュを削除できません: {self.directory} ({e})")
            return False

    def _cleanup_parts(self) -> None:
        try:
            for name in os.listdir(self.directory):
                if name.endswith(_PART):
                    os.remove(os.path.join(self.directory, name))
        except OSError:
            pass

    # --- 参照 ---
    def has_scan(self, scan_index: int) -> bool:
        return int(scan_index) in self.scans

    def scan(self, scan_index: int) -> Optional[CachedScan]:
        """スキャンのキャッシュを開く（mmap は使わない）。ファイルが欠けていれば None（呼び出し側はデコードに戻す）"""
        idx = int(scan_index)
        if idx in self._loaded:
            return self._loaded[idx]
        meta = self.scans.get(idx)
        if meta is None:
            return None
        base = self._scan_base(idx)
        kind = str(meta.get("xyz_kind", ""))
        if kind not in XYZ_KINDS:
            log_warning(f"読み直しキャッシュのスキャン #{idx} の座標形式が不明なためデコードに戻します: {kind}")
            return None
        try:
            chunks = np.load(base + "_chunks.npy")
            xyz = NpyFileArray(base + "_xyz.npy")
            rgb = NpyFileArray(base + "_rgb.npy") if meta.get("has_rgb") else None
            inten = NpyFileArray(base + "_int.npy") if meta.get("has_int") else None
        except (OSError, ValueError) as e:
            log_warning(f"読み直しキャッシュのスキャン #{idx} が読めないためデコードに戻します: {e}")
            return None
        n_valid = int(meta.get("n_valid", 0))
        if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) < n_valid or xyz.dtype != _XYZ_DTYPES[kind]:
            log_warning(f"読み直しキャッシュのスキャン #{idx} の形状が不正なためデコードに戻します")
            return None
        cs = CachedScan(
            scan_index=idx,
            n_valid=n_valid,
            xyz_kind=kind,
            xyz_scale=meta.get("xyz_scale"),
            xyz_offset=meta.get("xyz_offset"),
            chunks=np.asarray(chunks, dtype=np.float64).reshape(-1, 8),
            xyz=xyz,
            rgb=rgb,
            intensity=inten,
        )
        self._loaded[idx] = cs
        return cs

    # --- 追加 ---
    def can_write(self, n_points: int, xyz_kind: str = "int32") -> bool:
        """書き込み可能かつ空き容量が足りるか（足りなければ警告してキャッシュ無しで続ける）"""
        if not self.writable:
            return False
        try:
            free = shutil.disk_usage(self.directory).free
        except OSError:
            return False
        need = int(estimate_cache_bytes(n_points, xyz_kind) * FREE_SPACE_MARGIN)
        if free < need:
            log_warning(
                f"読み直しキャッシュを作る空き容量が足りません（必要 {need / 1e9:.1f}GB / 空き {free / 1e9:.1f}GB）。"
                f"キャッシュ無しで続けます: {self.directory}"
            )
            self.writable = False
            return False
        return True

    def writer(
        self,
        scan_index: int,
        n_points: int,
        has_rgb: bool,
        has_int: bool,
        xyz_kind: str = "int32",
        xyz_scale: Optional[Sequence[float]] = None,
        xyz_offset: Optional[Sequence[float]] = None,
    ) -> Optional[ScanCacheWriter]:
        try:
            return ScanCacheWriter(self, scan_index, n_points, has_rgb, has_int, xyz_kind, xyz_scale, xyz_offset)
        except (OSError, ValueError) as e:
            log_warning(f"読み直しキャッシュを書き始められないためキャッシュ無しで続けます: {e}")
            self.writable = False
            return None

    def _register_scan(
        self,
        scan_index: int,
        n_valid: int,
        has_rgb: bool,
        has_int: bool,
        xyz_kind: str,
        xyz_scale: Optional[List[float]],
        xyz_offset: Optional[List[float]],
    ) -> None:
        self.scans[int(scan_index)] = {
            "n_valid": int(n_valid),
            "has_rgb": bool(has_rgb),
            "has_int": bool(has_int),
            "xyz_kind": xyz_kind,
            "xyz_scale": xyz_scale,
            "xyz_offset": xyz_offset,
        }
        self._loaded.pop(int(scan_index), None)
        self._write_header()

    def total_bytes(self) -> int:
        total = 0
        try:
            for name in os.listdir(self.directory):
                total += os.path.getsize(os.path.join(self.directory, name))
        except OSError:
            pass
        return total


def remove_cache(source_file: str, cache_dir: Optional[str] = None) -> bool:
    """キャッシュフォルダを削除する（無ければ False）"""
    directory = cache_dir_for(source_file, cache_dir)
    if not os.path.isdir(directory):
        return False
    shutil.rmtree(directory, ignore_errors=True)
    return not os.path.isdir(directory)
