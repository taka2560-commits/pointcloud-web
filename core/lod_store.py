# -*- coding: utf-8 -*-
"""
LOD ストア（視点に応じて元ファイルの密度で描くための、空間順に並べ替えた全点データ）

間引き読込（k 点に 1 点）の画面は、どこまで寄ってもファイルの 1/k の密度のまま。読み直しキャッシュ
（core/section_cache.py、ファイル順の全点）を元に、次の形へ並べ替えたデータを <元ファイル>.cache/ に置く。

  レベル分け : 各点に疑似乱数でレベル 0〜L を割り当てる（レベル L が 1/2、L-1 が 1/4、…、レベル 0 が 1/2^L）。
               レベル 0〜ℓ を合わせると全点の 1/2^(L-ℓ) の一様な抜き取りになり、足していくほど密になる（重複なし）。
  空間の並び : 各レベルの中は、全体を 128×128×128 に割った格子のモートン順（Z 順）に並べる。
               八分木のどの節（深さ 0〜7）も、各レベルの中で連続した範囲になる。
  葉         : 点数が多い節は 8 分割し、葉（描画の単位）を決める。葉ごと・レベルごとの範囲は cells / starts から求まる。

  lod.json        … 版、元ファイルの (サイズ, 更新時刻ns)、点数、レベル数、格子の原点と一辺、座標の原点
  lod_xyz.npy     float32 (N, 3) … ワールド座標（pose 適用後・ファイルの軸）− 原点。表示専用なので float32（0.1mm 級）
  lod_rgb.npy     uint8   (N, 3) … 色（無ければ作らない）
  lod_int.npy     float16 (N,)   … 反射強度（無ければ作らない）
  lod_cells.npy   int32   (K,)   … 点のある格子（モートン番号、昇順）
  lod_starts.npy  int64   (レベル数, K+1) … レベル ℓ の中で格子 j の点が始まる位置（全体通し番号）

描画時は、葉の箱を画面へ投影して「画面上の広さ × 目標密度」ぶんの点数を葉ごとに求め、全体が予算に収まるように
葉ごとのレベルを決める（select_leaf_levels）。選ばれた範囲だけをファイルから読む（gather）。粗いレベルが先に並ぶ。
作成は 3 段（1: レベル × 格子の点数を数える、2: 行き先の区間ごとの一時ファイルへ振り分ける、3: 区間ごとに並べて書く）で、
全点をメモリに載せず、ファイルも mmap しない（core/npy_file.py の冒頭に理由）。
GUI にも OpenGL にも依存しない。
"""

import json
import math
import os
import shutil
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.e57_io import file_signature
from core.fsutil import replace_file
from core.npy_file import NpyFileArray, NpySequentialWriter
from core.logger import log_info, log_warning
from core.processor import _coord_mode_axes, scan_pose_transform, transform_coordinate_system
from core.section_cache import SectionCache, cache_dir_for, local_from_stored
from core.sysinfo import format_bytes

LOD_VERSION = 1
LOD_DEPTH = 7
N_AXIS = 1 << LOD_DEPTH            # 128
N_MORTON = 1 << (3 * LOD_DEPTH)    # 2,097,152
DEFAULT_BASE_POINTS = 1_500_000    # レベル 0（最も粗い）の目標点数
DEFAULT_LEAF_MAX_POINTS = 400_000  # これを超える節は 8 分割する
MAX_LEVELS = 14
BYTES_PER_POINT = 4 * 3 + 3 + 2
FREE_SPACE_MARGIN = 1.1
FALLBACK_RGB = (77, 204, 242)      # 色の無いスキャン（読み込みワーカーの (0.3, 0.8, 0.95) と同じ）
HIDDEN_VOXEL = 0.05                # m。非表示の点（ノイズなど）を重ね描きから隠す判定の格子（最も細かい段）
HIDDEN_VOXELS = (0.05, 0.2, 0.8, 3.2)  # m。細かい段から粗い段へ。どれかで「非表示の点だけの格子」に入れば隠す
_PART = ".part.npy"
# 並べ替えの小分け（2026-11-13）: 行き先を SEGMENT_POINTS 点ごとの区間に分け、区間ごとの一時ファイルへ順に書いてから、
# 区間を 1 つずつメモリ上で並べて最終ファイルへ順に書く。以前は 9GB の最終ファイルを mmap して飛び飛びに書いていたので、
# 書いた分がメモリに残り、5.4 億点でメモリが 97% まで埋まってディスクの応答が 2.4 秒まで落ちた。
SEGMENT_POINTS = 16_000_000   # 1 区間の点数（メモリ上で並べる単位。約 0.6GB）
MAX_SEGMENTS = 64             # 一時ファイルの数の上限（点数が多いときは 1 区間を大きくする）
# 一時ファイルの 1 点: 区間の中の行き先 int32 + 座標 float32×3 + 色 uint8×3 + 強度 float16 = 21 バイト
_BUCKET_DTYPE = np.dtype([("d", "<i4"), ("xyz", "<f4", (3,)), ("rgb", "u1", (3,)), ("i", "<f2")])
BUCKET_BYTES_PER_POINT = _BUCKET_DTYPE.itemsize
_FILES = ("lod_xyz.npy", "lod_rgb.npy", "lod_int.npy", "lod_cells.npy", "lod_starts.npy", "lod.json")

# 7bit の値を 3bit おきに広げる表（モートン符号用）
_SPREAD = np.zeros(N_AXIS, dtype=np.int64)
for _i in range(N_AXIS):
    _v = 0
    for _b in range(LOD_DEPTH):
        if _i & (1 << _b):
            _v |= 1 << (3 * _b)
    _SPREAD[_i] = _v


def morton_encode(ix: np.ndarray, iy: np.ndarray, iz: np.ndarray) -> np.ndarray:
    """格子番号 (0〜127) ×3 → モートン番号 (0〜2^21-1)。x が最下位ビット"""
    return _SPREAD[ix] | (_SPREAD[iy] << 1) | (_SPREAD[iz] << 2)


def morton_decode_prefix(prefix: int, depth: int) -> Tuple[int, int, int]:
    """深さ depth の節のモートン接頭辞 → その深さでの格子番号 (ix, iy, iz)"""
    ix = iy = iz = 0
    for b in range(depth):
        ix |= ((prefix >> (3 * b)) & 1) << b
        iy |= ((prefix >> (3 * b + 1)) & 1) << b
        iz |= ((prefix >> (3 * b + 2)) & 1) << b
    return ix, iy, iz


def level_count_for(n_points: int, base_points: int = DEFAULT_BASE_POINTS) -> int:
    """レベル数（L+1）。レベル 0 が base_points 程度になるように決める"""
    if n_points <= max(1, base_points):
        return 1
    return int(min(MAX_LEVELS, math.ceil(math.log2(n_points / float(base_points))) + 1))


def point_levels(start_index: int, n: int, n_levels: int) -> np.ndarray:
    """
    通し番号 start_index〜 の n 点に、疑似乱数（splitmix64）でレベルを割り当てる。
    レベル L=n_levels-1 が確率 1/2、L-1 が 1/4、…、レベル 0 が 1/2^L。通し番号だけで決まるので 2 回の走査で同じ値になる。
    """
    if n_levels <= 1:
        return np.zeros(n, dtype=np.int64)
    z = np.arange(start_index, start_index + n, dtype=np.uint64)
    z = z + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z = z ^ (z >> np.uint64(31))
    v = (z >> np.uint64(32)).astype(np.float64)  # 0〜2^32-1
    top = n_levels - 1
    lg = np.floor(np.log2(np.maximum(v, 1.0))).astype(np.int64)
    nlz = np.where(v < 1.0, 32, 31 - lg)  # 32bit での先頭ゼロの数（k 個以上になる確率は 2^-k）
    return top - np.minimum(top, nlz)


def _pose_points(scan, local: np.ndarray) -> np.ndarray:
    """ローカル座標 → ワールド座標（pose のみ。取り込み時の座標系変換は掛けない）"""
    pose = scan_pose_transform(scan)
    if pose is None:
        return local
    R, t = pose
    out = local.dot(R.T)
    out += t
    return out


def lod_paths(source_file: str, cache_dir: Optional[str] = None) -> Dict[str, str]:
    d = cache_dir_for(source_file, cache_dir)
    return {name: os.path.join(d, name) for name in _FILES}


def remove_lod(source_file: str, cache_dir: Optional[str] = None) -> None:
    """LOD のファイルだけを消す（読み直しキャッシュは残す）"""
    d = cache_dir_for(source_file, cache_dir)
    if not os.path.isdir(d):
        return
    for name in os.listdir(d):
        if name.startswith("lod"):
            try:
                os.remove(os.path.join(d, name))
            except OSError:
                pass


# ----------------------------------------------------------------------------------------------------------------------
# 作成
# ----------------------------------------------------------------------------------------------------------------------
def build_lod_store(
    scans: Sequence,
    cache_dir: Optional[str] = None,
    base_points: int = DEFAULT_BASE_POINTS,
    is_cancelled: Optional[Callable[[], bool]] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    segment_points: Optional[int] = None,
) -> List[Dict[str, object]]:
    """
    読み込み済みスキャンの読み込み元ファイルごとに LOD ストアを作る（読み直しキャッシュが揃っていること。
    揃っていなければ先に core.section_reload.ensure_section_cache を呼ぶ）。
    戻り値: ファイルごとの {"source", "n_points", "n_levels", "n_cells", "bytes", "elapsed_sec"}
    """
    groups: Dict[str, List] = {}
    for s in scans:
        schemas = getattr(s, "field_schemas", {}) or {}
        if getattr(s, "is_spherical_only", False) or not all(k in schemas for k in ("cartesianX", "cartesianY", "cartesianZ")):
            continue  # 直交座標の無いスキャンは読み直しキャッシュにも入らない
        if not getattr(s, "source_file", None) or getattr(s, "source_scan_index", None) is None:
            raise ValueError(f"スキャン「{s.name}」に読み込み元ファイルの記録がないため LOD データは作れません。")
        groups.setdefault(os.path.abspath(s.source_file), []).append(s)
    results = []
    n_files = len(groups)
    for f_i, (source, file_scans) in enumerate(groups.items()):
        def sub_progress(ratio: float, msg: str, _i=f_i):
            if progress_callback is not None:
                progress_callback((_i + max(0.0, min(1.0, ratio))) / n_files, msg)

        results.append(_build_one(source, file_scans, cache_dir, base_points, is_cancelled, sub_progress, segment_points))
    return results


def prepare_lod_for_scans(
    scans: Sequence,
    cache_dir: Optional[str] = None,
    base_points: int = DEFAULT_BASE_POINTS,
    is_cancelled: Optional[Callable[[], bool]] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Dict[str, object]:
    """
    高精細表示（LOD）に必要なデータを揃える: 読み直しキャッシュが無いスキャンをデコードして作り、続けて LOD ストアを作る。
    戻り値: {"cache": ensure_section_cache の結果, "lod": build_lod_store の結果, "bytes": LOD の合計バイト, "elapsed_sec"}
    """
    from core.section_reload import ensure_section_cache

    t0 = time.perf_counter()

    def cache_progress(ratio: float, msg: str):
        if progress_callback is not None:
            progress_callback(0.5 * max(0.0, min(1.0, ratio)), msg)

    def lod_progress(ratio: float, msg: str):
        if progress_callback is not None:
            progress_callback(0.5 + 0.5 * max(0.0, min(1.0, ratio)), msg)

    cache_info = ensure_section_cache(scans, cache_dir=cache_dir, is_cancelled=is_cancelled, progress_callback=cache_progress)
    lod_info = build_lod_store(
        scans, cache_dir=cache_dir, base_points=base_points, is_cancelled=is_cancelled, progress_callback=lod_progress
    )
    return {
        "cache": cache_info,
        "lod": lod_info,
        "bytes": int(sum(int(r["bytes"]) for r in lod_info)),
        "elapsed_sec": time.perf_counter() - t0,
    }


def _build_one(source, file_scans, cache_dir, base_points, is_cancelled, progress, segment_points=None) -> Dict[str, object]:
    t0 = time.perf_counter()
    cache = SectionCache.open(source, cache_dir=cache_dir, create=False)
    if cache is None:
        raise RuntimeError(f"読み直しキャッシュがありません（先にキャッシュを作成してください）: {source}")
    file_scans = sorted(file_scans, key=lambda s: int(s.source_scan_index))
    pairs = []
    for s in file_scans:
        cs = cache.scan(int(s.source_scan_index)) if cache.has_scan(int(s.source_scan_index)) else None
        if cs is None:
            raise RuntimeError(f"スキャン「{s.name}」の読み直しキャッシュがありません: {cache.directory}")
        pairs.append((s, cs))
    n_points = int(sum(cs.n_valid for _, cs in pairs))
    if n_points <= 0:
        raise ValueError(f"LOD データを作る点がありません: {source}")
    n_levels = level_count_for(n_points, base_points)
    has_rgb = any(cs.rgb is not None for _, cs in pairs)
    has_int = any(cs.intensity is not None for _, cs in pairs)

    try:
        free = shutil.disk_usage(cache.directory).free
    except OSError:
        free = 0
    # 並べ替えの途中は、一時ファイル（1 点 21 バイト）がいちばん大きい。区間を最終ファイルへ移すたびに消していく
    need = int(n_points * max(BYTES_PER_POINT, BUCKET_BYTES_PER_POINT) * FREE_SPACE_MARGIN)
    if free < need:
        raise RuntimeError(
            f"LOD データを作る空き容量が足りません（必要 {format_bytes(need)} / 空き {format_bytes(free)}）: {cache.directory}"
        )

    # 格子の範囲: キャッシュのチャンク表（ローカル座標の範囲）の 8 頂点を pose で移して合わせる（やや大きめになるが取りこぼさない）
    bmin = np.full(3, np.inf)
    bmax = np.full(3, -np.inf)
    for s, cs in pairs:
        if len(cs.chunks) == 0:
            continue
        c = cs.chunks[:, 2:8]
        corners = np.stack(
            [np.stack([c[:, ix], c[:, 2 + iy], c[:, 4 + iz]], axis=1) for ix in (0, 1) for iy in (0, 1) for iz in (0, 1)],
            axis=1,
        ).reshape(-1, 3)
        world = _pose_points(s, corners)
        bmin = np.minimum(bmin, world.min(axis=0))
        bmax = np.maximum(bmax, world.max(axis=0))
    extent = np.maximum(bmax - bmin, 1e-6)
    side = float(np.max(extent)) * (1.0 + 1e-6) + 1e-6
    cell = side / N_AXIS
    origin = (bmin + bmax) / 2.0

    def _check():
        if is_cancelled is not None and is_cancelled():
            raise InterruptedError("LOD データの作成がユーザーによって中止されました。")

    def _keys(s, local: np.ndarray, g0: int) -> Tuple[np.ndarray, np.ndarray]:
        """(ワールド座標, レベル × N_MORTON + モートン番号)"""
        world = _pose_points(s, local)
        ijk = np.floor((world - bmin) / cell).astype(np.int64)
        np.clip(ijk, 0, N_AXIS - 1, out=ijk)
        key = point_levels(g0, len(world), n_levels) * N_MORTON + morton_encode(ijk[:, 0], ijk[:, 1], ijk[:, 2])
        return world, key

    # --- 1 回目: レベル × 格子の点数 ---
    counts = np.zeros(n_levels * N_MORTON, dtype=np.int64)
    g = 0
    for s, cs in pairs:
        name = s.name or f"Scan {s.scan_index}"
        for k in range(len(cs.chunks)):
            _check()
            a, b = cs.chunk_slice(k)
            local = local_from_stored(cs.xyz[a:b], cs.xyz_kind, cs.xyz_scale, cs.xyz_offset)  # 1 回目は色・強度を読まない
            _, key = _keys(s, local, g)
            u, c = np.unique(key, return_counts=True)
            counts[u] += c
            g += len(local)
            progress(0.45 * g / n_points, f"LOD 作成 1/3: 「{name}」の点を数えています ({g:,}/{n_points:,} 点)")
    counts2 = counts.reshape(n_levels, N_MORTON)
    cells = np.flatnonzero(counts2.any(axis=0)).astype(np.int32)
    cnt = counts2[:, cells]
    level_sizes = cnt.sum(axis=1)
    level_base = np.concatenate([[0], np.cumsum(level_sizes)])
    starts = np.empty((n_levels, len(cells) + 1), dtype=np.int64)
    for lv in range(n_levels):
        starts[lv, 0] = level_base[lv]
        starts[lv, 1:] = level_base[lv] + np.cumsum(cnt[lv])
    # 書き込み位置（レベル × 格子ごとの現在位置）。counts の領域を使い回す
    cursor = counts
    cursor[:] = -1
    for lv in range(n_levels):
        cursor[lv * N_MORTON + cells.astype(np.int64)] = starts[lv, :-1]
    del counts2, cnt

    # --- 2 回目: 行き先の区間ごとの一時ファイルへ順に書く → 区間ごとに並べて最終ファイルへ順に書く ---
    paths = lod_paths(source, cache_dir)
    part = {k: paths[k][: -len(".npy")] + _PART for k in ("lod_xyz.npy", "lod_rgb.npy", "lod_int.npy")}
    seg_points = int(segment_points) if segment_points else max(SEGMENT_POINTS, -(-n_points // MAX_SEGMENTS))
    seg_points = max(1, min(seg_points, 2_000_000_000))  # 区間の中の行き先は int32 で持つ
    n_seg = -(-n_points // seg_points)
    lod_dir = os.path.dirname(paths["lod_xyz.npy"])
    bucket_paths = [os.path.join(lod_dir, f"lod_bucket{i:03d}{_PART}") for i in range(n_seg)]
    bucket_files: List[Optional[object]] = [None] * n_seg
    writers: List[NpySequentialWriter] = []
    for p_old in bucket_paths:  # leftovers of an interrupted run must not be appended to
        if os.path.exists(p_old):
            os.remove(p_old)
    try:
        g = 0
        for s, cs in pairs:
            name = s.name or f"Scan {s.scan_index}"
            for k in range(len(cs.chunks)):
                _check()
                local, rgb, ints = cs.load_chunk_raw(k)
                n = len(local)
                world, key = _keys(s, local, g)
                order = np.argsort(key, kind="stable")
                ks = key[order]
                first = np.flatnonzero(np.concatenate([[True], ks[1:] != ks[:-1]]))
                lens = np.diff(np.concatenate([first, [n]]))
                rank = np.arange(n, dtype=np.int64) - np.repeat(first, lens)
                dest = cursor[ks] + rank   # 行き先（キーの順に並べてあるので、小さい順になっている）
                xyz_o = (world[order] - origin).astype(np.float32)
                rgb_o = rgb[order] if rgb is not None else None
                int_o = ints[order] if ints is not None else None
                seg_lo = int(dest[0] // seg_points)
                seg_hi = int(dest[-1] // seg_points)
                edges = np.searchsorted(dest, np.arange(seg_lo, seg_hi + 2, dtype=np.int64) * seg_points)
                for j, seg in enumerate(range(seg_lo, seg_hi + 1)):
                    i0, i1 = int(edges[j]), int(edges[j + 1])
                    if i1 <= i0:
                        continue
                    rec = np.empty(i1 - i0, dtype=_BUCKET_DTYPE)
                    rec["d"] = dest[i0:i1] - seg * seg_points
                    rec["xyz"] = xyz_o[i0:i1]
                    rec["rgb"] = rgb_o[i0:i1] if rgb_o is not None else np.asarray(FALLBACK_RGB, dtype=np.uint8)
                    rec["i"] = int_o[i0:i1] if int_o is not None else np.float16(0.0)
                    if bucket_files[seg] is None:
                        bucket_files[seg] = open(bucket_paths[seg], "ab")
                    bucket_files[seg].write(memoryview(rec).cast("B"))
                cursor[ks[first]] += lens
                g += n
                progress(0.45 + 0.3 * g / n_points, f"LOD 作成 2/3: 「{name}」の点を振り分けています ({g:,}/{n_points:,} 点)")
        for i, f in enumerate(bucket_files):
            if f is not None:
                f.close()
                bucket_files[i] = None

        out_xyz = NpySequentialWriter(part["lod_xyz.npy"], np.float32, (n_points, 3))
        writers.append(out_xyz)
        out_rgb = out_int = None
        if has_rgb:
            out_rgb = NpySequentialWriter(part["lod_rgb.npy"], np.uint8, (n_points, 3))
            writers.append(out_rgb)
        if has_int:
            out_int = NpySequentialWriter(part["lod_int.npy"], np.float16, (n_points,))
            writers.append(out_int)
        done = 0
        for seg in range(n_seg):
            _check()
            seg_len = min(seg_points, n_points - seg * seg_points)
            rec = np.fromfile(bucket_paths[seg], dtype=_BUCKET_DTYPE) if os.path.exists(bucket_paths[seg]) else np.empty(0, dtype=_BUCKET_DTYPE)
            if len(rec) != seg_len:
                raise RuntimeError(f"LOD の並べ替えで点数が合いません（区間 {seg}: {len(rec):,} / {seg_len:,} 点）")
            d = rec["d"]
            buf = np.empty((seg_len, 3), dtype=np.float32)
            buf[d] = rec["xyz"]
            out_xyz.append(buf)
            if out_rgb is not None:
                buf_c = np.empty((seg_len, 3), dtype=np.uint8)
                buf_c[d] = rec["rgb"]
                out_rgb.append(buf_c)
            if out_int is not None:
                buf_i = np.empty(seg_len, dtype=np.float16)
                buf_i[d] = rec["i"]
                out_int.append(buf_i)
            del rec, d, buf
            os.remove(bucket_paths[seg])
            done += seg_len
            progress(0.75 + 0.22 * done / n_points, f"LOD 作成 3/3: 並べ替えた点を書き込んでいます ({done:,}/{n_points:,} 点)")
        for w in writers:
            w.close(pad=False)
        writers = []
        progress(0.97, "LOD データを確定しています...")
        for k_name, p in part.items():
            if os.path.exists(p):
                replace_file(p, paths[k_name])
        np.save(paths["lod_cells.npy"], cells)
        np.save(paths["lod_starts.npy"], starts)
        header = {
            "version": LOD_VERSION,
            "source": source,
            "signature": list(file_signature(source)),
            "n_points": n_points,
            "n_levels": n_levels,
            "depth": LOD_DEPTH,
            "grid_min": [float(v) for v in bmin],
            "grid_side": side,
            "origin": [float(v) for v in origin],
            "has_rgb": bool(has_rgb),
            "has_int": bool(has_int),
            "level_sizes": [int(v) for v in level_sizes],
        }
        tmp = paths["lod.json"] + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(header, f, ensure_ascii=False, indent=1)
        replace_file(tmp, paths["lod.json"])
    except BaseException:
        for f in bucket_files:
            if f is not None:
                try:
                    f.close()
                except Exception:
                    pass
        for w in writers:
            try:
                w.close(pad=False)
            except Exception:
                pass
        for p in list(part.values()) + bucket_paths:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except OSError:
                pass
        raise

    total_bytes = sum(os.path.getsize(paths[k]) for k in _FILES if os.path.exists(paths[k]))
    elapsed = time.perf_counter() - t0
    log_info(
        f"LOD データ作成: {os.path.basename(source)} {n_points:,} 点, {n_levels} レベル, 格子 {len(cells):,} 個, "
        f"{format_bytes(total_bytes)}, {elapsed:.1f} 秒"
    )
    progress(1.0, "LOD データの作成が完了しました")
    return {
        "source": source,
        "n_points": n_points,
        "n_levels": n_levels,
        "n_cells": int(len(cells)),
        "bytes": int(total_bytes),
        "elapsed_sec": float(elapsed),
    }


# ----------------------------------------------------------------------------------------------------------------------
# 読み出し
# ----------------------------------------------------------------------------------------------------------------------
class LodStore:
    """
    1 ファイル分の LOD ストア。葉の表と、葉ごとのレベル指定からの読み出しを提供する。
    点のファイルは mmap せず、要る範囲だけをファイルから読む（NpyFileArray）。mmap だと、見て回るうちに読んだ分が
    使用メモリに積み上がる。
    """

    def __init__(self, source_file: str, header: dict, paths: Dict[str, str], leaf_max_points: int):
        self.source_file = source_file
        self.n_points = int(header["n_points"])
        self.n_levels = int(header["n_levels"])
        self.grid_min = np.asarray(header["grid_min"], dtype=np.float64)
        self.grid_side = float(header["grid_side"])
        self.origin = np.asarray(header["origin"], dtype=np.float64)
        self.cells = np.load(paths["lod_cells.npy"]).astype(np.int64)
        self.starts = np.load(paths["lod_starts.npy"])
        self.xyz = NpyFileArray(paths["lod_xyz.npy"])
        self.rgb = NpyFileArray(paths["lod_rgb.npy"]) if header.get("has_rgb") else None
        self.intensity = NpyFileArray(paths["lod_int.npy"]) if header.get("has_int") else None
        if self.xyz.shape != (self.n_points, 3) or self.starts.shape != (self.n_levels, len(self.cells) + 1):
            raise ValueError("LOD データの形状が不正です")
        self._build_leaves(int(leaf_max_points))

    @classmethod
    def open(
        cls, source_file: str, cache_dir: Optional[str] = None, leaf_max_points: int = DEFAULT_LEAF_MAX_POINTS
    ) -> Optional["LodStore"]:
        """LOD ストアを開く。無い・版や署名が合わない・壊れている場合は None"""
        paths = lod_paths(source_file, cache_dir)
        try:
            with open(paths["lod.json"], "r", encoding="utf-8") as f:
                header = json.load(f)
        except (OSError, ValueError):
            return None
        try:
            if header.get("version") != LOD_VERSION or int(header.get("depth", -1)) != LOD_DEPTH:
                return None
            if tuple(header.get("signature", ())) != file_signature(source_file):
                return None
            return cls(os.path.abspath(source_file), header, paths, leaf_max_points)
        except (OSError, ValueError, KeyError) as e:
            log_warning(f"LOD データを開けません: {paths['lod.json']} ({e})")
            return None

    def _build_leaves(self, leaf_max: int) -> None:
        """点数が leaf_max を超える節を 8 分割して葉を決める（モートン順）。葉ごとの格子範囲・箱・レベル別累積点数を作る"""
        k = len(self.cells)
        tot = (self.starts[:, 1:] - self.starts[:, :-1]).sum(axis=0) if k else np.zeros(0, dtype=np.int64)
        prefix_sum = np.concatenate([[0], np.cumsum(tot)])
        lo_list: List[int] = []
        hi_list: List[int] = []
        depth_list: List[int] = []
        prefix_list: List[int] = []
        stack = [(0, 0)]
        while stack:
            depth, prefix = stack.pop()
            shift = 3 * (LOD_DEPTH - depth)
            lo = int(np.searchsorted(self.cells, prefix << shift, side="left"))
            hi = int(np.searchsorted(self.cells, (prefix + 1) << shift, side="left"))
            if hi <= lo:
                continue
            if depth < LOD_DEPTH and int(prefix_sum[hi] - prefix_sum[lo]) > leaf_max:
                for child in range(7, -1, -1):  # 0 から順に取り出されるよう逆順に積む
                    stack.append((depth + 1, (prefix << 3) | child))
                continue
            lo_list.append(lo)
            hi_list.append(hi)
            depth_list.append(depth)
            prefix_list.append(prefix)
        self.leaf_lo = np.asarray(lo_list, dtype=np.int64)
        self.leaf_hi = np.asarray(hi_list, dtype=np.int64)
        n_leaf = len(lo_list)
        mins = np.empty((n_leaf, 3), dtype=np.float64)
        sizes = np.empty(n_leaf, dtype=np.float64)
        for i, (d, p) in enumerate(zip(depth_list, prefix_list)):
            ix, iy, iz = morton_decode_prefix(p, d)
            size = self.grid_side / (1 << d)
            mins[i] = self.grid_min + np.array([ix, iy, iz], dtype=np.float64) * size
            sizes[i] = size
        self.leaf_min = mins
        self.leaf_max = mins + sizes[:, None]
        per_level = self.starts[:, self.leaf_hi] - self.starts[:, self.leaf_lo] if n_leaf else np.zeros((self.n_levels, 0), dtype=np.int64)
        self.leaf_cum = np.cumsum(per_level, axis=0).T.copy()  # (葉, レベル) 累積点数

    @property
    def n_leaves(self) -> int:
        return int(len(self.leaf_lo))

    def leaf_corners_world(self) -> np.ndarray:
        """葉の箱の 8 頂点 (葉, 8, 3)。ワールド座標（ファイルの軸）"""
        mn, mx = self.leaf_min, self.leaf_max
        sel = [(ix, iy, iz) for ix in (0, 1) for iy in (0, 1) for iz in (0, 1)]
        out = np.empty((len(mn), 8, 3), dtype=np.float64)
        for j, (ix, iy, iz) in enumerate(sel):
            out[:, j, 0] = mx[:, 0] if ix else mn[:, 0]
            out[:, j, 1] = mx[:, 1] if iy else mn[:, 1]
            out[:, j, 2] = mx[:, 2] if iz else mn[:, 2]
        return out

    def count_for(self, levels: np.ndarray) -> int:
        levels = np.minimum(np.asarray(levels, dtype=np.int64), self.n_levels - 1)
        sel = levels >= 0
        if not np.any(sel):
            return 0
        return int(self.leaf_cum[np.flatnonzero(sel), levels[sel]].sum())

    def gather(self, levels: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        葉ごとのレベル（-1 = 描かない）を受け取り、レベル 0〜指定レベルの点を読む。
        戻り値: (原点基準の座標 float32 (M,3), 色 uint8 (M,3) or None, 強度 float16 (M,) or None)。粗いレベルが先に並ぶ。
        """
        levels = np.minimum(np.asarray(levels, dtype=np.int64), self.n_levels - 1)
        seg_a: List[np.ndarray] = []
        seg_b: List[np.ndarray] = []
        for lv in range(self.n_levels):
            sel = np.flatnonzero(levels >= lv)
            if len(sel) == 0:
                break
            a = self.starts[lv, self.leaf_lo[sel]]
            b = self.starts[lv, self.leaf_hi[sel]]
            keep = b > a
            a, b = a[keep], b[keep]
            if len(a) == 0:
                continue
            brk = np.flatnonzero(a[1:] != b[:-1])  # 隣り合う範囲はつなげて読む
            seg_a.append(np.concatenate([[a[0]], a[1:][brk]]))
            seg_b.append(np.concatenate([b[:-1][brk], [b[-1]]]))
        if not seg_a:
            return (
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.uint8) if self.rgb is not None else None,
                np.empty(0, dtype=np.float16) if self.intensity is not None else None,
            )
        a_all = np.concatenate(seg_a)
        b_all = np.concatenate(seg_b)
        total = int((b_all - a_all).sum())
        xyz = np.empty((total, 3), dtype=np.float32)
        rgb = np.empty((total, 3), dtype=np.uint8) if self.rgb is not None else None
        inten = np.empty(total, dtype=np.float16) if self.intensity is not None else None
        a_list, b_list = a_all.tolist(), b_all.tolist()
        self.xyz.read_ranges(a_list, b_list, xyz)
        if rgb is not None:
            self.rgb.read_ranges(a_list, b_list, rgb)
        if inten is not None:
            self.intensity.read_ranges(a_list, b_list, inten)
        return xyz, rgb, inten

    def close(self) -> None:
        self.xyz = self.rgb = self.intensity = None


def select_leaf_levels(
    corners: np.ndarray,
    cum: np.ndarray,
    mvp: np.ndarray,
    viewport: Tuple[int, int],
    budget: int,
    px_density: float = 1.0,
    enabled: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    葉ごとの描画レベルを決める。
    :param corners: 葉の箱の 8 頂点 (葉, 8, 3)。mvp を掛ける座標系（ビューアの中心オフセット済み座標）
    :param cum: (葉, レベル) の累積点数
    :param mvp: 4×4（列ベクトルに左から掛ける形: clip = mvp @ [x, y, z, 1]）
    :param viewport: (幅 px, 高さ px)
    :param budget: 合計点数の上限
    :param px_density: 1 画素あたりの目標点数（葉が画面に占める広さ × これ が欲しい点数。葉の全点数で頭打ち）
    :param enabled: 葉ごとの有効フラグ（断面の箱に掛からない葉などを False にする）
    :return: (レベル (葉,) int。-1 は描かない, 集計)
    """
    n_leaf = len(corners)
    levels = np.full(n_leaf, -1, dtype=np.int64)
    info = {"n_leaves": float(n_leaf), "n_visible": 0.0, "n_points": 0.0, "scale": 1.0}
    if n_leaf == 0 or budget <= 0:
        return levels, info
    w_px, h_px = max(1, int(viewport[0])), max(1, int(viewport[1]))
    hom = np.concatenate([corners, np.ones((n_leaf, 8, 1), dtype=np.float64)], axis=2) @ np.asarray(mvp, dtype=np.float64).T
    x, y, z, w = hom[..., 0], hom[..., 1], hom[..., 2], hom[..., 3]
    outside = (
        (x < -w).all(axis=1) | (x > w).all(axis=1) | (y < -w).all(axis=1) | (y > w).all(axis=1)
        | (z < -w).all(axis=1) | (z > w).all(axis=1)
    )
    visible = ~outside
    if enabled is not None:
        visible &= np.asarray(enabled, dtype=bool)
    info["n_visible"] = float(np.count_nonzero(visible))
    if not np.any(visible):
        return levels, info

    front = w > 1e-9
    safe_w = np.where(front, w, 1.0)
    px = (np.where(front, x / safe_w, 0.0) * 0.5 + 0.5) * w_px
    py = (np.where(front, y / safe_w, 0.0) * 0.5 + 0.5) * h_px
    x0 = np.clip(px.min(axis=1), 0, w_px)
    x1 = np.clip(px.max(axis=1), 0, w_px)
    y0 = np.clip(py.min(axis=1), 0, h_px)
    y1 = np.clip(py.max(axis=1), 0, h_px)
    area = (x1 - x0) * (y1 - y0)
    area = np.where(front.all(axis=1), area, float(w_px * h_px))  # 一部がカメラの後ろにある葉は画面全体とみなす
    total = cum[:, -1].astype(np.float64)
    desire = np.where(visible, np.minimum(total, area * float(px_density)), 0.0)

    cum_f = cum.astype(np.float64)
    top = cum.shape[1] - 1
    rows = np.arange(n_leaf)

    def levels_for(scale: float) -> np.ndarray:
        d = desire * scale
        lv = np.minimum(top, (cum_f < d[:, None]).sum(axis=1))  # 欲しい点数以上になる最小のレベル
        lv = np.where(d < 0.5 * cum_f[:, 0], -1, lv)            # レベル 0 の半分も要らない葉は描かない
        return np.where(visible, lv, -1)

    def total_for(lv: np.ndarray) -> float:
        sel = lv >= 0
        return float(cum_f[rows[sel], lv[sel]].sum()) if np.any(sel) else 0.0

    lv = levels_for(1.0)
    scale = 1.0
    if total_for(lv) > budget:
        lo, hi = 0.0, 1.0
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            if total_for(levels_for(mid)) <= budget:
                lo = mid
            else:
                hi = mid
        scale = lo
        lv = levels_for(scale)
    levels = lv.astype(np.int64)
    info["n_points"] = total_for(levels)
    info["scale"] = scale
    return levels, info


def box_leaf_mask(corners_display: np.ndarray, box) -> np.ndarray:
    """断面ボックス（core.section_reload.SectionBox、表示ワールド座標）に掛かる可能性のある葉 (葉,) bool"""
    n_leaf = len(corners_display)
    mask = np.ones(n_leaf, dtype=bool)
    if box is None or n_leaf == 0:
        return mask
    margin = 0.001
    for axis, rng in enumerate((box.x_range, box.y_range, box.z_range)):
        if rng is not None:
            col = corners_display[:, :, axis]
            mask &= ~((col.max(axis=1) < float(rng[0]) - margin) | (col.min(axis=1) > float(rng[1]) + margin))
    if box.wall is not None:
        rad = math.radians(float(box.wall[0]))
        nx, ny = -math.sin(rad), math.cos(rad)
        d = (corners_display[:, :, 0] - float(box.wall[1])) * nx + (corners_display[:, :, 1] - float(box.wall[2])) * ny
        mask &= ~((d.max(axis=1) < float(box.wall[3]) - margin) | (d.min(axis=1) > float(box.wall[4]) + margin))
    return mask


class LodScene:
    """
    読み込み中の点群に対応する LOD ストアの集まり（読み込み元ファイルごとに 1 つ）。
    ビューアの座標（取り込み時の座標系変換後、中心オフセットを引いた float32）での選択と読み出しを提供する。
    """

    def __init__(self, stores: List[LodStore], coord_mode: str = "cartesian"):
        self.stores = stores
        self.coord_mode = coord_mode or "cartesian"
        self.n_points = int(sum(s.n_points for s in stores))
        self.max_levels = max((s.n_levels for s in stores), default=1)
        self._center: Optional[np.ndarray] = None
        self.corners_view: Optional[np.ndarray] = None  # (葉, 8, 3) 中心オフセット済み
        self.cum: Optional[np.ndarray] = None            # (葉, max_levels)
        self._leaf_slices: List[Tuple[int, int]] = []

    @classmethod
    def open_for_scans(
        cls, scans: Sequence, cache_dir: Optional[str] = None, leaf_max_points: int = DEFAULT_LEAF_MAX_POINTS
    ) -> Optional["LodScene"]:
        """読み込み済みスキャンの読み込み元すべてに有効な LOD ストアがあれば開く。1 つでも欠けていれば None"""
        sources: List[str] = []
        for s in scans:
            src = getattr(s, "source_file", None)
            if not src or not os.path.isfile(src):
                return None
            key = os.path.abspath(src)
            if key not in sources:
                sources.append(key)
        if not sources:
            return None
        stores = []
        for src in sources:
            st = LodStore.open(src, cache_dir=cache_dir, leaf_max_points=leaf_max_points)
            if st is None:
                return None
            stores.append(st)
        mode = getattr(scans[0], "coord_mode", "cartesian") or "cartesian"
        return cls(stores, mode)

    @property
    def n_leaves(self) -> int:
        return int(sum(s.n_leaves for s in self.stores))

    def prepare(self, center_offset: np.ndarray) -> None:
        """ビューアの中心オフセットに合わせて、葉の箱（ビューア座標）と累積点数の表を作る"""
        center = np.asarray(center_offset, dtype=np.float64).reshape(3)
        if self._center is not None and np.array_equal(center, self._center) and self.corners_view is not None:
            return
        self._center = center.copy()
        corners_list = []
        cum_list = []
        self._leaf_slices = []
        pos = 0
        for st in self.stores:
            cw = st.leaf_corners_world().reshape(-1, 3)
            cd = transform_coordinate_system(cw, self.coord_mode) - center
            corners_list.append(cd.reshape(-1, 8, 3))
            cum = st.leaf_cum
            if cum.shape[1] < self.max_levels:  # レベル数の少ないストアは最後の値で埋める
                pad = np.repeat(cum[:, -1:], self.max_levels - cum.shape[1], axis=1)
                cum = np.concatenate([cum, pad], axis=1)
            cum_list.append(cum)
            self._leaf_slices.append((pos, pos + st.n_leaves))
            pos += st.n_leaves
        self.corners_view = np.concatenate(corners_list) if corners_list else np.empty((0, 8, 3))
        self.cum = np.concatenate(cum_list) if cum_list else np.empty((0, self.max_levels), dtype=np.int64)

    def select(
        self, mvp: np.ndarray, viewport: Tuple[int, int], budget: int, px_density: float = 1.0, box=None
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """葉ごとのレベルを決める（prepare 済みであること）。box は表示ワールド座標の断面ボックス"""
        enabled = None
        if box is not None and box.is_active():
            enabled = box_leaf_mask(self.corners_view + self._center, box)
        return select_leaf_levels(self.corners_view, self.cum, mvp, viewport, budget, px_density, enabled)

    def gather_view(
        self, levels: np.ndarray, box=None
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], int]:
        """
        選ばれた点をビューア座標（中心オフセット済み float32）で読む。box があれば箱の中だけに絞る。
        戻り値: (座標 float32 (M,3), 色 uint8 (M,3) or None, 強度 float16 (M,) or None, 箱で絞る前の点数)
        """
        src, sign = _coord_mode_axes(self.coord_mode)
        pts_list, rgb_list, int_list = [], [], []
        any_rgb = any(st.rgb is not None for st in self.stores)
        any_int = any(st.intensity is not None for st in self.stores)
        for st, (a, b) in zip(self.stores, self._leaf_slices):
            xyz, rgb, inten = st.gather(levels[a:b])
            n = len(xyz)
            delta = transform_coordinate_system(st.origin.reshape(1, 3), self.coord_mode)[0] - self._center
            out = np.empty((n, 3), dtype=np.float32)
            for i in range(3):
                col = xyz[:, src[i]]
                out[:, i] = (-col if sign[i] < 0 else col) + np.float32(delta[i])
            pts_list.append(out)
            if any_rgb:
                rgb_list.append(rgb if rgb is not None else np.tile(np.asarray(FALLBACK_RGB, dtype=np.uint8), (n, 1)))
            if any_int:
                int_list.append(inten if inten is not None else np.zeros(n, dtype=np.float16))
        pts = np.concatenate(pts_list) if pts_list else np.empty((0, 3), dtype=np.float32)
        rgb = (np.concatenate(rgb_list) if rgb_list else np.empty((0, 3), dtype=np.uint8)) if any_rgb else None
        inten = (np.concatenate(int_list) if int_list else np.empty(0, dtype=np.float16)) if any_int else None
        n_before = len(pts)
        if box is not None and box.is_active() and n_before > 0:
            keep = np.flatnonzero(_box_contains_view(pts, self._center, box))
            pts = pts[keep]
            rgb = rgb[keep] if rgb is not None else None
            inten = inten[keep] if inten is not None else None
        return pts, rgb, inten, n_before

    def close(self) -> None:
        for st in self.stores:
            st.close()


def _box_contains_view(pts_view: np.ndarray, center: np.ndarray, box) -> np.ndarray:
    """ビューア座標 float32 の点が断面ボックス（表示ワールド座標）に入るか。箱の側を中心オフセットぶんずらして比べる"""
    mask = np.ones(len(pts_view), dtype=bool)
    for axis, rng in enumerate((box.x_range, box.y_range, box.z_range)):
        if rng is not None:
            col = pts_view[:, axis]
            mask &= (col >= np.float32(float(rng[0]) - center[axis])) & (col <= np.float32(float(rng[1]) - center[axis]))
    if box.wall is not None:
        rad = math.radians(float(box.wall[0]))
        nx, ny = np.float32(-math.sin(rad)), np.float32(math.cos(rad))
        cx = np.float32(float(box.wall[1]) - center[0])
        cy = np.float32(float(box.wall[2]) - center[1])
        d = (pts_view[:, 0] - cx) * nx + (pts_view[:, 1] - cy) * ny
        mask &= (d >= np.float32(box.wall[3])) & (d <= np.float32(box.wall[4]))
    return mask


# ----------------------------------------------------------------------------------------------------------------------
# 非表示の点（ノイズレイヤーなど）を重ね描きから隠すための格子
# ----------------------------------------------------------------------------------------------------------------------
def voxel_keys(points: np.ndarray, origin: np.ndarray, voxel: float = HIDDEN_VOXEL) -> np.ndarray:
    """点 (N, 3) を voxel 幅の格子番号に直し、1 本の int64 にまとめる（各軸 21bit）"""
    inv = 1.0 / float(voxel)
    off = np.int64(1 << 20)
    mask = np.int64((1 << 21) - 1)
    keys = np.zeros(len(points), dtype=np.int64)
    for axis in range(3):
        idx = np.floor((points[:, axis].astype(np.float64) - float(origin[axis])) * inv).astype(np.int64) + off
        np.clip(idx, 0, mask, out=idx)
        keys = (keys << np.int64(21)) | idx
    return keys


def hidden_voxel_sets(
    points: np.ndarray,
    visible: np.ndarray,
    origin: np.ndarray,
    voxels: Sequence[float] = HIDDEN_VOXELS,
    batch: int = 4_000_000,
) -> List[Tuple[float, np.ndarray]]:
    """
    読み込み済みの点 (N, 3) と表示フラグから、格子の大きさごとに「非表示の点だけがある格子」の番号
    （昇順・重複なし）を返す: [(格子幅, 番号), ...]。非表示の点が無ければ空のリスト。

    重ね描きの点がどれかの大きさでこの格子に入っていたら隠す。保存時の全点転写（最も近い判定点のラベルを引き継ぐ）
    に合わせた近似で、細かい格子（5cm）は面のそばの浮遊点、粗い格子（20cm・80cm・3.2m）は読み込みが間引かれて
    同じ細かい格子に判定点が無い「仲間の点」や、まるごと消した領域を拾う。
    表示中の点が 1 つでも入っている格子は隠さない（残した面を誤って消さない）。
    """
    visible = np.asarray(visible, dtype=bool)
    hidden_idx = np.flatnonzero(~visible)
    if len(hidden_idx) == 0:
        return []
    vis_idx = np.flatnonzero(visible)
    sets: List[Tuple[float, np.ndarray]] = []
    for voxel in voxels:
        hk = np.unique(voxel_keys(points[hidden_idx], origin, voxel))
        has_visible = np.zeros(len(hk), dtype=bool)
        for a in range(0, len(vis_idx), batch):
            vk = voxel_keys(points[vis_idx[a : a + batch]], origin, voxel)
            pos = np.searchsorted(hk, vk)
            pos[pos >= len(hk)] = len(hk) - 1
            hit = hk[pos] == vk
            has_visible[pos[hit]] = True
        sets.append((float(voxel), hk[~has_visible]))
    return sets


def not_hidden_mask(points: np.ndarray, hidden_sets: Optional[Sequence[Tuple[float, np.ndarray]]], origin: np.ndarray) -> np.ndarray:
    """点 (N, 3) のうち、どの大きさの「隠す格子」にも入っていないもの (N,) bool"""
    keep = np.ones(len(points), dtype=bool)
    if not hidden_sets or len(points) == 0:
        return keep
    for voxel, keys in hidden_sets:
        if len(keys) == 0:
            continue
        vk = voxel_keys(points, origin, voxel)
        pos = np.searchsorted(keys, vk)
        pos[pos >= len(keys)] = len(keys) - 1
        keep &= keys[pos] != vk
    return keep
