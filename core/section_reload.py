# -*- coding: utf-8 -*-
"""
断面ボックスの元ファイル読み直し（表示専用の重ね描き）の純粋関数

1 億点超の E57 は「k 点に 1 点」の間引きで読み込まれるため、断面ボックス内を全点表示しても密度は
ファイルの 1/k のまま。ここでは、断面ボックス（表示座標系: pose 適用後・座標系変換後）を受け取り、
読み込み元の E57 をスキャンごとにチャンクで流し読みして、ボックスに入る点だけを拾って返す。
全点をメモリに載せないので、ボックス内の点数（上限 max_points）ぶんのメモリで済む。

- スキャンの cartesianBounds（ヘッダの範囲）がボックスに掛からないスキャンは読まずに飛ばす。
- 読み直した点は表示専用で、選択・ノイズ移動の対象にはならない（編集は読み込み済みの点で行い、
  保存時の全点転写が元ファイルの全点へ写す）。
- ボックス内の点が max_points を超えたら、走査しながら 2 分の 1 ずつ間引いて（ほぼ一様）上限に収める。
  ボックス内の総点数 n_in_box は間引き前の正確な値を返す。
- キャッシュ（core/section_cache.py）: libe57 には読み飛ばしが無く毎回全点のデコードになる（5.4 億点で約 100 秒）ので、
  初回の読み直しでデコードした点を圧縮無しの自前形式で元ファイルの横に書き（1 点 17 バイト）、
  2 回目からはチャンク表の範囲が箱に掛かるチャンクだけを mmap で読む。書けない場合は従来どおりデコードする。
GUI にも OpenGL にも依存しない。読み込み元の照合（ファイルの有無・サイズ・更新時刻）は全点転写保存と同じ。
"""

import math
import os
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from pye57 import libe57

from core.e57_io import FieldSchema, e57_point_field_schemas, file_signature, iter_e57_point_chunks
from core.logger import log_info, log_warning
from core.processor import scan_pose_transform, transform_coordinate_system
from core.section_cache import SectionCache, local_from_stored

DEFAULT_MAX_POINTS = 10_000_000
DEFAULT_CHUNK_SIZE = 1_000_000
BOUNDS_MARGIN = 0.001  # m。cartesianBounds の丸めで境界の点を取りこぼさないための余裕
FALLBACK_COLOR = (0.3, 0.8, 0.95)  # 色の無いスキャンの表示色（読み込みワーカーと同じ）
_CARTESIAN = ("cartesianX", "cartesianY", "cartesianZ")
_RGB = ("colorRed", "colorGreen", "colorBlue")


@dataclass(frozen=True)
class SectionBox:
    """
    表示座標系で定義した断面ボックス。None の軸は制限なし。
    wall は 2 点ピック斜め壁の直交スライス (angle_deg, center_x, center_y, d_min, d_max)。
    ビューアの _get_section_inlier_mask() と同じ判定（軸範囲は両端を含む、壁は直交距離の範囲）。
    """

    x_range: Optional[Tuple[float, float]] = None
    y_range: Optional[Tuple[float, float]] = None
    z_range: Optional[Tuple[float, float]] = None
    wall: Optional[Tuple[float, float, float, float, float]] = None

    def is_active(self) -> bool:
        return any(r is not None for r in (self.x_range, self.y_range, self.z_range, self.wall))

    def _wall_normal(self) -> Tuple[float, float]:
        rad = math.radians(float(self.wall[0]))
        return -math.sin(rad), math.cos(rad)

    def contains(self, pts: np.ndarray) -> np.ndarray:
        """(N, 3) の表示座標に対するボックス内マスク"""
        n = len(pts)
        mask = np.ones(n, dtype=bool)
        if n == 0:
            return mask
        for axis, rng in enumerate((self.x_range, self.y_range, self.z_range)):
            if rng is not None:
                col = pts[:, axis]
                mask &= (col >= float(rng[0])) & (col <= float(rng[1]))
        if self.wall is not None:
            nx, ny = self._wall_normal()
            d = (pts[:, 0] - float(self.wall[1])) * nx + (pts[:, 1] - float(self.wall[2])) * ny
            mask &= (d >= float(self.wall[3])) & (d <= float(self.wall[4]))
        return mask

    def may_intersect(self, corners: np.ndarray, margin: float = BOUNDS_MARGIN) -> bool:
        """スキャンやチャンクの範囲（表示座標の 8 頂点）がボックスに掛かる可能性があるか。False なら読まなくてよい"""
        if corners is None or len(corners) == 0:
            return True
        for axis, rng in enumerate((self.x_range, self.y_range, self.z_range)):
            if rng is not None:
                col = corners[:, axis]
                if float(np.max(col)) < float(rng[0]) - margin or float(np.min(col)) > float(rng[1]) + margin:
                    return False
        if self.wall is not None:
            nx, ny = self._wall_normal()
            d = (corners[:, 0] - float(self.wall[1])) * nx + (corners[:, 1] - float(self.wall[2])) * ny
            if float(np.max(d)) < float(self.wall[3]) - margin or float(np.min(d)) > float(self.wall[4]) + margin:
                return False
        return True

    def describe(self) -> str:
        parts = []
        for name, rng in (("X", self.x_range), ("Y", self.y_range), ("Z", self.z_range)):
            if rng is not None:
                parts.append(f"{name} {rng[0]:.2f}〜{rng[1]:.2f}m")
        if self.wall is not None:
            parts.append(f"斜め壁{self.wall[0]:.1f}° {self.wall[3]:.2f}〜{self.wall[4]:.2f}m")
        return " / ".join(parts) if parts else "（断面なし）"


@dataclass
class SectionReloadResult:
    """読み直しの結果。points は表示座標（pose 適用後・座標系変換後）"""

    points: np.ndarray                 # (M, 3) float64
    colors: Optional[np.ndarray]       # (M, 3) float32 0〜1（どのスキャンにも色が無ければ None）
    intensities: Optional[np.ndarray]  # (M,) float32（どのスキャンにも強度が無ければ None）
    scan_ids: np.ndarray               # (M,) int32  読み込み時のスキャン番号（scans の添字）
    n_in_box: int = 0                  # ボックス内にあった元ファイルの点数（間引き前）
    n_read: int = 0                    # 走査した点数（キャッシュ使用時は読んだチャンクの点数）
    n_scans_read: int = 0
    n_scans_skipped: int = 0           # 範囲がボックスに掛からず飛ばしたスキャン数
    n_scans_unsupported: int = 0       # 直交座標を持たず読めなかったスキャン数
    capped: bool = False               # 上限 max_points で間引いたか
    elapsed_sec: float = 0.0
    cache_used_scans: int = 0          # キャッシュから読んだスキャン数
    cache_built_scans: int = 0         # 今回キャッシュに書いたスキャン数
    n_chunks_read: int = 0             # キャッシュから読んだチャンク数
    n_chunks_total: int = 0            # キャッシュ使用スキャンの総チャンク数
    cache_dir: Optional[str] = None    # 使った／作ったキャッシュフォルダ（無ければ None）
    cache_bytes: int = 0               # キャッシュフォルダの大きさ（バイト）

    @property
    def n_points(self) -> int:
        return int(len(self.points))


def _snapshot_value(node: Optional[dict]) -> Optional[float]:
    if not isinstance(node, dict):
        return None
    t = node.get("type")
    if t == "Float":
        return float(node["value"])
    if t == "ScaledInteger":
        return float(node["raw_value"]) * float(node["scale"]) + float(node["offset"])
    if t == "Integer":
        return float(node["value"])
    return None


def scan_bounds_corners(scan) -> Optional[np.ndarray]:
    """
    スキャンヘッダの cartesianBounds（ローカル座標）から、範囲の 8 頂点を表示座標へ変換して返す。
    cartesianBounds が無い・不完全なら None（＝範囲不明なので読む）。
    """
    snap = getattr(scan, "meta_snapshot", None)
    if not isinstance(snap, dict):
        return None
    cb = snap.get("children", {}).get("cartesianBounds")
    if not isinstance(cb, dict) or cb.get("type") != "Structure":
        return None
    ch = cb.get("children", {})
    vals = [_snapshot_value(ch.get(k)) for k in ("xMinimum", "xMaximum", "yMinimum", "yMaximum", "zMinimum", "zMaximum")]
    if any(v is None or not math.isfinite(v) for v in vals):
        return None
    xmin, xmax, ymin, ymax, zmin, zmax = vals
    corners = np.array(
        [[x, y, z] for x in (xmin, xmax) for y in (ymin, ymax) for z in (zmin, zmax)],
        dtype=np.float64,
    )
    return to_display_coordinates(scan, corners)


def to_display_coordinates(scan, local_pts: np.ndarray) -> np.ndarray:
    """スキャンのローカル生座標 (N, 3) を表示座標へ（pose を掛けてから取り込み時の座標系変換）"""
    pts = np.asarray(local_pts, dtype=np.float64)
    pose = scan_pose_transform(scan)
    if pose is not None:
        R, t = pose
        pts = pts.dot(R.T)
        pts += t
    return transform_coordinate_system(pts, getattr(scan, "coord_mode", "cartesian"))


def color_divisor(scan) -> float:
    """
    読み込みワーカーと同じ規則で色の正規化係数を決める（読み込み済みの色の最大値 > 255 なら 16bit、> 1 なら 8bit）。
    読み込み済みの点が無ければスキーマの最大値で判断する。
    """
    raw = getattr(scan, "raw_fields", {}) or {}
    max_c = 0.0
    if all(k in raw for k in _RGB) and len(raw["colorRed"]) > 0:
        max_c = max(float(np.max(raw[k])) for k in _RGB)
    else:
        schemas = getattr(scan, "field_schemas", {}) or {}
        for k in _RGB:
            sch = schemas.get(k)
            if sch is not None and sch.maximum is not None:
                max_c = max(max_c, float(sch.maximum))
    if max_c > 255.0:
        return 65535.0
    if max_c > 1.0:
        return 255.0
    return 1.0


def validate_reload_sources(scans: Sequence) -> None:
    """読み込み元ファイルの有無と変更の有無を確認する（全点転写保存と同じ基準）"""
    for s in scans:
        if not getattr(s, "source_file", None) or getattr(s, "source_scan_index", None) is None:
            raise ValueError(
                f"スキャン「{s.name}」に読み込み元ファイルの記録がないため、元ファイルからの読み直しはできません。"
            )
        if not os.path.isfile(s.source_file):
            raise FileNotFoundError(f"読み込み元ファイルが見つかりません: {s.source_file}")
        sig = getattr(s, "source_signature", None)
        if sig is not None and file_signature(s.source_file) != tuple(sig):
            raise RuntimeError(
                f"読み込み元ファイルが読込後に変更されています（サイズか更新時刻が異なります）: {s.source_file}\n"
                "元ファイルを開き直してから読み直してください。"
            )


class _Accumulator:
    """ボックス内の点を貯めつつ、上限を超えたら 2 分の 1 ずつ間引いて上限に収める"""

    def __init__(self, max_points: int, want_colors: bool, want_ints: bool):
        self.max_points = int(max_points) if max_points and max_points > 0 else 0
        self.want_colors = want_colors
        self.want_ints = want_ints
        self.pts: List[np.ndarray] = []
        self.cols: List[np.ndarray] = []
        self.ints: List[np.ndarray] = []
        self.sids: List[np.ndarray] = []
        self.total = 0
        self.keep_step = 1
        self.capped = False

    def add(self, pts: np.ndarray, cols: Optional[np.ndarray], ints: Optional[np.ndarray], scan_id: int) -> None:
        if len(pts) == 0:
            return
        if self.keep_step > 1:
            pts = pts[:: self.keep_step]
            cols = cols[:: self.keep_step] if cols is not None else None
            ints = ints[:: self.keep_step] if ints is not None else None
        n = len(pts)
        self.pts.append(np.ascontiguousarray(pts, dtype=np.float64))
        if self.want_colors:
            self.cols.append(
                np.ascontiguousarray(cols, dtype=np.float32)
                if cols is not None
                else np.tile(np.asarray(FALLBACK_COLOR, dtype=np.float32), (n, 1))
            )
        if self.want_ints:
            self.ints.append(
                np.ascontiguousarray(ints, dtype=np.float32) if ints is not None else np.zeros(n, dtype=np.float32)
            )
        self.sids.append(np.full(n, int(scan_id), dtype=np.int32))
        self.total += n
        while self.max_points > 0 and self.total > self.max_points:
            self._halve()

    def _halve(self) -> None:
        self.pts = [p[::2] for p in self.pts]
        self.cols = [c[::2] for c in self.cols]
        self.ints = [i[::2] for i in self.ints]
        self.sids = [s[::2] for s in self.sids]
        self.total = sum(len(p) for p in self.pts)
        self.keep_step *= 2
        self.capped = True

    def finish(self) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
        pts = np.concatenate(self.pts) if self.pts else np.empty((0, 3), dtype=np.float64)
        cols = (np.concatenate(self.cols) if self.cols else np.empty((0, 3), dtype=np.float32)) if self.want_colors else None
        ints = (np.concatenate(self.ints) if self.ints else np.empty(0, dtype=np.float32)) if self.want_ints else None
        sids = np.concatenate(self.sids) if self.sids else np.empty(0, dtype=np.int32)
        return pts, cols, ints, sids


def _xyz_storage_kind(file_schemas: Dict[str, FieldSchema]) -> Tuple[str, Optional[List[float]], Optional[List[float]]]:
    """
    元ファイルの cartesianX/Y/Z の定義から、キャッシュでの座標の持ち方を決める。
    ScaledInteger → 生整数（int32、範囲が収まらなければ int64）と scale/offset、Float → 同じ精度の float32/float64、
    それ以外（混在など）→ 計算済み float64。
    """
    schemas = [file_schemas.get(k) for k in _CARTESIAN]
    if all(sch is not None and sch.node_type == "ScaledInteger" for sch in schemas):
        fits32 = all(
            sch.minimum is not None and sch.maximum is not None and -(2**31) <= int(sch.minimum) and int(sch.maximum) <= 2**31 - 1
            for sch in schemas
        )
        return ("int32" if fits32 else "int64"), [float(sch.scale) for sch in schemas], [float(sch.offset) for sch in schemas]
    if all(sch is not None and sch.node_type == "Float" for sch in schemas):
        single = all(sch.precision == libe57.E57_SINGLE for sch in schemas)
        return ("float32" if single else "float64"), None, None
    return "float64", None, None


def _chunk_colors(chunk: Dict[str, np.ndarray], n: int, divisor: float) -> np.ndarray:
    """デコードしたチャンクの色を 0〜1 の float32 (n, 3) にする"""
    cols = np.empty((n, 3), dtype=np.float32)
    for i, k in enumerate(_RGB):
        cols[:, i] = chunk[k]
    if divisor != 1.0:
        cols /= np.float32(divisor)
    np.clip(cols, 0.0, 1.0, out=cols)
    return cols


def _decode_scan_chunks(s, cache: Optional[SectionCache], chunk_size: int, state: Dict[str, bool]):
    """
    1 スキャンを元ファイルからデコードして (ローカル座標 float64 (n,3), 色 0〜1 or None, 強度 or None, 読んだ点数) を
    チャンクごとに返すジェネレータ。cache が書き込み可能ならキャッシュへも書く。
    読み切ったら確定して state["built"]=True、途中で止まれば（例外・close）書きかけの部分ファイルを消す。
    state["building"] は書き込み中かどうか（進捗メッセージ用）。
    """
    schemas = s.field_schemas
    has_rgb = all(k in schemas for k in _RGB)
    has_int = "intensity" in schemas
    name = s.name or f"Scan {s.scan_index}"
    n_declared = int(getattr(s, "original_point_count", len(s.points)))
    fields = list(_CARTESIAN) + (list(_RGB) if has_rgb else []) + (["intensity"] if has_int else [])
    divisor = color_divisor(s) if has_rgb else 1.0
    # 座標の持ち方は元ファイルの定義で決める（生整数ならそのまま保持し、全点読込と同じ式で復元 → ビット単位で一致）
    xyz_kind, xyz_scale, xyz_offset = _xyz_storage_kind(e57_point_field_schemas(s.source_file, int(s.source_scan_index)))
    raw_xyz = xyz_kind in ("int32", "int64")
    writer = None
    if cache is not None and cache.can_write(n_declared, xyz_kind):
        writer = cache.writer(int(s.source_scan_index), n_declared, has_rgb, has_int, xyz_kind, xyz_scale, xyz_offset)
    state["building"] = writer is not None
    completed = False
    try:
        for chunk in iter_e57_point_chunks(
            s.source_file, int(s.source_scan_index), fields, chunk_size=chunk_size, raw_fields=(_CARTESIAN if raw_xyz else None)
        ):
            cx = chunk.get("cartesianX")
            if cx is None or "cartesianY" not in chunk or "cartesianZ" not in chunk:
                raise ValueError(f"スキャン「{name}」の読み直しで直交座標が読めませんでした: {s.source_file}")
            n_chunk = len(cx)
            if raw_xyz:
                store = np.column_stack([chunk[k] for k in _CARTESIAN]).astype(np.int64, copy=False)
                local = local_from_stored(store, "int64", xyz_scale, xyz_offset)
                if xyz_kind == "int32":
                    store = store.astype(np.int32)
            else:
                store_dtype = np.float32 if xyz_kind == "float32" else np.float64
                store = np.column_stack([np.asarray(chunk[k], dtype=store_dtype) for k in _CARTESIAN])
                local = store.astype(np.float64) if store_dtype != np.float64 else store
                finite = np.isfinite(local).all(axis=1)
                if not finite.all():
                    keep = np.flatnonzero(finite)
                    store = store[keep]
                    local = local[keep]
                    chunk = {k: v[keep] for k, v in chunk.items()}
            n_fin = len(local)
            rgb01 = _chunk_colors(chunk, n_fin, divisor) if (has_rgb and (writer is not None or n_fin > 0)) else None
            ints = None
            if has_int:
                ints = np.asarray(chunk["intensity"], dtype=np.float32)
                if not np.isfinite(ints).all():
                    ints = np.nan_to_num(ints, nan=0.0, posinf=0.0, neginf=0.0)
            if writer is not None:
                try:
                    writer.add(store, local, rgb01, ints)
                except (OSError, ValueError) as e:
                    log_warning(f"読み直しキャッシュの書き込みに失敗したためキャッシュ無しで続けます: {e}")
                    writer.abort()
                    writer = None
                    state["building"] = False
            del chunk, store
            yield local, rgb01, ints, n_chunk
        completed = True
    finally:
        if writer is not None:
            if completed:
                try:
                    writer.finish()
                    state["built"] = True
                except (OSError, ValueError) as e:
                    log_warning(f"読み直しキャッシュの確定に失敗しました（次回はデコードします）: {e}")
                    writer.abort()
            else:
                writer.abort()
        state["building"] = False


def ensure_section_cache(
    scans: Sequence,
    cache_dir: Optional[str] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    is_cancelled: Optional[Callable[[], bool]] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Dict[str, object]:
    """
    読み込み済みスキャンの読み込み元すべてについて、読み直しキャッシュを揃える（無いスキャンだけデコードして書く）。
    LOD データ（core/lod_store.py）を作る前提として使う。書けない場合（読み取り専用・空き容量不足）は RuntimeError。
    戻り値: {"built": 今回書いたスキャン数, "already": 既にあったスキャン数, "cache_dirs": [...], "cache_bytes": 合計バイト}
    """
    scans = [
        s for s in scans
        if not getattr(s, "is_spherical_only", False) and all(k in (getattr(s, "field_schemas", {}) or {}) for k in _CARTESIAN)
    ]
    if not scans:
        raise ValueError("直交座標を持つスキャンがありません。")
    validate_reload_sources(scans)
    caches: Dict[str, SectionCache] = {}
    todo: List[Tuple[object, SectionCache]] = []
    already = 0
    for s in scans:
        key = os.path.abspath(s.source_file)
        if key not in caches:
            cache = SectionCache.open(s.source_file, cache_dir=cache_dir, create=True)
            if cache is None:
                raise RuntimeError(f"読み直しキャッシュのフォルダを作れません: {s.source_file}")
            caches[key] = cache
        cache = caches[key]
        if cache.has_scan(int(s.source_scan_index)) and cache.scan(int(s.source_scan_index)) is not None:
            already += 1
        else:
            todo.append((s, cache))
    total = sum(int(getattr(s, "original_point_count", len(s.points))) for s, _ in todo)
    done = 0
    built = 0
    for s, cache in todo:
        name = s.name or f"Scan {s.scan_index}"
        n_declared = int(getattr(s, "original_point_count", len(s.points)))
        state = {"built": False, "building": False}
        base = done
        scan_read = 0
        gen = _decode_scan_chunks(s, cache, chunk_size, state)
        try:
            for _local, _rgb, _ints, n_chunk in gen:
                if is_cancelled is not None and is_cancelled():
                    raise InterruptedError("キャッシュ作成がユーザーによって中止されました。")
                if not state["building"]:
                    raise RuntimeError(
                        f"読み直しキャッシュを書き込めません（空き容量不足か書き込み不可）: {cache.directory}"
                    )
                scan_read += n_chunk
                done = base + min(n_declared, scan_read)
                if progress_callback is not None:
                    progress_callback(
                        min(1.0, done / total) if total > 0 else 1.0,
                        f"スキャン「{name}」を元ファイルからデコードしてキャッシュ作成中 ({scan_read:,}/{n_declared:,} 点)",
                    )
        finally:
            gen.close()
        if not state["built"]:
            raise RuntimeError(f"読み直しキャッシュを作成できませんでした: {cache.directory}")
        built += 1
        done = base + n_declared
    return {
        "built": built,
        "already": already,
        "cache_dirs": [c.directory for c in caches.values()],
        "cache_bytes": int(sum(c.total_bytes() for c in caches.values())),
    }


def reload_section_points(
    scans: Sequence,
    box: SectionBox,
    max_points: int = DEFAULT_MAX_POINTS,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    is_cancelled: Optional[Callable[[], bool]] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    use_cache: bool = True,
    cache_dir: Optional[str] = None,
) -> SectionReloadResult:
    """
    読み込み済みスキャン（間引き読込でよい）の読み込み元 E57 を流し読みし、断面ボックス内の点を表示座標で返す。

    :param scans: 読み込み中の E57ScanData の一覧（source_file / source_scan_index / pose / coord_mode を使う）
    :param box: 表示座標系の断面ボックス
    :param max_points: 返す点数の上限（0 以下で無制限）。超えた分はほぼ一様に間引く
    :param chunk_size: デコードのチャンク点数（キャッシュのチャンク粒度にもなる）
    :param is_cancelled: True を返すと InterruptedError で中断（チャンク境界で反映）
    :param progress_callback: (0.0〜1.0, メッセージ)
    :param use_cache: 読み直しキャッシュを使う／作る（書けなければ自動的にキャッシュ無しで続ける）
    :param cache_dir: キャッシュの置き場所（None なら元ファイルの横）
    """
    if not box.is_active():
        raise ValueError("断面が有効になっていません。X/Y/Z または斜め壁の断面を有効にしてから読み直してください。")
    scans = list(scans)
    if not scans:
        raise ValueError("点群が読み込まれていません。")
    validate_reload_sources(scans)

    t0 = time.perf_counter()
    want_colors = any(all(k in (getattr(s, "field_schemas", {}) or {}) for k in _RGB) for s in scans)
    want_ints = any("intensity" in (getattr(s, "field_schemas", {}) or {}) for s in scans)
    acc = _Accumulator(max_points, want_colors, want_ints)

    # 読むスキャンを決める（範囲がボックスに掛からないスキャンは飛ばす）
    plan: List[Tuple[int, object]] = []
    n_skipped = 0
    n_unsupported = 0
    for s_idx, s in enumerate(scans):
        schemas = getattr(s, "field_schemas", {}) or {}
        if getattr(s, "is_spherical_only", False) or not all(k in schemas for k in _CARTESIAN):
            log_warning(f"読み直し: スキャン「{s.name}」は直交座標を持たないため飛ばします。")
            n_unsupported += 1
            continue
        corners = scan_bounds_corners(s)
        if corners is not None and not box.may_intersect(corners):
            n_skipped += 1
            continue
        plan.append((s_idx, s))

    total_to_read = sum(int(getattr(s, "original_point_count", len(s.points))) for _, s in plan)
    n_read = 0
    n_in_box = 0
    done_weight = 0  # 進捗用（キャッシュから読んだスキャンも点数ぶん進める）
    caches: Dict[str, Optional[SectionCache]] = {}
    cache_used = 0
    cache_built = 0
    chunks_read = 0
    chunks_total = 0
    cache_dirs: List[str] = []

    def _check():
        if is_cancelled is not None and is_cancelled():
            raise InterruptedError("読み直しがユーザーによって中止されました。")

    def _report(msg: str):
        if progress_callback is not None:
            ratio = done_weight / total_to_read if total_to_read > 0 else 1.0
            progress_callback(min(1.0, ratio), msg)

    def _get_cache(source_file: str) -> Optional[SectionCache]:
        if not use_cache:
            return None
        key = os.path.abspath(source_file)
        if key not in caches:
            caches[key] = SectionCache.open(source_file, cache_dir=cache_dir, create=True)
            if caches[key] is not None and caches[key].directory not in cache_dirs:
                cache_dirs.append(caches[key].directory)
        return caches[key]

    for s_idx, s in plan:
        _check()
        schemas = s.field_schemas
        has_rgb = all(k in schemas for k in _RGB)
        has_int = "intensity" in schemas
        name = s.name or f"Scan {s.scan_index}"
        n_declared = int(getattr(s, "original_point_count", len(s.points)))
        scan_start_weight = done_weight
        cache = _get_cache(s.source_file)
        cached = cache.scan(int(s.source_scan_index)) if (cache is not None and cache.has_scan(int(s.source_scan_index))) else None

        if cached is not None:
            # --- キャッシュから: 箱に掛かるチャンクだけ読む ---
            m = len(cached.chunks)
            chunks_total += m
            hit = [k for k in range(m) if box.may_intersect(to_display_coordinates(s, cached.chunk_corners_local(k)))]
            _report(f"スキャン「{name}」をキャッシュから読み込み中 ({len(hit):,}/{m:,} チャンク, ボックス内 {n_in_box:,} 点)")
            for j, k in enumerate(hit):
                _check()
                local, rgb01, ints = cached.load_chunk(k)
                disp = to_display_coordinates(s, local)
                sel = np.flatnonzero(box.contains(disp))
                n_read += len(local)
                n_in_box += len(sel)
                if len(sel) > 0:
                    acc.add(
                        disp[sel],
                        rgb01[sel] if rgb01 is not None else None,
                        ints[sel] if ints is not None else None,
                        s_idx,
                    )
                del local, disp
                done_weight = scan_start_weight + int(n_declared * (j + 1) / max(1, len(hit)))
                _report(f"スキャン「{name}」をキャッシュから読み込み中 ({j + 1:,}/{len(hit):,} チャンク, ボックス内 {n_in_box:,} 点)")
            chunks_read += len(hit)
            cache_used += 1
            done_weight = scan_start_weight + n_declared
            continue

        # --- 元ファイルをデコード（可能ならキャッシュも書く） ---
        state = {"built": False, "building": False}
        _report(f"スキャン「{name}」を元ファイルから読み直し中 (0/{n_declared:,} 点)")
        scan_read = 0
        gen = _decode_scan_chunks(s, cache, chunk_size, state)
        try:
            for local, rgb01, ints, n_chunk in gen:
                _check()
                disp = to_display_coordinates(s, local)
                sel = np.flatnonzero(box.contains(disp))
                n_read += n_chunk
                scan_read += n_chunk
                n_in_box += len(sel)
                if len(sel) > 0:
                    acc.add(
                        disp[sel],
                        rgb01[sel] if rgb01 is not None else None,
                        ints[sel] if ints is not None else None,
                        s_idx,
                    )
                del local, disp
                done_weight = scan_start_weight + min(n_declared, scan_read)
                verb = "元ファイルから読み直し＋キャッシュ作成中" if state["building"] else "元ファイルから読み直し中"
                _report(f"スキャン「{name}」を{verb} ({scan_read:,}/{n_declared:,} 点, ボックス内 {n_in_box:,} 点)")
        finally:
            gen.close()  # 途中で止まった場合（中止・例外）は書きかけのキャッシュを消す
        if state["built"]:
            cache_built += 1
        done_weight = scan_start_weight + n_declared

    pts, cols, ints, sids = acc.finish()
    capped = acc.capped
    if acc.max_points > 0 and len(pts) > acc.max_points:
        step = int(math.ceil(len(pts) / acc.max_points))
        pts, sids = pts[::step], sids[::step]
        cols = cols[::step] if cols is not None else None
        ints = ints[::step] if ints is not None else None
        capped = True

    cache_bytes = sum(c.total_bytes() for c in caches.values() if c is not None)
    elapsed = time.perf_counter() - t0
    cache_note = ""
    if cache_used:
        cache_note += f", キャッシュから {cache_used} スキャン ({chunks_read:,}/{chunks_total:,} チャンク)"
    if cache_built:
        cache_note += f", キャッシュ作成 {cache_built} スキャン ({cache_bytes / 1e9:.2f}GB)"
    log_info(
        f"断面ボックスの読み直し: {box.describe()} → ボックス内 {n_in_box:,} 点 (表示 {len(pts):,} 点"
        f"{', 上限で間引き' if capped else ''}), 走査 {n_read:,} 点 / {len(plan)} スキャン"
        f" (範囲外で飛ばしたスキャン {n_skipped}){cache_note}, {elapsed:.1f} 秒"
    )
    return SectionReloadResult(
        points=np.ascontiguousarray(pts, dtype=np.float64),
        colors=np.ascontiguousarray(cols, dtype=np.float32) if cols is not None else None,
        intensities=np.ascontiguousarray(ints, dtype=np.float32) if ints is not None else None,
        scan_ids=np.ascontiguousarray(sids, dtype=np.int32),
        n_in_box=int(n_in_box),
        n_read=int(n_read),
        n_scans_read=len(plan),
        n_scans_skipped=int(n_skipped),
        n_scans_unsupported=int(n_unsupported),
        capped=bool(capped),
        elapsed_sec=float(elapsed),
        cache_used_scans=int(cache_used),
        cache_built_scans=int(cache_built),
        n_chunks_read=int(chunks_read),
        n_chunks_total=int(chunks_total),
        cache_dir=(cache_dirs[0] if cache_dirs else None),
        cache_bytes=int(cache_bytes),
    )
