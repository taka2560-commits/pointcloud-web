# -*- coding: utf-8 -*-
"""
1 スキャンを全点メモリに載せずに「元ファイルから読む → 残す点を選ぶ → 出力へ書く」を流す部品（全点で保存用）

なぜ要るか
    全点で保存（gui/worker_thread.py: FullResolutionSaveWorker）は、以前は 1 スキャンの全点を read_e57_scans で
    一度に読み、filter_by_indices で写してから E57StreamWriter.add_scan で書いていた。1 点あたり約 165 バイトを
    使うので、利用者の普段のファイル（5.4 億点・1 スキャン）では約 89GB になり、32GB の PC では保存できない
    （2026-11-12 の点検で判明）。ここではスキャンを 100 万点ずつ読み、選んだ点をそのまま出力のスキャンへ追記する。
    使うメモリはスキャンの点数に依らず、チャンク数個分（数百 MB）で済む。

出力は以前の方式と同じ
    - フィールドの並び・型・精度・プロトタイプの範囲・scale/offset、生の整数値、スキャンのメタデータ（pose・名前・
      guid など）は、以前の方式（全点を読んでから書く）と同じになる。tests/test_fullres_stream.py が 1 点ずつ照合する。
    - 座標が NaN / Inf の点は、読み込み（read_e57_scans）と同じく除く。
    - cartesianBounds は、残した点の範囲で書く（全点を残したときは元の値をそのまま書く）。点を書き終えてから
      スキャンに足す（libe57 はスキャンを data3D に繋いだ後でも子ノードを足せる）。

元ファイルの値がプロトタイプの範囲を外れている場合
    以前の方式は書く値の最小・最大でプロトタイプの範囲を広げていたが、流す方式では書き始める前に範囲を決める必要が
    ある。範囲外の値を見つけたら E57SchemaBoundsError を出す。呼び出し側は、先にスキャンを 1 回走査して範囲を
    測り（measure_field_ranges）、その範囲で書き直す。
"""

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np
from pye57 import libe57
from pye57.utils import get_node

from core.e57_io import (
    E57StreamWriter,
    FieldSchema,
    _extract_pose_safely,
    _open_points_node,
    _optimal_integer_dtype,
    _restore_structure_node,
    _schema_of_proto_field,
    _snapshot_structure_node,
)

CARTESIAN_FIELDS = ("cartesianX", "cartesianY", "cartesianZ")
DEFAULT_STREAM_CHUNK = 1_000_000


class E57SchemaBoundsError(ValueError):
    """元ファイルの値が、そのファイル自身のプロトタイプの範囲を外れていた（範囲を測ってから書き直す）"""


@dataclass
class ScanStreamTemplate:
    """1 スキャンを流して書くための、元ファイルのスキャンの定義（ヘッダだけ読んで作る）"""

    file_path: str
    scan_index: int
    n_declared: int
    field_schemas: "OrderedDict[str, FieldSchema]"
    meta_snapshot: Optional[dict]
    root_extensions: List[Tuple[str, str]]
    coordinate_metadata: Optional[str]
    is_spherical_only: bool
    rotation: Optional[np.ndarray] = None
    translation: Optional[np.ndarray] = None
    # 範囲を測って書き直すとき（E57SchemaBoundsError の後）に使う、フィールドごとの (最小, 最大)
    bounds_override: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    @property
    def has_orig_bounds(self) -> bool:
        snap = self.meta_snapshot or {}
        return isinstance(snap.get("children"), dict) and "cartesianBounds" in snap["children"]


def scan_stream_template(file_path: str, scan_index: int) -> ScanStreamTemplate:
    """元ファイルの 1 スキャンの定義を、点を読まずに返す"""
    imf = libe57.ImageFile(file_path, "r")
    try:
        root = imf.root()
        root_extensions = [(imf.extensionsPrefix(i), imf.extensionsUri(i)) for i in range(imf.extensionsCount())]
        coordinate_metadata = str(get_node(root, "coordinateMetadata").value()) if root.isDefined("coordinateMetadata") else None
        points_node, proto_node = _open_points_node(imf, file_path, scan_index)
        scan_node = get_node(get_node(root, "data3D"), scan_index)
        meta_snapshot = _snapshot_structure_node(scan_node)
        _has_pose, rotation, translation = _extract_pose_safely(scan_node)
        schemas: "OrderedDict[str, FieldSchema]" = OrderedDict()
        for f_idx in range(proto_node.childCount()):
            sch = _schema_of_proto_field(get_node(proto_node, f_idx))
            if sch is not None:
                schemas[sch.name] = sch
        has_cartesian = all(f in schemas for f in CARTESIAN_FIELDS)
        if not has_cartesian and "sphericalRange" not in schemas:
            raise ValueError(f"スキャン #{scan_index} に座標データ（cartesian/spherical）が見つかりません。")
        return ScanStreamTemplate(
            file_path=file_path,
            scan_index=int(scan_index),
            n_declared=int(points_node.childCount()),
            field_schemas=schemas,
            meta_snapshot=meta_snapshot,
            root_extensions=root_extensions,
            coordinate_metadata=coordinate_metadata,
            is_spherical_only=not has_cartesian,
            rotation=rotation,
            translation=translation,
        )
    finally:
        imf.close()


def _local_xyz(tpl: ScanStreamTemplate, raw: Dict[str, np.ndarray], n: int) -> np.ndarray:
    """チャンクの生の値からローカル座標 (n, 3) を作る（read_e57_scans と同じ式なので、同じ値になる）"""
    pts = np.empty((n, 3), dtype=np.float64)
    if not tpl.is_spherical_only:
        for axis, f in enumerate(CARTESIAN_FIELDS):
            sch = tpl.field_schemas[f]
            if sch.node_type == "ScaledInteger":
                np.multiply(raw[f], sch.scale, out=pts[:, axis], casting="unsafe")
                if sch.offset != 0.0:
                    pts[:, axis] += sch.offset
            else:
                pts[:, axis] = raw[f]
        return pts

    def as_float(name: str) -> np.ndarray:
        if name not in raw:
            return np.zeros(n, dtype=np.float64)
        sch = tpl.field_schemas[name]
        if sch.node_type == "ScaledInteger":
            return raw[name].astype(np.float64) * sch.scale + sch.offset
        return np.asarray(raw[name], dtype=np.float64)

    r, az, el = as_float("sphericalRange"), as_float("sphericalAzimuth"), as_float("sphericalElevation")
    cos_el = np.cos(el)
    pts[:, 0] = r * cos_el * np.cos(az)
    pts[:, 1] = r * cos_el * np.sin(az)
    pts[:, 2] = r * np.sin(el)
    return pts


def iter_scan_stream_chunks(
    tpl: ScanStreamTemplate,
    chunk_size: int = DEFAULT_STREAM_CHUNK,
    is_cancelled: Optional[Callable[[], bool]] = None,
) -> Iterator[Tuple[Dict[str, np.ndarray], np.ndarray]]:
    """
    1 スキャンをチャンクで読み、(生の値の辞書, ローカル座標 (n, 3)) を返すジェネレータ。
    生の値は ScaledInteger が int64 の生の整数、Float / Integer はファイルの型のまま（read_e57_scans と同じ型）。
    座標が NaN / Inf の点は除いてある。配列は毎回新しく作るので、呼び出し側で持っていてよい。
    """
    chunk_size = max(1, int(chunk_size))
    imf = libe57.ImageFile(tpl.file_path, "r")
    reader = None
    total = 0
    try:
        points_node, _proto = _open_points_node(imf, tpl.file_path, tpl.scan_index)
        n_points = int(points_node.childCount())
        if n_points <= 0:
            return
        c_size = min(chunk_size, n_points)
        buffers = libe57.VectorSourceDestBuffer()
        bufs: Dict[str, np.ndarray] = {}
        for name, sch in tpl.field_schemas.items():
            if sch.node_type == "Float":
                arr = np.empty(c_size, dtype=np.float32 if sch.precision == libe57.E57_SINGLE else np.float64)
                buffers.append(libe57.SourceDestBuffer(imf, name, arr, c_size, True, True))
            elif sch.node_type == "ScaledInteger":
                arr = np.empty(c_size, dtype=np.int64)
                buffers.append(libe57.SourceDestBuffer(imf, name, arr, c_size, True, False))
            else:
                arr = np.empty(c_size, dtype=_optimal_integer_dtype(sch.minimum, sch.maximum))
                buffers.append(libe57.SourceDestBuffer(imf, name, arr, c_size, True, True))
            bufs[name] = arr
        reader = points_node.reader(buffers)
        while True:
            if is_cancelled is not None and is_cancelled():
                raise InterruptedError("全点で保存がユーザーによって中止されました。")
            n = reader.read()
            if n <= 0:
                break
            total += n
            raw = {name: arr[:n].copy() for name, arr in bufs.items()}
            pts = _local_xyz(tpl, raw, n)
            if not np.isfinite(pts).all():
                ok = np.isfinite(pts).all(axis=1)
                pts = pts[ok]
                raw = {name: arr[ok] for name, arr in raw.items()}
            yield raw, pts
        if total == 0:
            raise ValueError(
                f"スキャン #{tpl.scan_index}: ヘッダーでは {n_points:,} 点が宣言されていますが 1 点も読み込めませんでした。"
                "E57 ファイルが破損している可能性があります。"
            )
    finally:
        if reader is not None:
            try:
                reader.close()
            except Exception:
                pass
        imf.close()


def measure_field_ranges(
    tpl: ScanStreamTemplate,
    chunk_size: int = DEFAULT_STREAM_CHUNK,
    is_cancelled: Optional[Callable[[], bool]] = None,
) -> Dict[str, Tuple[float, float]]:
    """スキャンを 1 回走査して、フィールドごとの (最小, 最大) を測る（プロトタイプの範囲を外れた値があったときの書き直し用）"""
    lo: Dict[str, float] = {}
    hi: Dict[str, float] = {}
    for raw, _pts in iter_scan_stream_chunks(tpl, chunk_size, is_cancelled):
        for name, arr in raw.items():
            if len(arr) == 0:
                continue
            a, b = np.min(arr), np.max(arr)
            lo[name] = a if name not in lo else min(lo[name], a)
            hi[name] = b if name not in hi else max(hi[name], b)
    return {name: (lo[name], hi[name]) for name in lo}


class ScanStreamSink:
    """
    出力ファイル（E57StreamWriter）の中に 1 スキャンを開き、選んだ点を少しずつ追記する。
    write() を何回でも呼び、最後に finish() を呼ぶ。失敗したら出力ファイルごと捨てる（E57StreamWriter.abort）。
    """

    def __init__(self, writer: E57StreamWriter, tpl: ScanStreamTemplate, name_override: Optional[str] = None,
                 capacity: int = DEFAULT_STREAM_CHUNK):
        self.writer = writer
        self.tpl = tpl
        self.n_written = 0
        self._finished = False
        self._capacity = max(1, int(capacity))
        self._lo = np.full(3, np.inf)
        self._hi = np.full(3, -np.inf)
        imf = writer._imf
        writer._register_scan_meta(tpl.root_extensions, tpl.coordinate_metadata)

        # 1. スキャンの構造ノード（元のメタデータを復元。cartesianBounds は点を書き終えてから決める）
        snap = tpl.meta_snapshot
        if snap is not None and snap.get("type") == "Structure":
            children = dict(snap.get("children", {}))
            if not tpl.is_spherical_only:
                children.pop("cartesianBounds", None)
            if name_override:
                children["name"] = {"type": "String", "value": str(name_override)}
            self.scan_node = _restore_structure_node({"type": "Structure", "children": children}, imf)
        else:
            import uuid

            self.scan_node = libe57.StructureNode(imf)
            self.scan_node.set("guid", libe57.StringNode(imf, "{%s}" % uuid.uuid4()))
            self.scan_node.set("name", libe57.StringNode(imf, name_override or f"Scan {tpl.scan_index}"))
        # pose は元のスキャンのものがメタデータに入っている（元ファイルから読んだスキャンなので、足すものは無い）

        # 2. プロトタイプと書き込みバッファ（元の並び・型・精度・scale/offset のまま）
        proto = libe57.StructureNode(imf)
        self._buffers = libe57.VectorSourceDestBuffer()
        self._bufs: Dict[str, np.ndarray] = {}
        self._limits: Dict[str, Tuple[float, float]] = {}
        cap = self._capacity
        for name, sch in tpl.field_schemas.items():
            min_v, max_v = sch.minimum, sch.maximum
            if name in tpl.bounds_override:
                a, b = tpl.bounds_override[name]
                min_v, max_v = min(min_v, a), max(max_v, b)
            if sch.node_type == "ScaledInteger":
                min_v, max_v = int(min_v), int(max_v)
                default_v = int(min(max(sch.default_value, min_v), max_v))
                proto.set(name, libe57.ScaledIntegerNode(imf, default_v, min_v, max_v, sch.scale, sch.offset))
                arr = np.empty(cap, dtype=np.int64)
                self._buffers.append(libe57.SourceDestBuffer(imf, name, arr, cap, True, False))
            elif sch.node_type == "Float":
                min_v, max_v = float(min_v), float(max_v)
                default_v = float(min(max(sch.default_value, min_v), max_v))
                proto.set(name, libe57.FloatNode(imf, default_v, sch.precision, min_v, max_v))
                arr = np.empty(cap, dtype=np.float32 if sch.precision == libe57.E57_SINGLE else np.float64)
                self._buffers.append(libe57.SourceDestBuffer(imf, name, arr, cap, True, True))
            else:
                min_v, max_v = int(min_v), int(max_v)
                default_v = int(min(max(sch.default_value, min_v), max_v))
                proto.set(name, libe57.IntegerNode(imf, default_v, min_v, max_v))
                arr = np.empty(cap, dtype=np.int64)
                self._buffers.append(libe57.SourceDestBuffer(imf, name, arr, cap, True, True))
            self._bufs[name] = arr
            self._limits[name] = (min_v, max_v)

        cv = libe57.CompressedVectorNode(imf, proto, libe57.VectorNode(imf, True))
        self.scan_node.set("points", cv)
        writer._data3d.append(self.scan_node)
        self._cv_writer = cv.writer(self._buffers)

    def write(self, raw: Dict[str, np.ndarray], local_xyz: np.ndarray, idx: np.ndarray) -> int:
        """チャンクの idx 番目の点を書き足す。書いた点数を返す"""
        if self._finished:
            raise RuntimeError("このスキャンは書き終えています。")
        idx = np.asarray(idx, dtype=np.int64)
        m_all = len(idx)
        if m_all == 0:
            return 0
        sel = {name: raw[name][idx] for name in self._bufs}
        for name, vals in sel.items():
            lo, hi = self._limits[name]
            vmin, vmax = np.min(vals), np.max(vals)
            if vmin < lo or vmax > hi:  # NaN は比べると偽になるので、以前の方式と同じく範囲の判定から外れる
                raise E57SchemaBoundsError(
                    f"スキャン #{self.tpl.scan_index} のフィールド {name} に、プロトタイプの範囲 [{lo}, {hi}] を外れた値"
                    f" [{vmin}, {vmax}] があります。"
                )
        if not self.tpl.is_spherical_only:
            sub = local_xyz[idx]
            self._lo = np.minimum(self._lo, sub.min(axis=0))
            self._hi = np.maximum(self._hi, sub.max(axis=0))
        for start in range(0, m_all, self._capacity):
            m = min(self._capacity, m_all - start)
            for name, buf in self._bufs.items():
                buf[:m] = sel[name][start:start + m]
            self._cv_writer.write(m)
        self.n_written += m_all
        return m_all

    def finish(self) -> int:
        """スキャンを閉じ、cartesianBounds を足して、出力ファイルの点検用の記録に載せる。書いた点数を返す"""
        if self._finished:
            return self.n_written
        self._cv_writer.close()
        self._finished = True
        tpl = self.tpl
        if not tpl.is_spherical_only:
            imf = self.writer._imf
            if tpl.has_orig_bounds and self.n_written == tpl.n_declared:
                # 全点を残した: 元の値をそのまま書く（以前の方式と同じ）
                bounds = _restore_structure_node(tpl.meta_snapshot["children"]["cartesianBounds"], imf)
            else:
                bounds = libe57.StructureNode(imf)
                if self.n_written > 0:
                    lo, hi = self._lo, self._hi
                else:
                    lo = hi = np.zeros(3)
                for axis, a in enumerate("xyz"):
                    bounds.set(f"{a}Minimum", libe57.FloatNode(imf, float(lo[axis]), libe57.E57_DOUBLE))
                    bounds.set(f"{a}Maximum", libe57.FloatNode(imf, float(hi[axis]), libe57.E57_DOUBLE))
            self.scan_node.set("cartesianBounds", bounds)
        self.writer._register_written(self.n_written, tpl.is_spherical_only)
        return self.n_written

    def close_quietly(self) -> None:
        """失敗したときの後始末（出力ファイルは呼び出し側が E57StreamWriter.abort で捨てる）"""
        if not self._finished:
            self._finished = True
            try:
                self._cv_writer.close()
            except Exception:
                pass
