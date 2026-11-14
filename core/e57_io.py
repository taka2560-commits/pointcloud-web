"""
E57ファイルの読み込みおよび書き込みを担当するモジュール（完全ロスレス・高精度維持対応版）
- libe57 C++バインディングを直接制御し、元の座標形式（FloatNode E57_DOUBLE / E57_SINGLE、ScaledIntegerNode）を完全保持
- スケーリング整数（ScaledIntegerNode）は生の整数値（int64）も保持し、丸め誤差ゼロ（バイナリ完全一致）を実現
- 元の全属性フィールド（RGB、反射強度、法線、タイムスタンプ、行列インデックス、マルチリターン等）を欠損なく保持
- ポーズ（回転クォータニオン・平行移動ベクトル）、スキャンヘッダ情報、ルート拡張名前空間を完全保持
- 出力時は一時ファイル経由のアトミック書き込みにより、エラー時のファイル破損（空ファイル化）を完全防止
"""

import math
import os
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pye57
from pye57 import libe57
from pye57.utils import get_fields, get_node
from pyquaternion import Quaternion

from core.logger import log_info, log_warning


@dataclass
class FieldSchema:
    """E57の単一点属性フィールドのプロトタイプ定義（型・精度・スケール・範囲）"""
    name: str
    node_type: str  # "Float", "ScaledInteger", "Integer"
    precision: Any = None  # libe57.E57_DOUBLE or libe57.E57_SINGLE
    minimum: Any = None
    maximum: Any = None
    scale: float = 1.0
    offset: float = 0.0
    default_value: Any = 0


def _optimal_integer_dtype(min_v: int, max_v: int) -> Any:
    """E57 IntegerNode の最小値・最大値に応じた最小メモリ消費の NumPy 整数型を返す"""
    if 0 <= min_v and max_v <= 255:
        return np.uint8
    if -128 <= min_v and max_v <= 127:
        return np.int8
    if 0 <= min_v and max_v <= 65535:
        return np.uint16
    if -32768 <= min_v and max_v <= 32767:
        return np.int16
    if 0 <= min_v and max_v <= 4294967295:
        return np.uint32
    if -2147483648 <= min_v and max_v <= 2147483647:
        return np.int32
    return np.int64


def _compute_points_signature(points: np.ndarray) -> Tuple[int, float, float, float, float]:
    """
    座標配列がインプレースで書き換えられたかをメモリ消費ゼロで厳密に検知するための軽量シグネチャを計算する
    """
    n = len(points)
    if n == 0:
        return (0, 0.0, 0.0, 0.0, 0.0)
    # 先頭・中央・末尾のサンプル値と全体の合計・最小・最大を組み合わせる
    mid = n // 2
    sample_sum = float(np.sum(points[0]) + np.sum(points[mid]) + np.sum(points[-1]))
    total_sum = float(np.sum(points))
    min_val = float(np.min(points))
    max_val = float(np.max(points))
    return (n, sample_sum, total_sum, min_val, max_val)


def _snapshot_structure_node(node: Any) -> Dict[str, Any]:
    """
    libe57のノードツリーをC++ファイルハンドルに依存しない純粋なPython辞書にディープコピーする。
    ImageFileがクローズされた後でもbad_weak_ptrを起こさず完全に元の構造を再構築可能にする。
    """
    if isinstance(node, libe57.FloatNode):
        return {
            "type": "Float",
            "value": float(node.value()),
            "precision": node.precision(),
            "minimum": float(node.minimum()),
            "maximum": float(node.maximum()),
        }
    elif isinstance(node, libe57.ScaledIntegerNode):
        return {
            "type": "ScaledInteger",
            "raw_value": int(node.rawValue()),
            "minimum": int(node.minimum()),
            "maximum": int(node.maximum()),
            "scale": float(node.scale()),
            "offset": float(node.offset()),
        }
    elif isinstance(node, libe57.IntegerNode):
        return {
            "type": "Integer",
            "value": int(node.value()),
            "minimum": int(node.minimum()),
            "maximum": int(node.maximum()),
        }
    elif isinstance(node, libe57.StringNode):
        return {
            "type": "String",
            "value": str(node.value()),
        }
    elif isinstance(node, libe57.StructureNode):
        children = {}
        for i in range(node.childCount()):
            child = get_node(node, i)
            name = child.elementName()
            # points と pointGroupingSchemes は点群データ本体・インデックス構造のため別途処理
            if name in ("points", "pointGroupingSchemes"):
                continue
            children[name] = _snapshot_structure_node(child)
        return {
            "type": "Structure",
            "children": children,
        }
    elif isinstance(node, libe57.VectorNode):
        items = []
        for i in range(node.childCount()):
            child = get_node(node, i)
            items.append(_snapshot_structure_node(child))
        return {
            "type": "Vector",
            "allow_hetero": bool(node.allowHeteroChildren()),
            "items": items,
        }
    return {"type": "Unknown"}


def _restore_structure_node(snapshot: Dict[str, Any], imf: libe57.ImageFile) -> Any:
    """スナップショット辞書からlibe57ノードツリーを元の型・精度通りに再構築する"""
    ntype = snapshot.get("type")
    if ntype == "Float":
        val = snapshot["value"]
        min_v = snapshot["minimum"]
        max_v = snapshot["maximum"]
        if val < min_v:
            min_v = val
        if val > max_v:
            max_v = val
        return libe57.FloatNode(
            imf,
            val,
            snapshot["precision"],
            min_v,
            max_v,
        )
    elif ntype == "ScaledInteger":
        return libe57.ScaledIntegerNode(
            imf,
            snapshot["raw_value"],
            snapshot["minimum"],
            snapshot["maximum"],
            snapshot["scale"],
            snapshot["offset"],
        )
    elif ntype == "Integer":
        val = snapshot["value"]
        min_v = snapshot["minimum"]
        max_v = snapshot["maximum"]
        if val < min_v:
            min_v = val
        if val > max_v:
            max_v = val
        return libe57.IntegerNode(
            imf,
            val,
            min_v,
            max_v,
        )
    elif ntype == "String":
        return libe57.StringNode(imf, snapshot["value"])
    elif ntype == "Structure":
        s_node = libe57.StructureNode(imf)
        for name, child_snap in snapshot.get("children", {}).items():
            restored_child = _restore_structure_node(child_snap, imf)
            if restored_child is not None:
                s_node.set(name, restored_child)
        return s_node
    elif ntype == "Vector":
        v_node = libe57.VectorNode(imf, snapshot.get("allow_hetero", True))
        for item_snap in snapshot.get("items", []):
            restored_item = _restore_structure_node(item_snap, imf)
            if restored_item is not None:
                v_node.append(restored_item)
        return v_node
    return None


class SafeScanHeader:
    """
    ImageFileクローズ後でもRuntimeError(bad_weak_ptr)を起こさずに
    安全にスキャンメタデータへアクセスできるヘッダクラス
    """

    def __init__(
        self,
        point_count: int,
        point_fields: List[str],
        scan_fields: List[str],
        has_pose_flag: bool,
        rotation: Optional[np.ndarray],
        translation: Optional[np.ndarray],
        meta_snapshot: Dict[str, Any],
    ):
        self._point_count = point_count
        self.point_fields = list(point_fields)
        self.scan_fields = list(scan_fields)
        self._has_pose = has_pose_flag
        self._rotation = rotation.copy() if rotation is not None else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self._translation = translation.copy() if translation is not None else np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.meta_snapshot = meta_snapshot

    @property
    def point_count(self) -> int:
        return self._point_count

    def has_pose(self) -> bool:
        return self._has_pose

    @property
    def rotation(self) -> np.ndarray:
        return self._rotation

    @property
    def rotation_matrix(self) -> np.ndarray:
        return Quaternion(self._rotation).rotation_matrix

    @property
    def translation(self) -> np.ndarray:
        return self._translation

    @property
    def guid(self) -> str:
        children = self.meta_snapshot.get("children", {})
        if "guid" in children:
            return children["guid"].get("value", "")
        return ""

    @property
    def name(self) -> str:
        children = self.meta_snapshot.get("children", {})
        if "name" in children:
            return children["name"].get("value", "")
        return ""


class E57ScanData:
    """単一スキャンの点群データ、属性フィールド、および元のスキーマ・メタデータを完全保持するクラス"""

    def __init__(
        self,
        scan_index: int,
        points: np.ndarray,
        raw_fields: Dict[str, np.ndarray],
        header: Any = None,
        rotation: Optional[np.ndarray] = None,
        translation: Optional[np.ndarray] = None,
        field_schemas: Optional[Dict[str, FieldSchema]] = None,
        raw_unscaled_fields: Optional[Dict[str, np.ndarray]] = None,
        meta_snapshot: Optional[Dict[str, Any]] = None,
        root_extensions: Optional[List[Tuple[str, str]]] = None,
        coordinate_metadata: Optional[str] = None,
        is_spherical_only: bool = False,
        name: Optional[str] = None,
        original_point_count: Optional[int] = None,
        is_subsampled: bool = False,
        source_file: Optional[str] = None,
        source_scan_index: Optional[int] = None,
        source_signature: Optional[Tuple[int, int]] = None,
    ):
        self.scan_index = scan_index
        # 読み込み元（全点転写保存で元ファイルを読み直すために使う）。合成データでは None
        self.source_file: Optional[str] = source_file
        self.source_scan_index: Optional[int] = source_scan_index
        self.source_signature: Optional[Tuple[int, int]] = source_signature  # (サイズ, 更新時刻 ns)
        pts_arr = np.asarray(points, dtype=np.float64)
        if pts_arr.ndim == 1 and pts_arr.size == 0:
            pts_arr = np.empty((0, 3), dtype=np.float64)
        self.points = pts_arr  # (N, 3) 形状の倍精度 XYZ 座標
        self.raw_fields = raw_fields  # 各種属性配列の辞書
        self.header = header
        self.rotation = np.asarray(rotation, dtype=np.float64) if rotation is not None else None
        self.translation = np.asarray(translation, dtype=np.float64) if translation is not None else None
        self.field_schemas: Dict[str, FieldSchema] = field_schemas or {}
        self.raw_unscaled_fields: Dict[str, np.ndarray] = raw_unscaled_fields or {}
        self.meta_snapshot: Optional[Dict[str, Any]] = meta_snapshot
        self.root_extensions: List[Tuple[str, str]] = root_extensions or []
        self.coordinate_metadata: Optional[str] = coordinate_metadata
        self.is_spherical_only: bool = is_spherical_only
        self._name: Optional[str] = name
        self.original_point_count: int = original_point_count if original_point_count is not None else len(self.points)
        self.is_subsampled: bool = bool(is_subsampled)
        # 読み込み時の軽量シグネチャ（座標値が外部でインプレース変更されたかの厳密判定用）。
        # 以前は座標の配列そのものへの参照（_initial_points_ref）も持っていたが、どこからも読まれていなかった。
        # 座標系を変えて読み込むと配列が差し替わるので、古い配列（1 点 24 バイト）を使わないまま持ち続けていた
        self._initial_points_signature = _compute_points_signature(self.points)
        # 取り込み時に適用された座標系変換モード（"cartesian" = 元ファイルのまま）。
        # core.processor.transform_scan_coordinate_system / restore_scan_original_coordinates が更新する。
        # 軸の入替・反転はシグネチャ（和・最小・最大）では検知できないため、変換済みであることを明示的に保持する。
        self.coord_mode: str = "cartesian"

    def mark_coordinates_modified(self) -> None:
        """
        座標が変更済みであることを明示的に記録する（以後 is_coordinates_modified() は必ず True）。
        軸入替のようにシグネチャが変わらない変換を行った直後に呼ぶこと。
        """
        self._initial_points_signature = None

    @property
    def name(self) -> str:
        """スキャン名（器械点名）を返す"""
        if self._name:
            return self._name
        if self.header is not None and hasattr(self.header, "name") and self.header.name:
            return str(self.header.name)
        if self.meta_snapshot and "children" in self.meta_snapshot:
            nm = self.meta_snapshot["children"].get("name", {}).get("value", "")
            if nm:
                return str(nm)
        return ""

    @name.setter
    def name(self, value: Optional[str]) -> None:
        """スキャン名（器械点名）を設定する"""
        self._name = str(value).strip() if value is not None else None

    @property
    def point_count(self) -> int:
        """点数を返す"""
        return len(self.points)

    @point_count.setter
    def point_count(self, value: Any) -> None:
        """
        防御的セッター: 外部代入による AttributeError('has no setter') を防止
        点数は常に len(self.points) が規定します
        """
        pass

    def is_coordinates_modified(self) -> bool:
        """
        座標配列 scan.points が読み込み時または抽出時から変更されたかを厳密に判定する。
        インプレース変更（scan.points[:, 0] += ...）も100%検知する。
        """
        n_points = len(self.points)
        if n_points == 0 or self.is_spherical_only:
            return False

        # 0. 取り込み時の座標系変換が未復元、または明示的に変更済みと記録されている場合は常に変更扱い
        if getattr(self, "coord_mode", "cartesian") not in ("cartesian", "", None):
            return True
        if self._initial_points_signature is None:
            return True

        # 1. 軽量シグネチャ比較（インプレース変更の高速・確実な検知）
        if _compute_points_signature(self.points) != self._initial_points_signature:
            return True

        # 2. ScaledInteger 座標の場合、生の整数配列から復元した値と一致するか確認
        for axis_idx, f_name in enumerate(("cartesianX", "cartesianY", "cartesianZ")):
            schema = self.field_schemas.get(f_name)
            if schema is not None and schema.node_type == "ScaledInteger":
                if f_name not in self.raw_unscaled_fields or len(self.raw_unscaled_fields[f_name]) != n_points:
                    return True
                expected = self.raw_unscaled_fields[f_name].astype(np.float64) * schema.scale + schema.offset
                if not np.allclose(self.points[:, axis_idx], expected, rtol=0, atol=max(1e-9, abs(schema.scale) * 0.5)):
                    return True

        # 3. raw_fields との整合確認
        if "cartesianX" in self.raw_fields and "cartesianY" in self.raw_fields and "cartesianZ" in self.raw_fields:
            if (
                len(self.raw_fields["cartesianX"]) != n_points
                or not np.allclose(self.points[:, 0], self.raw_fields["cartesianX"], rtol=0, atol=1e-6)
                or not np.allclose(self.points[:, 1], self.raw_fields["cartesianY"], rtol=0, atol=1e-6)
                or not np.allclose(self.points[:, 2], self.raw_fields["cartesianZ"], rtol=0, atol=1e-6)
            ):
                return True
        else:
            return True

        return False

    def validate(self) -> None:
        """
        書き込み前にスキャンデータの構造・型・数値の健全性を厳格に検証する。
        不整合や破損（NaN/Inf座標、次元不正など）があれば ValueError を送出する。
        """
        if not isinstance(self.points, np.ndarray):
            raise ValueError(f"スキャン #{self.scan_index}: points が NumPy 配列ではありません。")
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError(
                f"スキャン #{self.scan_index}: points の形状が不正です ({self.points.shape})。(N, 3) である必要があります。"
            )
        n_pts = len(self.points)
        if n_pts > 0 and not np.isfinite(self.points).all():
            bad_cnt = int(np.sum(~np.isfinite(self.points).all(axis=1)))
            raise ValueError(
                f"スキャン #{self.scan_index}: 座標データに非有限値 (NaN または Inf) が {bad_cnt} 点含まれています。"
            )
        if self.rotation is not None:
            rot = np.asarray(self.rotation, dtype=np.float64)
            if rot.shape != (4,) or not np.isfinite(rot).all():
                raise ValueError(f"スキャン #{self.scan_index}: rotation クォータニオンが不正です ({rot})。")
        if self.translation is not None:
            tr = np.asarray(self.translation, dtype=np.float64)
            if tr.shape != (3,) or not np.isfinite(tr).all():
                raise ValueError(f"スキャン #{self.scan_index}: translation ベクトルが不正です ({tr})。")

    def filter_by_indices(self, keep_indices: np.ndarray) -> "E57ScanData":
        """
        指定されたインデックスのみを抽出した新しいE57ScanDataを生成する。
        座標値の再計算や型変換を一切行わず、生の配列とスキーマをそのままスライスして引き継ぐ。
        """
        keep_indices = np.asarray(keep_indices, dtype=np.int64)
        n_orig = len(self.points)
        new_points = self.points[keep_indices].copy()

        new_fields = {}
        for key, val in self.raw_fields.items():
            if isinstance(val, np.ndarray) and len(val) == n_orig:
                # もし cartesianX/Y/Z が元の points とのビュー共有だった場合は new_points の列ビューを共有してメモリ節約
                if key == "cartesianX" and val.dtype == np.float64 and np.shares_memory(val, self.points):
                    new_fields[key] = new_points[:, 0]
                elif key == "cartesianY" and val.dtype == np.float64 and np.shares_memory(val, self.points):
                    new_fields[key] = new_points[:, 1]
                elif key == "cartesianZ" and val.dtype == np.float64 and np.shares_memory(val, self.points):
                    new_fields[key] = new_points[:, 2]
                else:
                    new_fields[key] = val[keep_indices].copy()
            else:
                new_fields[key] = val.copy() if hasattr(val, "copy") else val

        new_unscaled = {}
        for key, val in self.raw_unscaled_fields.items():
            if isinstance(val, np.ndarray) and len(val) == n_orig:
                new_unscaled[key] = val[keep_indices].copy()
            else:
                new_unscaled[key] = val.copy() if hasattr(val, "copy") else val

        new_header = None
        if isinstance(self.header, SafeScanHeader):
            new_header = SafeScanHeader(
                point_count=len(new_points),
                point_fields=self.header.point_fields,
                scan_fields=self.header.scan_fields,
                has_pose_flag=self.header.has_pose(),
                rotation=self.rotation,
                translation=self.translation,
                meta_snapshot=self.meta_snapshot or {},
            )
        else:
            new_header = self.header

        filtered = E57ScanData(
            scan_index=self.scan_index,
            points=new_points,
            raw_fields=new_fields,
            header=new_header,
            rotation=self.rotation.copy() if self.rotation is not None else None,
            translation=self.translation.copy() if self.translation is not None else None,
            field_schemas=dict(self.field_schemas),
            raw_unscaled_fields=new_unscaled,
            meta_snapshot=self.meta_snapshot,
            root_extensions=list(self.root_extensions),
            coordinate_metadata=self.coordinate_metadata,
            is_spherical_only=self.is_spherical_only,
            name=self._name,
            original_point_count=self.original_point_count,
            is_subsampled=self.is_subsampled,
            source_file=self.source_file,
            source_scan_index=self.source_scan_index,
            source_signature=self.source_signature,
        )
        # 取り込み時の座標系変換状態を引き継ぐ（抽出後も「変換済み」であることを失わないため）
        filtered.coord_mode = getattr(self, "coord_mode", "cartesian")
        if self._initial_points_signature is None:
            filtered.mark_coordinates_modified()
        return filtered


def _reconcile_read_count(
    scan_index: int,
    n_declared: int,
    n_read: int,
    raw_data: Dict[str, np.ndarray],
    raw_unscaled: Dict[str, np.ndarray],
) -> None:
    """
    E57 ヘッダーの宣言点数 (n_declared) と実際に読み込めた点数 (n_read) を照合する。
    - 一致: 何もしない
    - 短い: 事前確保済み配列 (np.empty) を n_read 長に切り詰めて未初期化領域を排除し、警告を記録
    - 0 点 (宣言 > 0): ファイル破損として例外
    """
    if n_declared <= 0 or n_read == n_declared:
        return
    if n_read <= 0:
        raise ValueError(
            f"スキャン #{scan_index}: ヘッダーでは {n_declared:,} 点が宣言されていますが 1 点も読み込めませんでした。"
            "E57 ファイルが破損している可能性があります。"
        )
    if n_read > n_declared:
        # libe57 はバッファ容量以上を返さないため通常は起こり得ないが、念のため宣言長に揃える
        n_read = n_declared
    log_warning(
        f"スキャン #{scan_index}: ヘッダー宣言 {n_declared:,} 点に対し実際に読み込めたのは {n_read:,} 点でした。"
        "読み込めた分だけを有効点として扱います（ファイル末尾が欠損している可能性があります）。"
    )
    for store in (raw_data, raw_unscaled):
        for key in list(store.keys()):
            arr = store[key]
            if isinstance(arr, np.ndarray) and len(arr) == n_declared:
                store[key] = arr[:n_read].copy()


def _extract_pose_safely(scan_node: libe57.StructureNode) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    スキャンノードからpose（rotationとtranslation）をXML要素の並び順に依存せず名前指定で安全・正確に取得する。
    poseが存在しない場合は (False, None, None) を返す。
    """
    if not scan_node.isDefined("pose"):
        return False, None, None

    pose_node = get_node(scan_node, "pose")
    rotation = None
    translation = None

    if pose_node.isDefined("rotation"):
        rot_node = get_node(pose_node, "rotation")
        w = float(get_node(rot_node, "w").value()) if rot_node.isDefined("w") else 1.0
        x = float(get_node(rot_node, "x").value()) if rot_node.isDefined("x") else 0.0
        y = float(get_node(rot_node, "y").value()) if rot_node.isDefined("y") else 0.0
        z = float(get_node(rot_node, "z").value()) if rot_node.isDefined("z") else 0.0
        rotation = np.array([w, x, y, z], dtype=np.float64)

    if pose_node.isDefined("translation"):
        tr_node = get_node(pose_node, "translation")
        tx = float(get_node(tr_node, "x").value()) if tr_node.isDefined("x") else 0.0
        ty = float(get_node(tr_node, "y").value()) if tr_node.isDefined("y") else 0.0
        tz = float(get_node(tr_node, "z").value()) if tr_node.isDefined("z") else 0.0
        translation = np.array([tx, ty, tz], dtype=np.float64)

    return True, rotation, translation


def file_signature(file_path: str) -> Tuple[int, int]:
    """ファイルの (サイズ, 更新時刻 ns)。読込後に元ファイルが差し替えられていないかの照合に使う"""
    st = os.stat(file_path)
    return (int(st.st_size), int(st.st_mtime_ns))


def subsample_step(total_points: int, point_budget: Optional[int]) -> int:
    """
    読み込み上限 point_budget に対する間引きの間隔（1 = 全点）。
    総点数が上限の 1.5 倍を超えるときだけ間引く（少し超えるだけなら全点を読む）。
    読み込み（iter_e57_scans）と、読み込み前のメモリの見積もり（core/memguard.py）が同じ式を使う。
    """
    if point_budget is not None and point_budget > 0 and total_points > int(point_budget * 1.5):
        return max(1, int(math.ceil(total_points / point_budget)))
    return 1


def subsampled_count(n_points: int, step: int) -> int:
    """step 点に 1 点で読んだときの点数（iter_e57_scans と同じ数え方: スキャンの先頭から step おき）"""
    n_points = max(0, int(n_points))
    return n_points if step <= 1 else (n_points + step - 1) // step


def read_e57_scans(
    file_path: str,
    progress_callback: Optional[Any] = None,
    point_budget: Optional[int] = None,
    chunk_size: int = 500000,
    scan_indices: Optional[Any] = None,
) -> List[E57ScanData]:
    """
    E57ファイルをロスレスで読み込み、各スキャンの点群データオブジェクトのリストを返す。
    大容量ファイル (数千万〜数億点) でもメモリ爆発 (30GB圧迫・ディスク100%スワップ) を完全に防止する
    チャンクストリーミング読み込み ＆ Point Budget サンプリング保護を内蔵。

    :param file_path: 入力E57ファイルのパス
    :param progress_callback: 進捗コールバック関数 (ratio: float, msg: str)
    :param point_budget: 最大表示点数 (None または <=0 の場合は全点ロスレス読み込み)
    :param chunk_size: ストリーミング読み込み時のチャンクサイズ (デフォルト50万点=約10MB)
    :param scan_indices: 読み込むスキャン番号の集合 (None なら全スキャン)。間引き率は全スキャンの総点数から決める
    :return: E57ScanData のリスト
    """
    return list(iter_e57_scans(file_path, progress_callback, point_budget, chunk_size, scan_indices))


def iter_e57_scans(
    file_path: str,
    progress_callback: Optional[Any] = None,
    point_budget: Optional[int] = None,
    chunk_size: int = 500000,
    scan_indices: Optional[Any] = None,
):
    """
    read_e57_scans のジェネレータ版。スキャンを 1 つ読み終えるごとに返すので、
    全点転写保存のように「1 スキャンずつ読んで処理して書き出す」用途でもメモリは 1 スキャン分で済む。
    各スキャンには読み込み元 (source_file / source_scan_index / source_signature) を記録する。
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"ファイルが見つかりません: {file_path}")

    if progress_callback:
        progress_callback(0.05, f"E57ヘッダー解析中: {os.path.basename(file_path)}")

    wanted = set(int(i) for i in scan_indices) if scan_indices is not None else None
    source_abs = os.path.abspath(file_path)
    source_sig = file_signature(file_path)
    imf = libe57.ImageFile(file_path, "r")
    try:
        root = imf.root()

        # ルートの拡張名前空間（extensions）をすべて取得
        root_extensions: List[Tuple[str, str]] = []
        for ext_idx in range(imf.extensionsCount()):
            prefix = imf.extensionsPrefix(ext_idx)
            uri = imf.extensionsUri(ext_idx)
            root_extensions.append((prefix, uri))

        coordinate_metadata = None
        if root.isDefined("coordinateMetadata"):
            coordinate_metadata = str(get_node(root, "coordinateMetadata").value())

        if not root.isDefined("data3D"):
            return

        data3d = get_node(root, "data3D")
        scan_count = data3d.childCount()

        # 大容量メモリ保護: 全スキャンの総点数を事前取得
        total_file_points = 0
        scan_counts: List[int] = []
        for s_idx in range(scan_count):
            s_node = get_node(data3d, s_idx)
            if s_node.isDefined("points"):
                cnt = get_node(s_node, "points").childCount()
            else:
                cnt = 0
            scan_counts.append(cnt)
            total_file_points += cnt

        # サンプリングステップの決定（予算指定あり、かつ総点数が予算の1.5倍超の場合）
        global_step = subsample_step(total_file_points, point_budget)

        for i in range(scan_count):
            if wanted is not None and i not in wanted:
                continue
            n_points = scan_counts[i]
            if progress_callback:
                progress_callback(
                    0.1 + (i / max(1, scan_count)) * 0.8,
                    f"スキャン #{i + 1}/{scan_count} を読み込み中 ({n_points:,} 点)...",
                )

            scan_node = get_node(data3d, i)
            scan_fields = get_fields(scan_node)
            meta_snapshot = _snapshot_structure_node(scan_node)
            has_pose_flag, rotation, translation = _extract_pose_safely(scan_node)

            if not scan_node.isDefined("points"):
                continue
            if n_points <= 0:
                # 点が 1 つも無いスキャン（ほかのソフトが書いた空のスキャン）は読み飛ばす。
                # 以前は「座標データが見つかりません」となり、ほかのスキャンに点があってもファイル全体を開けなかった
                log_info(f"スキャン #{i} は点が 0 点のため読み飛ばしました: {os.path.basename(file_path)}")
                continue

            points_node = get_node(scan_node, "points")
            proto_node = libe57.StructureNode(points_node.prototype())
            point_fields = get_fields(proto_node)

            field_schemas: Dict[str, FieldSchema] = {}
            scaled_int_fields: List[str] = []

            for f_idx in range(proto_node.childCount()):
                f_node = get_node(proto_node, f_idx)
                f_name = f_node.elementName()

                if isinstance(f_node, libe57.FloatNode):
                    prec = f_node.precision()
                    schema = FieldSchema(
                        name=f_name,
                        node_type="Float",
                        precision=prec,
                        minimum=float(f_node.minimum()),
                        maximum=float(f_node.maximum()),
                        default_value=float(f_node.value()),
                    )
                    field_schemas[f_name] = schema
                elif isinstance(f_node, libe57.ScaledIntegerNode):
                    schema = FieldSchema(
                        name=f_name,
                        node_type="ScaledInteger",
                        minimum=int(f_node.minimum()),
                        maximum=int(f_node.maximum()),
                        scale=float(f_node.scale()),
                        offset=float(f_node.offset()),
                        default_value=int(f_node.rawValue()),
                    )
                    field_schemas[f_name] = schema
                    scaled_int_fields.append(f_name)
                elif isinstance(f_node, libe57.IntegerNode):
                    min_v = int(f_node.minimum())
                    max_v = int(f_node.maximum())
                    schema = FieldSchema(
                        name=f_name,
                        node_type="Integer",
                        minimum=min_v,
                        maximum=max_v,
                        default_value=int(f_node.value()),
                    )
                    field_schemas[f_name] = schema

            raw_data: Dict[str, np.ndarray] = {}
            raw_unscaled: Dict[str, np.ndarray] = {}

            if n_points > 0 and len(field_schemas) > 0:
                if global_step > 1:
                    # ★ 大容量保護モード: 固定チャンクサイズ(約10MB)でストリーミング読み込みしながらサンプリング
                    # 5000万点・1億点あってもメモリ消費は数十MBのみに完全抑制
                    c_size = min(chunk_size, n_points)
                    chunk_buffers = libe57.VectorSourceDestBuffer()
                    chunk_raw_bufs: Dict[str, np.ndarray] = {}

                    for f_name, schema in field_schemas.items():
                        if schema.node_type == "Float":
                            dtype = np.float32 if schema.precision == libe57.E57_SINGLE else np.float64
                            arr = np.empty(c_size, dtype=dtype)
                            chunk_raw_bufs[f_name] = arr
                            chunk_buffers.append(libe57.SourceDestBuffer(imf, f_name, arr, c_size, True, True))
                        elif schema.node_type == "ScaledInteger":
                            arr = np.empty(c_size, dtype=np.int64)
                            chunk_raw_bufs[f_name] = arr
                            chunk_buffers.append(libe57.SourceDestBuffer(imf, f_name, arr, c_size, True, False))
                        elif schema.node_type == "Integer":
                            dtype = _optimal_integer_dtype(schema.minimum, schema.maximum)
                            arr = np.empty(c_size, dtype=dtype)
                            chunk_raw_bufs[f_name] = arr
                            chunk_buffers.append(libe57.SourceDestBuffer(imf, f_name, arr, c_size, True, True))

                    reader = points_node.reader(chunk_buffers)
                    sampled_raw: Dict[str, List[np.ndarray]] = {f: [] for f in field_schemas}
                    offset = 0
                    pts_read_so_far = 0

                    while True:
                        read_pts = reader.read()
                        if read_pts <= 0:
                            break
                        if offset < read_pts:
                            idx = np.arange(offset, read_pts, global_step)
                            for f_name, buf_arr in chunk_raw_bufs.items():
                                sampled_raw[f_name].append(buf_arr[:read_pts][idx].copy())
                            last_idx = int(idx[-1])
                            offset = (last_idx + global_step) - read_pts
                        else:
                            offset -= read_pts
                        pts_read_so_far += read_pts

                        if progress_callback:
                            base_prog = 0.1 + (i / max(1, scan_count)) * 0.8
                            scan_prog = (pts_read_so_far / max(1, n_points)) * (0.8 / max(1, scan_count))
                            progress_callback(
                                base_prog + scan_prog,
                                f"スキャン #{i + 1}/{scan_count} を保護読み込み中 ({pts_read_so_far:,}/{n_points:,}点)...",
                            )
                    reader.close()
                    del chunk_buffers, chunk_raw_bufs

                    # サンプリング結果をNumPy配列に結合
                    for f_name, schema in field_schemas.items():
                        pieces = sampled_raw.get(f_name, [])
                        if schema.node_type == "ScaledInteger":
                            raw_unscaled[f_name] = np.concatenate(pieces) if pieces else np.empty(0, dtype=np.int64)
                        elif schema.node_type == "Integer":
                            dtype = _optimal_integer_dtype(schema.minimum, schema.maximum)
                            raw_data[f_name] = np.concatenate(pieces) if pieces else np.empty(0, dtype=dtype)
                        else:
                            dtype = np.float32 if schema.precision == libe57.E57_SINGLE else np.float64
                            raw_data[f_name] = np.concatenate(pieces) if pieces else np.empty(0, dtype=dtype)
                    sampled_raw.clear()
                    del sampled_raw
                    import gc
                    gc.collect()

                else:
                    # 通常モード (global_step == 1): 全点ロスレス一括読み込み（最小限のメモリ型で事前確保）
                    buffers = libe57.VectorSourceDestBuffer()
                    for f_name, schema in field_schemas.items():
                        if schema.node_type == "Float":
                            dtype = np.float32 if schema.precision == libe57.E57_SINGLE else np.float64
                            arr = np.empty(n_points, dtype=dtype)
                            raw_data[f_name] = arr
                            buffers.append(libe57.SourceDestBuffer(imf, f_name, arr, n_points, True, True))
                        elif schema.node_type == "ScaledInteger":
                            unscaled_arr = np.empty(n_points, dtype=np.int64)
                            raw_unscaled[f_name] = unscaled_arr
                            buffers.append(libe57.SourceDestBuffer(imf, f_name, unscaled_arr, n_points, True, False))
                        elif schema.node_type == "Integer":
                            dtype = _optimal_integer_dtype(schema.minimum, schema.maximum)
                            arr = np.empty(n_points, dtype=dtype)
                            raw_data[f_name] = arr
                            buffers.append(libe57.SourceDestBuffer(imf, f_name, arr, n_points, True, True))

                    reader = points_node.reader(buffers)
                    total_read = 0
                    while True:
                        r_cnt = reader.read()
                        if r_cnt <= 0:
                            break
                        total_read += r_cnt
                    reader.close()

                    # 読み込めた点数をヘッダー宣言値と照合（np.empty で確保したバッファの未初期化領域が
                    # 座標として残らないよう、短く読めた場合は実読み込み長に切り詰める）
                    _reconcile_read_count(i, n_points, total_read, raw_data, raw_unscaled)

            # ScaledIntegerフィールドから倍精度float64配列を算出（cartesianX/Y/Z は直接 points 配列へ書き込んで一時メモリを節約）
            has_cartesian_all = all(
                (k in raw_data or k in raw_unscaled) for k in ("cartesianX", "cartesianY", "cartesianZ")
            )
            for f_name in scaled_int_fields:
                if has_cartesian_all and f_name in ("cartesianX", "cartesianY", "cartesianZ"):
                    continue
                sc = field_schemas[f_name]
                raw_data[f_name] = raw_unscaled[f_name].astype(np.float64) * sc.scale + sc.offset

            # 直交座標の取得（計算・表示用に倍精度float64の(N, 3)配列をピークメモリ最小で構築）
            is_spherical_only = False
            if has_cartesian_all:
                ref_arr = raw_unscaled.get("cartesianX") if "cartesianX" in raw_unscaled else raw_data.get("cartesianX")
                n_loaded = len(ref_arr) if ref_arr is not None else 0
                points = np.empty((n_loaded, 3), dtype=np.float64)
                if n_loaded > 0:
                    for axis_idx, f_name in enumerate(("cartesianX", "cartesianY", "cartesianZ")):
                        if f_name in raw_unscaled:
                            sc = field_schemas[f_name]
                            np.multiply(raw_unscaled[f_name], sc.scale, out=points[:, axis_idx], casting="unsafe")
                            if sc.offset != 0.0:
                                points[:, axis_idx] += sc.offset
                            raw_data[f_name] = points[:, axis_idx]
                        else:
                            orig_arr = raw_data[f_name]
                            points[:, axis_idx] = orig_arr
                            # E57_DOUBLE (float64) の場合は points の列ビューに置き換えて個別配列のメモリを即解放
                            # E57_SINGLE (float32) の場合は単精度ビット完全一致のため元の float32 配列を保持
                            if orig_arr.dtype == np.float64:
                                raw_data[f_name] = points[:, axis_idx]
            elif "sphericalRange" in raw_data:
                is_spherical_only = True
                r = np.asarray(raw_data["sphericalRange"], dtype=np.float64)
                az = np.asarray(raw_data.get("sphericalAzimuth", np.zeros_like(r)), dtype=np.float64)
                el = np.asarray(raw_data.get("sphericalElevation", np.zeros_like(r)), dtype=np.float64)

                cos_el = np.cos(el)
                x = r * cos_el * np.cos(az)
                y = r * cos_el * np.sin(az)
                z = r * np.sin(el)
                points = np.column_stack([x, y, z]) if len(x) > 0 else np.empty((0, 3), dtype=np.float64)
                del r, az, el, cos_el, x, y, z
            else:
                raise ValueError(f"スキャン #{i} に座標データ（cartesian/spherical）が見つかりません。")

            # 非有限座標 (NaN / Inf) の厳格サニタイズ: スキャナーの無効点等で NaN/Inf が混入している場合は安全に除外
            if len(points) > 0 and not np.isfinite(points).all():
                finite_mask = np.isfinite(points).all(axis=1)
                n_before_finite = len(points)
                points = points[finite_mask].copy()
                for k in list(raw_data.keys()):
                    val = raw_data[k]
                    if isinstance(val, np.ndarray) and len(val) == n_before_finite:
                        if k == "cartesianX" and val.dtype == np.float64:
                            raw_data[k] = points[:, 0]
                        elif k == "cartesianY" and val.dtype == np.float64:
                            raw_data[k] = points[:, 1]
                        elif k == "cartesianZ" and val.dtype == np.float64:
                            raw_data[k] = points[:, 2]
                        else:
                            raw_data[k] = val[finite_mask].copy()
                for k in list(raw_unscaled.keys()):
                    val = raw_unscaled[k]
                    if isinstance(val, np.ndarray) and len(val) == n_before_finite:
                        raw_unscaled[k] = val[finite_mask].copy()

            safe_header = SafeScanHeader(
                point_count=len(points),
                point_fields=point_fields,
                scan_fields=scan_fields,
                has_pose_flag=has_pose_flag,
                rotation=rotation,
                translation=translation,
                meta_snapshot=meta_snapshot,
            )

            scan_data = E57ScanData(
                scan_index=i,
                points=points,
                raw_fields=raw_data,
                header=safe_header,
                rotation=rotation,
                translation=translation,
                field_schemas=field_schemas,
                raw_unscaled_fields=raw_unscaled,
                meta_snapshot=meta_snapshot,
                root_extensions=root_extensions,
                coordinate_metadata=coordinate_metadata,
                is_spherical_only=is_spherical_only,
                original_point_count=n_points,
                is_subsampled=(global_step > 1),
                source_file=source_abs,
                source_scan_index=i,
                source_signature=source_sig,
            )
            yield scan_data
    finally:
        imf.close()

    if progress_callback:
        progress_callback(0.95, "点群データの統合中...")


def _open_points_node(imf: libe57.ImageFile, file_path: str, scan_index: int):
    """スキャンの points ノードとプロトタイプを返す（スキャンが無い・points が無いときは ValueError）"""
    root = imf.root()
    if not root.isDefined("data3D"):
        raise ValueError(f"E57 に data3D がありません: {file_path}")
    data3d = get_node(root, "data3D")
    if scan_index < 0 or scan_index >= data3d.childCount():
        raise ValueError(f"スキャン #{scan_index} は存在しません（スキャン数 {data3d.childCount()}）: {file_path}")
    scan_node = get_node(data3d, scan_index)
    if not scan_node.isDefined("points"):
        raise ValueError(f"スキャン #{scan_index} に points がありません: {file_path}")
    points_node = get_node(scan_node, "points")
    proto_node = libe57.StructureNode(points_node.prototype())
    return points_node, proto_node


def _schema_of_proto_field(f_node: Any) -> Optional[FieldSchema]:
    """プロトタイプの 1 フィールドを FieldSchema にする（Float / ScaledInteger / Integer 以外は None）"""
    f_name = f_node.elementName()
    if isinstance(f_node, libe57.FloatNode):
        return FieldSchema(
            name=f_name,
            node_type="Float",
            precision=f_node.precision(),
            minimum=float(f_node.minimum()),
            maximum=float(f_node.maximum()),
            default_value=float(f_node.value()),
        )
    if isinstance(f_node, libe57.ScaledIntegerNode):
        return FieldSchema(
            name=f_name,
            node_type="ScaledInteger",
            minimum=int(f_node.minimum()),
            maximum=int(f_node.maximum()),
            scale=float(f_node.scale()),
            offset=float(f_node.offset()),
            default_value=int(f_node.rawValue()),
        )
    if isinstance(f_node, libe57.IntegerNode):
        return FieldSchema(
            name=f_name,
            node_type="Integer",
            minimum=int(f_node.minimum()),
            maximum=int(f_node.maximum()),
            default_value=int(f_node.value()),
        )
    return None


def e57_point_field_schemas(file_path: str, scan_index: int) -> Dict[str, FieldSchema]:
    """
    ファイルの 1 スキャンの点フィールド定義（名前 → FieldSchema）をヘッダだけ読んで返す。
    読み込み済みスキャンの field_schemas は座標系変換で cartesianX/Y/Z が入れ替わっていることがあるので、
    元ファイルそのものの定義が要るとき（読み直しキャッシュ）はこちらを使う。
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"ファイルが見つかりません: {file_path}")
    imf = libe57.ImageFile(file_path, "r")
    try:
        _, proto_node = _open_points_node(imf, file_path, scan_index)
        schemas: Dict[str, FieldSchema] = {}
        for f_idx in range(proto_node.childCount()):
            sch = _schema_of_proto_field(get_node(proto_node, f_idx))
            if sch is not None:
                schemas[sch.name] = sch
        return schemas
    finally:
        imf.close()


@dataclass
class ScanLayout:
    """1 スキャンの点数・点フィールド定義・pose（ヘッダだけ読んで得る。読み込みに要るメモリの見積もりに使う）"""
    n_points: int
    schemas: Dict[str, FieldSchema]
    rotation: Optional[np.ndarray] = None
    translation: Optional[np.ndarray] = None


def e57_scan_layouts(file_path: str) -> List[ScanLayout]:
    """
    ファイルの全スキャンの配置（点数・点フィールド定義・pose）を、ヘッダだけ読んで返す。
    点は読まないので、十数 GB のファイルでも一瞬で済む（実データ 16.8GB・5.4 億点で 0.01 秒）。
    points の無いスキャンは点数 0 として返す（読み込みは、そのスキャンを飛ばす）。
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"ファイルが見つかりません: {file_path}")
    out: List[ScanLayout] = []
    imf = libe57.ImageFile(file_path, "r")
    try:
        root = imf.root()
        if not root.isDefined("data3D"):
            return out
        data3d = get_node(root, "data3D")
        for i in range(data3d.childCount()):
            scan_node = get_node(data3d, i)
            if not scan_node.isDefined("points"):
                out.append(ScanLayout(0, {}))
                continue
            points_node = get_node(scan_node, "points")
            proto_node = libe57.StructureNode(points_node.prototype())
            schemas: Dict[str, FieldSchema] = {}
            for f_idx in range(proto_node.childCount()):
                sch = _schema_of_proto_field(get_node(proto_node, f_idx))
                if sch is not None:
                    schemas[sch.name] = sch
            _, rotation, translation = _extract_pose_safely(scan_node)
            out.append(ScanLayout(int(points_node.childCount()), schemas, rotation, translation))
        return out
    finally:
        imf.close()


def iter_e57_point_chunks(
    file_path: str,
    scan_index: int,
    fields: Optional[Any] = None,
    chunk_size: int = 1_000_000,
    raw_fields: Optional[Any] = None,
):
    """
    1 スキャンの点をチャンク単位で流し読みするジェネレータ（断面ボックスの元ファイル読み直しなど、
    全点をメモリに載せずに走査したい用途向け）。

    各チャンクは {フィールド名: ndarray} の辞書。ScaledInteger は scale/offset を掛けた float64、
    Float / Integer はファイルの dtype のまま。配列は毎回新しく確保するので呼び出し側で保持してよい。
    raw_fields に挙げた ScaledInteger フィールドは scale/offset を掛けずに生の int64 で返す
    （読み直しキャッシュが生整数のまま保持して、全点読込と同じ式で座標を復元するため）。
    fields に無いフィールド名、またはプロトタイプに無いフィールド名は黙って無視する
    （呼び出し側は読み込み済みスキャンの field_schemas で有無を確認できる）。
    fields が None なら cartesianX/Y/Z のみ。スキャンが無い・points が無いときは ValueError。
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"ファイルが見つかりません: {file_path}")
    wanted = list(fields) if fields is not None else ["cartesianX", "cartesianY", "cartesianZ"]
    raw_set = set(raw_fields) if raw_fields is not None else set()
    chunk_size = max(1, int(chunk_size))

    imf = libe57.ImageFile(file_path, "r")
    reader = None
    try:
        points_node, proto_node = _open_points_node(imf, file_path, scan_index)
        n_points = points_node.childCount()
        if n_points <= 0:
            return
        c_size = min(chunk_size, n_points)

        buffers = libe57.VectorSourceDestBuffer()
        bufs: Dict[str, np.ndarray] = {}
        scaled: Dict[str, Tuple[float, float]] = {}
        for f_idx in range(proto_node.childCount()):
            f_node = get_node(proto_node, f_idx)
            f_name = f_node.elementName()
            if f_name not in wanted:
                continue
            if isinstance(f_node, libe57.FloatNode):
                dtype = np.float32 if f_node.precision() == libe57.E57_SINGLE else np.float64
                arr = np.empty(c_size, dtype=dtype)
                buffers.append(libe57.SourceDestBuffer(imf, f_name, arr, c_size, True, True))
            elif isinstance(f_node, libe57.ScaledIntegerNode):
                arr = np.empty(c_size, dtype=np.int64)
                buffers.append(libe57.SourceDestBuffer(imf, f_name, arr, c_size, True, False))
                if f_name not in raw_set:
                    scaled[f_name] = (float(f_node.scale()), float(f_node.offset()))
            elif isinstance(f_node, libe57.IntegerNode):
                arr = np.empty(c_size, dtype=_optimal_integer_dtype(int(f_node.minimum()), int(f_node.maximum())))
                buffers.append(libe57.SourceDestBuffer(imf, f_name, arr, c_size, True, True))
            else:
                continue
            bufs[f_name] = arr
        if not bufs:
            return

        reader = points_node.reader(buffers)
        while True:
            read_pts = reader.read()
            if read_pts <= 0:
                break
            chunk: Dict[str, np.ndarray] = {}
            for f_name, arr in bufs.items():
                if f_name in scaled:
                    sc, off = scaled[f_name]
                    vals = arr[:read_pts].astype(np.float64)
                    vals *= sc
                    if off != 0.0:
                        vals += off
                    chunk[f_name] = vals
                else:
                    chunk[f_name] = arr[:read_pts].copy()
            yield chunk
    finally:
        if reader is not None:
            try:
                reader.close()
            except Exception:
                pass
        imf.close()


def _write_single_scan_lossless(imf: libe57.ImageFile, data3d: libe57.VectorNode, scan: E57ScanData, scan_idx: int) -> None:
    """
    単一スキャンデータを元のスキーマ・精度・メタデータを完全保持したままImageFileに書き込む
    """
    # 書き込み前の構造・数値整合性チェック
    scan.validate()
    n_points = len(scan.points)

    # 座標が直接書き換えられたかどうかを厳密に判定（インプレース変更も検知）
    coords_modified = scan.is_coordinates_modified()

    # バウンディングボックス (cartesianBounds) を最新化すべきか判定
    # 点数が変化した場合、座標が変更された場合、または元ヘッダに cartesianBounds がない場合は実点群から正確に再計算する
    has_orig_bounds = bool(
        scan.meta_snapshot
        and isinstance(scan.meta_snapshot.get("children"), dict)
        and "cartesianBounds" in scan.meta_snapshot["children"]
    )
    should_update_bounds = (not scan.is_spherical_only) and (
        not has_orig_bounds or coords_modified or n_points != scan.original_point_count
    )

    # 1. スキャン構造ノードの構築（元のメタデータがあれば完全復元）
    # libe57 の StructureNode.set は既存キーの上書き時に ErrorSetTwice を送出するため、更新対象キーは復元前に調整する
    if scan.meta_snapshot is not None and scan.meta_snapshot.get("type") == "Structure":
        snap_children = dict(scan.meta_snapshot.get("children", {}))
        if should_update_bounds and "cartesianBounds" in snap_children:
            del snap_children["cartesianBounds"]
        if scan._name:
            snap_children["name"] = {"type": "String", "value": str(scan._name)}
        adjusted_snapshot = {"type": "Structure", "children": snap_children}
        scan_node = _restore_structure_node(adjusted_snapshot, imf)
    else:
        scan_node = libe57.StructureNode(imf)
        scan_node.set("guid", libe57.StringNode(imf, "{%s}" % uuid.uuid4()))
        scan_node.set("name", libe57.StringNode(imf, scan.name if scan.name else f"Scan {scan_idx}"))

    # 2. ポーズ（rotation / translation）の設定
    # 元のmeta_snapshotにposeが含まれておらず、かつscan.rotation/translationが明示指定されている場合のみ追加
    if not scan_node.isDefined("pose") and (scan.rotation is not None or scan.translation is not None):
        rot = scan.rotation if scan.rotation is not None else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        tr = scan.translation if scan.translation is not None else np.array([0.0, 0.0, 0.0], dtype=np.float64)
        pose_node = libe57.StructureNode(imf)
        rot_node = libe57.StructureNode(imf)
        rot_node.set("w", libe57.FloatNode(imf, float(rot[0]), libe57.E57_DOUBLE, -1.0, 1.0))
        rot_node.set("x", libe57.FloatNode(imf, float(rot[1]), libe57.E57_DOUBLE, -1.0, 1.0))
        rot_node.set("y", libe57.FloatNode(imf, float(rot[2]), libe57.E57_DOUBLE, -1.0, 1.0))
        rot_node.set("z", libe57.FloatNode(imf, float(rot[3]), libe57.E57_DOUBLE, -1.0, 1.0))
        pose_node.set("rotation", rot_node)
        tr_node = libe57.StructureNode(imf)
        tr_node.set("x", libe57.FloatNode(imf, float(tr[0]), libe57.E57_DOUBLE, float(libe57.E57_DOUBLE_MIN), float(libe57.E57_DOUBLE_MAX)))
        tr_node.set("y", libe57.FloatNode(imf, float(tr[1]), libe57.E57_DOUBLE, float(libe57.E57_DOUBLE_MIN), float(libe57.E57_DOUBLE_MAX)))
        tr_node.set("z", libe57.FloatNode(imf, float(tr[2]), libe57.E57_DOUBLE, float(libe57.E57_DOUBLE_MIN), float(libe57.E57_DOUBLE_MAX)))
        pose_node.set("translation", tr_node)
        scan_node.set("pose", pose_node)

    # 3. 書き込むデータ辞書の整理
    data_to_write: Dict[str, np.ndarray] = {}
    for k, v in scan.raw_fields.items():
        if isinstance(v, np.ndarray) and len(v) == n_points:
            data_to_write[k] = v

    if not scan.is_spherical_only:
        if coords_modified or "cartesianX" not in data_to_write or "cartesianY" not in data_to_write or "cartesianZ" not in data_to_write:
            data_to_write["cartesianX"] = scan.points[:, 0] if n_points > 0 else np.empty(0, dtype=np.float64)
            data_to_write["cartesianY"] = scan.points[:, 1] if n_points > 0 else np.empty(0, dtype=np.float64)
            data_to_write["cartesianZ"] = scan.points[:, 2] if n_points > 0 else np.empty(0, dtype=np.float64)

    # 4. cartesianBounds の設定・更新（実点群の正確な最小・最大値を記録）
    if not scan.is_spherical_only and not scan_node.isDefined("cartesianBounds"):
        bbox_node = libe57.StructureNode(imf)
        if n_points > 0:
            x_min, x_max = float(np.min(scan.points[:, 0])), float(np.max(scan.points[:, 0]))
            y_min, y_max = float(np.min(scan.points[:, 1])), float(np.max(scan.points[:, 1]))
            z_min, z_max = float(np.min(scan.points[:, 2])), float(np.max(scan.points[:, 2]))
        else:
            x_min = x_max = y_min = y_max = z_min = z_max = 0.0
        bbox_node.set("xMinimum", libe57.FloatNode(imf, x_min, libe57.E57_DOUBLE))
        bbox_node.set("xMaximum", libe57.FloatNode(imf, x_max, libe57.E57_DOUBLE))
        bbox_node.set("yMinimum", libe57.FloatNode(imf, y_min, libe57.E57_DOUBLE))
        bbox_node.set("yMaximum", libe57.FloatNode(imf, y_max, libe57.E57_DOUBLE))
        bbox_node.set("zMinimum", libe57.FloatNode(imf, z_min, libe57.E57_DOUBLE))
        bbox_node.set("zMaximum", libe57.FloatNode(imf, z_max, libe57.E57_DOUBLE))
        scan_node.set("cartesianBounds", bbox_node)

    # 5. プロトタイプ（points_prototype）と書き込みバッファの構築
    points_prototype = libe57.StructureNode(imf)

    # フィールド順序の決定：元のスキーマ順序を最優先し、新規フィールドがあれば後ろに追加
    ordered_fields: List[str] = []
    for f_name in scan.field_schemas.keys():
        if f_name in data_to_write or f_name in scan.raw_unscaled_fields:
            ordered_fields.append(f_name)
    for f_name in data_to_write.keys():
        if f_name not in ordered_fields:
            ordered_fields.append(f_name)

    chunk_size = max(1, min(n_points, 5000000)) if n_points > 0 else 1
    chunk_buffers: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}  # field -> (chunk_array, full_source_array)
    vec_buffers = libe57.VectorSourceDestBuffer()

    for f_name in ordered_fields:
        schema = scan.field_schemas.get(f_name)

        if schema is not None and schema.node_type == "ScaledInteger":
            # ScaledIntegerNode の完全復元
            is_coord_field = f_name in ("cartesianX", "cartesianY", "cartesianZ")
            use_raw_unscaled = (
                (not (coords_modified and is_coord_field))
                and f_name in scan.raw_unscaled_fields
                and len(scan.raw_unscaled_fields[f_name]) == n_points
            )
            if use_raw_unscaled:
                src_arr = np.ascontiguousarray(scan.raw_unscaled_fields[f_name], dtype=np.int64)
                min_v = schema.minimum
                max_v = schema.maximum
                if n_points > 0:
                    min_v = min(min_v, int(np.min(src_arr)))
                    max_v = max(max_v, int(np.max(src_arr)))
                default_v = int(min(max(schema.default_value, min_v), max_v))
                proto_field = libe57.ScaledIntegerNode(
                    imf, default_v, min_v, max_v, schema.scale, schema.offset
                )
                points_prototype.set(f_name, proto_field)
                chunk_arr = np.empty(chunk_size, dtype=np.int64)
                chunk_buffers[f_name] = (chunk_arr, src_arr)
                vec_buffers.append(libe57.SourceDestBuffer(imf, f_name, chunk_arr, chunk_size, True, False))
            else:
                # 座標が直接変更された場合はスケールを使って書き込む
                src_arr = np.ascontiguousarray(data_to_write[f_name], dtype=np.float64)
                raw_calc = np.round((src_arr - schema.offset) / schema.scale).astype(np.int64) if n_points > 0 else np.empty(0, dtype=np.int64)
                min_v = schema.minimum
                max_v = schema.maximum
                if n_points > 0:
                    min_v = min(min_v, int(np.min(raw_calc)))
                    max_v = max(max_v, int(np.max(raw_calc)))
                default_v = int(min(max(schema.default_value, min_v), max_v))
                proto_field = libe57.ScaledIntegerNode(
                    imf, default_v, min_v, max_v, schema.scale, schema.offset
                )
                points_prototype.set(f_name, proto_field)
                chunk_arr = np.empty(chunk_size, dtype=np.int64)
                chunk_buffers[f_name] = (chunk_arr, raw_calc)
                vec_buffers.append(libe57.SourceDestBuffer(imf, f_name, chunk_arr, chunk_size, True, False))

        elif schema is not None and schema.node_type == "Float":
            prec = schema.precision
            dtype = np.float32 if prec == libe57.E57_SINGLE else np.float64
            src_arr = np.ascontiguousarray(data_to_write[f_name], dtype=dtype)

            min_v = float(schema.minimum)
            max_v = float(schema.maximum)
            if n_points > 0:
                arr_min = float(np.min(src_arr))
                arr_max = float(np.max(src_arr))
                if arr_min < min_v:
                    min_v = arr_min
                if arr_max > max_v:
                    max_v = arr_max
            default_v = float(min(max(schema.default_value, min_v), max_v))
            proto_field = libe57.FloatNode(imf, default_v, prec, min_v, max_v)
            points_prototype.set(f_name, proto_field)
            chunk_arr = np.empty(chunk_size, dtype=dtype)
            chunk_buffers[f_name] = (chunk_arr, src_arr)
            vec_buffers.append(libe57.SourceDestBuffer(imf, f_name, chunk_arr, chunk_size, True, True))

        elif schema is not None and schema.node_type == "Integer":
            src_arr = np.ascontiguousarray(data_to_write[f_name], dtype=np.int64)
            min_v = int(schema.minimum)
            max_v = int(schema.maximum)
            if n_points > 0:
                arr_min = int(np.min(src_arr))
                arr_max = int(np.max(src_arr))
                if arr_min < min_v:
                    min_v = arr_min
                if arr_max > max_v:
                    max_v = arr_max
            default_v = int(min(max(schema.default_value, min_v), max_v))
            proto_field = libe57.IntegerNode(imf, default_v, min_v, max_v)
            points_prototype.set(f_name, proto_field)
            chunk_arr = np.empty(chunk_size, dtype=np.int64)
            chunk_buffers[f_name] = (chunk_arr, src_arr)
            vec_buffers.append(libe57.SourceDestBuffer(imf, f_name, chunk_arr, chunk_size, True, True))

        else:
            # スキーマがない新規フィールドの自動判定（座標や浮動小数点は劣化防止のためE57_DOUBLEを使用）
            arr = data_to_write[f_name]
            if f_name in ("colorRed", "colorGreen", "colorBlue"):
                src_arr = np.ascontiguousarray(arr, dtype=np.int64)
                min_v = 0
                max_v = max(255, int(np.max(src_arr)) if n_points > 0 else 255)
                points_prototype.set(f_name, libe57.IntegerNode(imf, 0, min_v, max_v))
                chunk_arr = np.empty(chunk_size, dtype=np.int64)
                chunk_buffers[f_name] = (chunk_arr, src_arr)
                vec_buffers.append(libe57.SourceDestBuffer(imf, f_name, chunk_arr, chunk_size, True, True))
            elif np.issubdtype(arr.dtype, np.integer):
                src_arr = np.ascontiguousarray(arr, dtype=np.int64)
                min_v = int(np.min(src_arr)) if n_points > 0 else 0
                max_v = int(np.max(src_arr)) if n_points > 0 else 0
                points_prototype.set(f_name, libe57.IntegerNode(imf, min_v, min_v, max_v))
                chunk_arr = np.empty(chunk_size, dtype=np.int64)
                chunk_buffers[f_name] = (chunk_arr, src_arr)
                vec_buffers.append(libe57.SourceDestBuffer(imf, f_name, chunk_arr, chunk_size, True, True))
            else:
                # 浮動小数点フィールド：元のdtypeがfloat32の場合のみE57_SINGLE、それ以外（float64等）はE57_DOUBLEで精度を完全保持
                is_single = (arr.dtype == np.float32) and f_name not in ("cartesianX", "cartesianY", "cartesianZ")
                prec = libe57.E57_SINGLE if is_single else libe57.E57_DOUBLE
                dtype = np.float32 if is_single else np.float64
                src_arr = np.ascontiguousarray(arr, dtype=dtype)
                min_v = float(np.min(src_arr)) if n_points > 0 else 0.0
                max_v = float(np.max(src_arr)) if n_points > 0 else 0.0
                center_v = (min_v + max_v) / 2.0
                points_prototype.set(f_name, libe57.FloatNode(imf, center_v, prec, min_v, max_v))
                chunk_arr = np.empty(chunk_size, dtype=dtype)
                chunk_buffers[f_name] = (chunk_arr, src_arr)
                vec_buffers.append(libe57.SourceDestBuffer(imf, f_name, chunk_arr, chunk_size, True, True))

    codecs = libe57.VectorNode(imf, True)
    compressed_points = libe57.CompressedVectorNode(imf, points_prototype, codecs)
    scan_node.set("points", compressed_points)
    data3d.append(scan_node)

    if n_points > 0:
        writer = compressed_points.writer(vec_buffers)
        current_index = 0
        while current_index < n_points:
            current_chunk = min(n_points - current_index, chunk_size)
            for f_name, (c_arr, s_arr) in chunk_buffers.items():
                c_arr[:current_chunk] = s_arr[current_index : current_index + current_chunk]
            writer.write(current_chunk)
            current_index += current_chunk
        writer.close()


def _verify_written_e57_file(file_path: str, expected_scans: List[Any]) -> None:
    """
    書き出し完了直後のE57ファイルを実際にリードモードで再オープンし、
    ヘッダ構造・スキャン数・各スキャンのレコード点数・必須座標属性が1点の狂いもなく書き込まれているかを厳格検証する。
    expected_scans は E57ScanData か、(点数, is_spherical_only) のタプル（ストリーム書込ではスキャン本体を保持しないため）。
    """
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        raise RuntimeError(f"出力ベリファイ失敗: 出力ファイルが空または存在しません ({file_path})")

    expected: List[Tuple[int, bool]] = []
    for item in expected_scans:
        if isinstance(item, E57ScanData):
            expected.append((len(item.points), bool(item.is_spherical_only)))
        else:
            expected.append((int(item[0]), bool(item[1])))

    verify_imf = libe57.ImageFile(file_path, "r")
    try:
        root = verify_imf.root()
        if not root.isDefined("data3D"):
            raise RuntimeError("出力ベリファイ失敗: E57ルートに data3D ノードが存在しません。")
        data3d = get_node(root, "data3D")
        actual_scan_count = data3d.childCount()
        if actual_scan_count != len(expected):
            raise RuntimeError(
                f"出力ベリファイ失敗: スキャン数が一致しません (期待: {len(expected)}, 実際: {actual_scan_count})"
            )

        for idx, (expected_pts, spherical_only) in enumerate(expected):
            s_node = get_node(data3d, idx)
            if not s_node.isDefined("points"):
                raise RuntimeError(f"出力ベリファイ失敗: スキャン #{idx} に points ノードが存在しません。")
            pts_node = get_node(s_node, "points")
            actual_pts = pts_node.childCount()
            if actual_pts != expected_pts:
                raise RuntimeError(
                    f"出力ベリファイ失敗: スキャン #{idx} の書き出し点数が一致しません (期待: {expected_pts:,} 点, 実際: {actual_pts:,} 点)"
                )
            proto = libe57.StructureNode(pts_node.prototype())
            fields = set(get_fields(proto))
            if not spherical_only:
                if not {"cartesianX", "cartesianY", "cartesianZ"}.issubset(fields):
                    raise RuntimeError(
                        f"出力ベリファイ失敗: スキャン #{idx} に必須座標フィールド (cartesianX/Y/Z) が欠落しています。"
                    )
            else:
                if "sphericalRange" not in fields:
                    raise RuntimeError(
                        f"出力ベリファイ失敗: 球面スキャン #{idx} に必須フィールド (sphericalRange) が欠落しています。"
                    )
    finally:
        verify_imf.close()


class E57StreamWriter:
    """
    スキャンを 1 つずつ追記できる E57 ロスレス書込。
    1 億点超の全点転写保存では全スキャンをメモリに持てないため、
    「1 スキャン読む → 判定を写す → add_scan() で書く」を繰り返す。

    - 一時ファイルへ書き、close() で再オープン検証 (_verify_written_e57_file) に合格したときだけ
      出力先へアトミック置換する。途中で abort() / 例外なら一時ファイルを消し、既存の出力先は無傷。
    - 拡張名前空間は add_scan() のたびに未登録分を追加する。
    - coordinateMetadata は最初に見つかったスキャンの値を使う。
    """

    def __init__(self, file_path: str):
        self.file_path = os.path.abspath(file_path)
        output_dir = os.path.dirname(self.file_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        self.temp_path = f"{self.file_path}.tmp_{os.getpid()}_{uuid.uuid4().hex[:6]}.e57"
        self._expected: List[Tuple[int, bool]] = []
        self._added_prefixes = {""}
        self._coord_meta: Optional[str] = None
        self._closed = False

        self._imf = libe57.ImageFile(self.temp_path, "w")
        try:
            root = self._imf.root()
            self._imf.extensionsAdd("", libe57.E57_V1_0_URI)
            root.set("formatName", libe57.StringNode(self._imf, "ASTM E57 3D Imaging Data File"))
            root.set("guid", libe57.StringNode(self._imf, "{%s}" % uuid.uuid4()))
            root.set("versionMajor", libe57.IntegerNode(self._imf, libe57.E57_FORMAT_MAJOR))
            root.set("versionMinor", libe57.IntegerNode(self._imf, libe57.E57_FORMAT_MINOR))
            root.set("e57LibraryVersion", libe57.StringNode(self._imf, libe57.E57_LIBRARY_ID))
            creation_date_time = libe57.StructureNode(self._imf)
            creation_date_time.set("dateTimeValue", libe57.FloatNode(self._imf, 0.0, libe57.E57_DOUBLE))
            creation_date_time.set("isAtomicClockReferenced", libe57.IntegerNode(self._imf, 0))
            root.set("creationDateTime", creation_date_time)
            self._data3d = libe57.VectorNode(self._imf, True)
            root.set("data3D", self._data3d)
            root.set("images2D", libe57.VectorNode(self._imf, True))
        except Exception:
            self.abort()
            raise

    @property
    def scan_count(self) -> int:
        return len(self._expected)

    @property
    def point_count(self) -> int:
        return sum(n for n, _ in self._expected)

    def _register_scan_meta(self, root_extensions, coordinate_metadata: Optional[str]) -> None:
        """スキャンの拡張名前空間のうち未登録のものを足し、最初に見つかった coordinateMetadata を覚える"""
        if self._closed:
            raise RuntimeError("E57StreamWriter は既に閉じられています。")
        for prefix, uri in root_extensions or []:
            if prefix not in self._added_prefixes:
                try:
                    self._imf.extensionsAdd(prefix, uri)
                except Exception:
                    pass
                self._added_prefixes.add(prefix)
        if self._coord_meta is None and coordinate_metadata:
            self._coord_meta = coordinate_metadata

    def _register_written(self, n_points: int, is_spherical_only: bool) -> None:
        """書き終えたスキャンを、close() の再オープン検証の期待値に載せる"""
        self._expected.append((int(n_points), bool(is_spherical_only)))

    def add_scan(self, scan: E57ScanData) -> None:
        """
        スキャンを検証してから追記する。失敗したら一時ファイルを消して例外を再送出する。
        1 スキャンを全点メモリに載せられないときは core/e57_stream.py の ScanStreamSink で少しずつ書く。
        """
        if self._closed:
            raise RuntimeError("E57StreamWriter は既に閉じられています。")
        try:
            scan.validate()
            self._register_scan_meta(scan.root_extensions, scan.coordinate_metadata)
            _write_single_scan_lossless(self._imf, self._data3d, scan, len(self._expected))
            self._register_written(len(scan.points), scan.is_spherical_only)
        except Exception:
            self.abort()
            raise

    def close(self) -> None:
        """書き終えて検証し、合格なら出力先へアトミック置換する"""
        if self._closed:
            return
        try:
            if self.scan_count == 0:
                raise ValueError("書き込むスキャンデータが空です。")
            self._imf.root().set("coordinateMetadata", libe57.StringNode(self._imf, self._coord_meta or ""))
            self._imf.close()
            self._imf = None
            _verify_written_e57_file(self.temp_path, self._expected)
            os.replace(self.temp_path, self.file_path)
            self._closed = True
        except Exception:
            self.abort()
            raise

    def abort(self) -> None:
        """書込を中止して一時ファイルを消す（既存の出力先には触れない）"""
        self._closed = True
        if getattr(self, "_imf", None) is not None:
            try:
                self._imf.close()
            except Exception:
                pass
            self._imf = None
        if os.path.exists(self.temp_path):
            try:
                os.remove(self.temp_path)
            except Exception:
                pass

    def __enter__(self) -> "E57StreamWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()


def write_e57_scans(file_path: str, scans: List[E57ScanData]) -> None:
    """
    複数のスキャンデータを新しいE57ファイルにロスレス（元の形式・精度・座標値を完全保持）で書き出す。
    1. 書き込み前の全スキャンデータ厳格バリデーション (validate)
    2. 一時ファイルへの書き込み
    3. 書き込み後の再オープン自動ベリファイ (_verify_written_e57_file)
    4. 検証合格後のアトミック置換 (os.replace)
    により、破損ファイルや点数欠損ファイルの生成を100%防止する。実体は E57StreamWriter。

    :param file_path: 出力先E57ファイルのパス
    :param scans: 書き込むE57ScanDataオブジェクトのリスト
    """
    if not scans:
        raise ValueError("書き込むスキャンデータが空です。")

    # 1. 書き込み前の事前バリデーション（1 つでも不正なら一時ファイルを作る前に止める）
    for scan in scans:
        scan.validate()

    writer = E57StreamWriter(file_path)
    for scan in scans:
        writer.add_scan(scan)
    writer.close()
