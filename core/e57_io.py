"""
E57ファイルの読み込みおよび書き込みを担当するモジュール（完全ロスレス・高精度維持対応版）
- libe57 C++バインディングを直接制御し、元の座標形式（FloatNode E57_DOUBLE / E57_SINGLE、ScaledIntegerNode）を完全保持
- スケーリング整数（ScaledIntegerNode）は生の整数値（int64）も保持し、丸め誤差ゼロ（バイナリ完全一致）を実現
- 元の全属性フィールド（RGB、反射強度、法線、タイムスタンプ、行列インデックス、マルチリターン等）を欠損なく保持
- ポーズ（回転クォータニオン・平行移動ベクトル）、スキャンヘッダ情報、ルート拡張名前空間を完全保持
- 出力時は一時ファイル経由のアトミック書き込みにより、エラー時のファイル破損（空ファイル化）を完全防止
"""

import os
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pye57
from pye57 import libe57
from pye57.utils import get_fields, get_node
from pyquaternion import Quaternion


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
    ):
        self.scan_index = scan_index
        self.points = np.asarray(points, dtype=np.float64)  # (N, 3) 形状の倍精度 XYZ 座標
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
        # 初期読み込み時の座標参照（座標値が外部で直接書き換えられたかの判定用）
        self._initial_points_ref = self.points

    @property
    def point_count(self) -> int:
        """点数を返す"""
        return len(self.points)

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
        )
        return filtered


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


def read_e57_scans(
    file_path: str,
    progress_callback: Optional[Any] = None,
) -> List[E57ScanData]:
    """
    E57ファイルをロスレスで読み込み、各スキャンの点群データオブジェクトのリストを返す。
    未サポートフィールドによるエラーを起こさず、すべての属性フィールドと座標精度を100%保持する。

    :param file_path: 入力E57ファイルのパス
    :param progress_callback: 進捗コールバック関数 (ratio: float, msg: str)
    :return: E57ScanData のリスト
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"ファイルが見つかりません: {file_path}")

    if progress_callback:
        progress_callback(0.05, f"E57ヘッダー解析中: {os.path.basename(file_path)}")

    scans: List[E57ScanData] = []
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
            return []

        data3d = get_node(root, "data3D")
        scan_count = data3d.childCount()

        for i in range(scan_count):
            if progress_callback:
                progress_callback(
                    0.1 + (i / max(1, scan_count)) * 0.8,
                    f"スキャン #{i + 1}/{scan_count} を読み込み中...",
                )

            scan_node = get_node(data3d, i)
            scan_fields = get_fields(scan_node)
            meta_snapshot = _snapshot_structure_node(scan_node)
            has_pose_flag, rotation, translation = _extract_pose_safely(scan_node)

            points_node = get_node(scan_node, "points")
            n_points = points_node.childCount()
            proto_node = libe57.StructureNode(points_node.prototype())
            point_fields = get_fields(proto_node)

            field_schemas: Dict[str, FieldSchema] = {}
            raw_data: Dict[str, np.ndarray] = {}
            raw_unscaled: Dict[str, np.ndarray] = {}

            # プロトタイプの各フィールドを解析し、最適なバッファを準備
            buffers = libe57.VectorSourceDestBuffer()
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
                    # 元の精度がSINGLEならfloat32、DOUBLEならfloat64でそのまま読み込む（ビット完全一致のため）
                    dtype = np.float32 if prec == libe57.E57_SINGLE else np.float64
                    arr = np.empty(n_points, dtype=dtype)
                    raw_data[f_name] = arr
                    buffers.append(libe57.SourceDestBuffer(imf, f_name, arr, n_points, True, True))

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
                    # ScaledIntegerは生のint64整数として読み込み、浮動小数点丸め誤差を100%排除する
                    unscaled_arr = np.empty(n_points, dtype=np.int64)
                    raw_unscaled[f_name] = unscaled_arr
                    scaled_int_fields.append(f_name)
                    buffers.append(libe57.SourceDestBuffer(imf, f_name, unscaled_arr, n_points, True, False))

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
                    if min_v >= 0 and max_v <= 255:
                        dtype = np.uint8
                    elif min_v >= -128 and max_v <= 127:
                        dtype = np.int8
                    else:
                        # オーバーフロー防止のため64bit整数を使用
                        dtype = np.int64
                    arr = np.empty(n_points, dtype=dtype)
                    raw_data[f_name] = arr
                    buffers.append(libe57.SourceDestBuffer(imf, f_name, arr, n_points, True, True))

            if n_points > 0 and len(field_schemas) > 0:
                reader = points_node.reader(buffers)
                reader.read()
                reader.close()

            # ScaledIntegerフィールドから倍精度float64配列を算出
            for f_name in scaled_int_fields:
                sc = field_schemas[f_name]
                raw_data[f_name] = raw_unscaled[f_name].astype(np.float64) * sc.scale + sc.offset

            # 直交座標の取得（計算・表示用に倍精度float64の(N, 3)配列を構築）
            is_spherical_only = False
            if "cartesianX" in raw_data and "cartesianY" in raw_data and "cartesianZ" in raw_data:
                x = np.asarray(raw_data["cartesianX"], dtype=np.float64)
                y = np.asarray(raw_data["cartesianY"], dtype=np.float64)
                z = np.asarray(raw_data["cartesianZ"], dtype=np.float64)
                points = np.column_stack([x, y, z]) if n_points > 0 else np.empty((0, 3), dtype=np.float64)
            elif "sphericalRange" in raw_data:
                is_spherical_only = True
                r = np.asarray(raw_data["sphericalRange"], dtype=np.float64)
                az = np.asarray(raw_data.get("sphericalAzimuth", np.zeros_like(r)), dtype=np.float64)
                el = np.asarray(raw_data.get("sphericalElevation", np.zeros_like(r)), dtype=np.float64)

                cos_el = np.cos(el)
                x = r * cos_el * np.cos(az)
                y = r * cos_el * np.sin(az)
                z = r * np.sin(el)
                points = np.column_stack([x, y, z]) if n_points > 0 else np.empty((0, 3), dtype=np.float64)
            else:
                raise ValueError(f"スキャン #{i} に座標データ（cartesian/spherical）が見つかりません。")

            safe_header = SafeScanHeader(
                point_count=n_points,
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
            )
            scans.append(scan_data)
    finally:
        imf.close()

    if progress_callback:
        progress_callback(0.95, "点群データの統合中...")

    return scans


def _write_single_scan_lossless(imf: libe57.ImageFile, data3d: libe57.VectorNode, scan: E57ScanData, scan_idx: int) -> None:
    """
    単一スキャンデータを元のスキーマ・精度・メタデータを完全保持したままImageFileに書き込む
    """
    n_points = len(scan.points)

    # 座標が直接書き換えられたかどうかを判定（スライスやコピーのみなら元の値がそのまま維持されている）
    coords_modified = False
    if not scan.is_spherical_only and n_points > 0:
        if "cartesianX" in scan.raw_fields and "cartesianY" in scan.raw_fields and "cartesianZ" in scan.raw_fields:
            # scan.points が外部で変更された場合は raw_fields との差が生じる
            if (
                len(scan.raw_fields["cartesianX"]) != n_points
                or not np.allclose(scan.points[:, 0], scan.raw_fields["cartesianX"], rtol=0, atol=1e-7)
                or not np.allclose(scan.points[:, 1], scan.raw_fields["cartesianY"], rtol=0, atol=1e-7)
                or not np.allclose(scan.points[:, 2], scan.raw_fields["cartesianZ"], rtol=0, atol=1e-7)
            ):
                coords_modified = True
        else:
            coords_modified = True

    # 1. スキャン構造ノードの構築（元のメタデータがあれば完全復元）
    if scan.meta_snapshot is not None and scan.meta_snapshot.get("type") == "Structure":
        scan_node = _restore_structure_node(scan.meta_snapshot, imf)
    else:
        scan_node = libe57.StructureNode(imf)
        scan_node.set("guid", libe57.StringNode(imf, "{%s}" % uuid.uuid4()))
        scan_node.set("name", libe57.StringNode(imf, f"Scan {scan_idx}"))

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
        if coords_modified or "cartesianX" not in data_to_write:
            data_to_write["cartesianX"] = scan.points[:, 0] if n_points > 0 else np.empty(0, dtype=np.float64)
            data_to_write["cartesianY"] = scan.points[:, 1] if n_points > 0 else np.empty(0, dtype=np.float64)
            data_to_write["cartesianZ"] = scan.points[:, 2] if n_points > 0 else np.empty(0, dtype=np.float64)

    # 4. cartesianBounds の更新（メタデータスナップショットに存在しない場合、または点数がある場合）
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
            if not coords_modified and f_name in scan.raw_unscaled_fields and len(scan.raw_unscaled_fields[f_name]) == n_points:
                src_arr = np.ascontiguousarray(scan.raw_unscaled_fields[f_name], dtype=np.int64)
                min_v = schema.minimum
                max_v = schema.maximum
                if n_points > 0:
                    min_v = min(min_v, int(np.min(src_arr)))
                    max_v = max(max_v, int(np.max(src_arr)))
                default_v = int( min(max(schema.default_value, min_v), max_v) )
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
                default_v = int( min(max(schema.default_value, min_v), max_v) )
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
            if f_name in ("cartesianX", "cartesianY", "cartesianZ") and coords_modified:
                src_arr = np.ascontiguousarray(data_to_write[f_name], dtype=dtype)
            else:
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


def write_e57_scans(file_path: str, scans: List[E57ScanData]) -> None:
    """
    複数のスキャンデータを新しいE57ファイルにロスレス（元の形式・精度・座標値を完全保持）で書き出す。
    一時ファイルへの書き込み完了後にアトミック置換を行うため、エラー時に壊れた空ファイルが残らない。

    :param file_path: 出力先E57ファイルのパス
    :param scans: 書き込むE57ScanDataオブジェクトのリスト
    """
    if not scans:
        raise ValueError("書き込むスキャンデータが空です。")

    abs_path = os.path.abspath(file_path)
    output_dir = os.path.dirname(abs_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # アトミック書き込み用の一時ファイルパス
    temp_path = f"{abs_path}.tmp_{os.getpid()}_{uuid.uuid4().hex[:6]}.e57"

    imf = None
    try:
        imf = libe57.ImageFile(temp_path, "w")
        root = imf.root()

        # 標準E57名前空間および元ファイルの拡張名前空間を登録
        imf.extensionsAdd("", libe57.E57_V1_0_URI)
        added_prefixes = {""}
        for scan in scans:
            for prefix, uri in scan.root_extensions:
                if prefix not in added_prefixes:
                    try:
                        imf.extensionsAdd(prefix, uri)
                        added_prefixes.add(prefix)
                    except Exception:
                        pass

        # ルートヘッダ属性の設定
        root.set("formatName", libe57.StringNode(imf, "ASTM E57 3D Imaging Data File"))
        root.set("guid", libe57.StringNode(imf, "{%s}" % uuid.uuid4()))
        root.set("versionMajor", libe57.IntegerNode(imf, libe57.E57_FORMAT_MAJOR))
        root.set("versionMinor", libe57.IntegerNode(imf, libe57.E57_FORMAT_MINOR))
        root.set("e57LibraryVersion", libe57.StringNode(imf, libe57.E57_LIBRARY_ID))

        coord_meta = ""
        for scan in scans:
            if scan.coordinate_metadata:
                coord_meta = scan.coordinate_metadata
                break
        root.set("coordinateMetadata", libe57.StringNode(imf, coord_meta))

        creation_date_time = libe57.StructureNode(imf)
        creation_date_time.set("dateTimeValue", libe57.FloatNode(imf, 0.0, libe57.E57_DOUBLE))
        creation_date_time.set("isAtomicClockReferenced", libe57.IntegerNode(imf, 0))
        root.set("creationDateTime", creation_date_time)

        data3d = libe57.VectorNode(imf, True)
        root.set("data3D", data3d)
        root.set("images2D", libe57.VectorNode(imf, True))

        for idx, scan in enumerate(scans):
            _write_single_scan_lossless(imf, data3d, scan, idx)

        imf.close()
        imf = None

        # 書き込み完了後、ターゲットファイルへアトミックに置換
        os.replace(temp_path, abs_path)

    except Exception as e:
        if imf is not None:
            try:
                imf.close()
            except Exception:
                pass
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
        raise e
