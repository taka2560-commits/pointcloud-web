"""
E57ファイルの読み込みおよび書き込みを担当するモジュール
"""

import os
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pye57


class E57ScanData:
    """単一スキャンの点群データおよびメタデータを保持するクラス"""

    def __init__(
        self,
        scan_index: int,
        points: np.ndarray,
        raw_fields: Dict[str, np.ndarray],
        header: Any = None,
        rotation: Optional[np.ndarray] = None,
        translation: Optional[np.ndarray] = None,
    ):
        self.scan_index = scan_index
        self.points = points  # (N, 3) 形状の XYZ 座標
        self.raw_fields = raw_fields  # 各種属性配列の辞書
        self.header = header
        self.rotation = rotation
        self.translation = translation

    @property
    def point_count(self) -> int:
        """点数を返す"""
        return len(self.points)

    def filter_by_indices(self, keep_indices: np.ndarray) -> "E57ScanData":
        """指定されたインデックスのみを抽出した新しいE57ScanDataを生成する"""
        new_points = self.points[keep_indices]
        new_fields = {}
        for key, val in self.raw_fields.items():
            if isinstance(val, np.ndarray) and len(val) == len(self.points):
                new_fields[key] = val[keep_indices]
            else:
                new_fields[key] = val

        return E57ScanData(
            scan_index=self.scan_index,
            points=new_points,
            raw_fields=new_fields,
            header=self.header,
            rotation=self.rotation,
            translation=self.translation,
        )


def read_e57_scans(
    file_path: str,
    progress_callback: Optional[Any] = None,
) -> List[E57ScanData]:
    """
    E57ファイルを読み込み、各スキャンの点群データオブジェクトのリストを返す。
    
    :param file_path: 入力E57ファイルのパス
    :param progress_callback: 進捗コールバック関数 (ratio: float, msg: str)
    :return: E57ScanData のリスト
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"ファイルが見つかりません: {file_path}")

    if progress_callback:
        progress_callback(0.05, f"E57ヘッダー解析中: {os.path.basename(file_path)}")

    scans = []
    with pye57.E57(file_path, mode="r") as e57:
        scan_count = e57.scan_count

        for i in range(scan_count):
            if progress_callback:
                progress_callback(
                    0.1 + (i / max(1, scan_count)) * 0.8,
                    f"スキャン #{i + 1}/{scan_count} を読み込み中...",
                )

            header = e57.get_header(i)
            raw_data = e57.read_scan_raw(i)

            # 直交座標の取得
            if "cartesianX" in raw_data and "cartesianY" in raw_data and "cartesianZ" in raw_data:
                x = np.asarray(raw_data["cartesianX"], dtype=np.float64)
                y = np.asarray(raw_data["cartesianY"], dtype=np.float64)
                z = np.asarray(raw_data["cartesianZ"], dtype=np.float64)
                points = np.column_stack([x, y, z])
            elif "sphericalRange" in raw_data:
                # 球面座標から直交座標への変換
                r = np.asarray(raw_data["sphericalRange"], dtype=np.float64)
                az = np.asarray(raw_data.get("sphericalAzimuth", np.zeros_like(r)), dtype=np.float64)
                el = np.asarray(raw_data.get("sphericalElevation", np.zeros_like(r)), dtype=np.float64)
                
                # 直交座標に変換
                cos_el = np.cos(el)
                x = r * cos_el * np.cos(az)
                y = r * cos_el * np.sin(az)
                z = r * np.sin(el)
                points = np.column_stack([x, y, z])
                
                # 直交座標フィールドもraw_dataに追加しておく
                raw_data["cartesianX"] = x
                raw_data["cartesianY"] = y
                raw_data["cartesianZ"] = z
            else:
                raise ValueError(f"スキャン #{i} に座標データ（cartesian/spherical）が見つかりません。")

            # 回転および並進ベクトルの取得
            rotation = getattr(header, "rotation", None)
            translation = getattr(header, "translation", None)

            scan_data = E57ScanData(
                scan_index=i,
                points=points,
                raw_fields=raw_data,
                header=header,
                rotation=rotation,
                translation=translation,
            )
            scans.append(scan_data)

    if progress_callback:
        progress_callback(0.95, "点群データの統合中...")

    return scans


def write_e57_scans(file_path: str, scans: List[E57ScanData]) -> None:
    """
    複数のスキャンデータを新しいE57ファイルに書き出す。
    
    :param file_path: 出力先E57ファイルのパス
    :param scans: 書き込むE57ScanDataオブジェクトのリスト
    """
    output_dir = os.path.dirname(os.path.abspath(file_path))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with pye57.E57(file_path, mode="w") as e57:
        for scan in scans:
            # write_scan_raw に渡すデータ辞書を作成
            scan_dict = {}
            for key, val in scan.raw_fields.items():
                if isinstance(val, np.ndarray):
                    scan_dict[key] = val
                else:
                    scan_dict[key] = val

            # 直交座標が更新されていることを保証
            scan_dict["cartesianX"] = scan.points[:, 0]
            scan_dict["cartesianY"] = scan.points[:, 1]
            scan_dict["cartesianZ"] = scan.points[:, 2]

            e57.write_scan_raw(
                scan_dict,
                rotation=scan.rotation,
                translation=scan.translation,
            )
