"""
Open3Dを用いたノイズ除去結果の3Dプレビュー可視化モジュール
"""

import threading
import numpy as np
import open3d as o3d


def show_point_cloud_preview(
    inliers: np.ndarray,
    outliers: np.ndarray,
    window_name: str = "ノイズ処理プレビュー (緑/青: 保持点, 赤: ノイズ点)",
):
    """
    Open3Dの3Dビューアを起動し、ノイズ処理前後の点群を色分けして表示する。
    - 保持点 (Inliers): 青緑色 (Cyan/Green)
    - ノイズ点 (Outliers): 鮮やかな赤色 (Red)

    :param inliers: 保持される点の座標配列 (N, 3)
    :param outliers: ノイズとして除去される点の座標配列 (M, 3)
    :param window_name: ウィンドウタイトル
    """
    geometries = []

    if len(inliers) > 0:
        pcd_inliers = o3d.geometry.PointCloud()
        pcd_inliers.points = o3d.utility.Vector3dVector(inliers)
        # 保持点をシアン色にペイント
        colors_inliers = np.tile([0.0, 0.75, 0.75], (len(inliers), 1))
        pcd_inliers.colors = o3d.utility.Vector3dVector(colors_inliers)
        geometries.append(pcd_inliers)

    if len(outliers) > 0:
        pcd_outliers = o3d.geometry.PointCloud()
        pcd_outliers.points = o3d.utility.Vector3dVector(outliers)
        # ノイズ点を鮮やかな赤色にペイント
        colors_outliers = np.tile([1.0, 0.1, 0.1], (len(outliers), 1))
        pcd_outliers.colors = o3d.utility.Vector3dVector(colors_outliers)
        geometries.append(pcd_outliers)

    if not geometries:
        return

    # 別スレッドで起動してメインUIのブロックを軽減
    def run_viewer():
        o3d.visualization.draw_geometries(
            geometries,
            window_name=window_name,
            width=1024,
            height=768,
            left=100,
            top=100,
        )

    t = threading.Thread(target=run_viewer, daemon=True)
    t.start()
