"""
レイヤー管理機能と手動削除・自動ノイズ処理の統合テスト
- 手動削除した点が自動ノイズ処理後に復活しないことの検証
- レイヤー表示切り替え (通常ON/OFF, ノイズON/OFF) の検証
- クリーン保存用データ抽出の検証
"""

import sys
import numpy as np
from PySide6.QtWidgets import QApplication

from core.processor import FilterConfig, PointCloudProcessor
from gui.gl_viewer import PointCloudViewer


def run_tests():
    app = QApplication.instance() or QApplication(sys.argv)

    print("--- 1. ビューアの初期化と点群設定テスト ---")
    viewer = PointCloudViewer()
    
    # 100点の点群を生成 (球体クラスタ + 外れ値)
    np.random.seed(42)
    cluster = np.random.normal(0, 1, (90, 3))
    outliers = np.random.uniform(10, 20, (10, 3))
    points = np.vstack([cluster, outliers])  # 合計100点
    
    viewer.set_point_cloud(points)
    
    assert viewer.layer_mask is not None, "layer_mask が初期化されていません"
    assert np.all(viewer.layer_mask), "初期状態では全点が通常レイヤーである必要があります"
    assert len(viewer.get_normal_indices()) == 100, "初期通常点数が100点ではありません"
    print("✓ 初期化テスト合格: 100点すべて通常レイヤー")

    print("\n--- 2. 手動削除 (ノイズレイヤー移動) テスト ---")
    # インデックス 10..29 (計20点) を手動削除
    manual_noise_idx = np.arange(10, 30)
    viewer.move_indices_to_noise(manual_noise_idx)

    assert np.sum(viewer.layer_mask) == 80, "手動削除後の通常点数が80点ではありません"
    assert np.sum(~viewer.layer_mask) == 20, "手動削除後のノイズ点数が20点ではありません"
    assert np.all(~viewer.layer_mask[manual_noise_idx]), "手動削除した点がノイズレイヤーにありません"
    print("✓ 手動削除テスト合格: 通常80点, ノイズ20点")

    print("\n--- 3. 通常点群のみを対象とした自動ノイズ処理テスト ---")
    # 通常レイヤーの点のみを取得
    normal_indices = viewer.get_normal_indices()
    assert len(normal_indices) == 80, "対象通常点数が80点ではありません"
    target_points = viewer.points_raw[normal_indices]

    # SORフィルタを実行
    processor = PointCloudProcessor(config=FilterConfig(use_sor=True, sor_neighbors=5, sor_std_ratio=1.0))
    keep_rel_idx, remove_rel_idx = processor.apply_filters_to_points(target_points)
    
    print(f"  フィルタ結果: 対象 {len(target_points)} 点中 {len(remove_rel_idx)} 点のノイズを検出")
    assert len(remove_rel_idx) > 0, "テストデータからノイズが検出されませんでした"

    # 相対インデックスを原本インデックスに変換してノイズレイヤーへ移動
    orig_remove_idx = normal_indices[remove_rel_idx]
    viewer.move_indices_to_noise(orig_remove_idx)

    # 検証: 手動削除した点(manual_noise_idx)が依然としてノイズレイヤーにあり、決して復活していないこと！
    assert np.all(~viewer.layer_mask[manual_noise_idx]), "手動削除した点が復活してしまいました！"
    
    expected_normal_count = 80 - len(orig_remove_idx)
    assert np.sum(viewer.layer_mask) == expected_normal_count, f"残存通常点数が不正です: 期待 {expected_normal_count}, 実際 {np.sum(viewer.layer_mask)}"
    print(f"✓ 自動ノイズ処理テスト合格: 手動削除点20点は完全に維持され、追加ノイズ {len(orig_remove_idx)} 点もノイズレイヤーへ移動 (残存通常: {expected_normal_count} 点)")

    print("\n--- 4. レイヤー表示切り替えテスト ---")
    # 初期状態: 通常ON, ノイズOFF
    viewer.set_layer_visibility(show_normal=True, show_noise=False)
    assert len(viewer.display_orig_indices) == expected_normal_count, "通常のみ表示時の描画点数が不一致です"

    # ノイズのみ表示
    viewer.set_layer_visibility(show_normal=False, show_noise=True)
    expected_noise_count = 20 + len(orig_remove_idx)
    assert len(viewer.display_orig_indices) == expected_noise_count, "ノイズのみ表示時の描画点数が不一致です"

    # 両方表示
    viewer.set_layer_visibility(show_normal=True, show_noise=True)
    assert len(viewer.display_orig_indices) == 100, "両方表示時の描画点数が100点ではありません"
    print("✓ レイヤー表示切替テスト合格: ON/OFFに応じて描画点数が正確に連動")

    print("\n--- 5. ノイズ復元テスト ---")
    # 全ノイズを復元
    all_noise_idx = np.where(~viewer.layer_mask)[0]
    viewer.restore_indices_to_normal(all_noise_idx)
    assert np.all(viewer.layer_mask), "復元後に全点が通常レイヤーに戻っていません"
    assert np.sum(viewer.layer_mask) == 100, "復元後の点数が100点ではありません"
    print("✓ ノイズ復元テスト合格: 全点が正常に通常レイヤーへ復元")

    print("\n===============================")
    print("全テストに合格しました！")
    print("===============================")


if __name__ == "__main__":
    run_tests()
