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

    print("\n--- 5. 部分ノイズ復元（ピンポイント復元）＆全ノイズ復元テスト ---")
    # 現在ノイズレイヤーにある点のうち先頭5点のみを選択して部分復元
    all_noise_idx = np.where(~viewer.layer_mask)[0]
    partial_restore_idx = all_noise_idx[:5]
    viewer.restore_indices_to_normal(partial_restore_idx)
    assert np.all(viewer.layer_mask[partial_restore_idx]), "部分復元した5点が通常レイヤーに戻っていません"
    assert np.sum(~viewer.layer_mask) == len(all_noise_idx) - 5, "残りのノイズ点数が不一致です"
    print("✓ 部分ノイズ復元テスト合格: 指定した5点のみが通常レイヤーへピンポイント復元")

    # 残りの全ノイズを復元
    remain_noise_idx = np.where(~viewer.layer_mask)[0]
    viewer.restore_indices_to_normal(remain_noise_idx)
    assert np.all(viewer.layer_mask), "全復元後に全点が通常レイヤーに戻っていません"
    assert np.sum(viewer.layer_mask) == 100, "復元後の点数が100点ではありません"
    print("✓ 全ノイズ復元テスト合格: 全点が正常に通常レイヤーへ復元")

    print("\n--- 6. Shift追加選択 / Ctrl除外選択 / 選択反転テスト ---")
    from PySide6.QtCore import Qt
    viewer.set_layer_visibility(show_normal=True, show_noise=False)
    # 初期選択: 0..9 (10点)
    viewer._apply_selection_with_modifiers(np.arange(0, 10), Qt.NoModifier)
    assert len(viewer.selected_indices) == 10, "新規選択が10点ではありません"
    # Shift追加選択: 5..14 (重複5点、合計15点になるはず)
    viewer._apply_selection_with_modifiers(np.arange(5, 15), Qt.ShiftModifier)
    assert len(viewer.selected_indices) == 15, "Shift追加選択後の点数が15点ではありません"
    # Ctrl除外選択: 0..4 (5点を除外し、10点になるはず)
    viewer._apply_selection_with_modifiers(np.arange(0, 5), Qt.ControlModifier)
    assert len(viewer.selected_indices) == 10, "Ctrl除外選択後の点数が10点ではありません"
    # 選択反転: 表示中100点のうち未選択だった90点が選択状態になるはず
    viewer.invert_selection()
    assert len(viewer.selected_indices) == 90, f"選択反転後の点数が90点ではありません: {len(viewer.selected_indices)}"
    viewer.clear_selection()
    print("✓ 修飾キー選択＆選択反転テスト合格")

    print("\n--- 7. 断面ボックス外クロップ＆カラーマップテスト ---")
    # Z軸の断面を [-1.0, 1.0] に設定し、断面外の通常点インデックスを取得
    viewer.set_section_z(True, -1.0, 1.0)
    outside_idx = viewer.get_outside_section_normal_indices()
    assert len(outside_idx) > 0, "断面外の点が検出されませんでした"
    viewer.move_indices_to_noise(outside_idx)
    viewer.set_section_z(False, 0.0, 0.0)
    assert np.sum(viewer.layer_mask) == 100 - len(outside_idx), "断面外クロップ後の通常点数が不一致です"
    print(f"✓ 断面外クロップテスト合格: 断面外の {len(outside_idx)} 点をノイズレイヤーへ一括移動")

    # カラーマップ（Viridis / Jet / 反射強度パーセンタイルストレッチ）の検証
    intensities = np.linspace(0.0, 100.0, 100, dtype=np.float32)
    viewer.set_point_cloud(points, intensities=intensities)
    viewer.set_color_mode(PointCloudViewer.COLOR_MODE_HEIGHT_VIRIDIS)
    assert viewer.active_colors.shape == (100, 3), "Viridisカラー配列の形状が不正です"
    viewer.set_color_mode(PointCloudViewer.COLOR_MODE_HEIGHT_JET)
    assert viewer.active_colors.shape == (100, 3), "Jetカラー配列の形状が不正です"
    viewer.set_color_mode(PointCloudViewer.COLOR_MODE_INTENSITY)
    assert viewer.active_colors.shape == (100, 3), "反射強度カラー配列の形状が不正です"

    # 異なる点数(50点)の点群を再ロードした際に古いlayer_maskやselected_indicesが残らないことを検証
    viewer._apply_selection_with_modifiers(np.arange(50, 90), Qt.NoModifier)
    viewer.move_indices_to_noise(np.arange(60, 80))
    smaller_points = np.random.normal(0, 1, (50, 3))
    viewer.set_point_cloud(smaller_points)
    assert len(viewer.layer_mask) == 50 and np.all(viewer.layer_mask), "再ロード後のlayer_maskが正しくリセットされていません"
    assert len(viewer.selected_indices) == 0, "再ロード後のselected_indicesがクリアされていません"
    print("✓ カラーマップ・反射強度ストレッチ・連続ロード状態リセット検証合格")

    print("\n===============================")
    print("全テストに合格しました！")
    print("===============================")


if __name__ == "__main__":
    run_tests()
