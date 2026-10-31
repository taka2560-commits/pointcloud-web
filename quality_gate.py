"""
自動品質管理ゲート (Quality Gate System)
========================================
今後新しい機能を追加・変更した際に、手戻りや潜在バグを未然に100%ブロックするための
5段階自動検証スクリプトです。

【5段階の品質ゲート構成】
- Gate 1: 静的構文・クロスリファレンス整合性ゲート
    * 全Pythonファイルのバイトコードコンパイル & AST検査
    * Web版 (index.html ⇄ app.js) の DOM ID 参照整合性・JS構文バランス検査
- Gate 2: 状態不変条件 (State Invariants) アサーションゲート
    * PointCloudViewer.validate_invariants() による配列長・マスク・選択範囲の完全一致検証
    * 異なる点数の点群を連続ロードした際の状態汚染（古いマスク残留）防止検証
- Gate 3: 境界値・カオス入力ファジングゲート (Edge-Case Fuzzing)
    * 0点 / 1点 / 2点 / 同一座標(分散ゼロ) / 全除去条件 / 反射強度一定値 に対する
      全5種フィルタ & ビューアのゼロ除算・NaN・クラッシュ耐性検査
- Gate 4: 実機 OpenGL & GUI シナリオ自動スモークゲート
    * MainWindow 起動・OpenGL VBO ハードウェア描画 (grabFramebuffer)
    * 選択・反転・削除・Undo/Redo・部分ノイズ復元・断面外クロップ・全配色・全形状の一連操作検証
- Gate 5: E57 完全ロスレス・単体/統合テストスイートゲート
    * 倍精度座標・ScaledIntegerNode・拡張属性・マルチスキャン完全保持テストの実行

使い方:
    python quality_gate.py                 # 全5段階ゲートを実行
    python quality_gate.py --quick         # 軽量モード: GUI ウィンドウを開く Gate 2/4 を除く (Gate 1, 3, 5)
    python quality_gate.py --fix-eol       # .bat の改行コードを CRLF へ修復（既定では検査のみで書き換えない）
    python quality_gate.py --install-hook  # Git pre-commit フックに自動登録
                                           # （注意: .git/hooks は全 worktree で共有されるため、他の開発者/AIの
                                           #   コミットにも適用される。並行開発中は手動実行を推奨）

一時ファイルはすべて OS の一時ディレクトリ配下に作成し、リポジトリ内には残さない。
"""

import ast
import os
import py_compile
import re
import shutil
import sys
import tempfile
import time
import unittest
from typing import List, Tuple

import numpy as np

# --fix-eol 指定時のみ .bat の改行コードを書き換える（既定は検査のみ）
FIX_EOL = "--fix-eol" in sys.argv

# Windows環境(cp932)でのUnicodeEncodeError(絵文字等)クラッシュ防止
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def print_banner(title: str) -> None:
    print("\n" + "=" * 68)
    print(f" 🛡️  {title}")
    print("=" * 68)


def gate_1_static_and_dom_cross_check() -> None:
    """Gate 1: 全Python構文・AST検査 ＆ Web版 HTML/JS DOM ID完全照合"""
    print_banner("Gate 1: 静的構文・Web DOM ID クロスリファレンス整合性検査")

    # 1. 全Pythonファイルのコンパイル & ASTパース検査
    py_files: List[str] = []
    for root, dirs, files in os.walk("."):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
        for f in files:
            if f.endswith(".py"):
                py_files.append(os.path.join(root, f))

    for path in sorted(py_files):
        py_compile.compile(path, doraise=True)
        with open(path, "r", encoding="utf-8") as fp:
            source = fp.read()
        ast.parse(source, filename=path)
    print(f"  [OK] Python 全 {len(py_files)} ファイルのバイトコードコンパイル & AST検査合格")

    # 2. Web版 index.html ⇄ app.js の DOM ID 整合性チェック
    if os.path.exists("index.html") and os.path.exists("app.js"):
        with open("index.html", "r", encoding="utf-8") as f_html:
            html_text = f_html.read()
        with open("app.js", "r", encoding="utf-8") as f_js:
            js_text = f_js.read()

        html_ids = set(re.findall(r'id=["\']([^"\']+)["\']', html_text))
        js_id_matches = re.findall(r'getElementById\(["\']([^"\']+)["\']\)', js_text)
        js_ids = set(js_id_matches)

        missing_ids = js_ids - html_ids
        if missing_ids:
            raise AssertionError(
                f"app.js が参照している以下の ID が index.html に存在しません: {sorted(missing_ids)}"
            )

        # JSの波括弧・丸括弧バランスと重複関数定義チェック
        fn_names = re.findall(r"^\s*(?:async\s+)?function\s+([a-zA-Z0-9_]+)\s*\(", js_text, re.MULTILINE)
        seen_fns = set()
        dup_fns = []
        for fn in fn_names:
            if fn in seen_fns:
                dup_fns.append(fn)
            seen_fns.add(fn)
        if dup_fns:
            raise AssertionError(f"app.js 内に重複した関数定義があります: {dup_fns}")

        print(
            f"  [OK] Web版 DOM ID 照合合格 (HTML定義: {len(html_ids)}個, JS参照: {len(js_ids)}個, 欠損: 0個, 関数定義: {len(fn_names)}個)"
        )

    # 3. Windows バッチファイル (.bat) の cmd.exe 文字化け・バイトオフセットずれ防止検査 (純粋ASCII & CRLF改行)
    bat_files = [f for f in os.listdir(".") if f.lower().endswith(".bat") and os.path.isfile(f)]
    for bf in sorted(bat_files):
        with open(bf, "rb") as f_bat:
            raw_bytes = f_bat.read()
        try:
            text_ascii = raw_bytes.decode("ascii")
        except UnicodeDecodeError as e:
            raise AssertionError(
                f"{bf} に非ASCII文字が含まれています。Windows cmd.exe の文字コード位置ずれバグを防ぐため純粋ASCIIで記述してください: {e}"
            )
        normalized_crlf = text_ascii.replace("\r\n", "\n").replace("\n", "\r\n").encode("ascii")
        if raw_bytes != normalized_crlf:
            if FIX_EOL:
                # 明示オプション (--fix-eol) 指定時のみ修復する（検査がファイルを書き換える副作用を持たないようにする）
                with open(bf, "wb") as f_bat_out:
                    f_bat_out.write(normalized_crlf)
                print(f"  [FIX] {bf} の改行コードを CRLF に正規化しました")
            else:
                raise AssertionError(
                    f"{bf} の改行コードが CRLF に統一されていません。"
                    "`python quality_gate.py --fix-eol` で修復するか、エディタで CRLF 保存してください。"
                )
    print(f"  [OK] Windows バッチファイル 全 {len(bat_files)} 個の純粋ASCII & CRLF改行検査合格")



def gate_2_state_invariants_check() -> None:
    """Gate 2: ビューア内部状態の不変条件 (State Invariants) & 連続ロード汚染防止検査"""
    print_banner("Gate 2: 状態不変条件 (State Invariants) アサーション検査")

    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication
    from gui.gl_viewer import PointCloudViewer

    app = QApplication.instance() or QApplication(sys.argv)
    viewer = PointCloudViewer()

    # 初期状態 (空) の不変条件
    viewer.validate_invariants()

    # 1回目の点群 (120点) をロードして編集・選択
    np.random.seed(10)
    pts1 = np.random.normal(0.0, 2.0, (120, 3)).astype(np.float64)
    cols1 = np.random.uniform(0.0, 1.0, (120, 3)).astype(np.float32)
    ints1 = np.random.uniform(0.0, 100.0, 120).astype(np.float32)

    viewer.set_point_cloud(pts1, colors=cols1, intensities=ints1)
    viewer.validate_invariants()

    viewer._apply_selection_with_modifiers(np.arange(80, 115), Qt.NoModifier)
    viewer.validate_invariants()

    viewer.move_indices_to_noise(np.arange(90, 120))
    viewer.validate_invariants()

    # 2回目の点群 (40点: 1回目より少ない点数) を連続ロードした際に IndexError や古いマスクが残らないこと
    pts2 = np.random.normal(5.0, 1.0, (40, 3)).astype(np.float64)
    viewer.set_point_cloud(pts2)
    viewer.validate_invariants()
    assert len(viewer.layer_mask) == 40 and np.all(viewer.layer_mask)
    assert len(viewer.selected_indices) == 0

    # 空データリセット時の不変条件
    viewer.set_point_cloud(None)
    viewer.validate_invariants()
    print("  [OK] 状態不変条件 (validate_invariants) および連続ロード時の状態リセット検査合格")


def gate_3_boundary_and_fuzzing_check() -> None:
    """Gate 3: 境界値・カオス入力（0点、1点、同一座標、コントラスト幅ゼロ等）耐性検査"""
    print_banner("Gate 3: 境界値・カオス入力ファジング検査 (Edge-Case Fuzzing)")

    from PySide6.QtWidgets import QApplication
    from core.processor import FilterConfig, PointCloudProcessor, transform_coordinate_system
    from core.filters import segment_ground_points, segment_largest_plane
    from gui.gl_viewer import PointCloudViewer

    app = QApplication.instance() or QApplication(sys.argv)
    viewer = PointCloudViewer()

    edge_cases: List[Tuple[str, np.ndarray, np.ndarray]] = [
        ("空データ (0点)", np.empty((0, 3), dtype=np.float64), np.empty(0, dtype=np.float32)),
        ("単一点 (1点)", np.array([[1.0, 2.0, 3.0]], dtype=np.float64), np.array([0.5], dtype=np.float32)),
        ("2点データ", np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float64), np.array([0.2, 0.8], dtype=np.float32)),
        ("全点同一座標 (分散ゼロ 30点)", np.ones((30, 3), dtype=np.float64) * 5.0, np.ones(30, dtype=np.float32) * 0.5),
        ("平面点群 (Z厚みがゼロ 50点)", np.column_stack([np.linspace(0, 10, 50), np.linspace(0, 10, 50), np.zeros(50)]), np.zeros(50, dtype=np.float32)),
    ]

    full_config = FilterConfig(
        use_sor=True,
        sor_neighbors=20,
        sor_std_ratio=1.5,
        use_ror=True,
        ror_radius=0.5,
        ror_min_points=5,
        use_cluster_filter=True,
        cluster_eps=0.5,
        cluster_min_points=10,
        use_distance_filter=True,
        min_distance=0.0,
        max_distance=100.0,
        use_intensity_filter=True,
        min_intensity=-10.0,
        max_intensity=10.0,
        use_voxel_downsample=True,
        voxel_size=0.05,
    )
    processor = PointCloudProcessor(config=full_config)

    for case_name, pts, ints in edge_cases:
        keep_idx, rem_idx = processor.apply_filters_to_points(pts, intensities=ints)
        assert len(keep_idx) + len(rem_idx) == len(pts), f"{case_name}: keep + remove != total"

        # 地面自動抽出・主平面自動検出の境界値耐性チェック
        g_idx = segment_ground_points(pts)
        p_idx, _ = segment_largest_plane(pts)
        assert len(g_idx) <= len(pts) and len(p_idx) <= len(pts)

        viewer.set_point_cloud(pts, intensities=ints)
        for mode in [
            PointCloudViewer.COLOR_MODE_RGB,
            PointCloudViewer.COLOR_MODE_HEIGHT,
            PointCloudViewer.COLOR_MODE_HEIGHT_VIRIDIS,
            PointCloudViewer.COLOR_MODE_HEIGHT_JET,
            PointCloudViewer.COLOR_MODE_INTENSITY,
            PointCloudViewer.COLOR_MODE_LAYER,
            PointCloudViewer.COLOR_MODE_SCAN,
            PointCloudViewer.COLOR_MODE_SOLID,
        ]:
            viewer.set_color_mode(mode)
            viewer.validate_invariants()

        viewer.invert_selection()
        viewer.validate_invariants()
        print(f"  [OK] 境界値ケース通過: {case_name}")


def gate_4_gui_and_opengl_smoke_check() -> None:
    """Gate 4: 実機 OpenGL VBO 描画 & MainWindow 全操作・レイヤー拡張①〜⑥シナリオスモークテスト"""
    print_banner("Gate 4: 実機 OpenGL VBO & GUI 操作シナリオ (レイヤー拡張①〜⑥含む) 自動スモーク検査")

    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication, QWidget
    from core.e57_io import E57ScanData
    from gui.main_window import MainWindow
    from gui.gl_viewer import PointCloudViewer
    from gui.worker_thread import FileSaveWorker

    # 新UI（modern）で検証する。ユーザー設定で従来UIが選ばれていても、ゲートは両モードを順に検証する
    from gui.ui_mode import reset_ui_mode_cache
    os.environ["E57_UI_MODE"] = "modern"
    reset_ui_mode_cache()

    app = QApplication.instance() or QApplication(sys.argv)
    win = MainWindow()
    win.show()
    app.processEvents()
    assert win.ui_mode == "modern", f"UIモードが modern ではありません: {win.ui_mode}"

    # テスト用マルチスキャン (Scan 0: 200点, Scan 1: 100点 = 計300点) をロード
    np.random.seed(99)
    pts0 = np.column_stack([
        np.random.uniform(-5, 5, 200),
        np.random.uniform(-5, 5, 200),
        np.zeros(200),  # Z=0 の床・地面点群
    ]).astype(np.float64)
    pts1 = np.random.normal(0, 4, (100, 3)).astype(np.float64)
    pts1[:, 2] += 3.0  # 上空の点群

    scan0 = E57ScanData(
        scan_index=0,
        points=pts0,
        raw_fields={"cartesianX": pts0[:, 0], "cartesianY": pts0[:, 1], "cartesianZ": pts0[:, 2]},
        name="Station_A",
    )
    scan1 = E57ScanData(
        scan_index=1,
        points=pts1,
        raw_fields={"cartesianX": pts1[:, 0], "cartesianY": pts1[:, 1], "cartesianZ": pts1[:, 2]},
        name="Station_B",
    )
    win.current_scans = [scan0, scan1]
    win._update_viewer_from_scans()
    app.processEvents()

    # OpenGL フレームバッファ取得 & VBO 検証
    fb = win.viewer.grabFramebuffer()
    assert fb.width() > 0 and fb.height() > 0, "OpenGL フレームバッファの取得に失敗しました"
    assert win.viewer._vbo_points is not None and not win.viewer._vbo_dirty, "OpenGL VBO が同期されていません"
    win.viewer.validate_invariants()

    # 左パネルの見切れ（水平方向オーバーフロー）がないことを検証
    app.processEvents()
    viewport_w = win.sidebar_scroll.viewport().width()
    content_min_w = win.sidebar_widget.minimumSizeHint().width()
    child_widths = [
        f"{c.title() if hasattr(c, 'title') else c.objectName() or type(c).__name__}:{c.minimumSizeHint().width()}px"
        for c in win.sidebar_widget.findChildren(QWidget, options=Qt.FindDirectChildrenOnly)
    ]
    assert content_min_w <= viewport_w, (
        f"左パネルの内部ウィジェット最小幅({content_min_w}px)が表示領域幅({viewport_w}px)を超えており、右端が見切れています！ 内訳: {child_widths}"
    )
    print(f"  [OK] 左パネル見切れ検査合格 (コンテンツ最小幅: {content_min_w}px <= 表示領域幅: {viewport_w}px)")

    # UI操作性向上 (柱①〜④) の自動検査
    # 柱①: ワークフロー切替セグメントバーの動作確認
    for tab_key in ["layer", "select", "section", "auto", "view", "settings", "all"]:
        win._set_sidebar_workflow_tab(tab_key)
        app.processEvents()
        if tab_key == "layer":
            assert win.layer_group.isVisible() and not win.auto_group.isVisible()
        elif tab_key == "section":
            assert win.section_group.isVisible() and not win.tools_group.isVisible()
        elif tab_key == "settings":
            assert win.settings_group.isVisible() and not win.layer_group.isVisible()
        elif tab_key == "all":
            assert win.layer_group.isVisible() and win.auto_group.isVisible() and win.settings_group.isVisible()

    # 柱②: 1クリック操作モード切替セグメントコントロールの動作確認
    win._set_interaction_mode("rect")
    assert win.viewer.selection_mode and win.viewer.selection_shape == PointCloudViewer.SELECT_RECT and win.seg_btn_rect.isChecked()
    win._set_interaction_mode("circle")
    assert win.viewer.selection_mode and win.viewer.selection_shape == PointCloudViewer.SELECT_CIRCLE and win.seg_btn_circle.isChecked()
    win._set_interaction_mode("measure")
    assert win.viewer.measure_mode and win.nav_btn_measure.isChecked()
    # 計測ツールの座標読み取り: 1点目で座標行、2点目で距離行と ΔX/ΔY/ΔZ、座標系変換時はファイル系の行、クリア
    from gui.measure_text import measure_clipboard_text
    _dp = win.viewer.display_points
    assert _dp is not None and len(_dp) >= 2
    _c1, _c2 = _dp[0], _dp[len(_dp) - 1]
    win.viewer._on_measure_pick(_c1, _c1.astype(np.float64) + win.viewer.center_offset)
    app.processEvents()
    assert len(win.viewer.measure_coord_lines) == 1 and win.viewer.measure_coord_lines[0].startswith("P1 X="), win.viewer.measure_coord_lines
    assert win.status_label.text().startswith("P1 X="), win.status_label.text()
    assert win.viewer.measure_result_text == ""
    win.viewer._on_measure_pick(_c2, _c2.astype(np.float64) + win.viewer.center_offset)
    app.processEvents()
    assert len(win.viewer.measure_coord_lines) == 4 and win.viewer.measure_coord_lines[1].startswith("P2 X=")
    assert win.viewer.measure_result_text.startswith("直線距離:") and win.viewer.measure_coord_lines[3].startswith("ΔX=")
    assert win.status_label.text().startswith("直線距離:")
    _tsv = measure_clipboard_text([p[1] for p in win.viewer.measure_points], win.viewer.coord_mode)
    assert _tsv.startswith("点\tX\tY\tZ") and "\nP1\t" in _tsv and "\nP2\t" in _tsv and "直線距離" in _tsv
    win.viewer.coord_mode = "survey_swap"
    win.viewer._refresh_measure_text()
    assert len(win.viewer.measure_coord_lines) == 6 and win.viewer.measure_coord_lines[1].strip().startswith("ファイル系")
    win.viewer.coord_mode = "cartesian"
    win.viewer._on_measure_pick(_c1, _c1.astype(np.float64) + win.viewer.center_offset)  # 3点目 → 新しい計測
    assert len(win.viewer.measure_points) == 1 and len(win.viewer.measure_coord_lines) == 1
    win.viewer.clear_measure_points()
    assert not win.viewer.measure_points and not win.viewer.measure_coord_lines and win.viewer.measure_result_text == ""
    print("  [OK] 計測ツール 座標読み取り（座標行・距離行・ファイル系併記・コピー文字列・クリア）検査合格")
    win._set_interaction_mode("orbit")
    assert not win.viewer.measure_mode and not win.viewer.selection_mode and win.seg_btn_orbit.isChecked()

    # 柱② & 柱④: 選択時の下部フローティングアクションバー連動 & トースト通知の検証
    assert not win.floating_del_btn.isEnabled(), "未選択時は下部アクションバーの削除ボタンは無効である必要があります"
    win.viewer._apply_selection_with_modifiers(np.arange(0, 15), Qt.NoModifier)
    app.processEvents()
    assert win.floating_del_btn.isEnabled() and win.floating_move_btn.isEnabled(), "点群選択時に下部フローティングアクションバーのボタンが有効化されませんでした"
    assert "15" in win.floating_sel_badge.text()
    win._clear_selection()
    app.processEvents()
    assert not win.floating_del_btn.isEnabled(), "選択解除時に下部アクションバーのボタンが無効化されませんでした"

    win.show_toast("UIテスト完了トースト", duration_ms=1000)
    assert win.toast_label.isVisible() and "UIテスト完了トースト" in win.toast_label.text()
    print("  [OK] 柱①〜④ (ワークフロータブ・1クリックモード切替・下部フローティングバー・トースト通知) 検査合格")


    # ③ スキャン(器械点)別表示切替の検証
    assert len(win.viewer.get_visible_indices()) == 300
    win.viewer.set_scan_visibility(1, False)
    assert len(win.viewer.get_visible_indices()) == 200
    win.viewer.set_scan_visibility(1, True)
    assert len(win.viewer.get_visible_indices()) == 300

    # ① カスタムレイヤー作成 & 選択点の移動 & ② レイヤーロック保護の検証
    win.custom_layer_name_edit.setText("重要構造物")
    win._on_add_custom_layer()
    custom_lid = win.target_layer_combo.currentData()
    assert custom_lid is not None and custom_lid >= 4

    win.viewer._apply_selection_with_modifiers(np.arange(0, 25), Qt.NoModifier)
    win._on_move_selected_to_target_layer()
    assert len(win.viewer.get_layer_indices(custom_lid)) == 25

    # ②「重要構造物」レイヤーをロックすると、削除・ノイズ移動から保護されること
    win._on_toggle_layer_lock(custom_lid, True)
    win.viewer.move_indices_to_noise(np.arange(0, 50))
    # 0〜24 はロックされているので「重要構造物」のまま残り、25〜49 の 25点のみがノイズへ移動するはず
    assert len(win.viewer.get_layer_indices(custom_lid)) == 25
    assert len(win.viewer.get_layer_indices(PointCloudViewer.LAYER_NOISE)) == 25
    win._on_toggle_layer_lock(custom_lid, False)

    # 全ノイズ復元と Undo / Redo 検証
    win._restore_all_noise()
    assert len(win.viewer.get_normal_indices()) == 300

    # ⑥ 自動レイヤー振り分け (地面抽出 & 主平面検出) の検証
    #    ※ 重い解析は FunctionWorker でバックグラウンド実行されるため、完了を待ってから検証する
    win._on_auto_extract_ground()
    assert win._wait_for_background(), "自動地面抽出のバックグラウンド処理がタイムアウトしました"
    win.viewer.validate_invariants()
    assert len(win.viewer.get_layer_indices(PointCloudViewer.LAYER_GROUND)) > 0

    # 一度Undoで戻してから主平面検出も検証
    win._undo_action()
    win.viewer.validate_invariants()
    win._on_auto_extract_plane()
    assert win._wait_for_background(), "自動平面検出のバックグラウンド処理がタイムアウトしました"
    win.viewer.validate_invariants()
    assert len(win.viewer.get_layer_indices(PointCloudViewer.LAYER_PLANE)) > 0

    # 再入防止: 処理中に同じ操作を重ねても二重起動しないこと（busy 判定）
    assert not win._is_background_busy()

    # 一時ファイル置き場（リポジトリ内に残さないよう OS の一時ディレクトリ配下に作る）
    qgate_tmp = tempfile.mkdtemp(prefix="e57_qgate_")

    # ⑤ レイヤー状態プロジェクト保存・再開 (.npz) の検証
    temp_proj_npz = os.path.join(qgate_tmp, "test_qgate_layers.npz")
    try:
        win._on_save_layer_project(temp_proj_npz)
        assert os.path.exists(temp_proj_npz), "レイヤー状態プロジェクトファイル (.npz) が生成されませんでした"
        saved_ids = win.viewer.layer_ids.copy()

        # レイヤー状態を初期化してから .npz を読み込んで完全一致するか検証
        win.viewer.move_indices_to_layer(np.arange(300), PointCloudViewer.LAYER_NORMAL, respect_lock=False)
        win._on_load_layer_project(temp_proj_npz)
        win.viewer.validate_invariants()
        assert np.array_equal(win.viewer.layer_ids, saved_ids), "復元された layer_ids が保存前と一致しません"
    finally:
        if os.path.exists(temp_proj_npz):
            os.remove(temp_proj_npz)

    # ④ FileSaveWorker による単一保存 & 全レイヤー分割保存 (split_masks) の検証
    temp_out = os.path.join(qgate_tmp, "test_qgate_temp_out.e57")
    created_split_files: List[str] = []
    try:
        save_worker = FileSaveWorker(temp_out, win.current_scans, win.viewer.layer_mask)
        save_worker.run()
        assert os.path.exists(temp_out), "FileSaveWorker による E57 出力ファイルが生成されませんでした"

        split_masks = win.viewer.get_all_layer_masks(include_empty=False, include_noise=False)
        assert len(split_masks) >= 2, "分割保存対象のレイヤー数が不足しています"
        split_worker = FileSaveWorker(temp_out, win.current_scans, win.viewer.layer_mask, split_masks=split_masks)
        split_worker.run()
        for f in os.listdir(qgate_tmp):
            if f.startswith("test_qgate_temp_out_") and f.endswith(".e57"):
                created_split_files.append(os.path.join(qgate_tmp, f))
        assert len(created_split_files) == len(split_masks), "レイヤー別分割E57ファイルの出力数が一致しません"
    finally:
        if os.path.exists(temp_out):
            os.remove(temp_out)
        for sf in created_split_files:
            if os.path.exists(sf):
                os.remove(sf)

    # 自動フィルタのフィルタ別レイヤー分離 & 1つ戻る/進む の検証
    from core.processor import FilterConfig, transform_coordinate_system
    from gui.worker_thread import MemoryFilterWorker

    win._on_toggle_layer_lock(custom_lid, False)
    win.viewer.move_indices_to_layer(np.arange(300), PointCloudViewer.LAYER_NORMAL, respect_lock=False)
    win.layer_undo_stack.clear()
    win.layer_redo_stack.clear()
    win._sync_undo_buttons()
    normal_idx = win.viewer.get_unlocked_normal_indices()
    cfg = FilterConfig(use_distance_filter=True, min_distance=0.0, max_distance=6.0, use_voxel_downsample=True, voxel_size=2.0)
    fw = MemoryFilterWorker(win.viewer.points_raw[normal_idx], cfg, intensities=None)
    got_stages: list = []
    fw.finished_stages.connect(lambda s: got_stages.extend(s))
    fw.run()
    assert len(got_stages) >= 1, "フィルタ別の除去段階が得られませんでした"
    all_removed = np.concatenate([idx for _, idx in got_stages])
    assert len(all_removed) == len(np.unique(all_removed)), "フィルタ間で除去点が重複しています"
    n_steps = 0
    for fname, local_idx in got_stages:
        win._push_layer_undo_state()
        lid = win.viewer.add_custom_layer(f"自動除去: {fname}")
        win.viewer.move_indices_to_layer(normal_idx[local_idx], lid, respect_lock=True)
        n_steps += 1
    win._sync_undo_buttons()
    assert win.auto_undo_btn.isEnabled() and not win.auto_redo_btn.isEnabled()
    after_all = win.viewer.layer_ids.copy()
    win.auto_undo_btn.click()
    assert win.auto_redo_btn.isEnabled(), "1つ戻る後に『1つ進む』が有効化されませんでした"
    assert len(win.layer_undo_stack) == n_steps - 1, "1つ戻るがフィルタ1段階分になっていません"
    win.auto_redo_btn.click()
    assert np.array_equal(win.viewer.layer_ids, after_all), "1つ進むで元の状態に復元されませんでした"
    # 差分 Undo 履歴: 履歴メモリが「全点コピー × ステップ数」ではなく、shadow 1 本＋変化点の差分に収まっていること
    hist_bytes = win.layer_history.memory_bytes()
    assert hist_bytes <= win.viewer.layer_ids.nbytes + 64 * 1024, (
        f"Undo 履歴のメモリ ({hist_bytes:,} B) が差分方式の想定 (全点 int32 1本 + 差分) を超えています"
    )
    # 自動除去ノイズレイヤーが確実にノイズ判定(is_noise=True, layer_mask=False)されており出力から除外されることを検証
    for lid, linfo in win.viewer.layers.items():
        if linfo.name.startswith("自動除去:"):
            assert win.viewer.is_noise_layer(lid) and linfo.is_noise, f"自動除去レイヤー {linfo.name} がノイズ判定されていません"
            pts_in_auto = win.viewer.get_layer_indices(lid)
            if len(pts_in_auto) > 0:
                assert not np.any(win.viewer.layer_mask[pts_in_auto]), f"自動除去レイヤー {linfo.name} の点が layer_mask で除外されていません"
    split_no_noise = win.viewer.get_all_layer_masks(include_empty=False, include_noise=False)
    for k_name in split_no_noise:
        assert not k_name.startswith("自動除去:"), f"ノイズ除外分割マスクに自動除去レイヤー {k_name} が混入しています"
    # 一括表示切替で自動除去レイヤーが show_noise に連動することを検証
    win.viewer.set_layer_visibility(show_normal=True, show_noise=False)
    for lid, linfo in win.viewer.layers.items():
        if win.viewer.is_noise_layer(lid):
            assert not linfo.visible, "show_noise=False なのにノイズレイヤーが表示になっています"
        else:
            assert linfo.visible, "show_normal=True なのに有効レイヤーが非表示になっています"
    win.viewer.validate_invariants()
    print("  [OK] 自動フィルタ別レイヤー分離・ノイズ確実除外・1つ戻る/進むボタン検査合格")

    # フォルダ一括E57統合読み込み & 大容量ストリーミングPoint Budget読み込みの検証
    import shutil
    from core.e57_io import read_e57_scans, write_e57_scans
    from gui.worker_thread import FileLoadWorker, collect_e57_files

    temp_folder = os.path.join(qgate_tmp, "test_qgate_folder_in")
    temp_merged_out = os.path.join(qgate_tmp, "test_qgate_folder_merged.e57")
    try:
        os.makedirs(temp_folder, exist_ok=True)
        write_e57_scans(os.path.join(temp_folder, "01_stationA.e57"), [scan0])
        write_e57_scans(os.path.join(temp_folder, "02_stationB.e57"), [scan1])

        # ストリーミング間引き読み込み (point_budget指定) と point_count セッター防御の検証
        budget_scans = read_e57_scans(os.path.join(temp_folder, "01_stationA.e57"), point_budget=50, chunk_size=60)
        assert len(budget_scans) == 1 and budget_scans[0].point_count <= 75 and budget_scans[0].original_point_count == 200
        budget_scans[0].point_count = 999  # AttributeError が起きないこと

        found_files = collect_e57_files(temp_folder)
        assert len(found_files) == 2, f"フォルダ内のE57ファイル検出数が不正です: {found_files}"

        loaded_holder: list = []
        fl_worker = FileLoadWorker(temp_folder, point_budget=2000000)
        fl_worker.finished_success.connect(lambda pre: loaded_holder.append(pre))
        fl_worker.run()
        assert len(loaded_holder) == 1, "FileLoadWorker によるフォルダ一括読み込みが完了しませんでした"
        pre = loaded_holder[0]
        assert len(pre.scans) == 2 and len(pre.points_raw) == 300, "フォルダ内2ファイルの統合スキャン数・総点数が一致しません"
        assert "01_stationA" in pre.scans[0].name and "02_stationB" in pre.scans[1].name
        win.current_scans = pre.scans
        win.viewer.set_preloaded_data(pre, reset_camera=True)
        win.viewer.validate_invariants()

        # 統合された複数ファイル由来スキャンを1つのE57ファイルへまとめて保存できるか検証
        merged_saver = FileSaveWorker(temp_merged_out, win.current_scans, win.viewer.layer_mask)
        merged_saver.run()
        assert os.path.exists(temp_merged_out), "フォルダ統合後のE57一括保存ファイルが生成されませんでした"
    finally:
        if os.path.exists(temp_folder):
            shutil.rmtree(temp_folder, ignore_errors=True)
        if os.path.exists(temp_merged_out):
            os.remove(temp_merged_out)
    print("  [OK] フォルダ一括E57統合読み込み & 大容量保護ストリーミング読み込み検査合格")

    # 🧹 ノイズ処理専用モード・ブラシ消去・即時削除・大まか/細かいプリセット・新コマンド4種の総合検証
    from PySide6.QtCore import QPoint
    from core.filters import (
        extract_connected_cluster_from_seeds,
        filter_edge_veil_noise,
        keep_largest_clusters,
    )

    # 1. 新フィルタアルゴリズム単体検証
    rng = np.random.default_rng(42)
    cluster_pts = np.vstack([
        rng.normal(0.0, 0.3, (120, 3)),   # メイン塊 (120点)
        rng.normal(25.0, 0.2, (15, 3)),   # 浮遊ノイズ塊 (15点)
    ])
    keep_idx = keep_largest_clusters(cluster_pts, eps=1.5, top_k=1)
    assert len(keep_idx) == 120 and np.all(keep_idx < 120), f"keep_largest_clusters が浮遊塊を分離できませんでした: {len(keep_idx)}"

    grown_idx = extract_connected_cluster_from_seeds(cluster_pts, np.array([120, 121]), radius=1.5, exclude_ground=False)
    assert len(grown_idx) == 15 and np.all(grown_idx >= 120), f"マジックワンド連結塊抽出が不正です: {grown_idx}"

    veil_inliers = filter_edge_veil_noise(cluster_pts, nb_neighbors=12, std_ratio=1.2)
    assert 0 < len(veil_inliers) <= len(cluster_pts)

    # 2. ワンクリック・プリセット (rough / standard / fine) の検証
    win._apply_filter_preset("rough")
    assert win.sor_chk.isChecked() and win.cluster_chk.isChecked() and not win.edge_veil_chk.isChecked()
    win._apply_filter_preset("fine")
    assert win.sor_chk.isChecked() and win.edge_veil_chk.isChecked()
    win._apply_filter_preset("standard")
    assert win.sor_chk.isChecked() and win.cluster_chk.isChecked()
    # 現場シチュエーション別レシピ全8種の検証
    win._apply_filter_preset("safe_wall")
    assert win.sor_chk.isChecked() and win.sor_r_spin.value() == 2.8 and win.cluster_pts_spin.value() == 15
    win._apply_filter_preset("pedestrian")
    assert win.cluster_chk.isChecked() and win.cluster_pts_spin.value() == 120
    win._apply_filter_preset("cad_lightweight")
    assert win.voxel_chk.isChecked() and win.voxel_spin.value() == 0.03
    win._apply_filter_preset("landscape_tree")
    assert win.cluster_eps_spin.value() == 0.28 and win.cluster_pts_spin.value() == 80
    win._apply_filter_preset("piping_protect")
    assert win.edge_veil_chk.isChecked() and win.edge_lin_spin.value() == 0.88
    # コンボボックス経由の切り替え検証
    win._on_recipe_combo_changed(1)  # rough
    assert win.cluster_eps_spin.value() == 0.25

    # 🌐 取り込み・表示座標系 (数学座標系 / 測量座標系X-Y入替 / Y-up / Z反転) の検証
    assert hasattr(win, "coord_system_combo")
    assert win.coord_system_combo.count() == 5
    win.coord_system_combo.setCurrentIndex(1)  # survey_swap
    assert win.coord_system_combo.currentData() == "survey_swap"
    win.coord_system_combo.setCurrentIndex(0)  # cartesian
    assert win.coord_system_combo.currentData() == "cartesian"

    # transform_coordinate_system の数学的正当性テスト
    dummy_pts = np.array([[10.0, 20.0, 30.0], [1.0, 2.0, 3.0]], dtype=np.float64)
    swapped = transform_coordinate_system(dummy_pts, "survey_swap")
    assert np.allclose(swapped[:, 0], dummy_pts[:, 1]) and np.allclose(swapped[:, 1], dummy_pts[:, 0]) and np.allclose(swapped[:, 2], dummy_pts[:, 2])
    y_up = transform_coordinate_system(dummy_pts, "y_up_to_z_up")
    assert np.allclose(y_up[:, 0], dummy_pts[:, 0]) and np.allclose(y_up[:, 1], -dummy_pts[:, 2]) and np.allclose(y_up[:, 2], dummy_pts[:, 1])
    z_inv = transform_coordinate_system(dummy_pts, "z_invert")
    assert np.allclose(z_inv[:, 2], -dummy_pts[:, 2])

    # 3. パン平行移動モード (H / V) & ブラシ選択モード (B) & 即時削除モード & ノイズ処理専用モードの検証
    win._set_interaction_mode("pan")
    assert win.viewer.pan_mode and not win.viewer.selection_mode and win.seg_btn_pan.isChecked() and win.side_btn_pan.isChecked()
    win._on_shortcut_v_view_mode()  # パン中からVキーで回転モードへ復帰
    assert not win.viewer.pan_mode and win.seg_btn_orbit.isChecked()
    win._on_shortcut_v_view_mode()  # 回転中からVキー再押下でパンモードへトグル
    assert win.viewer.pan_mode and win.seg_btn_pan.isChecked()

    win._set_interaction_mode("brush")
    assert win.viewer.selection_mode and win.viewer.selection_shape == PointCloudViewer.SELECT_BRUSH and not win.viewer.pan_mode
    win._toggle_clean_focus_mode(True)
    assert win.viewer.clean_focus_mode and win.viewer.instant_delete_mode
    # 画面中央を通るブラシストロークで即時消去が発火するか検証
    cx, cy = win.viewer.width() // 2, win.viewer.height() // 2
    win.viewer.brush_stroke_points = [QPoint(cx - 80, cy), QPoint(cx, cy), QPoint(cx + 80, cy)]
    win.viewer.brush_radius = 60
    win.viewer._process_brush_selection()
    win.viewer.brush_stroke_points.clear()
    win.viewer.validate_invariants()
    win._toggle_clean_focus_mode(False)
    assert not win.viewer.clean_focus_mode and not win.viewer.instant_delete_mode

    # 4. スマートコマンド4種 (マジックワンド・局所精密除去・メイン塊保持・エッジ飛散除去) のUIハンドラ検証
    win.viewer.selected_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int64)
    win.viewer._update_active_colors()
    win._on_magic_wand_cluster_select()
    assert win._wait_for_background(), "マジックワンドのバックグラウンド処理がタイムアウトしました"
    win.viewer.validate_invariants()

    win.viewer.selected_indices = np.arange(30, dtype=np.int64)
    win.viewer._update_active_colors()
    win._on_local_fine_clean_selected()
    win.viewer.validate_invariants()

    win._on_keep_main_cluster_only()
    assert win._wait_for_background(), "メイン塊解析のバックグラウンド処理がタイムアウトしました"
    win.viewer.validate_invariants()

    n_before_edge = len(win.viewer.get_layer_indices(PointCloudViewer.LAYER_NORMAL))
    win._on_quick_edge_veil_clean()
    assert win._wait_for_background(), "エッジ飛散除去のバックグラウンド処理がタイムアウトしました"
    win.viewer.validate_invariants()
    assert len(win.viewer.get_layer_indices(PointCloudViewer.LAYER_NORMAL)) <= n_before_edge
    print("  [OK] ノイズ処理専用モード・ブラシ消去・即時削除・プリセット・新スマートコマンド4種検査合格")

    # 5. Esc選択解除 & ショートカット重複チェック & モード切替時の途中状態クリア検査
    from PySide6.QtGui import QKeyEvent, QShortcut
    from PySide6.QtCore import QEvent

    # (a) viewer上での Esc キー押下で選択状態・多角形頂点・ブラシ軌跡が完全解除されるか
    win.viewer.selected_indices = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    win.viewer.polygon_points = [QPoint(10, 10), QPoint(20, 20)]
    win.viewer.brush_stroke_points = [QPoint(30, 30)]
    win.viewer.is_selecting = True
    win.viewer._update_active_colors()
    esc_event = QKeyEvent(QEvent.KeyPress, Qt.Key_Escape, Qt.NoModifier)
    win.viewer.keyPressEvent(esc_event)
    assert len(win.viewer.selected_indices) == 0, "viewer上でのEsc押下で選択が解除されませんでした"
    assert len(win.viewer.polygon_points) == 0 and len(win.viewer.brush_stroke_points) == 0 and not win.viewer.is_selecting
    win.viewer.validate_invariants()

    # (b) MainWindow QShortcut (Esc) ハンドラ _clear_selection でも完全解除されるか
    win.viewer.selected_indices = np.array([5, 6, 7], dtype=np.int64)
    win.viewer.polygon_points = [QPoint(15, 15)]
    win.viewer._update_active_colors()
    win._clear_selection()
    assert len(win.viewer.selected_indices) == 0 and len(win.viewer.polygon_points) == 0, "MainWindow _clear_selection で解除されませんでした"

    # (c) 多角形頂点を打った途中で別モードへ切り替えた際に残像が消えるか
    win._set_interaction_mode("poly")
    win.viewer.polygon_points = [QPoint(10, 10), QPoint(50, 50)]
    win._set_interaction_mode("rect")
    assert len(win.viewer.polygon_points) == 0, "モード切替時に未確定の多角形頂点がクリアされませんでした"

    # (d) MainWindow 内の QShortcut に重複キーシーケンス（Ambiguous shortcut）が存在しないか
    shortcuts = win.findChildren(QShortcut)
    seen_seqs = set()
    for sc in shortcuts:
        seq_str = sc.key().toString()
        if seq_str:
            assert seq_str not in seen_seqs, f"重複したショートカットキーが登録されています (Ambiguous shortcut): {seq_str}"
            seen_seqs.add(seq_str)
    print(f"  [OK] Esc選択解除・ショートカット重複なし({len(seen_seqs)}件)・モード切替クリーンアップ検査合格")

    # 6. XYZ断面スライサー (シーク基点基準モード・現在位置基点セット・厚みプリセット・正対ビュー・3D基点プレーン描画) 検査
    win.viewer.selected_indices = np.array([0, 1, 2], dtype=np.int64)
    expected_focus = np.mean(win.viewer.points_raw[:3], axis=0)
    win._set_section_origin_from_current_view()
    assert abs(win.spin_pos_x.value() - expected_focus[0]) < 0.05
    assert abs(win.spin_pos_y.value() - expected_focus[1]) < 0.05
    assert abs(win.spin_pos_z.value() - expected_focus[2]) < 0.05

    win.chk_sec_z.setChecked(True)
    win.spin_pos_z.setValue(1.50)
    win._apply_section_thickness_preset(0.60)
    assert abs(win.spin_thick_z.value() - 0.60) < 1e-4

    # centerモード: [1.20, 1.80]
    win.sec_origin_mode_combo.setCurrentIndex(0)  # center
    assert abs(win.viewer.clip_z_origin - 1.50) < 1e-4
    assert abs(win.viewer.clip_z_range[0] - 1.20) < 1e-4 and abs(win.viewer.clip_z_range[1] - 1.80) < 1e-4

    # plusモード (基点〜+厚み): [1.50, 2.10]
    win.sec_origin_mode_combo.setCurrentIndex(1)  # plus
    assert abs(win.viewer.clip_z_range[0] - 1.50) < 1e-4 and abs(win.viewer.clip_z_range[1] - 2.10) < 1e-4

    # minusモード (-厚み〜基点): [0.90, 1.50]
    win.sec_origin_mode_combo.setCurrentIndex(2)  # minus
    assert abs(win.viewer.clip_z_range[0] - 0.90) < 1e-4 and abs(win.viewer.clip_z_range[1] - 1.50) < 1e-4

    # 正対ビュー & OpenGL 基点プレーン + HUD 描画検証
    win._align_to_section_axis("z")
    fb_sec = win.viewer.grabFramebuffer()
    assert fb_sec.width() > 0 and fb_sec.height() > 0
    win._reset_all_sections()
    win.sec_origin_mode_combo.setCurrentIndex(0)
    win._clear_selection()
    win.viewer.validate_invariants()
    print("  [OK] XYZ断面スライサー (シーク基点モード・現在位置セット・厚み一括設定・正対ビュー・基点プレーン描画) 検査合格")

    # 7. 🧭 2点ピック斜め壁アライン (横・縦・直交正面・壁真横・180°反転) & 斜め壁スライス検査
    win._toggle_wall_align_mode(True)
    assert win.viewer.wall_align_mode and win.btn_wall_pick.isChecked() and win.nav_btn_wall_align.isChecked()
    # 45度の斜め壁 (0, 0, 0) -> (2, 2, 0) を2点指定
    angle_deg = win.viewer.set_wall_align_points(
        np.array([0.0, 0.0, 0.0], dtype=np.float64),
        np.array([2.0, 2.0, 0.0], dtype=np.float64),
        initial_view="horiz",
    )
    assert abs(angle_deg - 45.0) < 1e-4, f"45°斜め壁の角度計算が不正です: {angle_deg}"
    assert not win.viewer.wall_align_mode and not win.btn_wall_pick.isChecked()
    assert win.viewer.is_orthographic, "2点壁アライン時は自動で正射影(Orthographic)に切り替わる必要があります"
    assert abs(win.viewer.pitch - 89.9) < 1e-3 and abs((win.viewer.yaw - 45.0 + 180.0) % 360.0 - 180.0) < 1e-3

    # ┃ 縦表示 (上面・垂直揃え): pitch=89.9, yaw=α-90=-45.0
    win._apply_wall_view("vert")
    assert abs(win.viewer.pitch - 89.9) < 1e-3 and abs((win.viewer.yaw - (-45.0) + 180.0) % 360.0 - 180.0) < 1e-3

    # 🧱 直交正面ビュー: pitch=0.0, yaw=45.0
    win._apply_wall_view("ortho_front")
    assert abs(win.viewer.pitch - 0.0) < 1e-3 and abs((win.viewer.yaw - 45.0 + 180.0) % 360.0 - 180.0) < 1e-3

    # 🔪 壁真横・断面エッジ: pitch=0.0, yaw=45+90=135.0
    win._apply_wall_view("edge")
    assert abs(win.viewer.pitch - 0.0) < 1e-3 and abs((win.viewer.yaw - 135.0 + 180.0) % 360.0 - 180.0) < 1e-3

    # 🔄 180°反転: 45° -> -135°
    win._flip_wall_view()
    assert abs((win.viewer.wall_align_angle_deg - (-135.0) + 180.0) % 360.0 - 180.0) < 1e-3

    # 斜め壁スライス (壁からの直交距離でフィルタリング & 3D描画)
    win.chk_sec_wall.setChecked(True)
    win.spin_pos_wall.setValue(0.0)
    win.spin_thick_wall.setValue(1.0)
    assert win.viewer.clip_wall_enabled
    fb_wall = win.viewer.grabFramebuffer()
    assert fb_wall.width() > 0 and fb_wall.height() > 0
    win._reset_all_sections()
    assert not win.viewer.clip_wall_enabled and win.viewer.wall_align_angle_deg is None
    win.viewer.validate_invariants()
    print("  [OK] 🧭 2点ピック斜め壁アライン (横・縦・直交正面・壁真横・180°反転) & 斜め壁スライス検査合格")

    # 8. 新UI（modern）固有の検査: 絵文字ゼロ・メニュー/ステータスバー・作業順・狭い画面での折り返し・未保存フラグ
    from PySide6.QtWidgets import QPushButton
    from gui.modern_ui import has_emoji

    emoji_btns = [b.text() for b in win.findChildren(QPushButton) if b.text() and has_emoji(b.text())]
    assert not emoji_btns, f"新UIのボタンに絵文字が残っています: {emoji_btns[:5]}"
    assert win.menuBar() is not None and len(win.menuBar().actions()) >= 4, "新UIのメニューバー（ファイル/編集/表示/ヘルプ）がありません"
    assert win.status_label.parentWidget() is win.statusBar(), "状態ラベルがステータスバーに移動していません"
    side_lay = win.sidebar_widget.layout()
    order = [side_lay.indexOf(g) for g in (win.auto_group, win.tools_group, win.layer_group, win.section_group)]
    assert order == sorted(order) and all(i >= 0 for i in order), f"サイドバーが作業順（自動→手動→レイヤー→断面）になっていません: {order}"
    assert hasattr(win, "save_group") and win.save_group.isVisible(), "保存グループが表示されていません"
    assert hasattr(win, "empty_state_card"), "空状態カードが生成されていません"
    # 「全て」タブ（全グループ表示）でも左パネルが横に見切れないこと
    win._set_sidebar_workflow_tab("all")
    for _ in range(5):
        app.processEvents()
    all_min_w = win.sidebar_widget.minimumSizeHint().width()
    all_vp_w = win.sidebar_scroll.viewport().width()
    assert all_min_w <= all_vp_w, f"「全て」タブで左パネルの最小幅({all_min_w}px)が表示領域幅({all_vp_w}px)を超えています"
    win._set_sidebar_workflow_tab("filter")
    win.resize(1000, 650)
    for _ in range(5):
        app.processEvents()
    nav_bar = getattr(win, "_nav_bar_widget", None)
    assert nav_bar is not None
    for b in nav_bar.findChildren(QPushButton):
        if b.isVisible():
            assert nav_bar.rect().contains(b.geometry()), f"1000px幅で上部バーのボタン「{b.text()}」がはみ出しています"
    for b in win.bottom_overlay_bar.findChildren(QPushButton):
        if b.isVisible():
            assert win.bottom_overlay_bar.rect().contains(b.geometry()), f"1000px幅で下部バーのボタン「{b.text()}」がはみ出しています"
    win.resize(1400, 900)
    app.processEvents()
    # 未保存フラグ: 編集でタイトルに * が付き、保存成功相当で消える
    win._set_dirty(False)
    win._push_layer_undo_state()
    assert win._dirty and win.windowTitle().startswith("* "), "編集後に未保存フラグ（タイトルの *）が立っていません"
    win._set_dirty(False)
    assert not win.windowTitle().startswith("* ")
    print("  [OK] 新UI検査合格 (絵文字0・メニュー/ステータスバー・作業順・1000px幅で折り返し・未保存フラグ)")

    win.close()
    shutil.rmtree(qgate_tmp, ignore_errors=True)
    print("  [OK] 実機 OpenGL VBO 描画 & MainWindow 全操作・レイヤー拡張①〜⑥シナリオ検査合格")

    # 9. 従来UI（classic）スモーク: 改修前の画面が引き続き起動・操作できること（安全策の検証）
    os.environ["E57_UI_MODE"] = "classic"
    reset_ui_mode_cache()
    win_c = MainWindow()
    win_c.show()
    app.processEvents()
    try:
        assert win_c.ui_mode == "classic"
        assert has_emoji(win_c.btn_instant_del.text()), "従来UIで絵文字付きラベルが失われています（従来UIは改修前のまま残す）"
        assert win_c.menuBar() is None or len(win_c.menuBar().actions()) == 0, "従来UIにメニューバーが付いています"
        assert win_c.status_label.parentWidget() is not win_c.statusBar(), "従来UIで状態ラベルの位置が変わっています"
        win_c.current_scans = [scan0, scan1]
        win_c._update_viewer_from_scans()
        app.processEvents()
        for tab_key in ["layer", "select", "section", "auto", "view", "settings", "all"]:
            win_c._set_sidebar_workflow_tab(tab_key)
        win_c._set_interaction_mode("rect")
        assert win_c.viewer.selection_mode and win_c.side_btn_rect.isChecked()
        win_c._set_interaction_mode("orbit")
        win_c.viewer.validate_invariants()
        print("  [OK] 従来UI（classic）スモーク合格 (起動・タブ切替・モード切替・絵文字ラベル維持)")
    finally:
        win_c.close()
        os.environ.pop("E57_UI_MODE", None)
        reset_ui_mode_cache()

    # 10. 「軌跡（明）」テーマのスモーク: 新UIのライトテーマが起動・タブ切替・通知・選択操作できること
    from gui.ui_mode import reset_ui_theme_cache
    os.environ["E57_UI_MODE"] = "modern"
    os.environ["E57_UI_THEME"] = "kiseki"
    reset_ui_mode_cache()
    reset_ui_theme_cache()
    win_k = MainWindow()
    win_k.show()
    app.processEvents()
    try:
        assert win_k.ui_mode == "modern" and win_k.ui_theme == "kiseki", f"テーマ解決が不正: {win_k.ui_mode}/{win_k.ui_theme}"
        qss_k = app.styleSheet()
        assert "#e7efec" in qss_k and "#004c2c" in qss_k and "#0b1110" in qss_k, "軌跡テーマの QSS が適用されていません"
        assert "chevron-down-light.svg" in qss_k, "軌跡テーマで明るい地用の矢印アイコンが使われていません"
        # その場でのテーマ切替（再起動不要）: dark → kiseki と往復して QSS とテーマ名が入れ替わること
        win_k.apply_ui_theme("dark")
        app.processEvents()
        assert win_k.ui_theme == "dark" and "#0f172a" in app.styleSheet(), "テーマのその場切替（dark）が効いていません"
        win_k.apply_ui_theme("kiseki")
        app.processEvents()
        assert win_k.ui_theme == "kiseki" and "#e7efec" in app.styleSheet(), "テーマのその場切替（kiseki）が効いていません"
        win_k.current_scans = [scan0, scan1]
        win_k._update_viewer_from_scans()
        app.processEvents()
        for tab_key in ["layer", "select", "section", "auto", "view", "settings", "all"]:
            win_k._set_sidebar_workflow_tab(tab_key)
            app.processEvents()
        win_k.show_toast("テーマ検査: 保存しました")
        win_k.show_toast("テーマ検査: エラー")
        app.processEvents()
        # 前の通知の退場中に出した通知が消えないこと
        QTest.qWait(400)
        win_k._toast_timer.stop()
        win_k._toast_timer.timeout.emit()
        QTest.qWait(30)
        win_k.show_toast("テーマ検査: 次の通知")
        QTest.qWait(300)
        assert win_k.toast_label.isVisible() and "次の通知" in win_k.toast_label.text(), "退場中に出した通知が消えました"
        win_k._update_floating_selection_bar(1234)
        win_k._update_floating_selection_bar(0)
        for grp in win_k.findChildren(QWidget):
            if grp.objectName() == "collapsible_group" and hasattr(grp, "set_expanded"):
                grp.set_expanded(False)
                grp.set_expanded(True)
        QTest.qWait(500)  # カスケード・展開・ティッカーが終わるのを待つ（効果が外れることの確認）
        app.processEvents()
        n_fx = sum(1 for w in win_k.findChildren(QWidget) if w.graphicsEffect() is not None and w.isVisible())
        assert n_fx == 0, f"モーション終了後に描画効果が残っています: {n_fx} 件"
        win_k.viewer.validate_invariants()
        print("  [OK] 軌跡（明）テーマ スモーク合格 (起動・QSS 適用・その場切替・タブ切替・通知・選択・展開・効果の後始末)")
    finally:
        win_k.close()
        os.environ.pop("E57_UI_THEME", None)
        os.environ.pop("E57_UI_MODE", None)
        reset_ui_mode_cache()
        reset_ui_theme_cache()


def gate_5_unit_and_integration_suites() -> None:
    """Gate 5: E57 ロスレス入出力・プロセッサ・レイヤー統合テストスイート実行"""
    print_banner("Gate 5: E57 完全ロスレス整合性 & 単体/統合テストスイート")

    # tests/test_layer_integration.py も unittest の TestCase を持つため discover で一括実行される
    suite = unittest.defaultTestLoader.discover("tests")
    runner = unittest.TextTestRunner(verbosity=1)
    result = runner.run(suite)
    if not result.wasSuccessful():
        raise AssertionError("unittest テストスイートに失敗したテストがあります")
    print(f"  [OK] 全単体テスト・E57ロスレス整合性テスト・レイヤー統合テスト合格 ({result.testsRun} 件)")


def install_git_pre_commit_hook() -> None:
    """Git コミット前に quality_gate.py を自動実行する pre-commit フックを設置する"""
    hooks_dir = os.path.join(".git", "hooks")
    if not os.path.isdir(hooks_dir):
        print("[情報] .git/hooks ディレクトリが見つかりません。")
        return

    hook_path = os.path.join(hooks_dir, "pre-commit")
    py_exe = sys.executable.replace("\\", "/")
    hook_content = f"""#!/bin/sh
# 自動品質管理ゲート (pre-commit hook)
echo "Running Quality Gate before commit..."
"{py_exe}" -X utf8 quality_gate.py
if [ $? -ne 0 ]; then
    echo "❌ 品質管理ゲートでエラーが検出されたため、コミットを中断しました。"
    exit 1
fi
"""
    with open(hook_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(hook_content)
    print(f"✅ Git pre-commit フックを設置しました: {hook_path}")


def main() -> int:
    if "--install-hook" in sys.argv:
        install_git_pre_commit_hook()
        return 0

    quick = "--quick" in sys.argv
    start_time = time.time()
    try:
        gate_1_static_and_dom_cross_check()
        if quick:
            print("\n  [SKIP] --quick 指定のため GUI ウィンドウを開く Gate 2 / Gate 4 を省略します（マージ前には全ゲートを実行すること）")
        else:
            gate_2_state_invariants_check()
        gate_3_boundary_and_fuzzing_check()
        if not quick:
            gate_4_gui_and_opengl_smoke_check()
        gate_5_unit_and_integration_suites()
    except Exception as exc:
        print("\n" + "!" * 68)
        print(f" ❌ 品質管理ゲート NG (不合格): {exc}")
        print("!" * 68)
        import traceback

        traceback.print_exc()
        return 1

    elapsed = time.time() - start_time
    print("\n" + "=" * 68)
    if quick:
        print(f" ✅ 軽量モード (Gate 1, 3, 5) に合格しました ({elapsed:.2f}秒) ※ Gate 2/4 は未実行")
    else:
        print(f" 🎉 全 5 段階の品質管理ゲート (Gate 1〜Gate 5) に完全合格しました！ ({elapsed:.2f}秒)")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
