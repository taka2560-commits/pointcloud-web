# -*- coding: utf-8 -*-
"""
UI/UX レビュー用スクリーンショット自動取得ツール

メインウィンドウにサンプル点群を読み込み、各ワークフロータブ・進捗ダイアログ・選択状態などの
画面を PNG として docs/ui_review/ に保存する（実機 OpenGL が必要。品質ゲート Gate4 と同じ前提）。

使い方:
    python -X utf8 tools/capture_ui_screens.py [出力フォルダ] [幅] [高さ]
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QTest

from gui.main_window import MainWindow
from gui.progress_dialog import ModernProgressDialog
from gui.worker_thread import FileLoadWorker


def _grab_window_with_gl(win: MainWindow):
    """
    実画面をそのままキャプチャする（QScreen.grabWindow）。
    win.grab() は QOpenGLWidget 上の QPainter オーバーレイ（ギズモのバッジ・HUD）を含まず、
    grabFramebuffer() の合成はビューア上に重なるフローティングバーを隠してしまうため、
    ユーザーが実際に見ている画面と一致する方法として画面キャプチャを使う。
    （ウィンドウが他のウィンドウに隠れていないことが前提）
    """
    screen = win.screen() or QApplication.primaryScreen()
    try:
        # grabWindow(winId) は OpenGL 合成ウィンドウでは最初の1枚以降が更新されない（古いフレームが返る）ため、
        # デスクトップ全体（合成済み）を撮ってウィンドウのクライアント領域を物理ピクセルで切り出す。
        full = screen.grabWindow(0)
        dpr = full.devicePixelRatio() if full.devicePixelRatio() > 0 else screen.devicePixelRatio()
        g = win.geometry()
        sg = screen.geometry()
        x = int((g.x() - sg.x()) * dpr)
        y = int((g.y() - sg.y()) * dpr)
        w = int(g.width() * dpr)
        h = int(g.height() * dpr)
        return full.copy(x, y, w, h)
    except Exception as exc:
        print("  [warn] full-screen grab failed, fallback to win.grab():", exc)
        return win.grab()


def main() -> int:
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join("docs", "ui_review")
    width = int(sys.argv[2]) if len(sys.argv) > 2 else 1600
    height = int(sys.argv[3]) if len(sys.argv) > 3 else 900
    os.makedirs(out_dir, exist_ok=True)

    app = QApplication.instance() or QApplication(sys.argv)
    win = MainWindow()
    win.resize(width, height)
    win.show()
    app.processEvents()

    def shot(name: str, widget=None):
        # レイアウト・GL 描画・アニメーションが落ち着くまで待つ（実画面キャプチャのため）。
        # 明示的に viewer.update() を呼ばないと画面上の GL が再描画されず古い画面が撮れることがある
        win.viewer.update()
        QTest.qWait(500)
        for _ in range(6):
            app.processEvents()
        pix = _grab_window_with_gl(win) if widget is None else widget.grab()
        path = os.path.join(out_dir, f"{name}.png")
        pix.save(path)
        print("  saved", path)

    # 1. 起動直後（空状態）
    shot("01_empty_startup")

    # 2. サンプル点群を読み込み
    sample = os.path.join("samples", "サンプル点群.e57")
    if os.path.exists(sample):
        holder = []
        w = FileLoadWorker(sample, point_budget=2_000_000)
        w.finished_success.connect(lambda pre: holder.append(pre))
        w.finished_error.connect(lambda m: print("  [error] load:", m))
        w.run()
        print("  loaded scans:", len(holder[0].scans) if holder else 0, "points:", len(holder[0].points_raw) if holder else 0)
        if holder:
            pre = holder[0]
            win.current_scans = pre.scans
            win.viewer.set_preloaded_data(pre, reset_camera=True)
            win.viewer.fit_to_screen()
            win.viewer.update()
            QTest.qWait(800)
            app.processEvents()
    else:
        print("  [warn] sample not found; using synthetic points")
        pts = np.random.normal(0, 3, (20000, 3))
        win.viewer.set_point_cloud(pts)

    shot("02_loaded_default")

    # 3. ワークフロータブごと
    for key in ["layer", "select", "section", "auto", "view", "settings", "all"]:
        try:
            win._set_sidebar_workflow_tab(key)
            shot(f"03_tab_{key}")
        except Exception as exc:
            print("  [warn] tab", key, exc)
    win._set_sidebar_workflow_tab("all")

    # 4. 配色モード（高さ / 反射強度 / レイヤー）
    from gui.gl_viewer import PointCloudViewer
    for name, mode in [
        ("height", PointCloudViewer.COLOR_MODE_HEIGHT),
        ("intensity", PointCloudViewer.COLOR_MODE_INTENSITY),
        ("layer", PointCloudViewer.COLOR_MODE_LAYER),
    ]:
        try:
            win.viewer.set_color_mode(mode)
            shot(f"04_color_{name}")
        except Exception as exc:
            print("  [warn] color", name, exc)

    # 5. 選択状態（下部フローティングバー・バッジ）
    try:
        n = len(win.viewer.points_raw)
        win._set_interaction_mode("rect")
        win.viewer._apply_selection_with_modifiers(np.arange(0, max(1, n // 5)), Qt.NoModifier)
        app.processEvents()
        shot("05_selection_floating_bar")
        win._clear_selection()
    except Exception as exc:
        print("  [warn] selection", exc)

    # 6. ノイズレイヤー表示・断面スライサー ON
    try:
        win.viewer.move_indices_to_noise(np.arange(0, max(1, len(win.viewer.points_raw) // 8)))
        win.chk_layer_noise.setChecked(True)
        win.viewer.set_color_mode(PointCloudViewer.COLOR_MODE_LAYER)
        shot("06_noise_layer_visible")
        win.chk_sec_z.setChecked(True)
        win._align_to_section_axis("z")
        shot("07_section_slicer_z")
        win._reset_all_sections()
    except Exception as exc:
        print("  [warn] section", exc)

    # 7. トースト通知
    try:
        win.show_toast("✅ 保存が完了しました（トースト表示の例）", duration_ms=5000)
        shot("08_toast")
    except Exception as exc:
        print("  [warn] toast", exc)

    # 8. 進捗ダイアログ
    try:
        dlg = ModernProgressDialog("点群データ読み込み中", win)
        dlg.set_progress(63, "E57バイナリ読み込み中...", "ファイル: サンプル点群.e57")
        dlg.show()
        for _ in range(30):
            app.processEvents()
        shot("09_progress_dialog", dlg)
        dlg.close()
    except Exception as exc:
        print("  [warn] progress", exc)

    # 9. 右クリックメニュー相当（スクリーンショットはメニューを含めにくいため省略）、狭い幅での表示
    try:
        win.resize(1100, 700)
        QTest.qWait(600)
        for _ in range(10):
            app.processEvents()
        shot("10_narrow_1100x700")
    except Exception as exc:
        print("  [warn] narrow", exc)

    win.close()
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
