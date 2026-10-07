"""
PySide6によるE57点群ノイズ処理ツールのメインウィンドウ
3D点群ビューア（OpenGL）をシームレスに統合し、表示操作および手動/自動ノイズ除去を提供します。
"""

import os
from typing import Optional, List
import numpy as np

from PySide6.QtCore import Qt, QSize
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QCheckBox,
    QSpinBox,
    QDoubleSpinBox,
    QProgressBar,
    QTextEdit,
    QFileDialog,
    QMessageBox,
    QTabWidget,
    QSplitter,
    QComboBox,
    QSlider,
    QToolButton,
    QScrollArea,
)
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QKeySequence, QShortcut

from core.processor import FilterConfig, PointCloudProcessor, ProcessSummary
from core.e57_io import E57ScanData, read_e57_scans, write_e57_scans
from gui.worker_thread import ProcessWorker
from gui.gl_viewer import PointCloudViewer
from gui.styles import DARK_THEME_QSS


class MainWindow(QMainWindow):
    """メインウィンドウクラス"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("E57 点群ノイズ処理ツール - 3Dエディタ")
        self.resize(1350, 850)
        self.setMinimumSize(QSize(1000, 700))
        self.setStyleSheet(DARK_THEME_QSS)

        # ドラッグ＆ドロップの有効化
        self.setAcceptDrops(True)

        # 点群データと履歴管理 (Undo用)
        self.current_scans: List[E57ScanData] = []
        self.undo_stack: List[List[E57ScanData]] = []
        self.worker: Optional[ProcessWorker] = None

        self._init_ui()
        self._setup_shortcuts()

    def _init_ui(self):
        """UIコンポーネントの構築"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        root_layout = QHBoxLayout(central_widget)
        root_layout.setContentsMargins(6, 6, 6, 6)
        root_layout.setSpacing(6)

        # 左右スプリッター
        splitter = QSplitter(Qt.Horizontal)
        root_layout.addWidget(splitter)

        # --- 左パネル (操作・設定サイドバー) ---
        sidebar_scroll = QScrollArea()
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setFixedWidth(430)
        sidebar_scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        sidebar_widget = QWidget()
        sidebar_layout = QVBoxLayout(sidebar_widget)
        sidebar_layout.setContentsMargins(8, 8, 8, 8)
        sidebar_layout.setSpacing(10)

        # 1. ファイル入出力
        file_group = QGroupBox("ファイル入出力")
        file_layout = QGridLayout(file_group)

        file_layout.addWidget(QLabel("入力 E57:"), 0, 0)
        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("E57ファイルを指定 / ドロップ")
        self.input_edit.textChanged.connect(self._on_input_file_changed)
        file_layout.addWidget(self.input_edit, 0, 1)

        self.input_browse_btn = QPushButton("参照")
        self.input_browse_btn.setMaximumWidth(60)
        self.input_browse_btn.clicked.connect(self._browse_input_file)
        file_layout.addWidget(self.input_browse_btn, 0, 2)

        file_layout.addWidget(QLabel("出力 E57:"), 1, 0)
        self.output_edit = QLineEdit()
        file_layout.addWidget(self.output_edit, 1, 1)

        self.output_browse_btn = QPushButton("参照")
        self.output_browse_btn.setMaximumWidth(60)
        self.output_browse_btn.clicked.connect(self._browse_output_file)
        file_layout.addWidget(self.output_browse_btn, 1, 2)

        sidebar_layout.addWidget(file_group)

        # 2. 3D表示・カメラ操作パネル
        view_group = QGroupBox("3D表示・操作")
        view_layout = QGridLayout(view_group)

        # 視点プリセット
        view_layout.addWidget(QLabel("視点切替:"), 0, 0)
        view_btn_layout = QHBoxLayout()
        self.btn_view_iso = QPushButton("等角")
        self.btn_view_iso.clicked.connect(lambda: self.viewer.reset_view())
        self.btn_view_top = QPushButton("上面")
        self.btn_view_top.clicked.connect(lambda: self.viewer.set_view_top())
        self.btn_view_front = QPushButton("正面")
        self.btn_view_front.clicked.connect(lambda: self.viewer.set_view_front())
        self.btn_view_side = QPushButton("側面")
        self.btn_view_side.clicked.connect(lambda: self.viewer.set_view_side())

        for b in [self.btn_view_iso, self.btn_view_top, self.btn_view_front, self.btn_view_side]:
            b.setStyleSheet("padding: 4px 8px; font-size: 11px;")
            view_btn_layout.addWidget(b)
        view_layout.addLayout(view_btn_layout, 0, 1)

        # カラーモード
        view_layout.addWidget(QLabel("配色モード:"), 1, 0)
        self.color_combo = QComboBox()
        self.color_combo.addItems([
            PointCloudViewer.COLOR_MODE_RGB,
            PointCloudViewer.COLOR_MODE_HEIGHT,
            PointCloudViewer.COLOR_MODE_INTENSITY,
            PointCloudViewer.COLOR_MODE_SOLID,
        ])
        self.color_combo.currentTextChanged.connect(self._on_color_mode_changed)
        view_layout.addWidget(self.color_combo, 1, 1)

        # 点サイズ
        view_layout.addWidget(QLabel("点サイズ:"), 2, 0)
        size_layout = QHBoxLayout()
        self.size_slider = QSlider(Qt.Horizontal)
        self.size_slider.setRange(1, 10)
        self.size_slider.setValue(2)
        self.size_slider.valueChanged.connect(lambda val: self.viewer.set_point_size(val))
        self.size_label = QLabel("2 px")
        self.size_slider.valueChanged.connect(lambda val: self.size_label.setText(f"{val} px"))
        size_layout.addWidget(self.size_slider)
        size_layout.addWidget(self.size_label)
        view_layout.addLayout(size_layout, 2, 1)

        sidebar_layout.addWidget(view_group)

        # 3. 手動ノイズ除去ツールパネル
        manual_group = QGroupBox("手動ノイズ除去ツール")
        manual_layout = QVBoxLayout(manual_group)

        manual_btn_layout = QHBoxLayout()
        self.select_mode_btn = QPushButton("矩形選択モード")
        self.select_mode_btn.setCheckable(True)
        self.select_mode_btn.toggled.connect(self._toggle_selection_mode)
        manual_btn_layout.addWidget(self.select_mode_btn)

        self.clear_sel_btn = QPushButton("選択解除")
        self.clear_sel_btn.clicked.connect(self._clear_selection)
        manual_btn_layout.addWidget(self.clear_sel_btn)
        manual_layout.addLayout(manual_btn_layout)

        manual_action_layout = QHBoxLayout()
        self.delete_pts_btn = QPushButton("選択点を削除 (Delete)")
        self.delete_pts_btn.setStyleSheet("background-color: #b33939; font-weight: bold;")
        self.delete_pts_btn.clicked.connect(self._delete_selected_points)
        manual_action_layout.addWidget(self.delete_pts_btn)

        self.undo_btn = QPushButton("元に戻す (Ctrl+Z)")
        self.undo_btn.clicked.connect(self._undo_action)
        self.undo_btn.setEnabled(False)
        manual_action_layout.addWidget(self.undo_btn)
        manual_layout.addLayout(manual_action_layout)

        self.manual_status_label = QLabel("※「矩形選択モード」をONにして画面上をドラッグすると点を選択できます")
        self.manual_status_label.setWordWrap(True)
        self.manual_status_label.setStyleSheet("color: #888899; font-size: 11px;")
        manual_layout.addWidget(self.manual_status_label)

        self.save_manual_btn = QPushButton("手動編集した点群をE57保存")
        self.save_manual_btn.clicked.connect(self._save_current_scans)
        manual_layout.addWidget(self.save_manual_btn)

        sidebar_layout.addWidget(manual_group)

        # 4. 自動ノイズ除去フィルタ設定
        auto_group = QGroupBox("自動ノイズ除去フィルタ")
        auto_layout = QVBoxLayout(auto_group)

        tab_widget = QTabWidget()

        # SORタブ
        sor_tab = QWidget()
        sor_l = QGridLayout(sor_tab)
        self.sor_chk = QCheckBox("統計的外れ値除去 (SOR) を有効化")
        self.sor_chk.setChecked(True)
        sor_l.addWidget(self.sor_chk, 0, 0, 1, 2)

        sor_l.addWidget(QLabel("近傍探索点数:"), 1, 0)
        self.sor_k_spin = QSpinBox()
        self.sor_k_spin.setRange(5, 200)
        self.sor_k_spin.setValue(20)
        sor_l.addWidget(self.sor_k_spin, 1, 1)

        sor_l.addWidget(QLabel("標準偏差倍率:"), 2, 0)
        self.sor_r_spin = QDoubleSpinBox()
        self.sor_r_spin.setRange(0.1, 10.0)
        self.sor_r_spin.setSingleStep(0.1)
        self.sor_r_spin.setValue(2.0)
        sor_l.addWidget(self.sor_r_spin, 2, 1)
        tab_widget.addTab(sor_tab, "SOR")

        # RORタブ
        ror_tab = QWidget()
        ror_l = QGridLayout(ror_tab)
        self.ror_chk = QCheckBox("半径外れ値除去 (ROR) を有効化")
        ror_l.addWidget(self.ror_chk, 0, 0, 1, 2)
        ror_l.addWidget(QLabel("探索半径 (m):"), 1, 0)
        self.ror_radius_spin = QDoubleSpinBox()
        self.ror_radius_spin.setRange(0.001, 10.0)
        self.ror_radius_spin.setSingleStep(0.01)
        self.ror_radius_spin.setValue(0.05)
        ror_l.addWidget(self.ror_radius_spin, 1, 1)
        ror_l.addWidget(QLabel("最小点数:"), 2, 0)
        self.ror_pts_spin = QSpinBox()
        self.ror_pts_spin.setRange(1, 100)
        self.ror_pts_spin.setValue(16)
        ror_l.addWidget(self.ror_pts_spin, 2, 1)
        tab_widget.addTab(ror_tab, "ROR")

        # 距離・間引き
        other_tab = QWidget()
        other_l = QGridLayout(other_tab)
        self.voxel_chk = QCheckBox("ボクセル間引き")
        other_l.addWidget(self.voxel_chk, 0, 0, 1, 2)
        other_l.addWidget(QLabel("ボクセルサイズ (m):"), 1, 0)
        self.voxel_spin = QDoubleSpinBox()
        self.voxel_spin.setRange(0.001, 5.0)
        self.voxel_spin.setValue(0.02)
        other_l.addWidget(self.voxel_spin, 1, 1)
        tab_widget.addTab(other_tab, "間引き")

        auto_layout.addWidget(tab_widget)

        self.auto_exec_btn = QPushButton("自動ノイズ処理を一括実行")
        self.auto_exec_btn.setObjectName("execute_button")
        self.auto_exec_btn.clicked.connect(self._run_auto_processing)
        auto_layout.addWidget(self.auto_exec_btn)

        sidebar_layout.addWidget(auto_group)

        # 5. ステータス・ログ
        log_group = QGroupBox("ステータス")
        log_layout = QVBoxLayout(log_group)
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        log_layout.addWidget(self.progress_bar)
        self.status_label = QLabel("E57ファイルを読み込んでください")
        log_layout.addWidget(self.status_label)
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(90)
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        sidebar_layout.addWidget(log_group)

        sidebar_scroll.setWidget(sidebar_widget)
        splitter.addWidget(sidebar_scroll)

        # --- 右パネル (3D点群OpenGLビューア) ---
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)

        # 上部ナビゲーションバー・操作ヘルプ
        nav_bar = QWidget()
        nav_bar.setStyleSheet("background-color: #1e1e24; padding: 4px; border-bottom: 1px solid #333344;")
        nav_layout = QHBoxLayout(nav_bar)
        nav_layout.setContentsMargins(6, 2, 6, 2)

        help_label = QLabel("操作方法: [左ドラッグ] 視点回転 | [右/中ドラッグ] 平行移動(パン) | [ホイール] ズーム | [Delete] 選択点削除")
        help_label.setStyleSheet("color: #a0a0b0; font-size: 11px;")
        nav_layout.addWidget(help_label)
        nav_layout.addStretch()

        self.point_count_label = QLabel("点数: 0 点")
        self.point_count_label.setStyleSheet("color: #4da6ff; font-weight: bold; font-size: 12px;")
        nav_layout.addWidget(self.point_count_label)

        right_layout.addWidget(nav_bar)

        # 3D点群ビューアウィジェット
        self.viewer = PointCloudViewer()
        self.viewer.status_changed.connect(self._on_viewer_status)
        self.viewer.point_selected.connect(self._on_points_selected)
        right_layout.addWidget(self.viewer)

        splitter.addWidget(right_container)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

    def _setup_shortcuts(self):
        """ショートカットキーの設定"""
        # Deleteキーで選択点を削除
        del_shortcut = QShortcut(QKeySequence(Qt.Key_Delete), self)
        del_shortcut.activated.connect(self._delete_selected_points)

        # Ctrl+Zで元に戻す
        undo_shortcut = QShortcut(QKeySequence.Undo, self)
        undo_shortcut.activated.connect(self._undo_action)

    def _on_input_file_changed(self, text: str):
        path = text.strip()
        if path and os.path.exists(path):
            base, ext = os.path.splitext(path)
            suggested_out = f"{base}_cleaned{ext}"
            if not self.output_edit.text() or self.output_edit.text().endswith("_cleaned.e57"):
                self.output_edit.setText(suggested_out)
            # 即座に3Dビューアに読み込み
            self._load_file_to_viewer(path)

    def _browse_input_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "入力 E57 ファイルの選択",
            "",
            "E57 点群ファイル (*.e57);;すべてのファイル (*.*)",
        )
        if path:
            self.input_edit.setText(path)

    def _browse_output_file(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "出力先 E57 ファイルの指定",
            self.output_edit.text(),
            "E57 点群ファイル (*.e57);;すべてのファイル (*.*)",
        )
        if path:
            self.output_edit.setText(path)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if path.lower().endswith(".e57"):
                self.input_edit.setText(path)

    def _load_file_to_viewer(self, file_path: str):
        """E57ファイルを読み込み、3Dビューアに反映する"""
        try:
            self.status_label.setText(f"ファイル読み込み中: {os.path.basename(file_path)}...")
            self.log_text.append(f"読み込み中: {file_path}")

            scans = read_e57_scans(file_path)
            if not scans:
                QMessageBox.warning(self, "警告", "スキャンデータが見つかりませんでした。")
                return

            self.current_scans = scans
            self.undo_stack.clear()
            self.undo_btn.setEnabled(False)

            # 全スキャンの点群を統合してビューアに設定
            self._update_viewer_from_scans()

            total_pts = sum(s.point_count for s in self.current_scans)
            self.log_text.append(f"読み込み成功: スキャン数 {len(scans)}, 総点数 {total_pts:,} 点")
            self.status_label.setText("準備完了")
        except Exception as e:
            QMessageBox.critical(self, "読み込みエラー", f"E57ファイルの読み込みに失敗しました:\n{e}")
            self.status_label.setText("読み込みエラー")

    def _update_viewer_from_scans(self):
        """現在のスキャンリストから点群データをまとめてビューアに転送"""
        if not self.current_scans:
            self.viewer.set_point_cloud(None)
            self.point_count_label.setText("点数: 0 点")
            return

        all_pts = []
        all_cols = []
        all_ints = []

        for s in self.current_scans:
            all_pts.append(s.points)
            if "colorRed" in s.raw_fields and "colorGreen" in s.raw_fields and "colorBlue" in s.raw_fields:
                r = s.raw_fields["colorRed"]
                g = s.raw_fields["colorGreen"]
                b = s.raw_fields["colorBlue"]
                all_cols.append(np.column_stack([r, g, b]))
            if "intensity" in s.raw_fields:
                all_ints.append(s.raw_fields["intensity"])

        combined_pts = np.vstack(all_pts) if all_pts else np.empty((0, 3))
        combined_cols = np.vstack(all_cols) if len(all_cols) == len(all_pts) else None
        combined_ints = np.concatenate(all_ints) if len(all_ints) == len(all_pts) else None

        self.viewer.set_point_cloud(combined_pts, combined_cols, combined_ints, reset_camera=True)
        self.point_count_label.setText(f"点数: {len(combined_pts):,} 点")

    def _on_color_mode_changed(self, mode_name: str):
        self.viewer.set_color_mode(mode_name)

    def _toggle_selection_mode(self, enabled: bool):
        self.viewer.selection_mode = enabled
        if enabled:
            self.select_mode_btn.setText("矩形選択モード: ON")
            self.select_mode_btn.setStyleSheet("background-color: #0078d4; font-weight: bold;")
            self.manual_status_label.setText("【選択中】画面上をマウス左ドラッグで四角く囲んでください")
        else:
            self.select_mode_btn.setText("矩形選択モード")
            self.select_mode_btn.setStyleSheet("")
            self.manual_status_label.setText("※「矩形選択モード」をONにして画面上をドラッグすると点を選択できます")

    def _clear_selection(self):
        self.viewer.selected_indices = np.array([], dtype=np.int64)
        self.viewer._update_active_colors()
        self.viewer._update_gl_buffers()
        self.viewer.update()
        self.manual_status_label.setText("選択を解除しました")

    def _on_points_selected(self, indices: list):
        self.manual_status_label.setText(f"選択中: {len(indices):,} 点 (赤色ハイライト) | [Delete]で削除")

    def _delete_selected_points(self):
        """選択されているノイズ点を削除する"""
        selected_idx = self.viewer.selected_indices
        if len(selected_idx) == 0:
            return

        # 現在の状態をUndoスタックに退避（ディープコピー）
        import copy
        saved_scans = [
            E57ScanData(
                scan_index=s.scan_index,
                points=s.points.copy(),
                raw_fields={k: v.copy() if hasattr(v, "copy") else v for k, v in s.raw_fields.items()},
                header=s.header,
                rotation=s.rotation,
                translation=s.translation,
            )
            for s in self.current_scans
        ]
        self.undo_stack.append(saved_scans)
        self.undo_btn.setEnabled(True)

        # 単一スキャンの場合、インデックスで直接フィルタ
        if len(self.current_scans) == 1:
            scan = self.current_scans[0]
            keep_mask = np.ones(len(scan.points), dtype=bool)
            keep_mask[selected_idx] = False
            keep_indices = np.where(keep_mask)[0]
            self.current_scans[0] = scan.filter_by_indices(keep_indices)
        else:
            # 複数スキャンの場合、オフセット計算して各スキャンから削除
            current_offset = 0
            for i, scan in enumerate(self.current_scans):
                n_s = len(scan.points)
                s_sel = selected_idx[(selected_idx >= current_offset) & (selected_idx < current_offset + n_s)] - current_offset
                if len(s_sel) > 0:
                    keep_mask = np.ones(n_s, dtype=bool)
                    keep_mask[s_sel] = False
                    keep_indices = np.where(keep_mask)[0]
                    self.current_scans[i] = scan.filter_by_indices(keep_indices)
                current_offset += n_s

        del_count = len(selected_idx)
        self._clear_selection()
        self._update_viewer_from_scans()
        self.log_text.append(f"手動削除: {del_count:,} 点を除去しました (Undo可能)")
        self.status_label.setText(f"{del_count:,} 点を削除しました")

    def _undo_action(self):
        """直前の操作を取り消す (Undo)"""
        if not self.undo_stack:
            return

        self.current_scans = self.undo_stack.pop()
        self.undo_btn.setEnabled(len(self.undo_stack) > 0)
        self._clear_selection()
        self._update_viewer_from_scans()
        self.log_text.append("元に戻す (Undo) を実行しました")
        self.status_label.setText("操作を取り消しました")

    def _save_current_scans(self):
        """現在の手動編集結果をE57に書き出す"""
        if not self.current_scans:
            QMessageBox.warning(self, "警告", "保存する点群データがありません。")
            return

        out_path = self.output_edit.text().strip()
        if not out_path:
            self._browse_output_file()
            out_path = self.output_edit.text().strip()
            if not out_path:
                return

        try:
            self.status_label.setText("E57書き出し中...")
            write_e57_scans(out_path, self.current_scans)
            total_pts = sum(s.point_count for s in self.current_scans)
            QMessageBox.information(
                self,
                "保存完了",
                f"手動編集後の点群を保存しました！\n\n・総点数: {total_pts:,} 点\n・保存先: {out_path}",
            )
            self.log_text.append(f"保存完了: {out_path} ({total_pts:,} 点)")
            self.status_label.setText("保存完了")
        except Exception as e:
            QMessageBox.critical(self, "保存エラー", f"保存中にエラーが発生しました:\n{e}")
            self.status_label.setText("保存失敗")

    def _run_auto_processing(self):
        """自動フィルタ処理を実行"""
        in_path = self.input_edit.text().strip()
        out_path = self.output_edit.text().strip()

        if not in_path or not os.path.exists(in_path):
            QMessageBox.warning(self, "エラー", "有効な入力E57ファイルを指定してください。")
            return
        if not out_path:
            QMessageBox.warning(self, "エラー", "出力先E57ファイルを指定してください。")
            return

        config = FilterConfig(
            use_sor=self.sor_chk.isChecked(),
            sor_neighbors=self.sor_k_spin.value(),
            sor_std_ratio=self.sor_r_spin.value(),
            use_ror=self.ror_chk.isChecked(),
            ror_radius=self.ror_radius_spin.value(),
            ror_min_points=self.ror_pts_spin.value(),
            use_voxel_downsample=self.voxel_chk.isChecked(),
            voxel_size=self.voxel_spin.value(),
        )

        self.auto_exec_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.log_text.append("--- 自動ノイズ処理開始 ---")

        self.worker = ProcessWorker(in_path, out_path, config)
        self.worker.progress_changed.connect(lambda p, m: (self.progress_bar.setValue(p), self.status_label.setText(m)))
        self.worker.log_message.connect(self.log_text.append)
        self.worker.finished_success.connect(self._on_auto_finished)
        self.worker.finished_error.connect(self._on_auto_error)
        self.worker.start()

    def _on_auto_finished(self, summary: ProcessSummary):
        self.auto_exec_btn.setEnabled(True)
        self.progress_bar.setValue(100)
        self.status_label.setText("自動ノイズ処理が完了しました")

        # 処理後のファイルをビューアにロード
        self._load_file_to_viewer(summary.output_file)

        res_msg = (
            f"自動ノイズ処理が完了しました！\n\n"
            f"・元点数: {summary.total_initial_points:,} 点\n"
            f"・保持点数: {summary.total_final_points:,} 点\n"
            f"・除去点数: {summary.total_removed_points:,} 点 ({summary.total_removal_ratio:.2f}% 削減)\n"
            f"・所要時間: {summary.total_time_taken_sec:.2f} 秒\n\n"
            f"処理後の点群を3D画面にロードしました。"
        )
        QMessageBox.information(self, "完了", res_msg)

    def _on_auto_error(self, err_msg: str):
        self.auto_exec_btn.setEnabled(True)
        self.status_label.setText("エラーが発生しました")
        QMessageBox.critical(self, "処理エラー", f"エラー:\n{err_msg}")

    def _on_viewer_status(self, msg: str):
        self.status_label.setText(msg)
