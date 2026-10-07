"""
PySide6によるE57点群ノイズ処理ツールのメインウィンドウ
- 数学座標系（Z-up）に準拠した表示およびXYZ軸方向ギズモ
- 見ている画面に対するWASDフライスルー移動
- XYZ別シーク断面スライサー (各軸の位置・厚みスライダーによる断面抽出)
- 手動選択ツール（矩形、円形、多角形ポリゴン選択）
- 2点間寸法計測、全体フィット、正射影/透視投影、描画点数バジェット
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
    QScrollArea,
)
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QKeySequence, QShortcut, QColor

from core.processor import FilterConfig, PointCloudProcessor, ProcessSummary
from core.e57_io import E57ScanData, read_e57_scans, write_e57_scans
from gui.worker_thread import ProcessWorker, FileLoadWorker, PreloadedData, MemoryFilterWorker
from gui.progress_dialog import ModernProgressDialog
from gui.gl_viewer import PointCloudViewer
from gui.styles import DARK_THEME_QSS


class MainWindow(QMainWindow):
    """メインウィンドウクラス"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("E57 点群ノイズ処理ツール - 3Dエディタ")
        self.resize(1400, 900)
        self.setMinimumSize(QSize(1000, 650))
        self.setStyleSheet(DARK_THEME_QSS)

        self.setAcceptDrops(True)

        self.current_scans: List[E57ScanData] = []
        self.undo_stack: List[List[E57ScanData]] = []
        self.layer_undo_stack: List[np.ndarray] = []
        self.worker: Optional[ProcessWorker] = None
        self.load_worker: Optional[FileLoadWorker] = None
        self.memory_filter_worker: Optional[MemoryFilterWorker] = None
        self.progress_dialog: Optional[ModernProgressDialog] = None

        # 点群のワールド範囲キャッシュ
        self.world_min = np.array([-10.0, -10.0, -5.0])
        self.world_max = np.array([10.0, 10.0, 5.0])

        self._init_ui()
        self._setup_shortcuts()

    def _init_ui(self):
        """UIコンポーネントの構築"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        root_layout = QHBoxLayout(central_widget)
        root_layout.setContentsMargins(4, 4, 4, 4)
        root_layout.setSpacing(4)

        # 左右スプリッター（ユーザーが境界線をドラッグして幅を自由調整可能）
        self.splitter = QSplitter(Qt.Horizontal)
        root_layout.addWidget(self.splitter)

        # --- 左パネル (操作・設定サイドバー) ---
        self.sidebar_scroll = QScrollArea()
        self.sidebar_scroll.setWidgetResizable(True)
        self.sidebar_scroll.setMinimumWidth(280)
        self.sidebar_scroll.setMaximumWidth(700)
        self.sidebar_scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        sidebar_widget = QWidget()
        sidebar_layout = QVBoxLayout(sidebar_widget)
        sidebar_layout.setContentsMargins(6, 6, 6, 6)
        sidebar_layout.setSpacing(8)

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

        # 2. 3D表示・カメラ操作パネル (数学座標系: Z-up)
        view_group = QGroupBox("3D表示・カメラ操作 (数学座標系)")
        view_layout = QGridLayout(view_group)

        # 視点プリセット
        view_layout.addWidget(QLabel("視点切替:"), 0, 0)
        view_btn_layout = QHBoxLayout()
        self.btn_view_iso = QPushButton("等角")
        self.btn_view_iso.clicked.connect(lambda: self.viewer.reset_view())
        self.btn_view_top = QPushButton("上面 (XY)")
        self.btn_view_top.clicked.connect(lambda: self.viewer.set_view_top())
        self.btn_view_front = QPushButton("正面 (XZ)")
        self.btn_view_front.clicked.connect(lambda: self.viewer.set_view_front())
        self.btn_view_side = QPushButton("側面 (YZ)")
        self.btn_view_side.clicked.connect(lambda: self.viewer.set_view_side())

        for b in [self.btn_view_iso, self.btn_view_top, self.btn_view_front, self.btn_view_side]:
            b.setStyleSheet("padding: 4px 5px; font-size: 11px;")
            view_btn_layout.addWidget(b)
        view_layout.addLayout(view_btn_layout, 0, 1)

        # 投影切り替え & 全体フィット
        view_layout.addWidget(QLabel("カメラ制御:"), 1, 0)
        cam_ctrl_layout = QHBoxLayout()
        self.btn_fit = QPushButton("全体フィット (F)")
        self.btn_fit.clicked.connect(lambda: self.viewer.fit_to_screen())
        self.btn_fit.setStyleSheet("padding: 4px 6px; font-size: 11px; background-color: #2b5278;")
        cam_ctrl_layout.addWidget(self.btn_fit)

        self.btn_proj = QPushButton("透視投影 ⇄ 正射影 (P)")
        self.btn_proj.clicked.connect(lambda: self.viewer.toggle_projection())
        self.btn_proj.setStyleSheet("padding: 4px 6px; font-size: 11px; background-color: #2b5278;")
        cam_ctrl_layout.addWidget(self.btn_proj)
        view_layout.addLayout(cam_ctrl_layout, 1, 1)

        # 配色モード
        view_layout.addWidget(QLabel("配色モード:"), 2, 0)
        self.color_combo = QComboBox()
        self.color_combo.addItems([
            PointCloudViewer.COLOR_MODE_RGB,
            PointCloudViewer.COLOR_MODE_HEIGHT,
            PointCloudViewer.COLOR_MODE_INTENSITY,
            PointCloudViewer.COLOR_MODE_SOLID,
        ])
        self.color_combo.currentTextChanged.connect(self._on_color_mode_changed)
        view_layout.addWidget(self.color_combo, 2, 1)

        # 点サイズ
        view_layout.addWidget(QLabel("点サイズ:"), 3, 0)
        size_layout = QHBoxLayout()
        self.size_slider = QSlider(Qt.Horizontal)
        self.size_slider.setRange(1, 15)
        self.size_slider.setValue(2)
        self.size_slider.valueChanged.connect(lambda val: self.viewer.set_point_size(val))
        self.size_label = QLabel("2 px")
        self.size_slider.valueChanged.connect(lambda val: self.size_label.setText(f"{val} px"))
        size_layout.addWidget(self.size_slider)
        size_layout.addWidget(self.size_label)
        view_layout.addLayout(size_layout, 3, 1)

        # 点形状 (丸・四角)
        view_layout.addWidget(QLabel("点形状:"), 4, 0)
        self.shape_combo = QComboBox()
        self.shape_combo.addItem("丸 (スムーズ)", "circle")
        self.shape_combo.addItem("四角 (クッキリ)", "square")
        self.shape_combo.currentIndexChanged.connect(self._on_point_shape_changed)
        view_layout.addWidget(self.shape_combo, 4, 1)

        # 最大描画点数 (Point Budget)
        view_layout.addWidget(QLabel("描画点数:"), 5, 0)
        self.budget_combo = QComboBox()
        self.budget_combo.addItem("100万点 (超高速・推奨)", 1000000)
        self.budget_combo.addItem("200万点 (標準)", 2000000)
        self.budget_combo.addItem("500万点 (高精細)", 5000000)
        self.budget_combo.addItem("全点表示 (無制限)", 0)
        self.budget_combo.setCurrentIndex(1)
        self.budget_combo.currentIndexChanged.connect(self._on_budget_changed)
        view_layout.addWidget(self.budget_combo, 5, 1)

        # 背景色 & グリッド
        view_layout.addWidget(QLabel("環境設定:"), 6, 0)
        env_layout = QHBoxLayout()
        self.bg_combo = QComboBox()
        self.bg_combo.addItem("ダーク", QColor(20, 20, 26))
        self.bg_combo.addItem("ブラック", QColor(0, 0, 0))
        self.bg_combo.addItem("ホワイト", QColor(245, 245, 247))
        self.bg_combo.currentIndexChanged.connect(self._on_bg_color_changed)
        env_layout.addWidget(self.bg_combo)

        self.grid_chk = QCheckBox("グリッド")
        self.grid_chk.setChecked(True)
        self.grid_chk.toggled.connect(lambda checked: self.viewer.set_show_grid(checked))
        env_layout.addWidget(self.grid_chk)
        view_layout.addLayout(env_layout, 6, 1)

        sidebar_layout.addWidget(view_group)

        # 3. XYZ別 シーク断面スライサー
        section_group = QGroupBox("📐 XYZ断面シークスライサー")
        sec_layout = QVBoxLayout(section_group)

        # X断面
        x_box = QWidget()
        x_layout = QVBoxLayout(x_box)
        x_layout.setContentsMargins(0, 2, 0, 2)
        x_head = QHBoxLayout()
        self.chk_sec_x = QCheckBox("X断面スライス (左右)")
        self.chk_sec_x.toggled.connect(self._on_section_x_changed)
        x_head.addWidget(self.chk_sec_x)
        self.lbl_sec_x = QLabel("位置: 0.0m (厚み: 2.0m)")
        self.lbl_sec_x.setStyleSheet("font-size: 11px; color: #38bdf8;")
        x_head.addStretch()
        x_head.addWidget(self.lbl_sec_x)
        x_layout.addLayout(x_head)

        x_sliders = QHBoxLayout()
        self.slider_pos_x = QSlider(Qt.Horizontal)
        self.slider_pos_x.setRange(-1000, 1000)
        self.slider_pos_x.setValue(0)
        self.slider_pos_x.valueChanged.connect(self._on_section_x_changed)
        x_sliders.addWidget(QLabel("位置:"))
        x_sliders.addWidget(self.slider_pos_x)

        self.slider_thick_x = QSlider(Qt.Horizontal)
        self.slider_thick_x.setRange(1, 200)  # 0.1m - 20.0m
        self.slider_thick_x.setValue(20)      # 2.0m
        self.slider_thick_x.valueChanged.connect(self._on_section_x_changed)
        x_sliders.addWidget(QLabel("厚み:"))
        x_sliders.addWidget(self.slider_thick_x)
        x_layout.addLayout(x_sliders)
        sec_layout.addWidget(x_box)

        # Y断面
        y_box = QWidget()
        y_layout = QVBoxLayout(y_box)
        y_layout.setContentsMargins(0, 2, 0, 2)
        y_head = QHBoxLayout()
        self.chk_sec_y = QCheckBox("Y断面スライス (奥行)")
        self.chk_sec_y.toggled.connect(self._on_section_y_changed)
        y_head.addWidget(self.chk_sec_y)
        self.lbl_sec_y = QLabel("位置: 0.0m (厚み: 2.0m)")
        self.lbl_sec_y.setStyleSheet("font-size: 11px; color: #38bdf8;")
        y_head.addStretch()
        y_head.addWidget(self.lbl_sec_y)
        y_layout.addLayout(y_head)

        y_sliders = QHBoxLayout()
        self.slider_pos_y = QSlider(Qt.Horizontal)
        self.slider_pos_y.setRange(-1000, 1000)
        self.slider_pos_y.setValue(0)
        self.slider_pos_y.valueChanged.connect(self._on_section_y_changed)
        y_sliders.addWidget(QLabel("位置:"))
        y_sliders.addWidget(self.slider_pos_y)

        self.slider_thick_y = QSlider(Qt.Horizontal)
        self.slider_thick_y.setRange(1, 200)
        self.slider_thick_y.setValue(20)
        self.slider_thick_y.valueChanged.connect(self._on_section_y_changed)
        y_sliders.addWidget(QLabel("厚み:"))
        y_sliders.addWidget(self.slider_thick_y)
        y_layout.addLayout(y_sliders)
        sec_layout.addWidget(y_box)

        # Z断面
        z_box = QWidget()
        z_layout = QVBoxLayout(z_box)
        z_layout.setContentsMargins(0, 2, 0, 2)
        z_head = QHBoxLayout()
        self.chk_sec_z = QCheckBox("Z断面スライス (高さ/フロア)")
        self.chk_sec_z.toggled.connect(self._on_section_z_changed)
        z_head.addWidget(self.chk_sec_z)
        self.lbl_sec_z = QLabel("位置: 0.0m (厚み: 2.0m)")
        self.lbl_sec_z.setStyleSheet("font-size: 11px; color: #38bdf8;")
        z_head.addStretch()
        z_head.addWidget(self.lbl_sec_z)
        z_layout.addLayout(z_head)

        z_sliders = QHBoxLayout()
        self.slider_pos_z = QSlider(Qt.Horizontal)
        self.slider_pos_z.setRange(-1000, 1000)
        self.slider_pos_z.setValue(0)
        self.slider_pos_z.valueChanged.connect(self._on_section_z_changed)
        z_sliders.addWidget(QLabel("位置:"))
        z_sliders.addWidget(self.slider_pos_z)

        self.slider_thick_z = QSlider(Qt.Horizontal)
        self.slider_thick_z.setRange(1, 200)
        self.slider_thick_z.setValue(20)
        self.slider_thick_z.valueChanged.connect(self._on_section_z_changed)
        z_sliders.addWidget(QLabel("厚み:"))
        z_sliders.addWidget(self.slider_thick_z)
        z_layout.addLayout(z_sliders)
        sec_layout.addWidget(z_box)

        sidebar_layout.addWidget(section_group)

        # 4. レイヤー管理 (通常点群 vs ノイズ点群)
        layer_group = QGroupBox("📑 レイヤー管理 (表示切替 / 対象制御)")
        layer_layout = QVBoxLayout(layer_group)

        layer_row1 = QHBoxLayout()
        self.chk_layer_normal = QCheckBox("通常点群レイヤー")
        self.chk_layer_normal.setChecked(True)
        self.chk_layer_normal.toggled.connect(self._on_layer_visibility_changed)
        self.lbl_layer_normal_count = QLabel("0 点")
        self.lbl_layer_normal_count.setStyleSheet("color: #38bdf8; font-weight: bold;")
        layer_row1.addWidget(self.chk_layer_normal)
        layer_row1.addStretch()
        layer_row1.addWidget(self.lbl_layer_normal_count)
        layer_layout.addLayout(layer_row1)

        layer_row2 = QHBoxLayout()
        self.chk_layer_noise = QCheckBox("ノイズレイヤー (赤色)")
        self.chk_layer_noise.setChecked(False)
        self.chk_layer_noise.toggled.connect(self._on_layer_visibility_changed)
        self.lbl_layer_noise_count = QLabel("0 点")
        self.lbl_layer_noise_count.setStyleSheet("color: #f87171; font-weight: bold;")
        layer_row2.addWidget(self.chk_layer_noise)
        layer_row2.addStretch()
        layer_row2.addWidget(self.lbl_layer_noise_count)
        layer_layout.addLayout(layer_row2)

        layer_btn_layout = QHBoxLayout()
        self.btn_restore_noise = QPushButton("ノイズを通常にすべて復元")
        self.btn_restore_noise.setStyleSheet("font-size: 11px;")
        self.btn_restore_noise.clicked.connect(self._restore_all_noise)
        layer_btn_layout.addWidget(self.btn_restore_noise)
        layer_layout.addLayout(layer_btn_layout)

        layer_info = QLabel("※手動削除および自動フィルタ検出点はノイズレイヤーに移動します。自動処理は現在ONの通常点群のみを対象に実行されます。")
        layer_info.setWordWrap(True)
        layer_info.setStyleSheet("color: #94a3b8; font-size: 11px;")
        layer_layout.addWidget(layer_info)

        sidebar_layout.addWidget(layer_group)

        # 5. 手動選択・編集・計測ツール
        tools_group = QGroupBox("編集・選択ツール (矩形 / 円 / 多角形)")
        tools_layout = QVBoxLayout(tools_group)

        # 計測ツールトグルボタン
        self.btn_measure = QPushButton("📐 2点間寸法計測ツール (M)")
        self.btn_measure.setCheckable(True)
        self.btn_measure.setStyleSheet("background-color: #334155; font-weight: bold;")
        self.btn_measure.toggled.connect(self._toggle_measure_mode)
        tools_layout.addWidget(self.btn_measure)

        # 選択形状セレクタ & モード切替
        sel_shape_layout = QHBoxLayout()
        sel_shape_layout.addWidget(QLabel("選択形状:"))
        self.select_shape_combo = QComboBox()
        self.select_shape_combo.addItem("矩形 (四角)", PointCloudViewer.SELECT_RECT)
        self.select_shape_combo.addItem("円形 (サークル)", PointCloudViewer.SELECT_CIRCLE)
        self.select_shape_combo.addItem("多角形 (ポリゴン)", PointCloudViewer.SELECT_POLYGON)
        self.select_shape_combo.currentIndexChanged.connect(self._on_selection_shape_changed)
        sel_shape_layout.addWidget(self.select_shape_combo)

        self.select_mode_btn = QPushButton("手動選択モード")
        self.select_mode_btn.setCheckable(True)
        self.select_mode_btn.toggled.connect(self._toggle_selection_mode)
        sel_shape_layout.addWidget(self.select_mode_btn)
        tools_layout.addLayout(sel_shape_layout)

        # 多角形確定ボタン & 選択解除ボタン
        poly_action_layout = QHBoxLayout()
        self.btn_poly_confirm = QPushButton("多角形を確定 (Enter)")
        self.btn_poly_confirm.setStyleSheet("background-color: #0f766e; font-size: 11px;")
        self.btn_poly_confirm.clicked.connect(self._confirm_polygon_selection)
        self.btn_poly_confirm.setEnabled(False)
        poly_action_layout.addWidget(self.btn_poly_confirm)

        self.clear_sel_btn = QPushButton("選択解除 (Esc)")
        self.clear_sel_btn.clicked.connect(self._clear_selection)
        poly_action_layout.addWidget(self.clear_sel_btn)
        tools_layout.addLayout(poly_action_layout)

        manual_action_layout = QHBoxLayout()
        self.delete_pts_btn = QPushButton("選択点を削除 (Delete)")
        self.delete_pts_btn.setStyleSheet("background-color: #b33939; font-weight: bold;")
        self.delete_pts_btn.clicked.connect(self._delete_selected_points)
        manual_action_layout.addWidget(self.delete_pts_btn)

        self.undo_btn = QPushButton("元に戻す (Ctrl+Z)")
        self.undo_btn.clicked.connect(self._undo_action)
        self.undo_btn.setEnabled(False)
        manual_action_layout.addWidget(self.undo_btn)
        tools_layout.addLayout(manual_action_layout)

        self.manual_status_label = QLabel("※選択モードONで画面上の点を囲んで削除できます (断面スライス併用可)")
        self.manual_status_label.setWordWrap(True)
        self.manual_status_label.setStyleSheet("color: #888899; font-size: 11px;")
        tools_layout.addWidget(self.manual_status_label)

        self.save_manual_btn = QPushButton("手動編集した点群をE57保存")
        self.save_manual_btn.clicked.connect(self._save_current_scans)
        tools_layout.addWidget(self.save_manual_btn)

        sidebar_layout.addWidget(tools_group)

        # 5. 自動ノイズ除去フィルタ設定
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

        # 間引き
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

        # 6. ステータス・ログ
        log_group = QGroupBox("ステータス")
        log_layout = QVBoxLayout(log_group)
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        log_layout.addWidget(self.progress_bar)
        self.status_label = QLabel("E57ファイルを読み込んでください")
        log_layout.addWidget(self.status_label)
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(85)
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        sidebar_layout.addWidget(log_group)

        self.sidebar_scroll.setWidget(sidebar_widget)
        self.splitter.addWidget(self.sidebar_scroll)

        # --- 右パネル (3D点群OpenGLビューア) ---
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        # ナビゲーションバー
        nav_bar = QWidget()
        nav_bar.setFixedHeight(38)
        nav_bar.setStyleSheet("background-color: #17171c; border-bottom: 1px solid #333344;")
        nav_layout = QHBoxLayout(nav_bar)
        nav_layout.setContentsMargins(8, 2, 8, 2)

        self.btn_toggle_sidebar = QPushButton("◀ パネル格納")
        self.btn_toggle_sidebar.setStyleSheet("padding: 3px 8px; font-size: 11px; background-color: #334155;")
        self.btn_toggle_sidebar.clicked.connect(self._toggle_sidebar)
        nav_layout.addWidget(self.btn_toggle_sidebar)

        nav_btn_fit = QPushButton("フィット (F)")
        nav_btn_fit.setStyleSheet("padding: 3px 8px; font-size: 11px;")
        nav_btn_fit.clicked.connect(lambda: self.viewer.fit_to_screen())
        nav_layout.addWidget(nav_btn_fit)

        self.nav_btn_proj = QPushButton("投影: 透視")
        self.nav_btn_proj.setStyleSheet("padding: 3px 8px; font-size: 11px;")
        self.nav_btn_proj.clicked.connect(lambda: self.viewer.toggle_projection())
        nav_layout.addWidget(self.nav_btn_proj)

        self.nav_btn_measure = QPushButton("📐 計測 (M)")
        self.nav_btn_measure.setCheckable(True)
        self.nav_btn_measure.setStyleSheet("padding: 3px 8px; font-size: 11px;")
        self.nav_btn_measure.toggled.connect(self._toggle_measure_mode)
        nav_layout.addWidget(self.nav_btn_measure)

        help_label = QLabel(
            "操作: [左ドラッグ]回転 | [右/中]画面基準パン | [ホイール]ズーム | [W/A/S/D]画面基準移動 | [ダブルクリック]中心移動"
        )
        help_label.setStyleSheet("color: #94a3b8; font-size: 11px; margin-left: 6px;")
        nav_layout.addWidget(help_label)
        nav_layout.addStretch()

        self.point_count_label = QLabel("点数: 0 点")
        self.point_count_label.setStyleSheet("color: #38bdf8; font-weight: bold; font-size: 12px;")
        nav_layout.addWidget(self.point_count_label)

        right_layout.addWidget(nav_bar, 0)

        # 3D点群ビューア
        self.viewer = PointCloudViewer()
        self.viewer.status_changed.connect(self._on_viewer_status)
        self.viewer.point_selected.connect(self._on_points_selected)
        self.viewer.projection_changed.connect(self._on_projection_changed)
        self.viewer.bounds_ready.connect(self._on_bounds_ready)
        self.viewer.layers_updated.connect(self._on_layers_updated)
        self.viewer.measure_updated.connect(lambda msg: self.status_label.setText(msg))
        right_layout.addWidget(self.viewer, 1)

        self.splitter.addWidget(right_container)
        self.splitter.setSizes([360, 1040])
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)

    def _setup_shortcuts(self):
        """ショートカットキーの設定"""
        del_shortcut = QShortcut(QKeySequence(Qt.Key_Delete), self)
        del_shortcut.activated.connect(self._delete_selected_points)

        undo_shortcut = QShortcut(QKeySequence.Undo, self)
        undo_shortcut.activated.connect(self._undo_action)

    def _on_layers_updated(self, normal_count: int, noise_count: int):
        """レイヤー点数更新通知を受信"""
        self.lbl_layer_normal_count.setText(f"{normal_count:,} 点")
        self.lbl_layer_noise_count.setText(f"{noise_count:,} 点")
        self.point_count_label.setText(f"通常: {normal_count:,} 点 | ノイズ: {noise_count:,} 点")

    def _on_layer_visibility_changed(self):
        """通常/ノイズレイヤーの表示切り替え"""
        show_normal = self.chk_layer_normal.isChecked()
        show_noise = self.chk_layer_noise.isChecked()
        self.viewer.set_layer_visibility(show_normal, show_noise)

    def _restore_all_noise(self):
        """ノイズレイヤーの点をすべて通常レイヤーに復元"""
        if self.viewer.layer_mask is None:
            return
        noise_idx = np.where(~self.viewer.layer_mask)[0]
        if len(noise_idx) == 0:
            QMessageBox.information(self, "情報", "ノイズレイヤーに点はありません。")
            return
        self.layer_undo_stack.append(self.viewer.layer_mask.copy())
        self.undo_btn.setEnabled(True)
        self.viewer.restore_indices_to_normal(noise_idx)
        self.log_text.append(f"ノイズ復元: {len(noise_idx):,} 点を通常レイヤーへ復元しました")
        self.status_label.setText(f"{len(noise_idx):,} 点を通常レイヤーに復元しました")

    def _toggle_sidebar(self):
        is_visible = self.sidebar_scroll.isVisible()
        self.sidebar_scroll.setVisible(not is_visible)
        self.btn_toggle_sidebar.setText("▶ パネル展開" if is_visible else "◀ パネル格納")

    def _on_projection_changed(self, is_ortho: bool):
        mode_str = "正射影" if is_ortho else "透視投影"
        self.nav_btn_proj.setText(f"投影: {mode_str}")
        self.btn_proj.setText(f"投影: {mode_str} (P)")

    def _toggle_measure_mode(self, enabled: bool):
        self.viewer.measure_mode = enabled
        self.btn_measure.setChecked(enabled)
        self.nav_btn_measure.setChecked(enabled)
        if enabled:
            self.btn_measure.setStyleSheet("background-color: #0284c7; font-weight: bold;")
            self.nav_btn_measure.setStyleSheet("background-color: #0284c7; font-weight: bold;")
            self.status_label.setText("【計測モード】3D画面上の2点を左クリックしてください（右クリックでクリア）")
        else:
            self.btn_measure.setStyleSheet("background-color: #334155; font-weight: bold;")
            self.nav_btn_measure.setStyleSheet("")
            self.viewer.measure_points.clear()
            self.viewer.measure_result_text = ""
            self.viewer.update()
            self.status_label.setText("計測モードを終了しました")

    def _on_selection_shape_changed(self, index: int):
        shape = self.select_shape_combo.currentData()
        self.viewer.selection_shape = shape
        self.btn_poly_confirm.setEnabled(shape == PointCloudViewer.SELECT_POLYGON)
        if shape == PointCloudViewer.SELECT_POLYGON:
            self.manual_status_label.setText("【多角形選択】左クリックで頂点追加、ダブルクリックまたはEnterで確定")
        elif shape == PointCloudViewer.SELECT_CIRCLE:
            self.manual_status_label.setText("【円形選択】中心からドラッグして円を描いてください")
        else:
            self.manual_status_label.setText("【矩形選択】ドラッグして四角く囲んでください")

    def _confirm_polygon_selection(self):
        if len(self.viewer.polygon_points) >= 3:
            self.viewer._process_polygon_selection()
            self.viewer.polygon_points.clear()
            self.viewer.update()

    def _on_bounds_ready(self, b_min: np.ndarray, b_max: np.ndarray):
        """点群のワールド範囲に合わせて断面スライダーの範囲を初期化"""
        self.world_min = b_min
        self.world_max = b_max

        # Xスライダー
        self.slider_pos_x.setRange(int(b_min[0] * 10), int(b_max[0] * 10))
        self.slider_pos_x.setValue(int((b_min[0] + b_max[0]) * 5))

        # Yスライダー
        self.slider_pos_y.setRange(int(b_min[1] * 10), int(b_max[1] * 10))
        self.slider_pos_y.setValue(int((b_min[1] + b_max[1]) * 5))

        # Zスライダー
        self.slider_pos_z.setRange(int(b_min[2] * 10), int(b_max[2] * 10))
        self.slider_pos_z.setValue(int((b_min[2] + b_max[2]) * 5))

        self._on_section_x_changed()
        self._on_section_y_changed()
        self._on_section_z_changed()

    def _on_section_x_changed(self):
        enabled = self.chk_sec_x.isChecked()
        pos = self.slider_pos_x.value() / 10.0
        thick = self.slider_thick_x.value() / 10.0
        half = thick / 2.0
        self.lbl_sec_x.setText(f"X: {pos:.1f}m (厚: {thick:.1f}m)")
        self.viewer.set_section_x(enabled, pos - half, pos + half)

    def _on_section_y_changed(self):
        enabled = self.chk_sec_y.isChecked()
        pos = self.slider_pos_y.value() / 10.0
        thick = self.slider_thick_y.value() / 10.0
        half = thick / 2.0
        self.lbl_sec_y.setText(f"Y: {pos:.1f}m (厚: {thick:.1f}m)")
        self.viewer.set_section_y(enabled, pos - half, pos + half)

    def _on_section_z_changed(self):
        enabled = self.chk_sec_z.isChecked()
        pos = self.slider_pos_z.value() / 10.0
        thick = self.slider_thick_z.value() / 10.0
        half = thick / 2.0
        self.lbl_sec_z.setText(f"Z: {pos:.1f}m (厚: {thick:.1f}m)")
        self.viewer.set_section_z(enabled, pos - half, pos + half)

    def _on_point_shape_changed(self, index: int):
        self.viewer.set_point_shape(self.shape_combo.currentData())

    def _on_budget_changed(self, index: int):
        self.viewer.set_point_budget(self.budget_combo.currentData())

    def _on_bg_color_changed(self, index: int):
        self.viewer.set_bg_color(self.bg_combo.currentData())

    def _on_input_file_changed(self, text: str):
        path = text.strip()
        if path and os.path.exists(path):
            base, ext = os.path.splitext(path)
            suggested_out = f"{base}_cleaned{ext}"
            if not self.output_edit.text() or self.output_edit.text().endswith("_cleaned.e57"):
                self.output_edit.setText(suggested_out)
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
        """E57ファイルをバックグラウンドスレッドで前処理まで完全非同期読み込みし、画面中央に進捗モーダルを表示"""
        fname = os.path.basename(file_path)
        self.status_label.setText(f"点群読み込み中: {fname} (0%)")
        self.progress_bar.setValue(0)
        self.log_text.append(f"--- 点群読み込み開始: {fname} ---")

        # 既存ワーカーの終了待ち
        if self.load_worker and self.load_worker.isRunning():
            self.load_worker.terminate()
            self.load_worker.wait()

        # 画面中央の進捗モーダルダイアログを表示
        if self.progress_dialog:
            self.progress_dialog.close()
        self.progress_dialog = ModernProgressDialog("点群データ読み込み中", self)
        self.progress_dialog.set_progress(0, f"読み込み準備中...", f"ファイル: {fname}")
        self.progress_dialog.show()

        budget = self.budget_combo.currentData() if hasattr(self, "budget_combo") else 2000000
        self.load_worker = FileLoadWorker(file_path, point_budget=budget)

        def on_progress(percent: int, msg: str, sub_msg: str):
            if self.progress_dialog:
                self.progress_dialog.set_progress(percent, msg, sub_msg)
            self.progress_bar.setValue(percent)
            self.status_label.setText(f"{msg} ({percent}%)")

        def on_success(preloaded: PreloadedData):
            if self.progress_dialog:
                self.progress_dialog.set_progress(100, "完了！", f"{len(preloaded.points_raw):,} 点")
                self.progress_dialog.close()
                self.progress_dialog = None

            self.current_scans = preloaded.scans
            self.undo_stack.clear()
            self.layer_undo_stack.clear()
            self.undo_btn.setEnabled(False)

            # チェックボックス表示をリセット
            self.chk_layer_normal.setChecked(True)
            self.chk_layer_noise.setChecked(False)

            # 0ミリ秒で高速バインド (UIフリーズなし)
            self.viewer.set_preloaded_data(preloaded, reset_camera=True)

            total_pts = len(preloaded.points_raw)
            self.point_count_label.setText(f"点数: {total_pts:,} 点")
            self.progress_bar.setValue(100)
            self.status_label.setText(f"読み込み完了: {total_pts:,} 点")
            self.log_text.append(f"読み込み成功: スキャン数 {len(preloaded.scans)}, 総点数 {total_pts:,} 点")

        def on_error(err_msg: str):
            if self.progress_dialog:
                self.progress_dialog.close()
                self.progress_dialog = None
            self.progress_bar.setValue(0)
            self.status_label.setText("読み込みエラー")
            self.log_text.append(f"エラー: {err_msg}")
            QMessageBox.critical(self, "読み込みエラー", f"E57ファイルの読み込みに失敗しました:\n{err_msg}")

        self.load_worker.progress_changed.connect(on_progress)
        self.load_worker.finished_success.connect(on_success)
        self.load_worker.finished_error.connect(on_error)
        self.load_worker.start()

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
            shape_str = self.select_shape_combo.currentText()
            self.select_mode_btn.setText(f"選択モード: ON ({shape_str})")
            self.select_mode_btn.setStyleSheet("background-color: #0078d4; font-weight: bold;")
        else:
            self.select_mode_btn.setText("手動選択モード")
            self.select_mode_btn.setStyleSheet("")
            self.viewer.polygon_points.clear()
            self.viewer.update()

    def _clear_selection(self):
        self.viewer.selected_indices = np.array([], dtype=np.int64)
        self.viewer.polygon_points.clear()
        self.viewer._update_active_colors()
        self.viewer._update_display_arrays()
        self.viewer.update()
        self.manual_status_label.setText("選択を解除しました")

    def _on_points_selected(self, indices: list):
        self.manual_status_label.setText(f"選択中: {len(indices):,} 点 (赤色ハイライト) | [Delete]で削除")

    def _delete_selected_points(self):
        """選択されている点をノイズレイヤーへ移動する (手動削除)"""
        selected_idx = self.viewer.selected_indices
        if len(selected_idx) == 0 or self.viewer.layer_mask is None:
            return

        # Undoスタックに直前のレイヤーマスクを退避
        self.layer_undo_stack.append(self.viewer.layer_mask.copy())
        self.undo_btn.setEnabled(True)

        del_count = len(selected_idx)
        # ノイズレイヤーへ移動
        self.viewer.move_indices_to_noise(selected_idx)
        self._clear_selection()

        self.log_text.append(f"手動削除: {del_count:,} 点をノイズレイヤーへ移動しました (Undo可能)")
        self.status_label.setText(f"{del_count:,} 点をノイズレイヤーへ移動しました")

    def _undo_action(self):
        """直前のレイヤー状態または編集を元に戻す"""
        if not self.layer_undo_stack:
            return

        prev_mask = self.layer_undo_stack.pop()
        self.undo_btn.setEnabled(len(self.layer_undo_stack) > 0)
        self._clear_selection()

        self.viewer.layer_mask = prev_mask
        n_norm = int(np.sum(prev_mask))
        n_noise = int(len(prev_mask) - n_norm)
        self.viewer._update_active_colors()
        self.viewer._update_display_arrays()
        self.viewer.update()
        self.viewer.layers_updated.emit(n_norm, n_noise)

        self.log_text.append("元に戻す (Undo) を実行しました")
        self.status_label.setText("操作を取り消しました")

    def _save_current_scans(self):
        """通常点群レイヤーに属する点のみをE57ファイルとして保存 (ノイズ除外)"""
        if not self.current_scans or self.viewer.layer_mask is None:
            QMessageBox.warning(self, "警告", "保存する点群データがありません。")
            return

        out_path = self.output_edit.text().strip()
        if not out_path:
            self._browse_output_file()
            out_path = self.output_edit.text().strip()
            if not out_path:
                return

        try:
            self.status_label.setText("E57書き出し中 (通常点群のみ抽出)...")
            layer_mask = self.viewer.layer_mask

            # 通常点群のみを抽出したスキャンデータを生成
            filtered_scans = []
            if len(self.current_scans) == 1:
                scan = self.current_scans[0]
                keep_idx = np.where(layer_mask)[0]
                filtered_scans.append(scan.filter_by_indices(keep_idx))
            else:
                current_offset = 0
                for scan in self.current_scans:
                    n_s = len(scan.points)
                    s_mask = layer_mask[current_offset : current_offset + n_s]
                    keep_idx = np.where(s_mask)[0]
                    filtered_scans.append(scan.filter_by_indices(keep_idx))
                    current_offset += n_s

            write_e57_scans(out_path, filtered_scans)
            total_clean_pts = sum(s.point_count for s in filtered_scans)
            total_noise_pts = len(layer_mask) - total_clean_pts
            QMessageBox.information(
                self,
                "保存完了",
                f"通常点群レイヤーの点群をE57ファイルに保存しました！\n\n"
                f"・保存点数 (通常点群): {total_clean_pts:,} 点\n"
                f"・除外点数 (ノイズ): {total_noise_pts:,} 点\n"
                f"・保存先: {out_path}",
            )
            self.log_text.append(f"保存完了: {out_path} (通常点群: {total_clean_pts:,} 点, ノイズ除外: {total_noise_pts:,} 点)")
            self.status_label.setText("保存完了")
        except Exception as e:
            QMessageBox.critical(self, "保存エラー", f"保存中にエラーが発生しました:\n{e}")
            self.status_label.setText("保存失敗")

    def _run_auto_processing(self):
        """現在表示・有効な通常点群レイヤーのみを対象に自動ノイズ処理を実行 (手動削除点は保持・復活なし)"""
        if self.viewer.points_raw is None or len(self.viewer.points_raw) == 0:
            QMessageBox.warning(self, "エラー", "処理対象の点群が読み込まれていません。\n先にE57ファイルを開いてください。")
            return

        normal_indices = self.viewer.get_normal_indices()
        if len(normal_indices) == 0:
            QMessageBox.warning(self, "エラー", "現在通常点群レイヤーに点がありません。")
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

        if not (config.use_sor or config.use_ror or config.use_voxel_downsample):
            QMessageBox.warning(self, "警告", "少なくとも1つのフィルタを有効にしてください。")
            return

        self.auto_exec_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.log_text.append(f"--- 自動ノイズ処理開始: 通常レイヤー {len(normal_indices):,} 点対象 ---")

        # 進捗モーダルを表示
        if self.progress_dialog:
            self.progress_dialog.close()
        self.progress_dialog = ModernProgressDialog("自動ノイズフィルタ処理中", self)
        self.progress_dialog.set_progress(0, "フィルタ解析準備中...", f"対象: {len(normal_indices):,} 点")
        self.progress_dialog.show()

        target_points = self.viewer.points_raw[normal_indices]
        self.memory_filter_worker = MemoryFilterWorker(target_points, config)

        def on_progress(percent: int, msg: str, sub_msg: str):
            if self.progress_dialog:
                self.progress_dialog.set_progress(percent, msg, sub_msg)
            self.progress_bar.setValue(percent)
            self.status_label.setText(f"{msg} ({percent}%)")

        def on_success(keep_idx, remove_idx):
            if self.progress_dialog:
                self.progress_dialog.set_progress(100, "処理完了！", f"{len(remove_idx):,} 点のノイズを分離")
                self.progress_dialog.close()
                self.progress_dialog = None

            self.auto_exec_btn.setEnabled(True)
            self.progress_bar.setValue(100)

            # Undoスタックに退避
            self.layer_undo_stack.append(self.viewer.layer_mask.copy())
            self.undo_btn.setEnabled(True)

            # 検出ノイズのインデックスを原本全体のインデックスに変換
            orig_remove_idx = normal_indices[remove_idx]
            self.viewer.move_indices_to_noise(orig_remove_idx)

            n_removed = len(orig_remove_idx)
            n_remain = len(normal_indices) - n_removed
            ratio = (n_removed / len(normal_indices) * 100.0) if len(normal_indices) > 0 else 0.0

            res_msg = (
                f"自動ノイズ処理が完了しました！\n\n"
                f"・処理対象 (通常レイヤー): {len(normal_indices):,} 点\n"
                f"・残存通常点: {n_remain:,} 点\n"
                f"・検出ノイズ: {n_removed:,} 点 ({ratio:.2f}% 検出)\n\n"
                f"※検出されたノイズはノイズレイヤーに移動しました（手動削除点もそのまま維持されます）。"
            )
            self.log_text.append(f"自動ノイズ処理完了: 検出ノイズ {n_removed:,} 点をノイズレイヤーへ移動 (残存通常: {n_remain:,} 点)")
            self.status_label.setText(f"自動処理完了: ノイズ {n_removed:,} 点検出")
            QMessageBox.information(self, "完了", res_msg)

        def on_error(err_msg: str):
            if self.progress_dialog:
                self.progress_dialog.close()
                self.progress_dialog = None
            self.auto_exec_btn.setEnabled(True)
            self.progress_bar.setValue(0)
            self.status_label.setText("エラーが発生しました")
            self.log_text.append(f"エラー: {err_msg}")
            QMessageBox.critical(self, "処理エラー", f"自動ノイズ処理中にエラーが発生しました:\n{err_msg}")

        self.memory_filter_worker.progress_changed.connect(on_progress)
        self.memory_filter_worker.log_message.connect(self.log_text.append)
        self.memory_filter_worker.finished_success.connect(on_success)
        self.memory_filter_worker.finished_error.connect(on_error)
        self.memory_filter_worker.start()

    def _on_viewer_status(self, msg: str):
        self.status_label.setText(msg)
