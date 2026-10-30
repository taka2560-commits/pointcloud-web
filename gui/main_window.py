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

from PySide6.QtCore import Qt, QSize, QTimer, QPoint, QSettings, Signal
from PySide6.QtWidgets import (
    QApplication,
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
    QColorDialog,
    QInputDialog,
    QMenu,
    QFrame,
    QSizePolicy,
)
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QKeySequence, QShortcut, QColor

from core.processor import FilterConfig, PointCloudProcessor, ProcessSummary
from core.e57_io import E57ScanData, read_e57_scans, write_e57_scans
from core.filters import (
    segment_ground_points,
    segment_largest_plane,
    keep_largest_clusters,
    filter_edge_veil_noise,
    extract_connected_cluster_from_seeds,
    filter_statistical_outlier,
)
from gui.worker_thread import (
    ProcessWorker,
    FileLoadWorker,
    PreloadedData,
    MemoryFilterWorker,
    FileSaveWorker,
    collect_e57_files,
)
from gui.progress_dialog import ModernProgressDialog
from gui.gl_viewer import PointCloudViewer, LayerInfo
from gui.styles import DARK_THEME_QSS



class CollapsibleGroupBox(QGroupBox):
    """
    折りたたみ可能なグループボックス (アコーディオン式)。
    ヘッダーバーまたは右端の [▼/▶] ボタンをクリックしてコンテンツの開閉が可能です。
    不要なパネルを折りたたんで画面を広くスッキリ保てます。
    """
    toggled_collapse = Signal(bool)

    def __init__(self, title: str, parent=None, is_expanded: bool = True):
        super().__init__("", parent)
        self._raw_title = title
        self._is_expanded = is_expanded
        self.setObjectName("collapsible_group")

        self.setStyleSheet("""
            QGroupBox#collapsible_group {
                border: 1px solid #334155;
                border-radius: 6px;
                margin-top: 2px;
                padding: 2px;
                background-color: #172033;
            }
        """)

        # ルートレイアウト
        self._root_layout = QVBoxLayout(self)
        self._root_layout.setContentsMargins(4, 4, 4, 4)
        self._root_layout.setSpacing(2)

        # ヘッダーバー (タイトルラベル + 開閉トグルボタン)
        self._header_widget = QWidget(self)
        self._header_widget.setCursor(Qt.PointingHandCursor)
        self._header_widget.setStyleSheet("background: transparent;")
        header_layout = QHBoxLayout(self._header_widget)
        header_layout.setContentsMargins(4, 2, 4, 2)
        header_layout.setSpacing(6)

        self._title_label = QLabel(title)
        self._title_label.setStyleSheet("font-weight: bold; font-size: 11px; color: #38bdf8;")
        header_layout.addWidget(self._title_label, 1)

        self._toggle_btn = QPushButton("▼" if is_expanded else "▶")
        self._toggle_btn.setFixedSize(22, 20)
        self._toggle_btn.setStyleSheet("""
            QPushButton {
                background-color: #334155;
                color: #e2e8f0;
                border: 1px solid #475569;
                border-radius: 3px;
                font-size: 9px;
                font-weight: bold;
                padding: 0px;
            }
            QPushButton:hover {
                background-color: #0284c7;
                border-color: #38bdf8;
                color: #ffffff;
            }
        """)
        header_layout.addWidget(self._toggle_btn, 0)
        self._root_layout.addWidget(self._header_widget)

        # コンテンツウィジェット (実際のコントロール群を格納)
        self.content_widget = QWidget(self)
        self.content_widget.setStyleSheet("background: transparent;")
        self._root_layout.addWidget(self.content_widget)

        # クリックイベント接続
        self._toggle_btn.clicked.connect(self.toggle_collapse)
        self._header_widget.mousePressEvent = self._on_header_mouse_press

        self.set_expanded(is_expanded)

    def _on_header_mouse_press(self, event):
        if event.button() == Qt.LeftButton:
            self.toggle_collapse()

    def set_expanded(self, expanded: bool):
        self._is_expanded = bool(expanded)
        self.content_widget.setVisible(self._is_expanded)
        self._toggle_btn.setText("▼" if self._is_expanded else "▶")
        self.toggled_collapse.emit(self._is_expanded)

    def toggle_collapse(self):
        self.set_expanded(not self._is_expanded)

    def is_expanded(self) -> bool:
        return self._is_expanded

def get_system_ram_gb() -> float:
    """PCの搭載物理メモリ容量 (GB) を取得 (大容量点群の安全読込推奨用)"""
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return float(stat.ullTotalPhys / (1024 ** 3))
    except Exception:
        pass
    return 16.0


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
        self.layer_undo_stack: list = []
        self.layer_redo_stack: list = []
        self.undo_max_steps: int = 50          # 操作履歴(Undo)の最大保持数 (0で無制限)
        self._sidebar_user_width: int = 380     # 左パネル幅 (コマンド選択時等の画面揺れ防止固定用)
        self.worker: Optional[ProcessWorker] = None
        self.load_worker: Optional[FileLoadWorker] = None
        self.memory_filter_worker: Optional[MemoryFilterWorker] = None
        self.save_worker: Optional[FileSaveWorker] = None
        self.progress_dialog: Optional[ModernProgressDialog] = None

        # 点群のワールド範囲キャッシュ
        self.world_min = np.array([-10.0, -10.0, -5.0])
        self.world_max = np.array([10.0, 10.0, 5.0])

        # 断面スライダー操作時の超高速プレビュー ＆ 確定デバウンスタイマー
        self._sec_commit_timer = QTimer(self)
        self._sec_commit_timer.setSingleShot(True)
        self._sec_commit_timer.timeout.connect(self._commit_pending_section_slices)

        self._init_ui()
        self._setup_shortcuts()
        self._load_app_settings()

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
        self.sidebar_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.sidebar_scroll.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.sidebar_scroll.setMinimumWidth(360)
        self.sidebar_scroll.setMaximumWidth(700)
        self.sidebar_scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        sidebar_widget = QWidget()
        self.sidebar_widget = sidebar_widget
        sidebar_layout = QVBoxLayout(sidebar_widget)
        sidebar_layout.setContentsMargins(4, 4, 6, 4)
        sidebar_layout.setSpacing(4)

        # 1. ファイル入出力 (単一E57ファイル & フォルダ一括統合読み込み対応)
        file_group = CollapsibleGroupBox("📁 ファイル入出力")
        file_layout = QGridLayout(file_group.content_widget)
        file_group.toggled_collapse.connect(self._on_panel_collapse_toggled)

        file_layout.addWidget(QLabel("入力 E57:"), 0, 0)
        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("E57ファイル / フォルダを指定・ドロップ")
        self.input_edit.setMinimumWidth(70)
        self.input_edit.textChanged.connect(self._on_input_file_changed)
        file_layout.addWidget(self.input_edit, 0, 1)

        self.input_browse_btn = QPushButton("参照")
        self.input_browse_btn.setProperty("role", "outline")
        self.input_browse_btn.setMaximumWidth(48)
        self.input_browse_btn.setToolTip("単一のE57ファイルを選択して開きます")
        self.input_browse_btn.clicked.connect(self._browse_input_file)
        file_layout.addWidget(self.input_browse_btn, 0, 2)

        self.input_folder_btn = QPushButton("📁フォルダ")
        self.input_folder_btn.setProperty("role", "outline")
        self.input_folder_btn.setMaximumWidth(72)
        self.input_folder_btn.setToolTip("フォルダ内の全E57ファイルを一括で統合読み込みします（各ファイル＝器械点として表示切替・一括保存可能）")
        self.input_folder_btn.clicked.connect(self._browse_input_folder)
        file_layout.addWidget(self.input_folder_btn, 0, 3)

        file_layout.addWidget(QLabel("出力 E57:"), 1, 0)
        self.output_edit = QLineEdit()
        self.output_edit.setMinimumWidth(70)
        file_layout.addWidget(self.output_edit, 1, 1, 1, 2)

        self.output_browse_btn = QPushButton("参照")
        self.output_browse_btn.setProperty("role", "outline")
        self.output_browse_btn.setMaximumWidth(72)
        self.output_browse_btn.clicked.connect(self._browse_output_file)
        file_layout.addWidget(self.output_browse_btn, 1, 3)

        # ④ 保存対象モード選択 & E57保存 (どのタブを開いていても常に保存できるようファイル欄に集約)
        self.export_mode_combo = QComboBox()
        self.export_mode_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.export_mode_combo.setMinimumContentsLength(8)
        self.export_mode_combo.setToolTip("E57保存時の出力対象を選択します")
        self.export_mode_combo.addItem("有効レイヤー統合 (ノイズ除外)", "normal")
        self.export_mode_combo.addItem("現在表示ONのレイヤー・スキャンのみ", "visible")
        self.export_mode_combo.addItem("選択中の移動先レイヤー単体のみ", "single_layer")
        self.export_mode_combo.addItem("ノイズレイヤーのみ (除去点の確認出力)", "noise_only")
        self.export_mode_combo.addItem("全レイヤーを別々のE57へ一括分割保存", "split_all")
        file_layout.addWidget(QLabel("保存対象:"), 2, 0)
        file_layout.addWidget(self.export_mode_combo, 2, 1, 1, 2)

        self.save_manual_btn = QPushButton("💾 保存")
        self.save_manual_btn.setProperty("role", "primary")
        self.save_manual_btn.setToolTip("左の『保存対象』の内容で点群をE57ファイルへ保存します")
        self.save_manual_btn.setMaximumWidth(72)
        self.save_manual_btn.clicked.connect(self._save_current_scans)
        file_layout.addWidget(self.save_manual_btn, 2, 3)
        file_layout.setVerticalSpacing(4)
        file_layout.setContentsMargins(6, 4, 6, 4)

        self.file_group = file_group
        sidebar_layout.addWidget(file_group)

        # ★ 柱①: ワークフロー切替セグメントナビゲーションバー (タブ風切替で目的セクションに即ジャンプ)
        self.sidebar_nav_bar = QWidget()
        snav_layout = QHBoxLayout(self.sidebar_nav_bar)
        snav_layout.setContentsMargins(0, 2, 0, 2)
        snav_layout.setSpacing(2)
        self.sidebar_tab_btns = {}
        for tab_key, tab_label, tab_tip in [
            ("all", "🗂全て", "全機能セクションを一覧表示"),
            ("edit", "🧹編集", "手動選択・ブラシ・即削除・塊選択・計測"),
            ("section", "📐断面", "XYZ断面スライサー & 2点斜め壁スライス"),
            ("layer", "📑レイヤー", "レイヤー管理 (表示/ロック/自動分類/保存)"),
            ("filter", "🪄自動", "自動ノイズ除去フィルタ (大まか ⇄ 細かい)"),
            ("view", "🎥表示", "視点切替・2点壁視点・投影・配色"),
            ("settings", "⚙設定", "表示・描画点数・環境・感度設定・ショートカット"),
        ]:
            tb = QPushButton(tab_label)
            tb.setCheckable(True)
            tb.setChecked(tab_key == "all")
            tb.setProperty("role", "segment")
            tb.setStyleSheet("padding: 4px 2px; font-size: 11px;")
            tb.setToolTip(tab_tip)
            tb.clicked.connect(lambda _=False, k=tab_key: self._set_sidebar_workflow_tab(k))
            snav_layout.addWidget(tb, 1)
            self.sidebar_tab_btns[tab_key] = tb
        sidebar_layout.addWidget(self.sidebar_nav_bar)

        # パネル一括展開・折りたたみ操作バー (必要なパネルだけ残して広く使うためのアコーディオン)
        collapse_ctrl_row = QHBoxLayout()
        collapse_ctrl_row.setContentsMargins(0, 0, 0, 2)
        collapse_ctrl_row.setSpacing(4)
        self.btn_expand_all = QPushButton("📂 全て展開")
        self.btn_expand_all.setProperty("role", "outline")
        self.btn_expand_all.setStyleSheet("padding: 2px 6px; font-size: 10px;")
        self.btn_expand_all.setToolTip("すべての設定セクションを展開します")
        self.btn_expand_all.clicked.connect(self._expand_all_panels)
        collapse_ctrl_row.addWidget(self.btn_expand_all)

        self.btn_collapse_all = QPushButton("📁 全て折りたたみ")
        self.btn_collapse_all.setProperty("role", "outline")
        self.btn_collapse_all.setStyleSheet("padding: 2px 6px; font-size: 10px;")
        self.btn_collapse_all.setToolTip("すべての設定セクションを折りたたみ、タイトルバーのみにします")
        self.btn_collapse_all.clicked.connect(self._collapse_all_panels)
        collapse_ctrl_row.addWidget(self.btn_collapse_all)
        collapse_ctrl_row.addStretch()
        sidebar_layout.addLayout(collapse_ctrl_row)

        # 2. 3D表示・カメラ操作パネル (数学座標系: Z-up)
        view_group = CollapsibleGroupBox("🎥 3D表示・カメラ操作")
        self.view_group = view_group
        view_layout = QGridLayout(view_group.content_widget)
        view_group.toggled_collapse.connect(self._on_panel_collapse_toggled)

        # 視点プリセット
        view_layout.addWidget(QLabel("視点切替:"), 0, 0)
        view_btn_layout = QHBoxLayout()
        view_btn_layout.setSpacing(4)
        self.btn_view_iso = QPushButton("等角")
        self.btn_view_iso.clicked.connect(lambda: self.viewer.reset_view())
        self.btn_view_top = QPushButton("上面(XY)")
        self.btn_view_top.clicked.connect(lambda: self.viewer.set_view_top())
        self.btn_view_front = QPushButton("正面(XZ)")
        self.btn_view_front.clicked.connect(lambda: self.viewer.set_view_front())
        self.btn_view_side = QPushButton("側面(YZ)")
        self.btn_view_side.clicked.connect(lambda: self.viewer.set_view_side())

        for b in [self.btn_view_iso, self.btn_view_top, self.btn_view_front, self.btn_view_side]:
            b.setProperty("role", "outline")
            b.setStyleSheet("padding: 4px 4px; font-size: 11px;")
            view_btn_layout.addWidget(b)
        view_layout.addLayout(view_btn_layout, 0, 1)

        # ★ 2点ピック斜め壁アライン視点切替 (直交・横・縦表示)
        view_layout.addWidget(QLabel("2点壁視点:"), 1, 0)
        wall_view_col = QVBoxLayout()
        wall_view_col.setSpacing(3)
        wall_v_row1 = QHBoxLayout()
        wall_v_row1.setSpacing(3)
        self.btn_wall_pick = QPushButton("🧭 2点ピック(U)")
        self.btn_wall_pick.setCheckable(True)
        self.btn_wall_pick.setStyleSheet("background-color: #701a75; font-size: 10px; font-weight: bold; padding: 3px 4px;")
        self.btn_wall_pick.setToolTip("点群上の壁2点をクリックして、斜めの壁に直交する横・縦表示へ合わせます (ショートカット: U)")
        self.btn_wall_pick.toggled.connect(self._toggle_wall_align_mode)
        wall_v_row1.addWidget(self.btn_wall_pick, 1)

        self.btn_wall_horiz = QPushButton("━ 横(上面)")
        self.btn_wall_horiz.setProperty("role", "outline")
        self.btn_wall_horiz.setStyleSheet("padding: 3px 3px; font-size: 10px; font-weight: bold;")
        self.btn_wall_horiz.setToolTip("ピックした2点ラインが画面の『横(水平)』になり、直交方向が上下になる上面正射影ビュー")
        self.btn_wall_horiz.clicked.connect(lambda: self._apply_wall_view("horiz"))
        wall_v_row1.addWidget(self.btn_wall_horiz, 1)

        self.btn_wall_vert = QPushButton("┃ 縦(上面)")
        self.btn_wall_vert.setProperty("role", "outline")
        self.btn_wall_vert.setStyleSheet("padding: 3px 3px; font-size: 10px; font-weight: bold;")
        self.btn_wall_vert.setToolTip("ピックした2点ラインが画面の『縦(垂直)』になり、直交方向が左右になる上面正射影ビュー")
        self.btn_wall_vert.clicked.connect(lambda: self._apply_wall_view("vert"))
        wall_v_row1.addWidget(self.btn_wall_vert, 1)
        wall_view_col.addLayout(wall_v_row1)

        wall_v_row2 = QHBoxLayout()
        wall_v_row2.setSpacing(3)
        self.btn_wall_ortho = QPushButton("🧱 直交正面")
        self.btn_wall_ortho.setProperty("role", "outline")
        self.btn_wall_ortho.setStyleSheet("padding: 3px 3px; font-size: 10px; font-weight: bold;")
        self.btn_wall_ortho.setToolTip("ピックした2点ラインに対して直交方向から水平に壁正面を見る立面ビュー (横=壁, 縦=高さZ)")
        self.btn_wall_ortho.clicked.connect(lambda: self._apply_wall_view("ortho_front"))
        wall_v_row2.addWidget(self.btn_wall_ortho, 1)

        self.btn_wall_edge = QPushButton("🔪 壁真横")
        self.btn_wall_edge.setProperty("role", "outline")
        self.btn_wall_edge.setStyleSheet("padding: 3px 3px; font-size: 10px; font-weight: bold;")
        self.btn_wall_edge.setToolTip("ピックした2点ラインの延長方向から壁を真横(断面エッジ)で見るビュー (壁前後の浮遊ノイズ除去に最適)")
        self.btn_wall_edge.clicked.connect(lambda: self._apply_wall_view("edge"))
        wall_v_row2.addWidget(self.btn_wall_edge, 1)

        self.btn_wall_flip = QPushButton("🔄 180°反転")
        self.btn_wall_flip.setProperty("role", "outline")
        self.btn_wall_flip.setStyleSheet("padding: 3px 3px; font-size: 10px;")
        self.btn_wall_flip.setToolTip("現在の2点壁アライン視点の向き(表裏・上下左右)を180°反転します")
        self.btn_wall_flip.clicked.connect(self._flip_wall_view)
        wall_v_row2.addWidget(self.btn_wall_flip, 1)
        wall_view_col.addLayout(wall_v_row2)
        view_layout.addLayout(wall_view_col, 1, 1)

        # 投影切り替え & 全体フィット
        view_layout.addWidget(QLabel("カメラ制御:"), 2, 0)
        cam_ctrl_layout = QHBoxLayout()
        cam_ctrl_layout.setSpacing(4)
        self.btn_fit = QPushButton("全体フィット (F)")
        self.btn_fit.setProperty("role", "outline")
        self.btn_fit.clicked.connect(lambda: self.viewer.fit_to_screen())
        self.btn_fit.setStyleSheet("padding: 4px 6px; font-size: 11px;")
        cam_ctrl_layout.addWidget(self.btn_fit)

        self.btn_proj = QPushButton("透視 ⇄ 正射影 (P)")
        self.btn_proj.setProperty("role", "outline")
        self.btn_proj.clicked.connect(lambda: self.viewer.toggle_projection())
        self.btn_proj.setStyleSheet("padding: 4px 6px; font-size: 11px;")
        cam_ctrl_layout.addWidget(self.btn_proj)
        view_layout.addLayout(cam_ctrl_layout, 2, 1)

        # 配色モード (以前のパレット完全保持 + 新パレット直接選択)
        view_layout.addWidget(QLabel("配色・パレット:"), 3, 0)
        self.color_combo = QComboBox()
        self.color_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.color_combo.setMinimumContentsLength(10)
        self.color_combo.addItems(PointCloudViewer.ALL_COLOR_MODES)
        self.color_combo.currentTextChanged.connect(self._on_color_mode_changed)
        self.color_combo.setToolTip("点群の色分け表示を1クリックで切り替えます (以前のViridis/Jetおよび新パレットを完全網羅)")
        view_layout.addWidget(self.color_combo, 3, 1)

        # 📏 高さ色分けレンジ調整パネル (任意高さスライダー & 全高リセット)
        self._height_bounds_min = 0.0
        self._height_bounds_max = 10.0
        self._height_slider_scale = 100.0  # 0.01m (1cm) 単位

        self.height_range_frame = QFrame()
        self.height_range_frame.setObjectName("height_range_frame")
        self.height_range_frame.setStyleSheet("""
            QFrame#height_range_frame {
                background-color: #172033;
                border: 1px solid #334155;
                border-radius: 6px;
            }
        """)
        hr_layout = QVBoxLayout(self.height_range_frame)
        hr_layout.setContentsMargins(6, 6, 6, 6)
        hr_layout.setSpacing(4)

        # ヘッダー行: チェックボックス & 全高リセットボタン
        hr_header = QHBoxLayout()
        hr_header.setSpacing(4)
        self.chk_custom_height = QCheckBox("📏 任意高さで色分け")
        self.chk_custom_height.setChecked(False)
        self.chk_custom_height.setStyleSheet("font-weight: bold; color: #38bdf8; font-size: 11px;")
        self.chk_custom_height.setToolTip("ONにすると、指定した下限〜上限の高さ範囲でグラデーション色を集中させます")
        self.chk_custom_height.toggled.connect(self._on_custom_height_toggled)
        hr_header.addWidget(self.chk_custom_height, 1)

        self.btn_reset_height_range = QPushButton("🔄 全高リセット")
        self.btn_reset_height_range.setProperty("role", "outline")
        self.btn_reset_height_range.setStyleSheet("padding: 2px 5px; font-size: 10px;")
        self.btn_reset_height_range.setToolTip("点群全体の最低高さ・最高高さに範囲をリセットします")
        self.btn_reset_height_range.clicked.connect(self._reset_height_range)
        hr_header.addWidget(self.btn_reset_height_range)
        hr_layout.addLayout(hr_header)

        # 下限 (Min Z) スライダー & スピンボックス
        min_row = QHBoxLayout()
        min_row.setSpacing(4)
        lbl_min = QLabel("下限:")
        lbl_min.setStyleSheet("color: #60a5fa; font-weight: bold; font-size: 11px;")
        min_row.addWidget(lbl_min)

        self.slider_height_min = QSlider(Qt.Horizontal)
        self.slider_height_min.setRange(0, 1000)
        self.slider_height_min.setValue(0)
        self.slider_height_min.valueChanged.connect(self._on_slider_height_min_changed)
        self.slider_height_min.sliderReleased.connect(self._apply_custom_height_range)
        min_row.addWidget(self.slider_height_min, 1)

        self.spin_height_min = QDoubleSpinBox()
        self.spin_height_min.setDecimals(2)
        self.spin_height_min.setSingleStep(0.1)
        self.spin_height_min.setSuffix(" m")
        self.spin_height_min.setRange(-10000.0, 10000.0)
        self.spin_height_min.setValue(0.0)
        self.spin_height_min.setStyleSheet("min-width: 62px; font-size: 11px;")
        self.spin_height_min.valueChanged.connect(self._on_spin_height_min_changed)
        min_row.addWidget(self.spin_height_min)
        hr_layout.addLayout(min_row)

        # 上限 (Max Z) スライダー & スピンボックス
        max_row = QHBoxLayout()
        max_row.setSpacing(4)
        lbl_max = QLabel("上限:")
        lbl_max.setStyleSheet("color: #f87171; font-weight: bold; font-size: 11px;")
        max_row.addWidget(lbl_max)

        self.slider_height_max = QSlider(Qt.Horizontal)
        self.slider_height_max.setRange(0, 1000)
        self.slider_height_max.setValue(1000)
        self.slider_height_max.valueChanged.connect(self._on_slider_height_max_changed)
        self.slider_height_max.sliderReleased.connect(self._apply_custom_height_range)
        max_row.addWidget(self.slider_height_max, 1)

        self.spin_height_max = QDoubleSpinBox()
        self.spin_height_max.setDecimals(2)
        self.spin_height_max.setSingleStep(0.1)
        self.spin_height_max.setSuffix(" m")
        self.spin_height_max.setRange(-10000.0, 10000.0)
        self.spin_height_max.setValue(10.0)
        self.spin_height_max.setStyleSheet("min-width: 62px; font-size: 11px;")
        self.spin_height_max.valueChanged.connect(self._on_spin_height_max_changed)
        max_row.addWidget(self.spin_height_max)
        hr_layout.addLayout(max_row)

        # 現在の幅・情報ラベル
        self.lbl_height_range_info = QLabel("配色範囲: 全高自動 (スライダーで任意指定)")
        self.lbl_height_range_info.setStyleSheet("color: #94a3b8; font-size: 10px;")
        hr_layout.addWidget(self.lbl_height_range_info)

        view_layout.addWidget(self.height_range_frame, 4, 0, 1, 2)

        # 3D表示強化・撮影コントロール行
        view_layout.addWidget(QLabel("3D強化・撮影:"), 5, 0)
        enh_row = QHBoxLayout()
        enh_row.setSpacing(4)

        self.btn_toggle_shading = QPushButton("✨ 凹凸陰影")
        self.btn_toggle_shading.setCheckable(True)
        self.btn_toggle_shading.setChecked(True)
        self.btn_toggle_shading.setProperty("role", "segment")
        self.btn_toggle_shading.setStyleSheet("padding: 3px 5px; font-size: 10px; font-weight: bold;")
        self.btn_toggle_shading.setToolTip("点群表面の法線近似・エッジ角度変化による立体陰影・輪郭強調 (疑似EDL) をON/OFF切替")
        self.btn_toggle_shading.toggled.connect(lambda chk: self.viewer.set_shading_enabled(chk))
        enh_row.addWidget(self.btn_toggle_shading, 1)

        self.btn_toggle_adaptive_size = QPushButton("🔍 距離連動サイズ")
        self.btn_toggle_adaptive_size.setCheckable(True)
        self.btn_toggle_adaptive_size.setChecked(True)
        self.btn_toggle_adaptive_size.setProperty("role", "segment")
        self.btn_toggle_adaptive_size.setStyleSheet("padding: 3px 5px; font-size: 10px; font-weight: bold;")
        self.btn_toggle_adaptive_size.setToolTip("カメラが近づくほど点サイズを自動拡大して隙間を埋め、引くとシャープに引き締める自動補正")
        self.btn_toggle_adaptive_size.toggled.connect(lambda chk: self.viewer.set_adaptive_point_size(chk))
        enh_row.addWidget(self.btn_toggle_adaptive_size, 1)

        self.btn_screenshot = QPushButton("📷 スクショ")
        self.btn_screenshot.setProperty("role", "outline")
        self.btn_screenshot.setStyleSheet("padding: 3px 6px; font-size: 10px; font-weight: bold;")
        self.btn_screenshot.setToolTip("現在の3D表示を高画質キャプチャしてクリップボードにコピー＆画像保存ダイアログを表示")
        self.btn_screenshot.clicked.connect(self._on_capture_screenshot)
        enh_row.addWidget(self.btn_screenshot)

        view_layout.addLayout(enh_row, 5, 1)

        # スクショ設定行 (グリッドON/OFF, XYZ凡例ON/OFF, 文字類は自動除外)
        ss_opt_row = QHBoxLayout()
        ss_opt_row.setSpacing(4)
        lbl_ss = QLabel("スクショ設定:")
        lbl_ss.setStyleSheet("color: #cbd5e1; font-size: 11px;")
        ss_opt_row.addWidget(lbl_ss)

        self.chk_ss_grid = QCheckBox("グリッド")
        self.chk_ss_grid.setChecked(False)
        self.chk_ss_grid.setStyleSheet("font-size: 11px;")
        self.chk_ss_grid.setToolTip("スクリーンショットに背景グリッド(床格子)を含めるか設定します (文字類は自動で除外)")
        self.chk_ss_grid.toggled.connect(self._on_ss_options_changed)
        ss_opt_row.addWidget(self.chk_ss_grid)

        self.chk_ss_axes = QCheckBox("XYZ凡例")
        self.chk_ss_axes.setChecked(True)
        self.chk_ss_axes.setStyleSheet("font-size: 11px;")
        self.chk_ss_axes.setToolTip("スクリーンショットに左下のXYZ座標軸凡例を含めるか設定します (文字類は自動で除外)")
        self.chk_ss_axes.toggled.connect(self._on_ss_options_changed)
        ss_opt_row.addWidget(self.chk_ss_axes)
        ss_opt_row.addStretch()
        view_layout.addLayout(ss_opt_row, 6, 0, 1, 2)

        view_hint = QLabel("※点サイズ・形状・背景色・描画点数・マウス感度は『⚙設定』タブにあります")
        view_hint.setWordWrap(True)
        view_hint.setStyleSheet("color: #94a3b8; font-size: 10px; font-weight: normal;")
        view_layout.addWidget(view_hint, 7, 0, 1, 2)

        sidebar_layout.addWidget(view_group)

        # 3. XYZ別 シーク断面スライサー (シーク位置基点モード・厚み直接入力・視点追従対応)
        section_group = CollapsibleGroupBox("📐 XYZ断面スライサー (シーク基点 & 厚み)")
        sec_layout = QVBoxLayout(section_group.content_widget)
        section_group.toggled_collapse.connect(self._on_panel_collapse_toggled)
        sec_layout.setSpacing(5)
        self._sec_pos_scale = 100.0  # 0.01m (1cm) 単位の高精度シークスケール

        # ① 基点基準モード選択 (今シークしている位置を中心にするか、片側へ厚みを取るか)
        sec_mode_row = QHBoxLayout()
        sec_mode_row.setSpacing(4)
        sec_mode_row.addWidget(QLabel("基点:"))
        self.sec_origin_mode_combo = QComboBox()
        self.sec_origin_mode_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.sec_origin_mode_combo.setMinimumContentsLength(6)
        self.sec_origin_mode_combo.addItem("🔄 シーク位置を中心 (±厚み/2)", "center")
        self.sec_origin_mode_combo.addItem("➕ シーク基点から＋方向 (0〜+厚み)", "plus")
        self.sec_origin_mode_combo.addItem("➖ シーク基点から－方向 (-厚み〜0)", "minus")
        self.sec_origin_mode_combo.addItem("⬆️ シーク基点から＋側全域", "above")
        self.sec_origin_mode_combo.addItem("⬇️ シーク基点から－側全域", "below")
        self.sec_origin_mode_combo.setToolTip(
            "今シークしている位置（黄色の基準面）から、どの方向に厚みを持たせて断面を表示するかを切り替えます"
        )
        self.sec_origin_mode_combo.currentIndexChanged.connect(self._on_section_origin_mode_changed)
        sec_mode_row.addWidget(self.sec_origin_mode_combo, 1)
        sec_layout.addLayout(sec_mode_row)

        # 枠線表示切替行 (黄色の基点枠・水色の厚み枠の個別ON/OFF & 一括切替)
        sec_guide_row = QHBoxLayout()
        sec_guide_row.setSpacing(4)
        sec_guide_row.addWidget(QLabel("枠線:"))
        self.chk_sec_show_origin = QCheckBox("基点(黄)")
        self.chk_sec_show_origin.setChecked(True)
        self.chk_sec_show_origin.setToolTip("シーク基点の黄色いプレーン枠線・半透明面の表示ON/OFF (点群観察時に邪魔な場合はOFFに)")
        self.chk_sec_show_origin.toggled.connect(self._on_section_guide_visibility_changed)
        sec_guide_row.addWidget(self.chk_sec_show_origin)

        self.chk_sec_show_box = QCheckBox("厚み(水色)")
        self.chk_sec_show_box.setChecked(True)
        self.chk_sec_show_box.setToolTip("断面スライスの厚み範囲（水色枠）の表示ON/OFF")
        self.chk_sec_show_box.toggled.connect(self._on_section_guide_visibility_changed)
        sec_guide_row.addWidget(self.chk_sec_show_box)

        self.btn_toggle_sec_guides = QPushButton("切替(B)")
        self.btn_toggle_sec_guides.setProperty("role", "outline")
        self.btn_toggle_sec_guides.setMaximumWidth(52)
        self.btn_toggle_sec_guides.setStyleSheet("padding: 1px 3px; font-size: 10px;")
        self.btn_toggle_sec_guides.setToolTip("断面の基点枠・厚み枠の表示を一括トグル切替 (ショートカット: Bキー)")
        self.btn_toggle_sec_guides.clicked.connect(self._toggle_section_guides_ui)
        sec_guide_row.addWidget(self.btn_toggle_sec_guides)
        sec_guide_row.addStretch()
        sec_layout.addLayout(sec_guide_row)

        # ② 現在見ている場所/選択点を一発でシーク基点にセット & シーク時のカメラ追従
        sec_focus_row = QHBoxLayout()
        sec_focus_row.setSpacing(4)
        self.btn_sec_from_current = QPushButton("🎯 現在位置/選択点を基点に")
        self.btn_sec_from_current.setStyleSheet(
            "background-color: #0369a1; font-size: 11px; font-weight: bold; padding: 4px 6px;"
        )
        self.btn_sec_from_current.setToolTip(
            "今画面で拡大・パンして見ている中心位置（または選択中の点群の重心）を、XYZ断面スライダーのシーク基点に一発セットします"
        )
        self.btn_sec_from_current.clicked.connect(self._set_section_origin_from_current_view)
        sec_focus_row.addWidget(self.btn_sec_from_current, 1)

        self.chk_sec_follow_cam = QCheckBox("視点追従")
        self.chk_sec_follow_cam.setChecked(True)
        self.chk_sec_follow_cam.setToolTip(
            "ONにすると、断面スライダーをシークした際にその断面基点位置へカメラの注視点が自動で追従します"
        )
        sec_focus_row.addWidget(self.chk_sec_follow_cam)
        sec_layout.addLayout(sec_focus_row)

        # ③ ワンタッチ厚みプリセットバー (薄切り 0.1m 〜 10m)
        sec_thick_preset_row = QHBoxLayout()
        sec_thick_preset_row.setSpacing(3)
        lbl_tp = QLabel("厚み一括:")
        lbl_tp.setStyleSheet("color: #cbd5e1; font-size: 11px;")
        sec_thick_preset_row.addWidget(lbl_tp)
        for t_val, t_txt in [(0.1, "0.1m"), (0.5, "0.5m"), (1.0, "1m"), (2.0, "2m"), (5.0, "5m")]:
            btn_t = QPushButton(t_txt)
            btn_t.setProperty("role", "outline")
            btn_t.setStyleSheet("padding: 2px 3px; font-size: 10px; font-weight: bold;")
            btn_t.setToolTip(f"断面厚みを {t_val}m にワンクリック変更します")
            btn_t.clicked.connect(lambda _=False, v=t_val: self._apply_section_thickness_preset(v))
            sec_thick_preset_row.addWidget(btn_t, 1)
        sec_layout.addLayout(sec_thick_preset_row)

        # X断面
        x_box = QWidget()
        x_layout = QVBoxLayout(x_box)
        x_layout.setContentsMargins(0, 2, 0, 2)
        x_layout.setSpacing(2)
        x_head = QHBoxLayout()
        x_head.setSpacing(4)
        self.chk_sec_x = QCheckBox("X断面(左右)")
        self.chk_sec_x.toggled.connect(lambda: self._on_section_x_changed(from_user_seek=False))
        x_head.addWidget(self.chk_sec_x)
        self.lbl_sec_x = QLabel("基点0.00m (±1.00m)")
        self.lbl_sec_x.setStyleSheet("font-size: 10px; color: #38bdf8;")
        x_head.addStretch()
        x_head.addWidget(self.lbl_sec_x)
        self.btn_view_sec_x = QPushButton("正対")
        self.btn_view_sec_x.setProperty("role", "outline")
        self.btn_view_sec_x.setMaximumWidth(40)
        self.btn_view_sec_x.setStyleSheet("padding: 1px 4px; font-size: 10px;")
        self.btn_view_sec_x.setToolTip("X断面を有効化し、カメラを側面(YZ断面)へ正対させます")
        self.btn_view_sec_x.clicked.connect(lambda: self._align_to_section_axis("x"))
        x_head.addWidget(self.btn_view_sec_x)
        x_layout.addLayout(x_head)

        x_pos_row = QHBoxLayout()
        x_pos_row.setSpacing(4)
        x_pos_row.addWidget(QLabel("シーク:"))
        self.slider_pos_x = QSlider(Qt.Horizontal)
        self.slider_pos_x.setRange(-10000, 10000)
        self.slider_pos_x.setValue(0)
        self.slider_pos_x.valueChanged.connect(self._on_slider_pos_x_changed)
        self.slider_pos_x.sliderReleased.connect(self._commit_pending_section_slices)
        x_pos_row.addWidget(self.slider_pos_x, 1)
        self.spin_pos_x = QDoubleSpinBox()
        self.spin_pos_x.setDecimals(2)
        self.spin_pos_x.setRange(-1e7, 1e7)
        self.spin_pos_x.setSingleStep(0.1)
        self.spin_pos_x.setValue(0.0)
        self.spin_pos_x.setSuffix(" m")
        self.spin_pos_x.setMaximumWidth(78)
        self.spin_pos_x.setToolTip("X断面のシーク基点座標 (m)")
        self.spin_pos_x.valueChanged.connect(self._on_spin_pos_x_changed)
        x_pos_row.addWidget(self.spin_pos_x)
        self.sec_x_detail = QWidget()
        x_detail_l = QVBoxLayout(self.sec_x_detail)
        x_detail_l.setContentsMargins(14, 0, 0, 0)
        x_detail_l.setSpacing(2)
        x_detail_l.addLayout(x_pos_row)

        x_thick_row = QHBoxLayout()
        x_thick_row.setSpacing(4)
        x_thick_row.addWidget(QLabel("厚み幅:"))
        self.slider_thick_x = QSlider(Qt.Horizontal)
        self.slider_thick_x.setRange(1, 500)  # 0.1m - 50.0m
        self.slider_thick_x.setValue(20)      # 2.0m
        self.slider_thick_x.valueChanged.connect(self._on_slider_thick_x_changed)
        self.slider_thick_x.sliderReleased.connect(self._commit_pending_section_slices)
        x_thick_row.addWidget(self.slider_thick_x, 1)
        self.spin_thick_x = QDoubleSpinBox()
        self.spin_thick_x.setDecimals(2)
        self.spin_thick_x.setRange(0.01, 5000.0)
        self.spin_thick_x.setSingleStep(0.1)
        self.spin_thick_x.setValue(2.0)
        self.spin_thick_x.setSuffix(" m")
        self.spin_thick_x.setMaximumWidth(78)
        self.spin_thick_x.setToolTip("X断面の厚み・スライス幅 (0.01m=1cm 〜 5000m)")
        self.spin_thick_x.valueChanged.connect(self._on_spin_thick_x_changed)
        x_thick_row.addWidget(self.spin_thick_x)
        x_detail_l.addLayout(x_thick_row)
        x_layout.addWidget(self.sec_x_detail)
        # コンパクト化: 断面チェックONの軸だけシーク・厚みスライダーを展開表示
        self.sec_x_detail.setVisible(False)
        self.chk_sec_x.toggled.connect(self.sec_x_detail.setVisible)
        sec_layout.addWidget(x_box)

        # Y断面
        y_box = QWidget()
        y_layout = QVBoxLayout(y_box)
        y_layout.setContentsMargins(0, 2, 0, 2)
        y_layout.setSpacing(2)
        y_head = QHBoxLayout()
        y_head.setSpacing(4)
        self.chk_sec_y = QCheckBox("Y断面(奥行)")
        self.chk_sec_y.toggled.connect(lambda: self._on_section_y_changed(from_user_seek=False))
        y_head.addWidget(self.chk_sec_y)
        self.lbl_sec_y = QLabel("基点0.00m (±1.00m)")
        self.lbl_sec_y.setStyleSheet("font-size: 10px; color: #38bdf8;")
        y_head.addStretch()
        y_head.addWidget(self.lbl_sec_y)
        self.btn_view_sec_y = QPushButton("正対")
        self.btn_view_sec_y.setProperty("role", "outline")
        self.btn_view_sec_y.setMaximumWidth(40)
        self.btn_view_sec_y.setStyleSheet("padding: 1px 4px; font-size: 10px;")
        self.btn_view_sec_y.setToolTip("Y断面を有効化し、カメラを正面(XZ立面)へ正対させます")
        self.btn_view_sec_y.clicked.connect(lambda: self._align_to_section_axis("y"))
        y_head.addWidget(self.btn_view_sec_y)
        y_layout.addLayout(y_head)

        y_pos_row = QHBoxLayout()
        y_pos_row.setSpacing(4)
        y_pos_row.addWidget(QLabel("シーク:"))
        self.slider_pos_y = QSlider(Qt.Horizontal)
        self.slider_pos_y.setRange(-10000, 10000)
        self.slider_pos_y.setValue(0)
        self.slider_pos_y.valueChanged.connect(self._on_slider_pos_y_changed)
        self.slider_pos_y.sliderReleased.connect(self._commit_pending_section_slices)
        y_pos_row.addWidget(self.slider_pos_y, 1)
        self.spin_pos_y = QDoubleSpinBox()
        self.spin_pos_y.setDecimals(2)
        self.spin_pos_y.setRange(-1e7, 1e7)
        self.spin_pos_y.setSingleStep(0.1)
        self.spin_pos_y.setValue(0.0)
        self.spin_pos_y.setSuffix(" m")
        self.spin_pos_y.setMaximumWidth(78)
        self.spin_pos_y.setToolTip("Y断面のシーク基点座標 (m)")
        self.spin_pos_y.valueChanged.connect(self._on_spin_pos_y_changed)
        y_pos_row.addWidget(self.spin_pos_y)
        self.sec_y_detail = QWidget()
        y_detail_l = QVBoxLayout(self.sec_y_detail)
        y_detail_l.setContentsMargins(14, 0, 0, 0)
        y_detail_l.setSpacing(2)
        y_detail_l.addLayout(y_pos_row)

        y_thick_row = QHBoxLayout()
        y_thick_row.setSpacing(4)
        y_thick_row.addWidget(QLabel("厚み幅:"))
        self.slider_thick_y = QSlider(Qt.Horizontal)
        self.slider_thick_y.setRange(1, 500)
        self.slider_thick_y.setValue(20)
        self.slider_thick_y.valueChanged.connect(self._on_slider_thick_y_changed)
        self.slider_thick_y.sliderReleased.connect(self._commit_pending_section_slices)
        y_thick_row.addWidget(self.slider_thick_y, 1)
        self.spin_thick_y = QDoubleSpinBox()
        self.spin_thick_y.setDecimals(2)
        self.spin_thick_y.setRange(0.01, 5000.0)
        self.spin_thick_y.setSingleStep(0.1)
        self.spin_thick_y.setValue(2.0)
        self.spin_thick_y.setSuffix(" m")
        self.spin_thick_y.setMaximumWidth(78)
        self.spin_thick_y.setToolTip("Y断面の厚み・スライス幅 (0.01m=1cm 〜 5000m)")
        self.spin_thick_y.valueChanged.connect(self._on_spin_thick_y_changed)
        y_thick_row.addWidget(self.spin_thick_y)
        y_detail_l.addLayout(y_thick_row)
        y_layout.addWidget(self.sec_y_detail)
        # コンパクト化: 断面チェックONの軸だけシーク・厚みスライダーを展開表示
        self.sec_y_detail.setVisible(False)
        self.chk_sec_y.toggled.connect(self.sec_y_detail.setVisible)
        sec_layout.addWidget(y_box)

        # Z断面
        z_box = QWidget()
        z_layout = QVBoxLayout(z_box)
        z_layout.setContentsMargins(0, 2, 0, 2)
        z_layout.setSpacing(2)
        z_head = QHBoxLayout()
        z_head.setSpacing(4)
        self.chk_sec_z = QCheckBox("Z断面(高さ)")
        self.chk_sec_z.toggled.connect(lambda: self._on_section_z_changed(from_user_seek=False))
        z_head.addWidget(self.chk_sec_z)
        self.lbl_sec_z = QLabel("基点0.00m (±1.00m)")
        self.lbl_sec_z.setStyleSheet("font-size: 10px; color: #38bdf8;")
        z_head.addStretch()
        z_head.addWidget(self.lbl_sec_z)
        self.btn_view_sec_z = QPushButton("正対")
        self.btn_view_sec_z.setProperty("role", "outline")
        self.btn_view_sec_z.setMaximumWidth(40)
        self.btn_view_sec_z.setStyleSheet("padding: 1px 4px; font-size: 10px;")
        self.btn_view_sec_z.setToolTip("Z断面を有効化し、カメラを上面(XY水平断面)へ正対させます")
        self.btn_view_sec_z.clicked.connect(lambda: self._align_to_section_axis("z"))
        z_head.addWidget(self.btn_view_sec_z)
        z_layout.addLayout(z_head)

        z_pos_row = QHBoxLayout()
        z_pos_row.setSpacing(4)
        z_pos_row.addWidget(QLabel("シーク:"))
        self.slider_pos_z = QSlider(Qt.Horizontal)
        self.slider_pos_z.setRange(-10000, 10000)
        self.slider_pos_z.setValue(0)
        self.slider_pos_z.valueChanged.connect(self._on_slider_pos_z_changed)
        self.slider_pos_z.sliderReleased.connect(self._commit_pending_section_slices)
        z_pos_row.addWidget(self.slider_pos_z, 1)
        self.spin_pos_z = QDoubleSpinBox()
        self.spin_pos_z.setDecimals(2)
        self.spin_pos_z.setRange(-1e7, 1e7)
        self.spin_pos_z.setSingleStep(0.1)
        self.spin_pos_z.setValue(0.0)
        self.spin_pos_z.setSuffix(" m")
        self.spin_pos_z.setMaximumWidth(78)
        self.spin_pos_z.setToolTip("Z断面のシーク基点座標 (m)")
        self.spin_pos_z.valueChanged.connect(self._on_spin_pos_z_changed)
        z_pos_row.addWidget(self.spin_pos_z)
        self.sec_z_detail = QWidget()
        z_detail_l = QVBoxLayout(self.sec_z_detail)
        z_detail_l.setContentsMargins(14, 0, 0, 0)
        z_detail_l.setSpacing(2)
        z_detail_l.addLayout(z_pos_row)

        z_thick_row = QHBoxLayout()
        z_thick_row.setSpacing(4)
        z_thick_row.addWidget(QLabel("厚み幅:"))
        self.slider_thick_z = QSlider(Qt.Horizontal)
        self.slider_thick_z.setRange(1, 500)
        self.slider_thick_z.setValue(20)
        self.slider_thick_z.valueChanged.connect(self._on_slider_thick_z_changed)
        self.slider_thick_z.sliderReleased.connect(self._commit_pending_section_slices)
        z_thick_row.addWidget(self.slider_thick_z, 1)
        self.spin_thick_z = QDoubleSpinBox()
        self.spin_thick_z.setDecimals(2)
        self.spin_thick_z.setRange(0.01, 5000.0)
        self.spin_thick_z.setSingleStep(0.1)
        self.spin_thick_z.setValue(2.0)
        self.spin_thick_z.setSuffix(" m")
        self.spin_thick_z.setMaximumWidth(78)
        self.spin_thick_z.setToolTip("Z断面の厚み・スライス幅 (0.01m=1cm 〜 5000m)")
        self.spin_thick_z.valueChanged.connect(self._on_spin_thick_z_changed)
        z_thick_row.addWidget(self.spin_thick_z)
        z_detail_l.addLayout(z_thick_row)
        z_layout.addWidget(self.sec_z_detail)
        # コンパクト化: 断面チェックONの軸だけシーク・厚みスライダーを展開表示
        self.sec_z_detail.setVisible(False)
        self.chk_sec_z.toggled.connect(self.sec_z_detail.setVisible)
        sec_layout.addWidget(z_box)

        # ★ 2点ピック斜め壁アライン & 斜め壁直交スライス (斜めの壁に沿ったノイズ処理用)
        wall_box = QWidget()
        wall_box.setStyleSheet("background-color: #1a1626; border: 1px solid #4c1d95; border-radius: 6px;")
        wall_layout = QVBoxLayout(wall_box)
        wall_layout.setContentsMargins(5, 4, 5, 4)
        wall_layout.setSpacing(3)

        wall_head = QHBoxLayout()
        wall_head.setSpacing(4)
        self.chk_sec_wall = QCheckBox("斜め壁スライス")
        self.chk_sec_wall.setToolTip("ピックした2点ライン(斜めの壁)に沿って、壁と直交する前後範囲だけを断面スライス表示します")
        self.chk_sec_wall.toggled.connect(lambda: self._on_section_wall_changed(from_user_seek=False))
        wall_head.addWidget(self.chk_sec_wall)
        self.lbl_sec_wall = QLabel("未ピック(Uで2点指定)")
        self.lbl_sec_wall.setStyleSheet("font-size: 10px; color: #e879f9;")
        wall_head.addStretch()
        wall_head.addWidget(self.lbl_sec_wall)
        wall_layout.addLayout(wall_head)

        wall_btn_row1 = QHBoxLayout()
        wall_btn_row1.setSpacing(3)
        self.btn_sec_wall_pick = QPushButton("🧭 ピック(U)")
        self.btn_sec_wall_pick.setCheckable(True)
        self.btn_sec_wall_pick.setStyleSheet("background-color: #701a75; font-size: 10px; font-weight: bold; padding: 3px 4px;")
        self.btn_sec_wall_pick.setToolTip("点群上の壁2点をクリックして、斜めの壁に直交する横・縦表示へ合わせます (Uキー)")
        self.btn_sec_wall_pick.toggled.connect(self._toggle_wall_align_mode)
        wall_btn_row1.addWidget(self.btn_sec_wall_pick, 1)

        for w_txt, w_type, w_tip in [
            ("━ 横", "horiz", "上面視で斜め壁を『横(水平)』に揃え、直交方向を上下にします"),
            ("┃ 縦", "vert", "上面視で斜め壁を『縦(垂直)』に揃え、直交方向を左右にします"),
            ("🔪 真横", "edge", "斜めの壁の延長方向から真横(断面エッジ)で見ます (壁から浮いたノイズの除去に最適)"),
        ]:
            btn_sw = QPushButton(w_txt)
            btn_sw.setProperty("role", "outline")
            btn_sw.setStyleSheet("padding: 3px 3px; font-size: 10px; font-weight: bold;")
            btn_sw.setToolTip(w_tip + "\n※直交正面・180°反転は『🎥表示』タブ／右クリックメニューにあります")
            btn_sw.clicked.connect(lambda _=False, t=w_type: self._apply_wall_view(t))
            wall_btn_row1.addWidget(btn_sw, 1)
        wall_layout.addLayout(wall_btn_row1)

        # 壁シーク・厚みはスライスON時のみ展開表示 (コンパクト化)
        self.sec_wall_detail = QWidget()
        self.sec_wall_detail.setStyleSheet("border: none;")
        wall_detail_l = QVBoxLayout(self.sec_wall_detail)
        wall_detail_l.setContentsMargins(0, 0, 0, 0)
        wall_detail_l.setSpacing(3)

        wall_pos_row = QHBoxLayout()
        wall_pos_row.setSpacing(4)
        wall_pos_row.addWidget(QLabel("壁シーク:"))
        self.slider_pos_wall = QSlider(Qt.Horizontal)
        self.slider_pos_wall.setRange(-2000, 2000)  # -20.00m 〜 +20.00m
        self.slider_pos_wall.setValue(0)
        self.slider_pos_wall.valueChanged.connect(self._on_slider_pos_wall_changed)
        self.slider_pos_wall.sliderReleased.connect(self._commit_pending_section_slices)
        wall_pos_row.addWidget(self.slider_pos_wall, 1)
        self.spin_pos_wall = QDoubleSpinBox()
        self.spin_pos_wall.setDecimals(2)
        self.spin_pos_wall.setRange(-500.0, 500.0)
        self.spin_pos_wall.setSingleStep(0.05)
        self.spin_pos_wall.setValue(0.0)
        self.spin_pos_wall.setSuffix(" m")
        self.spin_pos_wall.setMaximumWidth(78)
        self.spin_pos_wall.setToolTip("2点壁ラインからの直交オフセットシーク位置 (0.00m = 2点壁ライン上)")
        self.spin_pos_wall.valueChanged.connect(self._on_spin_pos_wall_changed)
        wall_pos_row.addWidget(self.spin_pos_wall)
        wall_detail_l.addLayout(wall_pos_row)

        wall_thick_row = QHBoxLayout()
        wall_thick_row.setSpacing(4)
        wall_thick_row.addWidget(QLabel("壁厚み幅:"))
        self.slider_thick_wall = QSlider(Qt.Horizontal)
        self.slider_thick_wall.setRange(1, 300)
        self.slider_thick_wall.setValue(10)  # 初期値 1.0m
        self.slider_thick_wall.valueChanged.connect(self._on_slider_thick_wall_changed)
        self.slider_thick_wall.sliderReleased.connect(self._commit_pending_section_slices)
        wall_thick_row.addWidget(self.slider_thick_wall, 1)
        self.spin_thick_wall = QDoubleSpinBox()
        self.spin_thick_wall.setDecimals(2)
        self.spin_thick_wall.setRange(0.01, 1000.0)
        self.spin_thick_wall.setSingleStep(0.05)
        self.spin_thick_wall.setValue(1.0)
        self.spin_thick_wall.setSuffix(" m")
        self.spin_thick_wall.setMaximumWidth(78)
        self.spin_thick_wall.setToolTip("2点斜め壁スライスの直交厚み幅 (m)")
        self.spin_thick_wall.valueChanged.connect(self._on_spin_thick_wall_changed)
        wall_thick_row.addWidget(self.spin_thick_wall)
        wall_detail_l.addLayout(wall_thick_row)
        wall_layout.addWidget(self.sec_wall_detail)
        self.sec_wall_detail.setVisible(False)
        self.chk_sec_wall.toggled.connect(self.sec_wall_detail.setVisible)

        sec_layout.addWidget(wall_box)

        # 断面ボックス外クロップ & リセットボタン
        sec_action_layout = QHBoxLayout()
        sec_action_layout.setSpacing(4)
        self.btn_crop_section = QPushButton("✂️ 断面ボックス外をノイズへ移動")
        self.btn_crop_section.setProperty("role", "danger")
        self.btn_crop_section.setStyleSheet("background-color: #9a3412; font-size: 11px; font-weight: bold; padding: 5px 6px;")
        self.btn_crop_section.clicked.connect(self._crop_outside_section)
        sec_action_layout.addWidget(self.btn_crop_section, 1)

        self.btn_reset_section = QPushButton("断面解除")
        self.btn_reset_section.setProperty("role", "outline")
        self.btn_reset_section.setStyleSheet("font-size: 11px; padding: 5px 6px;")
        self.btn_reset_section.setMaximumWidth(68)
        self.btn_reset_section.clicked.connect(self._reset_all_sections)
        sec_action_layout.addWidget(self.btn_reset_section)
        sec_layout.addLayout(sec_action_layout)

        self.section_group = section_group
        sidebar_layout.addWidget(section_group)

        # 4. レイヤー管理 (マルチレイヤー・ロック・スキャン別・自動分類・プロジェクト保存)
        layer_group = CollapsibleGroupBox("📑 レイヤー管理 (表示 / 🔒保護 / 自動分類)")
        self.layer_group = layer_group
        layer_layout = QVBoxLayout(layer_group.content_widget)
        layer_group.toggled_collapse.connect(self._on_panel_collapse_toggled)

        layer_row1 = QHBoxLayout()
        self.chk_layer_normal = QCheckBox("有効点群レイヤー全体")
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

        # マルチレイヤー動的リストコンテナ (表示ON/OFF, 色変更, 🔒ロック切替)
        self.layer_list_widget = QWidget()
        self.layer_list_layout = QVBoxLayout(self.layer_list_widget)
        self.layer_list_layout.setContentsMargins(2, 2, 2, 2)
        self.layer_list_layout.setSpacing(3)
        layer_layout.addWidget(self.layer_list_widget)

        # ① カスタムレイヤー追加 & 選択点のレイヤー移動
        add_layer_row = QHBoxLayout()
        add_layer_row.setSpacing(4)
        self.custom_layer_name_edit = QLineEdit()
        self.custom_layer_name_edit.setMinimumWidth(80)
        self.custom_layer_name_edit.setPlaceholderText("新規レイヤー名 (例: 構造物, 植生)")
        add_layer_row.addWidget(self.custom_layer_name_edit, 1)

        self.btn_add_layer = QPushButton("＋追加")
        self.btn_add_layer.setStyleSheet("background-color: #0f766e; font-size: 11px; font-weight: bold; padding: 5px 8px;")
        self.btn_add_layer.clicked.connect(self._on_add_custom_layer)
        add_layer_row.addWidget(self.btn_add_layer)
        layer_layout.addLayout(add_layer_row)

        move_layer_row = QHBoxLayout()
        move_layer_row.setSpacing(4)
        self.target_layer_combo = QComboBox()
        self.target_layer_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.target_layer_combo.setMinimumContentsLength(6)
        move_layer_row.addWidget(self.target_layer_combo, 1)

        self.btn_move_to_layer = QPushButton("選択点を移動")
        self.btn_move_to_layer.setProperty("role", "primary")
        self.btn_move_to_layer.setStyleSheet("font-size: 11px; font-weight: bold; padding: 5px 6px;")
        self.btn_move_to_layer.clicked.connect(self._on_move_selected_to_target_layer)
        move_layer_row.addWidget(self.btn_move_to_layer)

        self.btn_rename_layer = QPushButton("名変更")
        self.btn_rename_layer.setProperty("role", "outline")
        self.btn_rename_layer.setStyleSheet("font-size: 11px; padding: 5px 6px;")
        self.btn_rename_layer.clicked.connect(self._on_rename_target_layer)
        move_layer_row.addWidget(self.btn_rename_layer)
        layer_layout.addLayout(move_layer_row)

        # ⑥ 自動レイヤー振り分け (地面抽出 / 主平面検出)
        auto_seg_row = QHBoxLayout()
        auto_seg_row.setSpacing(4)
        self.btn_auto_ground = QPushButton("🏞️ 地面を自動抽出")
        self.btn_auto_ground.setToolTip("局所最低標高グリッド法により地面を自動検出し、地面レイヤーへ移動します")
        self.btn_auto_ground.setStyleSheet("background-color: #78350f; font-size: 11px; font-weight: bold; padding: 5px 6px;")
        self.btn_auto_ground.clicked.connect(self._on_auto_extract_ground)
        auto_seg_row.addWidget(self.btn_auto_ground, 1)

        self.btn_auto_plane = QPushButton("🧱 主平面を自動検出")
        self.btn_auto_plane.setToolTip("RANSACにより支配的な平面(床・壁・天井)を自動検出し、平面レイヤーへ移動します")
        self.btn_auto_plane.setStyleSheet("background-color: #065f46; font-size: 11px; font-weight: bold; padding: 5px 6px;")
        self.btn_auto_plane.clicked.connect(self._on_auto_extract_plane)
        auto_seg_row.addWidget(self.btn_auto_plane, 1)
        layer_layout.addLayout(auto_seg_row)

        # ノイズ復元ボタン行
        layer_btn_layout = QHBoxLayout()
        layer_btn_layout.setSpacing(4)
        self.btn_restore_selected_noise = QPushButton("選択ノイズのみ復元")
        self.btn_restore_selected_noise.setStyleSheet("background-color: #0369a1; font-size: 11px; font-weight: bold; padding: 5px 6px;")
        self.btn_restore_selected_noise.clicked.connect(self._restore_selected_noise)
        layer_btn_layout.addWidget(self.btn_restore_selected_noise, 1)

        self.btn_restore_noise = QPushButton("全ノイズを復元")
        self.btn_restore_noise.setProperty("role", "outline")
        self.btn_restore_noise.setStyleSheet("font-size: 11px; padding: 5px 6px;")
        self.btn_restore_noise.clicked.connect(self._restore_all_noise)
        layer_btn_layout.addWidget(self.btn_restore_noise, 1)
        layer_layout.addLayout(layer_btn_layout)

        # ③ スキャン(器械点/ステーション)別表示切替リスト
        self.scan_group_label = QLabel("📡 スキャン (器械点) 別の表示切替:")
        self.scan_group_label.setStyleSheet("color: #94a3b8; font-size: 11px; font-weight: bold; margin-top: 2px;")
        layer_layout.addWidget(self.scan_group_label)
        self.scan_list_widget = QWidget()
        self.scan_list_layout = QVBoxLayout(self.scan_list_widget)
        self.scan_list_layout.setContentsMargins(2, 0, 2, 0)
        self.scan_list_layout.setSpacing(2)
        layer_layout.addWidget(self.scan_list_widget)

        # ⑤ 作業途中のレイヤー状態プロジェクト保存・再開 (.npz)
        proj_btn_row = QHBoxLayout()
        proj_btn_row.setSpacing(4)
        self.btn_save_layer_proj = QPushButton("💾 状態保存 (.npz)")
        self.btn_save_layer_proj.setProperty("role", "outline")
        self.btn_save_layer_proj.setStyleSheet("font-size: 11px; padding: 5px 6px;")
        self.btn_save_layer_proj.clicked.connect(self._on_save_layer_project)
        proj_btn_row.addWidget(self.btn_save_layer_proj, 1)

        self.btn_load_layer_proj = QPushButton("📂 状態読込 (.npz)")
        self.btn_load_layer_proj.setProperty("role", "outline")
        self.btn_load_layer_proj.setStyleSheet("font-size: 11px; padding: 5px 6px;")
        self.btn_load_layer_proj.clicked.connect(self._on_load_layer_project)
        proj_btn_row.addWidget(self.btn_load_layer_proj, 1)
        layer_layout.addLayout(proj_btn_row)

        layer_info = QLabel("※🔒ロックしたレイヤーは手動削除・自動ノイズ除去・断面クロップから保護されます。")
        layer_info.setWordWrap(True)
        layer_info.setStyleSheet("color: #94a3b8; font-size: 11px;")
        layer_layout.addWidget(layer_info)

        sidebar_layout.addWidget(layer_group)

        # 5. 手動選択・編集・直感ノイズ処理ツール
        tools_group = CollapsibleGroupBox("✂️ 選択・編集・ノイズツール")
        self.tools_group = tools_group
        tools_layout = QVBoxLayout(tools_group.content_widget)
        tools_group.toggled_collapse.connect(self._on_panel_collapse_toggled)

        # ★ 専用モード切替 & 即時削除モード行
        mode_top_row = QHBoxLayout()
        mode_top_row.setSpacing(4)
        self.btn_clean_mode_side = QPushButton("🎯 ノイズ専用(N)")
        self.btn_clean_mode_side.setCheckable(True)
        self.btn_clean_mode_side.setToolTip(
            "ノイズ処理に特化した高コントラスト表示・ブラシ消去・囲み即削除モードへワンタッチで切り替えます (ショートカット: N)"
        )
        self.btn_clean_mode_side.setStyleSheet(
            "background-color: #1e293b; border: 1px solid #38bdf8; color: #38bdf8; font-size: 11px; font-weight: bold; padding: 4px 4px;"
        )
        self.btn_clean_mode_side.toggled.connect(self._toggle_clean_focus_mode)
        mode_top_row.addWidget(self.btn_clean_mode_side, 1)

        self.chk_instant_del = QPushButton("⚡ 囲み即削除: OFF")
        self.chk_instant_del.setCheckable(True)
        self.chk_instant_del.setToolTip(
            "ONにすると、矩形・円・多角形・ブラシで囲んだ瞬間にDeleteキー不要で即座にノイズレイヤーへ消去します"
        )
        self.chk_instant_del.setStyleSheet("font-size: 11px; font-weight: bold; padding: 4px 4px;")
        self.chk_instant_del.toggled.connect(self._toggle_instant_delete_mode)
        mode_top_row.addWidget(self.chk_instant_del, 1)
        tools_layout.addLayout(mode_top_row)

        # ★ 操作モード・ワンクリック切替ショートカットボタン群 (回転V / パンH / ブラシB / 矩形R / 円形C / 多角形L)
        quick_mode_row1 = QHBoxLayout()
        quick_mode_row1.setSpacing(3)
        self.side_btn_orbit = QPushButton("🖐 回転(V)")
        self.side_btn_orbit.setCheckable(True)
        self.side_btn_orbit.setChecked(True)
        self.side_btn_orbit.setProperty("role", "segment")
        self.side_btn_orbit.setStyleSheet("padding: 4px 2px; font-size: 11px; font-weight: bold;")
        self.side_btn_orbit.setToolTip("左ドラッグで視点回転モード (ショートカット: V / 再押下でパンと交互切替)")
        self.side_btn_orbit.clicked.connect(lambda: self._set_interaction_mode("orbit"))
        quick_mode_row1.addWidget(self.side_btn_orbit, 1)

        self.side_btn_pan = QPushButton("✋ パン(H)")
        self.side_btn_pan.setCheckable(True)
        self.side_btn_pan.setProperty("role", "segment")
        self.side_btn_pan.setStyleSheet("padding: 4px 2px; font-size: 11px; font-weight: bold;")
        self.side_btn_pan.setToolTip("左ドラッグで画面平行移動(パン)モード (ショートカット: H または V / Space長押しでも一時パン可)")
        self.side_btn_pan.clicked.connect(lambda: self._set_interaction_mode("pan"))
        quick_mode_row1.addWidget(self.side_btn_pan, 1)

        self.side_btn_brush = QPushButton("🧹 ブラシ(B)")
        self.side_btn_brush.setCheckable(True)
        self.side_btn_brush.setProperty("role", "segment")
        self.side_btn_brush.setStyleSheet("padding: 4px 2px; font-size: 11px; font-weight: bold;")
        self.side_btn_brush.setToolTip("円形ブラシでなぞって選択・消去 (ショートカット: B)")
        self.side_btn_brush.clicked.connect(lambda: self._set_interaction_mode("brush"))
        quick_mode_row1.addWidget(self.side_btn_brush, 1)
        tools_layout.addLayout(quick_mode_row1)

        quick_mode_row2 = QHBoxLayout()
        quick_mode_row2.setSpacing(3)
        self.side_btn_rect = QPushButton("⬜ 矩形(R)")
        self.side_btn_rect.setCheckable(True)
        self.side_btn_rect.setProperty("role", "segment")
        self.side_btn_rect.setStyleSheet("padding: 4px 2px; font-size: 11px; font-weight: bold;")
        self.side_btn_rect.setToolTip("四角形ドラッグ選択モード (ショートカット: R)")
        self.side_btn_rect.clicked.connect(lambda: self._set_interaction_mode("rect"))
        quick_mode_row2.addWidget(self.side_btn_rect, 1)

        self.side_btn_circle = QPushButton("⭕ 円形(C)")
        self.side_btn_circle.setCheckable(True)
        self.side_btn_circle.setProperty("role", "segment")
        self.side_btn_circle.setStyleSheet("padding: 4px 2px; font-size: 11px; font-weight: bold;")
        self.side_btn_circle.setToolTip("中心から円形ドラッグ選択モード (ショートカット: C)")
        self.side_btn_circle.clicked.connect(lambda: self._set_interaction_mode("circle"))
        quick_mode_row2.addWidget(self.side_btn_circle, 1)

        self.side_btn_poly = QPushButton("⬠ 多角形(L)")
        self.side_btn_poly.setCheckable(True)
        self.side_btn_poly.setProperty("role", "segment")
        self.side_btn_poly.setStyleSheet("padding: 4px 2px; font-size: 11px; font-weight: bold;")
        self.side_btn_poly.setToolTip("頂点クリックで多角形選択モード (ショートカット: L)")
        self.side_btn_poly.clicked.connect(lambda: self._set_interaction_mode("polygon"))
        quick_mode_row2.addWidget(self.side_btn_poly, 1)
        tools_layout.addLayout(quick_mode_row2)

        # 選択形状セレクタ & モード切替
        sel_shape_layout = QHBoxLayout()
        sel_shape_layout.setSpacing(4)
        sel_shape_layout.addWidget(QLabel("形状:"))
        self.select_shape_combo = QComboBox()
        self.select_shape_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.select_shape_combo.setMinimumContentsLength(6)
        self.select_shape_combo.addItem("🧹 ブラシ (なぞり円)", PointCloudViewer.SELECT_BRUSH)
        self.select_shape_combo.addItem("⬜ 矩形 (四角)", PointCloudViewer.SELECT_RECT)
        self.select_shape_combo.addItem("⭕ 円形 (サークル)", PointCloudViewer.SELECT_CIRCLE)
        self.select_shape_combo.addItem("⬠ 多角形 (ポリゴン)", PointCloudViewer.SELECT_POLYGON)
        self.select_shape_combo.setCurrentIndex(1)  # 初期値は矩形
        self.select_shape_combo.currentIndexChanged.connect(self._on_selection_shape_changed)
        sel_shape_layout.addWidget(self.select_shape_combo, 1)

        self.select_mode_btn = QPushButton("手動選択モード")
        self.select_mode_btn.setCheckable(True)
        self.select_mode_btn.setStyleSheet("font-size: 11px; padding: 5px 8px;")
        self.select_mode_btn.toggled.connect(self._toggle_selection_mode)
        sel_shape_layout.addWidget(self.select_mode_btn)
        tools_layout.addLayout(sel_shape_layout)

        # ブラシサイズ調整スライダー行
        brush_row = QHBoxLayout()
        brush_row.setSpacing(4)
        brush_row.addWidget(QLabel("ブラシ径:"))
        self.brush_slider = QSlider(Qt.Horizontal)
        self.brush_slider.setRange(6, 150)
        self.brush_slider.setValue(28)
        self.brush_slider.setToolTip("ブラシ選択・消しゴムの円サイズ ([ / ] キーまたは Ctrl+マウスホイールでも調整可能)")
        self.brush_slider.valueChanged.connect(self._on_brush_slider_changed)
        brush_row.addWidget(self.brush_slider, 1)
        self.lbl_brush_radius = QLabel("28 px")
        self.lbl_brush_radius.setStyleSheet("color: #38bdf8; font-size: 11px; min-width: 36px;")
        brush_row.addWidget(self.lbl_brush_radius)
        tools_layout.addLayout(brush_row)

        # 多角形確定ボタン & 選択反転 & 選択解除ボタン
        poly_action_layout = QHBoxLayout()
        poly_action_layout.setSpacing(4)
        self.btn_poly_confirm = QPushButton("多角形確定(Enter)")
        self.btn_poly_confirm.setStyleSheet("background-color: #0f766e; font-size: 11px; padding: 5px 4px;")
        self.btn_poly_confirm.clicked.connect(self._confirm_polygon_selection)
        self.btn_poly_confirm.setEnabled(False)
        poly_action_layout.addWidget(self.btn_poly_confirm, 1)

        self.btn_invert_sel = QPushButton("選択反転(I)")
        self.btn_invert_sel.setProperty("role", "outline")
        self.btn_invert_sel.setStyleSheet("font-size: 11px; padding: 5px 4px;")
        self.btn_invert_sel.clicked.connect(self._invert_selection)
        poly_action_layout.addWidget(self.btn_invert_sel, 1)

        self.clear_sel_btn = QPushButton("選択解除(Esc)")
        self.clear_sel_btn.setProperty("role", "outline")
        self.clear_sel_btn.setStyleSheet("font-size: 11px; padding: 5px 4px;")
        self.clear_sel_btn.clicked.connect(self._clear_selection)
        poly_action_layout.addWidget(self.clear_sel_btn, 1)
        tools_layout.addLayout(poly_action_layout)

        # ★ スマートノイズ処理コマンド (大まかな塊処理 & 細かい局所・ヒゲ処理)
        smart_cmd_row1 = QHBoxLayout()
        smart_cmd_row1.setSpacing(4)
        self.btn_magic_wand = QPushButton("🪄 連結塊選択(G)")
        self.btn_magic_wand.setToolTip(
            "現在選択中の点から3D空間で繋がっている同一オブジェクト(浮遊ゴミ・人・車・看板等)を地面を巻き込まずに一括選択します (ショートカット: G)"
        )
        self.btn_magic_wand.setStyleSheet("background-color: #4f46e5; font-size: 11px; font-weight: bold; padding: 4px 4px;")
        self.btn_magic_wand.clicked.connect(self._on_magic_wand_cluster_select)
        smart_cmd_row1.addWidget(self.btn_magic_wand, 1)

        self.btn_local_clean = QPushButton("📦 選択内のみ除去")
        self.btn_local_clean.setToolTip(
            "全体のディテールを守りながら、現在マウスで選択している範囲内だけに細かいSOR・エッジ飛散ノイズ除去を適用します"
        )
        self.btn_local_clean.setStyleSheet("background-color: #0369a1; font-size: 11px; font-weight: bold; padding: 4px 4px;")
        self.btn_local_clean.clicked.connect(self._on_local_fine_clean_selected)
        smart_cmd_row1.addWidget(self.btn_local_clean, 1)
        tools_layout.addLayout(smart_cmd_row1)

        smart_cmd_row2 = QHBoxLayout()
        smart_cmd_row2.setSpacing(4)
        self.btn_keep_main_cluster = QPushButton("🏝️ メイン塊のみ残す")
        self.btn_keep_main_cluster.setToolTip(
            "【大まかなノイズ一括処理】主要な構造物の大きな塊だけを残し、周囲に浮かぶ飛散ノイズや孤立塊を一括でノイズレイヤーへ除去します"
        )
        self.btn_keep_main_cluster.setStyleSheet("background-color: #0f766e; font-size: 11px; font-weight: bold; padding: 4px 4px;")
        self.btn_keep_main_cluster.clicked.connect(self._on_keep_main_cluster_only)
        smart_cmd_row2.addWidget(self.btn_keep_main_cluster, 1)

        self.btn_quick_edge_clean = QPushButton("🪶 エッジ・ヒゲ除去")
        self.btn_quick_edge_clean.setToolTip(
            "【細かいノイズ精密処理】構造物の輪郭エッジから飛び出す混合ピクセル(ベールノイズ)や表面の微細ヒゲをピンポイント除去します"
        )
        self.btn_quick_edge_clean.setStyleSheet("background-color: #b45309; font-size: 11px; font-weight: bold; padding: 4px 4px;")
        self.btn_quick_edge_clean.clicked.connect(self._on_quick_edge_veil_clean)
        smart_cmd_row2.addWidget(self.btn_quick_edge_clean, 1)
        tools_layout.addLayout(smart_cmd_row2)

        manual_action_layout = QHBoxLayout()
        manual_action_layout.setSpacing(4)
        self.delete_pts_btn = QPushButton("🗑 選択削除(Del)")
        self.delete_pts_btn.setProperty("role", "danger")
        self.delete_pts_btn.setStyleSheet("font-size: 11px; font-weight: bold; padding: 5px 6px;")
        self.delete_pts_btn.clicked.connect(self._delete_selected_points)
        manual_action_layout.addWidget(self.delete_pts_btn, 1)

        self.undo_btn = QPushButton("↩ 戻す(Ctrl+Z)")
        self.undo_btn.setProperty("role", "outline")
        self.undo_btn.setStyleSheet("font-size: 11px; padding: 5px 6px;")
        self.undo_btn.clicked.connect(self._undo_action)
        self.undo_btn.setEnabled(False)
        manual_action_layout.addWidget(self.undo_btn, 1)

        self.redo_btn = QPushButton("↪ やり直し(Ctrl+Y)")
        self.redo_btn.setProperty("role", "outline")
        self.redo_btn.setStyleSheet("font-size: 11px; padding: 5px 6px;")
        self.redo_btn.clicked.connect(self._redo_action)
        self.redo_btn.setEnabled(False)
        manual_action_layout.addWidget(self.redo_btn, 1)
        tools_layout.addLayout(manual_action_layout)

        # 計測ツールトグルボタン
        self.btn_measure = QPushButton("📐 2点間寸法計測ツール (M)")
        self.btn_measure.setCheckable(True)
        self.btn_measure.setProperty("role", "outline")
        self.btn_measure.setStyleSheet("font-size: 11px; font-weight: bold; padding: 4px 6px;")
        self.btn_measure.toggled.connect(self._toggle_measure_mode)
        tools_layout.addWidget(self.btn_measure)

        self.manual_status_label = QLabel("※[Alt+左ドラッグ]選択中も視点回転 / [Shift]追加 / [Ctrl]除外")
        self.manual_status_label.setWordWrap(True)
        self.manual_status_label.setStyleSheet("color: #888899; font-size: 11px;")
        tools_layout.addWidget(self.manual_status_label)

        sidebar_layout.addWidget(tools_group)

        # 6. 自動ノイズ除去フィルタ設定 (大まか ⇄ 細かい ワンクリックプリセット搭載)
        auto_group = CollapsibleGroupBox("🪄 ノイズフィルタ設定 (大まか ⇄ 細部)")
        self.auto_group = auto_group
        auto_layout = QVBoxLayout(auto_group.content_widget)
        auto_group.toggled_collapse.connect(self._on_panel_collapse_toggled)

        # ★ ワンクリック・プリセットバー (粗取り / 標準 / 精密仕上げ)
        preset_lbl = QLabel("🎚️ ワンクリック・プリセット (目的に応じて自動設定):")
        preset_lbl.setStyleSheet("color: #cbd5e1; font-size: 11px; font-weight: bold;")
        auto_layout.addWidget(preset_lbl)

        preset_row = QHBoxLayout()
        preset_row.setSpacing(4)
        self.btn_preset_rough = QPushButton("🟢 大まか(粗取り)")
        self.btn_preset_rough.setToolTip("浮遊する大きな塊ゴミや遠方の飛び点を大まかに一掃する設定をセットします")
        self.btn_preset_rough.setStyleSheet("background-color: #065f46; font-size: 11px; font-weight: bold; padding: 5px 4px;")
        self.btn_preset_rough.clicked.connect(lambda: self._apply_filter_preset("rough"))
        preset_row.addWidget(self.btn_preset_rough, 1)

        self.btn_preset_std = QPushButton("🟡 標準バランス")
        self.btn_preset_std.setToolTip("一般的な建築・土木スキャンに適したバランス型のSOR+クラスタ除去設定をセットします")
        self.btn_preset_std.setStyleSheet("background-color: #1e40af; font-size: 11px; font-weight: bold; padding: 5px 4px;")
        self.btn_preset_std.clicked.connect(lambda: self._apply_filter_preset("standard"))
        preset_row.addWidget(self.btn_preset_std, 1)

        self.btn_preset_fine = QPushButton("🟠 細かい(精密)")
        self.btn_preset_fine.setToolTip("構造物表面の微細ヒゲや輪郭エッジの飛散ノイズ(ベール)を緻密に除去する設定をセットします")
        self.btn_preset_fine.setStyleSheet("background-color: #9a3412; font-size: 11px; font-weight: bold; padding: 5px 4px;")
        self.btn_preset_fine.clicked.connect(lambda: self._apply_filter_preset("fine"))
        preset_row.addWidget(self.btn_preset_fine, 1)
        auto_layout.addLayout(preset_row)

        tab_widget = QTabWidget()
        tab_widget.setUsesScrollButtons(True)
        self.filter_tab_widget = tab_widget

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

        # クラスタ（浮遊小塊ノイズ除去）タブ
        cluster_tab = QWidget()
        cluster_l = QGridLayout(cluster_tab)
        self.cluster_chk = QCheckBox("孤立クラスタ(塊ノイズ)除去を有効化")
        cluster_l.addWidget(self.cluster_chk, 0, 0, 1, 2)
        cluster_l.addWidget(QLabel("連結距離 eps (m):"), 1, 0)
        self.cluster_eps_spin = QDoubleSpinBox()
        self.cluster_eps_spin.setRange(0.01, 20.0)
        self.cluster_eps_spin.setSingleStep(0.05)
        self.cluster_eps_spin.setValue(0.15)
        cluster_l.addWidget(self.cluster_eps_spin, 1, 1)
        cluster_l.addWidget(QLabel("最小塊点数 (点):"), 2, 0)
        self.cluster_pts_spin = QSpinBox()
        self.cluster_pts_spin.setRange(2, 50000)
        self.cluster_pts_spin.setValue(30)
        cluster_l.addWidget(self.cluster_pts_spin, 2, 1)
        tab_widget.addTab(cluster_tab, "クラスタ")

        # エッジ飛散・ヒゲ（ベールノイズ）精密除去タブ
        edge_tab = QWidget()
        edge_l = QGridLayout(edge_tab)
        self.edge_veil_chk = QCheckBox("エッジ飛散・ヒゲ(ベール)精密除去を有効化")
        edge_l.addWidget(self.edge_veil_chk, 0, 0, 1, 2)
        edge_l.addWidget(QLabel("局所近傍点数:"), 1, 0)
        self.edge_k_spin = QSpinBox()
        self.edge_k_spin.setRange(6, 60)
        self.edge_k_spin.setValue(16)
        edge_l.addWidget(self.edge_k_spin, 1, 1)
        edge_l.addWidget(QLabel("線状ヒゲ感度:"), 2, 0)
        self.edge_lin_spin = QDoubleSpinBox()
        self.edge_lin_spin.setRange(0.30, 0.98)
        self.edge_lin_spin.setSingleStep(0.05)
        self.edge_lin_spin.setValue(0.72)
        edge_l.addWidget(self.edge_lin_spin, 2, 1)
        tab_widget.addTab(edge_tab, "エッジ・ヒゲ")

        # 距離・反射強度タブ
        dist_int_tab = QWidget()
        di_l = QGridLayout(dist_int_tab)
        self.dist_chk = QCheckBox("距離フィルタ (原点からの距離 [m])")
        di_l.addWidget(self.dist_chk, 0, 0, 1, 2)
        self.dist_min_spin = QDoubleSpinBox()
        self.dist_min_spin.setRange(0.0, 10000.0)
        self.dist_min_spin.setValue(0.5)
        self.dist_min_spin.setPrefix("最小: ")
        self.dist_min_spin.setSuffix(" m")
        self.dist_min_spin.setMinimumWidth(60)
        di_l.addWidget(self.dist_min_spin, 1, 0)
        self.dist_max_spin = QDoubleSpinBox()
        self.dist_max_spin.setRange(0.1, 50000.0)
        self.dist_max_spin.setValue(150.0)
        self.dist_max_spin.setPrefix("最大: ")
        self.dist_max_spin.setSuffix(" m")
        self.dist_max_spin.setMinimumWidth(60)
        di_l.addWidget(self.dist_max_spin, 1, 1)

        self.int_chk = QCheckBox("反射強度フィルタ (Intensity)")
        di_l.addWidget(self.int_chk, 2, 0, 1, 2)
        self.int_min_spin = QDoubleSpinBox()
        self.int_min_spin.setDecimals(3)
        self.int_min_spin.setRange(-100000.0, 100000.0)
        self.int_min_spin.setSingleStep(0.01)
        self.int_min_spin.setValue(0.01)
        self.int_min_spin.setPrefix("最小: ")
        self.int_min_spin.setMinimumWidth(60)
        di_l.addWidget(self.int_min_spin, 3, 0)
        self.int_max_spin = QDoubleSpinBox()
        self.int_max_spin.setDecimals(3)
        self.int_max_spin.setRange(-100000.0, 100000.0)
        self.int_max_spin.setSingleStep(0.1)
        self.int_max_spin.setValue(65535.0)
        self.int_max_spin.setPrefix("最大: ")
        self.int_max_spin.setMinimumWidth(60)
        di_l.addWidget(self.int_max_spin, 3, 1)
        tab_widget.addTab(dist_int_tab, "距離・強度")

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

        # フィルタ単位 (1フィルタ = 1段階) で戻る/進むボタン
        auto_step_row = QHBoxLayout()
        auto_step_row.setSpacing(4)
        self.auto_undo_btn = QPushButton("◀ 1つ戻る")
        self.auto_undo_btn.setProperty("role", "outline")
        self.auto_undo_btn.setToolTip("直前の処理を1段階取り消す (自動除去はフィルタ別レイヤー単位) Ctrl+Z")
        self.auto_undo_btn.setEnabled(False)
        self.auto_undo_btn.clicked.connect(self._undo_action)
        auto_step_row.addWidget(self.auto_undo_btn, 1)
        self.auto_redo_btn = QPushButton("1つ進む ▶")
        self.auto_redo_btn.setProperty("role", "outline")
        self.auto_redo_btn.setToolTip("取り消した処理を1段階やり直す Ctrl+Y")
        self.auto_redo_btn.setEnabled(False)
        self.auto_redo_btn.clicked.connect(self._redo_action)
        auto_step_row.addWidget(self.auto_redo_btn, 1)
        auto_layout.addLayout(auto_step_row)

        sidebar_layout.addWidget(auto_group)

        # 7. ⚙ 設定パネル (表示・パフォーマンス・マウス操作・設定保存・ショートカット一覧)
        settings_group = self._build_settings_panel()
        self.settings_group = settings_group
        sidebar_layout.addWidget(settings_group)

        # 8. ステータス・ログ (アコーディオン折りたたみ対応)
        log_group = CollapsibleGroupBox("📋 処理状況・ログ")
        self.log_group = log_group
        log_layout = QVBoxLayout(log_group.content_widget)
        log_group.toggled_collapse.connect(self._on_panel_collapse_toggled)
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        log_layout.addWidget(self.progress_bar)

        status_row = QHBoxLayout()
        self.status_label = QLabel("E57ファイルを読み込んでください")
        self.status_label.setWordWrap(True)
        status_row.addWidget(self.status_label, 1)

        self.btn_toggle_log = QPushButton("▼ ログ")
        self.btn_toggle_log.setProperty("role", "outline")
        self.btn_toggle_log.setStyleSheet("padding: 2px 6px; font-size: 10px;")
        self.btn_toggle_log.setCheckable(True)
        self.btn_toggle_log.setChecked(True)
        self.btn_toggle_log.toggled.connect(lambda checked: self.log_text.setVisible(checked))
        status_row.addWidget(self.btn_toggle_log)
        log_layout.addLayout(status_row)

        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(70)
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        log_layout.setContentsMargins(6, 4, 6, 4)
        log_layout.setSpacing(3)
        sidebar_layout.addWidget(log_group)
        sidebar_layout.addStretch()

        self.sidebar_scroll.setWidget(sidebar_widget)
        self.splitter.addWidget(self.sidebar_scroll)

        # --- 右パネル (3D点群OpenGLビューア + セグメントバー + 選択アクションバー + トースト通知) ---
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        # ★ 柱②: 上部ナビゲーションバー (1クリック操作モード・セグメントコントロール搭載)
        nav_bar = QWidget()
        nav_bar.setFixedHeight(40)
        nav_bar.setStyleSheet("background-color: #17171c; border-bottom: 1px solid #333344;")
        nav_layout = QHBoxLayout(nav_bar)
        nav_layout.setContentsMargins(8, 3, 8, 3)
        nav_layout.setSpacing(4)

        self.btn_toggle_sidebar = QPushButton("◀ パネル格納")
        self.btn_toggle_sidebar.setProperty("role", "outline")
        self.btn_toggle_sidebar.setStyleSheet("padding: 3px 8px; font-size: 11px;")
        self.btn_toggle_sidebar.clicked.connect(self._toggle_sidebar)
        nav_layout.addWidget(self.btn_toggle_sidebar)

        # セグメントコントロール: [🖐回転(V)] [✋パン(H)] [🧹ブラシ(B)] [⬜矩形(R)] [⭕円(C)] [⬠多角(L)] [📐計測(M)] + [⚡即削除] [🎯専用(N)]
        self.seg_btn_orbit = QPushButton("🖐 回転(V)")
        self.seg_btn_orbit.setCheckable(True)
        self.seg_btn_orbit.setChecked(True)
        self.seg_btn_orbit.setProperty("role", "segment")
        self.seg_btn_orbit.setStyleSheet("padding: 3px 5px; font-size: 11px;")
        self.seg_btn_orbit.setToolTip("視点回転モード: 左ドラッグで回転、右ドラッグでパン (ショートカット: V / 再押下でパンと交互切替)")
        self.seg_btn_orbit.clicked.connect(lambda: self._set_interaction_mode("orbit"))
        nav_layout.addWidget(self.seg_btn_orbit)

        self.seg_btn_pan = QPushButton("✋ パン(H)")
        self.seg_btn_pan.setCheckable(True)
        self.seg_btn_pan.setProperty("role", "segment")
        self.seg_btn_pan.setStyleSheet("padding: 3px 5px; font-size: 11px;")
        self.seg_btn_pan.setToolTip("パン(平行移動)モード: 左ドラッグでそのまま画面を平行移動します (ショートカット: H または V / Space長押しでも一時パン可)")
        self.seg_btn_pan.clicked.connect(lambda: self._set_interaction_mode("pan"))
        nav_layout.addWidget(self.seg_btn_pan)

        self.seg_btn_brush = QPushButton("🧹 ブラシ(B)")
        self.seg_btn_brush.setCheckable(True)
        self.seg_btn_brush.setProperty("role", "segment")
        self.seg_btn_brush.setStyleSheet("padding: 3px 5px; font-size: 11px;")
        self.seg_btn_brush.setToolTip("円形ブラシで画面上をなぞって直感的に選択・消去します (ショートカット: B / [ ]でサイズ変更)")
        self.seg_btn_brush.clicked.connect(lambda: self._set_interaction_mode("brush"))
        nav_layout.addWidget(self.seg_btn_brush)

        self.seg_btn_rect = QPushButton("⬜ 矩形(R)")
        self.seg_btn_rect.setCheckable(True)
        self.seg_btn_rect.setProperty("role", "segment")
        self.seg_btn_rect.setStyleSheet("padding: 3px 5px; font-size: 11px;")
        self.seg_btn_rect.setToolTip("1クリックで矩形(四角)選択モードへ切替 (ショートカット: R)")
        self.seg_btn_rect.clicked.connect(lambda: self._set_interaction_mode("rect"))
        nav_layout.addWidget(self.seg_btn_rect)

        self.seg_btn_circle = QPushButton("⭕ 円(C)")
        self.seg_btn_circle.setCheckable(True)
        self.seg_btn_circle.setProperty("role", "segment")
        self.seg_btn_circle.setStyleSheet("padding: 3px 5px; font-size: 11px;")
        self.seg_btn_circle.setToolTip("1クリックで円形(サークル)選択モードへ切替 (ショートカット: C)")
        self.seg_btn_circle.clicked.connect(lambda: self._set_interaction_mode("circle"))
        nav_layout.addWidget(self.seg_btn_circle)

        self.seg_btn_poly = QPushButton("⬠ 多角(L)")
        self.seg_btn_poly.setCheckable(True)
        self.seg_btn_poly.setProperty("role", "segment")
        self.seg_btn_poly.setStyleSheet("padding: 3px 5px; font-size: 11px;")
        self.seg_btn_poly.setToolTip("1クリックで多角形(ポリゴン)選択モードへ切替 (ショートカット: L)")
        self.seg_btn_poly.clicked.connect(lambda: self._set_interaction_mode("polygon"))
        nav_layout.addWidget(self.seg_btn_poly)

        self.btn_instant_del = QPushButton("⚡ 即削除")
        self.btn_instant_del.setCheckable(True)
        self.btn_instant_del.setToolTip("ONにすると、囲んだ瞬間・ブラシでなぞった瞬間にDeleteキー不要で即ノイズ消去します")
        self.btn_instant_del.setStyleSheet("padding: 3px 6px; font-size: 11px; font-weight: bold;")
        self.btn_instant_del.toggled.connect(self._toggle_instant_delete_mode)
        nav_layout.addWidget(self.btn_instant_del)

        self.btn_clean_mode = QPushButton("🎯 専用(N)")
        self.btn_clean_mode.setCheckable(True)
        self.btn_clean_mode.setToolTip("ノイズ処理に特化した高コントラスト・ブラシ即消去ワークスペースへワンタッチ切替 (N)")
        self.btn_clean_mode.setStyleSheet(
            "background-color: #1e293b; border: 1px solid #38bdf8; color: #38bdf8; padding: 3px 6px; font-size: 11px; font-weight: bold;"
        )
        self.btn_clean_mode.toggled.connect(self._toggle_clean_focus_mode)
        nav_layout.addWidget(self.btn_clean_mode)

        self.nav_btn_measure = QPushButton("📐 計測(M)")
        self.nav_btn_measure.setCheckable(True)
        self.nav_btn_measure.setProperty("role", "segment")
        self.nav_btn_measure.setStyleSheet("padding: 3px 6px; font-size: 11px;")
        self.nav_btn_measure.setToolTip("2点間寸法計測モード (M)")
        self.nav_btn_measure.toggled.connect(self._toggle_measure_mode)
        nav_layout.addWidget(self.nav_btn_measure)

        self.nav_btn_wall_align = QPushButton("🧭 2点壁(U)")
        self.nav_btn_wall_align.setCheckable(True)
        self.nav_btn_wall_align.setProperty("role", "segment")
        self.nav_btn_wall_align.setStyleSheet("padding: 3px 6px; font-size: 11px;")
        self.nav_btn_wall_align.setToolTip("点群2点をピックして斜めの壁に直交する横・縦表示へ切り替えます (ショートカット: U)")
        self.nav_btn_wall_align.toggled.connect(self._toggle_wall_align_mode)
        nav_layout.addWidget(self.nav_btn_wall_align)

        nav_btn_fit = QPushButton("フィット(F)")
        nav_btn_fit.setProperty("role", "outline")
        nav_btn_fit.setStyleSheet("padding: 3px 6px; font-size: 11px;")
        nav_btn_fit.clicked.connect(lambda: self.viewer.fit_to_screen())
        nav_layout.addWidget(nav_btn_fit)

        self.nav_btn_proj = QPushButton("投影: 透視")
        self.nav_btn_proj.setProperty("role", "outline")
        self.nav_btn_proj.setStyleSheet("padding: 3px 6px; font-size: 11px;")
        self.nav_btn_proj.clicked.connect(lambda: self.viewer.toggle_projection())
        nav_layout.addWidget(self.nav_btn_proj)

        self.nav_btn_screenshot = QPushButton("📷 撮影")
        self.nav_btn_screenshot.setProperty("role", "outline")
        self.nav_btn_screenshot.setStyleSheet("padding: 3px 6px; font-size: 11px; font-weight: bold; color: #38bdf8;")
        self.nav_btn_screenshot.setToolTip("3D画面を高画質キャプチャ (クリップボードコピー ＆ 画像ファイル保存)")
        self.nav_btn_screenshot.clicked.connect(self._on_capture_screenshot)
        nav_layout.addWidget(self.nav_btn_screenshot)

        nav_layout.addStretch()

        self.point_count_label = QLabel("点数: 0 点")
        self.point_count_label.setStyleSheet(
            "color: #38bdf8; font-weight: bold; font-size: 12px; background-color: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 2px 10px;"
        )
        nav_layout.addWidget(self.point_count_label)

        right_layout.addWidget(nav_bar, 0)

        # 3D点群ビューア
        self.viewer = PointCloudViewer()
        self.viewer.status_changed.connect(self._on_viewer_status)
        self.viewer.point_selected.connect(self._on_points_selected)
        self.viewer.projection_changed.connect(self._on_projection_changed)
        self.viewer.bounds_ready.connect(self._on_bounds_ready)
        self.viewer.layers_updated.connect(self._on_layers_updated)
        self.viewer.layer_list_changed.connect(self._refresh_layer_and_scan_lists)
        self.viewer.measure_updated.connect(lambda msg: self.status_label.setText(msg))
        self.viewer.context_menu_requested.connect(self._show_viewer_context_menu)
        self.viewer.instant_erase_requested.connect(self._on_instant_erase_requested)
        self.viewer.brush_radius_changed.connect(self._on_viewer_brush_radius_changed)
        self.viewer.wall_aligned.connect(self._on_wall_aligned)
        self.viewer.section_guides_changed.connect(self._on_viewer_section_guides_changed)
        right_layout.addWidget(self.viewer, 1)

        # ★ 柱② & 柱④: 3Dビューア下部 ボトムアクションバー (選択時クイック操作 + トースト通知表示)
        self.bottom_overlay_bar = QWidget()
        self.bottom_overlay_bar.setFixedHeight(36)
        self.bottom_overlay_bar.setStyleSheet("background-color: #141419; border-top: 1px solid #2d2d3a;")
        bot_layout = QHBoxLayout(self.bottom_overlay_bar)
        bot_layout.setContentsMargins(10, 3, 10, 3)
        bot_layout.setSpacing(6)

        self.floating_sel_badge = QLabel("未選択 (ブラシ/矩形/円/多角形で点を囲むとここで即操作できます)")
        self.floating_sel_badge.setStyleSheet("color: #94a3b8; font-size: 11px; font-weight: bold;")
        bot_layout.addWidget(self.floating_sel_badge)

        self.floating_del_btn = QPushButton("🗑 ノイズ削除(Del)")
        self.floating_del_btn.setProperty("role", "danger")
        self.floating_del_btn.setStyleSheet("padding: 3px 8px; font-size: 11px;")
        self.floating_del_btn.clicked.connect(self._delete_selected_points)
        self.floating_del_btn.setEnabled(False)
        bot_layout.addWidget(self.floating_del_btn)

        self.floating_wand_btn = QPushButton("🪄 連結塊を選択(G)")
        self.floating_wand_btn.setToolTip("選択した点と3Dで繋がっている塊全体を一発選択します (G)")
        self.floating_wand_btn.setStyleSheet("background-color: #4f46e5; padding: 3px 8px; font-size: 11px; font-weight: bold;")
        self.floating_wand_btn.clicked.connect(self._on_magic_wand_cluster_select)
        self.floating_wand_btn.setEnabled(False)
        bot_layout.addWidget(self.floating_wand_btn)

        self.floating_local_clean_btn = QPushButton("📦 選択内のみ精密除去")
        self.floating_local_clean_btn.setToolTip("選択範囲内だけに細かいSOR・エッジ飛散除去を適用します")
        self.floating_local_clean_btn.setStyleSheet("background-color: #0369a1; padding: 3px 8px; font-size: 11px; font-weight: bold;")
        self.floating_local_clean_btn.clicked.connect(self._on_local_fine_clean_selected)
        self.floating_local_clean_btn.setEnabled(False)
        bot_layout.addWidget(self.floating_local_clean_btn)

        self.floating_move_btn = QPushButton("📁 レイヤー移動")
        self.floating_move_btn.setProperty("role", "primary")
        self.floating_move_btn.setStyleSheet("padding: 3px 8px; font-size: 11px;")
        self.floating_move_btn.clicked.connect(self._on_move_selected_to_target_layer)
        self.floating_move_btn.setEnabled(False)
        bot_layout.addWidget(self.floating_move_btn)

        self.floating_restore_btn = QPushButton("♻ 選択ノイズ復元")
        self.floating_restore_btn.setProperty("role", "outline")
        self.floating_restore_btn.setStyleSheet("padding: 3px 8px; font-size: 11px;")
        self.floating_restore_btn.clicked.connect(self._restore_selected_noise)
        self.floating_restore_btn.setEnabled(False)
        bot_layout.addWidget(self.floating_restore_btn)

        self.floating_invert_btn = QPushButton("🔄 反転(I)")
        self.floating_invert_btn.setProperty("role", "outline")
        self.floating_invert_btn.setStyleSheet("padding: 3px 8px; font-size: 11px;")
        self.floating_invert_btn.clicked.connect(self._invert_selection)
        bot_layout.addWidget(self.floating_invert_btn)

        self.floating_clear_btn = QPushButton("✖ 解除(Esc)")
        self.floating_clear_btn.setProperty("role", "outline")
        self.floating_clear_btn.setStyleSheet("padding: 3px 8px; font-size: 11px;")
        self.floating_clear_btn.clicked.connect(self._clear_selection)
        self.floating_clear_btn.setEnabled(False)
        bot_layout.addWidget(self.floating_clear_btn)

        bot_layout.addStretch()

        # トースト通知ピル (操作完了時にハイライト表示)
        self.toast_label = QLabel("")
        self.toast_label.setStyleSheet(
            "background-color: #065f46; color: #ecfdf5; border: 1px solid #10b981; border-radius: 10px; padding: 2px 10px; font-size: 11px; font-weight: bold;"
        )
        self.toast_label.setVisible(False)
        bot_layout.addWidget(self.toast_label)

        self._toast_timer = QTimer(self)
        self._toast_timer.setSingleShot(True)
        self._toast_timer.timeout.connect(lambda: self.toast_label.setVisible(False))

        right_layout.addWidget(self.bottom_overlay_bar, 0)

        self.splitter.addWidget(right_container)
        self.splitter.setSizes([380, 1020])
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.splitterMoved.connect(self._on_splitter_moved)

        self._refresh_layer_and_scan_lists()

    def _setup_shortcuts(self):
        """ショートカットキーの設定 (どのボタンにフォーカスがあっても確実に動作するよう一元登録)"""
        esc_shortcut = QShortcut(QKeySequence(Qt.Key_Escape), self)
        esc_shortcut.activated.connect(self._clear_selection)

        del_shortcut = QShortcut(QKeySequence(Qt.Key_Delete), self)
        del_shortcut.activated.connect(self._delete_selected_points)

        bs_shortcut = QShortcut(QKeySequence(Qt.Key_Backspace), self)
        bs_shortcut.activated.connect(self._delete_selected_points)

        ret_shortcut = QShortcut(QKeySequence(Qt.Key_Return), self)
        ret_shortcut.activated.connect(self._confirm_polygon_selection)

        enter_shortcut = QShortcut(QKeySequence(Qt.Key_Enter), self)
        enter_shortcut.activated.connect(self._confirm_polygon_selection)

        undo_shortcut = QShortcut(QKeySequence("Ctrl+Z"), self)
        undo_shortcut.activated.connect(self._undo_action)

        # Windows環境で QKeySequence.Redo と "Ctrl+Y" の二重登録による Ambiguous 競合を防ぐため明示指定
        redo_ctrl_y = QShortcut(QKeySequence("Ctrl+Y"), self)
        redo_ctrl_y.activated.connect(self._redo_action)

        redo_ctrl_shift_z = QShortcut(QKeySequence("Ctrl+Shift+Z"), self)
        redo_ctrl_shift_z.activated.connect(self._redo_action)

        inv_shortcut = QShortcut(QKeySequence("I"), self)
        inv_shortcut.activated.connect(self._invert_selection)

        fit_shortcut = QShortcut(QKeySequence("F"), self)
        fit_shortcut.activated.connect(lambda: self.viewer.fit_to_screen())

        proj_shortcut = QShortcut(QKeySequence("P"), self)
        proj_shortcut.activated.connect(lambda: self.viewer.toggle_projection())

        meas_shortcut = QShortcut(QKeySequence("M"), self)
        meas_shortcut.activated.connect(lambda: self._toggle_measure_mode(not self.viewer.measure_mode))

        brush_dec_shortcut = QShortcut(QKeySequence("["), self)
        brush_dec_shortcut.activated.connect(lambda: self.viewer.set_brush_radius(self.viewer.brush_radius - 4))

        brush_inc_shortcut = QShortcut(QKeySequence("]"), self)
        brush_inc_shortcut.activated.connect(lambda: self.viewer.set_brush_radius(self.viewer.brush_radius + 4))

        clean_mode_shortcut = QShortcut(QKeySequence("N"), self)
        clean_mode_shortcut.activated.connect(lambda: self._toggle_clean_focus_mode(not self.viewer.clean_focus_mode))

        view_orbit_pan_shortcut = QShortcut(QKeySequence("V"), self)
        view_orbit_pan_shortcut.activated.connect(self._on_shortcut_v_view_mode)

        pan_shortcut = QShortcut(QKeySequence("H"), self)
        pan_shortcut.activated.connect(lambda: self._set_interaction_mode("pan"))

        brush_shortcut = QShortcut(QKeySequence("B"), self)
        brush_shortcut.activated.connect(lambda: self._set_interaction_mode("brush"))

        rect_shortcut = QShortcut(QKeySequence("R"), self)
        rect_shortcut.activated.connect(lambda: self._set_interaction_mode("rect"))

        circle_shortcut = QShortcut(QKeySequence("C"), self)
        circle_shortcut.activated.connect(lambda: self._set_interaction_mode("circle"))

        poly_shortcut = QShortcut(QKeySequence("L"), self)
        poly_shortcut.activated.connect(lambda: self._set_interaction_mode("polygon"))

        wand_shortcut = QShortcut(QKeySequence("G"), self)
        wand_shortcut.activated.connect(self._on_magic_wand_cluster_select)

        wall_align_shortcut = QShortcut(QKeySequence("U"), self)
        wall_align_shortcut.activated.connect(lambda: self._toggle_wall_align_mode(not self.viewer.wall_align_mode))

    def _on_shortcut_v_view_mode(self):
        """Vキー: 選択/計測モード中なら即座に画面操作へ戻り、すでに回転モード中なら『🖐回転 ⇄ ✋パン』を交互に切替"""
        if not self.viewer.selection_mode and not self.viewer.measure_mode and not self.viewer.pan_mode:
            self._set_interaction_mode("pan")
        else:
            self._set_interaction_mode("orbit")

    def _capture_layer_snapshot(self) -> dict:
        """現在のレイヤー割り当てとレイヤー定義のスナップショットを取得"""
        self.viewer._ensure_layer_sync()
        return {
            "layer_ids": self.viewer.layer_ids.copy() if self.viewer.layer_ids is not None else None,
            "layer_mask": self.viewer.layer_mask.copy() if self.viewer.layer_mask is not None else None,
            "layers": {
                lid: LayerInfo(
                    layer_id=linfo.layer_id,
                    name=linfo.name,
                    visible=linfo.visible,
                    locked=linfo.locked,
                    color=linfo.color,
                    use_layer_color=linfo.use_layer_color,
                    is_noise=linfo.is_noise,
                )
                for lid, linfo in self.viewer.layers.items()
            },
        }

    def _restore_layer_snapshot(self, snapshot):
        """スナップショット（辞書または後方互換のbool配列）からレイヤー状態を復元"""
        if isinstance(snapshot, np.ndarray):
            self.viewer.layer_mask = snapshot.copy()
            self.viewer._ensure_layer_sync()
        elif isinstance(snapshot, dict):
            if snapshot.get("layers"):
                self.viewer.layers = {
                    lid: LayerInfo(
                        layer_id=linfo.layer_id,
                        name=linfo.name,
                        visible=linfo.visible,
                        locked=linfo.locked,
                        color=linfo.color,
                        use_layer_color=linfo.use_layer_color,
                        is_noise=getattr(linfo, "is_noise", lid == 0 or str(linfo.name).startswith("自動除去:")),
                    )
                    for lid, linfo in snapshot["layers"].items()
                }
            if snapshot.get("layer_ids") is not None:
                self.viewer.layer_ids = snapshot["layer_ids"].copy()
                self.viewer._sync_mask_from_ids()
            elif snapshot.get("layer_mask") is not None:
                self.viewer.layer_mask = snapshot["layer_mask"].copy()
                self.viewer._ensure_layer_sync()

        self.viewer._update_active_colors()
        self.viewer._update_display_arrays()
        self.viewer.update()
        self.viewer._emit_layer_counts()

    def _push_layer_undo_state(self):
        """レイヤー変更前の状態をUndoスタックに保存し、Redoスタックをクリア（大規模点群向け自動メモリ保護付き）"""
        if self.viewer.layer_mask is not None:
            self.layer_undo_stack.append(self._capture_layer_snapshot())
            n_pts = len(self.viewer.points_raw) if self.viewer.points_raw is not None else 0
            if n_pts > 10_000_000:
                max_steps = 5
            elif n_pts > 3_000_000:
                max_steps = 10
            else:
                max_steps = getattr(self, "undo_max_steps", 50)
            while len(self.layer_undo_stack) > max_steps:
                self.layer_undo_stack.pop(0)
            self.layer_redo_stack.clear()
            self._sync_undo_buttons()

    def _sync_undo_buttons(self):
        """戻す/やり直しボタン (手動編集パネル・自動フィルタパネル両方) の有効状態を履歴スタックと同期"""
        can_undo = len(self.layer_undo_stack) > 0
        can_redo = len(self.layer_redo_stack) > 0
        self.undo_btn.setEnabled(can_undo)
        self.redo_btn.setEnabled(can_redo)
        if hasattr(self, "auto_undo_btn"):
            self.auto_undo_btn.setEnabled(can_undo)
            self.auto_redo_btn.setEnabled(can_redo)

    def _on_layers_updated(self, normal_count: int, noise_count: int):
        """レイヤー点数更新通知を受信"""
        self.lbl_layer_normal_count.setText(f"{normal_count:,} 点")
        self.lbl_layer_noise_count.setText(f"{noise_count:,} 点")
        self.point_count_label.setText(f"有効: {normal_count:,} 点 | ノイズ: {noise_count:,} 点")
        self._refresh_layer_and_scan_lists()

    def _on_layer_visibility_changed(self):
        """有効レイヤー全体/ノイズレイヤーの表示切り替え"""
        show_normal = self.chk_layer_normal.isChecked()
        show_noise = self.chk_layer_noise.isChecked()
        self.viewer.set_layer_visibility(show_normal, show_noise)

    def _clear_layout(self, layout):
        """レイアウト内の子ウィジェットをすべて安全に削除"""
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
            elif item.layout() is not None:
                self._clear_layout(item.layout())

    def _refresh_layer_and_scan_lists(self):
        """マルチレイヤー一覧（表示切替・色変更・🔒ロック）およびスキャン一覧UIを最新状態に更新"""
        if not hasattr(self, "layer_list_layout") or not hasattr(self, "viewer"):
            return

        # 1. 移動先レイヤーコンボボックスの更新
        prev_target_id = self.target_layer_combo.currentData()
        self.target_layer_combo.blockSignals(True)
        self.target_layer_combo.clear()
        sorted_layer_ids = sorted(self.viewer.layers.keys(), key=lambda x: (self.viewer.is_noise_layer(x), x))
        for lid in sorted_layer_ids:
            linfo = self.viewer.layers[lid]
            lock_mark = "🔒" if linfo.locked else ""
            self.target_layer_combo.addItem(f"{lock_mark}[ID:{lid}] {linfo.name}", lid)
        if prev_target_id in self.viewer.layers:
            idx = self.target_layer_combo.findData(prev_target_id)
            if idx >= 0:
                self.target_layer_combo.setCurrentIndex(idx)
        self.target_layer_combo.blockSignals(False)

        # 2. マルチレイヤー各行の生成
        self._clear_layout(self.layer_list_layout)
        counts = self.viewer.get_layer_counts()
        for lid in sorted_layer_ids:
            linfo = self.viewer.layers[lid]
            cnt = counts.get(lid, 0)
            row_w = QWidget()
            row_l = QHBoxLayout(row_w)
            row_l.setContentsMargins(0, 1, 0, 1)
            row_l.setSpacing(4)

            chk = QCheckBox(linfo.name)
            chk.setMinimumWidth(60)
            chk.setChecked(linfo.visible)
            chk.toggled.connect(lambda checked, layer_id=lid: self.viewer.set_single_layer_visibility(layer_id, checked))
            row_l.addWidget(chk, 1)

            r_i = int(linfo.color[0] * 255)
            g_i = int(linfo.color[1] * 255)
            b_i = int(linfo.color[2] * 255)
            col_btn = QPushButton("")
            col_btn.setFixedSize(18, 18)
            col_btn.setToolTip("レイヤーの表示色を変更")
            col_btn.setStyleSheet(
                f"background-color: rgb({r_i},{g_i},{b_i}); border: 1px solid #cbd5e1; border-radius: 3px; padding: 0px;"
            )
            col_btn.clicked.connect(lambda _=False, layer_id=lid: self._on_pick_layer_color(layer_id))
            row_l.addWidget(col_btn)

            lock_btn = QPushButton("🔒" if linfo.locked else "🔓")
            lock_btn.setFixedSize(26, 20)
            lock_btn.setToolTip("レイヤーのロック(保護)切替: ロック中は選択・削除・自動フィルタから保護されます")
            lock_btn.setStyleSheet(
                "background-color: #b45309; font-size: 10px; padding: 0px;"
                if linfo.locked
                else "background-color: #334155; font-size: 10px; padding: 0px;"
            )
            lock_btn.clicked.connect(
                lambda _=False, layer_id=lid, cur_lock=linfo.locked: self._on_toggle_layer_lock(layer_id, not cur_lock)
            )
            row_l.addWidget(lock_btn)

            cnt_lbl = QLabel(f"{cnt:,}点")
            cnt_lbl.setStyleSheet(
                "color: #cbd5e1; font-size: 11px; min-width: 50px;"
                if not self.viewer.is_noise_layer(lid)
                else "color: #f87171; font-size: 11px; min-width: 50px;"
            )
            cnt_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            row_l.addWidget(cnt_lbl)

            self.layer_list_layout.addWidget(row_w)

        # 3. スキャン(器械点)別表示リストの更新
        self._clear_layout(self.scan_list_layout)
        scan_counts = self.viewer.get_scan_counts()
        if len(scan_counts) <= 1 and len(self.current_scans) <= 1:
            self.scan_group_label.setText(f"📡 スキャン (器械点): 全 {len(scan_counts) or 1} ステーション")
        else:
            self.scan_group_label.setText(f"📡 スキャン (器械点) 別の表示切替 ({len(scan_counts)} 箇所):")
        for sid, scnt in sorted(scan_counts.items()):
            scan_name = (
                self.current_scans[sid].name
                if 0 <= sid < len(self.current_scans) and self.current_scans[sid].name
                else f"Scan {sid}"
            )
            s_chk = QCheckBox(f"器械点 [{sid}] {scan_name} ({scnt:,} 点)")
            s_chk.setMinimumWidth(60)
            s_chk.setChecked(self.viewer.scan_visibility.get(sid, True))
            s_chk.setStyleSheet("font-size: 11px;")
            s_chk.toggled.connect(lambda checked, scan_id=sid: self.viewer.set_scan_visibility(scan_id, checked))
            self.scan_list_layout.addWidget(s_chk)

    def _on_add_custom_layer(self):
        """① 新しいカスタムレイヤーを作成（入力欄が空なら連番名）"""
        name = self.custom_layer_name_edit.text().strip()
        if not name:
            name = f"カスタムレイヤー {len(self.viewer.layers) - 1}"
        new_id = self.viewer.add_custom_layer(name)
        self.custom_layer_name_edit.clear()
        idx = self.target_layer_combo.findData(new_id)
        if idx >= 0:
            self.target_layer_combo.setCurrentIndex(idx)
        self.log_text.append(f"新規レイヤー作成: 「{self.viewer.layers[new_id].name}」 (ID:{new_id})")
        self.status_label.setText(f"レイヤー「{self.viewer.layers[new_id].name}」を追加しました")
        self.show_toast(f"✨ レイヤー「{self.viewer.layers[new_id].name}」を追加しました")

    def _on_move_selected_to_target_layer(self):
        """① 選択中の点をコンボボックスで指定したレイヤーへ移動"""
        if self.viewer.points_raw is None:
            return
        selected_idx = self.viewer.selected_indices
        if len(selected_idx) == 0:
            QMessageBox.information(self, "レイヤー移動", "移動したい点を先に3D画面上で選択してください。")
            return
        target_id = self.target_layer_combo.currentData()
        if target_id is None:
            return

        self._push_layer_undo_state()
        moved = self.viewer.move_indices_to_layer(selected_idx, int(target_id), respect_lock=True)
        self._clear_selection()
        lname = self.viewer.layers[int(target_id)].name
        self.log_text.append(f"レイヤー移動: {moved:,} 点を「{lname}」へ移動しました")
        self.status_label.setText(f"{moved:,} 点を「{lname}」レイヤーへ移動しました")
        self.show_toast(f"📁 {moved:,} 点を「{lname}」へ移動しました")

    def _on_rename_target_layer(self):
        """① 選択中の対象レイヤーの名前を変更"""
        target_id = self.target_layer_combo.currentData()
        if target_id is None or int(target_id) not in self.viewer.layers:
            return
        lid = int(target_id)
        cur_name = self.viewer.layers[lid].name
        typed_name = self.custom_layer_name_edit.text().strip()
        if not typed_name:
            typed_name, ok = QInputDialog.getText(self, "レイヤー名変更", "新しいレイヤー名を入力:", text=cur_name)
            if not ok or not typed_name.strip():
                return
        else:
            self.custom_layer_name_edit.clear()
        self.viewer.rename_layer(lid, typed_name.strip())
        self.status_label.setText(f"レイヤー名を「{typed_name.strip()}」に変更しました")
        self.show_toast(f"✏️ レイヤー名を「{typed_name.strip()}」に変更しました")

    def _on_pick_layer_color(self, layer_id: int):
        """① 指定レイヤーの表示色をカラーダイアログで変更"""
        if layer_id not in self.viewer.layers:
            return
        cur = self.viewer.layers[layer_id].color
        initial = QColor(int(cur[0] * 255), int(cur[1] * 255), int(cur[2] * 255))
        color = QColorDialog.getColor(initial, self, f"レイヤー「{self.viewer.layers[layer_id].name}」の色を選択")
        if color.isValid():
            self.viewer.set_layer_color(layer_id, (color.redF(), color.greenF(), color.blueF()), use_layer_color=True)
            self.status_label.setText(f"レイヤー「{self.viewer.layers[layer_id].name}」の色を更新しました")
            self.show_toast(f"🎨 レイヤー「{self.viewer.layers[layer_id].name}」の色を更新しました")

    def _on_toggle_layer_lock(self, layer_id: int, locked: bool):
        """② 指定レイヤーの🔒ロック(保護)状態を切り替え"""
        if layer_id not in self.viewer.layers:
            return
        self.viewer.set_layer_locked(layer_id, locked)
        lname = self.viewer.layers[layer_id].name
        state_str = "🔒ロック(保護 ON)" if locked else "🔓ロック解除"
        self.log_text.append(f"レイヤー保護: 「{lname}」を {state_str} に設定しました")
        self.status_label.setText(f"レイヤー「{lname}」: {state_str}")
        self.show_toast(f"{state_str}: 「{lname}」")

    def _on_auto_extract_ground(self):
        """⑥ 地面（グラウンド）の点を自動抽出して地面レイヤーへ振り分け"""
        if self.viewer.points_raw is None or len(self.viewer.points_raw) == 0:
            QMessageBox.information(self, "自動地面抽出", "先に点群データを読み込んでください。")
            return
        cand_idx = self.viewer.get_unlocked_normal_indices()
        if len(cand_idx) == 0:
            QMessageBox.information(self, "自動地面抽出", "抽出対象の未ロック有効点群がありません。")
            return

        ground_local_idx = segment_ground_points(self.viewer.points_raw[cand_idx], grid_size=1.0, height_threshold=0.25)
        if len(ground_local_idx) == 0:
            QMessageBox.information(self, "自動地面抽出", "地面と判定される点が見つかりませんでした。")
            return

        ground_orig_idx = cand_idx[ground_local_idx]
        self._push_layer_undo_state()
        lid = self.viewer.ensure_preset_layer(
            PointCloudViewer.LAYER_GROUND, "地面 (自動抽出)", (0.55, 0.40, 0.22)
        )
        moved = self.viewer.move_indices_to_layer(ground_orig_idx, lid, respect_lock=True)
        self.log_text.append(f"自動地面抽出: {moved:,} 点を「地面 (自動抽出)」レイヤーへ分離しました")
        self.status_label.setText(f"地面 {moved:,} 点を地面レイヤーへ自動振り分けしました (🔒ロックで保護可能)")
        self.show_toast(f"🏞️ 地面 {moved:,} 点を自動抽出しました")

    def _on_auto_extract_plane(self):
        """⑥ RANSACにより支配的な主平面（壁・床・天井）を自動検出して平面レイヤーへ振り分け"""
        if self.viewer.points_raw is None or len(self.viewer.points_raw) == 0:
            QMessageBox.information(self, "自動平面検出", "先に点群データを読み込んでください。")
            return
        cand_idx = self.viewer.get_unlocked_normal_indices()
        if len(cand_idx) < 3:
            QMessageBox.information(self, "自動平面検出", "検出対象の未ロック有効点群が不足しています。")
            return

        plane_local_idx, _ = segment_largest_plane(
            self.viewer.points_raw[cand_idx], distance_threshold=0.06, ransac_n=3, num_iterations=400
        )
        if len(plane_local_idx) == 0:
            QMessageBox.information(self, "自動平面検出", "平面が見つかりませんでした。")
            return

        plane_orig_idx = cand_idx[plane_local_idx]
        self._push_layer_undo_state()
        lid = self.viewer.ensure_preset_layer(
            PointCloudViewer.LAYER_PLANE, "主平面 (壁/床)", (0.20, 0.82, 0.55)
        )
        moved = self.viewer.move_indices_to_layer(plane_orig_idx, lid, respect_lock=True)
        self.log_text.append(f"自動平面検出: {moved:,} 点を「主平面 (壁/床)」レイヤーへ分離しました")
        self.status_label.setText(f"主平面 {moved:,} 点を平面レイヤーへ自動振り分けしました")
        self.show_toast(f"🧱 主平面 {moved:,} 点を自動検出しました")

    def _on_save_layer_project(self, file_path: Optional[str] = None):
        """⑤ 作業途中のレイヤー状態を .npz ファイルに保存"""
        if self.viewer.points_raw is None:
            QMessageBox.warning(self, "警告", "保存する点群データがありません。")
            return
        if not file_path:
            default_base = os.path.splitext(self.input_edit.text().strip() or "project")[0] + "_layers.npz"
            file_path, _ = QFileDialog.getSaveFileName(
                self, "レイヤー状態プロジェクトの保存", default_base, "レイヤー状態ファイル (*.npz)"
            )
            if not file_path:
                return
        try:
            self.viewer.save_layer_project(file_path)
            self.log_text.append(f"レイヤー状態保存完了: {file_path}")
            self.status_label.setText(f"レイヤー状態を保存しました: {os.path.basename(file_path)}")
            self.show_toast(f"💾 レイヤー状態を保存しました ({os.path.basename(file_path)})")
        except Exception as e:
            QMessageBox.critical(self, "保存エラー", f"レイヤー状態の保存に失敗しました:\n{e}")

    def _on_load_layer_project(self, file_path: Optional[str] = None):
        """⑤ 保存されたレイヤー状態 (.npz) を読み込んで作業を再開"""
        if self.viewer.points_raw is None:
            QMessageBox.warning(self, "警告", "先に元のE57点群ファイルを読み込んでください。")
            return
        if not file_path:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "レイヤー状態プロジェクトの読み込み", "", "レイヤー状態ファイル (*.npz)"
            )
            if not file_path:
                return
        try:
            self._push_layer_undo_state()
            self.viewer.load_layer_project(file_path)
            self.chk_layer_normal.blockSignals(True)
            self.chk_layer_noise.blockSignals(True)
            self.chk_layer_normal.setChecked(self.viewer.show_normal_layer)
            self.chk_layer_noise.setChecked(self.viewer.show_noise_layer)
            self.chk_layer_normal.blockSignals(False)
            self.chk_layer_noise.blockSignals(False)
            self.log_text.append(f"レイヤー状態読込完了: {file_path}")
            self.status_label.setText(f"レイヤー状態を復元しました: {os.path.basename(file_path)}")
            self.show_toast(f"📂 レイヤー状態を復元しました ({os.path.basename(file_path)})")
        except Exception as e:
            QMessageBox.critical(self, "読込エラー", f"レイヤー状態の読み込みに失敗しました:\n{e}")

    def _restore_selected_noise(self):
        """選択されている点のうちノイズレイヤーにある点のみを通常レイヤーにピンポイント復元"""
        if self.viewer.layer_mask is None:
            return
        selected_idx = self.viewer.selected_indices
        if len(selected_idx) == 0:
            QMessageBox.information(
                self,
                "部分ノイズ復元",
                "復元したいノイズ点を先に選択してください。\n"
                "（「ノイズレイヤー(赤色)」にチェックを入れて画面上で囲むと選択できます）",
            )
            return

        noise_in_sel = selected_idx[~self.viewer.layer_mask[selected_idx]]
        if len(noise_in_sel) == 0:
            QMessageBox.information(self, "情報", "選択範囲内にノイズレイヤーの点はありません。")
            return

        self._push_layer_undo_state()
        self.viewer.restore_indices_to_normal(noise_in_sel)
        self._clear_selection()
        self.log_text.append(f"部分ノイズ復元: 選択された {len(noise_in_sel):,} 点を通常レイヤーへ復元しました")
        self.status_label.setText(f"選択ノイズ {len(noise_in_sel):,} 点を通常レイヤーに復元しました")
        self.show_toast(f"♻ 選択ノイズ {len(noise_in_sel):,} 点を通常レイヤーへ復元しました")

    def _restore_all_noise(self):
        """ノイズレイヤーの点をすべて通常レイヤーに復元"""
        if self.viewer.layer_mask is None:
            return
        noise_idx = np.where(~self.viewer.layer_mask)[0]
        if len(noise_idx) == 0:
            QMessageBox.information(self, "情報", "ノイズレイヤーに点はありません。")
            return
        self._push_layer_undo_state()
        self.viewer.restore_indices_to_normal(noise_idx)
        self.log_text.append(f"全ノイズ復元: {len(noise_idx):,} 点を通常レイヤーへ復元しました")
        self.status_label.setText(f"{len(noise_idx):,} 点を通常レイヤーに復元しました")
        self.show_toast(f"♻ 全ノイズ {len(noise_idx):,} 点を通常レイヤーへ復元しました")

    def _build_settings_panel(self) -> QGroupBox:
        """⚙ 設定パネル: 表示・描画品質・マウス感度・設定保存・ショートカット一覧を集約"""
        settings_group = CollapsibleGroupBox("⚙️ 設定 & ガイド")
        set_layout = QVBoxLayout(settings_group.content_widget)
        settings_group.toggled_collapse.connect(self._on_panel_collapse_toggled)
        set_layout.setContentsMargins(6, 6, 6, 6)
        set_layout.setSpacing(6)

        tab_sub = QTabWidget()
        tab_sub.setStyleSheet("QTabBar::tab { padding: 4px 6px; font-size: 11px; }")

        # 1. 表示・品質タブ
        tab_disp = QWidget()
        l_disp = QGridLayout(tab_disp)
        l_disp.setContentsMargins(4, 6, 4, 6)
        l_disp.setSpacing(5)

        # 点サイズ
        l_disp.addWidget(QLabel("点サイズ:"), 0, 0)
        size_layout = QHBoxLayout()
        self.size_slider = QSlider(Qt.Horizontal)
        self.size_slider.setRange(1, 15)
        self.size_slider.setValue(2)
        self.size_slider.valueChanged.connect(lambda val: self.viewer.set_point_size(val))
        self.size_label = QLabel("2 px")
        self.size_label.setStyleSheet("color: #38bdf8; min-width: 32px;")
        self.size_slider.valueChanged.connect(lambda val: self.size_label.setText(f"{val} px"))
        size_layout.addWidget(self.size_slider)
        size_layout.addWidget(self.size_label)
        l_disp.addLayout(size_layout, 0, 1)

        # 点形状
        l_disp.addWidget(QLabel("点形状:"), 1, 0)
        self.shape_combo = QComboBox()
        self.shape_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.shape_combo.setMinimumContentsLength(8)
        self.shape_combo.addItem("丸 (スムーズ)", "circle")
        self.shape_combo.addItem("四角 (クッキリ)", "square")
        self.shape_combo.currentIndexChanged.connect(self._on_point_shape_changed)
        l_disp.addWidget(self.shape_combo, 1, 1)

        # 描画点数 (Point Budget)
        l_disp.addWidget(QLabel("描画点数:"), 2, 0)
        self.budget_combo = QComboBox()
        self.budget_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.budget_combo.setMinimumContentsLength(8)
        self.budget_combo.addItem("100万点 (超高速・推奨)", 1000000)
        self.budget_combo.addItem("200万点 (標準)", 2000000)
        self.budget_combo.addItem("500万点 (高精細)", 5000000)
        self.budget_combo.addItem("全点表示 (無制限)", 0)
        self.budget_combo.setCurrentIndex(1)
        self.budget_combo.currentIndexChanged.connect(self._on_budget_changed)
        l_disp.addWidget(self.budget_combo, 2, 1)

        # 背景色 & グリッド
        l_disp.addWidget(QLabel("環境設定:"), 3, 0)
        env_layout = QHBoxLayout()
        self.bg_combo = QComboBox()
        self.bg_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.bg_combo.setMinimumContentsLength(6)
        self.bg_combo.addItem("ダーク", QColor(20, 20, 26))
        self.bg_combo.addItem("ブラック", QColor(0, 0, 0))
        self.bg_combo.addItem("ホワイト", QColor(245, 245, 247))
        self.bg_combo.currentIndexChanged.connect(self._on_bg_color_changed)
        env_layout.addWidget(self.bg_combo)

        self.grid_chk = QCheckBox("グリッド")
        self.grid_chk.setChecked(True)
        self.grid_chk.toggled.connect(lambda checked: self.viewer.set_show_grid(checked))
        env_layout.addWidget(self.grid_chk)
        l_disp.addLayout(env_layout, 3, 1)

        # 初期配色モード (E57読込時に自動適用)
        l_disp.addWidget(QLabel("初期配色:"), 4, 0)
        self.settings_default_color_combo = QComboBox()
        self.settings_default_color_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.settings_default_color_combo.setMinimumContentsLength(8)
        self.settings_default_color_combo.addItems(PointCloudViewer.ALL_COLOR_MODES)
        self.settings_default_color_combo.currentTextChanged.connect(self._on_default_color_mode_changed)
        self.settings_default_color_combo.setToolTip("点群ファイルを読み込んだ瞬間に自動適用する配色モードです")
        l_disp.addWidget(self.settings_default_color_combo, 4, 1)

        # カメラ視野角 (FOV)
        l_disp.addWidget(QLabel("視野角(FOV):"), 5, 0)
        fov_row = QHBoxLayout()
        self.slider_fov = QSlider(Qt.Horizontal)
        self.slider_fov.setRange(20, 85)
        self.slider_fov.setValue(45)
        self.lbl_fov = QLabel("45°")
        self.lbl_fov.setStyleSheet("color: #38bdf8; min-width: 32px;")
        self.slider_fov.valueChanged.connect(self._on_fov_changed)
        self.slider_fov.setToolTip("カメラの視野角 (画角) を調整します (広角 60〜75°: 室内狭小用 ⇄ 狭角 30°: 建築立面図面用)")
        fov_row.addWidget(self.slider_fov)
        fov_row.addWidget(self.lbl_fov)
        l_disp.addLayout(fov_row, 5, 1)

        # 深度フォグ (立体感・奥行き強調)
        self.chk_depth_fog = QCheckBox("深度フォグ (奥行き・立体感強調)")
        self.chk_depth_fog.setChecked(False)
        self.chk_depth_fog.toggled.connect(lambda checked: self.viewer.set_depth_fog_enabled(checked))
        self.chk_depth_fog.setToolTip("視点からの距離に応じてわずかに深度陰影をつけ、単色や暗い点群でも凹凸や壁の立体感を強調します")
        l_disp.addWidget(self.chk_depth_fog, 6, 0, 1, 2)

        tab_sub.addTab(tab_disp, "🎨 表示")

        # 2. マウス操作・カメラ感度タブ
        tab_ctrl = QWidget()
        l_ctrl = QGridLayout(tab_ctrl)
        l_ctrl.setContentsMargins(4, 6, 4, 6)
        l_ctrl.setSpacing(5)

        # 回転感度
        l_ctrl.addWidget(QLabel("視点回転速度:"), 0, 0)
        r_row = QHBoxLayout()
        self.slider_rot_speed = QSlider(Qt.Horizontal)
        self.slider_rot_speed.setRange(2, 20)  # 0.2x 〜 2.0x
        self.slider_rot_speed.setValue(10)     # 1.0x
        self.lbl_rot_speed = QLabel("1.0x")
        self.lbl_rot_speed.setStyleSheet("color: #38bdf8; min-width: 32px;")
        self.slider_rot_speed.valueChanged.connect(self._on_rot_speed_changed)
        r_row.addWidget(self.slider_rot_speed)
        r_row.addWidget(self.lbl_rot_speed)
        l_ctrl.addLayout(r_row, 0, 1)

        # パン移動感度
        l_ctrl.addWidget(QLabel("パン移動感度:"), 1, 0)
        p_row = QHBoxLayout()
        self.slider_pan_speed = QSlider(Qt.Horizontal)
        self.slider_pan_speed.setRange(2, 25)
        self.slider_pan_speed.setValue(10)
        self.lbl_pan_speed = QLabel("1.0x")
        self.lbl_pan_speed.setStyleSheet("color: #38bdf8; min-width: 32px;")
        self.slider_pan_speed.valueChanged.connect(self._on_pan_speed_changed)
        p_row.addWidget(self.slider_pan_speed)
        p_row.addWidget(self.lbl_pan_speed)
        l_ctrl.addLayout(p_row, 1, 1)

        # ズーム感度
        l_ctrl.addWidget(QLabel("ズーム速度:"), 2, 0)
        z_row = QHBoxLayout()
        self.slider_zoom_speed = QSlider(Qt.Horizontal)
        self.slider_zoom_speed.setRange(2, 25)
        self.slider_zoom_speed.setValue(10)
        self.lbl_zoom_speed = QLabel("1.0x")
        self.lbl_zoom_speed.setStyleSheet("color: #38bdf8; min-width: 32px;")
        self.slider_zoom_speed.valueChanged.connect(self._on_zoom_speed_changed)
        z_row.addWidget(self.slider_zoom_speed)
        z_row.addWidget(self.lbl_zoom_speed)
        l_ctrl.addLayout(z_row, 2, 1)

        # WASD移動速度
        l_ctrl.addWidget(QLabel("WASD移動:"), 3, 0)
        w_row = QHBoxLayout()
        self.slider_wasd_speed = QSlider(Qt.Horizontal)
        self.slider_wasd_speed.setRange(2, 25)
        self.slider_wasd_speed.setValue(10)
        self.lbl_wasd_speed = QLabel("1.0x")
        self.lbl_wasd_speed.setStyleSheet("color: #38bdf8; min-width: 32px;")
        self.slider_wasd_speed.valueChanged.connect(self._on_wasd_speed_changed)
        w_row.addWidget(self.slider_wasd_speed)
        w_row.addWidget(self.lbl_wasd_speed)
        l_ctrl.addLayout(w_row, 3, 1)

        # 断面シーク刻み幅
        l_ctrl.addWidget(QLabel("断面シーク刻み:"), 4, 0)
        self.combo_sec_step = QComboBox()
        self.combo_sec_step.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.combo_sec_step.setMinimumContentsLength(8)
        self.combo_sec_step.addItem("1mm (0.001m: 極精密)", 0.001)
        self.combo_sec_step.addItem("5mm (0.005m: 精密)", 0.005)
        self.combo_sec_step.addItem("1cm (0.01m: 標準)", 0.01)
        self.combo_sec_step.addItem("5cm (0.05m: 建築)", 0.05)
        self.combo_sec_step.addItem("10cm (0.1m: 造成・広域)", 0.1)
        self.combo_sec_step.setCurrentIndex(2)  # 初期値: 1cm
        self.combo_sec_step.currentIndexChanged.connect(self._on_sec_step_changed)
        self.combo_sec_step.setToolTip("XYZ断面スライダーおよび数値入力のシークステップ刻み幅を設定します")
        l_ctrl.addWidget(self.combo_sec_step, 4, 1)

        # 縦回転反転
        self.chk_invert_orbit = QCheckBox("縦回転(ピッチ)の向きを反転")
        self.chk_invert_orbit.toggled.connect(lambda c: setattr(self.viewer, "invert_orbit_y", c))
        l_ctrl.addWidget(self.chk_invert_orbit, 5, 0, 1, 2)

        tab_sub.addTab(tab_ctrl, "🖐 操作感度")

        # 3. システム・メモリ管理タブ
        tab_sys = QWidget()
        l_sys = QGridLayout(tab_sys)
        l_sys.setContentsMargins(4, 6, 4, 6)
        l_sys.setSpacing(5)

        l_sys.addWidget(QLabel("大容量点群読込上限:"), 0, 0)
        self.combo_load_budget = QComboBox()
        self.combo_load_budget.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.combo_load_budget.setMinimumContentsLength(8)
        self.combo_load_budget.addItem("自動最適化 (PCメモリ推奨)", -1)
        self.combo_load_budget.addItem("3,000万点 (軽量・安定優先)", 30000000)
        self.combo_load_budget.addItem("6,000万点 (標準・推奨)", 60000000)
        self.combo_load_budget.addItem("1億点 (大容量・超高精細)", 100000000)
        self.combo_load_budget.addItem("無制限 (全点ロード)", 0)
        self.combo_load_budget.setCurrentIndex(0)
        self.combo_load_budget.setToolTip("5〜10GB等の巨大点群を開く際、PCのRAM枯渇やフリーズを防ぐ安全上限を設定します")
        l_sys.addWidget(self.combo_load_budget, 0, 1)

        l_sys.addWidget(QLabel("Undo最大保持数:"), 1, 0)
        self.combo_undo_limit = QComboBox()
        self.combo_undo_limit.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.combo_undo_limit.setMinimumContentsLength(8)
        self.combo_undo_limit.addItem("20回 (省メモリ推奨)", 20)
        self.combo_undo_limit.addItem("50回 (標準)", 50)
        self.combo_undo_limit.addItem("100回 (大容量PC向け)", 100)
        self.combo_undo_limit.addItem("無制限 (メモリ重視)", 0)
        self.combo_undo_limit.setCurrentIndex(1)  # 50回
        self.combo_undo_limit.currentIndexChanged.connect(self._on_undo_limit_changed)
        self.combo_undo_limit.setToolTip("手動消去やレイヤー移動の「元に戻す (Ctrl+Z)」の最大履歴保持数を設定します (メモリ枯渇防止)")
        l_sys.addWidget(self.combo_undo_limit, 1, 1)

        self.chk_auto_save_settings = QCheckBox("終了時に設定を自動保存する")
        self.chk_auto_save_settings.setChecked(True)
        self.chk_auto_save_settings.setToolTip("アプリケーション終了時に現在の表示・感度設定をPCに自動保存します")
        l_sys.addWidget(self.chk_auto_save_settings, 2, 0, 1, 2)

        sys_hint = QLabel("※5〜10GBの中規模点群でもメモリ枯渇を起こさず、快適なヌルヌル動作を維持します。")
        sys_hint.setWordWrap(True)
        sys_hint.setStyleSheet("color: #94a3b8; font-size: 10px; font-weight: normal;")
        l_sys.addWidget(sys_hint, 3, 0, 1, 2)

        tab_sub.addTab(tab_sys, "⚙️ システム")

        # 3. ショートカット一覧タブ
        tab_keys = QWidget()
        l_keys = QVBoxLayout(tab_keys)
        l_keys.setContentsMargins(4, 6, 4, 6)
        l_keys.setSpacing(4)
        sc_text = QLabel(
            "<b>【主要ショートカット一覧】</b><br>"
            "・<b>V</b>: 視点回転 ⇄ パン切り替え<br>"
            "・<b>H</b>: パン平行移動モード<br>"
            "・<b>B</b>: ブラシなぞり選択/消去<br>"
            "・<b>R</b>: 矩形ドラッグ選択<br>"
            "・<b>C</b>: 円形ドラッグ選択<br>"
            "・<b>L</b>: 多角形クリック選択 (Enter確定)<br>"
            "・<b>G</b>: 3D連結塊一括選択 (マジックワンド)<br>"
            "・<b>M</b>: 2点間寸法計測<br>"
            "・<b>U</b>: 2点ピック斜め壁アライン<br>"
            "・<b>N</b>: ノイズ処理専用モード<br>"
            "・<b>Del / Backspace</b>: 選択点をノイズへ消去<br>"
            "・<b>Ctrl+Z / Ctrl+Y</b>: 元に戻す / やり直し<br>"
            "・<b>I</b>: 選択反転 | <b>Esc</b>: 選択解除<br>"
            "・<b>F</b>: 全体フィット | <b>P</b>: 透視 ⇄ 正射影<br>"
            "・<b>[ / ]</b>: ブラシ径縮小 / 拡大<br>"
            "・<b>Space (長押し)</b>: 一時パン移動<br>"
            "・<b>Alt + 左ドラッグ</b>: 選択モード中でも視点回転<br>"
            "・<b>W / A / S / D / Q / E</b>: 3D画面フライスルー移動"
        )
        sc_text.setWordWrap(True)
        sc_text.setStyleSheet("font-size: 11px; line-height: 1.35; color: #cbd5e1;")
        l_keys.addWidget(sc_text)
        tab_sub.addTab(tab_keys, "⌨️ キー一覧")

        set_layout.addWidget(tab_sub)

        # 設定保存・初期化ボタン行
        btn_set_row = QHBoxLayout()
        btn_set_row.setSpacing(4)
        self.btn_save_settings = QPushButton("💾 設定を保存")
        self.btn_save_settings.setProperty("role", "primary")
        self.btn_save_settings.setStyleSheet("font-size: 11px; padding: 4px 6px;")
        self.btn_save_settings.setToolTip("現在の表示・感度設定をPCに保存し、次回起動時も自動適用します")
        self.btn_save_settings.clicked.connect(self._save_app_settings)
        btn_set_row.addWidget(self.btn_save_settings, 1)

        self.btn_reset_settings = QPushButton("🔄 初期設定")
        self.btn_reset_settings.setProperty("role", "outline")
        self.btn_reset_settings.setStyleSheet("font-size: 11px; padding: 4px 6px;")
        self.btn_reset_settings.setToolTip("各種設定を初期値にリセットします")
        self.btn_reset_settings.clicked.connect(self._reset_app_settings)
        btn_set_row.addWidget(self.btn_reset_settings, 1)
        set_layout.addLayout(btn_set_row)

        return settings_group

    def _on_rot_speed_changed(self, val: int):
        speed = val / 10.0
        self.lbl_rot_speed.setText(f"{speed:.1f}x")
        self.viewer.rotate_speed = 0.5 * speed

    def _on_pan_speed_changed(self, val: int):
        speed = val / 10.0
        self.lbl_pan_speed.setText(f"{speed:.1f}x")
        self.viewer.pan_speed_factor = speed

    def _on_zoom_speed_changed(self, val: int):
        speed = val / 10.0
        self.lbl_zoom_speed.setText(f"{speed:.1f}x")
        self.viewer.zoom_speed_factor = speed

    def _on_wasd_speed_changed(self, val: int):
        speed = val / 10.0
        self.lbl_wasd_speed.setText(f"{speed:.1f}x")
        self.viewer.wasd_speed_factor = speed

    def _on_default_color_mode_changed(self, mode_name: str):
        if hasattr(self.viewer, "default_color_mode"):
            self.viewer.default_color_mode = mode_name

    def _on_fov_changed(self, val: int):
        self.lbl_fov.setText(f"{val}°")
        self.viewer.set_fov(float(val))

    def _on_sec_step_changed(self, index: int):
        step_val = float(self.combo_sec_step.currentData() or 0.01)
        self._sec_pos_scale = 1.0 / step_val
        for sp in (
            getattr(self, "spin_pos_x", None),
            getattr(self, "spin_pos_y", None),
            getattr(self, "spin_pos_z", None),
            getattr(self, "spin_pos_wall", None),
        ):
            if sp:
                sp.setSingleStep(step_val)

    def _on_undo_limit_changed(self, index: int):
        self.undo_max_steps = int(self.combo_undo_limit.currentData() or 50)

    def _save_app_settings(self, silent: bool = False):
        """設定を QSettings に保存"""
        s = QSettings("E57PointCloudTool", "Settings")
        s.setValue("point_size", self.size_slider.value())
        s.setValue("point_shape", self.shape_combo.currentData())
        s.setValue("budget_index", self.budget_combo.currentIndex())
        s.setValue("bg_index", self.bg_combo.currentIndex())
        s.setValue("grid", self.grid_chk.isChecked())
        if hasattr(self, "settings_default_color_combo"):
            s.setValue("default_color_mode", self.settings_default_color_combo.currentText())
        if hasattr(self, "slider_fov"):
            s.setValue("fov", self.slider_fov.value())
        if hasattr(self, "chk_depth_fog"):
            s.setValue("depth_fog", self.chk_depth_fog.isChecked())
        s.setValue("rot_speed", self.slider_rot_speed.value())
        s.setValue("pan_speed", self.slider_pan_speed.value())
        s.setValue("zoom_speed", self.slider_zoom_speed.value())
        if hasattr(self, "slider_wasd_speed"):
            s.setValue("wasd_speed", self.slider_wasd_speed.value())
        if hasattr(self, "combo_sec_step"):
            s.setValue("sec_step_idx", self.combo_sec_step.currentIndex())
        if hasattr(self, "combo_load_budget"):
            s.setValue("load_budget_idx", self.combo_load_budget.currentIndex())
        if hasattr(self, "combo_undo_limit"):
            s.setValue("undo_limit_idx", self.combo_undo_limit.currentIndex())
        if hasattr(self, "chk_auto_save_settings"):
            s.setValue("auto_save", self.chk_auto_save_settings.isChecked())
        s.setValue("invert_orbit", self.chk_invert_orbit.isChecked())
        if not silent:
            self.show_toast("💾 設定を保存しました (次回起動時も適用)")

    def _load_app_settings(self):
        """保存された設定を QSettings から読み込み、UIおよびViewerへ完全反映"""
        s = QSettings("E57PointCloudTool", "Settings")
        if not s.contains("point_size"):
            return
        try:
            pt_size = s.value("point_size", 2, type=int)
            self.size_slider.setValue(pt_size)
            self.viewer.set_point_size(pt_size)
            self.size_label.setText(f"{pt_size} px")

            shape = s.value("point_shape", "circle", type=str)
            idx = self.shape_combo.findData(shape)
            if idx >= 0:
                self.shape_combo.setCurrentIndex(idx)
            self._on_point_shape_changed(self.shape_combo.currentIndex())

            b_idx = s.value("budget_index", 1, type=int)
            if 0 <= b_idx < self.budget_combo.count():
                self.budget_combo.setCurrentIndex(b_idx)
            self._on_budget_changed(self.budget_combo.currentIndex())

            bg_idx = s.value("bg_index", 0, type=int)
            if 0 <= bg_idx < self.bg_combo.count():
                self.bg_combo.setCurrentIndex(bg_idx)
            self._on_bg_color_changed(self.bg_combo.currentIndex())

            grid_on = s.value("grid", True, type=bool)
            self.grid_chk.setChecked(grid_on)
            self.viewer.set_show_grid(grid_on)

            def_col = s.value("default_color_mode", PointCloudViewer.COLOR_MODE_RGB, type=str)
            if hasattr(self, "settings_default_color_combo"):
                c_idx = self.settings_default_color_combo.findText(def_col)
                if c_idx >= 0:
                    self.settings_default_color_combo.setCurrentIndex(c_idx)
            self.viewer.default_color_mode = def_col

            fov_val = s.value("fov", 45, type=int)
            if hasattr(self, "slider_fov"):
                self.slider_fov.setValue(fov_val)
                self._on_fov_changed(fov_val)

            fog_on = s.value("depth_fog", False, type=bool)
            if hasattr(self, "chk_depth_fog"):
                self.chk_depth_fog.setChecked(fog_on)
                self.viewer.set_depth_fog_enabled(fog_on)

            rot_v = s.value("rot_speed", 10, type=int)
            self.slider_rot_speed.setValue(rot_v)
            self._on_rot_speed_changed(rot_v)

            pan_v = s.value("pan_speed", 10, type=int)
            self.slider_pan_speed.setValue(pan_v)
            self._on_pan_speed_changed(pan_v)

            zoom_v = s.value("zoom_speed", 10, type=int)
            self.slider_zoom_speed.setValue(zoom_v)
            self._on_zoom_speed_changed(zoom_v)

            if hasattr(self, "slider_wasd_speed"):
                wasd_v = s.value("wasd_speed", 10, type=int)
                self.slider_wasd_speed.setValue(wasd_v)
                self._on_wasd_speed_changed(wasd_v)

            sec_idx = s.value("sec_step_idx", 2, type=int)
            if hasattr(self, "combo_sec_step") and 0 <= sec_idx < self.combo_sec_step.count():
                self.combo_sec_step.setCurrentIndex(sec_idx)
                self._on_sec_step_changed(sec_idx)

            undo_idx = s.value("undo_limit_idx", 1, type=int)
            if hasattr(self, "combo_undo_limit") and 0 <= undo_idx < self.combo_undo_limit.count():
                self.combo_undo_limit.setCurrentIndex(undo_idx)
                self._on_undo_limit_changed(undo_idx)

            load_b_idx = s.value("load_budget_idx", 0, type=int)
            if hasattr(self, "combo_load_budget") and 0 <= load_b_idx < self.combo_load_budget.count():
                self.combo_load_budget.setCurrentIndex(load_b_idx)

            if hasattr(self, "chk_auto_save_settings"):
                self.chk_auto_save_settings.setChecked(s.value("auto_save", True, type=bool))

            inv_v = s.value("invert_orbit", False, type=bool)
            self.chk_invert_orbit.setChecked(inv_v)
            self.viewer.invert_orbit_y = inv_v
        except Exception:
            pass

    def _reset_app_settings(self):
        """設定を初期値に戻す"""
        self.size_slider.setValue(2)
        self.size_label.setText("2 px")
        self.viewer.set_point_size(2)

        self.shape_combo.setCurrentIndex(0)
        self._on_point_shape_changed(0)

        self.budget_combo.setCurrentIndex(1)
        self._on_budget_changed(1)

        self.bg_combo.setCurrentIndex(0)
        self._on_bg_color_changed(0)

        self.grid_chk.setChecked(True)
        self.viewer.set_show_grid(True)

        if hasattr(self, "settings_default_color_combo"):
            self.settings_default_color_combo.setCurrentIndex(0)
        self.viewer.default_color_mode = PointCloudViewer.COLOR_MODE_RGB

        if hasattr(self, "slider_fov"):
            self.slider_fov.setValue(45)
            self._on_fov_changed(45)

        if hasattr(self, "chk_depth_fog"):
            self.chk_depth_fog.setChecked(False)
            self.viewer.set_depth_fog_enabled(False)

        self.slider_rot_speed.setValue(10)
        self.lbl_rot_speed.setText("1.0x")
        self.viewer.rotate_speed = 0.5

        self.slider_pan_speed.setValue(10)
        self.lbl_pan_speed.setText("1.0x")
        self.viewer.pan_speed_factor = 1.0

        self.slider_zoom_speed.setValue(10)
        self.lbl_zoom_speed.setText("1.0x")
        self.viewer.zoom_speed_factor = 1.0

        if hasattr(self, "slider_wasd_speed"):
            self.slider_wasd_speed.setValue(10)
            self.lbl_wasd_speed.setText("1.0x")
            self.viewer.wasd_speed_factor = 1.0

        if hasattr(self, "combo_sec_step"):
            self.combo_sec_step.setCurrentIndex(2)
            self._on_sec_step_changed(2)

        if hasattr(self, "combo_load_budget"):
            self.combo_load_budget.setCurrentIndex(0)

        if hasattr(self, "combo_undo_limit"):
            self.combo_undo_limit.setCurrentIndex(1)
            self._on_undo_limit_changed(1)

        if hasattr(self, "chk_auto_save_settings"):
            self.chk_auto_save_settings.setChecked(True)

        self.chk_invert_orbit.setChecked(False)
        self.viewer.invert_orbit_y = False

        self.viewer.update()
        self.show_toast("🔄 設定を初期値に戻しました")

    def closeEvent(self, event):
        """アプリケーション終了時の処理 (設定の自動保存・ワーカー安全終了)"""
        try:
            if getattr(self, "chk_auto_save_settings", None) is None or self.chk_auto_save_settings.isChecked():
                self._save_app_settings(silent=True)
        except Exception:
            pass
        for w in (self.load_worker, self.worker, self.save_worker, self.memory_filter_worker):
            if w and w.isRunning():
                w.terminate()
                w.wait()
        super().closeEvent(event)

    def show_toast(self, message: str, duration_ms: int = 2600):
        """★ 柱④: 3Dビューア下部オーバーレイバーにピル型トースト通知を表示"""
        if hasattr(self, "toast_label") and hasattr(self, "_toast_timer"):
            self.toast_label.setText(message)
            self.toast_label.setVisible(True)
            self._toast_timer.start(duration_ms)

    def _set_sidebar_workflow_tab(self, tab_key: str):
        """★ 柱①: 左サイドバーのワークフロー切替 (全て / 編集 / 断面 / レイヤー / 自動除去 / 表示 / 設定 / ノイズ専用)"""
        alias_map = {"select": "edit", "auto": "filter"}
        norm_key = alias_map.get(tab_key, tab_key)
        for k, btn in self.sidebar_tab_btns.items():
            btn.blockSignals(True)
            btn.setChecked(k == norm_key)
            btn.blockSignals(False)

        has_set = hasattr(self, "settings_group")
        if norm_key == "clean":
            self.layer_group.setVisible(False)
            self.tools_group.setVisible(True)
            self.section_group.setVisible(True)
            self.auto_group.setVisible(True)
            self.view_group.setVisible(False)
            if has_set:
                self.settings_group.setVisible(False)
        elif norm_key == "all":
            self.layer_group.setVisible(True)
            self.tools_group.setVisible(True)
            self.section_group.setVisible(True)
            self.auto_group.setVisible(True)
            self.view_group.setVisible(True)
            if has_set:
                self.settings_group.setVisible(True)
        elif norm_key == "edit":
            self.layer_group.setVisible(False)
            self.tools_group.setVisible(True)
            self.section_group.setVisible(False)
            self.auto_group.setVisible(False)
            self.view_group.setVisible(False)
            if has_set:
                self.settings_group.setVisible(False)
        elif norm_key == "section":
            self.layer_group.setVisible(False)
            self.tools_group.setVisible(False)
            self.section_group.setVisible(True)
            self.auto_group.setVisible(False)
            self.view_group.setVisible(False)
            if has_set:
                self.settings_group.setVisible(False)
        elif norm_key == "layer":
            self.layer_group.setVisible(True)
            self.tools_group.setVisible(False)
            self.section_group.setVisible(False)
            self.auto_group.setVisible(False)
            self.view_group.setVisible(False)
            if has_set:
                self.settings_group.setVisible(False)
        elif norm_key == "filter":
            self.layer_group.setVisible(False)
            self.tools_group.setVisible(False)
            self.section_group.setVisible(False)
            self.auto_group.setVisible(True)
            self.view_group.setVisible(False)
            if has_set:
                self.settings_group.setVisible(False)
        elif norm_key == "view":
            self.layer_group.setVisible(False)
            self.tools_group.setVisible(False)
            self.section_group.setVisible(False)
            self.auto_group.setVisible(False)
            self.view_group.setVisible(True)
            if has_set:
                self.settings_group.setVisible(False)
        elif norm_key == "settings":
            self.layer_group.setVisible(False)
            self.tools_group.setVisible(False)
            self.section_group.setVisible(False)
            self.auto_group.setVisible(False)
            self.view_group.setVisible(False)
            if has_set:
                self.settings_group.setVisible(True)
        self._lock_sidebar_width()

    def _set_interaction_mode(self, mode: str):
        """★ 柱②: 1クリックで 視点回転(V) / パン(H) / ブラシ(B) / 矩形(R) / 円形(C) / 多角形(L) / 計測(M) を切り替え"""
        if getattr(self.viewer, "wall_align_mode", False):
            self._toggle_wall_align_mode(False)
        self.viewer.polygon_points.clear()
        self.viewer.brush_stroke_points.clear()
        self.viewer.is_selecting = False
        if mode == "orbit":
            self.viewer.pan_mode = False
            self.viewer.unsetCursor()
            self.select_mode_btn.blockSignals(True)
            self.select_mode_btn.setChecked(False)
            self.select_mode_btn.blockSignals(False)
            self._toggle_selection_mode(False)
            if self.viewer.measure_mode:
                self._toggle_measure_mode(False)
            self.viewer.update()
            self.status_label.setText("【🖐 視点回転モード (V)】左ドラッグで回転、右/中ドラッグでパン、ホイールでズーム")
            self.show_toast("🖐 視点回転モード (V)")
        elif mode == "pan":
            self.viewer.pan_mode = True
            self.viewer.setCursor(Qt.OpenHandCursor)
            self.select_mode_btn.blockSignals(True)
            self.select_mode_btn.setChecked(False)
            self.select_mode_btn.blockSignals(False)
            self._toggle_selection_mode(False)
            if self.viewer.measure_mode:
                self._toggle_measure_mode(False)
            self.viewer.update()
            self.status_label.setText("【✋ パン移動モード (H / V)】左ドラッグでそのまま画面を平行移動、Alt+左/中ドラッグで回転")
            self.show_toast("✋ パン平行移動モード (左ドラッグで移動)")
        elif mode == "measure":
            self.viewer.pan_mode = False
            self.viewer.unsetCursor()
            self._toggle_measure_mode(True)
        elif mode in ("brush", "rect", "circle", "polygon"):
            self.viewer.pan_mode = False
            self.viewer.unsetCursor()
            if self.viewer.measure_mode:
                self._toggle_measure_mode(False)
            shape_map = {
                "brush": PointCloudViewer.SELECT_BRUSH,
                "rect": PointCloudViewer.SELECT_RECT,
                "circle": PointCloudViewer.SELECT_CIRCLE,
                "polygon": PointCloudViewer.SELECT_POLYGON,
            }
            target_shape = shape_map[mode]
            idx = self.select_shape_combo.findData(target_shape)
            if idx >= 0:
                self.select_shape_combo.setCurrentIndex(idx)
            self.select_mode_btn.blockSignals(True)
            self.select_mode_btn.setChecked(True)
            self.select_mode_btn.blockSignals(False)
            self._toggle_selection_mode(True)
        self._sync_segment_buttons()
        self.viewer.setFocus()

    def _sync_segment_buttons(self):
        """上部および左パネルのモード切替ショートカットボタンのチェック状態を現在の選択・パン・計測モードと同期"""
        if not hasattr(self, "seg_btn_orbit"):
            return
        is_sel = bool(self.viewer.selection_mode)
        is_meas = bool(self.viewer.measure_mode)
        is_pan = bool(getattr(self.viewer, "pan_mode", False)) and not is_sel and not is_meas
        is_orbit = not is_sel and not is_meas and not is_pan
        shape = self.viewer.selection_shape

        btn_states = [
            (self.seg_btn_orbit, is_orbit),
            (getattr(self, "seg_btn_pan", None), is_pan),
            (self.seg_btn_brush, is_sel and shape == PointCloudViewer.SELECT_BRUSH),
            (self.seg_btn_rect, is_sel and shape == PointCloudViewer.SELECT_RECT),
            (self.seg_btn_circle, is_sel and shape == PointCloudViewer.SELECT_CIRCLE),
            (self.seg_btn_poly, is_sel and shape == PointCloudViewer.SELECT_POLYGON),
            (getattr(self, "side_btn_orbit", None), is_orbit),
            (getattr(self, "side_btn_pan", None), is_pan),
            (getattr(self, "side_btn_brush", None), is_sel and shape == PointCloudViewer.SELECT_BRUSH),
            (getattr(self, "side_btn_rect", None), is_sel and shape == PointCloudViewer.SELECT_RECT),
            (getattr(self, "side_btn_circle", None), is_sel and shape == PointCloudViewer.SELECT_CIRCLE),
            (getattr(self, "side_btn_poly", None), is_sel and shape == PointCloudViewer.SELECT_POLYGON),
        ]
        for btn, checked in btn_states:
            if btn is not None:
                btn.blockSignals(True)
                btn.setChecked(checked)
                btn.blockSignals(False)

    def _show_viewer_context_menu(self, global_pos: QPoint):
        """★ 柱④: 3Dビューア上の右クリック文脈メニュー (対象に応じた操作をその場で開く)"""
        menu = QMenu(self)
        n_sel = len(self.viewer.selected_indices) if self.viewer.selected_indices is not None else 0

        if n_sel > 0:
            act_del = menu.addAction(f"🗑 選択した {n_sel:,} 点をノイズへ削除 (Del)")
            act_del.triggered.connect(self._delete_selected_points)

            act_wand = menu.addAction(f"🪄 選択点から3D連結塊を一括選択 (G)")
            act_wand.triggered.connect(self._on_magic_wand_cluster_select)

            act_local = menu.addAction(f"📦 選択範囲内のみ細かくノイズ除去")
            act_local.triggered.connect(self._on_local_fine_clean_selected)

            move_sub = menu.addMenu(f"📁 選択した {n_sel:,} 点をレイヤーへ移動")
            for lid in sorted(self.viewer.layers.keys(), key=lambda x: (x == 0, x)):
                linfo = self.viewer.layers[lid]
                lock_mark = "🔒" if linfo.locked else ""
                act_mv = move_sub.addAction(f"{lock_mark}[ID:{lid}] {linfo.name}")
                act_mv.triggered.connect(lambda _=False, target_id=lid: self._move_selected_to_specific_layer(target_id))

            act_rest = menu.addAction("♻ 選択内のノイズ点を通常レイヤーへ復元")
            act_rest.triggered.connect(self._restore_selected_noise)

            act_inv = menu.addAction("🔄 選択を反転 (I)")
            act_inv.triggered.connect(self._invert_selection)

            act_clr = menu.addAction("✖ 選択を解除 (Esc)")
            act_clr.triggered.connect(self._clear_selection)
            menu.addSeparator()

        clean_sub = menu.addMenu("🧹 クイック・ノイズ処理コマンド")
        clean_sub.addAction(
            f"🎯 ノイズ専用モード切替 ({'現在: ON' if self.viewer.clean_focus_mode else '現在: OFF'}) (N)",
            lambda: self._toggle_clean_focus_mode(not self.viewer.clean_focus_mode),
        )
        clean_sub.addAction(
            f"⚡ 囲み・なぞり即削除 ({'現在: ON' if self.viewer.instant_delete_mode else '現在: OFF'})",
            lambda: self._toggle_instant_delete_mode(not self.viewer.instant_delete_mode),
        )
        clean_sub.addSeparator()
        clean_sub.addAction("🏝️ メイン塊のみ残す (大まかな浮遊塊を一括除去)", self._on_keep_main_cluster_only)
        clean_sub.addAction("🪶 表面・輪郭のエッジ飛散ヒゲを精密除去", self._on_quick_edge_veil_clean)

        mode_sub = menu.addMenu("🧭 操作モード切替")
        mode_sub.addAction("🖐 視点回転モード (V)", lambda: self._set_interaction_mode("orbit"))
        mode_sub.addAction("✋ パン平行移動モード (H / V)", lambda: self._set_interaction_mode("pan"))
        mode_sub.addAction("🧹 ブラシなぞり選択/消去モード (B)", lambda: self._set_interaction_mode("brush"))
        mode_sub.addAction("⬜ 矩形選択モード (R)", lambda: self._set_interaction_mode("rect"))
        mode_sub.addAction("⭕ 円形選択モード (C)", lambda: self._set_interaction_mode("circle"))
        mode_sub.addAction("⬠ 多角形選択モード (L)", lambda: self._set_interaction_mode("polygon"))
        mode_sub.addAction("📐 2点間寸法計測モード (M)", lambda: self._toggle_measure_mode(not self.viewer.measure_mode))

        view_sub = menu.addMenu("🎥 視点プリセット / 斜め壁アライン")
        view_sub.addAction("等角ビュー (Iso)", lambda: self.viewer.reset_view())
        view_sub.addAction("上面ビュー (XY)", lambda: self.viewer.set_view_top())
        view_sub.addAction("正面ビュー (XZ)", lambda: self.viewer.set_view_front())
        view_sub.addAction("側面ビュー (YZ)", lambda: self.viewer.set_view_side())
        view_sub.addSeparator()
        view_sub.addAction("🧭 2点ピック斜め壁アライン開始 (U)", lambda: self._toggle_wall_align_mode(True))
        view_sub.addAction("━ 斜め壁: 横表示 (上面・水平揃え)", lambda: self._apply_wall_view("horiz"))
        view_sub.addAction("┃ 斜め壁: 縦表示 (上面・垂直揃え)", lambda: self._apply_wall_view("vert"))
        view_sub.addAction("🧱 斜め壁: 直交正面ビュー", lambda: self._apply_wall_view("ortho_front"))
        view_sub.addAction("🔪 斜め壁: 壁真横・断面エッジ", lambda: self._apply_wall_view("edge"))
        view_sub.addAction("🔄 斜め壁: 180°向き反転", self._flip_wall_view)

        if (
            self.viewer.clip_x_enabled
            or self.viewer.clip_y_enabled
            or self.viewer.clip_z_enabled
            or self.viewer.clip_wall_enabled
        ):
            guides_on = self.viewer.show_section_origin_plane or self.viewer.show_section_box
            state_txt = "OFFにする (点群のみ表示)" if guides_on else "ONにする"
            act_sec_guide = menu.addAction(f"📐 断面基点枠・厚み枠を{state_txt} (B)")
            act_sec_guide.triggered.connect(self._toggle_section_guides_ui)

        menu.addSeparator()
        menu.addAction("🔍 全体フィット (F)", lambda: self.viewer.fit_to_screen())
        menu.addAction("🎥 透視 ⇄ 正射影切替 (P)", lambda: self.viewer.toggle_projection())
        if self.layer_undo_stack:
            menu.addAction("↩ 元に戻す (Ctrl+Z)", self._undo_action)
        if self.layer_redo_stack:
            menu.addAction("↪ やり直し (Ctrl+Y)", self._redo_action)

        menu.exec(global_pos)

    def _move_selected_to_specific_layer(self, target_id: int):
        """右クリック文脈メニューから指定レイヤーへ直接移動"""
        idx = self.target_layer_combo.findData(target_id)
        if idx >= 0:
            self.target_layer_combo.setCurrentIndex(idx)
        self._on_move_selected_to_target_layer()


    def _on_splitter_moved(self, pos: int, index: int):
        """ユーザーがスプリッター境界線を手動ドラッグした時の幅を記憶"""
        if index == 1:
            sizes = self.splitter.sizes()
            if len(sizes) >= 2 and sizes[0] > 0:
                self._sidebar_user_width = sizes[0]

    def _lock_sidebar_width(self):
        """コマンド選択・タブ切替・パネル開閉時に左パネル幅が勝手に変動して画面が動くのを完全に防止"""
        if hasattr(self, "splitter"):
            curr_sizes = self.splitter.sizes()
            if len(curr_sizes) >= 2 and curr_sizes[0] > 0:
                target_w = getattr(self, "_sidebar_user_width", 380)
                total = sum(curr_sizes)
                if total > 0:
                    left = max(340, min(target_w, total - 200))
                    right = total - left
                    if abs(curr_sizes[0] - left) > 1:
                        self.splitter.setSizes([left, right])

    def _on_panel_collapse_toggled(self, is_expanded: bool):
        """パネルの折りたたみ/展開時に幅を再固定"""
        self._lock_sidebar_width()

    def _expand_all_panels(self):
        """全パネルを一括展開"""
        for g in [
            getattr(self, "file_group", None),
            getattr(self, "view_group", None),
            getattr(self, "section_group", None),
            getattr(self, "layer_group", None),
            getattr(self, "tools_group", None),
            getattr(self, "auto_group", None),
            getattr(self, "settings_group", None),
            getattr(self, "log_group", None),
        ]:
            if g is not None and hasattr(g, "set_expanded"):
                g.set_expanded(True)
        self._lock_sidebar_width()
        self.show_toast("📂 全パネルを展開しました")

    def _collapse_all_panels(self):
        """全パネルを一括折りたたみ"""
        for g in [
            getattr(self, "file_group", None),
            getattr(self, "view_group", None),
            getattr(self, "section_group", None),
            getattr(self, "layer_group", None),
            getattr(self, "tools_group", None),
            getattr(self, "auto_group", None),
            getattr(self, "settings_group", None),
            getattr(self, "log_group", None),
        ]:
            if g is not None and hasattr(g, "set_expanded"):
                g.set_expanded(False)
        self._lock_sidebar_width()
        self.show_toast("📁 全パネルを折りたたみました")


    def _on_ss_options_changed(self):
        """スクリーンショット設定の変更をビューアへ反映"""
        self.viewer.set_screenshot_include_grid(self.chk_ss_grid.isChecked())
        self.viewer.set_screenshot_include_axes(self.chk_ss_axes.isChecked())

    def _on_capture_screenshot(self):
        """⑧ 高画質スクリーンショットのキャプチャ (クリップボード自動コピー ＆ ファイル保存)"""
        import datetime
        now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"screenshot_{now_str}.png"

        # まずクリップボードに即時コピー
        # 文字類(HUDなど)は常に省き、グリッド・XYZ凡例は設定値に従って撮影
        img = self.viewer.capture_screenshot(save_path=None, copy_to_clipboard=True)
        if img is None or img.isNull():
            self.show_toast("⚠️ スクリーンショットの取得に失敗しました")
            return

        self.show_toast("📋 スクショをクリップボードにコピーしました！")

        # ファイル保存ダイアログ
        out_path, _ = QFileDialog.getSaveFileName(
            self,
            "3Dスクリーンショットの保存 (クリップボードにもコピー済)",
            default_filename,
            "PNG画像 (*.png);;JPEG画像 (*.jpg *.jpeg);;すべてのファイル (*.*)",
        )
        if out_path:
            success = img.save(out_path)
            if success:
                self.show_toast(f"📷 画像を保存しました: {os.path.basename(out_path)}")
                self.log_text.append(f"スクリーンショット保存: {out_path}")

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
        self.btn_measure.blockSignals(True)
        self.nav_btn_measure.blockSignals(True)
        self.btn_measure.setChecked(enabled)
        self.nav_btn_measure.setChecked(enabled)
        self.btn_measure.blockSignals(False)
        self.nav_btn_measure.blockSignals(False)
        if enabled:
            self.viewer.pan_mode = False
            self.viewer.unsetCursor()
            if self.viewer.selection_mode:
                self.select_mode_btn.blockSignals(True)
                self.select_mode_btn.setChecked(False)
                self.select_mode_btn.blockSignals(False)
                self._toggle_selection_mode(False)
            self.btn_measure.setStyleSheet("background-color: #0284c7; font-weight: bold;")
            self.status_label.setText("【計測モード】3D画面上の2点を左クリックしてください（右クリックまたはEscでクリア）")
        else:
            self.btn_measure.setStyleSheet("font-weight: bold;")
            self.viewer.measure_points.clear()
            self.viewer.measure_result_text = ""
            self.viewer.update()
            self.status_label.setText("計測モードを終了しました")
        self._sync_segment_buttons()
        self.viewer.setFocus()

    def _on_selection_shape_changed(self, index: int):
        shape = self.select_shape_combo.currentData()
        self.viewer.selection_shape = shape
        self.viewer.polygon_points.clear()
        self.viewer.brush_stroke_points.clear()
        self.viewer.is_selecting = False
        self.btn_poly_confirm.setEnabled(shape == PointCloudViewer.SELECT_POLYGON)
        if shape == PointCloudViewer.SELECT_POLYGON:
            self.manual_status_label.setText("【多角形選択】左クリックで頂点追加、ダブルクリックまたはEnterで確定")
        elif shape == PointCloudViewer.SELECT_CIRCLE:
            self.manual_status_label.setText("【円形選択】中心からドラッグして円を描いてください")
        elif shape == PointCloudViewer.SELECT_BRUSH:
            self.manual_status_label.setText("【ブラシ選択/消去】なぞって選択・消去 ([ / ] または Ctrl+ホイールでサイズ変更)")
        else:
            self.manual_status_label.setText("【矩形選択】ドラッグして四角く囲んでください")
        self.viewer.update()
        self._sync_segment_buttons()
        self.viewer.setFocus()

    def _on_brush_slider_changed(self, val: int):
        self.lbl_brush_radius.setText(f"{val} px")
        self.viewer.set_brush_radius(val)

    def _on_viewer_brush_radius_changed(self, val: int):
        self.brush_slider.blockSignals(True)
        self.brush_slider.setValue(int(val))
        self.brush_slider.blockSignals(False)
        self.lbl_brush_radius.setText(f"{val} px")

    def _toggle_instant_delete_mode(self, enabled: bool):
        """⚡ 囲み・なぞり即削除モードのON/OFF切替"""
        self.viewer.instant_delete_mode = bool(enabled)
        for btn, on_text, off_text in [
            (self.btn_instant_del, "⚡ 即削除: ON", "⚡ 即削除"),
            (self.chk_instant_del, "⚡ 囲み即削除: ON", "⚡ 囲み即削除: OFF"),
        ]:
            btn.blockSignals(True)
            btn.setChecked(bool(enabled))
            btn.setText(on_text if enabled else off_text)
            if enabled:
                btn.setStyleSheet(
                    "background-color: #b91c1c; border: 1px solid #f87171; color: #ffffff; font-size: 11px; font-weight: bold; padding: 4px 6px;"
                )
            else:
                btn.setStyleSheet("font-size: 11px; font-weight: bold; padding: 4px 6px;")
            btn.blockSignals(False)
        self.viewer.update()
        if enabled:
            self.status_label.setText("⚡ 即時削除モード ON: 囲んだ瞬間・ブラシでなぞった瞬間にノイズレイヤーへ消去します")
            self.show_toast("⚡ 囲み・なぞり即削除モード ON (Ctrl+Zでいつでも戻せます)")
        else:
            self.status_label.setText("即時削除モード OFF: 通常選択モードに戻りました")
            self.show_toast("即時削除モード OFF")

    def _toggle_clean_focus_mode(self, enabled: bool):
        """🎯 ノイズ処理専用モード (高コントラスト背景・点強調・ブラシ即削除・専用UI表示) のワンタッチ切替"""
        self.viewer.clean_focus_mode = bool(enabled)
        for btn in (self.btn_clean_mode, self.btn_clean_mode_side):
            btn.blockSignals(True)
            btn.setChecked(bool(enabled))
            if enabled:
                btn.setStyleSheet(
                    "background-color: #0284c7; border: 1px solid #7dd3fc; color: #ffffff; font-size: 11px; font-weight: bold; padding: 4px 8px;"
                )
            else:
                btn.setStyleSheet(
                    "background-color: #1e293b; border: 1px solid #38bdf8; color: #38bdf8; font-size: 11px; font-weight: bold; padding: 4px 8px;"
                )
            btn.blockSignals(False)

        if enabled:
            self._prev_bg_idx = self.bg_combo.currentIndex()
            self._prev_pt_size = self.size_slider.value()
            self.bg_combo.setCurrentIndex(1)  # ブラック高コントラスト背景
            if self.size_slider.value() < 3:
                self.size_slider.setValue(3)
            self._set_sidebar_workflow_tab("clean")
            self._toggle_instant_delete_mode(True)
            self._set_interaction_mode("brush")
            self.status_label.setText(
                "🎯【ノイズ処理専用モード ON】ブラシや矩形で囲むだけで即消去！ Alt+左ドラッグで視点回転、Nキーで終了"
            )
            self.show_toast("🎯 ノイズ処理専用モード ON (なぞる・囲むだけで即消去)")
        else:
            if hasattr(self, "_prev_bg_idx"):
                self.bg_combo.setCurrentIndex(self._prev_bg_idx)
            if hasattr(self, "_prev_pt_size"):
                self.size_slider.setValue(self._prev_pt_size)
            self._set_sidebar_workflow_tab("all")
            self._toggle_instant_delete_mode(False)
            self.viewer.update()
            self.status_label.setText("ノイズ処理専用モードを終了しました")
            self.show_toast("通常ワークスペースへ戻りました")

    def _on_instant_erase_requested(self, erase_indices: np.ndarray):
        """⚡ 即時削除モードで囲まれた・なぞられた点群を1ステップで即座にノイズレイヤーへ消去"""
        if self.viewer.points_raw is None or len(erase_indices) == 0:
            return
        self._push_layer_undo_state()
        moved = self.viewer.move_indices_to_layer(erase_indices, PointCloudViewer.LAYER_NOISE, respect_lock=True)
        self._clear_selection()
        if moved > 0:
            self.log_text.append(f"⚡ 即時消去: {moved:,} 点をノイズレイヤーへ移動しました (Ctrl+Zで取消可)")
            self.status_label.setText(f"⚡ {moved:,} 点を即座にノイズ消去しました (Ctrl+Zで元に戻せます)")
            self.show_toast(f"⚡ {moved:,} 点を即消去しました")

    def _confirm_polygon_selection(self):
        if len(self.viewer.polygon_points) >= 3:
            self.viewer._process_polygon_selection()
            self.viewer.polygon_points.clear()
            self.viewer.update()

    def _on_bounds_ready(self, b_min: np.ndarray, b_max: np.ndarray):
        """点群のワールド範囲に合わせて断面スライダー・スピンボックスの範囲を初期化（非有限値・オーバーフロー安全ガード付き）"""
        safe_min = np.nan_to_num(np.asarray(b_min, dtype=np.float64), nan=0.0, posinf=1e6, neginf=-1e6)
        safe_max = np.nan_to_num(np.asarray(b_max, dtype=np.float64), nan=0.0, posinf=1e6, neginf=-1e6)
        # Qt QSlider.setRange の 32bit C int オーバーフロー防止のためクランプ
        safe_min = np.clip(safe_min, -1e7, 1e7)
        safe_max = np.clip(safe_max, -1e7, 1e7)
        safe_max = np.maximum(safe_min, safe_max)

        self.world_min = safe_min
        self.world_max = safe_max

        max_span = float(np.max(safe_max - safe_min))
        self._sec_pos_scale = 100.0 if max_span <= 10000.0 else 10.0
        scale = self._sec_pos_scale
        max_thick_slider = max(200, min(5000, int(max(10.0, max_span) * 10)))

        for idx, (s_pos, sp_pos, s_th, sp_th) in enumerate([
            (self.slider_pos_x, self.spin_pos_x, self.slider_thick_x, self.spin_thick_x),
            (self.slider_pos_y, self.spin_pos_y, self.slider_thick_y, self.spin_thick_y),
            (self.slider_pos_z, self.spin_pos_z, self.slider_thick_z, self.spin_thick_z),
        ]):
            mid = 0.5 * (safe_min[idx] + safe_max[idx])
            sp_pos.blockSignals(True)
            sp_pos.setRange(float(safe_min[idx] - 10.0), float(safe_max[idx] + 10.0))
            sp_pos.setValue(float(mid))
            sp_pos.blockSignals(False)

            s_pos.blockSignals(True)
            s_pos.setRange(int(round(safe_min[idx] * scale)), int(round(safe_max[idx] * scale)))
            s_pos.setValue(int(round(mid * scale)))
            s_pos.blockSignals(False)

            s_th.blockSignals(True)
            s_th.setRange(1, max_thick_slider)
            s_th.setValue(int(round(sp_th.value() * 10.0)))
            s_th.blockSignals(False)

        self._on_section_x_changed()
        self._on_section_y_changed()
        self._on_section_z_changed()

        # 📏 高さ色分けスライダー・スピンボックスの初期化
        self._init_height_range_from_bounds(safe_min[2], safe_max[2])

    def _init_height_range_from_bounds(self, z_min: float, z_max: float):
        """点群のワールドZ範囲に基づいて高さ色分けスライダー・スピンボックスを初期化"""
        if not hasattr(self, "slider_height_min"):
            return
        z_min = float(z_min)
        z_max = float(z_max)
        if z_min > z_max:
            z_min, z_max = z_max, z_min
        if abs(z_max - z_min) < 1e-4:
            z_max = z_min + 1.0

        self._height_bounds_min = z_min
        self._height_bounds_max = z_max

        span = z_max - z_min
        self._height_slider_scale = 100.0 if span <= 5000.0 else 10.0
        scale = self._height_slider_scale

        margin = max(5.0, span * 0.2)
        s_min_val = int(round(z_min * scale))
        s_max_val = int(round(z_max * scale))

        self.spin_height_min.blockSignals(True)
        self.spin_height_max.blockSignals(True)
        self.spin_height_min.setRange(z_min - margin, z_max + margin)
        self.spin_height_max.setRange(z_min - margin, z_max + margin)
        self.spin_height_min.setValue(z_min)
        self.spin_height_max.setValue(z_max)
        self.spin_height_min.blockSignals(False)
        self.spin_height_max.blockSignals(False)

        self.slider_height_min.blockSignals(True)
        self.slider_height_max.blockSignals(True)
        self.slider_height_min.setRange(s_min_val, s_max_val)
        self.slider_height_max.setRange(s_min_val, s_max_val)
        self.slider_height_min.setValue(s_min_val)
        self.slider_height_max.setValue(s_max_val)
        self.slider_height_min.blockSignals(False)
        self.slider_height_max.blockSignals(False)

        # 初期状態は全高自動
        if hasattr(self, "chk_custom_height"):
            self.chk_custom_height.blockSignals(True)
            self.chk_custom_height.setChecked(False)
            self.chk_custom_height.blockSignals(False)

        if hasattr(self, "lbl_height_range_info"):
            self.lbl_height_range_info.setText(f"配色範囲: 全高自動 ({z_min:.2f}m 〜 {z_max:.2f}m)")
        self.viewer.set_custom_height_range(False, z_min, z_max)

    def _on_custom_height_toggled(self, checked: bool):
        """任意高さ色分けチェックボックス切り替え"""
        self._apply_custom_height_range()
        if checked:
            min_z = self.spin_height_min.value()
            max_z = self.spin_height_max.value()
            self.show_toast(f"📏 任意高さ色分け: {min_z:.2f}m 〜 {max_z:.2f}m")
        else:
            self.show_toast("📏 高さ色分け: 全高自動")

    def _reset_height_range(self):
        """全高リセットボタン押下時: 点群全体の標高範囲に戻す"""
        z_min = getattr(self, "_height_bounds_min", 0.0)
        z_max = getattr(self, "_height_bounds_max", 10.0)
        scale = getattr(self, "_height_slider_scale", 100.0)

        self.spin_height_min.blockSignals(True)
        self.spin_height_max.blockSignals(True)
        self.slider_height_min.blockSignals(True)
        self.slider_height_max.blockSignals(True)

        self.spin_height_min.setValue(z_min)
        self.spin_height_max.setValue(z_max)
        self.slider_height_min.setValue(int(round(z_min * scale)))
        self.slider_height_max.setValue(int(round(z_max * scale)))

        self.spin_height_min.blockSignals(False)
        self.spin_height_max.blockSignals(False)
        self.slider_height_min.blockSignals(False)
        self.slider_height_max.blockSignals(False)

        if hasattr(self, "chk_custom_height"):
            self.chk_custom_height.blockSignals(True)
            self.chk_custom_height.setChecked(False)
            self.chk_custom_height.blockSignals(False)

        if hasattr(self, "lbl_height_range_info"):
            self.lbl_height_range_info.setText(f"配色範囲: 全高自動 ({z_min:.2f}m 〜 {z_max:.2f}m)")
        self.viewer.set_custom_height_range(False, z_min, z_max)
        self.show_toast("🔄 高さ色分けレンジを全高にリセットしました")

    def _on_slider_height_min_changed(self, val: int):
        """下限スライダー変更時 (直感ドラッグ)"""
        scale = getattr(self, "_height_slider_scale", 100.0)
        z_val = val / scale
        self.spin_height_min.blockSignals(True)
        self.spin_height_min.setValue(z_val)
        self.spin_height_min.blockSignals(False)

        # 下限が上限を超えないよう上限を連動
        max_val = self.slider_height_max.value()
        if val > max_val:
            self.slider_height_max.blockSignals(True)
            self.spin_height_max.blockSignals(True)
            self.slider_height_max.setValue(val)
            self.spin_height_max.setValue(z_val)
            self.slider_height_max.blockSignals(False)
            self.spin_height_max.blockSignals(False)

        # 操作時は自動的に任意モードをONにする
        if hasattr(self, "chk_custom_height") and not self.chk_custom_height.isChecked():
            self.chk_custom_height.blockSignals(True)
            self.chk_custom_height.setChecked(True)
            self.chk_custom_height.blockSignals(False)

        # ドラッグ中は超高速プレビュー (60fps) でヌルヌル追従
        self._preview_custom_height_range()

    def _on_slider_height_max_changed(self, val: int):
        """上限スライダー変更時 (直感ドラッグ)"""
        scale = getattr(self, "_height_slider_scale", 100.0)
        z_val = val / scale
        self.spin_height_max.blockSignals(True)
        self.spin_height_max.setValue(z_val)
        self.spin_height_max.blockSignals(False)

        # 上限が下限を下回らないよう下限を連動
        min_val = self.slider_height_min.value()
        if val < min_val:
            self.slider_height_min.blockSignals(True)
            self.spin_height_min.blockSignals(True)
            self.slider_height_min.setValue(val)
            self.spin_height_min.setValue(z_val)
            self.slider_height_min.blockSignals(False)
            self.spin_height_min.blockSignals(False)

        if hasattr(self, "chk_custom_height") and not self.chk_custom_height.isChecked():
            self.chk_custom_height.blockSignals(True)
            self.chk_custom_height.setChecked(True)
            self.chk_custom_height.blockSignals(False)

        # ドラッグ中は超高速プレビュー (60fps) でヌルヌル追従
        self._preview_custom_height_range()

    def _on_spin_height_min_changed(self, val: float):
        """下限スピンボックス変更時 (数値入力)"""
        scale = getattr(self, "_height_slider_scale", 100.0)
        s_val = int(round(val * scale))
        self.slider_height_min.blockSignals(True)
        self.slider_height_min.setValue(s_val)
        self.slider_height_min.blockSignals(False)

        max_val = self.spin_height_max.value()
        if val > max_val:
            self.spin_height_max.blockSignals(True)
            self.slider_height_max.blockSignals(True)
            self.spin_height_max.setValue(val)
            self.slider_height_max.setValue(s_val)
            self.spin_height_max.blockSignals(False)
            self.slider_height_max.blockSignals(False)

        if hasattr(self, "chk_custom_height") and not self.chk_custom_height.isChecked():
            self.chk_custom_height.blockSignals(True)
            self.chk_custom_height.setChecked(True)
            self.chk_custom_height.blockSignals(False)

        self._apply_custom_height_range()

    def _on_spin_height_max_changed(self, val: float):
        """上限スピンボックス変更時 (数値入力)"""
        scale = getattr(self, "_height_slider_scale", 100.0)
        s_val = int(round(val * scale))
        self.slider_height_max.blockSignals(True)
        self.slider_height_max.setValue(s_val)
        self.slider_height_max.blockSignals(False)

        min_val = self.spin_height_min.value()
        if val < min_val:
            self.spin_height_min.blockSignals(True)
            self.slider_height_min.blockSignals(True)
            self.spin_height_min.setValue(val)
            self.slider_height_min.setValue(s_val)
            self.spin_height_min.blockSignals(False)
            self.slider_height_min.blockSignals(False)

        if hasattr(self, "chk_custom_height") and not self.chk_custom_height.isChecked():
            self.chk_custom_height.blockSignals(True)
            self.chk_custom_height.setChecked(True)
            self.chk_custom_height.blockSignals(False)

        self._apply_custom_height_range()

    def _preview_custom_height_range(self):
        """スライダードラッグ中専用: 画面表示中の点群だけを超高速(60fps)でプレビュー更新"""
        if not hasattr(self, "spin_height_min") or not hasattr(self, "viewer"):
            return
        min_z = float(self.spin_height_min.value())
        max_z = float(self.spin_height_max.value())
        if min_z > max_z:
            min_z, max_z = max_z, min_z
        span = max_z - min_z
        enabled = self.chk_custom_height.isChecked() if hasattr(self, "chk_custom_height") else False

        if hasattr(self, "lbl_height_range_info"):
            if enabled:
                self.lbl_height_range_info.setText(f"範囲: {min_z:.2f}m 〜 {max_z:.2f}m (高低差: {span:.2f}m)")
            else:
                self.lbl_height_range_info.setText(f"配色範囲: 全高自動 (待機: {min_z:.2f}m〜{max_z:.2f}m)")

        self.viewer.preview_custom_height_range(enabled, min_z, max_z)

    def _apply_custom_height_range(self):
        """スライダー・スピンボックスの値をビューアーに適用して即時カラー反映"""
        if not hasattr(self, "spin_height_min") or not hasattr(self, "viewer"):
            return
        min_z = float(self.spin_height_min.value())
        max_z = float(self.spin_height_max.value())
        if min_z > max_z:
            min_z, max_z = max_z, min_z
        span = max_z - min_z
        enabled = self.chk_custom_height.isChecked() if hasattr(self, "chk_custom_height") else False

        if hasattr(self, "lbl_height_range_info"):
            if enabled:
                self.lbl_height_range_info.setText(f"範囲: {min_z:.2f}m 〜 {max_z:.2f}m (高低差: {span:.2f}m)")
            else:
                self.lbl_height_range_info.setText(f"配色範囲: 全高自動 (待機: {min_z:.2f}m〜{max_z:.2f}m)")

        self.viewer.set_custom_height_range(enabled, min_z, max_z)

    def _compute_section_interval(self, pos: float, thick: float):
        """現在の基点基準モードに応じて (min_val, max_val, ラベル文字列) を算出"""
        mode = self.sec_origin_mode_combo.currentData() if hasattr(self, "sec_origin_mode_combo") else "center"
        thick = max(0.01, float(thick))
        pos = float(pos)
        if mode == "plus":
            return pos, pos + thick, f"基点{pos:.2f}m (0〜+{thick:.2f}m)"
        elif mode == "minus":
            return pos - thick, pos, f"基点{pos:.2f}m (-{thick:.2f}m〜0)"
        elif mode == "above":
            return pos, 1e9, f"基点{pos:.2f}m 以上全域"
        elif mode == "below":
            return -1e9, pos, f"基点{pos:.2f}m 以下全域"
        else:
            half = thick / 2.0
            return pos - half, pos + half, f"基点{pos:.2f}m (±{half:.2f}m)"

    def _on_section_guide_visibility_changed(self):
        """GUIチェックボックスからビューアの基点枠・厚み枠表示を更新"""
        self.viewer.set_show_section_origin_plane(self.chk_sec_show_origin.isChecked())
        self.viewer.set_show_section_box(self.chk_sec_show_box.isChecked())

    def _on_viewer_section_guides_changed(self, show_origin: bool, show_box: bool):
        """ビューア側（Bキーなど）の枠表示状態をGUIチェックボックスへ反映"""
        self.chk_sec_show_origin.blockSignals(True)
        self.chk_sec_show_box.blockSignals(True)
        self.chk_sec_show_origin.setChecked(show_origin)
        self.chk_sec_show_box.setChecked(show_box)
        self.chk_sec_show_origin.blockSignals(False)
        self.chk_sec_show_box.blockSignals(False)

    def _toggle_section_guides_ui(self):
        """断面ガイド枠を一括トグル切替"""
        is_on = self.viewer.toggle_section_guides()
        txt = "表示" if is_on else "非表示 (点群のみ)"
        self.show_toast(f"📐 断面枠: {txt} (B)")

    def _on_section_origin_mode_changed(self):
        """断面スライスの基点基準モード切替時ハンドラ"""
        self._on_section_x_changed()
        self._on_section_y_changed()
        self._on_section_z_changed()
        self._on_section_wall_changed()
        mode_text = self.sec_origin_mode_combo.currentText()
        self.status_label.setText(f"📐 断面基点モード切替: {mode_text}")
        self.show_toast(f"📐 {mode_text}")

    def _set_section_origin_from_current_view(self):
        """現在画面で見ているカメラ注視点（または選択点群の重心）をXYZ断面スライダーのシーク基点に一発セット"""
        focus_pt = self.viewer.get_current_focus_world_point()
        scale = getattr(self, "_sec_pos_scale", 100.0)
        for idx, (s_pos, sp_pos) in enumerate([
            (self.slider_pos_x, self.spin_pos_x),
            (self.slider_pos_y, self.spin_pos_y),
            (self.slider_pos_z, self.spin_pos_z),
        ]):
            val = float(focus_pt[idx])
            sp_pos.blockSignals(True)
            sp_pos.setValue(val)
            sp_pos.blockSignals(False)
            s_pos.blockSignals(True)
            s_pos.setValue(int(round(val * scale)))
            s_pos.blockSignals(False)

        self._on_section_x_changed()
        self._on_section_y_changed()
        self._on_section_z_changed()
        src_str = "選択点群の中心" if len(self.viewer.selected_indices) > 0 else "現在のカメラ注目位置"
        msg = f"🎯 {src_str} (X:{focus_pt[0]:.2f}m, Y:{focus_pt[1]:.2f}m, Z:{focus_pt[2]:.2f}m) を断面シーク基点にセットしました"
        self.status_label.setText(msg)
        self.show_toast(f"🎯 基点セット: ({focus_pt[0]:.2f}, {focus_pt[1]:.2f}, {focus_pt[2]:.2f})")

    def _apply_section_thickness_preset(self, thick_val: float):
        """ワンクリックで断面厚みを一括変更する"""
        thick_val = max(0.01, float(thick_val))
        any_checked = (
            self.chk_sec_x.isChecked()
            or self.chk_sec_y.isChecked()
            or self.chk_sec_z.isChecked()
            or self.chk_sec_wall.isChecked()
        )
        targets = []
        if not any_checked or self.chk_sec_x.isChecked():
            targets.append((self.slider_thick_x, self.spin_thick_x, self._on_section_x_changed))
        if not any_checked or self.chk_sec_y.isChecked():
            targets.append((self.slider_thick_y, self.spin_thick_y, self._on_section_y_changed))
        if not any_checked or self.chk_sec_z.isChecked():
            targets.append((self.slider_thick_z, self.spin_thick_z, self._on_section_z_changed))
        if not any_checked or self.chk_sec_wall.isChecked():
            targets.append((self.slider_thick_wall, self.spin_thick_wall, self._on_section_wall_changed))

        for s_th, sp_th, cb in targets:
            sp_th.blockSignals(True)
            sp_th.setValue(thick_val)
            sp_th.blockSignals(False)
            s_th.blockSignals(True)
            s_th.setValue(max(s_th.minimum(), min(s_th.maximum(), int(round(thick_val * 10.0)))))
            s_th.blockSignals(False)
            cb()
        self.status_label.setText(f"📐 断面スライスの厚みを {thick_val:.2f}m に設定しました")
        self.show_toast(f"📐 断面厚み: {thick_val:.2f} m")

    def _align_to_section_axis(self, axis: str):
        """指定断面軸をONにし、そのシーク基点を中心に断面正対ビューへ切り替える"""
        ax = axis.lower()
        if ax == "x" and not self.chk_sec_x.isChecked():
            self.chk_sec_x.setChecked(True)
        elif ax == "y" and not self.chk_sec_y.isChecked():
            self.chk_sec_y.setChecked(True)
        elif ax == "z" and not self.chk_sec_z.isChecked():
            self.chk_sec_z.setChecked(True)
        self.viewer.align_view_to_section(ax)
        self.status_label.setText(f"📐 {ax.upper()}断面のシーク基点にカメラを正対させました")
        self.show_toast(f"📐 {ax.upper()}断面 正対ビュー")

    # --- 🧭 2点ピック斜め壁アライン (横・縦・直交視点 & 斜め壁スライス) ハンドラ ---
    def _sync_wall_pick_buttons(self, checked: bool):
        for b in [
            getattr(self, "btn_wall_pick", None),
            getattr(self, "btn_sec_wall_pick", None),
            getattr(self, "nav_btn_wall_align", None),
        ]:
            if b is not None:
                b.blockSignals(True)
                b.setChecked(bool(checked))
                b.blockSignals(False)

    def _toggle_wall_align_mode(self, checked: bool):
        """🧭 2点ピック斜め壁アラインモードの開始・終了 (Uキー)"""
        self.viewer.wall_align_mode = bool(checked)
        self._sync_wall_pick_buttons(checked)
        if checked:
            self._prev_sel_before_wall_pick = self.viewer.selection_mode
            self.viewer.wall_align_points.clear()
            if self.viewer.measure_mode:
                self._toggle_measure_mode(False)
            self.viewer.setCursor(Qt.CrossCursor)
            self.status_label.setText("🧭【2点壁ピックモード(U)】斜めの壁に沿って 1点目 と 2点目 をクリックしてください")
            self.show_toast("🧭 斜めの壁上の2点をクリックしてください")
        else:
            if not self.viewer.pan_mode:
                self.viewer.unsetCursor()
            self.status_label.setText("2点壁ピックモードを終了しました")
        self.viewer.update()
        self.viewer.setFocus()

    def _on_wall_aligned(self, angle_deg: float, p1_world: np.ndarray, p2_world: np.ndarray):
        """点群上の2点ピック完了時に呼ばれるコールバック"""
        self._sync_wall_pick_buttons(False)
        if not self.viewer.pan_mode:
            self.viewer.unsetCursor()
        self.spin_pos_wall.blockSignals(True)
        self.spin_pos_wall.setValue(0.0)
        self.spin_pos_wall.blockSignals(False)
        self.slider_pos_wall.blockSignals(True)
        self.slider_pos_wall.setValue(0)
        self.slider_pos_wall.blockSignals(False)
        self._on_section_wall_changed(from_user_seek=False)

        v_type = self.viewer.wall_align_view_type
        v_names = {
            "horiz": "━ 横表示(上面・水平揃え)",
            "vert": "┃ 縦表示(上面・垂直揃え)",
            "ortho_front": "🧱 直交正面(水平立面)",
            "edge": "🔪 壁断面・真横",
        }
        v_str = v_names.get(v_type, "━ 横表示")
        self.log_text.append(
            f"🧭 2点壁アライン完了: 角度 {angle_deg:.2f}° -> {v_str} (正射影 ON)"
        )
        self.status_label.setText(
            f"🧭 斜め壁({angle_deg:.1f}°)に合わせて『{v_str}』へ切り替えました！そのまま矩形(R)・ブラシ(B)で壁沿いノイズを除去できます"
        )
        self.show_toast(f"🧭 壁角 {angle_deg:.1f}°: {v_str} に切替完了！")
        self.viewer.setFocus()

    def _apply_wall_view(self, view_type: str):
        """ピック済み2点ラインに対して 横(horiz) / 縦(vert) / 直交正面(ortho_front) / 壁真横(edge) へ1クリック切替"""
        ok = self.viewer.apply_wall_aligned_view(view_type, auto_ortho=True)
        if not ok:
            # まだ2点がピックされていない場合は、このモードを予約して即座に2点ピックモードを起動する
            self.viewer.wall_align_view_type = view_type
            self._toggle_wall_align_mode(True)
            return
        v_names = {
            "horiz": "━ 横表示 (上面・壁を水平配置)",
            "vert": "┃ 縦表示 (上面・壁を垂直配置)",
            "ortho_front": "🧱 直交正面 (水平立面)",
            "edge": "🔪 壁断面・真横 (エッジ視)",
        }
        self.show_toast(f"🧭 {v_names.get(view_type, view_type)}")
        self.viewer.setFocus()

    def _flip_wall_view(self):
        """2点壁アラインの向きを 180° 反転"""
        if not self.viewer.flip_wall_aligned_view():
            self._toggle_wall_align_mode(True)
            return
        self.show_toast(f"🔄 壁アライン向きを180°反転 ({self.viewer.wall_align_angle_deg:.1f}°)")
        self.viewer.setFocus()

    def _commit_pending_section_slices(self):
        """スライダー操作停止時(120ms後)またはマウス解放時に、全点データに対する確定断面スライスを実行"""
        if hasattr(self, "_sec_commit_timer") and self._sec_commit_timer.isActive():
            self._sec_commit_timer.stop()
        if not hasattr(self, "viewer") or self.viewer.points_centered is None:
            return
        if self.chk_sec_x.isChecked():
            self._on_section_x_changed(from_user_seek=False, is_preview=False)
        if self.chk_sec_y.isChecked():
            self._on_section_y_changed(from_user_seek=False, is_preview=False)
        if self.chk_sec_z.isChecked():
            self._on_section_z_changed(from_user_seek=False, is_preview=False)
        if hasattr(self, "chk_sec_wall") and self.chk_sec_wall.isChecked():
            self._on_section_wall_changed(from_user_seek=False, is_preview=False)

    def _on_slider_pos_wall_changed(self, val: int):
        pos = val / 100.0
        self.spin_pos_wall.blockSignals(True)
        self.spin_pos_wall.setValue(pos)
        self.spin_pos_wall.blockSignals(False)
        self._on_section_wall_changed(from_user_seek=True, is_preview=True)
        self._sec_commit_timer.start(120)

    def _on_spin_pos_wall_changed(self, pos: float):
        self.slider_pos_wall.blockSignals(True)
        self.slider_pos_wall.setValue(max(self.slider_pos_wall.minimum(), min(self.slider_pos_wall.maximum(), int(round(pos * 100.0)))))
        self.slider_pos_wall.blockSignals(False)
        self._on_section_wall_changed(from_user_seek=True, is_preview=False)

    def _on_slider_thick_wall_changed(self, val: int):
        thick = val / 10.0
        self.spin_thick_wall.blockSignals(True)
        self.spin_thick_wall.setValue(thick)
        self.spin_thick_wall.blockSignals(False)
        self._on_section_wall_changed(from_user_seek=False, is_preview=True)
        self._sec_commit_timer.start(120)

    def _on_spin_thick_wall_changed(self, thick: float):
        self.slider_thick_wall.blockSignals(True)
        self.slider_thick_wall.setValue(max(self.slider_thick_wall.minimum(), min(self.slider_thick_wall.maximum(), int(round(thick * 10.0)))))
        self.slider_thick_wall.blockSignals(False)
        self._on_section_wall_changed(from_user_seek=False, is_preview=False)

    def _on_section_wall_changed(self, from_user_seek: bool = False, is_preview: bool = False):
        enabled = self.chk_sec_wall.isChecked()
        pos = self.spin_pos_wall.value()
        thick = self.spin_thick_wall.value()
        min_v, max_v, desc = self._compute_section_interval(pos, thick)
        if self.viewer.wall_align_angle_deg is not None:
            self.lbl_sec_wall.setText(f"壁{self.viewer.wall_align_angle_deg:.1f}° {desc}")
        else:
            self.lbl_sec_wall.setText("未ピック(Uで2点指定)")
        self.viewer.set_section_wall(enabled, min_v, max_v, origin_val=pos, is_preview=is_preview)
        if from_user_seek and enabled and self.chk_sec_follow_cam.isChecked() and self.viewer.wall_align_angle_deg is not None and self.viewer.wall_align_center_world is not None:
            rad = math.radians(self.viewer.wall_align_angle_deg)
            nx, ny = -math.sin(rad), math.cos(rad)
            cx = float(self.viewer.wall_align_center_world[0] + pos * nx - self.viewer.center_offset[0])
            cy = float(self.viewer.wall_align_center_world[1] + pos * ny - self.viewer.center_offset[1])
            self.viewer.camera_target.setX(cx)
            self.viewer.camera_target.setY(cy)
            self.viewer.update()

    # --- X断面 スライダー / スピンボックス連動 ---
    def _on_slider_pos_x_changed(self, val: int):
        scale = getattr(self, "_sec_pos_scale", 100.0)
        pos = val / scale
        self.spin_pos_x.blockSignals(True)
        self.spin_pos_x.setValue(pos)
        self.spin_pos_x.blockSignals(False)
        self._on_section_x_changed(from_user_seek=True, is_preview=True)
        self._sec_commit_timer.start(120)

    def _on_spin_pos_x_changed(self, pos: float):
        scale = getattr(self, "_sec_pos_scale", 100.0)
        self.slider_pos_x.blockSignals(True)
        self.slider_pos_x.setValue(int(round(pos * scale)))
        self.slider_pos_x.blockSignals(False)
        self._on_section_x_changed(from_user_seek=True, is_preview=False)

    def _on_slider_thick_x_changed(self, val: int):
        thick = val / 10.0
        self.spin_thick_x.blockSignals(True)
        self.spin_thick_x.setValue(thick)
        self.spin_thick_x.blockSignals(False)
        self._on_section_x_changed(from_user_seek=False, is_preview=True)
        self._sec_commit_timer.start(120)

    def _on_spin_thick_x_changed(self, thick: float):
        self.slider_thick_x.blockSignals(True)
        self.slider_thick_x.setValue(max(self.slider_thick_x.minimum(), min(self.slider_thick_x.maximum(), int(round(thick * 10.0)))))
        self.slider_thick_x.blockSignals(False)
        self._on_section_x_changed(from_user_seek=False, is_preview=False)

    def _on_section_x_changed(self, from_user_seek: bool = False, is_preview: bool = False):
        enabled = self.chk_sec_x.isChecked()
        pos = self.spin_pos_x.value()
        thick = self.spin_thick_x.value()
        min_v, max_v, desc = self._compute_section_interval(pos, thick)
        self.lbl_sec_x.setText(desc)
        self.viewer.set_section_x(enabled, min_v, max_v, origin_val=pos, is_preview=is_preview)
        if from_user_seek and enabled and self.chk_sec_follow_cam.isChecked():
            self.viewer.focus_camera_on_world_axis(0, pos)

    # --- Y断面 スライダー / スピンボックス連動 ---
    def _on_slider_pos_y_changed(self, val: int):
        scale = getattr(self, "_sec_pos_scale", 100.0)
        pos = val / scale
        self.spin_pos_y.blockSignals(True)
        self.spin_pos_y.setValue(pos)
        self.spin_pos_y.blockSignals(False)
        self._on_section_y_changed(from_user_seek=True, is_preview=True)
        self._sec_commit_timer.start(120)

    def _on_spin_pos_y_changed(self, pos: float):
        scale = getattr(self, "_sec_pos_scale", 100.0)
        self.slider_pos_y.blockSignals(True)
        self.slider_pos_y.setValue(int(round(pos * scale)))
        self.slider_pos_y.blockSignals(False)
        self._on_section_y_changed(from_user_seek=True, is_preview=False)

    def _on_slider_thick_y_changed(self, val: int):
        thick = val / 10.0
        self.spin_thick_y.blockSignals(True)
        self.spin_thick_y.setValue(thick)
        self.spin_thick_y.blockSignals(False)
        self._on_section_y_changed(from_user_seek=False, is_preview=True)
        self._sec_commit_timer.start(120)

    def _on_spin_thick_y_changed(self, thick: float):
        self.slider_thick_y.blockSignals(True)
        self.slider_thick_y.setValue(max(self.slider_thick_y.minimum(), min(self.slider_thick_y.maximum(), int(round(thick * 10.0)))))
        self.slider_thick_y.blockSignals(False)
        self._on_section_y_changed(from_user_seek=False, is_preview=False)

    def _on_section_y_changed(self, from_user_seek: bool = False, is_preview: bool = False):
        enabled = self.chk_sec_y.isChecked()
        pos = self.spin_pos_y.value()
        thick = self.spin_thick_y.value()
        min_v, max_v, desc = self._compute_section_interval(pos, thick)
        self.lbl_sec_y.setText(desc)
        self.viewer.set_section_y(enabled, min_v, max_v, origin_val=pos, is_preview=is_preview)
        if from_user_seek and enabled and self.chk_sec_follow_cam.isChecked():
            self.viewer.focus_camera_on_world_axis(1, pos)

    # --- Z断面 スライダー / スピンボックス連動 ---
    def _on_slider_pos_z_changed(self, val: int):
        scale = getattr(self, "_sec_pos_scale", 100.0)
        pos = val / scale
        self.spin_pos_z.blockSignals(True)
        self.spin_pos_z.setValue(pos)
        self.spin_pos_z.blockSignals(False)
        self._on_section_z_changed(from_user_seek=True, is_preview=True)
        self._sec_commit_timer.start(120)

    def _on_spin_pos_z_changed(self, pos: float):
        scale = getattr(self, "_sec_pos_scale", 100.0)
        self.slider_pos_z.blockSignals(True)
        self.slider_pos_z.setValue(int(round(pos * scale)))
        self.slider_pos_z.blockSignals(False)
        self._on_section_z_changed(from_user_seek=True, is_preview=False)

    def _on_slider_thick_z_changed(self, val: int):
        thick = val / 10.0
        self.spin_thick_z.blockSignals(True)
        self.spin_thick_z.setValue(thick)
        self.spin_thick_z.blockSignals(False)
        self._on_section_z_changed(from_user_seek=False, is_preview=True)
        self._sec_commit_timer.start(120)

    def _on_spin_thick_z_changed(self, thick: float):
        self.slider_thick_z.blockSignals(True)
        self.slider_thick_z.setValue(max(self.slider_thick_z.minimum(), min(self.slider_thick_z.maximum(), int(round(thick * 10.0)))))
        self.slider_thick_z.blockSignals(False)
        self._on_section_z_changed(from_user_seek=False, is_preview=False)

    def _on_section_z_changed(self, from_user_seek: bool = False, is_preview: bool = False):
        enabled = self.chk_sec_z.isChecked()
        pos = self.spin_pos_z.value()
        thick = self.spin_thick_z.value()
        min_v, max_v, desc = self._compute_section_interval(pos, thick)
        self.lbl_sec_z.setText(desc)
        self.viewer.set_section_z(enabled, min_v, max_v, origin_val=pos, is_preview=is_preview)
        if from_user_seek and enabled and self.chk_sec_follow_cam.isChecked():
            self.viewer.focus_camera_on_world_axis(2, pos)

    def _reset_all_sections(self):
        """XYZおよび斜め壁すべての断面スライスを解除して全域表示に戻す"""
        self.chk_sec_x.setChecked(False)
        self.chk_sec_y.setChecked(False)
        self.chk_sec_z.setChecked(False)
        self.chk_sec_wall.setChecked(False)
        self.viewer.clear_wall_alignment()
        self.lbl_sec_wall.setText("未ピック(Uで2点指定)")
        self.status_label.setText("すべての断面スライス・2点壁基準を解除しました")

    def _crop_outside_section(self):
        """現在有効なXYZ断面ボックス・斜め壁スライスの外側にある未ロック有効点群を一括でノイズレイヤーへ移動"""
        if self.viewer.points_raw is None or len(self.viewer.points_raw) == 0:
            return
        if not (
            self.chk_sec_x.isChecked()
            or self.chk_sec_y.isChecked()
            or self.chk_sec_z.isChecked()
            or self.chk_sec_wall.isChecked()
        ):
            QMessageBox.information(
                self,
                "断面ボックス外クロップ",
                "先にX・Y・Zまたは斜め壁スライスのいずれかを有効にして、残したい範囲を枠線で囲んでください。",
            )
            return

        outside_idx = self.viewer.get_outside_section_normal_indices()
        if len(outside_idx) == 0:
            QMessageBox.information(self, "情報", "断面ボックスの外側に移動対象の未ロック有効点群はありません。")
            return

        self._push_layer_undo_state()
        self.viewer.move_indices_to_noise(outside_idx)
        self.log_text.append(f"断面ボックス外クロップ: {len(outside_idx):,} 点をノイズレイヤーへ一括移動しました")
        self.status_label.setText(f"断面ボックス外の {len(outside_idx):,} 点をノイズへ移動しました (Ctrl+Zで取消可)")

    def _invert_selection(self):
        """現在表示中の範囲内で選択状態を反転"""
        self.viewer.invert_selection()

    def _on_point_shape_changed(self, index: int):
        self.viewer.set_point_shape(self.shape_combo.currentData())

    def _on_budget_changed(self, index: int):
        self.viewer.set_point_budget(self.budget_combo.currentData())

    def _on_bg_color_changed(self, index: int):
        self.viewer.set_bg_color(self.bg_combo.currentData())

    def _on_input_file_changed(self, text: str):
        path = text.strip()
        if not path or not os.path.exists(path):
            return
        if os.path.isdir(path):
            e57_files = collect_e57_files(path)
            if not e57_files:
                self.status_label.setText("指定フォルダ内に .e57 ファイルが見つかりません")
                QMessageBox.warning(
                    self,
                    "フォルダ読み込み",
                    f"指定されたフォルダ内に E57 ファイル (*.e57) が見つかりませんでした:\n{path}",
                )
                return
            norm_dir = os.path.normpath(path)
            suggested_out = f"{norm_dir}_cleaned.e57"
            if not self.output_edit.text() or self.output_edit.text().endswith("_cleaned.e57"):
                self.output_edit.setText(suggested_out)
            self._load_file_to_viewer(norm_dir)
        elif os.path.isfile(path):
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

    def _browse_input_folder(self):
        """フォルダを選択して、フォルダ内の全E57ファイルを一括で統合読み込みする"""
        folder = QFileDialog.getExistingDirectory(
            self,
            "E57 点群が含まれるフォルダの選択",
            "",
        )
        if folder:
            self.input_edit.setText(folder)

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
            if os.path.isdir(path) or path.lower().endswith(".e57"):
                self.input_edit.setText(path)

    def _load_file_to_viewer(self, file_path: str):
        """E57ファイルまたはフォルダ内の全E57をバックグラウンドスレッドで完全非同期読み込みし、画面中央に進捗モーダルを表示"""
        is_dir = os.path.isdir(file_path)
        fname = os.path.basename(os.path.normpath(file_path))
        if is_dir:
            n_files = len(collect_e57_files(file_path))
            self.status_label.setText(f"フォルダ一括読み込み中: {fname} ({n_files} ファイル)")
            self.log_text.append(f"--- フォルダ一括読み込み開始: {fname} ({n_files} 個のE57ファイルを統合) ---")
            sub_init = f"フォルダ: {fname} ({n_files} ファイル)"
        else:
            self.status_label.setText(f"点群読み込み中: {fname} (0%)")
            self.log_text.append(f"--- 点群読み込み開始: {fname} ---")
            sub_init = f"ファイル: {fname}"
        self.progress_bar.setValue(0)

        # 既存ワーカーの終了待ち
        if self.load_worker and self.load_worker.isRunning():
            self.load_worker.terminate()
            self.load_worker.wait()

        # 画面中央の進捗モーダルダイアログを表示
        if self.progress_dialog:
            self.progress_dialog.close()
        self.progress_dialog = ModernProgressDialog("点群データ読み込み中", self)
        self.progress_dialog.set_progress(0, "読み込み準備中...", sub_init)
        self.progress_dialog.show()

        budget = self.budget_combo.currentData() if hasattr(self, "budget_combo") else 2000000

        # 大容量点群保護: PCの空きRAMに合わせて安全ロード上限を決定
        max_pts_setting = self.combo_load_budget.currentData() if hasattr(self, "combo_load_budget") else -1
        if max_pts_setting == -1:
            ram_gb = get_system_ram_gb()
            if ram_gb >= 60.0:
                eff_max_load = 120000000
            elif ram_gb >= 28.0:
                eff_max_load = 60000000  # 32GB RAM PC向け: 6,000万点
            else:
                eff_max_load = 30000000  # 16GB RAM PC向け: 3,000万点
        elif max_pts_setting > 0:
            eff_max_load = max_pts_setting
        else:
            eff_max_load = None  # 無制限全点ロード

        self.load_worker = FileLoadWorker(file_path, point_budget=budget, max_load_points=eff_max_load)

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
            self.layer_redo_stack.clear()
            self._sync_undo_buttons()

            # チェックボックス表示と断面スライス状態をリセット
            self.chk_layer_normal.setChecked(True)
            self.chk_layer_noise.setChecked(False)
            self.chk_sec_x.setChecked(False)
            self.chk_sec_y.setChecked(False)
            self.chk_sec_z.setChecked(False)
            if hasattr(self, "chk_sec_wall"):
                self.chk_sec_wall.setChecked(False)
            self.viewer.clear_wall_alignment()
            if hasattr(self, "lbl_sec_wall"):
                self.lbl_sec_wall.setText("未ピック(Uで2点指定)")

            # 0ミリ秒で高速バインド (UIフリーズなし)
            self.viewer.set_preloaded_data(preloaded, reset_camera=True)

            # 初期配色モードの自動適用
            def_mode = getattr(self.viewer, "default_color_mode", PointCloudViewer.COLOR_MODE_RGB)
            if hasattr(self, "color_combo") and def_mode in PointCloudViewer.ALL_COLOR_MODES:
                idx = self.color_combo.findText(def_mode)
                if idx >= 0:
                    self.color_combo.setCurrentIndex(idx)
                    self.viewer.set_color_mode(def_mode)

            total_pts = len(preloaded.points_raw)
            orig_total = getattr(preloaded, "original_total_pts", total_pts)
            is_down = getattr(preloaded, "is_downsampled", False)

            self.progress_bar.setValue(100)
            if is_down:
                self.point_count_label.setText(f"表示: {total_pts:,} 点 (元: {orig_total:,} 点)")
                self.status_label.setText(f"大容量保護読み込み完了: {len(preloaded.scans)} スキャン (元: {orig_total:,} 点中 {total_pts:,} 点を表示)")
                self.log_text.append(f"大容量メモリ保護適用: 元 {orig_total:,} 点から {total_pts:,} 点をサンプリング表示 (メモリ圧迫・スワップ防止)")
                self.show_toast(f"⚡ 大容量保護: {total_pts:,} 点を高速表示 (メモリ圧迫防止)")
            else:
                self.point_count_label.setText(f"点数: {total_pts:,} 点")
                self.status_label.setText(f"読み込み完了: {len(preloaded.scans)} スキャン / 計 {total_pts:,} 点")
                self.log_text.append(f"読み込み成功: スキャン(器械点)数 {len(preloaded.scans)}, 総点数 {total_pts:,} 点")
                self.show_toast(f"📂 {len(preloaded.scans)} スキャン ({total_pts:,} 点) を読み込みました")

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
        """現在のスキャンリストから点群データとスキャンIDをまとめてビューアに転送"""
        if not self.current_scans:
            self.viewer.set_point_cloud(None)
            self.point_count_label.setText("点数: 0 点")
            return

        all_pts = []
        all_scan_ids = []
        has_any_color = any(
            "colorRed" in s.raw_fields and "colorGreen" in s.raw_fields and "colorBlue" in s.raw_fields
            for s in self.current_scans if s.point_count > 0
        )
        has_any_int = any("intensity" in s.raw_fields for s in self.current_scans if s.point_count > 0)

        all_cols = []
        all_ints = []

        for s_idx, s in enumerate(self.current_scans):
            if s.point_count == 0:
                continue
            all_pts.append(s.points)
            all_scan_ids.append(np.full(s.point_count, s_idx, dtype=np.int32))
            if has_any_color:
                if "colorRed" in s.raw_fields and "colorGreen" in s.raw_fields and "colorBlue" in s.raw_fields:
                    r = np.asarray(s.raw_fields["colorRed"], dtype=np.float32)
                    g = np.asarray(s.raw_fields["colorGreen"], dtype=np.float32)
                    b = np.asarray(s.raw_fields["colorBlue"], dtype=np.float32)
                    c = np.column_stack([r, g, b])
                    max_c = float(np.max(c)) if len(c) > 0 else 1.0
                    if max_c > 255.0:
                        c = c / 65535.0
                    elif max_c > 1.0:
                        c = c / 255.0
                    all_cols.append(np.clip(c, 0.0, 1.0))
                else:
                    all_cols.append(np.tile([0.3, 0.8, 0.95], (s.point_count, 1)).astype(np.float32))
            if has_any_int:
                if "intensity" in s.raw_fields:
                    all_ints.append(np.asarray(s.raw_fields["intensity"], dtype=np.float32))
                else:
                    all_ints.append(np.zeros(s.point_count, dtype=np.float32))

        combined_pts = np.vstack(all_pts) if all_pts else np.empty((0, 3), dtype=np.float64)
        combined_cols = np.vstack(all_cols) if all_cols else None
        combined_ints = np.concatenate(all_ints) if all_ints else None
        combined_scan_ids = np.concatenate(all_scan_ids) if all_scan_ids else None

        self.viewer.set_point_cloud(
            combined_pts, combined_cols, combined_ints, reset_camera=True, scan_ids=combined_scan_ids
        )
        self.point_count_label.setText(f"点数: {len(combined_pts):,} 点")

        # 初期配色モードの自動適用
        def_mode = getattr(self.viewer, "default_color_mode", PointCloudViewer.COLOR_MODE_RGB)
        if hasattr(self, "color_combo") and def_mode in PointCloudViewer.ALL_COLOR_MODES:
            idx = self.color_combo.findText(def_mode)
            if idx >= 0:
                self.color_combo.setCurrentIndex(idx)
                self.viewer.set_color_mode(def_mode)

    def _on_color_mode_changed(self, mode_name: str):
        self.viewer.set_color_mode(mode_name)
        if hasattr(self, "settings_palette_combo") and hasattr(self.viewer, "color_palette"):
            idx = self.settings_palette_combo.findData(self.viewer.color_palette)
            if idx >= 0:
                self.settings_palette_combo.blockSignals(True)
                self.settings_palette_combo.setCurrentIndex(idx)
                self.settings_palette_combo.blockSignals(False)
        self.show_toast(f"🎨 配色切替: {mode_name}")

    def _on_color_palette_changed(self, index: int):
        pal_key = self.settings_palette_combo.itemData(index) if hasattr(self, "settings_palette_combo") else "turbo"
        if not pal_key:
            pal_key = PointCloudViewer.PALETTE_TURBO

        self.viewer.set_color_palette(pal_key)

        # color_combo 側の選択肢とも同期
        pal_to_mode = {
            PointCloudViewer.PALETTE_TURBO: PointCloudViewer.COLOR_MODE_HEIGHT,
            PointCloudViewer.PALETTE_VIRIDIS: PointCloudViewer.COLOR_MODE_HEIGHT_VIRIDIS,
            PointCloudViewer.PALETTE_JET: PointCloudViewer.COLOR_MODE_HEIGHT_JET,
            PointCloudViewer.PALETTE_PLASMA: PointCloudViewer.COLOR_MODE_HEIGHT_PLASMA,
            PointCloudViewer.PALETTE_TERRAIN: PointCloudViewer.COLOR_MODE_HEIGHT_TERRAIN,
            PointCloudViewer.PALETTE_HEAT: PointCloudViewer.COLOR_MODE_HEIGHT_HEAT,
            PointCloudViewer.PALETTE_COOLWARM: PointCloudViewer.COLOR_MODE_HEIGHT_COOLWARM,
            PointCloudViewer.PALETTE_GRAYSCALE: PointCloudViewer.COLOR_MODE_INTENSITY,
        }
        target_mode = pal_to_mode.get(pal_key)
        if target_mode and hasattr(self, "color_combo"):
            idx = self.color_combo.findText(target_mode)
            if idx >= 0:
                self.color_combo.blockSignals(True)
                self.color_combo.setCurrentIndex(idx)
                self.color_combo.blockSignals(False)

        pal_name = PointCloudViewer.PALETTE_LABELS.get(pal_key, pal_key)
        self.show_toast(f"🎨 色分けパレット: {pal_name}")

    def _toggle_selection_mode(self, enabled: bool):
        self.viewer.selection_mode = enabled
        if enabled:
            if self.viewer.measure_mode:
                self._toggle_measure_mode(False)
            shape_str = self.select_shape_combo.currentText()
            self.select_mode_btn.setText(f"選択: ON ({shape_str})")
            self.select_mode_btn.setStyleSheet("background-color: #0078d4; font-size: 11px; font-weight: bold; padding: 5px 8px;")
        else:
            self.select_mode_btn.setText("手動選択モード")
            self.select_mode_btn.setStyleSheet("font-size: 11px; padding: 5px 8px;")
            self.viewer.polygon_points.clear()
            self.viewer.update()
        self._sync_segment_buttons()

    def _update_floating_selection_bar(self, count: int):
        """選択点数に応じて3Dビュー下部アクションバーの表示・ボタン有効化を更新"""
        if not hasattr(self, "floating_sel_badge"):
            return
        has_sel = count > 0
        self.floating_del_btn.setEnabled(has_sel)
        if hasattr(self, "floating_wand_btn"):
            self.floating_wand_btn.setEnabled(has_sel)
        if hasattr(self, "floating_local_clean_btn"):
            self.floating_local_clean_btn.setEnabled(has_sel)
        self.floating_move_btn.setEnabled(has_sel)
        self.floating_restore_btn.setEnabled(has_sel)
        self.floating_clear_btn.setEnabled(has_sel)
        if has_sel:
            self.floating_sel_badge.setText(f"✨ {count:,} 点を選択中:")
            self.floating_sel_badge.setStyleSheet(
                "color: #fde047; background-color: #422006; border: 1px solid #eab308; border-radius: 8px; padding: 2px 8px; font-size: 11px; font-weight: bold;"
            )
        else:
            self.floating_sel_badge.setText("未選択 (ブラシ/矩形/円/多角形で点を囲むとここで即操作できます)")
            self.floating_sel_badge.setStyleSheet("color: #94a3b8; font-size: 11px; font-weight: bold;")

    def _apply_filter_preset(self, preset_key: str):
        """🎚️ ワンクリック・プリセット (大まか粗取り / 標準バランス / 細かい精密仕上げ) を自動設定"""
        if preset_key == "rough":
            # 🟢 大まか(粗取り): 浮遊塊・飛び石を一掃
            self.sor_chk.setChecked(True)
            self.sor_k_spin.setValue(25)
            self.sor_r_spin.setValue(2.2)
            self.ror_chk.setChecked(False)
            self.cluster_chk.setChecked(True)
            self.cluster_eps_spin.setValue(0.25)
            self.cluster_pts_spin.setValue(45)
            self.edge_veil_chk.setChecked(False)
            if hasattr(self, "filter_tab_widget"):
                self.filter_tab_widget.setCurrentIndex(2)
            msg = "🟢 プリセット「大まか(粗取り)」を設定: 浮遊塊除去(eps=0.25m, 45点) + SOR(k=25, 倍率2.2)"
        elif preset_key == "fine":
            # 🟠 細かい(精密仕上げ): 表面微細ヒゲ & エッジ飛散ベールノイズを緻密除去
            self.sor_chk.setChecked(True)
            self.sor_k_spin.setValue(16)
            self.sor_r_spin.setValue(1.3)
            self.ror_chk.setChecked(True)
            self.ror_radius_spin.setValue(0.03)
            self.ror_pts_spin.setValue(8)
            self.cluster_chk.setChecked(False)
            self.edge_veil_chk.setChecked(True)
            self.edge_k_spin.setValue(16)
            self.edge_lin_spin.setValue(0.70)
            if hasattr(self, "filter_tab_widget"):
                self.filter_tab_widget.setCurrentIndex(3)
            msg = "🟠 プリセット「細かい(精密仕上げ)」を設定: エッジ飛散・ヒゲ除去 + 高感度SOR(1.3) + 狭半径ROR"
        else:
            # 🟡 標準バランス
            self.sor_chk.setChecked(True)
            self.sor_k_spin.setValue(20)
            self.sor_r_spin.setValue(1.8)
            self.ror_chk.setChecked(False)
            self.cluster_chk.setChecked(True)
            self.cluster_eps_spin.setValue(0.15)
            self.cluster_pts_spin.setValue(25)
            self.edge_veil_chk.setChecked(False)
            if hasattr(self, "filter_tab_widget"):
                self.filter_tab_widget.setCurrentIndex(0)
            msg = "🟡 プリセット「標準バランス」を設定: SOR(k=20, 倍率1.8) + 孤立クラスタ(eps=0.15m, 25点)"

        self.log_text.append(msg)
        self.status_label.setText(msg)
        self.show_toast(msg)

    def _on_magic_wand_cluster_select(self):
        """🪄 マジックワンド: 現在選択されている点をシードとして3D空間で繋がっている塊全体を一括選択 (または即時削除)"""
        if self.viewer.points_raw is None or len(self.viewer.points_raw) == 0:
            return
        seed_orig = self.viewer.selected_indices
        if seed_orig is None or len(seed_orig) == 0:
            QMessageBox.information(
                self,
                "🪄 連結塊の一括選択 (マジックワンド)",
                "先に消したい塊（浮遊ゴミ・人・車・看板など）の一部をブラシや矩形で少しだけ選択してください。\n"
                "そこから3D空間で繋がっている塊全体を自動追跡して丸ごと選択します。",
            )
            return

        selectable_mask = self.viewer._get_selectable_mask()
        cand_orig = np.where(selectable_mask)[0]
        if len(cand_orig) == 0:
            return

        # cand_orig 内でのシード位置を特定
        seed_in_cand = np.where(np.isin(cand_orig, seed_orig))[0]
        if len(seed_in_cand) == 0:
            return

        eps = float(self.cluster_eps_spin.value()) if hasattr(self, "cluster_eps_spin") else 0.18
        cluster_local = extract_connected_cluster_from_seeds(
            self.viewer.points_raw[cand_orig],
            seed_in_cand,
            eps=max(0.05, eps),
            max_cluster_ratio=0.45,
        )
        cluster_orig = cand_orig[cluster_local]
        if len(cluster_orig) == 0:
            return

        if self.viewer.instant_delete_mode:
            self._push_layer_undo_state()
            moved = self.viewer.move_indices_to_layer(cluster_orig, PointCloudViewer.LAYER_NOISE, respect_lock=True)
            self._clear_selection()
            self.log_text.append(f"🪄 連結塊の即時消去: シード {len(seed_orig):,} 点から塊全体 {moved:,} 点をノイズへ移動")
            self.status_label.setText(f"🪄 連結塊 {moved:,} 点を丸ごと即消去しました (Ctrl+Zで取消可)")
            self.show_toast(f"🪄 連結塊 {moved:,} 点を丸ごと即消去しました")
        else:
            self.viewer.selected_indices = np.asarray(cluster_orig, dtype=np.int64)
            self.viewer._update_active_colors()
            self.viewer._update_display_arrays()
            self.viewer.update()
            self.viewer.point_selected.emit(self.viewer.selected_indices.tolist())
            self.log_text.append(f"🪄 連結塊選択: シード {len(seed_orig):,} 点から同一塊 {len(cluster_orig):,} 点を一括選択しました")
            self.status_label.setText(f"🪄 繋がっている塊 {len(cluster_orig):,} 点を一括選択しました (Delで即削除)")
            self.show_toast(f"🪄 連結塊 {len(cluster_orig):,} 点を一括選択しました")

    def _on_local_fine_clean_selected(self):
        """📦 選択範囲内のみ局所精密ノイズ除去: 選択中の点群だけに細かいSOR+エッジ飛散ヒゲ除去を適用"""
        if self.viewer.points_raw is None or len(self.viewer.points_raw) == 0:
            return
        sel_orig = self.viewer.selected_indices
        if sel_orig is None or len(sel_orig) < 5:
            QMessageBox.information(
                self,
                "📦 選択範囲内のみ精密除去",
                "先に細かくノイズを除去したいエリアをブラシ・矩形・円・多角形で選択してください（5点以上）。\n"
                "選択範囲外の構造ディテールを一切損なわずに、選択エリア内のヒゲや飛び点だけを除去します。",
            )
            return

        sub_pts = self.viewer.points_raw[sel_orig]
        keep_edge = filter_edge_veil_noise(
            sub_pts,
            nb_neighbors=min(16, max(4, len(sub_pts) - 1)),
            linearity_threshold=float(self.edge_lin_spin.value()) if hasattr(self, "edge_lin_spin") else 0.70,
            sor_std_ratio=1.25,
        )
        keep_mask_local = np.zeros(len(sel_orig), dtype=bool)
        keep_mask_local[keep_edge] = True
        noise_orig = sel_orig[~keep_mask_local]

        if len(noise_orig) == 0:
            self.status_label.setText("📦 選択範囲内に除去すべき微細ノイズは見つかりませんでした")
            self.show_toast("📦 選択範囲内に微細ノイズはありませんでした")
            return

        self._push_layer_undo_state()
        moved = self.viewer.move_indices_to_layer(noise_orig, PointCloudViewer.LAYER_NOISE, respect_lock=True)
        self._clear_selection()
        self.log_text.append(
            f"📦 選択範囲内の局所精密除去: 選択 {len(sel_orig):,} 点中 {moved:,} 点の微細ノイズを除去しました"
        )
        self.status_label.setText(f"📦 選択範囲内から微細ノイズ {moved:,} 点を除去しました (Ctrl+Zで取消可)")
        self.show_toast(f"📦 選択内から微細ノイズ {moved:,} 点を除去しました")

    def _on_keep_main_cluster_only(self):
        """🏝️ メイン塊のみ残す (大まかな浮遊塊ノイズ一括除去): 最大連結クラスタ以外を一括でノイズレイヤーへ分離"""
        if self.viewer.points_raw is None or len(self.viewer.points_raw) == 0:
            QMessageBox.information(self, "🏝️ メイン塊のみ残す", "先に点群データを読み込んでください。")
            return

        normal_indices = self.viewer.get_unlocked_normal_indices()
        if len(normal_indices) < 4:
            QMessageBox.information(self, "🏝️ メイン塊のみ残す", "対象となる未ロック有効点群がありません。")
            return

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self.status_label.setText("🏝️ メイン構造物を高速解析中 (ボクセルクラスタリング)...")
        QApplication.processEvents()
        try:
            eps = float(self.cluster_eps_spin.value()) if hasattr(self, "cluster_eps_spin") else 0.25
            keep_local = keep_largest_clusters(
                self.viewer.points_raw[normal_indices],
                eps=max(0.08, eps),
                top_k=1,
                min_keep_ratio=0.05,
            )
            keep_mask = np.zeros(len(normal_indices), dtype=bool)
            keep_mask[keep_local] = True
            remove_orig = normal_indices[~keep_mask]

            if len(remove_orig) == 0:
                self.status_label.setText("🏝️ 周囲に分離可能な浮遊塊ノイズはありませんでした")
                self.show_toast("🏝️ 周囲に浮遊塊ノイズはありませんでした")
                return

            self._push_layer_undo_state()
            layer_name = "自動除去: 浮遊塊一括"
            lid = next((k for k, v in self.viewer.layers.items() if v.name == layer_name), None)
            if lid is None:
                lid = self.viewer.add_custom_layer(layer_name, is_noise=True)
            moved = self.viewer.move_indices_to_layer(remove_orig, lid, respect_lock=True)
            self._sync_undo_buttons()

            self.log_text.append(
                f"🏝️ メイン塊のみ保持 (粗取り): 周囲の浮遊塊 {moved:,} 点を「{layer_name}」へ一括除去しました (残存: {len(normal_indices) - moved:,} 点)"
            )
            self.status_label.setText(f"🏝️ メイン塊以外の浮遊ノイズ {moved:,} 点を一括除去しました (Ctrl+Zで取消可)")
            self.show_toast(f"🏝️ 浮遊塊 {moved:,} 点を一括除去しました")
        finally:
            QApplication.restoreOverrideCursor()

    def _on_quick_edge_veil_clean(self):
        """🪶 表面・輪郭のエッジ飛散ヒゲ(ベールノイズ)をワンクリックで精密除去"""
        if self.viewer.points_raw is None or len(self.viewer.points_raw) == 0:
            QMessageBox.information(self, "🪶 表面・輪郭ヒゲ除去", "先に点群データを読み込んでください。")
            return

        normal_indices = self.viewer.get_unlocked_normal_indices()
        if len(normal_indices) < 6:
            return

        k_nb = int(self.edge_k_spin.value()) if hasattr(self, "edge_k_spin") else 16
        lin_th = float(self.edge_lin_spin.value()) if hasattr(self, "edge_lin_spin") else 0.72

        keep_local = filter_edge_veil_noise(
            self.viewer.points_raw[normal_indices],
            nb_neighbors=k_nb,
            linearity_threshold=lin_th,
            sor_std_ratio=1.25,
        )
        keep_mask = np.zeros(len(normal_indices), dtype=bool)
        keep_mask[keep_local] = True
        remove_orig = normal_indices[~keep_mask]

        if len(remove_orig) == 0:
            self.status_label.setText("🪶 エッジ飛散・ヒゲノイズは検出されませんでした")
            self.show_toast("🪶 エッジ飛散・ヒゲは検出されませんでした")
            return

        self._push_layer_undo_state()
        layer_name = "自動除去: エッジ飛散"
        lid = next((k for k, v in self.viewer.layers.items() if v.name == layer_name), None)
        if lid is None:
            lid = self.viewer.add_custom_layer(layer_name, is_noise=True)
        moved = self.viewer.move_indices_to_layer(remove_orig, lid, respect_lock=True)
        self._sync_undo_buttons()

        self.log_text.append(
            f"🪶 エッジ飛散・ヒゲ精密除去: {moved:,} 点を「{layer_name}」へ分離しました"
        )
        self.status_label.setText(f"🪶 表面・輪郭ヒゲ {moved:,} 点を精密除去しました (Ctrl+Zで取消可)")
        self.show_toast(f"🪶 表面・輪郭ヒゲ {moved:,} 点を精密除去しました")

    def _clear_selection(self):
        if getattr(self.viewer, "wall_align_mode", False):
            self._toggle_wall_align_mode(False)
        self.viewer.clear_selection()
        self.manual_status_label.setText("選択を解除しました (Esc)")
        self._update_floating_selection_bar(0)
        self.viewer.setFocus()

    def _on_points_selected(self, indices: list):
        n = len(indices)
        self.manual_status_label.setText(
            f"選択中: {n:,} 点 | [Del]削除 / [選択点を移動]で別レイヤーへ"
        )
        self._update_floating_selection_bar(n)

    def _delete_selected_points(self):
        """選択されている点をノイズレイヤーへ移動する (手動削除)"""
        selected_idx = self.viewer.selected_indices
        if len(selected_idx) == 0 or self.viewer.layer_mask is None:
            return

        # Undoスタックに直前のレイヤースナップショットを退避
        self._push_layer_undo_state()

        del_count = len(selected_idx)
        # ノイズレイヤーへ移動
        self.viewer.move_indices_to_noise(selected_idx)
        self._clear_selection()

        self.log_text.append(f"手動削除: {del_count:,} 点をノイズレイヤーへ移動しました (Undo可能)")
        self.status_label.setText(f"{del_count:,} 点をノイズレイヤーへ移動しました")
        self.show_toast(f"🗑 {del_count:,} 点をノイズレイヤーへ移動しました (Ctrl+Zで取消可)")

    def _undo_action(self):
        """直前のレイヤー状態または編集を元に戻す (Ctrl+Z)"""
        if not self.layer_undo_stack or self.viewer.layer_mask is None:
            return

        self.layer_redo_stack.append(self._capture_layer_snapshot())
        prev_snapshot = self.layer_undo_stack.pop()
        self._sync_undo_buttons()
        self._clear_selection()

        self._restore_layer_snapshot(prev_snapshot)

        self.log_text.append("元に戻す (Undo) を実行しました")
        self.status_label.setText("操作を取り消しました (Ctrl+Yでやり直し可能)")
        self.show_toast("↩ 操作を元に戻しました (Undo)")

    def _redo_action(self):
        """取り消したレイヤー操作をやり直す (Ctrl+Y)"""
        if not self.layer_redo_stack or self.viewer.layer_mask is None:
            return

        self.layer_undo_stack.append(self._capture_layer_snapshot())
        next_snapshot = self.layer_redo_stack.pop()
        self._sync_undo_buttons()
        self._clear_selection()

        self._restore_layer_snapshot(next_snapshot)

        self.log_text.append("やり直し (Redo) を実行しました")
        self.status_label.setText("操作をやり直しました")
        self.show_toast("↪ 操作をやり直しました (Redo)")

    def _save_current_scans(self):
        """④ 選択された保存モード(有効レイヤー統合/表示中のみ/特定レイヤーのみ/ノイズのみ/全レイヤー分割)でE57保存"""
        if not self.current_scans or self.viewer.layer_mask is None:
            QMessageBox.warning(self, "警告", "保存する点群データがありません。")
            return

        self.viewer._ensure_layer_sync()
        export_mode = self.export_mode_combo.currentData() if hasattr(self, "export_mode_combo") else "normal"
        split_masks = None
        layer_mask = self.viewer.layer_mask

        if export_mode == "split_all":
            split_masks = self.viewer.get_all_layer_masks(include_empty=False, include_noise=False)
            if not split_masks:
                QMessageBox.warning(self, "警告", "分割出力可能な有効レイヤーがありません。")
                return
        elif export_mode == "visible":
            layer_mask = self.viewer._get_layer_and_scan_visibility_mask()
            if int(np.sum(layer_mask)) == 0:
                QMessageBox.warning(self, "警告", "現在表示ONの点数が 0 点のため保存できません。")
                return
        elif export_mode == "single_layer":
            target_id = self.target_layer_combo.currentData()
            if target_id is None:
                return
            layer_mask = (self.viewer.layer_ids == int(target_id))
            if int(np.sum(layer_mask)) == 0:
                QMessageBox.warning(self, "警告", "選択中の対象レイヤーに点が 0 点のため保存できません。")
                return
        elif export_mode == "noise_only":
            layer_mask = ~self.viewer.layer_mask
            if int(np.sum(layer_mask)) == 0:
                QMessageBox.warning(self, "警告", "ノイズレイヤーの点数が 0 点のため保存できません。")
                return
        else:
            layer_mask = self.viewer.layer_mask
            if int(np.sum(layer_mask)) == 0:
                QMessageBox.warning(self, "警告", "有効点群レイヤーの点数が 0 点のため保存できません。")
                return

        out_path = self.output_edit.text().strip()
        if not out_path:
            self._browse_output_file()
            out_path = self.output_edit.text().strip()
            if not out_path:
                return

        if not out_path.lower().endswith(".e57"):
            out_path += ".e57"
            self.output_edit.setText(out_path)

        # 実務安全保護①: 入力元のE57ファイルと同一パスへの直接上書き確認
        in_path = self.input_edit.text().strip() if hasattr(self, "input_edit") else ""
        if in_path and os.path.isfile(in_path) and not split_masks:
            if os.path.normcase(os.path.abspath(out_path)) == os.path.normcase(os.path.abspath(in_path)):
                reply = QMessageBox.question(
                    self,
                    "原本ファイルの上書き確認",
                    f"出力先が入力元のE57ファイルと同じパスに設定されています。\n\n"
                    f"対象ファイル:\n{out_path}\n\n"
                    f"元のファイルを直接上書きして保存してもよろしいですか？",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                if reply != QMessageBox.Yes:
                    return

        # 実務安全保護②: メモリ保護による間引き読み込みデータが含まれている場合の確認
        if any(getattr(s, "is_subsampled", False) for s in self.current_scans):
            orig_sum = sum(getattr(s, "original_point_count", s.point_count) for s in self.current_scans)
            cur_sum = sum(s.point_count for s in self.current_scans)
            reply = QMessageBox.question(
                self,
                "間引き読み込みデータの保存確認",
                f"現在読み込まれている点群は、メモリ保護のため元データ（{orig_sum:,} 点）から "
                f"{cur_sum:,} 点にサブサンプリング（間引き）されています。\n\n"
                f"このまま保存すると、間引かれた {cur_sum:,} 点のみが出力されます。保存を続行しますか？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        if self.save_worker and self.save_worker.isRunning():
            return

        self.save_manual_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("E57書き出し中 (元の座標精度・属性形式を維持して抽出)...")

        if self.progress_dialog:
            self.progress_dialog.close()
        self.progress_dialog = ModernProgressDialog("E57点群データ保存中", self)
        self.progress_dialog.set_progress(5, "保存データの準備中...", f"保存先: {os.path.basename(out_path)}")
        self.progress_dialog.show()

        self.save_worker = FileSaveWorker(out_path, self.current_scans, layer_mask, split_masks=split_masks)

        def on_save_progress(percent: int, msg: str, sub_msg: str):
            if self.progress_dialog:
                self.progress_dialog.set_progress(percent, msg, sub_msg)
            self.progress_bar.setValue(percent)
            self.status_label.setText(f"{msg} ({percent}%)")

        def on_save_success(total_clean_pts: int, total_noise_pts: int, saved_path: str):
            if self.progress_dialog:
                self.progress_dialog.set_progress(100, "保存完了！", f"{total_clean_pts:,} 点を出力")
                self.progress_dialog.close()
                self.progress_dialog = None
            self.save_manual_btn.setEnabled(True)
            self.progress_bar.setValue(100)
            self.log_text.append(
                f"保存完了: {saved_path} (出力点数: {total_clean_pts:,} 点, 除外: {total_noise_pts:,} 点)"
            )
            self.status_label.setText("保存完了")
            QMessageBox.information(
                self,
                "保存完了",
                f"点群データをE57ファイルに保存しました！\n\n"
                f"・出力点数: {total_clean_pts:,} 点\n"
                f"・除外点数: {total_noise_pts:,} 点\n"
                f"・保存先: {saved_path}\n\n"
                f"※元の座標値精度・属性フィールド・ポーズ情報はすべて完全に維持されています。",
            )

        def on_save_error(err_msg: str):
            if self.progress_dialog:
                self.progress_dialog.close()
                self.progress_dialog = None
            self.save_manual_btn.setEnabled(True)
            self.progress_bar.setValue(0)
            self.status_label.setText("保存失敗")
            self.log_text.append(f"保存エラー: {err_msg}")
            QMessageBox.critical(self, "保存エラー", f"保存中にエラーが発生しました:\n{err_msg}")

        self.save_worker.progress_changed.connect(on_save_progress)
        self.save_worker.finished_success.connect(on_save_success)
        self.save_worker.finished_error.connect(on_save_error)
        self.save_worker.start()

    def _run_auto_processing(self):
        """現在有効かつ未ロックのレイヤー点のみを対象に自動ノイズ処理を実行 (ロック中レイヤーと手動削除点は完全保護)"""
        if self.viewer.points_raw is None or len(self.viewer.points_raw) == 0:
            QMessageBox.warning(self, "エラー", "処理対象の点群が読み込まれていません。\n先にE57ファイルを開いてください。")
            return

        normal_indices = self.viewer.get_unlocked_normal_indices()
        if len(normal_indices) == 0:
            QMessageBox.warning(self, "エラー", "現在処理可能な未ロックの有効点群がありません。")
            return

        config = FilterConfig(
            use_sor=self.sor_chk.isChecked(),
            sor_neighbors=self.sor_k_spin.value(),
            sor_std_ratio=self.sor_r_spin.value(),
            use_ror=self.ror_chk.isChecked(),
            ror_radius=self.ror_radius_spin.value(),
            ror_min_points=self.ror_pts_spin.value(),
            use_cluster_filter=self.cluster_chk.isChecked(),
            cluster_eps=self.cluster_eps_spin.value(),
            cluster_min_points=self.cluster_pts_spin.value(),
            use_edge_veil_filter=self.edge_veil_chk.isChecked() if hasattr(self, "edge_veil_chk") else False,
            edge_neighbors=self.edge_k_spin.value() if hasattr(self, "edge_k_spin") else 16,
            edge_linearity_thresh=self.edge_lin_spin.value() if hasattr(self, "edge_lin_spin") else 0.72,
            use_distance_filter=self.dist_chk.isChecked(),
            min_distance=self.dist_min_spin.value(),
            max_distance=self.dist_max_spin.value(),
            use_intensity_filter=self.int_chk.isChecked(),
            min_intensity=self.int_min_spin.value(),
            max_intensity=self.int_max_spin.value(),
            use_voxel_downsample=self.voxel_chk.isChecked(),
            voxel_size=self.voxel_spin.value(),
        )

        if not (
            config.use_sor
            or config.use_ror
            or config.use_cluster_filter
            or config.use_edge_veil_filter
            or config.use_distance_filter
            or config.use_intensity_filter
            or config.use_voxel_downsample
        ):
            QMessageBox.warning(self, "警告", "少なくとも1つのフィルタを有効にしてください。")
            return

        self.auto_exec_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.log_text.append(f"--- 自動ノイズ処理開始: 未ロック有効レイヤー {len(normal_indices):,} 点対象 ---")

        # 進捗モーダルを表示
        if self.progress_dialog:
            self.progress_dialog.close()
        self.progress_dialog = ModernProgressDialog("自動ノイズフィルタ処理中", self)
        self.progress_dialog.set_progress(0, "フィルタ解析準備中...", f"対象: {len(normal_indices):,} 点")
        self.progress_dialog.show()

        target_points = self.viewer.points_raw[normal_indices]
        target_intensities = (
            self.viewer.intensities_raw[normal_indices]
            if self.viewer.intensities_raw is not None and len(self.viewer.intensities_raw) == len(self.viewer.points_raw)
            else None
        )
        self.memory_filter_worker = MemoryFilterWorker(
            target_points,
            config,
            intensities=target_intensities,
        )

        def on_progress(percent: int, msg: str, sub_msg: str):
            if self.progress_dialog:
                self.progress_dialog.set_progress(percent, msg, sub_msg)
            self.progress_bar.setValue(percent)
            self.status_label.setText(f"{msg} ({percent}%)")

        stage_results: list = []

        def on_stages(stages):
            stage_results.extend(stages)

        def on_cancelled():
            if self.progress_dialog:
                self.progress_dialog.close()
                self.progress_dialog = None
            self.auto_exec_btn.setEnabled(True)
            self.progress_bar.setValue(0)
            self.status_label.setText("自動ノイズ処理を中断しました")
            self.show_toast("⚠️ 自動ノイズ処理を安全に中断しました")

        def on_success(keep_idx, remove_idx):
            if self.progress_dialog:
                self.progress_dialog.set_progress(100, "処理完了！", f"{len(remove_idx):,} 点のノイズを分離")
                self.progress_dialog.close()
                self.progress_dialog = None

            self.auto_exec_btn.setEnabled(True)
            self.progress_bar.setValue(100)

            if len(stage_results) == 0 or len(remove_idx) == 0:
                self.status_label.setText("検出されたノイズはありませんでした")
                self.show_toast("✨ 検出ノイズ 0 点（除去対象なし）")
                return

            # 移動前にUndo状態を記録（1回の操作として元に戻せるようにする）
            self._push_layer_undo_state()

            # バッチ移動リストの構築
            moves = []
            detail_lines = []
            for fname, local_idx in stage_results:
                orig_idx = normal_indices[np.asarray(local_idx, dtype=np.int64)]
                layer_name = f"自動除去: {fname}"
                lid = next((k for k, v in self.viewer.layers.items() if v.name == layer_name), None)
                if lid is None:
                    lid = self.viewer.add_custom_layer(layer_name)
                moves.append((orig_idx, lid))
                detail_lines.append(f"・{layer_name}: {len(orig_idx):,} 点")

            # 1回のバッチ更新で一括レイヤー移動（画面再描画・VBO転送は最後に1度だけ実行）
            n_removed = self.viewer.batch_move_indices_to_layers(moves, respect_lock=True)
            self._sync_undo_buttons()

            n_remain = len(normal_indices) - n_removed
            ratio = (n_removed / len(normal_indices) * 100.0) if len(normal_indices) > 0 else 0.0

            res_msg = (
                f"自動ノイズ処理が完了しました！\n\n"
                f"・処理対象 (未ロック有効レイヤー): {len(normal_indices):,} 点\n"
                f"・残存有効点: {n_remain:,} 点\n"
                f"・検出ノイズ: {n_removed:,} 点 ({ratio:.2f}% 検出)\n"
                + ("\n【フィルタ別レイヤー】\n" + "\n".join(detail_lines) + "\n" if detail_lines else "")
                + f"\n※「◀ 1つ戻る (Ctrl+Z)」で自動除去を一括で取り消せます。"
                f"（🔒ロック中レイヤーと手動削除点はそのまま保護されます）"
            )
            self.log_text.append(f"自動ノイズ処理完了: {n_removed:,} 点を {len(detail_lines)} 個のフィルタ別レイヤーへ一括移動 (残存: {n_remain:,} 点)")
            self.status_label.setText(f"自動処理完了: ノイズ {n_removed:,} 点検出")
            self.show_toast(f"🪄 自動除去: {n_removed:,} 点を高速分離完了")
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

        # キャンセルハンドラを接続
        self.progress_dialog.cancelled.connect(self.memory_filter_worker.cancel)

        self.memory_filter_worker.progress_changed.connect(on_progress)
        self.memory_filter_worker.log_message.connect(self.log_text.append)
        self.memory_filter_worker.finished_stages.connect(on_stages)
        self.memory_filter_worker.finished_success.connect(on_success)
        self.memory_filter_worker.finished_error.connect(on_error)
        self.memory_filter_worker.cancelled.connect(on_cancelled)
        self.memory_filter_worker.start()

    def _on_viewer_status(self, msg: str):
        self.status_label.setText(msg)

