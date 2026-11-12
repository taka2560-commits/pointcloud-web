"""
PySide6 / OpenGLによる3D点群ビューアウィジェット
- 数学座標系: 上面視で上がY、右がX、高さがZ (Z-up) に厳密に準拠
- 見ている画面に対するWASDフライスルー移動 (Forward/Right/Upベクトル)
- 視点切替プリセット (上面: XY平面, 正面: XZ平面(高さZ), 側面: YZ平面(高さZ), 等角)
- XYZ別シーク断面スライサー (X/Y/Z独立クリッピング & 断面枠線可視化)
- 手動選択ツール: 矩形選択、円形選択、多角形(ポリゴン)選択
- 2点間寸法計測ツール、全体フィット、正射影/透視投影切替、表示バジェット
"""

import json
import math
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List, Dict
import numpy as np
import OpenGL.GL as gl

from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import (
    QMatrix4x4,
    QVector3D,
    QVector4D,
    QColor,
    QPainter,
    QPen,
    QBrush,
    QFont,
    QImage,
    QMouseEvent,
    QWheelEvent,
    QKeyEvent,
)
from PySide6.QtCore import Qt, QPoint, QRect, QTimer, Signal
from gui.measure_text import measure_clipboard_text, measure_distance_text, measure_hud_lines, measure_status_text


@dataclass
class LayerInfo:
    """レイヤー情報データクラス"""
    layer_id: int
    name: str
    visible: bool = True
    locked: bool = False
    color: Tuple[float, float, float] = (0.20, 0.82, 0.55)
    use_layer_color: bool = True
    is_noise: bool = False  # True: ノイズ(除去点)扱い (ID=0 または 自動除去フィルタレイヤー)


class PointCloudViewer(QOpenGLWidget):
    """3D点群ビューアクラス"""

    point_selected = Signal(list)       # 選択された点インデックスリスト
    status_changed = Signal(str)        # ステータスメッセージ
    measure_updated = Signal(str)       # 計測結果メッセージ
    MEASURE_PICK_MAX_PX = 12.0          # 計測で点を拾うときのクリック位置からの許容距離 [px]
    MEASURE_SNAP_NOTICE_PX = 3.0        # これより離れた点を拾ったらステータスで知らせる [px]
    projection_changed = Signal(bool)   # 投影モード変更通知 (True: 正射影, False: 透視投影)
    bounds_ready = Signal(object, object)  # 点群のワールド範囲 (min_xyz, max_xyz)
    layers_updated = Signal(int, int)      # レイヤー点数更新 (有効レイヤー点数, ノイズ点数)
    layer_list_changed = Signal()          # マルチレイヤー構成・状態変更通知
    context_menu_requested = Signal(QPoint)  # 右クリック文脈メニュー表示要求 (グローバル座標)
    instant_erase_requested = Signal(object)  # 即時削除モード時の1アクション消去要求 (np.ndarray)
    brush_radius_changed = Signal(int)        # ブラシサイズ変更通知 (px)
    wall_aligned = Signal(float, object, object)  # 2点ピック斜め壁視点合わせ完了通知 (angle_deg, p1_world, p2_world)
    section_guides_changed = Signal(bool, bool)  # 断面ガイド枠表示変更通知 (show_origin_plane, show_box)

    COLOR_MODE_RGB = "RGB"
    COLOR_MODE_HEIGHT = "高さ (Z)"
    COLOR_MODE_HEIGHT_VIRIDIS = "高さ (Z: Viridis)"
    COLOR_MODE_HEIGHT_JET = "高さ (Z: Jet)"
    COLOR_MODE_HEIGHT_PLASMA = "高さ (Z: Plasma / 紫桃黄)"
    COLOR_MODE_HEIGHT_TERRAIN = "高さ (Z: 地形土木 / 緑茶白)"
    COLOR_MODE_HEIGHT_HEAT = "高さ (Z: Heat / 温熱鉄)"
    COLOR_MODE_HEIGHT_COOLWARM = "高さ (Z: Cool-Warm / 青白赤)"
    COLOR_MODE_INTENSITY = "反射強度"
    COLOR_MODE_INTENSITY_HEAT = "反射強度 (Heat / 温熱鉄)"
    COLOR_MODE_INTENSITY_PLASMA = "反射強度 (Plasma / 高コントラスト)"
    COLOR_MODE_INTENSITY_TURBO = "反射強度 (Turbo / レインボー)"
    COLOR_MODE_INTENSITY_VIRIDIS = "反射強度 (Viridis / 知覚均一)"
    COLOR_MODE_LAYER = "レイヤー別色分け"
    COLOR_MODE_SCAN = "スキャン別 (器械点)"
    COLOR_MODE_SOLID = "単色"

    # 全配色モード一覧 (コンボボックス用)
    ALL_COLOR_MODES = [
        COLOR_MODE_RGB,
        COLOR_MODE_HEIGHT,
        COLOR_MODE_HEIGHT_VIRIDIS,
        COLOR_MODE_HEIGHT_JET,
        COLOR_MODE_HEIGHT_PLASMA,
        COLOR_MODE_HEIGHT_TERRAIN,
        COLOR_MODE_HEIGHT_HEAT,
        COLOR_MODE_HEIGHT_COOLWARM,
        COLOR_MODE_INTENSITY,
        COLOR_MODE_INTENSITY_HEAT,
        COLOR_MODE_INTENSITY_PLASMA,
        COLOR_MODE_INTENSITY_TURBO,
        COLOR_MODE_INTENSITY_VIRIDIS,
        COLOR_MODE_LAYER,
        COLOR_MODE_SCAN,
        COLOR_MODE_SOLID,
    ]

    # カラーパレット定数 (高さ・反射強度の任意色分け)
    PALETTE_TURBO = "turbo"
    PALETTE_VIRIDIS = "viridis"
    PALETTE_PLASMA = "plasma"
    PALETTE_JET = "jet"
    PALETTE_TERRAIN = "terrain"
    PALETTE_HEAT = "heat"
    PALETTE_COOLWARM = "coolwarm"
    PALETTE_GRAYSCALE = "grayscale"

    PALETTE_LABELS = {
        PALETTE_TURBO: "Turbo (レインボー)",
        PALETTE_VIRIDIS: "Viridis (知覚均一・紫緑黄)",
        PALETTE_PLASMA: "Plasma (高コントラスト・紫桃黄)",
        PALETTE_JET: "Jet (クラシック虹色)",
        PALETTE_TERRAIN: "Terrain (地形土木・緑茶白)",
        PALETTE_HEAT: "Heat (温熱鉄・黒赤黄白)",
        PALETTE_COOLWARM: "Cool-Warm (青白赤)",
        PALETTE_GRAYSCALE: "Grayscale (白黒)",
    }

    # 標準レイヤーID定数
    LAYER_NOISE = 0
    LAYER_NORMAL = 1
    LAYER_GROUND = 2
    LAYER_PLANE = 3

    # カスタムレイヤー用デフォルトカラーパレット
    LAYER_PALETTE = [
        (0.55, 0.40, 0.22),  # ブラウン (地面)
        (0.20, 0.82, 0.55),  # エメラルド (平面/壁)
        (0.65, 0.35, 0.95),  # パープル (設備/配管)
        (0.95, 0.55, 0.15),  # オレンジ (構造物)
        (0.20, 0.85, 0.85),  # シアン
        (0.95, 0.35, 0.70),  # ピンク
    ]

    # スキャン(器械点)色分け用パレット
    SCAN_PALETTE = np.array([
        [0.22, 0.74, 0.97],
        [0.98, 0.55, 0.20],
        [0.25, 0.85, 0.45],
        [0.85, 0.35, 0.95],
        [0.95, 0.85, 0.20],
        [0.20, 0.90, 0.80],
        [0.95, 0.35, 0.55],
        [0.55, 0.85, 0.25],
    ], dtype=np.float32)

    # 選択形状
    # 断面ボックス内の全点表示の上限。この PC（内蔵 GPU）で 1,000 万点は約 130ms/フレーム、VBO 転送は約 3 秒
    SECTION_DETAIL_MAX_POINTS = 10_000_000

    SELECT_RECT = "rect"
    SELECT_CIRCLE = "circle"
    SELECT_POLYGON = "polygon"
    SELECT_BRUSH = "brush"

    # 1024段事前計算カラーマップLUTキャッシュ (クラス共通・超高速引き当て用)
    _COLORMAP_LUTS: Optional[Dict[str, np.ndarray]] = None

    @property
    def _vbo_dirty(self) -> bool:
        """後方互換用: 頂点座標またはカラーのいずれかが未同期ならTrue"""
        return getattr(self, "_vbo_points_dirty", True) or getattr(self, "_vbo_colors_dirty", True)

    @_vbo_dirty.setter
    def _vbo_dirty(self, val: bool):
        self._vbo_points_dirty = bool(val)
        self._vbo_colors_dirty = bool(val)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)

        # カメラパラメータ (数学座標系: 上面で右がX、上がY、高さがZ)
        self.camera_target = QVector3D(0.0, 0.0, 0.0)  # 注視点
        self.camera_distance = 15.0                     # 距離
        self.yaw = 45.0                                # 水平角 (度)
        self.pitch = 30.0                              # 仰角 (度: 0°=水平正面, 89.9°=真上見下ろし)
        self.fov = 45.0                                # 視野角 (度: 15°〜90°)
        self.depth_fog_enabled = False                 # 深度フォグ・立体感強調
        self.default_color_mode = self.COLOR_MODE_RGB  # 起動・読込時デフォルト配色
        self.is_orthographic = False                   # 正射影フラグ

        # 点群原本データ
        self.points_raw = None        # (N, 3) float64 原本
        self.points_centered = None   # (N, 3) float32 中心オフセット済み原本
        self.center_offset = np.zeros(3, dtype=np.float64)
        self.colors_raw = None        # (N, 3) float32
        self.intensities_raw = None   # (N,) float32
        self.active_colors = None     # (N, 3) float32 描画用カラー原本

        # カラー正規化キャッシュ (もっさり感解消: 毎回計算せず0ms参照)
        self._cached_normalized_heights: Optional[np.ndarray] = None
        self._cached_normalized_intensities: Optional[np.ndarray] = None
        self._cached_shading_factors: Optional[np.ndarray] = None
        self._cached_height_range: Optional[Tuple[float, float]] = None      # 高さ色分けの (z_min, z_max)
        self._cached_intensity_stretch: Optional[Tuple[float, float]] = None  # 反射強度の (p2, p98)

        # 3D表示強化設定 (① 凹凸・陰影強調 / ② 距離連動ポイントサイズ)
        self.shading_enabled: bool = True
        self.shading_strength: float = 0.65
        self.adaptive_point_size: bool = True

        # 表示用バジェット配列 (Point Budget対応) & OpenGL VBOキャッシュ
        self.display_points = None    # 描画対象 (M, 3) float32
        self.display_colors = None    # 描画対象カラー (M, 3) float32
        self.display_orig_indices = None  # 原本インデックスとのマッピング
        self._preview_base_points = None  # スライダープレビュー用ベース配列 (ヌルヌル追従用)
        self._preview_base_colors = None  # スライダープレビュー用ベース配列
        self.point_budget = 2000000   # 初期値: 200万点 (0で無制限)
        # 断面ボックス内の全点表示: 断面有効中はボックス内の点を表示予算で間引かず描く（上限 SECTION_DETAIL_MAX_POINTS）
        self.section_full_detail: bool = True
        self.section_box_count: int = 0          # 直近の確定再構築でボックス内にあった点数（断面無効時は 0）
        self.section_detail_capped: bool = False  # ボックス内が上限を超えて間引いたか
        self._vbo_points = None
        self._vbo_colors = None
        self._vbo_points_dirty = True
        self._vbo_colors_dirty = True
        self._vbo_count = 0
        # 断面ボックスの元ファイル読み直し（表示専用の重ね描き。core/section_reload.py の結果を set_section_overlay() で受け取る）
        # 現在の断面ボックスに紐づき、ボックスが変わる・断面を解除する・点群を入れ替えると自動で消える
        self.overlay_points: Optional[np.ndarray] = None   # (M, 3) float32 中心オフセット済み
        self.overlay_colors: Optional[np.ndarray] = None   # (M, 3) float32
        self._overlay_source = None                        # (points_centered, rgb, intensities, scan_ids) 配色変更時の再着色用
        self.overlay_box_total: int = 0                    # ボックス内にあった元ファイルの点数（間引き前）
        self.overlay_capped: bool = False                  # 上限で間引いたか
        self._overlay_box_key = None                       # 重ね描きを作ったときの断面ボックス
        self.overlay_kind: str = ""                        # "section"（断面の読み直し）/ "lod"（視点連動の高精細表示）
        self._vbo_overlay = None
        self._vbo_overlay_colors = None
        self._vbo_overlay_dirty = True
        self._vbo_overlay_count = 0

        # 表示設定
        self.point_size = 2.0
        self.point_shape = "circle"
        self.color_mode = self.COLOR_MODE_RGB
        self.color_palette = self.PALETTE_TURBO
        self.bg_color = QColor(20, 20, 26)
        self.show_grid = True
        self.bounds_min = np.array([-10, -10, -5], dtype=np.float32)
        self.bounds_max = np.array([10, 10, 5], dtype=np.float32)
        self.world_bounds_min = np.zeros(3, dtype=np.float64)
        self.world_bounds_max = np.zeros(3, dtype=np.float64)

        # 任意高さ色分けレンジ (高さスライダー連動)
        self.custom_height_enabled: bool = False
        self.custom_height_min: float = 0.0
        self.custom_height_max: float = 10.0

        # レイヤー管理 (マルチレイヤー + 通常/ノイズ後方互換マスク)
        self.layer_mask = None         # (N,) bool (True: 有効レイヤー[ID!=0], False: ノイズ[ID==0])
        self.layer_ids = None          # (N,) int32 (0: ノイズ, 1: 通常, 2+: カスタム/自動抽出レイヤー)
        self.layers: Dict[int, LayerInfo] = {}
        self._init_default_layers()
        self.show_normal_layer = True  # 通常レイヤー表示
        self.show_noise_layer = False  # ノイズレイヤー表示 (初期値: OFF)

        # スキャン(器械点/ステーション)別管理
        self.scan_ids = None           # (N,) int32
        self.scan_visibility: Dict[int, bool] = {}

        # XYZ断面シークスライサー設定 (シーク基点座標 + 断面範囲)
        self.clip_x_enabled = False
        self.clip_x_range = [-1e6, 1e6]
        self.clip_x_origin = 0.0
        self.clip_y_enabled = False
        self.clip_y_range = [-1e6, 1e6]
        self.clip_y_origin = 0.0
        self.clip_z_enabled = False
        self.clip_z_range = [-1e6, 1e6]
        self.clip_z_origin = 0.0
        self.show_section_origin_plane: bool = True  # シーク基点プレーン枠(黄)の表示
        self.show_section_box: bool = True           # 断面厚み範囲ボックス枠(水色)の表示

        # マウス操作用
        self.last_mouse_pos = QPoint()
        self.current_mouse_pos = QPoint()
        self.is_rotating = False
        self.is_panning = False
        self.pan_mode = False          # ✋ 左ドラッグ平行移動(パン)専用モード
        self.space_pan_active = False  # Spaceキー押下中の一時パンモード

        # 手動選択モード & ノイズ処理専用UX
        self.selection_mode = False
        self.selection_shape = self.SELECT_RECT
        self.selection_start = QPoint()
        self.selection_end = QPoint()
        self.is_selecting = False
        self.polygon_points: List[QPoint] = []
        self.brush_radius = 28
        self.brush_stroke_points: List[QPoint] = []
        self.instant_delete_mode = False
        self.clean_focus_mode = False
        self.selected_indices = np.array([], dtype=np.int64)
        self._last_selection_modifiers = Qt.NoModifier

        # 2点間寸法計測ツール
        self.measure_mode = False
        self.measure_points: List[Tuple[np.ndarray, np.ndarray]] = []
        self.measure_result_text = ""
        self.measure_coord_lines: List[str] = []   # HUD に出す座標・距離の行（gui/measure_text.py）
        self.coord_mode: str = "cartesian"         # 読み込み時の座標系変換（計測座標をファイル系でも表示するため）
        self.last_pick_px: Optional[float] = None  # 直近のピックのクリック位置からの距離 [px]

        # 🧭 2点ピック斜め壁アライン視点・直交スライス
        self.wall_align_mode = False
        self.wall_align_points: List[Tuple[np.ndarray, np.ndarray]] = []
        self.wall_align_angle_deg: Optional[float] = None
        self.wall_align_center_world: Optional[np.ndarray] = None
        self.wall_align_view_type: str = "horiz"  # "horiz"(上面横) / "vert"(上面縦) / "ortho_front"(壁正面) / "edge"(壁断面)
        self.clip_wall_enabled = False
        self.clip_wall_range = [-1.0, 1.0]
        self.clip_wall_origin = 0.0

        # マウス操作・カメラ感度設定
        self.rotate_speed: float = 0.5
        self.pan_speed_factor: float = 1.0
        self.zoom_speed_factor: float = 1.0
        self.wasd_speed_factor: float = 1.0
        self.invert_orbit_y: bool = False

        # ⑦ インタラクティブ軸ギズモ & スムーズ視点回転アニメーション
        self._gizmo_hit_targets: Dict[str, Tuple[float, float, float]] = {}
        self._hovered_gizmo_axis: Optional[str] = None
        self._cam_anim_timer = QTimer(self)
        self._cam_anim_timer.setInterval(16)
        self._cam_anim_timer.timeout.connect(self._step_camera_animation)
        self._cam_anim_start_pitch: float = 30.0
        self._cam_anim_start_yaw: float = 45.0
        self._cam_anim_target_pitch: float = 30.0
        self._cam_anim_target_yaw: float = 45.0
        self._cam_anim_step: int = 0
        self._cam_anim_total_steps: int = 10

        # ⑧ 高画質スクリーンショット撮影時のHUD一時非表示フラグ
        self._hide_hud_for_capture: bool = False
        # スクリーンショット設定 (文字類は常に除外、グリッド・凡例は任意設定)
        self.screenshot_include_grid: bool = False  # スクショ時にグリッドを含めるか
        self.screenshot_include_axes: bool = True   # スクショ時にXYZ凡例を含めるか
        self._is_capturing_screenshot: bool = False
        self._capture_override_grid: Optional[bool] = None
        self._capture_override_axes: Optional[bool] = None

    def initializeGL(self):
        """OpenGL初期化処理 & GPU環境ログ記録"""
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthFunc(gl.GL_LEQUAL)
        gl.glEnable(gl.GL_POINT_SMOOTH)
        gl.glHint(gl.GL_POINT_SMOOTH_HINT, gl.GL_NICEST)
        gl.glDisable(gl.GL_LIGHTING)
        try:
            from core.logger import log_info
            renderer = gl.glGetString(gl.GL_RENDERER)
            version = gl.glGetString(gl.GL_VERSION)
            vendor = gl.glGetString(gl.GL_VENDOR)
            r_str = renderer.decode('utf-8', errors='ignore') if isinstance(renderer, bytes) else str(renderer)
            v_str = version.decode('utf-8', errors='ignore') if isinstance(version, bytes) else str(version)
            ven_str = vendor.decode('utf-8', errors='ignore') if isinstance(vendor, bytes) else str(vendor)
            log_info(f"🖥️ OpenGL描画環境: GPU={r_str} | ベンダー={ven_str} | OpenGL={v_str}")
        except Exception:
            pass

    def resizeGL(self, width: int, height: int):
        """ウィンドウリサイズ処理"""
        dpr = self.devicePixelRatio()
        gl.glViewport(0, 0, int(width * dpr), int(height * dpr))

    def paintGL(self):
        """OpenGL描画ループ（高DPI・断面・数学座標系・ギズモ・VBO高速描画）"""
        dpr = self.devicePixelRatio()
        vp_w = int(self.width() * dpr)
        vp_h = int(self.height() * dpr)
        gl.glViewport(0, 0, vp_w, vp_h)

        bg = self.bg_color
        gl.glClearColor(bg.redF(), bg.greenF(), bg.blueF(), 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        aspect = self.width() / max(1, self.height())
        near_plane = max(0.1, self.camera_distance * 0.01)
        far_plane = max(500.0, self.camera_distance * 50.0)

        # 1. プロジェクション行列の設定
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        if self.is_orthographic:
            ortho_size = max(0.5, self.camera_distance * 0.35)
            gl.glOrtho(
                -ortho_size * aspect,
                ortho_size * aspect,
                -ortho_size,
                ortho_size,
                -far_plane,
                far_plane,
            )
        else:
            proj = QMatrix4x4()
            proj.perspective(self.fov, aspect, near_plane, far_plane)
            gl.glLoadMatrixf(proj.data())

        # 2. モデルビュー行列の設定 (数学座標系lookAt)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        view = self._get_view_matrix()
        gl.glLoadMatrixf(view.data())

        # 3. 地面グリッド描画 (XY平面, 高さZ基準)
        is_capture = getattr(self, "_is_capturing_screenshot", False)
        if is_capture:
            show_grid_now = getattr(self, "_capture_override_grid", False)
        else:
            show_grid_now = self.show_grid

        if show_grid_now:
            self._draw_grid()

        # 4. 点群の描画 (VBO GPUキャッシュ対応・フォールバック付き)
        if self.display_points is not None and len(self.display_points) > 0:
            if getattr(self, "depth_fog_enabled", False) and not self.is_orthographic:
                gl.glEnable(gl.GL_FOG)
                gl.glFogi(gl.GL_FOG_MODE, gl.GL_LINEAR)
                bg = self.bg_color
                gl.glFogfv(gl.GL_FOG_COLOR, [bg.redF(), bg.greenF(), bg.blueF(), 1.0])
                near_fog = max(0.5, self.camera_distance * 0.4)
                far_fog = max(5.0, self.camera_distance * 2.6)
                gl.glFogf(gl.GL_FOG_START, near_fog)
                gl.glFogf(gl.GL_FOG_END, far_fog)
            else:
                gl.glDisable(gl.GL_FOG)

            # ② カメラ距離に応じた点サイズ自動スケーリング (アダプティブ・ポイントサイズ)
            eff_pt_size = float(self.point_size)
            if getattr(self, "adaptive_point_size", False):
                span = float(np.linalg.norm(self.bounds_max - self.bounds_min)) if self.bounds_max is not None else 20.0
                ref_dist = max(2.0, span * 0.55)
                dist_ratio = ref_dist / max(0.2, float(self.camera_distance))
                scale = max(0.65, min(2.8, math.pow(dist_ratio, 0.42)))
                eff_pt_size = max(1.0, min(14.0, eff_pt_size * scale))
            gl.glPointSize(eff_pt_size)

            if self.point_shape == "circle":
                gl.glEnable(gl.GL_POINT_SMOOTH)
            else:
                gl.glDisable(gl.GL_POINT_SMOOTH)

            gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
            gl.glEnableClientState(gl.GL_COLOR_ARRAY)

            drawn_with_vbo = False
            try:
                if self._vbo_points is None:
                    self._vbo_points = gl.glGenBuffers(1)
                    self._vbo_colors = gl.glGenBuffers(1)
                    self._vbo_points_dirty = True
                    self._vbo_colors_dirty = True

                # 頂点座標VBO更新 (座標変更時のみ転送: カラーのみ変更時はスキップして超高速化)
                if self._vbo_points_dirty:
                    self._vbo_count = len(self.display_points)
                    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._vbo_points)
                    try:
                        gl.glBufferData(gl.GL_ARRAY_BUFFER, self.display_points.nbytes, self.display_points, gl.GL_DYNAMIC_DRAW)
                    except Exception:
                        pts_bytes = self.display_points.tobytes()
                        gl.glBufferData(gl.GL_ARRAY_BUFFER, len(pts_bytes), pts_bytes, gl.GL_DYNAMIC_DRAW)
                    self._vbo_points_dirty = False

                # カラーVBO更新 (カラー変更時のみ転送)
                if self._vbo_colors_dirty:
                    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._vbo_colors)
                    try:
                        gl.glBufferData(gl.GL_ARRAY_BUFFER, self.display_colors.nbytes, self.display_colors, gl.GL_DYNAMIC_DRAW)
                    except Exception:
                        cols_bytes = self.display_colors.tobytes()
                        gl.glBufferData(gl.GL_ARRAY_BUFFER, len(cols_bytes), cols_bytes, gl.GL_DYNAMIC_DRAW)
                    self._vbo_colors_dirty = False

                gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._vbo_points)
                gl.glVertexPointer(3, gl.GL_FLOAT, 0, None)
                gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._vbo_colors)
                gl.glColorPointer(3, gl.GL_FLOAT, 0, None)

                # ★ カメラ操作時アダプティブLOD: 回転・パン操作中は点数を半減させて滑らかな60fpsを維持
                render_count = self._vbo_count
                if (self.is_rotating or self.is_panning) and render_count > 800000:
                    render_count = max(400000, render_count // 2)

                gl.glDrawArrays(gl.GL_POINTS, 0, render_count)
                gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
                drawn_with_vbo = True
            except Exception:
                gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

            if not drawn_with_vbo:
                gl.glVertexPointer(3, gl.GL_FLOAT, 0, self.display_points)
                gl.glColorPointer(3, gl.GL_FLOAT, 0, self.display_colors)
                gl.glDrawArrays(gl.GL_POINTS, 0, len(self.display_points))

            # 4.5. 断面ボックスの元ファイル読み直し（表示専用の重ね描き）を同じ点サイズ・フォグで描く
            self._draw_section_overlay()

            gl.glDisableClientState(gl.GL_COLOR_ARRAY)
            gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
            gl.glDisable(gl.GL_FOG)

        # 5. 断面スライスボックス枠線描画
        self._draw_section_box()

        # 6. 計測ライン & 2点ピック斜め壁基準ラインの描画 (スクショ撮影時は除外)
        if not is_capture:
            if len(self.measure_points) > 0:
                self._draw_measure_lines()
            if len(self.wall_align_points) > 0:
                self._draw_wall_align_lines()

        # 7. XYZ座標軸ギズモ & 8. 2Dオーバーレイ描画
        if is_capture:
            # スクリーンショット撮影時: 文字類・HUD・案内テキストは完全に省き、XYZ凡例のみ設定に従う
            show_axes_now = getattr(self, "_capture_override_axes", True)
            if show_axes_now:
                self._draw_axes()
                painter = QPainter(self)
                painter.setRenderHint(QPainter.Antialiasing)
                self._draw_axes_labels_2d(painter)
                painter.end()
        else:
            if not getattr(self, "_hide_hud_for_capture", False):
                self._draw_axes()
                self._draw_2d_overlays()

    def _get_view_matrix(self) -> QMatrix4x4:
        """
        数学座標系（上がY、右がX、高さがZ）に完全準拠したビュー行列
        - pitch = 89.9°: 上面 (Top: 画面右がX、画面上がY、高さZを見下ろし)
        - pitch = 0.0°, yaw = 0.0°: 正面 (Front: 画面右がX、画面上がZ(高さ)、奥がY)
        - pitch = 0.0°, yaw = -90.0°: 側面 (Side: 画面右がY、画面上がZ(高さ))
        - pitch = 30.0°, yaw = 45.0°: 等角 (Iso: 3D鳥瞰)
        """
        rad_pitch = math.radians(self.pitch)
        rad_yaw = math.radians(self.yaw)

        # カメラの相対位置
        xcam = self.camera_distance * math.cos(rad_pitch) * math.sin(rad_yaw)
        ycam = -self.camera_distance * math.cos(rad_pitch) * math.cos(rad_yaw)
        zcam = self.camera_distance * math.sin(rad_pitch)

        eye = self.camera_target + QVector3D(xcam, ycam, zcam)
        target = self.camera_target

        # 視線ベクトル (Target - Eye)
        v_view = np.array([-xcam, -ycam, -zcam], dtype=np.float32) / max(1e-6, self.camera_distance)

        # 画面右方向ベクトル (水平面上で画面右)
        r = np.array([math.cos(rad_yaw), math.sin(rad_yaw), 0.0], dtype=np.float32)

        # 画面上方向ベクトル = R x V_view
        u = np.cross(r, v_view)
        norm_u = np.linalg.norm(u)
        if norm_u < 1e-5:
            u = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        else:
            u = u / norm_u

        up = QVector3D(float(u[0]), float(u[1]), float(u[2]))

        view = QMatrix4x4()
        view.lookAt(eye, target, up)
        return view

    def _get_camera_vectors(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        見ている画面に対する基準ベクトル (Forward, Right, Up)
        - Forward: 画面の奥（視線方向）
        - Right: 画面の右
        - Up: 画面の上
        """
        rad_pitch = math.radians(self.pitch)
        rad_yaw = math.radians(self.yaw)

        xcam = self.camera_distance * math.cos(rad_pitch) * math.sin(rad_yaw)
        ycam = -self.camera_distance * math.cos(rad_pitch) * math.cos(rad_yaw)
        zcam = self.camera_distance * math.sin(rad_pitch)

        forward = np.array([-xcam, -ycam, -zcam], dtype=np.float32) / max(1e-6, self.camera_distance)
        right = np.array([math.cos(rad_yaw), math.sin(rad_yaw), 0.0], dtype=np.float32)
        up = np.cross(right, forward)
        norm_u = np.linalg.norm(up)
        if norm_u < 1e-5:
            up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        else:
            up = up / norm_u

        return forward, right, up

    def _draw_grid(self):
        """数学座標系の地面グリッド（XY平面、高さZ基準）を描画"""
        gl.glPushAttrib(gl.GL_ALL_ATTRIB_BITS)
        gl.glDisable(gl.GL_LIGHTING)
        gl.glLineWidth(1.0)

        z_floor = float(self.bounds_min[2]) if self.points_raw is not None else 0.0
        grid_size = max(20.0, float(self.camera_distance * 1.5))
        step = 5.0 if grid_size > 50 else (1.0 if grid_size > 10 else 0.5)
        half = int(grid_size / step)

        gl.glBegin(gl.GL_LINES)
        gl.glColor4f(0.25, 0.28, 0.35, 0.4)
        for i in range(-half, half + 1):
            coord = i * step
            # X方向ライン (左右)
            gl.glVertex3f(-half * step, coord, z_floor)
            gl.glVertex3f(half * step, coord, z_floor)
            # Y方向ライン (奥行/上)
            gl.glVertex3f(coord, -half * step, z_floor)
            gl.glVertex3f(coord, half * step, z_floor)
        gl.glEnd()
        gl.glPopAttrib()

    def _draw_section_box(self):
        """断面の範囲枠線（厚みボックス）およびシーク基点プレーンを描画"""
        if not (self.clip_x_enabled or self.clip_y_enabled or self.clip_z_enabled or self.clip_wall_enabled):
            return
        if self.points_raw is None:
            return
        # 基点枠・厚み枠が両方非表示の場合は描画をスキップして点群のみ表示
        if not getattr(self, "show_section_origin_plane", True) and not getattr(self, "show_section_box", True):
            return
        if not self.isValid():
            return

        gl.glPushAttrib(gl.GL_ALL_ATTRIB_BITS)
        gl.glDisable(gl.GL_LIGHTING)

        # ワールド境界内にクランプしてボックス描画を視覚的にわかりやすく維持
        bx_min, bx_max = float(self.bounds_min[0]), float(self.bounds_max[0])
        by_min, by_max = float(self.bounds_min[1]), float(self.bounds_max[1])
        bz_min, bz_max = float(self.bounds_min[2]), float(self.bounds_max[2])

        x0 = max(bx_min, float(self.clip_x_range[0] - self.center_offset[0])) if self.clip_x_enabled else bx_min
        x1 = min(bx_max, float(self.clip_x_range[1] - self.center_offset[0])) if self.clip_x_enabled else bx_max
        y0 = max(by_min, float(self.clip_y_range[0] - self.center_offset[1])) if self.clip_y_enabled else by_min
        y1 = min(by_max, float(self.clip_y_range[1] - self.center_offset[1])) if self.clip_y_enabled else by_max
        z0 = max(bz_min, float(self.clip_z_range[0] - self.center_offset[2])) if self.clip_z_enabled else bz_min
        z1 = min(bz_max, float(self.clip_z_range[1] - self.center_offset[2])) if self.clip_z_enabled else bz_max

        # 1. 厚み範囲ボックス（シアン枠線）
        if getattr(self, "show_section_box", True):
            gl.glLineWidth(1.5)
            gl.glColor4f(0.0, 0.9, 1.0, 0.65)
            gl.glBegin(gl.GL_LINES)
            # 底面
            gl.glVertex3f(x0, y0, z0); gl.glVertex3f(x1, y0, z0)
            gl.glVertex3f(x1, y0, z0); gl.glVertex3f(x1, y1, z0)
            gl.glVertex3f(x1, y1, z0); gl.glVertex3f(x0, y1, z0)
            gl.glVertex3f(x0, y1, z0); gl.glVertex3f(x0, y0, z0)
            # 上面
            gl.glVertex3f(x0, y0, z1); gl.glVertex3f(x1, y0, z1)
            gl.glVertex3f(x1, y0, z1); gl.glVertex3f(x1, y1, z1)
            gl.glVertex3f(x1, y1, z1); gl.glVertex3f(x0, y1, z1)
            gl.glVertex3f(x0, y1, z1); gl.glVertex3f(x0, y0, z1)
            # 柱
            gl.glVertex3f(x0, y0, z0); gl.glVertex3f(x0, y0, z1)
            gl.glVertex3f(x1, y0, z0); gl.glVertex3f(x1, y0, z1)
            gl.glVertex3f(x1, y1, z0); gl.glVertex3f(x1, y1, z1)
            gl.glVertex3f(x0, y1, z0); gl.glVertex3f(x0, y1, z1)
            gl.glEnd()

        # 2. 今シークしている「基点（Origin）断面プレーン」をゴールド枠＋半透明面で強調描画
        if getattr(self, "show_section_origin_plane", True):
            gl.glEnable(gl.GL_BLEND)
            gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
            gl.glDepthMask(gl.GL_FALSE)
            if self.clip_x_enabled:
                ox = float(np.clip(self.clip_x_origin - self.center_offset[0], bx_min, bx_max))
                gl.glColor4f(1.0, 0.78, 0.10, 0.10)
                gl.glBegin(gl.GL_QUADS)
                gl.glVertex3f(ox, y0, z0); gl.glVertex3f(ox, y1, z0)
                gl.glVertex3f(ox, y1, z1); gl.glVertex3f(ox, y0, z1)
                gl.glEnd()
                gl.glLineWidth(2.4)
                gl.glColor4f(1.0, 0.80, 0.15, 0.95)
                gl.glBegin(gl.GL_LINE_LOOP)
                gl.glVertex3f(ox, y0, z0); gl.glVertex3f(ox, y1, z0)
                gl.glVertex3f(ox, y1, z1); gl.glVertex3f(ox, y0, z1)
                gl.glEnd()

            if self.clip_y_enabled:
                oy = float(np.clip(self.clip_y_origin - self.center_offset[1], by_min, by_max))
                gl.glColor4f(0.25, 0.95, 0.45, 0.10)
                gl.glBegin(gl.GL_QUADS)
                gl.glVertex3f(x0, oy, z0); gl.glVertex3f(x1, oy, z0)
                gl.glVertex3f(x1, oy, z1); gl.glVertex3f(x0, oy, z1)
                gl.glEnd()
                gl.glLineWidth(2.4)
                gl.glColor4f(1.0, 0.80, 0.15, 0.95)
                gl.glBegin(gl.GL_LINE_LOOP)
                gl.glVertex3f(x0, oy, z0); gl.glVertex3f(x1, oy, z0)
                gl.glVertex3f(x1, oy, z1); gl.glVertex3f(x0, oy, z1)
                gl.glEnd()

            if self.clip_z_enabled:
                oz = float(np.clip(self.clip_z_origin - self.center_offset[2], bz_min, bz_max))
                gl.glColor4f(1.0, 0.78, 0.10, 0.10)
                gl.glBegin(gl.GL_QUADS)
                gl.glVertex3f(x0, y0, oz); gl.glVertex3f(x1, y0, oz)
                gl.glVertex3f(x1, y1, oz); gl.glVertex3f(x0, y1, oz)
                gl.glEnd()
                gl.glLineWidth(2.4)
                gl.glColor4f(1.0, 0.80, 0.15, 0.95)
                gl.glBegin(gl.GL_LINE_LOOP)
                gl.glVertex3f(x0, y0, oz); gl.glVertex3f(x1, y0, oz)
                gl.glVertex3f(x1, y1, oz); gl.glVertex3f(x0, y1, oz)
                gl.glEnd()

            # 3. 2点ピック斜め壁スライス有効時の斜め基準プレーン・厚み枠線描画
            if self.clip_wall_enabled and self.wall_align_angle_deg is not None and self.wall_align_center_world is not None:
                rad = math.radians(self.wall_align_angle_deg)
                tx, ty = math.cos(rad), math.sin(rad)
                nx, ny = -math.sin(rad), math.cos(rad)
                cx = float(self.wall_align_center_world[0] - self.center_offset[0])
                cy = float(self.wall_align_center_world[1] - self.center_offset[1])
                half_len = max(5.0, float(np.linalg.norm(self.bounds_max[:2] - self.bounds_min[:2])) * 0.55)
                d_orig = float(self.clip_wall_origin)
                ox_c = cx + d_orig * nx
                oy_c = cy + d_orig * ny
                p_a = (ox_c - half_len * tx, oy_c - half_len * ty)
                p_b = (ox_c + half_len * tx, oy_c + half_len * ty)

                gl.glColor4f(0.95, 0.45, 0.95, 0.12)
                gl.glBegin(gl.GL_QUADS)
                gl.glVertex3f(p_a[0], p_a[1], z0); gl.glVertex3f(p_b[0], p_b[1], z0)
                gl.glVertex3f(p_b[0], p_b[1], z1); gl.glVertex3f(p_a[0], p_a[1], z1)
                gl.glEnd()
                gl.glLineWidth(2.5)
                gl.glColor4f(0.95, 0.55, 1.0, 0.95)
                gl.glBegin(gl.GL_LINE_LOOP)
                gl.glVertex3f(p_a[0], p_a[1], z0); gl.glVertex3f(p_b[0], p_b[1], z0)
                gl.glVertex3f(p_b[0], p_b[1], z1); gl.glVertex3f(p_a[0], p_a[1], z1)
                gl.glEnd()

            gl.glDepthMask(gl.GL_TRUE)
        gl.glPopAttrib()

    def _draw_wall_align_lines(self):
        """2点ピック斜め壁アラインの基準ピンと壁ライン・直交法線ガイドを描画"""
        gl.glPushAttrib(gl.GL_ALL_ATTRIB_BITS)
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glLineWidth(2.8)

        p1 = self.wall_align_points[0][0]
        gl.glPointSize(11.0)
        gl.glBegin(gl.GL_POINTS)
        gl.glColor3f(0.95, 0.45, 0.95)
        gl.glVertex3f(float(p1[0]), float(p1[1]), float(p1[2]))
        gl.glEnd()

        if len(self.wall_align_points) >= 2:
            p2 = self.wall_align_points[1][0]
            gl.glBegin(gl.GL_POINTS)
            gl.glColor3f(0.20, 0.95, 0.85)
            gl.glVertex3f(float(p2[0]), float(p2[1]), float(p2[2]))
            gl.glEnd()

            # 2点を結ぶ壁基準ライン (マゼンタ〜シアン)
            gl.glBegin(gl.GL_LINES)
            gl.glColor3f(0.95, 0.55, 1.0)
            gl.glVertex3f(float(p1[0]), float(p1[1]), float(p1[2]))
            gl.glVertex3f(float(p2[0]), float(p2[1]), float(p2[2]))

            # 中点から直交法線方向のガイド線
            mx = 0.5 * (float(p1[0]) + float(p2[0]))
            my = 0.5 * (float(p1[1]) + float(p2[1]))
            mz = 0.5 * (float(p1[2]) + float(p2[2]))
            dx = float(p2[0]) - float(p1[0])
            dy = float(p2[1]) - float(p1[1])
            seg_len = math.hypot(dx, dy)
            if seg_len > 1e-5:
                nx = -dy / seg_len
                ny = dx / seg_len
                n_scale = max(0.5, min(3.0, seg_len * 0.35))
                gl.glColor3f(1.0, 0.85, 0.15)
                gl.glVertex3f(mx, my, mz)
                gl.glVertex3f(mx + nx * n_scale, my + ny * n_scale, mz)
            gl.glEnd()

        gl.glPopAttrib()

    def _draw_measure_lines(self):
        """2点間寸法計測のラインを描画"""
        gl.glPushAttrib(gl.GL_ALL_ATTRIB_BITS)
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glLineWidth(2.5)

        p1 = self.measure_points[0][0]
        gl.glPointSize(10.0)
        gl.glBegin(gl.GL_POINTS)
        gl.glColor3f(1.0, 0.8, 0.0)
        gl.glVertex3f(p1[0], p1[1], p1[2])
        gl.glEnd()

        if len(self.measure_points) >= 2:
            p2 = self.measure_points[1][0]
            gl.glBegin(gl.GL_POINTS)
            gl.glColor3f(0.0, 1.0, 0.8)
            gl.glVertex3f(p2[0], p2[1], p2[2])
            gl.glEnd()

            gl.glBegin(gl.GL_LINES)
            gl.glColor3f(1.0, 0.9, 0.1)
            gl.glVertex3f(p1[0], p1[1], p1[2])
            gl.glVertex3f(p2[0], p2[1], p2[2])
            gl.glEnd()

        gl.glPopAttrib()

    def _draw_axes(self):
        """画面左下に数学座標系のXYZ軸ギズモを描画 (カメラ回転と完全同期)"""
        gl.glPushAttrib(gl.GL_ALL_ATTRIB_BITS)
        gl.glDisable(gl.GL_DEPTH_TEST)

        aspect = self.width() / max(1, self.height())
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glPushMatrix()
        gl.glLoadIdentity()
        gl.glOrtho(-aspect, aspect, -1.0, 1.0, -10.0, 10.0)

        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glPushMatrix()
        gl.glLoadIdentity()
        gl.glTranslatef(-aspect + 0.18, -0.78, 0.0)

        # カメラの回転成分のみを適用 (ターゲット平行移動を除去)
        view = self._get_view_matrix()
        view.setColumn(3, QVector4D(0.0, 0.0, 0.0, 1.0))
        gl.glMultMatrixf(view.data())
        gl.glScalef(0.14, 0.14, 0.14)

        gl.glLineWidth(3.5)
        gl.glBegin(gl.GL_LINES)
        # X軸: 赤 (右)
        gl.glColor3f(1.0, 0.25, 0.25)
        gl.glVertex3f(0.0, 0.0, 0.0); gl.glVertex3f(1.0, 0.0, 0.0)
        # Y軸: 緑 (上 / 奥)
        gl.glColor3f(0.25, 1.0, 0.25)
        gl.glVertex3f(0.0, 0.0, 0.0); gl.glVertex3f(0.0, 1.0, 0.0)
        # Z軸: 青 (高さ)
        gl.glColor3f(0.3, 0.65, 1.0)
        gl.glVertex3f(0.0, 0.0, 0.0); gl.glVertex3f(0.0, 0.0, 1.0)
        gl.glEnd()

        gl.glPopMatrix()
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glPopMatrix()
        gl.glPopAttrib()

    def _draw_2d_overlays(self):
        """2Dオーバーレイ描画"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 1. 軸ラベル (X, Y, Z) の描画
        self._draw_axes_labels_2d(painter)

        # 即時削除モード(消しゴム状態)か通常選択かで枠・ブラシ色を切り替え
        is_erase_visual = bool(self.instant_delete_mode and not (self._last_selection_modifiers & Qt.ControlModifier))
        stroke_col = QColor(248, 113, 113, 235) if is_erase_visual else QColor(0, 215, 255, 230)
        fill_col = QColor(239, 68, 68, 55) if is_erase_visual else QColor(0, 150, 255, 45)

        # 2. 矩形選択枠
        if self.is_selecting and self.selection_shape == self.SELECT_RECT:
            rect = QRect(self.selection_start, self.selection_end).normalized()
            pen = QPen(stroke_col, 2, Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(QBrush(fill_col))
            painter.drawRect(rect)

        # 3. 円形選択枠
        elif self.is_selecting and self.selection_shape == self.SELECT_CIRCLE:
            p1 = self.selection_start
            p2 = self.selection_end
            radius = int(math.hypot(p2.x() - p1.x(), p2.y() - p1.y()))
            pen = QPen(stroke_col, 2, Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(QBrush(fill_col))
            painter.drawEllipse(p1, radius, radius)

        # 4. 多角形選択枠
        elif self.selection_mode and self.selection_shape == self.SELECT_POLYGON and len(self.polygon_points) > 0:
            pen = QPen(stroke_col, 2, Qt.SolidLine)
            painter.setPen(pen)
            for i in range(len(self.polygon_points) - 1):
                painter.drawLine(self.polygon_points[i], self.polygon_points[i + 1])
            dash_pen = QPen(stroke_col, 1.5, Qt.DashLine)
            painter.setPen(dash_pen)
            painter.drawLine(self.polygon_points[-1], self.current_mouse_pos)

            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(255, 200, 0, 240)))
            for pt in self.polygon_points:
                painter.drawEllipse(pt, 4, 4)

        # 4.5. ブラシ選択 / なぞり消しゴム (SELECT_BRUSH) の軌跡 & 円形カーソル描画
        elif self.selection_mode and self.selection_shape == self.SELECT_BRUSH:
            r = max(4, int(self.brush_radius))
            # ドラッグ中のストローク軌跡
            if self.is_selecting and len(self.brush_stroke_points) > 0:
                trail_col = QColor(239, 68, 68, 85) if is_erase_visual else QColor(14, 165, 233, 85)
                trail_pen = QPen(trail_col, r * 2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
                painter.setPen(trail_pen)
                painter.setBrush(Qt.NoBrush)
                if len(self.brush_stroke_points) == 1:
                    painter.drawPoint(self.brush_stroke_points[0])
                else:
                    for i in range(len(self.brush_stroke_points) - 1):
                        painter.drawLine(self.brush_stroke_points[i], self.brush_stroke_points[i + 1])

            # 現在のマウス位置にブラシ円形ガイドを描画
            if not self.current_mouse_pos.isNull():
                ring_pen = QPen(stroke_col, 2.0, Qt.SolidLine)
                painter.setPen(ring_pen)
                painter.setBrush(QBrush(fill_col))
                painter.drawEllipse(self.current_mouse_pos, r, r)
                # 中心クロスヘア
                cx, cy = self.current_mouse_pos.x(), self.current_mouse_pos.y()
                painter.setPen(QPen(QColor(255, 255, 255, 180), 1))
                painter.drawLine(cx - 4, cy, cx + 4, cy)
                painter.drawLine(cx, cy - 4, cx, cy + 4)

        # 4.8. ノイズ処理専用モード / 即時削除 / ブラシモード / パン移動モード時の上部HUDガイドバッジ
        if (self.clean_focus_mode or self.instant_delete_mode or self.pan_mode or self.space_pan_active or (self.selection_mode and self.selection_shape == self.SELECT_BRUSH)) and not self.measure_mode:
            hud_parts = []
            if self.pan_mode or self.space_pan_active:
                hud_parts.append("✋ パン移動モード (左ドラッグで平行移動)")
            if self.clean_focus_mode:
                hud_parts.append("🎯 ノイズ処理専用モード")
            if self.instant_delete_mode:
                hud_parts.append("⚡ 囲み・なぞり即削除 ON")
            if self.selection_mode and self.selection_shape == self.SELECT_BRUSH and not self.space_pan_active:
                hud_parts.append(f"🧹 ブラシ径: {self.brush_radius}px ([ / ] または Ctrl+ホイール)")
            hud_parts.append("V:回転/パン | H:パン | B:ブラシ | R:矩形 | Space+左ドラッグ:一時パン")
            hud_text = "  ｜  ".join(hud_parts)

            font_hud = QFont("Segoe UI", 9, QFont.Bold)
            painter.setFont(font_hud)
            fm = painter.fontMetrics()
            tw = fm.horizontalAdvance(hud_text) + 28
            bx = max(12, (self.width() - tw) // 2)
            by = 10
            bg_hud = QColor(127, 29, 29, 215) if self.instant_delete_mode else QColor(15, 23, 42, 215)
            border_hud = QColor(248, 113, 113, 220) if self.instant_delete_mode else QColor(56, 189, 248, 200)
            painter.setPen(QPen(border_hud, 1.5))
            painter.setBrush(QBrush(bg_hud))
            painter.drawRoundedRect(QRect(bx, by, tw, 28), 14, 14)
            painter.setPen(QColor(254, 242, 242) if self.instant_delete_mode else QColor(224, 242, 254))
            painter.drawText(QRect(bx, by, tw, 28), Qt.AlignCenter, hud_text)

        # 4.9. XYZ断面スライス / 2点斜め壁スライス有効時のシーク基点・厚みHUDオーバーレイ表示
        if (self.clip_x_enabled or self.clip_y_enabled or self.clip_z_enabled or self.clip_wall_enabled) and not self.measure_mode:
            sec_parts = []
            if self.clip_wall_enabled and self.wall_align_angle_deg is not None:
                tw_wall = max(0.0, float(self.clip_wall_range[1] - self.clip_wall_range[0]))
                sec_parts.append(
                    f"斜め壁({self.wall_align_angle_deg:.1f}°)基点:{self.clip_wall_origin:.2f}m [厚{tw_wall:.2f}m]"
                )
            if self.clip_x_enabled:
                tx = max(0.0, float(self.clip_x_range[1] - self.clip_x_range[0]))
                sec_parts.append(
                    f"X基点:{self.clip_x_origin:.2f}m [{self.clip_x_range[0]:.2f}〜{self.clip_x_range[1]:.2f}m / 厚{tx:.2f}m]"
                )
            if self.clip_y_enabled:
                ty = max(0.0, float(self.clip_y_range[1] - self.clip_y_range[0]))
                sec_parts.append(
                    f"Y基点:{self.clip_y_origin:.2f}m [{self.clip_y_range[0]:.2f}〜{self.clip_y_range[1]:.2f}m / 厚{ty:.2f}m]"
                )
            if self.clip_z_enabled:
                tz = max(0.0, float(self.clip_z_range[1] - self.clip_z_range[0]))
                sec_parts.append(
                    f"Z基点:{self.clip_z_origin:.2f}m [{self.clip_z_range[0]:.2f}〜{self.clip_z_range[1]:.2f}m / 厚{tz:.2f}m]"
                )
            vis_cnt = len(self.display_points) if self.display_points is not None else 0
            guide_tag = ""
            if not getattr(self, "show_section_origin_plane", True) and not getattr(self, "show_section_box", True):
                guide_tag = " (枠非表示 / Shift+B:再表示)"
            elif not getattr(self, "show_section_origin_plane", True):
                guide_tag = " (基点枠OFF / B:切替)"
            elif not getattr(self, "show_section_box", True):
                guide_tag = " (厚み枠OFF / B:切替)"
            else:
                guide_tag = " (黄枠=基点 / 水色枠=厚み)"
            if self.section_detail_capped:
                detail_tag = f" (ボックス内{self.section_box_count:,}点を上限{self.SECTION_DETAIL_MAX_POINTS // 10000:,}万点で間引き)"
            elif self.section_full_detail and self.point_budget > 0:
                detail_tag = " (ボックス内全点)"
            else:
                detail_tag = ""
            overlay_tag = ""
            if self.overlay_points is not None and len(self.overlay_points) > 0:
                if self.overlay_kind == "lod":
                    overlay_tag = f"  ｜  🔍 高精細(LOD) {len(self.overlay_points):,}点"
                else:
                    overlay_tag = f"  ｜  🔍 元ファイルから{len(self.overlay_points):,}点を重ね描き"
                    if self.overlay_capped:
                        overlay_tag += f" (ボックス内{self.overlay_box_total:,}点を上限で間引き)"
            sec_text = f"📐 断面シーク中{guide_tag}  ｜  {'  /  '.join(sec_parts)}  ｜  表示: {vis_cnt:,}点{detail_tag}{overlay_tag}"

            font_sec = QFont("Segoe UI", 9, QFont.Bold)
            painter.setFont(font_sec)
            fm_sec = painter.fontMetrics()
            tw_s = fm_sec.horizontalAdvance(sec_text) + 24
            bx_s = max(12, (self.width() - tw_s) // 2)
            by_s = 44 if (self.clean_focus_mode or self.instant_delete_mode or self.pan_mode or self.space_pan_active or (self.selection_mode and self.selection_shape == self.SELECT_BRUSH)) else 10
            painter.setPen(QPen(QColor(250, 204, 21, 220), 1.5))
            painter.setBrush(QBrush(QColor(15, 23, 42, 220)))
            painter.drawRoundedRect(QRect(bx_s, by_s, tw_s, 26), 13, 13)
            painter.setPen(QColor(254, 240, 138))
            painter.drawText(QRect(bx_s, by_s, tw_s, 26), Qt.AlignCenter, sec_text)

        # 5. 計測情報 & 2点ピック斜め壁視点合わせガイド表示
        if self.wall_align_mode:
            painter.setPen(QPen(QColor(232, 121, 249, 230), 1.5))
            painter.setBrush(QBrush(QColor(15, 23, 42, 225)))
            painter.drawRoundedRect(QRect(14, 14, 410, 56), 8, 8)
            painter.setPen(QColor(240, 171, 252))
            painter.setFont(QFont("Segoe UI", 10, QFont.Bold))
            step_msg = "壁に沿った 1点目 (始点) をクリック" if len(self.wall_align_points) == 0 else "壁に沿った 2点目 (終点) をクリック → 自動で直交視点へ"
            painter.drawText(24, 35, f"🧭 2点ピック斜め壁合わせ(U): {step_msg}")
            painter.setPen(QColor(203, 213, 225))
            painter.setFont(QFont("Segoe UI", 9, QFont.Normal))
            painter.drawText(24, 54, "※2点ピック後、[━横][┃縦][🧱正面][🔪断面]で壁沿いノイズを簡単に除去できます")
        elif self.measure_mode or self.measure_result_text:
            # 計測 HUD: 1点目で座標（座標系変換して読み込んだときはファイル系も）、2点目で距離と ΔX/ΔY/ΔZ
            n_pts = len(self.measure_points)
            lines = list(self.measure_coord_lines) or ["計測点 1 をクリックすると座標を表示します"]
            if n_pts == 0:
                title = "📐 計測モード: 座標の読み取り"
            elif n_pts == 1:
                title = "📐 計測: 2点目をクリックで距離"
            else:
                title = "📐 2点間寸法計測結果"
            hint = "Ctrl+C: 座標をコピー / 右クリック: クリア / M: 終了" if n_pts else "※右クリックでキャンセル / M: 終了"
            box_h = 30 + 18 * len(lines) + 22
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(15, 23, 42, 220)))
            painter.drawRoundedRect(QRect(14, 14, 560, box_h), 8, 8)

            painter.setPen(QColor(56, 189, 248) if n_pts >= 2 else QColor(250, 204, 21))
            painter.setFont(QFont("Segoe UI", 11, QFont.Bold))
            painter.drawText(26, 36, title)

            painter.setFont(QFont("Consolas", 10, QFont.Normal))
            y = 56
            for line in lines:
                painter.setPen(QColor(148, 163, 184) if line.startswith("   ") else QColor(241, 245, 249))
                painter.drawText(26, y, line)
                y += 18
            painter.setPen(QColor(148, 163, 184))
            painter.setFont(QFont("Segoe UI", 9, QFont.Normal))
            painter.drawText(26, y + 2, hint)

        painter.end()

    def _draw_axes_labels_2d(self, painter: QPainter):
        """⑦ ギズモ先端のX, Y, Zおよび中心ハブをインタラクティブな丸バッジとして2D描画 (クリックで視点切替)"""
        w, h = self.width(), self.height()
        aspect = w / max(1, h)
        cx = 0.18
        cy = -0.78

        # カメラ回転行列を取得
        view = self._get_view_matrix()
        view.setColumn(3, QVector4D(0.0, 0.0, 0.0, 1.0))

        base_x = (-aspect + cx + aspect) * 0.5 * w
        base_y = (1.0 - (cy + 1.0) * 0.5) * h

        self._gizmo_hit_targets = {}
        hovered = getattr(self, "_hovered_gizmo_axis", None)

        # 中心ハブ (クリックで等角[Iso]視点へ)
        iso_r = 9 if hovered == "ISO" else 7
        painter.setPen(QPen(QColor(226, 232, 240, 220 if hovered == "ISO" else 150), 1.5))
        painter.setBrush(QBrush(QColor(56, 189, 248, 220) if hovered == "ISO" else QColor(30, 41, 59, 200)))
        painter.drawEllipse(QPoint(int(base_x), int(base_y)), iso_r, iso_r)
        self._gizmo_hit_targets["ISO"] = (float(base_x), float(base_y), 11.0)

        font = QFont("Segoe UI", 8, QFont.Bold)
        painter.setFont(font)

        # 各軸先端の投影と丸型バッジ描画
        for label, vec, fill_col, border_col in [
            ("X", QVector3D(1.0, 0.0, 0.0), QColor(220, 38, 38, 230), QColor(252, 165, 165)),
            ("Y", QVector3D(0.0, 1.0, 0.0), QColor(22, 163, 74, 230), QColor(134, 239, 172)),
            ("Z", QVector3D(0.0, 0.0, 1.0), QColor(37, 99, 235, 230), QColor(147, 197, 253)),
        ]:
            v_rot = view.map(vec) * 0.17
            sx = int(base_x + v_rot.x() * 0.5 * w)
            sy = int(base_y - v_rot.y() * 0.5 * h)
            is_hov = (hovered == label)
            rad = 12 if is_hov else 10
            painter.setPen(QPen(QColor(255, 255, 255) if is_hov else border_col, 2.0 if is_hov else 1.2))
            painter.setBrush(QBrush(fill_col.lighter(120) if is_hov else fill_col))
            painter.drawEllipse(QPoint(sx, sy), rad, rad)
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(QRect(sx - rad, sy - rad, rad * 2, rad * 2), Qt.AlignCenter, label)
            self._gizmo_hit_targets[label] = (float(sx), float(sy), 14.0)

        if hovered:
            tip_map = {
                "X": "クリック: 側面視点 (YZ平面)",
                "Y": "クリック: 正面視点 (XZ平面)",
                "Z": "クリック: 上面視点 (XY平面)",
                "ISO": "クリック: 等角3D鳥瞰視点",
            }
            tip_str = tip_map.get(hovered, "")
            if tip_str:
                painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
                painter.setPen(QColor(125, 211, 252))
                painter.drawText(int(base_x - 36), int(base_y + 42), tip_str)

    def is_noise_layer(self, layer_id: int) -> bool:
        """指定したlayer_idがノイズ(除去点)扱いかどうかを判定 (ID=0 または 自動除去フィルタレイヤー)"""
        if layer_id == self.LAYER_NOISE:
            return True
        info = self.layers.get(layer_id)
        if info is not None:
            return bool(getattr(info, "is_noise", False) or (isinstance(info.name, str) and info.name.startswith("自動除去:")))
        return False

    def _init_default_layers(self):
        """標準レイヤー(通常点群・ノイズ)の初期定義を生成"""
        self.layers = {
            self.LAYER_NORMAL: LayerInfo(
                layer_id=self.LAYER_NORMAL,
                name="通常点群",
                visible=True,
                locked=False,
                color=(0.22, 0.74, 0.97),
                use_layer_color=False,
                is_noise=False,
            ),
            self.LAYER_NOISE: LayerInfo(
                layer_id=self.LAYER_NOISE,
                name="ノイズ (除去点)",
                visible=False,
                locked=False,
                color=(0.95, 0.25, 0.25),
                use_layer_color=True,
                is_noise=True,
            ),
        }

    def _ensure_layer_sync(self):
        """layer_maskが外部から直接変更された場合でもlayer_idsとの整合性を維持する"""
        if self.points_raw is None:
            self.layer_mask = None
            self.layer_ids = None
            return
        n = len(self.points_raw)
        if self.layer_ids is None or len(self.layer_ids) != n:
            if self.layer_mask is not None and len(self.layer_mask) == n:
                self.layer_ids = np.where(self.layer_mask, self.LAYER_NORMAL, self.LAYER_NOISE).astype(np.int32)
            else:
                self.layer_ids = np.full(n, self.LAYER_NORMAL, dtype=np.int32)
                self.layer_mask = np.ones(n, dtype=bool)
            return
        if self.layer_mask is not None and len(self.layer_mask) == n:
            noise_ids = [lid for lid in self.layers if self.is_noise_layer(lid)]
            if self.LAYER_NOISE not in noise_ids:
                noise_ids.append(self.LAYER_NOISE)
            expected_mask = ~np.isin(self.layer_ids, noise_ids)
            if not np.array_equal(self.layer_mask, expected_mask):
                to_noise = (~self.layer_mask) & np.isin(self.layer_ids, noise_ids, invert=True)
                to_normal = self.layer_mask & np.isin(self.layer_ids, noise_ids)
                self.layer_ids[to_noise] = self.LAYER_NOISE
                self.layer_ids[to_normal] = self.LAYER_NORMAL

    def _sync_mask_from_ids(self):
        """layer_idsの変更をlayer_maskへ反映する (ID=0および自動除去ノイズレイヤーをFalse[除外]に設定)"""
        if self.layer_ids is not None:
            noise_ids = [lid for lid in self.layers if self.is_noise_layer(lid)]
            if self.LAYER_NOISE not in noise_ids:
                noise_ids.append(self.LAYER_NOISE)
            self.layer_mask = ~np.isin(self.layer_ids, noise_ids)

    def set_point_cloud(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
        intensities: Optional[np.ndarray] = None,
        reset_camera: bool = True,
        scan_ids: Optional[np.ndarray] = None,
    ):
        """点群データを設定する"""
        self._clear_color_cache()
        self.clear_section_overlay()  # 重ね描きは前の点群・前のボックスのもの
        self.clear_measure_points()  # 前の点群の計測点は新しい点群では無効
        if points is None or len(points) == 0:
            self.coord_mode = "cartesian"
            self.points_raw = None
            self.points_centered = None
            self.active_colors = None
            self.display_points = None
            self.display_colors = None
            self.display_orig_indices = None
            self.layer_mask = None
            self.layer_ids = None
            self.scan_ids = None
            self.scan_visibility.clear()
            self._init_default_layers()
            self.selected_indices = np.empty(0, dtype=np.int64)
            self.polygon_points.clear()
            self._vbo_dirty = True
            self.update()
            return

        pts_arr = np.asarray(points, dtype=np.float64)
        if not np.isfinite(pts_arr).all():
            pts_arr = np.nan_to_num(pts_arr, copy=True, nan=0.0, posinf=0.0, neginf=0.0)
        self.points_raw = pts_arr
        n = len(self.points_raw)
        self._init_default_layers()
        self.layers[self.LAYER_NORMAL].visible = self.show_normal_layer
        self.layers[self.LAYER_NOISE].visible = self.show_noise_layer
        self.layer_ids = np.full(n, self.LAYER_NORMAL, dtype=np.int32)
        self.layer_mask = np.ones(n, dtype=bool)
        self.selected_indices = np.empty(0, dtype=np.int64)
        self.polygon_points.clear()

        if scan_ids is not None and len(scan_ids) == n:
            self.scan_ids = np.asarray(scan_ids, dtype=np.int32)
        else:
            self.scan_ids = np.zeros(n, dtype=np.int32)
        unique_scans = np.unique(self.scan_ids)
        self.scan_visibility = {int(s): True for s in unique_scans}

        min_b = np.min(self.points_raw, axis=0)
        max_b = np.max(self.points_raw, axis=0)
        self.world_bounds_min = min_b.copy()
        self.world_bounds_max = max_b.copy()

        self.center_offset = (min_b + max_b) / 2.0
        self.points_centered = np.ascontiguousarray((self.points_raw - self.center_offset), dtype=np.float32)

        self.bounds_min = (min_b - self.center_offset).astype(np.float32)
        self.bounds_max = (max_b - self.center_offset).astype(np.float32)

        if colors is not None and len(colors) == len(points):
            c = np.asarray(colors, dtype=np.float32)
            if not np.isfinite(c).all():
                c = np.nan_to_num(c, copy=True, nan=0.0, posinf=1.0, neginf=0.0)
            max_c = float(np.max(c)) if len(c) > 0 else 1.0
            if max_c > 255.0:
                c = c / 65535.0
            elif max_c > 1.0:
                c = c / 255.0
            self.colors_raw = np.clip(c, 0.0, 1.0)
        else:
            self.colors_raw = None

        if intensities is not None and len(intensities) == len(points):
            ints_arr = np.asarray(intensities, dtype=np.float32)
            if not np.isfinite(ints_arr).all():
                ints_arr = np.nan_to_num(ints_arr, copy=True, nan=0.0, posinf=0.0, neginf=0.0)
            self.intensities_raw = ints_arr
        else:
            self.intensities_raw = None

        self._update_active_colors()
        self._update_display_arrays()

        if reset_camera:
            self.fit_to_screen()

        self.layers_updated.emit(len(self.points_raw), 0)
        self.layer_list_changed.emit()
        self.bounds_ready.emit(self.world_bounds_min, self.world_bounds_max)
        self.update()
        self.status_changed.emit(f"点群読み込み完了: {len(points):,} 点")

    def set_preloaded_data(self, preloaded, reset_camera: bool = True):
        """バックグラウンドで前処理済みのデータを0ミリ秒で超高速バインド (UIフリーズ皆無)"""
        self._clear_color_cache()
        self.clear_section_overlay()
        if preloaded is None or preloaded.points_raw is None or len(preloaded.points_raw) == 0:
            self.set_point_cloud(None)
            return

        self.clear_measure_points()  # 前の点群の計測点は新しい点群では無効
        # 読み込み時の座標系変換を計測ツールへ伝える（ピックした座標をファイル系でも表示するため）
        scans = getattr(preloaded, "scans", None)
        self.coord_mode = (getattr(scans[0], "coord_mode", "cartesian") or "cartesian") if scans else "cartesian"

        self.points_raw = preloaded.points_raw
        self.points_centered = preloaded.points_centered
        self.center_offset = preloaded.center_offset
        self.colors_raw = preloaded.colors_raw
        self.intensities_raw = preloaded.intensities_raw
        self.active_colors = preloaded.active_colors
        self.world_bounds_min = preloaded.world_bounds_min
        self.world_bounds_max = preloaded.world_bounds_max
        self.bounds_min = preloaded.bounds_min
        self.bounds_max = preloaded.bounds_max
        self.display_points = preloaded.display_points
        self.display_colors = preloaded.display_colors
        self.display_orig_indices = preloaded.display_orig_indices

        n = len(self.points_raw)
        self._init_default_layers()
        self.layers[self.LAYER_NORMAL].visible = self.show_normal_layer
        self.layers[self.LAYER_NOISE].visible = self.show_noise_layer
        self.layer_ids = np.full(n, self.LAYER_NORMAL, dtype=np.int32)
        self.layer_mask = np.ones(n, dtype=bool)
        self.selected_indices = np.empty(0, dtype=np.int64)
        self.polygon_points.clear()
        self._vbo_dirty = True

        scan_ids = getattr(preloaded, "scan_ids", None)
        if scan_ids is not None and len(scan_ids) == n:
            self.scan_ids = np.asarray(scan_ids, dtype=np.int32)
        else:
            self.scan_ids = np.zeros(n, dtype=np.int32)
        unique_scans = np.unique(self.scan_ids)
        self.scan_visibility = {int(s): True for s in unique_scans}

        # 現在のカラーモードがRGB以外の場合、または断面スライスが有効な場合は再反映
        if self.color_mode != self.COLOR_MODE_RGB or self.clip_x_enabled or self.clip_y_enabled or self.clip_z_enabled:
            self._update_active_colors()
            self._update_display_arrays()

        if reset_camera:
            self.fit_to_screen()

        self.layers_updated.emit(len(self.points_raw), 0)
        self.layer_list_changed.emit()
        self.bounds_ready.emit(self.world_bounds_min, self.world_bounds_max)
        self.update()
        self.status_changed.emit(f"点群読み込み完了: {len(self.points_raw):,} 点")

    def _clear_color_cache(self):
        """点群ロード時や点群変更時にカラー正規化キャッシュをクリア"""
        self._cached_normalized_heights = None
        self._cached_normalized_intensities = None
        self._cached_shading_factors = None
        self._cached_height_range = None
        self._cached_intensity_stretch = None

    def _compute_shading_factors(self) -> np.ndarray:
        """
        ① 凹凸・エッジ輪郭強調 (疑似EDL/法線陰影シェーディング) 係数を高速ベクトル算出 (キャッシュで2回目以降0ms)
        点群の局所接線ベクトル外積(法線近似)・角度変化(角・エッジ検出)・反射強度微細コントラストを統合します。
        """
        n = len(self.points_centered) if self.points_centered is not None else 0
        if self._cached_shading_factors is not None and len(self._cached_shading_factors) == n:
            return self._cached_shading_factors
        if self.points_centered is None or n < 4:
            res = np.ones(n, dtype=np.float32)
            self._cached_shading_factors = res
            return res

        iv_norm = self._normalize_intensities() if (self.intensities_raw is not None and len(self.intensities_raw) == n) else None
        res = self._shading_factors_of(self.points_centered, iv_norm)
        self._cached_shading_factors = res
        return res

    @staticmethod
    def _shading_factors_of(pts: np.ndarray, iv_norm: Optional[np.ndarray]) -> np.ndarray:
        """陰影係数の本体（点列 pts と正規化済み反射強度から算出。重ね描きにも同じ式を使う）"""
        n = len(pts)
        if n < 4:
            return np.ones(n, dtype=np.float32)
        v1 = np.empty_like(pts)
        v1[:-1] = pts[1:] - pts[:-1]
        v1[-1] = v1[-2]

        step2 = min(7, max(2, n // 20))
        v2 = np.empty_like(pts)
        v2[:-step2] = pts[step2:] - pts[:-step2]
        v2[-step2:] = v2[-step2 - 1]

        # 局所法線近似ベクトル (v1 × v2)
        normals = np.cross(v1, v2)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        valid_n = norms[:, 0] > 1e-7
        normals[valid_n] /= norms[valid_n]
        normals[~valid_n] = [0.0, 0.0, 1.0]

        # 斜め上方の仮想光源 (0.45, -0.35, 0.82) との内積で立体陰影を生成
        light_dot = np.abs(normals[:, 0] * 0.45 - normals[:, 1] * 0.35 + normals[:, 2] * 0.82)

        # 隣接ベクトル間の角度変化でエッジ・角(コーナー)を検出 (EDL風輪郭強調)
        len1 = np.maximum(1e-6, np.linalg.norm(v1, axis=1, keepdims=True))
        u1 = v1 / len1
        edge_factor = np.ones(n, dtype=np.float32)
        edge_factor[1:] = 0.65 + 0.35 * np.abs(np.sum(u1[1:] * u1[:-1], axis=1))

        shade = (0.68 + 0.44 * light_dot) * edge_factor

        # 反射強度がある場合は微細な表面テクスチャ陰影を15%ブレンド
        if iv_norm is not None and len(iv_norm) == n:
            shade = shade * (0.90 + 0.20 * iv_norm)

        return np.clip(shade, 0.50, 1.15).astype(np.float32)

    def _height_range(self) -> Tuple[float, float]:
        """高さ色分けの (z_min, z_max)（中心オフセット済み座標）。任意高さレンジ有効時はその範囲、無効時は点群全体の範囲"""
        if getattr(self, "custom_height_enabled", False):
            # 任意高さ範囲が有効な場合: ワールド標高基準から中心オフセットを引いたローカル座標で正規化
            cz = float(self.center_offset[2]) if self.center_offset is not None else 0.0
            return float(self.custom_height_min - cz), float(self.custom_height_max - cz)
        if self._cached_height_range is None:
            if self.points_centered is None or len(self.points_centered) == 0:
                return 0.0, 1.0
            z = self.points_centered[:, 2]
            self._cached_height_range = (float(np.min(z)), float(np.max(z)))
        return self._cached_height_range

    def _normalize_heights(self) -> np.ndarray:
        """点群のZ高さを 0.0〜1.0 に正規化 (キャッシュ利用で2回目以降0ms即時参照、任意高さレンジ対応)"""
        n = len(self.points_centered) if self.points_centered is not None else 0
        if self._cached_normalized_heights is not None and len(self._cached_normalized_heights) == n:
            return self._cached_normalized_heights
        if self.points_centered is None or n == 0:
            return np.zeros(0, dtype=np.float32)
        z = self.points_centered[:, 2]
        z_min, z_max = self._height_range()
        diff = max(1e-5, z_max - z_min)
        res = np.clip((z - z_min) / diff, 0.0, 1.0).astype(np.float32)
        self._cached_normalized_heights = res
        return res

    def _intensity_stretch(self) -> Tuple[float, float]:
        """反射強度ストレッチの (p_low, p_high)（2%〜98% パーセンタイル）。重ね描きも同じ値で正規化する"""
        if self._cached_intensity_stretch is not None:
            return self._cached_intensity_stretch
        if self.intensities_raw is None or len(self.intensities_raw) == 0:
            return 0.0, 1.0
        iv = self.intensities_raw
        if len(iv) > 10:
            p_low, p_high = np.percentile(iv, [2.0, 98.0])
            if p_high - p_low < 1e-6:
                p_low, p_high = float(np.min(iv)), float(np.max(iv))
        else:
            p_low, p_high = float(np.min(iv)), float(np.max(iv))
        self._cached_intensity_stretch = (float(p_low), float(p_high))
        return self._cached_intensity_stretch

    def _normalize_intensities(self) -> np.ndarray:
        """点群の反射強度を 2%〜98% パーセンタイルでストレッチ正規化 (キャッシュ利用で50〜150msのラグ解消)"""
        n = len(self.intensities_raw) if self.intensities_raw is not None else 0
        if self._cached_normalized_intensities is not None and len(self._cached_normalized_intensities) == n:
            return self._cached_normalized_intensities
        if self.intensities_raw is None or n == 0:
            return np.zeros(0, dtype=np.float32)
        iv = self.intensities_raw
        p_low, p_high = self._intensity_stretch()
        diff = max(1e-5, p_high - p_low)
        res = np.clip((iv - p_low) / diff, 0.0, 1.0).astype(np.float32)
        self._cached_normalized_intensities = res
        return res

    def _is_height_mode(self, mode: str) -> bool:
        return "高さ" in mode or mode in (
            self.COLOR_MODE_HEIGHT,
            self.COLOR_MODE_HEIGHT_VIRIDIS,
            self.COLOR_MODE_HEIGHT_JET,
            self.COLOR_MODE_HEIGHT_PLASMA,
            self.COLOR_MODE_HEIGHT_TERRAIN,
            self.COLOR_MODE_HEIGHT_HEAT,
            self.COLOR_MODE_HEIGHT_COOLWARM,
        )

    def _is_intensity_mode(self, mode: str) -> bool:
        return "反射強度" in mode or mode in (
            self.COLOR_MODE_INTENSITY,
            self.COLOR_MODE_INTENSITY_HEAT,
            self.COLOR_MODE_INTENSITY_PLASMA,
            self.COLOR_MODE_INTENSITY_TURBO,
            self.COLOR_MODE_INTENSITY_VIRIDIS,
        )

    def _height_colormap(self, t: np.ndarray, mode: str) -> np.ndarray:
        """高さ (Z) 系モードのパレット選択 (Turbo, Viridis, Jet, Plasma, Terrain, Heat, Cool-Warm, Grayscale)"""
        if "Viridis" in mode or self.color_palette == self.PALETTE_VIRIDIS:
            return self._colormap_viridis(t)
        if "Jet" in mode or self.color_palette == self.PALETTE_JET:
            return self._colormap_jet(t)
        if "Plasma" in mode or self.color_palette == self.PALETTE_PLASMA:
            return self._colormap_plasma(t)
        if "地形" in mode or "Terrain" in mode or self.color_palette == self.PALETTE_TERRAIN:
            return self._colormap_terrain(t)
        if "Heat" in mode or "温熱" in mode or self.color_palette == self.PALETTE_HEAT:
            return self._colormap_heat(t)
        if "Cool-Warm" in mode or "青白赤" in mode or self.color_palette == self.PALETTE_COOLWARM:
            return self._colormap_coolwarm(t)
        if self.color_palette == self.PALETTE_GRAYSCALE:
            return self._colormap_grayscale(t)
        # 標準の高さ (Z) -> 従来と同じ Turbo レインボー！
        return self._colormap_turbo(t)

    def _intensity_colormap(self, t: np.ndarray, mode: str) -> np.ndarray:
        """反射強度系モードのパレット選択 (白黒, Heat, Plasma, Turbo, Viridis)"""
        if "Heat" in mode or "温熱" in mode or self.color_palette == self.PALETTE_HEAT:
            return self._colormap_heat(t)
        if "Plasma" in mode or self.color_palette == self.PALETTE_PLASMA:
            return self._colormap_plasma(t)
        if "Turbo" in mode or "レインボー" in mode or self.color_palette == self.PALETTE_TURBO:
            return self._colormap_turbo(t)
        if "Viridis" in mode or self.color_palette == self.PALETTE_VIRIDIS:
            return self._colormap_viridis(t)
        # 従来の標準「反射強度」-> 白黒グレースケール！
        return self._colormap_grayscale(t)

    def _update_active_colors(self):
        """原本全体のカラー配列を生成（カラーマップ・レイヤー色・スキャン別色分け・反射強度ストレッチ対応）"""
        if self.points_centered is None:
            return

        self._ensure_layer_sync()
        n = len(self.points_centered)
        mode = self.color_mode

        # 1. RGB (実色写真)
        if mode in (self.COLOR_MODE_RGB, "RGB", "RGB (実色写真)"):
            if self.colors_raw is not None:
                self.active_colors = self.colors_raw.copy()
            elif self.intensities_raw is not None:
                # 写真がない場合は反射強度でフォールバック
                t = self._normalize_intensities()
                self.active_colors = self._colormap_grayscale(t)
            else:
                # 写真も強度もない場合は高さでフォールバック
                t = self._normalize_heights()
                self.active_colors = self._colormap_turbo(t)

        # 2. 高さ (Z) 系カラーマップ (Turbo, Viridis, Jet, Plasma, Terrain, Heat, Cool-Warm)
        elif self._is_height_mode(mode):
            self.active_colors = self._height_colormap(self._normalize_heights(), mode)

        # 3. 反射強度系カラーマップ (白黒, Heat, Plasma, Turbo, Viridis)
        elif self._is_intensity_mode(mode):
            if self.intensities_raw is not None:
                self.active_colors = self._intensity_colormap(self._normalize_intensities(), mode)
            else:
                # 反射強度データがない場合は高さでフォールバック
                t = self._normalize_heights()
                self.active_colors = self._colormap_turbo(t)

        # 4. スキャン別 (器械点)
        elif mode in (self.COLOR_MODE_SCAN, "スキャン別 (器械点)") and self.scan_ids is not None:
            palette_len = len(self.SCAN_PALETTE)
            idx_mod = np.mod(np.maximum(0, self.scan_ids), palette_len)
            self.active_colors = self.SCAN_PALETTE[idx_mod].copy()

        # 5. レイヤー別色分け
        elif mode in (self.COLOR_MODE_LAYER, "レイヤー別色分け") and self.layer_ids is not None:
            self.active_colors = np.tile([0.22, 0.74, 0.97], (n, 1)).astype(np.float32)
            for lid, linfo in self.layers.items():
                mask = (self.layer_ids == lid)
                if np.any(mask):
                    self.active_colors[mask] = linfo.color

        # 6. 単色またはその他
        else:
            self.active_colors = np.tile([0.3, 0.8, 0.95], (n, 1)).astype(np.float32)

        # ① 凹凸・エッジ輪郭強調 (疑似EDL/陰影コントラスト強調) の適用
        if getattr(self, "shading_enabled", False):
            shade = self._compute_shading_factors()
            strength = float(getattr(self, "shading_strength", 0.65))
            factor = 1.0 + (shade - 1.0) * strength
            self.active_colors = np.clip(self.active_colors * factor[:, np.newaxis], 0.0, 1.0).astype(np.float32)

        # 各レイヤー固有色の上書き (ノイズレイヤーやuse_layer_color=Trueのカスタムレイヤー)
        if self.layer_ids is not None and self.color_mode != self.COLOR_MODE_LAYER:
            for lid, linfo in self.layers.items():
                if lid == self.LAYER_NOISE or linfo.use_layer_color:
                    mask = (self.layer_ids == lid)
                    if np.any(mask):
                        self.active_colors[mask] = linfo.color

        # 選択中の点を黄色でハイライト
        if len(self.selected_indices) > 0:
            valid = self.selected_indices[self.selected_indices < n]
            self.active_colors[valid] = [1.0, 0.9, 0.1]

        self.active_colors = np.ascontiguousarray(self.active_colors, dtype=np.float32)
        # 重ね描き（元ファイル読み直し）も同じ配色で作り直す
        if self._overlay_source is not None:
            self._recolor_overlay()

    def _get_layer_and_scan_visibility_mask(self) -> np.ndarray:
        """現在のマルチレイヤー表示設定およびスキャン(器械点)表示設定を統合した可視マスクを返す"""
        if self.points_raw is None or len(self.points_raw) == 0:
            return np.empty(0, dtype=bool)
        self._ensure_layer_sync()
        n = len(self.points_raw)

        layer_vis = np.zeros(n, dtype=bool)
        if self.layer_ids is not None:
            for lid, linfo in self.layers.items():
                if linfo.visible:
                    layer_vis |= (self.layer_ids == lid)
        else:
            layer_vis[:] = True

        if self.scan_ids is not None and len(self.scan_visibility) > 0:
            if all(self.scan_visibility.values()):
                scan_vis = np.ones(n, dtype=bool)
            else:
                scan_vis = np.zeros(n, dtype=bool)
                for sid, vis in self.scan_visibility.items():
                    if vis:
                        scan_vis |= (self.scan_ids == sid)
            return layer_vis & scan_vis

        return layer_vis

    def _get_unlocked_mask(self) -> np.ndarray:
        """ロックされていないレイヤーに属する点のブールマスクを返す"""
        if self.points_raw is None or len(self.points_raw) == 0:
            return np.empty(0, dtype=bool)
        self._ensure_layer_sync()
        n = len(self.points_raw)
        if self.layer_ids is None:
            return np.ones(n, dtype=bool)
        unlocked = np.ones(n, dtype=bool)
        for lid, linfo in self.layers.items():
            if linfo.locked:
                unlocked &= (self.layer_ids != lid)
        return unlocked

    def _get_section_inlier_mask(self) -> np.ndarray:
        """現在のXYZ断面スライスおよび2点ピック斜め壁スライスの統合インライアマスクを返す (float32 SIMD高速化)"""
        if self.points_centered is None or len(self.points_centered) == 0:
            return np.empty(0, dtype=bool)

        # 断面スライスがすべて無効の場合は即座に全Trueを返却 (余計な演算をゼロ化)
        if not (self.clip_x_enabled or self.clip_y_enabled or self.clip_z_enabled or self.clip_wall_enabled):
            return np.ones(len(self.points_centered), dtype=bool)

        pts = self.points_centered
        cx, cy, cz = float(self.center_offset[0]), float(self.center_offset[1]), float(self.center_offset[2])
        in_sec = np.ones(len(pts), dtype=bool)

        if self.clip_x_enabled:
            in_sec &= (pts[:, 0] >= (self.clip_x_range[0] - cx)) & (pts[:, 0] <= (self.clip_x_range[1] - cx))
        if self.clip_y_enabled:
            in_sec &= (pts[:, 1] >= (self.clip_y_range[0] - cy)) & (pts[:, 1] <= (self.clip_y_range[1] - cy))
        if self.clip_z_enabled:
            in_sec &= (pts[:, 2] >= (self.clip_z_range[0] - cz)) & (pts[:, 2] <= (self.clip_z_range[1] - cz))
        if self.clip_wall_enabled and self.wall_align_angle_deg is not None and self.wall_align_center_world is not None:
            rad = math.radians(self.wall_align_angle_deg)
            nx = -math.sin(rad)
            ny = math.cos(rad)
            cx_rel = float(self.wall_align_center_world[0] - cx)
            cy_rel = float(self.wall_align_center_world[1] - cy)
            d_perp = (pts[:, 0] - cx_rel) * nx + (pts[:, 1] - cy_rel) * ny
            in_sec &= (d_perp >= self.clip_wall_range[0]) & (d_perp <= self.clip_wall_range[1])
        return in_sec

    def _update_display_arrays_fast_preview(self):
        """
        断面スライダーのドラッグ中に、全点(数千万点)の走査を完全スキップし、
        現在表示されている点群(100〜200万点)に対してわずか数ミリ秒で即時断面プレビューを反映。
        スライダーの動きに断面が吸い付くように60fpsでヌルヌル追従します。
        """
        self._drop_overlay_if_box_changed()
        if self._preview_base_points is None or len(self._preview_base_points) == 0:
            self._update_display_arrays()
            return

        cx, cy, cz = float(self.center_offset[0]), float(self.center_offset[1]), float(self.center_offset[2])
        pts = self._preview_base_points
        mask = np.ones(len(pts), dtype=bool)

        if self.clip_x_enabled:
            mask &= (pts[:, 0] >= (self.clip_x_range[0] - cx)) & (pts[:, 0] <= (self.clip_x_range[1] - cx))
        if self.clip_y_enabled:
            mask &= (pts[:, 1] >= (self.clip_y_range[0] - cy)) & (pts[:, 1] <= (self.clip_y_range[1] - cy))
        if self.clip_z_enabled:
            mask &= (pts[:, 2] >= (self.clip_z_range[0] - cz)) & (pts[:, 2] <= (self.clip_z_range[1] - cz))
        if self.clip_wall_enabled and self.wall_align_angle_deg is not None and self.wall_align_center_world is not None:
            rad = math.radians(self.wall_align_angle_deg)
            nx = -math.sin(rad)
            ny = math.cos(rad)
            cx_rel = float(self.wall_align_center_world[0] - cx)
            cy_rel = float(self.wall_align_center_world[1] - cy)
            d_perp = (pts[:, 0] - cx_rel) * nx + (pts[:, 1] - cy_rel) * ny
            mask &= (d_perp >= self.clip_wall_range[0]) & (d_perp <= self.clip_wall_range[1])

        surv = np.where(mask)[0]
        self.display_points = np.ascontiguousarray(pts[surv])
        if self._preview_base_colors is not None and len(self._preview_base_colors) == len(pts):
            self.display_colors = np.ascontiguousarray(self._preview_base_colors[surv])
        else:
            self.display_colors = np.empty((len(surv), 3), dtype=np.float32)
        self._vbo_dirty = True

    def _update_display_arrays(self):
        """断面スライス・レイヤー可視性・スキャン可視性・Point Budgetを適用して描画バッファを生成"""
        self._vbo_dirty = True
        self._drop_overlay_if_box_changed()
        if self.points_raw is None or len(self.points_raw) == 0:
            self.display_points = None
            self.display_colors = None
            self.display_orig_indices = None
            self._preview_base_points = None
            self._preview_base_colors = None
            return

        # 断面マスク (XYZ断面 + 2点斜め壁スライス)
        in_sec = self._get_section_inlier_mask()

        # レイヤー＆スキャン可視性マスク
        layer_scan_vis = self._get_layer_and_scan_visibility_mask()

        mask = in_sec & layer_scan_vis
        surviving_indices = np.where(mask)[0]
        n_surv = len(surviving_indices)

        if n_surv == 0:
            self.display_points = np.empty((0, 3), dtype=np.float32)
            self.display_colors = np.empty((0, 3), dtype=np.float32)
            self.display_orig_indices = np.empty(0, dtype=np.int64)
            self._preview_base_points = self.display_points
            self._preview_base_colors = self.display_colors
            return

        # 断面有効中はボックス内の点を表示予算で間引かず描く（上限 SECTION_DETAIL_MAX_POINTS。表示予算「無制限」はそのまま）
        budget = self.point_budget
        section_active = self.is_section_active()
        self.section_box_count = n_surv if section_active else 0
        if section_active and self.section_full_detail and budget > 0:
            budget = max(budget, int(self.SECTION_DETAIL_MAX_POINTS))
        self.section_detail_capped = bool(section_active and self.section_full_detail and budget > 0 and n_surv > budget)
        if budget <= 0 or n_surv <= budget:
            sampled_idx = surviving_indices
        else:
            step = max(1, int(math.ceil(n_surv / budget)))
            sampled_idx = surviving_indices[::step]

        self.display_orig_indices = sampled_idx
        self.display_points = np.ascontiguousarray(self.points_centered[sampled_idx])
        self.display_colors = np.ascontiguousarray(self.active_colors[sampled_idx])
        self._preview_base_points = self.display_points
        self._preview_base_colors = self.display_colors

    def validate_invariants(self) -> None:
        """
        品質ゲート用: ビューア内部の全配列・マスク・レイヤーID・スキャンID・選択インデックスの不変条件(Invariants)を検証する。
        不整合がある場合は AssertionError を送出して即座にバグ箇所を特定可能にする。
        """
        if self.points_raw is None:
            assert self.points_centered is None, "points_rawがNoneなのにpoints_centeredが残っています"
            assert self.layer_mask is None, "points_rawがNoneなのにlayer_maskが残っています"
            assert self.layer_ids is None, "points_rawがNoneなのにlayer_idsが残っています"
            assert len(self.selected_indices) == 0, "points_rawがNoneなのにselected_indicesが残っています"
            return

        self._ensure_layer_sync()
        n = len(self.points_raw)
        assert self.points_raw.ndim == 2 and self.points_raw.shape[1] == 3, f"points_rawの形状が不正です: {self.points_raw.shape}"
        assert self.points_centered is not None and self.points_centered.shape == (n, 3), "points_centeredの形状がpoints_rawと不一致です"
        assert self.active_colors is not None and self.active_colors.shape == (n, 3), "active_colorsの形状がpoints_rawと不一致です"
        assert self.layer_mask is not None and self.layer_mask.shape == (n,), f"layer_maskの長さ({len(self.layer_mask) if self.layer_mask is not None else None})が点数({n})と不一致です"
        noise_ids = [lid for lid in self.layers if self.is_noise_layer(lid)]
        if self.LAYER_NOISE not in noise_ids:
            noise_ids.append(self.LAYER_NOISE)
        expected_mask = ~np.isin(self.layer_ids, noise_ids)
        assert np.array_equal(self.layer_mask, expected_mask), "layer_maskとlayer_idsの整合性が崩れています"

        if self.scan_ids is not None:
            assert self.scan_ids.shape == (n,), "scan_idsの長さがpoints_rawと不一致です"
        if self.colors_raw is not None:
            assert self.colors_raw.shape == (n, 3), "colors_rawの形状がpoints_rawと不一致です"
        if self.intensities_raw is not None:
            assert self.intensities_raw.shape == (n,), "intensities_rawの長さがpoints_rawと不一致です"

        if len(self.selected_indices) > 0:
            assert np.all((self.selected_indices >= 0) & (self.selected_indices < n)), "selected_indicesに範囲外のインデックスが含まれています"
            assert len(np.unique(self.selected_indices)) == len(self.selected_indices), "selected_indicesに重複インデックスが含まれています"

        if self.display_points is not None:
            n_disp = len(self.display_points)
            assert self.display_colors is not None and len(self.display_colors) == n_disp, "display_pointsとdisplay_colorsの長さが不一致です"
            assert self.display_orig_indices is not None and len(self.display_orig_indices) == n_disp, "display_pointsとdisplay_orig_indicesの長さが不一致です"
            assert not np.any(np.isnan(self.display_points)), "display_pointsにNaNが含まれています"
            assert not np.any(np.isnan(self.display_colors)), "display_colorsにNaNが含まれています"

        if self.overlay_points is not None:
            n_ov = len(self.overlay_points)
            assert self.overlay_points.shape == (n_ov, 3), f"overlay_pointsの形状が不正です: {self.overlay_points.shape}"
            assert self.overlay_colors is not None and len(self.overlay_colors) == n_ov, "overlay_pointsとoverlay_colorsの長さが不一致です"
            assert not np.any(np.isnan(self.overlay_points)), "overlay_pointsにNaNが含まれています"
            assert self.overlay_kind in ("section", "lod"), f"重ね描きの種別が不正です: {self.overlay_kind!r}"
            if self.overlay_kind == "section":
                assert self.is_section_active(), "断面が無効なのに断面の重ね描き(overlay_points)が残っています"
            assert self._overlay_box_key == self._section_box_key(), "重ね描きが現在の断面ボックスと一致していません"

    # --- レイヤー管理メソッド (マルチレイヤー・ロック・スキャン・プロジェクト保存対応) ---
    def _emit_layer_counts(self):
        """有効点数・ノイズ点数の更新シグナルとレイヤー一覧変更シグナルを発火"""
        if self.layer_mask is None:
            return
        n_norm = int(np.sum(self.layer_mask))
        n_noise = int(len(self.layer_mask) - n_norm)
        self.layers_updated.emit(n_norm, n_noise)
        self.layer_list_changed.emit()

    def set_layer_visibility(self, show_normal: bool, show_noise: bool):
        """有効レイヤー群およびノイズレイヤーの一括表示/非表示を切り替え (後方互換)"""
        self.show_normal_layer = show_normal
        self.show_noise_layer = show_noise
        for lid, linfo in self.layers.items():
            if self.is_noise_layer(lid):
                linfo.visible = show_noise
            else:
                linfo.visible = show_normal
        self._update_display_arrays()
        self.update()
        self.layer_list_changed.emit()

    def set_single_layer_visibility(self, layer_id: int, visible: bool):
        """指定した単一レイヤーの表示/非表示を切り替え"""
        if layer_id not in self.layers:
            return
        self.layers[layer_id].visible = bool(visible)
        if layer_id == self.LAYER_NOISE:
            self.show_noise_layer = bool(visible)
        elif layer_id == self.LAYER_NORMAL:
            self.show_normal_layer = bool(visible)
        self._update_display_arrays()
        self.update()
        self.layer_list_changed.emit()

    def set_layer_locked(self, layer_id: int, locked: bool):
        """指定レイヤーのロック(保護)状態を切り替え。ロック時はそのレイヤーの選択状態も自動解除する"""
        if layer_id not in self.layers:
            return
        self.layers[layer_id].locked = bool(locked)
        if locked and self.layer_ids is not None and len(self.selected_indices) > 0:
            self._ensure_layer_sync()
            keep = self.layer_ids[self.selected_indices] != layer_id
            if not np.all(keep):
                self.selected_indices = self.selected_indices[keep]
                self.point_selected.emit(self.selected_indices.tolist())
        self._update_active_colors()
        self._update_display_arrays()
        self.update()
        self.layer_list_changed.emit()

    def add_custom_layer(
        self,
        name: str,
        color: Optional[Tuple[float, float, float]] = None,
        layer_id: Optional[int] = None,
        is_noise: bool = False,
    ) -> int:
        """新しいカスタムレイヤーを作成してそのlayer_idを返す"""
        clean_name = name.strip()
        auto_noise = bool(is_noise or (clean_name.startswith("自動除去:")))

        if layer_id is None:
            existing_ids = set(self.layers.keys()) | {0, 1, 2, 3}
            new_id = 4
            while new_id in existing_ids:
                new_id += 1
        else:
            new_id = int(layer_id)

        if color is None:
            palette_idx = max(0, new_id - 2) % len(self.LAYER_PALETTE)
            color = self.LAYER_PALETTE[palette_idx]

        self.layers[new_id] = LayerInfo(
            layer_id=new_id,
            name=clean_name or f"レイヤー {new_id}",
            visible=not auto_noise,  # 自動除去ノイズレイヤーは初期状態で非表示(画面から除外)
            locked=False,
            color=tuple(float(c) for c in color),
            use_layer_color=True,
            is_noise=auto_noise,
        )
        self.layer_list_changed.emit()
        return new_id

    def ensure_preset_layer(
        self,
        layer_id: int,
        default_name: str,
        default_color: Tuple[float, float, float],
    ) -> int:
        """地面(ID=2)や主平面(ID=3)などのプリセットレイヤーが存在しなければ作成してIDを返す"""
        if layer_id not in self.layers:
            self.layers[layer_id] = LayerInfo(
                layer_id=layer_id,
                name=default_name,
                visible=True,
                locked=False,
                color=default_color,
                use_layer_color=True,
                is_noise=False,
            )
            self.layer_list_changed.emit()
        return layer_id

    def rename_layer(self, layer_id: int, new_name: str):
        """指定レイヤーの名前を変更"""
        if layer_id in self.layers and new_name.strip():
            self.layers[layer_id].name = new_name.strip()
            self._sync_mask_from_ids()
            self.layer_list_changed.emit()

    def set_layer_color(self, layer_id: int, color: Tuple[float, float, float], use_layer_color: bool = True):
        """指定レイヤーの表示色を変更"""
        if layer_id in self.layers:
            self.layers[layer_id].color = (float(color[0]), float(color[1]), float(color[2]))
            self.layers[layer_id].use_layer_color = bool(use_layer_color)
            self._update_active_colors()
            self._update_display_colors_only()
            self.update()
            self.layer_list_changed.emit()

    def move_indices_to_layer(self, indices: np.ndarray, target_layer_id: int, respect_lock: bool = True) -> int:
        """
        指定されたインデックスの点を target_layer_id のレイヤーへ移動する。
        respect_lock=True の場合、ロックされているレイヤーに属する点は保護され移動しない。
        また移動先レイヤー自体がロックされている場合も移動しない。
        戻り値: 実際に移動した点数
        """
        if self.points_raw is None or len(indices) == 0:
            return 0
        self._ensure_layer_sync()
        if target_layer_id not in self.layers:
            self.add_custom_layer(f"レイヤー {target_layer_id}", layer_id=target_layer_id)

        if respect_lock and self.layers[target_layer_id].locked:
            self.status_changed.emit(f"移動先レイヤー「{self.layers[target_layer_id].name}」がロックされているため移動できません")
            return 0

        idx = np.asarray(indices, dtype=np.int64)
        idx = idx[(idx >= 0) & (idx < len(self.points_raw))]
        if len(idx) == 0:
            return 0

        if respect_lock:
            unlocked_mask = self._get_unlocked_mask()
            idx = idx[unlocked_mask[idx]]
            if len(idx) == 0:
                self.status_changed.emit("対象点がロック中レイヤーで保護されているため変更されませんでした")
                return 0

        self.layer_ids[idx] = int(target_layer_id)
        self._sync_mask_from_ids()
        # レイヤー別カラーモード以外では元の色配列は不変のため、全点の再計算をスキップして高速化
        if self.color_mode == self.COLOR_MODE_LAYER or self.active_colors is None or len(self.active_colors) != len(self.points_raw):
            self._update_active_colors()
        self._update_display_arrays()
        self.update()
        self._emit_layer_counts()
        return len(idx)

    def batch_move_indices_to_layers(
        self,
        moves: list,
        respect_lock: bool = True,
    ) -> int:
        """
        複数レイヤーへの点移動を一括（バッチ）で高速実行する。
        各移動ごとに全点VBOやカラーを再構築するのを防ぎ、全てのレイヤーIDを代入した後に
        1回だけマスク同期・VBO構築・再描画を行うことで劇的に高速化する。

        :param moves: [(点インデックス配列, 移動先layer_id), ...] のリスト
        :param respect_lock: ロック中レイヤーの点を保護するかどうか
        :return: 実際に移動した総点数
        """
        if self.points_raw is None or not moves:
            return 0
        self._ensure_layer_sync()

        total_moved = 0
        n_pts = len(self.points_raw)
        unlocked_mask = self._get_unlocked_mask() if respect_lock else None

        for indices, target_layer_id in moves:
            if indices is None or len(indices) == 0:
                continue
            target_layer_id = int(target_layer_id)
            if target_layer_id not in self.layers:
                self.add_custom_layer(f"レイヤー {target_layer_id}", layer_id=target_layer_id)

            if respect_lock and self.layers[target_layer_id].locked:
                continue

            idx = np.asarray(indices, dtype=np.int64)
            idx = idx[(idx >= 0) & (idx < n_pts)]
            if len(idx) == 0:
                continue

            if unlocked_mask is not None:
                idx = idx[unlocked_mask[idx]]
                if len(idx) == 0:
                    continue

            self.layer_ids[idx] = target_layer_id
            total_moved += len(idx)

        if total_moved > 0:
            self._sync_mask_from_ids()
            if self.color_mode == self.COLOR_MODE_LAYER or self.active_colors is None or len(self.active_colors) != n_pts:
                self._update_active_colors()
            self._update_display_arrays()
            self.update()
            self._emit_layer_counts()

        return total_moved

    def move_indices_to_noise(self, indices: np.ndarray, respect_lock: bool = True):
        """指定されたインデックスの点をノイズレイヤーへ移動 (手動削除 / 自動検出)"""
        self.move_indices_to_layer(indices, self.LAYER_NOISE, respect_lock=respect_lock)

    def restore_indices_to_normal(self, indices: np.ndarray, respect_lock: bool = True):
        """指定されたインデックスの点を通常レイヤーへ復元"""
        self.move_indices_to_layer(indices, self.LAYER_NORMAL, respect_lock=respect_lock)

    def get_normal_indices(self) -> np.ndarray:
        """現在有効レイヤー(ノイズ以外すべて)に属する点の原本インデックスを取得"""
        self._ensure_layer_sync()
        if self.layer_mask is None:
            return np.arange(len(self.points_raw)) if self.points_raw is not None else np.empty(0, dtype=np.int64)
        return np.where(self.layer_mask)[0]

    def get_unlocked_normal_indices(self) -> np.ndarray:
        """現在有効レイヤーかつロックされていない点の原本インデックスを取得 (自動フィルタ・自動分類の対象用)"""
        self._ensure_layer_sync()
        if self.layer_mask is None:
            return np.empty(0, dtype=np.int64)
        mask = self.layer_mask & self._get_unlocked_mask()
        return np.where(mask)[0]

    def get_layer_indices(self, layer_id: int) -> np.ndarray:
        """指定したlayer_idに属する点の原本インデックスを取得"""
        self._ensure_layer_sync()
        if self.layer_ids is None:
            return np.empty(0, dtype=np.int64)
        return np.where(self.layer_ids == int(layer_id))[0]

    def get_visible_indices(self) -> np.ndarray:
        """現在表示ONになっているレイヤー＆スキャンの点インデックスを取得"""
        if self.points_raw is None:
            return np.empty(0, dtype=np.int64)
        vis_mask = self._get_layer_and_scan_visibility_mask()
        return np.where(vis_mask)[0]

    def get_all_layer_masks(self, include_empty: bool = False, include_noise: bool = False) -> Dict[str, np.ndarray]:
        """全レイヤーの {レイヤー名: boolマスク} 辞書を返す (レイヤー別一括分割E57保存用)"""
        self._ensure_layer_sync()
        if self.layer_ids is None:
            return {}
        result: Dict[str, np.ndarray] = {}
        for lid, linfo in self.layers.items():
            if not include_noise and self.is_noise_layer(lid):
                continue
            mask = (self.layer_ids == lid)
            if include_empty or np.any(mask):
                key = f"{linfo.name}_ID{lid}" if linfo.name in result else linfo.name
                result[key] = mask
        return result

    def get_layer_counts(self) -> Dict[int, int]:
        """各レイヤーIDごとの点数を辞書で返す"""
        self._ensure_layer_sync()
        if self.layer_ids is None:
            return {lid: 0 for lid in self.layers}
        unique_ids, counts = np.unique(self.layer_ids, return_counts=True)
        count_map = {int(k): int(v) for k, v in zip(unique_ids, counts)}
        return {lid: count_map.get(lid, 0) for lid in self.layers}

    def set_scan_visibility(self, scan_id: int, visible: bool):
        """特定のスキャン(器械点)の表示/非表示を切り替え"""
        self.scan_visibility[int(scan_id)] = bool(visible)
        self._update_display_arrays()
        self.update()

    def get_scan_counts(self) -> Dict[int, int]:
        """各スキャン(器械点)IDごとの点数を辞書で返す"""
        if self.scan_ids is None:
            return {}
        unique_ids, counts = np.unique(self.scan_ids, return_counts=True)
        return {int(k): int(v) for k, v in zip(unique_ids, counts)}

    def save_layer_project(self, filepath: str) -> None:
        """作業途中のレイヤー状態(layer_ids, レイヤー定義, スキャン表示設定)を.npzファイルに保存"""
        self._ensure_layer_sync()
        if self.points_raw is None or self.layer_ids is None:
            raise ValueError("保存する点群データがありません")

        layers_meta = [asdict(linfo) for linfo in self.layers.values()]
        scan_vis_meta = {str(k): bool(v) for k, v in self.scan_visibility.items()}
        meta_json = json.dumps(
            {
                "point_count": len(self.points_raw),
                "layers": layers_meta,
                "scan_visibility": scan_vis_meta,
            },
            ensure_ascii=False,
        )
        np.savez_compressed(
            filepath,
            layer_ids=self.layer_ids,
            meta_json=np.array(meta_json),
        )

    def load_layer_project(self, filepath: str) -> None:
        """保存されたレイヤー状態(.npz)を読み込んで現在の点群に復元"""
        if self.points_raw is None:
            raise ValueError("先に点群ファイルを読み込んでください")

        with np.load(filepath, allow_pickle=False) as data:
            loaded_ids = np.asarray(data["layer_ids"], dtype=np.int32)
            if len(loaded_ids) != len(self.points_raw):
                raise ValueError(
                    f"レイヤープロジェクトの点数({len(loaded_ids):,}点)が現在の点群点数({len(self.points_raw):,}点)と一致しません"
                )
            meta_str = str(data["meta_json"].item())
            meta = json.loads(meta_str)

        new_layers: Dict[int, LayerInfo] = {}
        for item in meta.get("layers", []):
            lid = int(item["layer_id"])
            col = tuple(float(c) for c in item.get("color", (0.2, 0.8, 0.5)))
            lname = str(item.get("name", f"レイヤー {lid}"))
            is_noise_flag = bool(
                item.get("is_noise", lid == self.LAYER_NOISE or lname.startswith("自動除去:"))
            )
            new_layers[lid] = LayerInfo(
                layer_id=lid,
                name=lname,
                visible=bool(item.get("visible", not is_noise_flag)),
                locked=bool(item.get("locked", False)),
                color=col,
                use_layer_color=bool(item.get("use_layer_color", lid != self.LAYER_NORMAL)),
                is_noise=is_noise_flag,
            )

        if self.LAYER_NORMAL not in new_layers:
            new_layers[self.LAYER_NORMAL] = LayerInfo(
                self.LAYER_NORMAL, "通常点群", True, False, (0.22, 0.74, 0.97), False, False
            )
        if self.LAYER_NOISE not in new_layers:
            new_layers[self.LAYER_NOISE] = LayerInfo(
                self.LAYER_NOISE, "ノイズ (除去点)", False, False, (0.95, 0.25, 0.25), True, True
            )

        self.layers = new_layers
        self.layer_ids = loaded_ids.copy()
        self._sync_mask_from_ids()
        self.show_normal_layer = self.layers[self.LAYER_NORMAL].visible
        self.show_noise_layer = self.layers[self.LAYER_NOISE].visible

        for k_str, vis in meta.get("scan_visibility", {}).items():
            sid = int(k_str)
            if sid in self.scan_visibility:
                self.scan_visibility[sid] = bool(vis)

        self._update_active_colors()
        self._update_display_arrays()
        self.update()
        self._emit_layer_counts()

    def get_outside_section_normal_indices(self) -> np.ndarray:
        """現在のXYZ断面スライス・2点斜め壁スライス範囲外にあるロックされていない有効レイヤー点のインデックスを取得 (断面外クロップ用)"""
        if self.points_raw is None or len(self.points_raw) == 0:
            return np.empty(0, dtype=np.int64)
        if not (self.clip_x_enabled or self.clip_y_enabled or self.clip_z_enabled or self.clip_wall_enabled):
            return np.empty(0, dtype=np.int64)

        self._ensure_layer_sync()
        in_sec = self._get_section_inlier_mask()

        outside_mask = (~in_sec) & self._get_unlocked_mask()
        if self.layer_mask is not None:
            outside_mask &= self.layer_mask
        return np.where(outside_mask)[0]

    # --- 断面スライサー制御 (シーク基点座標・厚み・カメラ追従・断面正対ビュー対応) ---
    def set_section_x(self, enabled: bool, min_val: float, max_val: float, origin_val: Optional[float] = None, is_preview: bool = False):
        self.clip_x_enabled = enabled
        self.clip_x_range = [min_val, max_val]
        self.clip_x_origin = float(origin_val) if origin_val is not None else 0.5 * (min_val + max_val)
        if is_preview:
            self._update_display_arrays_fast_preview()
        else:
            self._update_display_arrays()
        self.update()

    def set_section_y(self, enabled: bool, min_val: float, max_val: float, origin_val: Optional[float] = None, is_preview: bool = False):
        self.clip_y_enabled = enabled
        self.clip_y_range = [min_val, max_val]
        self.clip_y_origin = float(origin_val) if origin_val is not None else 0.5 * (min_val + max_val)
        if is_preview:
            self._update_display_arrays_fast_preview()
        else:
            self._update_display_arrays()
        self.update()

    def set_section_z(self, enabled: bool, min_val: float, max_val: float, origin_val: Optional[float] = None, is_preview: bool = False):
        self.clip_z_enabled = enabled
        self.clip_z_range = [min_val, max_val]
        self.clip_z_origin = float(origin_val) if origin_val is not None else 0.5 * (min_val + max_val)
        if is_preview:
            self._update_display_arrays_fast_preview()
        else:
            self._update_display_arrays()
        self.update()

    def get_current_focus_world_point(self) -> np.ndarray:
        """
        現在注目しているワールド座標 (X, Y, Z) を返す。
        - 点群が選択されている場合: 選択点群の重心座標
        - 未選択の場合: 現在のカメラ注視点 (camera_target + center_offset) を点群範囲内にクランプした座標
        """
        if self.points_raw is not None and len(self.points_raw) > 0:
            if len(self.selected_indices) > 0:
                valid_sel = self.selected_indices[(self.selected_indices >= 0) & (self.selected_indices < len(self.points_raw))]
                if len(valid_sel) > 0:
                    return np.mean(self.points_raw[valid_sel], axis=0).astype(np.float64)
            cam_world = self.center_offset + np.array(
                [self.camera_target.x(), self.camera_target.y(), self.camera_target.z()], dtype=np.float64
            )
            return np.clip(cam_world, self.world_bounds_min, self.world_bounds_max)
        return np.zeros(3, dtype=np.float64)

    def focus_camera_on_world_axis(self, axis: int, world_val: float):
        """指定した軸 (0:X, 1:Y, 2:Z) のシーク基点ワールド座標へカメラ注視点を追従移動する"""
        if self.points_raw is None:
            return
        local_val = float(world_val - self.center_offset[axis])
        if axis == 0:
            self.camera_target.setX(local_val)
        elif axis == 1:
            self.camera_target.setY(local_val)
        elif axis == 2:
            self.camera_target.setZ(local_val)
        self.update()

    def align_view_to_section(self, axis: str):
        """
        指定した断面軸 ('x', 'y', 'z') に対してカメラを正対させ、シーク基点を中心に断面図として観察しやすくする
        - X断面: 側面視 (YZ平面を正面から観察)
        - Y断面: 正面視 (XZ立面を正面から観察)
        - Z断面: 上面視 (XY水平輪切り断面を真上から観察)
        """
        if self.points_raw is not None:
            if self.clip_x_enabled:
                self.camera_target.setX(float(self.clip_x_origin - self.center_offset[0]))
            if self.clip_y_enabled:
                self.camera_target.setY(float(self.clip_y_origin - self.center_offset[1]))
            if self.clip_z_enabled:
                self.camera_target.setZ(float(self.clip_z_origin - self.center_offset[2]))

        ax = axis.lower()
        if ax == "x":
            self.set_view_side()
        elif ax == "y":
            self.set_view_front()
        elif ax == "z":
            self.set_view_top()
        self.update()

    # --- 🧭 2点ピック斜め壁アライン (直交・横・縦表示 & 斜め壁断面スライス) ---
    def set_section_wall(self, enabled: bool, min_val: float, max_val: float, origin_val: Optional[float] = None, is_preview: bool = False):
        """2点ピック斜め壁ラインに対する直交方向の断面スライス範囲を設定"""
        self.clip_wall_enabled = bool(enabled and self.wall_align_angle_deg is not None)
        self.clip_wall_range = [float(min_val), float(max_val)]
        self.clip_wall_origin = float(origin_val) if origin_val is not None else 0.5 * (float(min_val) + float(max_val))
        if is_preview:
            self._update_display_arrays_fast_preview()
        else:
            self._update_display_arrays()
        self.update()

    def set_wall_align_points(self, p1_world: np.ndarray, p2_world: np.ndarray, initial_view: str = "horiz") -> float:
        """
        点群上の2点 (p1_world, p2_world) から斜め壁の水平角度・中心座標を算出し、指定の直交・平行視点へ切り替える
        """
        p1_w = np.asarray(p1_world, dtype=np.float64)
        p2_w = np.asarray(p2_world, dtype=np.float64)
        p1_c = (p1_w - self.center_offset).astype(np.float32)
        p2_c = (p2_w - self.center_offset).astype(np.float32)
        self.wall_align_points = [(p1_c, p1_w), (p2_c, p2_w)]
        self.wall_align_mode = False

        dx = float(p2_w[0] - p1_w[0])
        dy = float(p2_w[1] - p1_w[1])
        if math.hypot(dx, dy) < 1e-6:
            dx, dy = 1.0, 0.0
        self.wall_align_angle_deg = math.degrees(math.atan2(dy, dx))
        self.wall_align_center_world = 0.5 * (p1_w + p2_w)
        self.apply_wall_aligned_view(initial_view, auto_ortho=True)
        self.wall_aligned.emit(float(self.wall_align_angle_deg), p1_w, p2_w)
        return float(self.wall_align_angle_deg)

    def apply_wall_aligned_view(self, view_type: str = "horiz", auto_ortho: bool = True) -> bool:
        """
        ピック済み2点ライン（または直前の計測2点）を基準に、直交・横・縦表示へ切り替える
        - "horiz"      : ━ 横表示 (上面視で斜め壁が画面の水平[横]になり、直交方向が上下[縦])
        - "vert"       : ┃ 縦表示 (上面視で斜め壁が画面の垂直[縦]になり、直交方向が左右[横])
        - "ortho_front": 🧱 直交正面 (水平から斜め壁に対して直交正対: 横=壁方向, 縦=高さZ)
        - "edge"       : 🔪 壁断面・真横 (斜め壁の延長方向から真横に見る: 壁が縦1本線になり表裏ノイズを分離)
        """
        if self.wall_align_angle_deg is None or self.wall_align_center_world is None:
            # もし計測ツールで2点ピック済みなら、それを自動的に壁基準点として採用する
            if len(self.measure_points) >= 2:
                self.set_wall_align_points(self.measure_points[0][1], self.measure_points[1][1], initial_view=view_type)
                return True
            return False

        self.wall_align_view_type = view_type
        if self.points_raw is not None:
            local_c = self.wall_align_center_world - self.center_offset
            self.camera_target = QVector3D(float(local_c[0]), float(local_c[1]), float(local_c[2]))

        if auto_ortho and not self.is_orthographic:
            self.is_orthographic = True
            self.projection_changed.emit(True)

        alpha = float(self.wall_align_angle_deg)
        if view_type == "horiz":
            # 真上から見下ろし、2点ラインが画面の横(水平)一文字に揃う
            self.pitch = 89.9
            self.yaw = alpha
            mode_label = "━ 横表示 (上面・壁を水平配置)"
        elif view_type == "vert":
            # 真上から見下ろし、2点ラインが画面の縦(垂直)一文字に揃う
            self.pitch = 89.9
            self.yaw = alpha - 90.0
            mode_label = "┃ 縦表示 (上面・壁を垂直配置)"
        elif view_type == "ortho_front":
            # 水平から2点ラインに対して直交方向に壁正面を見る (横=壁、縦=高さZ)
            self.pitch = 0.0
            self.yaw = alpha
            mode_label = "🧱 直交正面 (壁に水平正対・横=壁/縦=Z)"
        elif view_type == "edge":
            # 水平から2点ライン延長方向に壁を真横(エッジ断面)で見る (横=壁直交法線、縦=高さZ)
            self.pitch = 0.0
            self.yaw = alpha + 90.0
            mode_label = "🔪 壁断面・真横 (壁の延長方向からエッジ視)"
        else:
            self.pitch = 89.9
            self.yaw = alpha
            mode_label = "━ 横表示"

        self.status_changed.emit(f"🧭 2点壁アライン ({alpha:.1f}°): {mode_label}")
        self.update()
        return True

    def flip_wall_aligned_view(self) -> bool:
        """現在の2点壁アライン基準角を 180° 反転させる (表裏・上下左右の反転)"""
        if self.wall_align_angle_deg is None:
            return False
        new_ang = float(self.wall_align_angle_deg) + 180.0
        if new_ang > 180.0:
            new_ang -= 360.0
        self.wall_align_angle_deg = new_ang
        return self.apply_wall_aligned_view(self.wall_align_view_type, auto_ortho=False)

    def clear_wall_alignment(self):
        """2点ピック斜め壁の基準ピンおよび斜め壁スライスをクリア"""
        self.wall_align_mode = False
        self.wall_align_points.clear()
        self.wall_align_angle_deg = None
        self.wall_align_center_world = None
        if self.clip_wall_enabled:
            self.clip_wall_enabled = False
            self._update_display_arrays()
        self.update()

    def set_point_budget(self, budget: int):
        self.point_budget = budget
        self._update_display_arrays()
        self.update()

    def is_section_active(self) -> bool:
        """XYZ 断面または 2 点斜め壁スライスのいずれかが有効か"""
        return bool(self.clip_x_enabled or self.clip_y_enabled or self.clip_z_enabled or self.clip_wall_enabled)

    def set_section_full_detail(self, enabled: bool):
        """断面ボックス内の全点表示（表示予算で間引かない。上限 SECTION_DETAIL_MAX_POINTS）の ON/OFF"""
        enabled = bool(enabled)
        if self.section_full_detail == enabled:
            return
        self.section_full_detail = enabled
        if self.is_section_active():
            self._update_display_arrays()
            self.update()

    # --- 断面ボックスの元ファイル読み直し（表示専用の重ね描き） ---
    def section_box_spec(self) -> Optional[dict]:
        """
        現在の断面ボックスを表示座標系（pose 適用後・座標系変換後のワールド座標）で返す。
        core.section_reload.SectionBox(**spec) にそのまま渡せる。断面が無効なら None。
        """
        if not self.is_section_active():
            return None
        spec: dict = {"x_range": None, "y_range": None, "z_range": None, "wall": None}
        if self.clip_x_enabled:
            spec["x_range"] = (float(self.clip_x_range[0]), float(self.clip_x_range[1]))
        if self.clip_y_enabled:
            spec["y_range"] = (float(self.clip_y_range[0]), float(self.clip_y_range[1]))
        if self.clip_z_enabled:
            spec["z_range"] = (float(self.clip_z_range[0]), float(self.clip_z_range[1]))
        if self.clip_wall_enabled and self.wall_align_angle_deg is not None and self.wall_align_center_world is not None:
            spec["wall"] = (
                float(self.wall_align_angle_deg),
                float(self.wall_align_center_world[0]),
                float(self.wall_align_center_world[1]),
                float(self.clip_wall_range[0]),
                float(self.clip_wall_range[1]),
            )
        return spec

    def _section_box_key(self):
        """重ね描きがどの断面ボックスのものかを比べるためのキー（断面無効なら None）"""
        spec = self.section_box_spec()
        if spec is None:
            return None
        return (spec["x_range"], spec["y_range"], spec["z_range"], spec["wall"])

    @property
    def has_section_overlay(self) -> bool:
        return self.overlay_points is not None and len(self.overlay_points) > 0

    def set_section_overlay(
        self,
        points_world: np.ndarray,
        colors: Optional[np.ndarray] = None,
        intensities: Optional[np.ndarray] = None,
        scan_ids: Optional[np.ndarray] = None,
        n_in_box: int = 0,
        capped: bool = False,
    ):
        """
        断面ボックスを元ファイルから全点で読み直した点（表示座標）を、表示専用に重ね描きする。
        現在の断面ボックスに紐づき、ボックスが変わる・断面を解除する・点群を入れ替えると自動で消える。
        選択・ノイズ移動・計測の対象にはならない。
        """
        if self.points_centered is None or points_world is None or len(points_world) == 0 or not self.is_section_active():
            self.clear_section_overlay()
            self.update()
            return
        pts_c = np.ascontiguousarray(np.asarray(points_world, dtype=np.float64) - self.center_offset, dtype=np.float32)
        n = len(pts_c)
        rgb = np.asarray(colors, dtype=np.float32) if colors is not None else None
        if rgb is not None and (rgb.ndim != 2 or rgb.shape != (n, 3)):
            rgb = None
        ints = np.asarray(intensities, dtype=np.float32) if intensities is not None else None
        if ints is not None and ints.shape != (n,):
            ints = None
        sids = np.asarray(scan_ids, dtype=np.int32) if scan_ids is not None else None
        if sids is not None and sids.shape != (n,):
            sids = None
        self._overlay_source = (pts_c, rgb, ints, sids)
        self.overlay_points = pts_c
        self.overlay_box_total = int(n_in_box)
        self.overlay_capped = bool(capped)
        self._overlay_box_key = self._section_box_key()
        self.overlay_kind = "section"
        self._recolor_overlay()
        self.update()

    def set_view_overlay(
        self,
        points_centered: np.ndarray,
        colors: Optional[np.ndarray] = None,
        intensities: Optional[np.ndarray] = None,
        n_total: int = 0,
        capped: bool = False,
    ):
        """
        視点連動の高精細表示（LOD）の点を表示専用に重ね描きする。points_centered は中心オフセット済みの float32 (M, 3)。
        断面が無くても表示できる。断面ボックスが変わる・点群を入れ替えると消える（作り直しは gui/lod_controller.py）。
        """
        if self.points_centered is None or points_centered is None or len(points_centered) == 0:
            self.clear_section_overlay()
            self.update()
            return
        pts_c = np.ascontiguousarray(points_centered, dtype=np.float32)
        n = len(pts_c)
        rgb = np.asarray(colors, dtype=np.float32) if colors is not None else None
        if rgb is not None and rgb.shape != (n, 3):
            rgb = None
        ints = np.asarray(intensities, dtype=np.float32) if intensities is not None else None
        if ints is not None and ints.shape != (n,):
            ints = None
        self._overlay_source = (pts_c, rgb, ints, None)
        self.overlay_points = pts_c
        self.overlay_box_total = int(n_total)
        self.overlay_capped = bool(capped)
        self._overlay_box_key = self._section_box_key()
        self.overlay_kind = "lod"
        self._recolor_overlay()
        self.update()

    def camera_mvp(self) -> Tuple[np.ndarray, int, int]:
        """
        現在のカメラの (投影 × ビュー行列 4×4, 幅 px, 高さ px)。paintGL と同じ投影。clip = mvp @ [x, y, z, 1]
        （x, y, z は中心オフセット済み座標）。高精細表示（LOD）が画面に入る範囲と広さを求めるのに使う。
        """
        w, h = max(1, self.width()), max(1, self.height())
        aspect = w / h
        near_plane = max(0.1, self.camera_distance * 0.01)
        far_plane = max(500.0, self.camera_distance * 50.0)
        proj = QMatrix4x4()
        if self.is_orthographic:
            ortho_size = max(0.5, self.camera_distance * 0.35)
            proj.ortho(-ortho_size * aspect, ortho_size * aspect, -ortho_size, ortho_size, -far_plane, far_plane)
        else:
            proj.perspective(self.fov, aspect, near_plane, far_plane)
        m = proj * self._get_view_matrix()
        return np.array(m.copyDataTo(), dtype=np.float64).reshape(4, 4), int(w), int(h)

    def clear_section_overlay(self):
        """重ね描きを消す（断面ボックスの変更・断面解除・点群入替・ユーザー操作）"""
        if self.overlay_points is None and self._overlay_source is None:
            return
        self.overlay_points = None
        self.overlay_colors = None
        self._overlay_source = None
        self.overlay_box_total = 0
        self.overlay_capped = False
        self._overlay_box_key = None
        self.overlay_kind = ""
        self._vbo_overlay_dirty = True

    def _drop_overlay_if_box_changed(self):
        """断面ボックスが重ね描きを作ったときと違えば消す（表示配列の再構築・プレビューのたびに確認）"""
        if self._overlay_source is not None and self._section_box_key() != self._overlay_box_key:
            self.clear_section_overlay()

    def _recolor_overlay(self):
        """重ね描きの色を現在の配色モード（RGB / 高さ / 反射強度 / スキャン別 / 単色）と陰影に合わせて作り直す"""
        if self._overlay_source is None:
            return
        pts_c, rgb, ints, sids = self._overlay_source
        n = len(pts_c)
        mode = self.color_mode

        def _t_height() -> np.ndarray:
            z_min, z_max = self._height_range()
            return np.clip((pts_c[:, 2] - z_min) / max(1e-5, z_max - z_min), 0.0, 1.0).astype(np.float32)

        def _t_int() -> np.ndarray:
            p_low, p_high = self._intensity_stretch()
            return np.clip((ints - p_low) / max(1e-5, p_high - p_low), 0.0, 1.0).astype(np.float32)

        if mode in (self.COLOR_MODE_RGB, "RGB", "RGB (実色写真)"):
            if rgb is not None:
                cols = rgb.copy()
            elif ints is not None:
                cols = self._colormap_grayscale(_t_int())
            else:
                cols = self._colormap_turbo(_t_height())
        elif self._is_height_mode(mode):
            cols = self._height_colormap(_t_height(), mode)
        elif self._is_intensity_mode(mode):
            cols = self._intensity_colormap(_t_int(), mode) if ints is not None else self._colormap_turbo(_t_height())
        elif mode in (self.COLOR_MODE_SCAN, "スキャン別 (器械点)") and sids is not None:
            cols = self.SCAN_PALETTE[np.mod(np.maximum(0, sids), len(self.SCAN_PALETTE))].copy()
        elif mode in (self.COLOR_MODE_LAYER, "レイヤー別色分け") and self.LAYER_NORMAL in self.layers:
            cols = np.tile(np.asarray(self.layers[self.LAYER_NORMAL].color, dtype=np.float32), (n, 1))
        else:
            cols = np.tile(np.asarray([0.3, 0.8, 0.95], dtype=np.float32), (n, 1))

        # 陰影は「隣の点＝スキャン順で隣」を前提にした計算なので、空間順に並べ替えた LOD の重ね描きには掛けない
        if getattr(self, "shading_enabled", False) and self.overlay_kind != "lod":
            shade = self._shading_factors_of(pts_c, _t_int() if ints is not None else None)
            factor = 1.0 + (shade - 1.0) * float(getattr(self, "shading_strength", 0.65))
            cols = np.clip(cols * factor[:, np.newaxis], 0.0, 1.0)

        # 通常レイヤーの固有色（use_layer_color）は本体でもレイヤー別以外で上書きされるので合わせる
        normal_layer = self.layers.get(self.LAYER_NORMAL)
        if normal_layer is not None and normal_layer.use_layer_color and self.color_mode != self.COLOR_MODE_LAYER:
            cols[:] = np.asarray(normal_layer.color, dtype=np.float32)

        self.overlay_colors = np.ascontiguousarray(cols, dtype=np.float32)
        self._vbo_overlay_dirty = True

    def _draw_section_overlay(self):
        """
        重ね描き（元ファイル読み直し）を点群と同じ点サイズ・フォグで描く。
        paintGL の点群描画ブロック内（VERTEX/COLOR ClientState 有効）から呼ばれる前提。VBO が使えなければクライアント配列で描く。
        """
        pts = self.overlay_points
        cols = self.overlay_colors
        if pts is None or cols is None or len(pts) == 0 or len(cols) != len(pts):
            return
        try:
            if self._vbo_overlay is None:
                self._vbo_overlay = gl.glGenBuffers(1)
                self._vbo_overlay_colors = gl.glGenBuffers(1)
                self._vbo_overlay_dirty = True
            if self._vbo_overlay_dirty:
                self._vbo_overlay_count = len(pts)
                gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._vbo_overlay)
                gl.glBufferData(gl.GL_ARRAY_BUFFER, pts.nbytes, pts, gl.GL_DYNAMIC_DRAW)
                gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._vbo_overlay_colors)
                gl.glBufferData(gl.GL_ARRAY_BUFFER, cols.nbytes, cols, gl.GL_DYNAMIC_DRAW)
                self._vbo_overlay_dirty = False
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._vbo_overlay)
            gl.glVertexPointer(3, gl.GL_FLOAT, 0, None)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._vbo_overlay_colors)
            gl.glColorPointer(3, gl.GL_FLOAT, 0, None)
            # 点群本体と同じく、回転・パン中は点数を半減して追従を軽くする
            render_count = self._vbo_overlay_count
            if (self.is_rotating or self.is_panning) and render_count > 800000:
                render_count = max(400000, render_count // 2)
            gl.glDrawArrays(gl.GL_POINTS, 0, render_count)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        except Exception:
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
            gl.glVertexPointer(3, gl.GL_FLOAT, 0, pts)
            gl.glColorPointer(3, gl.GL_FLOAT, 0, cols)
            gl.glDrawArrays(gl.GL_POINTS, 0, len(pts))

    def fit_to_screen(self):
        """全体フィット (F)"""
        if self.bounds_min is None or self.bounds_max is None:
            return
        diag = np.linalg.norm(self.bounds_max - self.bounds_min)
        self.camera_target = QVector3D(0.0, 0.0, 0.0)
        self.camera_distance = max(2.0, float(diag) * 1.5)
        self.yaw = 45.0
        self.pitch = 30.0
        self.update()
        self.status_changed.emit("全体フィット実行 (F)")

    def toggle_projection(self):
        """透視投影 / 正射影 切替 (P)"""
        self.is_orthographic = not self.is_orthographic
        self.projection_changed.emit(self.is_orthographic)
        mode_str = "正射影" if self.is_orthographic else "透視投影"
        self.status_changed.emit(f"投影切替: {mode_str} (P)")
        self.update()

    def set_fov(self, fov: float):
        """視野角(FOV)を設定 (15°〜90°)"""
        self.fov = float(np.clip(fov, 15.0, 90.0))
        self.update()

    def set_depth_fog_enabled(self, enabled: bool):
        """深度フォグ・立体感強調の有効/無効"""
        self.depth_fog_enabled = bool(enabled)
        self.update()

    def set_shading_enabled(self, enabled: bool):
        """① 凹凸・エッジ輪郭強調 (疑似EDL/陰影コントラスト強調) の有効/無効"""
        self.shading_enabled = bool(enabled)
        self._update_active_colors()
        self._update_display_colors_only()
        self.update()
        state_str = "有効" if self.shading_enabled else "無効"
        self.status_changed.emit(f"✨ 凹凸・エッジ陰影強調: {state_str}")

    def set_adaptive_point_size(self, enabled: bool):
        """② 距離連動ポイントサイズ自動スケーリングの有効/無効"""
        self.adaptive_point_size = bool(enabled)
        self.update()
        state_str = "有効 (ズームインで自動拡大)" if self.adaptive_point_size else "固定サイズ"
        self.status_changed.emit(f"🔍 距離連動ポイントサイズ: {state_str}")

    def animate_to_view(self, target_pitch: float, target_yaw: float):
        """⑦ 指定したピッチ・ヨー角へ滑らかにアニメーション回転"""
        self._cam_anim_start_pitch = float(self.pitch)
        self._cam_anim_start_yaw = float(self.yaw)
        self._cam_anim_target_pitch = float(target_pitch)

        # 角度差を最短ルート (-180°〜+180°) に補正
        diff_yaw = (float(target_yaw) - self.yaw + 180.0) % 360.0 - 180.0
        self._cam_anim_target_yaw = self.yaw + diff_yaw

        self._cam_anim_step = 0
        self._cam_anim_total_steps = 10
        self._cam_anim_timer.start()

    def _step_camera_animation(self):
        """カメラ回転アニメーションのステップ更新 (Cubic Ease-Out)"""
        self._cam_anim_step += 1
        p = min(1.0, self._cam_anim_step / max(1, self._cam_anim_total_steps))
        # 3次Ease-Out (滑らかに減速して停止)
        ease = 1.0 - (1.0 - p) ** 3

        self.pitch = self._cam_anim_start_pitch + (self._cam_anim_target_pitch - self._cam_anim_start_pitch) * ease
        self.yaw = self._cam_anim_start_yaw + (self._cam_anim_target_yaw - self._cam_anim_start_yaw) * ease
        self.update()

        if self._cam_anim_step >= self._cam_anim_total_steps:
            self._cam_anim_timer.stop()
            self.pitch = self._cam_anim_target_pitch
            self.yaw = (self._cam_anim_target_yaw + 180.0) % 360.0 - 180.0
            self.update()

    def _check_gizmo_hit(self, pos: QPoint) -> Optional[str]:
        """画面座標がXYZ軸ギズモの各バッジ(または中心ハブ)上にあるかを判定"""
        px, py = float(pos.x()), float(pos.y())
        targets = getattr(self, "_gizmo_hit_targets", {})
        for label, (gx, gy, rad) in targets.items():
            if math.hypot(px - gx, py - gy) <= rad:
                return label
        return None

    def _on_gizmo_axis_clicked(self, axis: str):
        """⑦ 軸ギズモクリック時の視点切替ハンドラ"""
        if axis == "Z":
            # 上面 ⇄ 下面 のトグル切替
            if abs(self.pitch - 89.9) < 1.0:
                self.animate_to_view(-89.9, 0.0)
                self.status_changed.emit("視点切替: 下面 (-Z)")
            else:
                self.animate_to_view(89.9, 0.0)
                self.status_changed.emit("視点切替: 上面 (+Z / XY平面)")
        elif axis == "Y":
            # 正面 ⇄ 背面 のトグル切替
            if abs(self.pitch) < 1.0 and abs(self.yaw % 360.0) < 1.0:
                self.animate_to_view(0.0, 180.0)
                self.status_changed.emit("視点切替: 背面 (-Y)")
            else:
                self.animate_to_view(0.0, 0.0)
                self.status_changed.emit("視点切替: 正面 (+Y / XZ平面)")
        elif axis == "X":
            # 右側面 ⇄ 左側面 のトグル切替
            if abs(self.pitch) < 1.0 and abs(((self.yaw + 90.0) % 360.0)) < 1.0:
                self.animate_to_view(0.0, 90.0)
                self.status_changed.emit("視点切替: 左側面 (-X)")
            else:
                self.animate_to_view(0.0, -90.0)
                self.status_changed.emit("視点切替: 右側面 (+X / YZ平面)")
        elif axis == "ISO":
            self.animate_to_view(30.0, 45.0)
            self.status_changed.emit("視点切替: 等角 (3D鳥瞰)")

    def set_screenshot_include_grid(self, val: bool):
        """スクリーンショット時にグリッドを含めるかを設定"""
        self.screenshot_include_grid = bool(val)

    def set_screenshot_include_axes(self, val: bool):
        """スクリーンショット時にXYZ座標軸凡例を含めるかを設定"""
        self.screenshot_include_axes = bool(val)

    def capture_screenshot(
        self,
        save_path: Optional[str] = None,
        copy_to_clipboard: bool = True,
        include_hud: bool = False,
        include_grid: Optional[bool] = None,
        include_axes: Optional[bool] = None,
    ) -> Optional[QImage]:
        """
        ⑧ 高画質スクリーンショット保存機能 (クリップボード同時コピー対応)
        - include_grid: 背景グリッドを含めるか (Noneの場合は設定値 self.screenshot_include_grid)
        - include_axes: XYZ座標軸凡例を含めるか (Noneの場合は設定値 self.screenshot_include_axes)
        - 文字類 (上部HUD・操作ガイド・計測テキスト等) は常に自動で完全に省かれます
        """
        grid_val = self.screenshot_include_grid if include_grid is None else bool(include_grid)
        axes_val = self.screenshot_include_axes if include_axes is None else bool(include_axes)

        self._is_capturing_screenshot = True
        self._capture_override_grid = grid_val
        self._capture_override_axes = axes_val
        prev_hud_flag = getattr(self, "_hide_hud_for_capture", False)
        self._hide_hud_for_capture = not bool(include_hud)
        try:
            # OpenGLフレームバッファを直接高解像度キャプチャ
            if self.isValid():
                self.repaint()
                img = self.grabFramebuffer()
            else:
                # ヘッドレス環境や非表示時のフォールバック
                pixmap = self.grab()
                img = pixmap.toImage()
        finally:
            self._is_capturing_screenshot = False
            self._capture_override_grid = None
            self._capture_override_axes = None
            self._hide_hud_for_capture = prev_hud_flag
            self.update()

        if img is None or img.isNull():
            return None

        # クリップボードへコピー
        if copy_to_clipboard:
            clipboard = QApplication.clipboard()
            if clipboard is not None:
                clipboard.setImage(img)

        # ファイル保存
        if save_path:
            img.save(save_path)

        return img

    def set_point_shape(self, shape: str):
        self.point_shape = shape
        self.update()

    def set_show_grid(self, show: bool):
        self.show_grid = show
        self.update()

    def set_bg_color(self, color: QColor):
        self.bg_color = color
        self.update()

    def set_color_mode(self, mode: str):
        self.color_mode = mode
        if "Viridis" in mode:
            self.color_palette = self.PALETTE_VIRIDIS
        elif "Jet" in mode:
            self.color_palette = self.PALETTE_JET
        elif "Plasma" in mode:
            self.color_palette = self.PALETTE_PLASMA
        elif "地形" in mode or "Terrain" in mode:
            self.color_palette = self.PALETTE_TERRAIN
        elif "Heat" in mode or "温熱" in mode:
            self.color_palette = self.PALETTE_HEAT
        elif "Cool-Warm" in mode or "青白赤" in mode:
            self.color_palette = self.PALETTE_COOLWARM
        elif mode == self.COLOR_MODE_INTENSITY:
            self.color_palette = self.PALETTE_GRAYSCALE
        elif mode == self.COLOR_MODE_HEIGHT:
            self.color_palette = self.PALETTE_TURBO
        self._update_active_colors()
        self._update_display_colors_only()
        self.update()

    def set_color_palette(self, palette: str):
        """高さ(Z)または反射強度の色分けパレットを変更"""
        self.color_palette = palette.lower()
        self._update_active_colors()
        self._update_display_colors_only()
        self.update()

    def preview_custom_height_range(self, enabled: bool, min_z: float, max_z: float):
        """
        スライダードラッグ操作専用の超高速カラープレビュー (60fps遅延ゼロ)。
        原本全体(数千万点)の巨大配列の再計算をスキップし、現在画面に描画されている点群(display_points、通常100万〜200万点)
        のZ座標のみを対象に数ミリ秒でカラーマップを引き当ててVBOへ即時反映します。
        全体表示中であっても大容量点群であっても完全にヌルヌル追従します。
        """
        self.custom_height_enabled = bool(enabled)
        self.custom_height_min = float(min(min_z, max_z))
        self.custom_height_max = float(max(min_z, max_z))
        self._cached_normalized_heights = None

        if self.display_points is None or len(self.display_points) == 0:
            return

        mode = self.color_mode
        is_height_related = (
            "高さ" in mode
            or mode in (
                self.COLOR_MODE_HEIGHT,
                self.COLOR_MODE_HEIGHT_VIRIDIS,
                self.COLOR_MODE_HEIGHT_JET,
                self.COLOR_MODE_HEIGHT_PLASMA,
                self.COLOR_MODE_HEIGHT_TERRAIN,
                self.COLOR_MODE_HEIGHT_HEAT,
                self.COLOR_MODE_HEIGHT_COOLWARM,
            )
            or (mode == self.COLOR_MODE_RGB and self.colors_raw is None and self.intensities_raw is None)
            or (mode == self.COLOR_MODE_INTENSITY and self.intensities_raw is None)
        )
        if not is_height_related:
            return

        # 画面表示中の点群(display_points)のZ座標から直接高速正規化 (100万〜200万点なら約1ms)
        z = self.display_points[:, 2]
        cz = float(self.center_offset[2]) if self.center_offset is not None else 0.0
        if self.custom_height_enabled:
            z_min = float(self.custom_height_min - cz)
            z_max = float(self.custom_height_max - cz)
        else:
            z_min, z_max = float(np.min(z)), float(np.max(z))
        diff = max(1e-5, z_max - z_min)
        t = np.clip((z - z_min) / diff, 0.0, 1.0).astype(np.float32)

        # 1024段事前計算LUTから一括インデックス引き当て
        if "Viridis" in mode or self.color_palette == self.PALETTE_VIRIDIS:
            new_colors = self._colormap_viridis(t)
        elif "Jet" in mode or self.color_palette == self.PALETTE_JET:
            new_colors = self._colormap_jet(t)
        elif "Plasma" in mode or self.color_palette == self.PALETTE_PLASMA:
            new_colors = self._colormap_plasma(t)
        elif "地形" in mode or "Terrain" in mode or self.color_palette == self.PALETTE_TERRAIN:
            new_colors = self._colormap_terrain(t)
        elif "Heat" in mode or "温熱" in mode or self.color_palette == self.PALETTE_HEAT:
            new_colors = self._colormap_heat(t)
        elif "Cool-Warm" in mode or "青白赤" in mode or self.color_palette == self.PALETTE_COOLWARM:
            new_colors = self._colormap_coolwarm(t)
        elif self.color_palette == self.PALETTE_GRAYSCALE:
            new_colors = self._colormap_grayscale(t)
        else:
            new_colors = self._colormap_turbo(t)

        # 凹凸陰影(疑似EDL)のブレンド
        if getattr(self, "shading_enabled", False):
            if (
                self.display_orig_indices is not None
                and self._cached_shading_factors is not None
                and len(self._cached_shading_factors) == len(self.points_centered)
            ):
                sh = self._cached_shading_factors[self.display_orig_indices, np.newaxis]
                new_colors = np.clip(new_colors * sh, 0.0, 1.0).astype(np.float32)

        # 選択中ハイライトのブレンド
        if len(self.selected_indices) > 0 and self.display_orig_indices is not None:
            is_selected = np.isin(self.display_orig_indices, self.selected_indices)
            if np.any(is_selected):
                sel_col = np.array([0.95, 0.25, 0.25] if self.instant_delete_mode else [0.0, 0.85, 1.0], dtype=np.float32)
                new_colors[is_selected] = new_colors[is_selected] * 0.45 + sel_col * 0.55

        self.display_colors = np.ascontiguousarray(new_colors, dtype=np.float32)
        self._vbo_colors_dirty = True
        self.update()

    def set_custom_height_range(self, enabled: bool, min_z: float, max_z: float):
        """
        任意の高さ色分け範囲を設定し、原本全体を含めて完全確定同期。
        スライダー解放時や数値入力時に呼び出されます。
        """
        self.custom_height_enabled = bool(enabled)
        self.custom_height_min = float(min(min_z, max_z))
        self.custom_height_max = float(max(min_z, max_z))
        self._cached_normalized_heights = None

        # 高さ系カラーモード、またはRGB/強度フォールバックで高さを使っている場合にカラー即時更新
        mode = self.color_mode
        is_height_related = (
            "高さ" in mode
            or mode in (
                self.COLOR_MODE_HEIGHT,
                self.COLOR_MODE_HEIGHT_VIRIDIS,
                self.COLOR_MODE_HEIGHT_JET,
                self.COLOR_MODE_HEIGHT_PLASMA,
                self.COLOR_MODE_HEIGHT_TERRAIN,
                self.COLOR_MODE_HEIGHT_HEAT,
                self.COLOR_MODE_HEIGHT_COOLWARM,
            )
            or (mode == self.COLOR_MODE_RGB and self.colors_raw is None and self.intensities_raw is None)
            or (mode == self.COLOR_MODE_INTENSITY and self.intensities_raw is None)
        )
        if is_height_related:
            self._update_active_colors()
            self._update_display_colors_only()
            self.update()

    def _update_display_colors_only(self):
        """
        頂点座標の再サンプリングや重い断面判定を行わず、表示用カラー配列とカラーVBOのみを最速更新。
        カラー切り替えや選択ハイライト更新時のもっさり感を完全に解消します。
        """
        if (
            self.display_orig_indices is not None
            and self.active_colors is not None
            and len(self.display_orig_indices) > 0
            and self.display_points is not None
            and len(self.display_points) == len(self.display_orig_indices)
        ):
            self.display_colors = np.ascontiguousarray(self.active_colors[self.display_orig_indices], dtype=np.float32)
            self._vbo_colors_dirty = True
        else:
            self._update_display_arrays()

    def set_point_size(self, size: float):
        self.point_size = max(0.5, float(size))
        self.update()

    @classmethod
    def _init_colormap_luts(cls):
        """1024段の事前計算カラーマップ Look-Up Table (LUT) を構築 (初回1回のみ・計96KB)"""
        if getattr(cls, "_COLORMAP_LUTS", None) is not None:
            return
        t = np.linspace(0.0, 1.0, 1024, dtype=np.float32)
        luts = {}

        # 1. Turbo
        r = np.clip(1.5 - np.abs(t * 4.0 - 3.0), 0.0, 1.0)
        g = np.clip(1.5 - np.abs(t * 4.0 - 2.0), 0.0, 1.0)
        b = np.clip(1.5 - np.abs(t * 4.0 - 1.0), 0.0, 1.0)
        luts[cls.PALETTE_TURBO] = np.column_stack([r, g, b]).astype(np.float32)

        # 2. Viridis
        r = np.clip(0.267 + t * (-0.005 + t * (1.28 + t * (-0.55))), 0.0, 1.0)
        g = np.clip(0.004 + t * (1.01 + t * (-0.11)), 0.0, 1.0)
        b = np.clip(0.329 + t * (1.42 + t * (-2.65 + t * 1.05)), 0.0, 1.0)
        luts[cls.PALETTE_VIRIDIS] = np.column_stack([r, g, b]).astype(np.float32)

        # 3. Jet
        r = np.clip(1.5 - np.abs(4.0 * t - 3.0), 0.0, 1.0)
        g = np.clip(1.5 - np.abs(4.0 * t - 2.0), 0.0, 1.0)
        b = np.clip(1.5 - np.abs(4.0 * t - 1.0), 0.0, 1.0)
        luts[cls.PALETTE_JET] = np.column_stack([r ** 0.85, g ** 0.9, b ** 0.85]).astype(np.float32)

        # 4. Plasma
        stops_plasma = [
            (0.00, (0.05, 0.03, 0.53)),
            (0.25, (0.45, 0.10, 0.65)),
            (0.50, (0.80, 0.25, 0.50)),
            (0.75, (0.95, 0.55, 0.25)),
            (1.00, (0.94, 0.97, 0.13)),
        ]
        xp_p = [s[0] for s in stops_plasma]
        luts[cls.PALETTE_PLASMA] = np.column_stack([
            np.interp(t, xp_p, [s[1][0] for s in stops_plasma]),
            np.interp(t, xp_p, [s[1][1] for s in stops_plasma]),
            np.interp(t, xp_p, [s[1][2] for s in stops_plasma]),
        ]).astype(np.float32)

        # 5. Terrain
        stops_terrain = [
            (0.00, (0.15, 0.40, 0.65)),
            (0.20, (0.25, 0.65, 0.35)),
            (0.45, (0.55, 0.75, 0.35)),
            (0.70, (0.75, 0.55, 0.30)),
            (0.88, (0.60, 0.35, 0.20)),
            (1.00, (0.95, 0.95, 0.95)),
        ]
        xp_t = [s[0] for s in stops_terrain]
        luts[cls.PALETTE_TERRAIN] = np.column_stack([
            np.interp(t, xp_t, [s[1][0] for s in stops_terrain]),
            np.interp(t, xp_t, [s[1][1] for s in stops_terrain]),
            np.interp(t, xp_t, [s[1][2] for s in stops_terrain]),
        ]).astype(np.float32)

        # 6. Heat
        stops_heat = [
            (0.00, (0.00, 0.00, 0.00)),
            (0.33, (0.80, 0.05, 0.00)),
            (0.66, (1.00, 0.85, 0.00)),
            (1.00, (1.00, 1.00, 1.00)),
        ]
        xp_h = [s[0] for s in stops_heat]
        luts[cls.PALETTE_HEAT] = np.column_stack([
            np.interp(t, xp_h, [s[1][0] for s in stops_heat]),
            np.interp(t, xp_h, [s[1][1] for s in stops_heat]),
            np.interp(t, xp_h, [s[1][2] for s in stops_heat]),
        ]).astype(np.float32)

        # 7. Coolwarm
        stops_coolwarm = [
            (0.00, (0.23, 0.30, 0.75)),
            (0.50, (0.92, 0.92, 0.92)),
            (1.00, (0.70, 0.015, 0.15)),
        ]
        xp_c = [s[0] for s in stops_coolwarm]
        luts[cls.PALETTE_COOLWARM] = np.column_stack([
            np.interp(t, xp_c, [s[1][0] for s in stops_coolwarm]),
            np.interp(t, xp_c, [s[1][1] for s in stops_coolwarm]),
            np.interp(t, xp_c, [s[1][2] for s in stops_coolwarm]),
        ]).astype(np.float32)

        # 8. Grayscale
        luts[cls.PALETTE_GRAYSCALE] = np.column_stack([t, t, t]).astype(np.float32)

        cls._COLORMAP_LUTS = luts

    def _apply_lut(self, t: np.ndarray, palette_name: str) -> np.ndarray:
        """1024段の事前計算LUTから一瞬でRGB配列を引き当て (計算負荷を大幅削減)"""
        if getattr(self.__class__, "_COLORMAP_LUTS", None) is None:
            self._init_colormap_luts()
        p = str(palette_name).lower()
        lut = self.__class__._COLORMAP_LUTS.get(p, self.__class__._COLORMAP_LUTS[self.PALETTE_TURBO])
        if len(t) == 0:
            return np.empty((0, 3), dtype=np.float32)
        idx = np.clip((t * 1023.0).astype(np.int32), 0, 1023)
        return lut[idx].copy()

    @staticmethod
    def _colormap_interp(t: np.ndarray, stops: list) -> np.ndarray:
        """任意のキーストップ配列から 0.0〜1.0 のRGBを高速ベクトル補間"""
        t = np.clip(t, 0.0, 1.0)
        xp = [s[0] for s in stops]
        r = np.interp(t, xp, [s[1][0] for s in stops])
        g = np.interp(t, xp, [s[1][1] for s in stops])
        b = np.interp(t, xp, [s[1][2] for s in stops])
        return np.column_stack([r, g, b]).astype(np.float32)

    def _colormap_turbo(self, t: np.ndarray) -> np.ndarray:
        return self._apply_lut(t, self.PALETTE_TURBO)

    def _colormap_viridis(self, t: np.ndarray) -> np.ndarray:
        return self._apply_lut(t, self.PALETTE_VIRIDIS)

    def _colormap_plasma(self, t: np.ndarray) -> np.ndarray:
        return self._apply_lut(t, self.PALETTE_PLASMA)

    def _colormap_jet(self, t: np.ndarray) -> np.ndarray:
        return self._apply_lut(t, self.PALETTE_JET)

    def _colormap_terrain(self, t: np.ndarray) -> np.ndarray:
        return self._apply_lut(t, self.PALETTE_TERRAIN)

    def _colormap_heat(self, t: np.ndarray) -> np.ndarray:
        return self._apply_lut(t, self.PALETTE_HEAT)

    def _colormap_coolwarm(self, t: np.ndarray) -> np.ndarray:
        return self._apply_lut(t, self.PALETTE_COOLWARM)

    def _colormap_grayscale(self, t: np.ndarray) -> np.ndarray:
        return self._apply_lut(t, self.PALETTE_GRAYSCALE)

    def apply_colormap_by_name(self, t: np.ndarray, palette_name: str) -> np.ndarray:
        """パレット名に応じてカラーマップを適用 (LUT引き当て)"""
        return self._apply_lut(t, palette_name)

    # --- 視点プリセット (数学座標系: 上面で右がX、上がY、高さがZ) ---
    def reset_view(self):
        """等角 (Iso: 3D鳥瞰)"""
        self.pitch = 30.0
        self.yaw = 45.0
        self.camera_target = QVector3D(0.0, 0.0, 0.0)
        self.update()

    def set_view_top(self):
        """上面 (Top): Z軸上空から見下ろし (画面右がX、画面上がY、高さZ)"""
        self.pitch = 89.9
        self.yaw = 0.0
        self.update()

    def set_view_front(self):
        """正面 (Front): 手前(-Y)から奥(+Y)を見る (画面右がX、画面上がZ(高さ))"""
        self.pitch = 0.0
        self.yaw = 0.0
        self.update()

    def set_view_side(self):
        """側面 (Side): 東(+X)から西(-X)を見る (画面右がY、画面上がZ(高さ))"""
        self.pitch = 0.0
        self.yaw = -90.0
        self.update()

    def set_show_section_origin_plane(self, show: bool):
        """シーク基点プレーン（黄色枠・半透明面）の表示ON/OFFを設定"""
        if self.show_section_origin_plane != bool(show):
            self.show_section_origin_plane = bool(show)
            self.section_guides_changed.emit(self.show_section_origin_plane, self.show_section_box)
            self.update()

    def set_show_section_box(self, show: bool):
        """断面厚み範囲ボックス（水色枠）の表示ON/OFFを設定"""
        if self.show_section_box != bool(show):
            self.show_section_box = bool(show)
            self.section_guides_changed.emit(self.show_section_origin_plane, self.show_section_box)
            self.update()

    def toggle_section_guides(self) -> bool:
        """断面ガイド枠（基点枠・厚み枠）の表示を一括トグル切替 (ショートカット: Shift+B)"""
        new_state = not (self.show_section_origin_plane or self.show_section_box)
        self.show_section_origin_plane = new_state
        self.show_section_box = new_state
        self.section_guides_changed.emit(self.show_section_origin_plane, self.show_section_box)
        state_txt = "ON" if new_state else "OFF (点群のみ表示中 / Shift+Bで再表示)"
        self.status_changed.emit(f"📐 断面ガイド枠表示: {state_txt}")
        self.update()
        return new_state

    def set_brush_radius(self, radius: int):
        """ブラシ選択・消しゴムの半径(px)を設定 (6〜150px)"""
        new_r = max(6, min(150, int(radius)))
        if new_r != self.brush_radius:
            self.brush_radius = new_r
            self.brush_radius_changed.emit(self.brush_radius)
            self.status_changed.emit(f"ブラシサイズ変更: 半径 {self.brush_radius} px")
        self.update()

    # --- マウス操作 ---
    def mousePressEvent(self, event: QMouseEvent):
        self.last_mouse_pos = event.position().toPoint()
        self.current_mouse_pos = event.position().toPoint()
        self._last_selection_modifiers = event.modifiers()

        # ⑦ 軸ギズモ (X, Y, Z, ISO) クリックによるワンタッチ視点切替
        if event.button() == Qt.LeftButton:
            gizmo_hit = self._check_gizmo_hit(event.position().toPoint())
            if gizmo_hit:
                self._on_gizmo_axis_clicked(gizmo_hit)
                return

        if event.button() == Qt.RightButton:
            self._right_press_pos = event.position().toPoint()
            self._right_click_consumed = False

        if event.button() == Qt.RightButton and self.wall_align_mode:
            self.wall_align_points.clear()
            self._right_click_consumed = True
            self.update()
            return

        if event.button() == Qt.RightButton and self.measure_mode:
            self.clear_measure_points()
            self._right_click_consumed = True
            self.update()
            return

        if event.button() == Qt.RightButton and self.selection_mode and self.selection_shape == self.SELECT_POLYGON:
            if len(self.polygon_points) > 0:
                self.polygon_points.pop()
                self._right_click_consumed = True
                self.update()
                return

        # Spaceキー長押し中 + 左ドラッグ は、選択/ブラシモード中でも最優先で一時パン(平行移動)を行う
        if event.button() == Qt.LeftButton and self.space_pan_active:
            self.is_panning = True
            self.setCursor(Qt.ClosedHandCursor)
            return

        # 選択モード中でも Alt+左ドラッグ でシームレスに視点回転可能にする
        if event.button() == Qt.LeftButton and (event.modifiers() & Qt.AltModifier):
            self.is_rotating = True
            return

        # 🧭 2点ピック斜め壁アラインモードでの点ピック
        if self.wall_align_mode and event.button() == Qt.LeftButton:
            hit = self._pick_nearest_point(event.position().toPoint())
            if hit is not None:
                centered_pt, world_pt = hit
                if len(self.wall_align_points) >= 2:
                    self.wall_align_points = [(centered_pt, world_pt)]
                    self.status_changed.emit("🧭 2点壁ピック: 1点目を設定しました。壁に沿った2点目をクリックしてください")
                else:
                    self.wall_align_points.append((centered_pt, world_pt))
                    if len(self.wall_align_points) == 1:
                        self.status_changed.emit("🧭 2点壁ピック: 1点目を設定しました。壁に沿った2点目をクリックしてください")
                    elif len(self.wall_align_points) == 2:
                        p1 = self.wall_align_points[0][1]
                        p2 = self.wall_align_points[1][1]
                        self.wall_align_mode = False
                        self.set_wall_align_points(p1, p2, initial_view=self.wall_align_view_type)
                self.update()
            return

        if self.measure_mode and event.button() == Qt.LeftButton:
            hit = self._pick_nearest_point(event.position().toPoint(), max_px=self.MEASURE_PICK_MAX_PX)
            if hit is not None:
                self._on_measure_pick(hit[0], hit[1], snap_px=self.last_pick_px)
            else:
                self.status_changed.emit("クリック位置の近くに点がありません。拡大するか、点の上をクリックしてください")
            return

        if self.selection_mode and event.button() == Qt.LeftButton:
            if self.selection_shape == self.SELECT_POLYGON:
                self.polygon_points.append(event.position().toPoint())
                self.update()
            elif self.selection_shape == self.SELECT_BRUSH:
                self.is_selecting = True
                self.brush_stroke_points = [event.position().toPoint()]
                self.update()
            else:
                self.is_selecting = True
                self.selection_start = event.position().toPoint()
                self.selection_end = event.position().toPoint()
                self.update()
            return

        if event.button() == Qt.LeftButton:
            if self.pan_mode:
                self.is_panning = True
                self.setCursor(Qt.ClosedHandCursor)
            else:
                self.is_rotating = True
        elif event.button() in (Qt.RightButton, Qt.MiddleButton):
            if self.pan_mode and event.button() == Qt.MiddleButton:
                self.is_rotating = True
            else:
                self.is_panning = True

    def mouseMoveEvent(self, event: QMouseEvent):
        self.current_mouse_pos = event.position().toPoint()
        dx = event.position().x() - self.last_mouse_pos.x()
        dy = event.position().y() - self.last_mouse_pos.y()
        self.last_mouse_pos = event.position().toPoint()

        if self.is_selecting:
            if self.selection_shape == self.SELECT_BRUSH:
                if len(self.brush_stroke_points) == 0 or (event.position().toPoint() - self.brush_stroke_points[-1]).manhattanLength() >= 3:
                    self.brush_stroke_points.append(event.position().toPoint())
            else:
                self.selection_end = event.position().toPoint()
            self.update()
        elif self.is_rotating:
            # 左クリック視点回転操作
            rot_step = float(getattr(self, "rotate_speed", 0.5))
            self.yaw -= dx * rot_step
            pitch_sign = -1.0 if getattr(self, "invert_orbit_y", False) else 1.0
            self.pitch = max(-89.9, min(89.9, self.pitch + dy * rot_step * pitch_sign))
            self.update()
        elif self.is_panning:
            # 見ている画面に対するパン平行移動
            _, right, up = self._get_camera_vectors()
            pan_speed = self.camera_distance * 0.0015 * float(getattr(self, "pan_speed_factor", 1.0))
            delta = (-dx * pan_speed * right) + (dy * pan_speed * up)
            self.camera_target.setX(self.camera_target.x() + float(delta[0]))
            self.camera_target.setY(self.camera_target.y() + float(delta[1]))
            self.camera_target.setZ(self.camera_target.z() + float(delta[2]))
            self.update()
        elif self.selection_mode and self.selection_shape in (self.SELECT_POLYGON, self.SELECT_BRUSH):
            self.update()
        else:
            # ⑦ 軸ギズモのホバー判定
            hit = self._check_gizmo_hit(event.position().toPoint())
            if hit != getattr(self, "_hovered_gizmo_axis", None):
                self._hovered_gizmo_axis = hit
                if hit:
                    self.setCursor(Qt.PointingHandCursor)
                elif not (self.pan_mode or self.space_pan_active):
                    self.unsetCursor()
                self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self.is_selecting:
            self.is_selecting = False
            self.selection_end = event.position().toPoint()
            self._last_selection_modifiers = event.modifiers()
            if self.selection_shape == self.SELECT_BRUSH:
                if len(self.brush_stroke_points) == 0:
                    self.brush_stroke_points.append(event.position().toPoint())
                self._process_brush_selection()
                self.brush_stroke_points.clear()
            else:
                self._process_selection()
            self.update()

        if event.button() == Qt.RightButton:
            press_pos = getattr(self, "_right_press_pos", None)
            consumed = getattr(self, "_right_click_consumed", False)
            if not consumed and press_pos is not None:
                if (event.position().toPoint() - press_pos).manhattanLength() <= 5:
                    self.context_menu_requested.emit(event.globalPosition().toPoint())

        was_interacting = self.is_rotating or self.is_panning
        self.is_rotating = False
        self.is_panning = False
        if self.pan_mode or self.space_pan_active:
            self.setCursor(Qt.OpenHandCursor)
        else:
            self.unsetCursor()
        if was_interacting:
            # 操作終了時にフル解像度(100%点数)で再描画
            self.update()

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        if self.selection_mode and self.selection_shape == self.SELECT_POLYGON:
            if len(self.polygon_points) >= 3:
                self._last_selection_modifiers = event.modifiers()
                self._process_polygon_selection()
                self.polygon_points.clear()
                self.update()
            return

        if event.button() == Qt.LeftButton:
            hit = self._pick_nearest_point(event.position().toPoint())
            if hit is not None:
                centered_pt, world_pt = hit
                self.camera_target = QVector3D(float(centered_pt[0]), float(centered_pt[1]), float(centered_pt[2]))
                self.update()
                self.status_changed.emit(
                    f"中心フォーカス: X={world_pt[0]:.2f}, Y={world_pt[1]:.2f}, Z={world_pt[2]:.2f}"
                )

    def wheelEvent(self, event: QWheelEvent):
        delta = event.angleDelta().y()
        if (event.modifiers() & Qt.ControlModifier) and self.selection_mode and self.selection_shape == self.SELECT_BRUSH:
            step = 4 if delta > 0 else -4
            self.set_brush_radius(self.brush_radius + step)
            return
        zoom_mult = float(getattr(self, "zoom_speed_factor", 1.0))
        base_factor = 0.85 if delta > 0 else 1.18
        factor = 1.0 + (base_factor - 1.0) * zoom_mult
        factor = max(0.2, min(2.0, factor))
        self.camera_distance = max(0.1, self.camera_distance * factor)
        self.update()

    def keyPressEvent(self, event: QKeyEvent):
        """
        見ている画面に対するWASDフライスルー移動
        - W: 画面の視線方向（奥）へ前進
        - S: 画面の視線方向（手前）へ後退
        - A: 画面の左方向へ平行移動
        - D: 画面の右方向へ平行移動
        - Q: 画面の上方向へ上昇
        - E: 画面の下方向へ下降
        """
        key = event.key()
        wasd_mult = float(getattr(self, "wasd_speed_factor", 1.0))
        move_speed = max(0.2, self.camera_distance * 0.05) * wasd_mult
        forward, right, up = self._get_camera_vectors()

        # 見ている画面に対する直感的なWASD移動 (W:画面上, S:画面下, A:画面左, D:画面右, Q:画面奥, E:画面手前)
        if key == Qt.Key_W:
            self.camera_target.setX(self.camera_target.x() + float(up[0] * move_speed))
            self.camera_target.setY(self.camera_target.y() + float(up[1] * move_speed))
            self.camera_target.setZ(self.camera_target.z() + float(up[2] * move_speed))
            self.update()
        elif key == Qt.Key_S:
            self.camera_target.setX(self.camera_target.x() - float(up[0] * move_speed))
            self.camera_target.setY(self.camera_target.y() - float(up[1] * move_speed))
            self.camera_target.setZ(self.camera_target.z() - float(up[2] * move_speed))
            self.update()
        elif key == Qt.Key_A:
            self.camera_target.setX(self.camera_target.x() - float(right[0] * move_speed))
            self.camera_target.setY(self.camera_target.y() - float(right[1] * move_speed))
            self.camera_target.setZ(self.camera_target.z() - float(right[2] * move_speed))
            self.update()
        elif key == Qt.Key_D:
            self.camera_target.setX(self.camera_target.x() + float(right[0] * move_speed))
            self.camera_target.setY(self.camera_target.y() + float(right[1] * move_speed))
            self.camera_target.setZ(self.camera_target.z() + float(right[2] * move_speed))
            self.update()
        elif key == Qt.Key_Q:
            self.camera_target.setX(self.camera_target.x() + float(forward[0] * move_speed))
            self.camera_target.setY(self.camera_target.y() + float(forward[1] * move_speed))
            self.camera_target.setZ(self.camera_target.z() + float(forward[2] * move_speed))
            self.update()
        elif key == Qt.Key_E:
            self.camera_target.setX(self.camera_target.x() - float(forward[0] * move_speed))
            self.camera_target.setY(self.camera_target.y() - float(forward[1] * move_speed))
            self.camera_target.setZ(self.camera_target.z() - float(forward[2] * move_speed))
            self.update()
        elif key in (Qt.Key_Return, Qt.Key_Enter):
            if self.selection_mode and self.selection_shape == self.SELECT_POLYGON and len(self.polygon_points) >= 3:
                self._last_selection_modifiers = event.modifiers()
                self._process_polygon_selection()
                self.polygon_points.clear()
                self.update()
        elif key == Qt.Key_Escape:
            self.clear_selection()
        elif key == Qt.Key_F:
            self.fit_to_screen()
        elif key == Qt.Key_P:
            self.toggle_projection()
        elif key == Qt.Key_I:
            self.invert_selection()
        elif key == Qt.Key_B and (event.modifiers() & Qt.ShiftModifier):
            self.toggle_section_guides()
        elif key == Qt.Key_BracketLeft:
            self.set_brush_radius(self.brush_radius - 4)
        elif key == Qt.Key_BracketRight:
            self.set_brush_radius(self.brush_radius + 4)
        elif key == Qt.Key_Space and not event.isAutoRepeat():
            self.space_pan_active = True
            self.setCursor(Qt.OpenHandCursor)
            self.update()
        elif key == Qt.Key_C and (event.modifiers() & Qt.ControlModifier) and self.measure_mode:
            # 計測中の Ctrl+C: 座標（と距離）をタブ区切りでコピー
            self.copy_measure_to_clipboard()
        elif key == Qt.Key_M:
            self.measure_mode = not self.measure_mode
            if not self.measure_mode:
                self.clear_measure_points()
            self.update()
            mode_s = "ON" if self.measure_mode else "OFF"
            self.status_changed.emit(f"寸法計測ツール: {mode_s} (M)")
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            self.space_pan_active = False
            self.is_panning = False
            if self.pan_mode:
                self.setCursor(Qt.OpenHandCursor)
            else:
                self.unsetCursor()
            self.update()
        else:
            super().keyReleaseEvent(event)

    # ------------------------------------------------------------------ 計測（座標の読み取り・コピー）
    def _on_measure_pick(self, centered_pt: np.ndarray, world_pt: np.ndarray, snap_px: Optional[float] = None) -> None:
        """計測モードで点をピック: 1点目で座標を表示、2点目で距離と差分、3点目以降は新しい計測を始める"""
        if len(self.measure_points) >= 2:
            self.measure_points = []
        self.measure_points.append((np.asarray(centered_pt), np.asarray(world_pt, dtype=np.float64)))
        self._refresh_measure_text()
        msg = measure_status_text([p[1] for p in self.measure_points], self.coord_mode)
        if snap_px is not None and snap_px > self.MEASURE_SNAP_NOTICE_PX:
            msg += f"  ※クリック位置から {snap_px:.0f}px 離れた点を拾いました。拡大すると正確に拾えます"
        self.measure_updated.emit(msg)
        self.update()

    def _refresh_measure_text(self) -> None:
        """measure_points と coord_mode から HUD の行と距離テキストを作り直す"""
        world = [p[1] for p in self.measure_points]
        self.measure_coord_lines = measure_hud_lines(world, self.coord_mode)
        self.measure_result_text = measure_distance_text(world)

    def clear_measure_points(self) -> None:
        """計測点・HUD の行・距離テキストをまとめて消す（右クリック / Esc / M で終了）"""
        self.measure_points.clear()
        self.measure_result_text = ""
        self.measure_coord_lines = []

    def copy_measure_to_clipboard(self) -> bool:
        """計測中の座標（と距離）をタブ区切りでクリップボードへ。Excel / CAD にそのまま貼れる"""
        if not self.measure_points:
            self.status_changed.emit("コピーする計測点がありません。計測モード(M)で点をクリックしてください")
            return False
        text = measure_clipboard_text([p[1] for p in self.measure_points], self.coord_mode)
        clipboard = QApplication.clipboard()
        if clipboard is None:
            return False
        clipboard.setText(text)
        self.status_changed.emit(f"計測座標 {len(self.measure_points)} 点をクリップボードにコピーしました（タブ区切り）")
        return True

    def _pick_nearest_point(self, click_pos: QPoint, max_px: float = 30.0) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        クリック位置に見えている点を拾い、(中心化座標, ワールド座標) を返す。max_px 以内に点が無ければ None。
        - 表示中の全点を対象にする（間引くとクリックした点と別の点を拾うため）
        - カーソル付近で最も手前の面に絞ってから、画面上で最も近い点を選ぶ（奥の点を拾わない）
        - ワールド座標は float32 の表示用配列ではなく points_raw の原本値を返す
        - クリック位置からの距離 [px] を self.last_pick_px に残す
        """
        self.last_pick_px = None
        if self.display_points is None or len(self.display_points) == 0:
            return None

        pts = self.display_points
        scr_x, scr_y, valid_z = self._project_to_screen(pts)
        dists_sq = (scr_x - click_pos.x()) ** 2 + (scr_y - click_pos.y()) ** 2
        cand = np.where(valid_z & (dists_sq <= float(max_px) ** 2))[0]
        if len(cand) == 0:
            return None

        cand_d = np.sqrt(dists_sq[cand])
        depth = self._view_depth(pts[cand])
        front = float(depth[cand_d <= float(cand_d.min()) + 1.5].min())
        on_front = depth <= front + self._world_per_pixel(front) * 8.0
        best_pos = int(np.argmin(np.where(on_front, cand_d, np.inf)))

        best_idx = int(cand[best_pos])
        self.last_pick_px = float(cand_d[best_pos])
        best_centered = pts[best_idx]
        return best_centered, self._display_point_world(best_idx, best_centered)

    def _view_depth(self, pts_centered: np.ndarray) -> np.ndarray:
        """視点からの奥行き（カメラ座標系。大きいほど奥）"""
        view_np = np.array(self._get_view_matrix().data(), dtype=np.float32).reshape((4, 4))
        ones = np.ones((len(pts_centered), 1), dtype=np.float32)
        return -(np.hstack([pts_centered, ones]) @ view_np)[:, 2]

    def _world_per_pixel(self, depth: float) -> float:
        """奥行き depth の位置で画面 1px が何メートルに当たるか"""
        height = max(1, self.height())
        if self.is_orthographic:
            return 2.0 * max(0.5, self.camera_distance * 0.35) / height
        return 2.0 * math.tan(math.radians(self.fov) * 0.5) * max(depth, 1e-6) / height

    def _display_point_world(self, display_idx: int, centered_pt: np.ndarray) -> np.ndarray:
        """
        表示点のワールド座標を原本 points_raw から返す。
        断面スライダーの高速プレビュー中は display_orig_indices と対応しないので、
        中心化座標が一致するときだけ原本を使い、それ以外は表示用の値から復元する。
        """
        orig = self.display_orig_indices
        if orig is not None and self.points_raw is not None and len(orig) == len(self.display_points):
            orig_idx = int(orig[display_idx])
            if np.array_equal(self.points_centered[orig_idx], centered_pt):
                return np.array(self.points_raw[orig_idx], dtype=np.float64)
        return np.asarray(centered_pt, dtype=np.float64) + self.center_offset

    def _project_to_screen(self, pts_centered: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        aspect = self.width() / max(1, self.height())
        proj = QMatrix4x4()
        near_plane = max(0.1, self.camera_distance * 0.01)
        far_plane = max(500.0, self.camera_distance * 50.0)

        if self.is_orthographic:
            ortho_size = max(0.5, self.camera_distance * 0.35)
            proj.ortho(-ortho_size * aspect, ortho_size * aspect, -ortho_size, ortho_size, -far_plane, far_plane)
        else:
            proj.perspective(self.fov, aspect, near_plane, far_plane)

        view = self._get_view_matrix()
        mvp = proj * view
        mvp_np = np.array(mvp.data(), dtype=np.float32).reshape((4, 4))

        ones = np.ones((len(pts_centered), 1), dtype=np.float32)
        pts4 = np.hstack([pts_centered, ones])
        clip_pts = pts4 @ mvp_np

        w = clip_pts[:, 3]
        valid_z = w > 0.001
        ndc_x = clip_pts[:, 0] / np.where(valid_z, w, 1.0)
        ndc_y = clip_pts[:, 1] / np.where(valid_z, w, 1.0)

        scr_x = (ndc_x + 1.0) * 0.5 * self.width()
        scr_y = (1.0 - ndc_y) * 0.5 * self.height()

        return scr_x, scr_y, valid_z

    def _get_selectable_mask(self) -> np.ndarray:
        """断面スライス・現在表示ONのレイヤー＆スキャン・かつ未ロックのレイヤーに基づく選択対象マスクを取得 (SIMD高速化)"""
        if self.points_centered is None or len(self.points_centered) == 0:
            return np.empty(0, dtype=bool)
        in_section = self._get_section_inlier_mask()
        layer_scan_vis = self._get_layer_and_scan_visibility_mask()
        unlocked_mask = self._get_unlocked_mask()
        return in_section & layer_scan_vis & unlocked_mask

    def _apply_selection_with_modifiers(self, new_selected: np.ndarray, label_prefix="範囲選択", modifiers=None):
        """Shiftキーで追加選択、Ctrlキーで選択除外、通常で新規選択を適用 (即時削除モード時は即消去を発火)"""
        if not isinstance(label_prefix, str):
            modifiers = label_prefix
            label_prefix = "範囲選択"
        mods = modifiers if modifiers is not None else self._last_selection_modifiers

        # ⚡ 即時削除モード(囲み・なぞり即削除)がONかつCtrl(除外)が押されていない場合は1ステップでノイズ消去を発火
        if self.instant_delete_mode and not (mods & Qt.ControlModifier):
            erase_targets = np.asarray(new_selected, dtype=np.int64)
            if len(erase_targets) > 0:
                self.instant_erase_requested.emit(erase_targets)
            else:
                self.status_changed.emit(f"{label_prefix}: 対象範囲内に点はありませんでした")
            return

        if (mods & Qt.ShiftModifier) and len(self.selected_indices) > 0:
            combined = np.union1d(self.selected_indices, new_selected).astype(np.int64)
            action_str = f"{label_prefix} (Shift追加)"
        elif (mods & Qt.ControlModifier) and len(self.selected_indices) > 0:
            combined = np.setdiff1d(self.selected_indices, new_selected).astype(np.int64)
            action_str = f"{label_prefix} (Ctrl除外)"
        else:
            combined = np.asarray(new_selected, dtype=np.int64)
            action_str = label_prefix

        self.selected_indices = combined
        self._update_active_colors()
        self._update_display_colors_only()
        self.update()
        self.point_selected.emit(self.selected_indices.tolist())
        self.status_changed.emit(f"{action_str}完了: {len(self.selected_indices):,} 点を選択中")

    def invert_selection(self):
        """現在表示・断面内の選択可能な点群に対して、選択状態と非選択状態を反転する"""
        if self.points_raw is None or len(self.points_raw) == 0:
            return
        selectable_indices = np.where(self._get_selectable_mask())[0]
        if len(selectable_indices) == 0:
            return
        inverted = np.setdiff1d(selectable_indices, self.selected_indices).astype(np.int64)
        self.selected_indices = inverted
        self._update_active_colors()
        self._update_display_colors_only()
        self.update()
        self.point_selected.emit(self.selected_indices.tolist())
        self.status_changed.emit(f"選択反転完了: {len(self.selected_indices):,} 点を選択中")

    def clear_selection(self):
        """選択状態・ドラッグ途中状態・計測ピンをすべて解除する (Esc)"""
        self.selected_indices = np.empty(0, dtype=np.int64)
        self.polygon_points.clear()
        self.brush_stroke_points.clear()
        self.is_selecting = False
        if self.measure_mode and (len(self.measure_points) > 0 or self.measure_result_text):
            self.clear_measure_points()
        self._update_active_colors()
        self._update_display_colors_only()
        self.update()
        self.point_selected.emit([])
        self.status_changed.emit("選択を解除しました (Esc)")

    def _process_selection(self):
        if self.points_centered is None or len(self.points_centered) == 0:
            return

        cand_indices = np.where(self._get_selectable_mask())[0]
        if len(cand_indices) == 0:
            return

        pts = self.points_centered[cand_indices]
        scr_x, scr_y, valid_z = self._project_to_screen(pts)

        if self.selection_shape == self.SELECT_RECT:
            rect = QRect(self.selection_start, self.selection_end).normalized()
            if rect.width() < 4 or rect.height() < 4:
                return
            in_shape = (
                valid_z
                & (scr_x >= rect.left())
                & (scr_x <= rect.right())
                & (scr_y >= rect.top())
                & (scr_y <= rect.bottom())
            )
        elif self.selection_shape == self.SELECT_CIRCLE:
            cx, cy = self.selection_start.x(), self.selection_start.y()
            radius = math.hypot(self.selection_end.x() - cx, self.selection_end.y() - cy)
            if radius < 3:
                return
            dists_sq = (scr_x - cx) ** 2 + (scr_y - cy) ** 2
            in_shape = valid_z & (dists_sq <= (radius ** 2))
        else:
            return

        selected = cand_indices[in_shape]
        self._apply_selection_with_modifiers(selected, "範囲選択")

    def _process_polygon_selection(self):
        if self.points_centered is None or len(self.points_centered) == 0 or len(self.polygon_points) < 3:
            return

        cand_indices = np.where(self._get_selectable_mask())[0]
        if len(cand_indices) == 0:
            return

        pts = self.points_centered[cand_indices]
        scr_x, scr_y, valid_z = self._project_to_screen(pts)

        poly = np.array([[p.x(), p.y()] for p in self.polygon_points], dtype=np.float32)
        n_poly = len(poly)
        inside = np.zeros(len(pts), dtype=bool)

        j = n_poly - 1
        for i in range(n_poly):
            xi, yi = poly[i]
            xj, yj = poly[j]
            cond = ((yi > scr_y) != (yj > scr_y)) & (
                scr_x < (xj - xi) * (scr_y - yi) / np.maximum(1e-6, np.abs(yj - yi)) * np.sign(yj - yi) + xi
            )
            inside ^= cond
            j = i

        selected = cand_indices[valid_z & inside]
        self._apply_selection_with_modifiers(selected, "多角形選択")

    def _process_brush_selection(self):
        """ブラシなぞり軌跡 (カプセル連続線分) 内の点群を高速抽出して選択または即削除する"""
        if self.points_centered is None or len(self.points_centered) == 0 or len(self.brush_stroke_points) == 0:
            return

        cand_indices = np.where(self._get_selectable_mask())[0]
        if len(cand_indices) == 0:
            return

        pts = self.points_centered[cand_indices]
        scr_x, scr_y, valid_z = self._project_to_screen(pts)

        r = float(max(4, self.brush_radius))
        r_sq = r * r

        stroke_xy = np.array([[p.x(), p.y()] for p in self.brush_stroke_points], dtype=np.float32)
        min_x = float(np.min(stroke_xy[:, 0])) - r
        max_x = float(np.max(stroke_xy[:, 0])) + r
        min_y = float(np.min(stroke_xy[:, 1])) - r
        max_y = float(np.max(stroke_xy[:, 1])) + r

        bbox_mask = (
            valid_z
            & (scr_x >= min_x)
            & (scr_x <= max_x)
            & (scr_y >= min_y)
            & (scr_y <= max_y)
        )
        sub_idx = np.where(bbox_mask)[0]
        if len(sub_idx) == 0:
            self._apply_selection_with_modifiers(np.empty(0, dtype=np.int64), "ブラシ選択")
            return

        px = scr_x[sub_idx]
        py = scr_y[sub_idx]
        hit_sub = np.zeros(len(sub_idx), dtype=bool)

        if len(stroke_xy) == 1:
            cx, cy = stroke_xy[0, 0], stroke_xy[0, 1]
            hit_sub = ((px - cx) ** 2 + (py - cy) ** 2) <= r_sq
        else:
            for i in range(len(stroke_xy) - 1):
                x0, y0 = stroke_xy[i, 0], stroke_xy[i, 1]
                x1, y1 = stroke_xy[i + 1, 0], stroke_xy[i + 1, 1]
                dx_seg = x1 - x0
                dy_seg = y1 - y0
                seg_len_sq = dx_seg * dx_seg + dy_seg * dy_seg
                if seg_len_sq < 1e-4:
                    dist_sq = (px - x0) ** 2 + (py - y0) ** 2
                else:
                    t = np.clip(((px - x0) * dx_seg + (py - y0) * dy_seg) / seg_len_sq, 0.0, 1.0)
                    proj_x = x0 + t * dx_seg
                    proj_y = y0 + t * dy_seg
                    dist_sq = (px - proj_x) ** 2 + (py - proj_y) ** 2
                hit_sub |= (dist_sq <= r_sq)

        selected = cand_indices[sub_idx[hit_sub]]
        self._apply_selection_with_modifiers(selected, "ブラシ選択")


