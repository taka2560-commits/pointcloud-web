"""
PySide6 / OpenGLによる3D点群ビューアウィジェット
- 数学座標系: 上面視で上がY、右がX、高さがZ (Z-up) に厳密に準拠
- 見ている画面に対するWASDフライスルー移動 (Forward/Right/Upベクトル)
- 視点切替プリセット (上面: XY平面, 正面: XZ平面(高さZ), 側面: YZ平面(高さZ), 等角)
- XYZ別シーク断面スライサー (X/Y/Z独立クリッピング & 断面枠線可視化)
- 手動選択ツール: 矩形選択、円形選択、多角形(ポリゴン)選択
- 2点間寸法計測ツール、全体フィット、正射影/透視投影切替、表示バジェット
"""

import math
from typing import Optional, Tuple, List
import numpy as np
import OpenGL.GL as gl

from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtGui import (
    QMatrix4x4,
    QVector3D,
    QVector4D,
    QColor,
    QPainter,
    QPen,
    QBrush,
    QFont,
    QMouseEvent,
    QWheelEvent,
    QKeyEvent,
)
from PySide6.QtCore import Qt, QPoint, QRect, Signal


class PointCloudViewer(QOpenGLWidget):
    """3D点群ビューアクラス"""

    point_selected = Signal(list)       # 選択された点インデックスリスト
    status_changed = Signal(str)        # ステータスメッセージ
    measure_updated = Signal(str)       # 計測結果メッセージ
    projection_changed = Signal(bool)   # 投影モード変更通知 (True: 正射影, False: 透視投影)
    bounds_ready = Signal(object, object)  # 点群のワールド範囲 (min_xyz, max_xyz)

    COLOR_MODE_RGB = "RGB"
    COLOR_MODE_HEIGHT = "高さ (Z)"
    COLOR_MODE_INTENSITY = "反射強度"
    COLOR_MODE_SOLID = "単色"

    # 選択形状
    SELECT_RECT = "rect"
    SELECT_CIRCLE = "circle"
    SELECT_POLYGON = "polygon"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)

        # カメラパラメータ (数学座標系: 上面で右がX、上がY、高さがZ)
        self.camera_target = QVector3D(0.0, 0.0, 0.0)  # 注視点
        self.camera_distance = 15.0                     # 距離
        self.yaw = 45.0                                # 水平角 (度)
        self.pitch = 30.0                              # 仰角 (度: 0°=水平正面, 89.9°=真上見下ろし)
        self.fov = 45.0                                # 視野角 (度)
        self.is_orthographic = False                   # 正射影フラグ

        # 点群原本データ
        self.points_raw = None        # (N, 3) float64 原本
        self.points_centered = None   # (N, 3) float32 中心オフセット済み原本
        self.center_offset = np.zeros(3, dtype=np.float64)
        self.colors_raw = None        # (N, 3) float32
        self.intensities_raw = None   # (N,) float32
        self.active_colors = None     # (N, 3) float32 描画用カラー原本

        # 表示用バジェット配列 (Point Budget対応)
        self.display_points = None    # 描画対象 (M, 3) float32
        self.display_colors = None    # 描画対象カラー (M, 3) float32
        self.display_orig_indices = None  # 原本インデックスとのマッピング
        self.point_budget = 2000000   # 初期値: 200万点 (0で無制限)

        # 表示設定
        self.point_size = 2.0
        self.point_shape = "circle"
        self.color_mode = self.COLOR_MODE_RGB
        self.bg_color = QColor(20, 20, 26)
        self.show_grid = True
        self.bounds_min = np.array([-10, -10, -5], dtype=np.float32)
        self.bounds_max = np.array([10, 10, 5], dtype=np.float32)
        self.world_bounds_min = np.zeros(3, dtype=np.float64)
        self.world_bounds_max = np.zeros(3, dtype=np.float64)

        # XYZ断面シークスライサー設定
        self.clip_x_enabled = False
        self.clip_x_range = [-1e6, 1e6]
        self.clip_y_enabled = False
        self.clip_y_range = [-1e6, 1e6]
        self.clip_z_enabled = False
        self.clip_z_range = [-1e6, 1e6]

        # マウス操作用
        self.last_mouse_pos = QPoint()
        self.current_mouse_pos = QPoint()
        self.is_rotating = False
        self.is_panning = False

        # 手動選択モード
        self.selection_mode = False
        self.selection_shape = self.SELECT_RECT
        self.selection_start = QPoint()
        self.selection_end = QPoint()
        self.is_selecting = False
        self.polygon_points: List[QPoint] = []
        self.selected_indices = np.array([], dtype=np.int64)

        # 2点間寸法計測ツール
        self.measure_mode = False
        self.measure_points: List[Tuple[np.ndarray, np.ndarray]] = []
        self.measure_result_text = ""

    def initializeGL(self):
        """OpenGL初期化処理"""
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthFunc(gl.GL_LEQUAL)
        gl.glEnable(gl.GL_POINT_SMOOTH)
        gl.glHint(gl.GL_POINT_SMOOTH_HINT, gl.GL_NICEST)
        gl.glDisable(gl.GL_LIGHTING)

    def resizeGL(self, width: int, height: int):
        """ウィンドウリサイズ処理"""
        dpr = self.devicePixelRatio()
        gl.glViewport(0, 0, int(width * dpr), int(height * dpr))

    def paintGL(self):
        """OpenGL描画ループ（高DPI・断面・数学座標系・ギズモ）"""
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
        if self.show_grid:
            self._draw_grid()

        # 4. 点群の描画
        if self.display_points is not None and len(self.display_points) > 0:
            gl.glPointSize(float(self.point_size))

            if self.point_shape == "circle":
                gl.glEnable(gl.GL_POINT_SMOOTH)
            else:
                gl.glDisable(gl.GL_POINT_SMOOTH)

            gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
            gl.glEnableClientState(gl.GL_COLOR_ARRAY)

            gl.glVertexPointer(3, gl.GL_FLOAT, 0, self.display_points)
            gl.glColorPointer(3, gl.GL_FLOAT, 0, self.display_colors)

            gl.glDrawArrays(gl.GL_POINTS, 0, len(self.display_points))

            gl.glDisableClientState(gl.GL_COLOR_ARRAY)
            gl.glDisableClientState(gl.GL_VERTEX_ARRAY)

        # 5. 断面スライスボックス枠線描画
        self._draw_section_box()

        # 6. 計測ラインの描画
        if len(self.measure_points) > 0:
            self._draw_measure_lines()

        # 7. XYZ座標軸ギズモ描画
        self._draw_axes()

        # 8. 2Dオーバーレイ描画
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
        """断面の範囲枠線を描画"""
        if not (self.clip_x_enabled or self.clip_y_enabled or self.clip_z_enabled):
            return
        if self.points_raw is None:
            return

        gl.glPushAttrib(gl.GL_ALL_ATTRIB_BITS)
        gl.glDisable(gl.GL_LIGHTING)
        gl.glLineWidth(1.5)
        gl.glColor4f(0.0, 0.9, 1.0, 0.6)

        x0 = (self.clip_x_range[0] - self.center_offset[0]) if self.clip_x_enabled else self.bounds_min[0]
        x1 = (self.clip_x_range[1] - self.center_offset[0]) if self.clip_x_enabled else self.bounds_max[0]
        y0 = (self.clip_y_range[0] - self.center_offset[1]) if self.clip_y_enabled else self.bounds_min[1]
        y1 = (self.clip_y_range[1] - self.center_offset[1]) if self.clip_y_enabled else self.bounds_max[1]
        z0 = (self.clip_z_range[0] - self.center_offset[2]) if self.clip_z_enabled else self.bounds_min[2]
        z1 = (self.clip_z_range[1] - self.center_offset[2]) if self.clip_z_enabled else self.bounds_max[2]

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

        # 2. 矩形選択枠
        if self.is_selecting and self.selection_shape == self.SELECT_RECT:
            rect = QRect(self.selection_start, self.selection_end).normalized()
            pen = QPen(QColor(0, 210, 255, 230), 2, Qt.DashLine)
            painter.setPen(pen)
            brush = QBrush(QColor(0, 150, 255, 45))
            painter.setBrush(brush)
            painter.drawRect(rect)

        # 3. 円形選択枠
        elif self.is_selecting and self.selection_shape == self.SELECT_CIRCLE:
            p1 = self.selection_start
            p2 = self.selection_end
            radius = int(math.hypot(p2.x() - p1.x(), p2.y() - p1.y()))
            pen = QPen(QColor(0, 220, 255, 230), 2, Qt.DashLine)
            painter.setPen(pen)
            brush = QBrush(QColor(0, 150, 255, 45))
            painter.setBrush(brush)
            painter.drawEllipse(p1, radius, radius)

        # 4. 多角形選択枠
        elif self.selection_mode and self.selection_shape == self.SELECT_POLYGON and len(self.polygon_points) > 0:
            pen = QPen(QColor(0, 220, 255, 230), 2, Qt.SolidLine)
            painter.setPen(pen)
            for i in range(len(self.polygon_points) - 1):
                painter.drawLine(self.polygon_points[i], self.polygon_points[i + 1])
            dash_pen = QPen(QColor(0, 220, 255, 180), 1.5, Qt.DashLine)
            painter.setPen(dash_pen)
            painter.drawLine(self.polygon_points[-1], self.current_mouse_pos)

            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(255, 200, 0, 240)))
            for pt in self.polygon_points:
                painter.drawEllipse(pt, 4, 4)

        # 5. 計測情報表示
        if self.measure_result_text:
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(15, 23, 42, 220)))
            painter.drawRoundedRect(QRect(14, 14, 380, 75), 8, 8)

            painter.setPen(QColor(56, 189, 248))
            painter.setFont(QFont("Segoe UI", 11, QFont.Bold))
            painter.drawText(26, 36, "📐 2点間寸法計測結果")

            painter.setPen(QColor(241, 245, 249))
            painter.setFont(QFont("Consolas", 10, QFont.Normal))
            painter.drawText(26, 56, self.measure_result_text)
            painter.drawText(26, 74, "※右クリックでクリア / Mキーで終了")
        elif self.measure_mode:
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(15, 23, 42, 200)))
            painter.drawRoundedRect(QRect(14, 14, 320, 50), 8, 8)

            painter.setPen(QColor(250, 204, 21))
            painter.setFont(QFont("Segoe UI", 10, QFont.Bold))
            msg = "計測点 1 をクリック" if len(self.measure_points) == 0 else "計測点 2 をクリック"
            painter.drawText(24, 34, f"📐 計測モード: {msg}")
            painter.drawText(24, 52, "※右クリックでキャンセル")

        painter.end()

    def _draw_axes_labels_2d(self, painter: QPainter):
        """ギズモ先端のX, Y, Z文字を2D描画"""
        w, h = self.width(), self.height()
        aspect = w / max(1, h)
        cx = 0.18
        cy = -0.78

        font = QFont("Segoe UI", 9, QFont.Bold)
        painter.setFont(font)

        # カメラ回転行列を取得
        view = self._get_view_matrix()
        view.setColumn(3, QVector4D(0.0, 0.0, 0.0, 1.0))

        base_x = (-aspect + cx + aspect) * 0.5 * w
        base_y = (1.0 - (cy + 1.0) * 0.5) * h

        # 各軸先端の投影
        for label, vec, col in [
            ("X", QVector3D(1.0, 0.0, 0.0), QColor(255, 80, 80)),
            ("Y", QVector3D(0.0, 1.0, 0.0), QColor(60, 230, 80)),
            ("Z", QVector3D(0.0, 0.0, 1.0), QColor(80, 160, 255)),
        ]:
            v_rot = view.map(vec) * 0.18
            sx = int(base_x + v_rot.x() * 0.5 * w)
            sy = int(base_y - v_rot.y() * 0.5 * h)
            painter.setPen(col)
            painter.drawText(sx, sy, label)

    def set_point_cloud(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
        intensities: Optional[np.ndarray] = None,
        reset_camera: bool = True,
    ):
        """点群データを設定する"""
        if points is None or len(points) == 0:
            self.points_raw = None
            self.points_centered = None
            self.active_colors = None
            self.display_points = None
            self.display_colors = None
            self.update()
            return

        self.points_raw = np.asarray(points, dtype=np.float64)

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
            if np.max(c) > 1.0:
                c = c / 255.0
            self.colors_raw = np.clip(c, 0.0, 1.0)
        else:
            self.colors_raw = None

        if intensities is not None and len(intensities) == len(points):
            self.intensities_raw = np.asarray(intensities, dtype=np.float32)
        else:
            self.intensities_raw = None

        self._update_active_colors()
        self._update_display_arrays()

        if reset_camera:
            self.fit_to_screen()

        self.bounds_ready.emit(self.world_bounds_min, self.world_bounds_max)
        self.update()
        self.status_changed.emit(f"点群読み込み完了: {len(points):,} 点")

    def _update_active_colors(self):
        """原本全体のカラー配列を生成"""
        if self.points_centered is None:
            return

        n = len(self.points_centered)

        if self.color_mode == self.COLOR_MODE_RGB and self.colors_raw is not None:
            self.active_colors = self.colors_raw.copy()
        elif self.color_mode == self.COLOR_MODE_HEIGHT:
            z = self.points_centered[:, 2]  # 高さZ
            z_min, z_max = np.min(z), np.max(z)
            t = (z - z_min) / max(1e-5, (z_max - z_min))
            self.active_colors = self._colormap_turbo(t)
        elif self.color_mode == self.COLOR_MODE_INTENSITY and self.intensities_raw is not None:
            iv = self.intensities_raw
            i_min, i_max = np.min(iv), np.max(iv)
            t = (iv - i_min) / max(1e-5, (i_max - i_min))
            self.active_colors = np.column_stack([t, t, t]).astype(np.float32)
        else:
            self.active_colors = np.tile([0.3, 0.8, 0.95], (n, 1)).astype(np.float32)

        if len(self.selected_indices) > 0:
            valid = self.selected_indices[self.selected_indices < n]
            self.active_colors[valid] = [1.0, 0.1, 0.1]

        self.active_colors = np.ascontiguousarray(self.active_colors, dtype=np.float32)

    def _update_display_arrays(self):
        """断面スライス＆Point Budgetを適用して描画バッファを生成"""
        if self.points_raw is None or len(self.points_raw) == 0:
            self.display_points = None
            self.display_colors = None
            self.display_orig_indices = None
            return

        mask = np.ones(len(self.points_raw), dtype=bool)
        if self.clip_x_enabled:
            mask &= (self.points_raw[:, 0] >= self.clip_x_range[0]) & (self.points_raw[:, 0] <= self.clip_x_range[1])
        if self.clip_y_enabled:
            mask &= (self.points_raw[:, 1] >= self.clip_y_range[0]) & (self.points_raw[:, 1] <= self.clip_y_range[1])
        if self.clip_z_enabled:
            mask &= (self.points_raw[:, 2] >= self.clip_z_range[0]) & (self.points_raw[:, 2] <= self.clip_z_range[1])

        surviving_indices = np.where(mask)[0]
        n_surv = len(surviving_indices)

        if n_surv == 0:
            self.display_points = np.empty((0, 3), dtype=np.float32)
            self.display_colors = np.empty((0, 3), dtype=np.float32)
            self.display_orig_indices = np.empty(0, dtype=np.int64)
            return

        budget = self.point_budget
        if budget <= 0 or n_surv <= budget:
            sampled_idx = surviving_indices
        else:
            step = max(1, int(math.ceil(n_surv / budget)))
            sampled_idx = surviving_indices[::step]

        self.display_orig_indices = sampled_idx
        self.display_points = np.ascontiguousarray(self.points_centered[sampled_idx])
        self.display_colors = np.ascontiguousarray(self.active_colors[sampled_idx])

    # --- 断面スライサー制御 ---
    def set_section_x(self, enabled: bool, min_val: float, max_val: float):
        self.clip_x_enabled = enabled
        self.clip_x_range = [min_val, max_val]
        self._update_display_arrays()
        self.update()

    def set_section_y(self, enabled: bool, min_val: float, max_val: float):
        self.clip_y_enabled = enabled
        self.clip_y_range = [min_val, max_val]
        self._update_display_arrays()
        self.update()

    def set_section_z(self, enabled: bool, min_val: float, max_val: float):
        self.clip_z_enabled = enabled
        self.clip_z_range = [min_val, max_val]
        self._update_display_arrays()
        self.update()

    def set_point_budget(self, budget: int):
        self.point_budget = budget
        self._update_display_arrays()
        self.update()

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
        self._update_active_colors()
        self._update_display_arrays()
        self.update()

    def set_point_size(self, size: float):
        self.point_size = max(0.5, float(size))
        self.update()

    def _colormap_turbo(self, t: np.ndarray) -> np.ndarray:
        t = np.clip(t, 0.0, 1.0)
        r = np.clip(1.5 - np.abs(t * 4.0 - 3.0), 0.0, 1.0)
        g = np.clip(1.5 - np.abs(t * 4.0 - 2.0), 0.0, 1.0)
        b = np.clip(1.5 - np.abs(t * 4.0 - 1.0), 0.0, 1.0)
        return np.column_stack([r, g, b]).astype(np.float32)

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

    # --- マウス操作 ---
    def mousePressEvent(self, event: QMouseEvent):
        self.last_mouse_pos = event.pos()

        if event.button() == Qt.RightButton and self.measure_mode:
            self.measure_points.clear()
            self.measure_result_text = ""
            self.update()
            return

        if event.button() == Qt.RightButton and self.selection_mode and self.selection_shape == self.SELECT_POLYGON:
            if len(self.polygon_points) > 0:
                self.polygon_points.pop()
                self.update()
            return

        if self.measure_mode and event.button() == Qt.LeftButton:
            hit = self._pick_nearest_point(event.pos())
            if hit is not None:
                centered_pt, world_pt = hit
                self.measure_points.append((centered_pt, world_pt))
                if len(self.measure_points) == 2:
                    p1 = self.measure_points[0][1]
                    p2 = self.measure_points[1][1]
                    dist_3d = np.linalg.norm(p2 - p1)
                    dist_xy = np.linalg.norm(p2[:2] - p1[:2])
                    dist_z = abs(p2[2] - p1[2])
                    self.measure_result_text = (
                        f"直線距離: {dist_3d:.3f}m | 水平: {dist_xy:.3f}m | 高低差: {dist_z:.3f}m"
                    )
                    self.measure_updated.emit(self.measure_result_text)
                    self.status_changed.emit(f"計測完了: 直線距離 {dist_3d:.3f} m")
                elif len(self.measure_points) > 2:
                    self.measure_points = [(centered_pt, world_pt)]
                    self.measure_result_text = ""
                self.update()
            return

        if self.selection_mode and event.button() == Qt.LeftButton:
            if self.selection_shape == self.SELECT_POLYGON:
                self.polygon_points.append(event.pos())
                self.update()
            else:
                self.is_selecting = True
                self.selection_start = event.pos()
                self.selection_end = event.pos()
                self.update()
            return

        if event.button() == Qt.LeftButton:
            self.is_rotating = True
        elif event.button() in (Qt.RightButton, Qt.MiddleButton):
            self.is_panning = True

    def mouseMoveEvent(self, event: QMouseEvent):
        self.current_mouse_pos = event.pos()
        dx = event.position().x() - self.last_mouse_pos.x()
        dy = event.position().y() - self.last_mouse_pos.y()
        self.last_mouse_pos = event.pos()

        if self.is_selecting:
            self.selection_end = event.pos()
            self.update()
        elif self.selection_mode and self.selection_shape == self.SELECT_POLYGON:
            self.update()
        elif self.is_rotating:
            self.yaw += dx * 0.5
            self.pitch = max(-89.9, min(89.9, self.pitch - dy * 0.5))
            self.update()
        elif self.is_panning:
            # 見ている画面に対するパン平行移動
            _, right, up = self._get_camera_vectors()
            pan_speed = self.camera_distance * 0.0015
            delta = (-dx * pan_speed * right) + (dy * pan_speed * up)
            self.camera_target.setX(self.camera_target.x() + float(delta[0]))
            self.camera_target.setY(self.camera_target.y() + float(delta[1]))
            self.camera_target.setZ(self.camera_target.z() + float(delta[2]))
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self.is_selecting:
            self.is_selecting = False
            self.selection_end = event.pos()
            self._process_selection()
            self.update()

        self.is_rotating = False
        self.is_panning = False

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        if self.selection_mode and self.selection_shape == self.SELECT_POLYGON:
            if len(self.polygon_points) >= 3:
                self._process_polygon_selection()
                self.polygon_points.clear()
                self.update()
            return

        if event.button() == Qt.LeftButton:
            hit = self._pick_nearest_point(event.pos())
            if hit is not None:
                centered_pt, world_pt = hit
                self.camera_target = QVector3D(float(centered_pt[0]), float(centered_pt[1]), float(centered_pt[2]))
                self.update()
                self.status_changed.emit(
                    f"中心フォーカス: X={world_pt[0]:.2f}, Y={world_pt[1]:.2f}, Z={world_pt[2]:.2f}"
                )

    def wheelEvent(self, event: QWheelEvent):
        delta = event.angleDelta().y()
        factor = 0.85 if delta > 0 else 1.18
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
        move_speed = max(0.2, self.camera_distance * 0.05)
        forward, right, up = self._get_camera_vectors()

        if key == Qt.Key_W:
            self.camera_target.setX(self.camera_target.x() + float(forward[0] * move_speed))
            self.camera_target.setY(self.camera_target.y() + float(forward[1] * move_speed))
            self.camera_target.setZ(self.camera_target.z() + float(forward[2] * move_speed))
            self.update()
        elif key == Qt.Key_S:
            self.camera_target.setX(self.camera_target.x() - float(forward[0] * move_speed))
            self.camera_target.setY(self.camera_target.y() - float(forward[1] * move_speed))
            self.camera_target.setZ(self.camera_target.z() - float(forward[2] * move_speed))
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
            self.camera_target.setX(self.camera_target.x() + float(up[0] * move_speed))
            self.camera_target.setY(self.camera_target.y() + float(up[1] * move_speed))
            self.camera_target.setZ(self.camera_target.z() + float(up[2] * move_speed))
            self.update()
        elif key == Qt.Key_E:
            self.camera_target.setX(self.camera_target.x() - float(up[0] * move_speed))
            self.camera_target.setY(self.camera_target.y() - float(up[1] * move_speed))
            self.camera_target.setZ(self.camera_target.z() - float(up[2] * move_speed))
            self.update()
        elif key in (Qt.Key_Return, Qt.Key_Enter):
            if self.selection_mode and self.selection_shape == self.SELECT_POLYGON and len(self.polygon_points) >= 3:
                self._process_polygon_selection()
                self.polygon_points.clear()
                self.update()
        elif key == Qt.Key_Escape:
            self.polygon_points.clear()
            self.update()
        elif key == Qt.Key_F:
            self.fit_to_screen()
        elif key == Qt.Key_P:
            self.toggle_projection()
        elif key == Qt.Key_M:
            self.measure_mode = not self.measure_mode
            if not self.measure_mode:
                self.measure_points.clear()
                self.measure_result_text = ""
            self.update()
            mode_s = "ON" if self.measure_mode else "OFF"
            self.status_changed.emit(f"寸法計測ツール: {mode_s} (M)")
        else:
            super().keyPressEvent(event)

    def _pick_nearest_point(self, click_pos: QPoint) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if self.display_points is None or len(self.display_points) == 0:
            return None

        pts = self.display_points
        step = max(1, len(pts) // 50000)
        sub_pts = pts[::step]

        scr_x, scr_y, valid_z = self._project_to_screen(sub_pts)

        cx, cy = click_pos.x(), click_pos.y()
        dists_sq = (scr_x - cx) ** 2 + (scr_y - cy) ** 2
        dists_sq[~valid_z] = 1e9

        min_idx = np.argmin(dists_sq)
        if dists_sq[min_idx] < (30.0 ** 2):
            best_centered = sub_pts[min_idx]
            best_world = best_centered.astype(np.float64) + self.center_offset
            return best_centered, best_world

        return None

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

    def _process_selection(self):
        if self.points_centered is None or len(self.points_centered) == 0:
            return

        in_section = np.ones(len(self.points_raw), dtype=bool)
        if self.clip_x_enabled:
            in_section &= (self.points_raw[:, 0] >= self.clip_x_range[0]) & (self.points_raw[:, 0] <= self.clip_x_range[1])
        if self.clip_y_enabled:
            in_section &= (self.points_raw[:, 1] >= self.clip_y_range[0]) & (self.points_raw[:, 1] <= self.clip_y_range[1])
        if self.clip_z_enabled:
            in_section &= (self.points_raw[:, 2] >= self.clip_z_range[0]) & (self.points_raw[:, 2] <= self.clip_z_range[1])

        cand_indices = np.where(in_section)[0]
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
        self.selected_indices = selected
        self._update_active_colors()
        self._update_display_arrays()
        self.point_selected.emit(selected.tolist())
        self.status_changed.emit(f"範囲選択完了: {len(selected):,} 点を選択中")

    def _process_polygon_selection(self):
        if self.points_centered is None or len(self.points_centered) == 0 or len(self.polygon_points) < 3:
            return

        in_section = np.ones(len(self.points_raw), dtype=bool)
        if self.clip_x_enabled:
            in_section &= (self.points_raw[:, 0] >= self.clip_x_range[0]) & (self.points_raw[:, 0] <= self.clip_x_range[1])
        if self.clip_y_enabled:
            in_section &= (self.points_raw[:, 1] >= self.clip_y_range[0]) & (self.points_raw[:, 1] <= self.clip_y_range[1])
        if self.clip_z_enabled:
            in_section &= (self.points_raw[:, 2] >= self.clip_z_range[0]) & (self.points_raw[:, 2] <= self.clip_z_range[1])

        cand_indices = np.where(in_section)[0]
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
        self.selected_indices = selected
        self._update_active_colors()
        self._update_display_arrays()
        self.point_selected.emit(selected.tolist())
        self.status_changed.emit(f"多角形選択完了: {len(selected):,} 点を選択中")
