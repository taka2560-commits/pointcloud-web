"""
PySide6 / OpenGLによる3D点群ビューアウィジェット
高DPI対応、正射影/透視投影切替、WASD移動、全体フィット、
ダブルクリック中心移動、2点間寸法計測、表示点数バジェット(間引き表示)を完備。
"""

import math
from typing import Optional, Tuple, List
import numpy as np
import OpenGL.GL as gl

from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtGui import (
    QMatrix4x4,
    QVector3D,
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
    measure_updated = Signal(str)       # 計測結果メッセージ (直線・水平・高低差)
    projection_changed = Signal(bool)   # 投影モード変更通知 (True: 正射影, False: 透視投影)

    COLOR_MODE_RGB = "RGB"
    COLOR_MODE_HEIGHT = "高さ (Z)"
    COLOR_MODE_INTENSITY = "反射強度"
    COLOR_MODE_SOLID = "単色"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)  # キーボードイベントを確実に受領

        # カメラパラメータ
        self.camera_target = QVector3D(0.0, 0.0, 0.0)  # 注視点（点群中心）
        self.camera_distance = 15.0                     # カメラ距離
        self.yaw = 45.0                                # 水平角（度）
        self.pitch = 30.0                              # 仰角（度）
        self.fov = 45.0                                # 視野角（度）
        self.is_orthographic = False                   # 正射影フラグ (False=透視投影, True=正射影)

        # 点群データ (NumPy配列)
        self.points_raw = None        # (N, 3) float64 原本
        self.points_centered = None   # (N, 3) float32 中心オフセット済み原本
        self.center_offset = np.zeros(3, dtype=np.float64)
        self.colors_raw = None        # (N, 3) float32
        self.intensities_raw = None   # (N,) float32
        self.active_colors = None     # (N, 3) float32 描画用カラー原本

        # 表示用間引き配列 (Point Budget対応)
        self.display_points = None    # 描画対象 (M, 3) float32
        self.display_colors = None    # 描画対象カラー (M, 3) float32
        self.display_indices = None   # 原本インデックスとのマッピング
        self.point_budget = 2000000   # 最大描画点数 (初期値: 200万点、0で全点無制限)

        # 表示設定
        self.point_size = 2.0
        self.point_shape = "circle"   # "circle" または "square"
        self.color_mode = self.COLOR_MODE_RGB
        self.bg_color = QColor(20, 20, 26)
        self.show_grid = True
        self.bounds_min = np.array([-10, -10, -5], dtype=np.float32)
        self.bounds_max = np.array([10, 10, 5], dtype=np.float32)

        # マウス操作用
        self.last_mouse_pos = QPoint()
        self.is_rotating = False
        self.is_panning = False

        # 手動矩形選択モード
        self.selection_mode = False
        self.selection_start = QPoint()
        self.selection_end = QPoint()
        self.is_selecting = False
        self.selected_indices = np.array([], dtype=np.int64)

        # 2点間寸法計測ツール
        self.measure_mode = False
        self.measure_points: List[Tuple[np.ndarray, np.ndarray]] = []  # [(centered_pos, world_pos)]
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
        """OpenGL描画ループ（高DPI・正射影・計測線・グリッド対応）"""
        # 高DPIスケーリングに合わせた正確なビューポート設定
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

        # 1. プロジェクション行列の設定 (透視投影 / 正射影)
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

        # 2. モデルビュー行列の設定
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        view = self._get_view_matrix()
        gl.glLoadMatrixf(view.data())

        # 3. 地面グリッドの描画
        if self.show_grid:
            self._draw_grid()

        # 4. 点群の描画
        if self.display_points is not None and len(self.display_points) > 0:
            gl.glPointSize(float(self.point_size))

            if self.point_shape == "circle":
                gl.glEnable(gl.GL_POINT_SMOOTH)
            else:
                gl.glDisable(gl.GL_POINT_SMOOTH)

            # 頂点配列とカラー配列を直接バインドして高速描画
            gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
            gl.glEnableClientState(gl.GL_COLOR_ARRAY)

            gl.glVertexPointer(3, gl.GL_FLOAT, 0, self.display_points)
            gl.glColorPointer(3, gl.GL_FLOAT, 0, self.display_colors)

            gl.glDrawArrays(gl.GL_POINTS, 0, len(self.display_points))

            gl.glDisableClientState(gl.GL_COLOR_ARRAY)
            gl.glDisableClientState(gl.GL_VERTEX_ARRAY)

        # 5. 計測ラインの描画 (OpenGL)
        if len(self.measure_points) > 0:
            self._draw_measure_lines()

        # 6. 座標軸ギズモの描画 (画面左下)
        self._draw_axes()

        # 7. 2Dオーバーレイ描画 (矩形選択枠 / 計測情報)
        self._draw_2d_overlays()

    def _get_view_matrix(self) -> QMatrix4x4:
        """カメラのビュー行列を計算する"""
        view = QMatrix4x4()
        view.translate(0.0, 0.0, -self.camera_distance)
        view.rotate(self.pitch, 1.0, 0.0, 0.0)
        view.rotate(self.yaw, 0.0, 0.0, 1.0)
        view.translate(-self.camera_target)
        return view

    def _draw_grid(self):
        """地面グリッドを描画"""
        gl.glPushAttrib(gl.GL_ALL_ATTRIB_BITS)
        gl.glDisable(gl.GL_LIGHTING)
        gl.glLineWidth(1.0)

        # 点群のZ最小値を基準とする
        z_floor = float(self.bounds_min[2]) if self.points_raw is not None else 0.0

        grid_size = max(20.0, float(self.camera_distance * 1.5))
        step = 5.0 if grid_size > 50 else (1.0 if grid_size > 10 else 0.5)

        half = int(grid_size / step)

        gl.glBegin(gl.GL_LINES)
        gl.glColor4f(0.25, 0.28, 0.35, 0.4)
        for i in range(-half, half + 1):
            coord = i * step
            # X方向ライン
            gl.glVertex3f(-half * step, coord, z_floor)
            gl.glVertex3f(half * step, coord, z_floor)
            # Y方向ライン
            gl.glVertex3f(coord, -half * step, z_floor)
            gl.glVertex3f(coord, half * step, z_floor)
        gl.glEnd()

        gl.glPopAttrib()

    def _draw_measure_lines(self):
        """2点間寸法計測のラインを描画"""
        gl.glPushAttrib(gl.GL_ALL_ATTRIB_BITS)
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glLineWidth(2.5)

        # 1点目マーカー
        p1 = self.measure_points[0][0]
        gl.glPointSize(10.0)
        gl.glBegin(gl.GL_POINTS)
        gl.glColor3f(1.0, 0.8, 0.0)
        gl.glVertex3f(p1[0], p1[1], p1[2])
        gl.glEnd()

        if len(self.measure_points) >= 2:
            p2 = self.measure_points[1][0]
            # 2点目マーカー
            gl.glBegin(gl.GL_POINTS)
            gl.glColor3f(0.0, 1.0, 0.8)
            gl.glVertex3f(p2[0], p2[1], p2[2])
            gl.glEnd()

            # 結ぶ直線
            gl.glBegin(gl.GL_LINES)
            gl.glColor3f(1.0, 0.9, 0.1)
            gl.glVertex3f(p1[0], p1[1], p1[2])
            gl.glVertex3f(p2[0], p2[1], p2[2])
            gl.glEnd()

        gl.glPopAttrib()

    def _draw_axes(self):
        """画面左下に小さなXYZ軸ギズモを描画する"""
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
        gl.glTranslatef(-aspect + 0.15, -0.82, 0.0)
        gl.glRotatef(self.pitch, 1.0, 0.0, 0.0)
        gl.glRotatef(self.yaw, 0.0, 0.0, 1.0)
        gl.glScalef(0.12, 0.12, 0.12)

        gl.glLineWidth(3.0)
        gl.glBegin(gl.GL_LINES)
        # X軸: 赤
        gl.glColor3f(1.0, 0.2, 0.2)
        gl.glVertex3f(0.0, 0.0, 0.0)
        gl.glVertex3f(1.0, 0.0, 0.0)
        # Y軸: 緑
        gl.glColor3f(0.2, 1.0, 0.2)
        gl.glVertex3f(0.0, 0.0, 0.0)
        gl.glVertex3f(0.0, 1.0, 0.0)
        # Z軸: 青
        gl.glColor3f(0.3, 0.6, 1.0)
        gl.glVertex3f(0.0, 0.0, 0.0)
        gl.glVertex3f(0.0, 0.0, 1.0)
        gl.glEnd()

        gl.glPopMatrix()
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glPopMatrix()
        gl.glPopAttrib()

    def _draw_2d_overlays(self):
        """矩形選択枠や寸法計測情報の2Dオーバーレイ描画"""
        if not self.is_selecting and not self.measure_result_text and not self.measure_mode:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 矩形選択中
        if self.is_selecting:
            rect = QRect(self.selection_start, self.selection_end).normalized()
            pen = QPen(QColor(0, 210, 255, 230), 2, Qt.DashLine)
            painter.setPen(pen)
            brush = QBrush(QColor(0, 150, 255, 45))
            painter.setBrush(brush)
            painter.drawRect(rect)

        # 計測モード通知または結果表示
        if self.measure_result_text:
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(15, 23, 42, 220)))
            box_rect = QRect(14, 14, 380, 75)
            painter.drawRoundedRect(box_rect, 8, 8)

            painter.setPen(QColor(56, 189, 248))
            font = QFont("Segoe UI", 11, QFont.Bold)
            painter.setFont(font)
            painter.drawText(26, 36, "📐 2点間寸法計測結果")

            painter.setPen(QColor(241, 245, 249))
            font_res = QFont("Consolas", 10, QFont.Normal)
            painter.setFont(font_res)
            painter.drawText(26, 56, self.measure_result_text)
            painter.drawText(26, 74, "※右クリックでクリア / Mキーで終了")
        elif self.measure_mode:
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(15, 23, 42, 200)))
            box_rect = QRect(14, 14, 320, 50)
            painter.drawRoundedRect(box_rect, 8, 8)

            painter.setPen(QColor(250, 204, 21))
            font = QFont("Segoe UI", 10, QFont.Bold)
            painter.setFont(font)
            msg = "計測点 1 をクリックしてください" if len(self.measure_points) == 0 else "計測点 2 をクリックしてください"
            painter.drawText(24, 34, f"📐 計測モード: {msg}")
            painter.drawText(24, 52, "※右クリックでキャンセル")

        painter.end()

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

        # 中心オフセット（大座標系でのグラフィックジッター完全防止）
        min_b = np.min(self.points_raw, axis=0)
        max_b = np.max(self.points_raw, axis=0)
        self.center_offset = (min_b + max_b) / 2.0
        self.points_centered = np.ascontiguousarray((self.points_raw - self.center_offset), dtype=np.float32)

        self.bounds_min = (min_b - self.center_offset).astype(np.float32)
        self.bounds_max = (max_b - self.center_offset).astype(np.float32)

        # カラーデータの正規化
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

        # 全体のカラー配列を生成
        self._update_active_colors()
        # 表示用バジェット配列を生成
        self._update_display_arrays()

        # カメラのリセット
        if reset_camera:
            self.fit_to_screen()

        self.update()
        self.status_changed.emit(f"点群読み込み完了: {len(points):,} 点")

    def _update_active_colors(self):
        """カラーモードに応じて原本全体のカラー配列を生成"""
        if self.points_centered is None:
            return

        n = len(self.points_centered)

        if self.color_mode == self.COLOR_MODE_RGB and self.colors_raw is not None:
            self.active_colors = self.colors_raw.copy()
        elif self.color_mode == self.COLOR_MODE_HEIGHT:
            z = self.points_centered[:, 2]
            z_min, z_max = np.min(z), np.max(z)
            t = (z - z_min) / max(1e-5, (z_max - z_min))
            self.active_colors = self._colormap_turbo(t)
        elif self.color_mode == self.COLOR_MODE_INTENSITY and self.intensities_raw is not None:
            iv = self.intensities_raw
            i_min, i_max = np.min(iv), np.max(iv)
            t = (iv - i_min) / max(1e-5, (i_max - i_min))
            self.active_colors = np.column_stack([t, t, t]).astype(np.float32)
        else:
            # デフォルトシアンカラー
            self.active_colors = np.tile([0.3, 0.8, 0.95], (n, 1)).astype(np.float32)

        # 選択中の点を赤色でハイライト
        if len(self.selected_indices) > 0:
            valid = self.selected_indices[self.selected_indices < n]
            self.active_colors[valid] = [1.0, 0.1, 0.1]

        self.active_colors = np.ascontiguousarray(self.active_colors, dtype=np.float32)

    def _update_display_arrays(self):
        """Point Budget（最大描画点数）に応じて表示用バッファを生成"""
        if self.points_centered is None:
            self.display_points = None
            self.display_colors = None
            self.display_indices = None
            return

        n = len(self.points_centered)
        budget = self.point_budget

        if budget <= 0 or n <= budget:
            # 全点表示
            self.display_points = self.points_centered
            self.display_colors = self.active_colors
            self.display_indices = None
        else:
            # 均等スライス間引き (2,770万点でも瞬時に生成可能)
            step = max(1, int(math.ceil(n / budget)))
            self.display_indices = np.arange(0, n, step)
            self.display_points = np.ascontiguousarray(self.points_centered[self.display_indices])
            self.display_colors = np.ascontiguousarray(self.active_colors[self.display_indices])

    def set_point_budget(self, budget: int):
        """最大描画点数を設定 (0: 全点, 1000000: 100万点 等)"""
        self.point_budget = budget
        self._update_display_arrays()
        self.update()

    def fit_to_screen(self):
        """点群全体を画面中央にぴったりフィット (Fキー)"""
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
        """透視投影と正射影の切り替え (Pキー)"""
        self.is_orthographic = not self.is_orthographic
        self.projection_changed.emit(self.is_orthographic)
        mode_str = "正射影 (平行投影)" if self.is_orthographic else "透視投影 (パースペクティブ)"
        self.status_changed.emit(f"投影モード切替: {mode_str} (P)")
        self.update()

    def set_orthographic(self, ortho: bool):
        self.is_orthographic = ortho
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
        """鮮やかな高さグラデーション"""
        t = np.clip(t, 0.0, 1.0)
        r = np.clip(1.5 - np.abs(t * 4.0 - 3.0), 0.0, 1.0)
        g = np.clip(1.5 - np.abs(t * 4.0 - 2.0), 0.0, 1.0)
        b = np.clip(1.5 - np.abs(t * 4.0 - 1.0), 0.0, 1.0)
        return np.column_stack([r, g, b]).astype(np.float32)

    # --- 視点プリセット ---
    def reset_view(self):
        self.yaw = 45.0
        self.pitch = 30.0
        self.camera_target = QVector3D(0.0, 0.0, 0.0)
        self.update()

    def set_view_top(self):
        self.yaw = 0.0
        self.pitch = 89.9
        self.update()

    def set_view_front(self):
        self.yaw = 0.0
        self.pitch = 0.0
        self.update()

    def set_view_side(self):
        self.yaw = -90.0
        self.pitch = 0.0
        self.update()

    # --- マウス操作 ---
    def mousePressEvent(self, event: QMouseEvent):
        self.last_mouse_pos = event.pos()

        if event.button() == Qt.RightButton and self.measure_mode:
            # 計測のリセット
            self.measure_points.clear()
            self.measure_result_text = ""
            self.update()
            return

        if self.measure_mode and event.button() == Qt.LeftButton:
            # クリック箇所の最も近い点を取得
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
            self.is_selecting = True
            self.selection_start = event.pos()
            self.selection_end = event.pos()
            self.update()
        elif event.button() == Qt.LeftButton:
            self.is_rotating = True
        elif event.button() in (Qt.RightButton, Qt.MiddleButton):
            self.is_panning = True

    def mouseMoveEvent(self, event: QMouseEvent):
        dx = event.position().x() - self.last_mouse_pos.x()
        dy = event.position().y() - self.last_mouse_pos.y()
        self.last_mouse_pos = event.pos()

        if self.is_selecting:
            self.selection_end = event.pos()
            self.update()
        elif self.is_rotating:
            self.yaw += dx * 0.5
            self.pitch = max(-89.0, min(89.0, self.pitch + dy * 0.5))
            self.update()
        elif self.is_panning:
            pan_speed = self.camera_distance * 0.0015
            rad_y = math.radians(self.yaw)
            sin_y, cos_y = math.sin(rad_y), math.cos(rad_y)
            self.camera_target.setX(self.camera_target.x() - dx * pan_speed * cos_y)
            self.camera_target.setY(self.camera_target.y() - dx * pan_speed * sin_y)
            self.camera_target.setZ(self.camera_target.z() + dy * pan_speed)
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self.is_selecting:
            self.is_selecting = False
            self.selection_end = event.pos()
            self._process_selection_rect()
            self.update()

        self.is_rotating = False
        self.is_panning = False

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        """ダブルクリックでその場所を中心フォーカス"""
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
        """WASDキー移動 & 各種ショートカット"""
        key = event.key()
        move_speed = max(0.2, self.camera_distance * 0.05)
        rad_y = math.radians(self.yaw)
        sin_y, cos_y = math.sin(rad_y), math.cos(rad_y)

        if key == Qt.Key_W:
            # 前進 (注視点方向へ寄る)
            self.camera_target.setX(self.camera_target.x() + sin_y * move_speed)
            self.camera_target.setY(self.camera_target.y() - cos_y * move_speed)
            self.update()
        elif key == Qt.Key_S:
            # 後退
            self.camera_target.setX(self.camera_target.x() - sin_y * move_speed)
            self.camera_target.setY(self.camera_target.y() + cos_y * move_speed)
            self.update()
        elif key == Qt.Key_A:
            # 左移動
            self.camera_target.setX(self.camera_target.x() - cos_y * move_speed)
            self.camera_target.setY(self.camera_target.y() - sin_y * move_speed)
            self.update()
        elif key == Qt.Key_D:
            # 右移動
            self.camera_target.setX(self.camera_target.x() + cos_y * move_speed)
            self.camera_target.setY(self.camera_target.y() + sin_y * move_speed)
            self.update()
        elif key == Qt.Key_Q:
            # 上昇
            self.camera_target.setZ(self.camera_target.z() + move_speed)
            self.update()
        elif key == Qt.Key_E:
            # 下降
            self.camera_target.setZ(self.camera_target.z() - move_speed)
            self.update()
        elif key == Qt.Key_F:
            # 全体フィット
            self.fit_to_screen()
        elif key == Qt.Key_P:
            # 投影切り替え
            self.toggle_projection()
        elif key == Qt.Key_M:
            # 計測モードトグル
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
        """スクリーン上のクリック位置から最も近い点を探索 (表示点群から高速探索)"""
        if self.display_points is None or len(self.display_points) == 0:
            return None

        # サンプリング点が多すぎる場合は最大5万点に絞って高速スクリーン投影
        pts = self.display_points
        step = max(1, len(pts) // 50000)
        sub_pts = pts[::step]

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

        ones = np.ones((len(sub_pts), 1), dtype=np.float32)
        pts4 = np.hstack([sub_pts, ones])
        clip_pts = pts4 @ mvp_np

        w = clip_pts[:, 3]
        valid_z = w > 0.001
        ndc_x = clip_pts[:, 0] / np.where(valid_z, w, 1.0)
        ndc_y = clip_pts[:, 1] / np.where(valid_z, w, 1.0)

        scr_x = (ndc_x + 1.0) * 0.5 * self.width()
        scr_y = (1.0 - ndc_y) * 0.5 * self.height()

        cx, cy = click_pos.x(), click_pos.y()
        dists_sq = (scr_x - cx) ** 2 + (scr_y - cy) ** 2
        dists_sq[~valid_z] = 1e9

        min_idx = np.argmin(dists_sq)
        if dists_sq[min_idx] < (30.0 ** 2):  # 30ピクセル以内の最近傍
            best_centered = sub_pts[min_idx]
            best_world = best_centered.astype(np.float64) + self.center_offset
            return best_centered, best_world

        return None

    def _process_selection_rect(self):
        """スクリーン上の矩形選択範囲に含まれる原本点を判定"""
        if self.points_centered is None or len(self.points_centered) == 0:
            return

        rect = QRect(self.selection_start, self.selection_end).normalized()
        if rect.width() < 4 or rect.height() < 4:
            return

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

        pts = self.points_centered
        ones = np.ones((len(pts), 1), dtype=np.float32)
        pts4 = np.hstack([pts, ones])

        clip_pts = pts4 @ mvp_np

        w = clip_pts[:, 3]
        valid_z = w > 0.001

        ndc_x = clip_pts[:, 0] / np.where(valid_z, w, 1.0)
        ndc_y = clip_pts[:, 1] / np.where(valid_z, w, 1.0)

        scr_x = (ndc_x + 1.0) * 0.5 * self.width()
        scr_y = (1.0 - ndc_y) * 0.5 * self.height()

        in_rect = (
            valid_z
            & (scr_x >= rect.left())
            & (scr_x <= rect.right())
            & (scr_y >= rect.top())
            & (scr_y <= rect.bottom())
        )

        selected = np.where(in_rect)[0]
        self.selected_indices = selected
        self._update_active_colors()
        self._update_display_arrays()
        self.point_selected.emit(selected.tolist())
        self.status_changed.emit(f"範囲選択完了: {len(selected):,} 点を選択中")
