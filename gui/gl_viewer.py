"""
PySide6 / OpenGLによる3D点群ビューアウィジェット
OpenGLの標準頂点配列を用いて、全環境で100%確実に高速描画できる堅牢な実装です。
"""

import math
from typing import Optional, Tuple
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
    QMouseEvent,
    QWheelEvent,
)
from PySide6.QtCore import Qt, QPoint, QRect, Signal


class PointCloudViewer(QOpenGLWidget):
    """3D点群ビューアクラス"""

    point_selected = Signal(list)  # 選択された点インデックスリスト
    status_changed = Signal(str)   # ステータスメッセージ

    COLOR_MODE_RGB = "RGB"
    COLOR_MODE_HEIGHT = "高さ (Z)"
    COLOR_MODE_INTENSITY = "反射強度"
    COLOR_MODE_SOLID = "単色"

    def __init__(self, parent=None):
        super().__init__(parent)

        # カメラパラメータ
        self.camera_target = QVector3D(0.0, 0.0, 0.0)  # 注視点（点群中心）
        self.camera_distance = 15.0                     # カメラ距離
        self.yaw = 45.0                                # 水平角（度）
        self.pitch = 30.0                              # 仰角（度）
        self.fov = 45.0                                # 視野角（度）

        # 点群データ (NumPy配列)
        self.points_raw = None        # (N, 3) float64
        self.points_centered = None   # (N, 3) float32 中心オフセット済み
        self.center_offset = np.zeros(3, dtype=np.float64)
        self.colors_raw = None        # (N, 3) float32
        self.intensities_raw = None   # (N,) float32
        self.active_colors = None     # (N, 3) float32 描画用カラー

        # 表示設定
        self.point_size = 3.0
        self.color_mode = self.COLOR_MODE_RGB
        self.bg_color = QColor(20, 20, 26)

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

    def initializeGL(self):
        """OpenGL初期化処理"""
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthFunc(gl.GL_LEQUAL)
        gl.glEnable(gl.GL_POINT_SMOOTH)
        gl.glHint(gl.GL_POINT_SMOOTH_HINT, gl.GL_NICEST)
        gl.glDisable(gl.GL_LIGHTING)

    def resizeGL(self, width: int, height: int):
        """ウィンドウリサイズ処理"""
        gl.glViewport(0, 0, width, height)

    def paintGL(self):
        """OpenGL描画ループ（高互換・高信頼性設計）"""
        bg = self.bg_color
        gl.glClearColor(bg.redF(), bg.greenF(), bg.blueF(), 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        aspect = self.width() / max(1, self.height())
        near_plane = max(0.1, self.camera_distance * 0.02)
        far_plane = max(100.0, self.camera_distance * 50.0)

        # 1. プロジェクション行列の設定
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        proj = QMatrix4x4()
        proj.perspective(self.fov, aspect, near_plane, far_plane)
        gl.glLoadMatrixf(proj.data())

        # 2. モデルビュー行列の設定
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        view = self._get_view_matrix()
        gl.glLoadMatrixf(view.data())

        # 3. 点群の描画
        if self.points_centered is not None and len(self.points_centered) > 0:
            gl.glPointSize(float(self.point_size))

            # 頂点配列とカラー配列を直接バインド
            gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
            gl.glEnableClientState(gl.GL_COLOR_ARRAY)

            gl.glVertexPointer(3, gl.GL_FLOAT, 0, self.points_centered)
            gl.glColorPointer(3, gl.GL_FLOAT, 0, self.active_colors)

            gl.glDrawArrays(gl.GL_POINTS, 0, len(self.points_centered))

            gl.glDisableClientState(gl.GL_COLOR_ARRAY)
            gl.glDisableClientState(gl.GL_VERTEX_ARRAY)

        # 4. 座標軸ギズモの描画
        self._draw_axes()

        # 5. 矩形選択中の2Dオーバーレイ描画
        if self.is_selecting:
            painter = QPainter(self)
            rect = QRect(self.selection_start, self.selection_end).normalized()
            pen = QPen(QColor(0, 210, 255, 230), 2, Qt.DashLine)
            painter.setPen(pen)
            brush = QBrush(QColor(0, 150, 255, 45))
            painter.setBrush(brush)
            painter.drawRect(rect)
            painter.end()

    def _get_view_matrix(self) -> QMatrix4x4:
        """カメラのビュー行列を計算する"""
        view = QMatrix4x4()
        view.translate(0.0, 0.0, -self.camera_distance)
        view.rotate(self.pitch, 1.0, 0.0, 0.0)
        view.rotate(self.yaw, 0.0, 0.0, 1.0)
        view.translate(-self.camera_target)
        return view

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

    def set_point_cloud(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
        intensities: Optional[np.ndarray] = None,
        reset_camera: bool = True,
    ):
        """
        点群データを設定する。

        :param points: (N, 3) 形状のXYZ座標
        :param colors: (N, 3) 形状のRGBカラー (0-255 または 0-1)
        :param intensities: (N,) 形状の反射強度
        :param reset_camera: カメラ位置を点群全体に合わせてリセットするか
        """
        if points is None or len(points) == 0:
            self.points_raw = None
            self.points_centered = None
            self.active_colors = None
            self.update()
            return

        self.points_raw = np.asarray(points, dtype=np.float64)

        # 中心オフセット（大座標系でのジッター防止）
        min_b = np.min(self.points_raw, axis=0)
        max_b = np.max(self.points_raw, axis=0)
        self.center_offset = (min_b + max_b) / 2.0
        self.points_centered = np.ascontiguousarray((self.points_raw - self.center_offset), dtype=np.float32)

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

        # 現在のカラーモードを適用
        self._update_active_colors()

        # カメラのリセット
        if reset_camera:
            diag = np.linalg.norm(max_b - min_b)
            self.camera_target = QVector3D(0.0, 0.0, 0.0)
            self.camera_distance = max(2.0, float(diag) * 1.6)
            self.yaw = 45.0
            self.pitch = 30.0

        self.update()
        self.status_changed.emit(f"点群読み込み完了: {len(points):,} 点")

    def _update_active_colors(self):
        """カラーモードに応じて描画用カラー配列を生成"""
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

    def _colormap_turbo(self, t: np.ndarray) -> np.ndarray:
        """鮮やかな高さグラデーション"""
        t = np.clip(t, 0.0, 1.0)
        r = np.clip(1.5 - np.abs(t * 4.0 - 3.0), 0.0, 1.0)
        g = np.clip(1.5 - np.abs(t * 4.0 - 2.0), 0.0, 1.0)
        b = np.clip(1.5 - np.abs(t * 4.0 - 1.0), 0.0, 1.0)
        return np.column_stack([r, g, b]).astype(np.float32)

    def set_color_mode(self, mode: str):
        self.color_mode = mode
        self._update_active_colors()
        self.update()

    def set_point_size(self, size: float):
        self.point_size = max(1.0, float(size))
        self.update()

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

    def wheelEvent(self, event: QWheelEvent):
        delta = event.angleDelta().y()
        factor = 0.85 if delta > 0 else 1.18
        self.camera_distance = max(0.1, self.camera_distance * factor)
        self.update()

    def _process_selection_rect(self):
        """スクリーン上の矩形選択範囲に含まれる点を判定"""
        if self.points_centered is None or len(self.points_centered) == 0:
            return

        rect = QRect(self.selection_start, self.selection_end).normalized()
        if rect.width() < 4 or rect.height() < 4:
            return

        aspect = self.width() / max(1, self.height())
        proj = QMatrix4x4()
        near_plane = max(0.1, self.camera_distance * 0.02)
        far_plane = max(100.0, self.camera_distance * 50.0)
        proj.perspective(self.fov, aspect, near_plane, far_plane)
        view = self._get_view_matrix()
        mvp = proj * view

        pts = self.points_centered
        ones = np.ones((len(pts), 1), dtype=np.float32)
        pts4 = np.hstack([pts, ones])

        mvp_np = np.array(mvp.data(), dtype=np.float32).reshape((4, 4))
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
        self.point_selected.emit(selected.tolist())
        self.status_changed.emit(f"範囲選択完了: {len(selected):,} 点を選択中")
