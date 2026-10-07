"""
ユーザー指定デザイン (--motion-accent: #22bdd6) に完全準拠した
モダンダーク・シアンアクセントの進捗ダイアログ (モーダル)
"""

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QWidget,
    QGraphicsDropShadowEffect,
)
from PySide6.QtGui import QColor, QFont


class ModernProgressDialog(QDialog):
    """
    画面中央に表示される滑らかな進捗ダイアログ
    ユーザー提示の motion-stage (#22bdd6) 仕様を完全再現
    """

    def __init__(self, title: str = "点群データ読み込み中", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedSize(460, 210)

        self._current_value = 0
        self._target_value = 0

        self._init_ui()

        # 滑らかな数値・フィル補間アニメーション用タイマー (60FPS)
        self._anim_timer = QTimer(self)
        self._anim_timer.timeout.connect(self._update_smooth_progress)
        self._anim_timer.start(16)

    def _init_ui(self):
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(12, 12, 12, 12)

        # メインカードコンテナ
        container = QWidget()
        container.setObjectName("progressContainer")
        container.setStyleSheet("""
            QWidget#progressContainer {
                background-color: #0b1120;
                border: 1px solid #1e293b;
                border-radius: 12px;
            }
        """)

        # ドロップシャドウ
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(28)
        shadow.setColor(QColor(0, 0, 0, 180))
        shadow.setOffset(0, 8)
        container.setGraphicsEffect(shadow)

        layout = QVBoxLayout(container)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(10)

        # 1. 上部ヘッダー (ラベル & パーセント数値)
        head_layout = QHBoxLayout()

        self.label_tag = QLabel("LOADING")
        self.label_tag.setStyleSheet("""
            color: #22bdd6;
            font-size: 11px;
            font-weight: 800;
            letter-spacing: 1.5px;
            font-family: 'Segoe UI', sans-serif;
        """)
        head_layout.addWidget(self.label_tag)
        head_layout.addStretch()

        self.label_percent = QLabel("0%")
        self.label_percent.setStyleSheet("""
            color: #ffffff;
            font-size: 18px;
            font-weight: 700;
            font-family: 'Segoe UI', 'Consolas', monospace;
        """)
        head_layout.addWidget(self.label_percent)
        layout.addLayout(head_layout)

        # 2. メインメッセージ (ファイル名・処理状態)
        self.label_title = QLabel("点群ファイルを解析しています...")
        self.label_title.setStyleSheet("""
            color: #f1f5f9;
            font-size: 13px;
            font-weight: 600;
        """)
        layout.addWidget(self.label_title)

        # 3. プログレスバー (トラック & フィル)
        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self.bar.setTextVisible(False)  # 数値は右上に大きく表示するためバー内は非表示
        self.bar.setFixedHeight(12)
        self.bar.setStyleSheet("""
            QProgressBar {
                background-color: #1e293b;
                border: none;
                border-radius: 6px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0ea5e9, stop:1 #22bdd6);
                border-radius: 6px;
            }
        """)
        layout.addWidget(self.bar)

        # 4. 下部キャプション (詳細サブテキスト)
        self.label_sub = QLabel("巨大データの座標・属性を展開中...")
        self.label_sub.setStyleSheet("""
            color: #94a3b8;
            font-size: 11px;
        """)
        layout.addWidget(self.label_sub)

        root_layout.addWidget(container)

    def set_progress(self, percent: int, title: str = "", sub_text: str = ""):
        """進捗を更新 (アニメーションにより滑らかに補間)"""
        self._target_value = max(0, min(100, percent))
        if title:
            self.label_title.setText(title)
        if sub_text:
            self.label_sub.setText(sub_text)

    def _update_smooth_progress(self):
        """毎フレーム滑らかにパーセントを補間"""
        if self._current_value < self._target_value:
            # 滑らかに目標値へ近づける
            step = max(0.5, (self._target_value - self._current_value) * 0.25)
            self._current_value = min(float(self._target_value), self._current_value + step)
            int_val = int(round(self._current_value))
            self.bar.setValue(int_val)
            self.label_percent.setText(f"{int_val}%")
        elif self._current_value > self._target_value:
            self._current_value = float(self._target_value)
            int_val = int(self._current_value)
            self.bar.setValue(int_val)
            self.label_percent.setText(f"{int_val}%")
