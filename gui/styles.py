"""
GUIのスタイルシート定義（モダン・ダークテーマ）
"""

DARK_THEME_QSS = """
QMainWindow {
    background-color: #1e1e24;
    color: #e0e0e0;
    font-family: "Segoe UI", "Meiryo UI", sans-serif;
    font-size: 13px;
}

QWidget {
    background-color: #1e1e24;
    color: #e0e0e0;
}

QGroupBox {
    border: 1px solid #3d3d4d;
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 14px;
    font-weight: bold;
    color: #4da6ff;
    background-color: #252530;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 14px;
    padding: 0 6px;
    background-color: #252530;
}

QLabel {
    color: #cccccc;
}

QLineEdit {
    background-color: #16161c;
    border: 1px solid #444455;
    border-radius: 5px;
    padding: 6px 10px;
    color: #ffffff;
    selection-background-color: #0078d4;
}

QLineEdit:focus {
    border: 1px solid #4da6ff;
}

QPushButton {
    background-color: #2e6296;
    color: #ffffff;
    border: none;
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: bold;
    min-height: 20px;
}

QPushButton:hover {
    background-color: #3978b8;
}

QPushButton:pressed {
    background-color: #1b4975;
}

QPushButton:disabled {
    background-color: #333340;
    color: #777788;
}

QPushButton#execute_button {
    background-color: #28a745;
    font-size: 14px;
    padding: 10px 20px;
}

QPushButton#execute_button:hover {
    background-color: #34c759;
}

QPushButton#execute_button:pressed {
    background-color: #1e7e34;
}

QPushButton#preview_button {
    background-color: #6f42c1;
    font-size: 13px;
}

QPushButton#preview_button:hover {
    background-color: #8553e6;
}

QCheckBox {
    color: #e0e0e0;
    spacing: 8px;
}

QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border-radius: 4px;
    border: 1px solid #555566;
    background-color: #16161c;
}

QCheckBox::indicator:checked {
    background-color: #0078d4;
    border-color: #0078d4;
}

QSpinBox, QDoubleSpinBox {
    background-color: #16161c;
    border: 1px solid #444455;
    border-radius: 5px;
    padding: 4px 8px;
    color: #ffffff;
}

QSpinBox:focus, QDoubleSpinBox:focus {
    border: 1px solid #4da6ff;
}

QProgressBar {
    border: 1px solid #3d3d4d;
    border-radius: 6px;
    text-align: center;
    background-color: #16161c;
    color: #ffffff;
    font-weight: bold;
    height: 22px;
}

QProgressBar::chunk {
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0078d4, stop:1 #00bcbc);
    border-radius: 5px;
}

QTextEdit {
    background-color: #16161c;
    border: 1px solid #3d3d4d;
    border-radius: 6px;
    padding: 8px;
    color: #b0ffb0;
    font-family: "Consolas", "Courier New", monospace;
    font-size: 12px;
}

QTabWidget::pane {
    border: 1px solid #3d3d4d;
    border-radius: 6px;
    background-color: #252530;
}

QTabBar::tab {
    background-color: #1e1e24;
    color: #aaaaaa;
    padding: 8px 18px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
}

QTabBar::tab:selected {
    background-color: #252530;
    color: #4da6ff;
    font-weight: bold;
}
"""
