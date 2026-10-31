"""
GUIのスタイルシート定義

- DARK_THEME_QSS: 従来UI（classic）のテーマ。改修前の見た目を保つため変更しない。
- 新UI（modern）の QSS は gui/design_tokens.py のトークンから生成する（get_theme_qss("modern")）。
"""


def get_theme_qss(ui_mode: str) -> str:
    """UI モードに応じた QSS を返す（"modern" はトークン生成、それ以外は従来テーマ）"""
    if ui_mode == "modern":
        from gui.design_tokens import build_modern_qss

        return build_modern_qss()
    return DARK_THEME_QSS


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
    margin-top: 10px;
    padding: 12px 8px 8px 8px;
    font-weight: bold;
    color: #4da6ff;
    background-color: #252530;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 5px;
    background-color: #252530;
}

QLabel {
    color: #cccccc;
}

QLineEdit {
    background-color: #16161c;
    border: 1px solid #444455;
    border-radius: 5px;
    padding: 5px 8px;
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
    padding: 5px 8px;
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
    border: 1px solid #1e293b;
    border-radius: 6px;
    text-align: center;
    background-color: #0f172a;
    color: #ffffff;
    font-weight: bold;
    font-size: 12px;
    height: 24px;
}

QProgressBar::chunk {
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0ea5e9, stop:1 #22bdd6);
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

QPushButton[role="primary"] {
    background-color: #2563eb;
    color: #ffffff;
    border: 1px solid #3b82f6;
}
QPushButton[role="primary"]:hover {
    background-color: #1d4ed8;
}

QPushButton[role="outline"] {
    background-color: #1e293b;
    color: #cbd5e1;
    border: 1px solid #475569;
    font-weight: normal;
}
QPushButton[role="outline"]:hover {
    background-color: #334155;
    color: #ffffff;
    border-color: #64748b;
}
QPushButton[role="outline"]:disabled {
    background-color: #181820;
    color: #555566;
    border-color: #2d2d3a;
}

QPushButton[role="danger"] {
    background-color: #b91c1c;
    color: #ffffff;
    border: 1px solid #ef4444;
}
QPushButton[role="danger"]:hover {
    background-color: #dc2626;
}

QPushButton[role="segment"] {
    background-color: #1e293b;
    color: #94a3b8;
    border: 1px solid #334155;
    border-radius: 5px;
    padding: 4px 8px;
    font-size: 11px;
}
QPushButton[role="segment"]:hover {
    background-color: #334155;
    color: #e2e8f0;
}
QPushButton[role="segment"]:checked {
    background-color: #0284c7;
    color: #ffffff;
    border: 1px solid #38bdf8;
    font-weight: bold;
}

QMenu {
    background-color: #1e293b;
    color: #e2e8f0;
    border: 1px solid #475569;
    border-radius: 6px;
    padding: 4px;
}
QMenu::item {
    padding: 6px 20px 6px 12px;
    border-radius: 4px;
}
QMenu::item:selected {
    background-color: #0284c7;
    color: #ffffff;
}
QMenu::separator {
    height: 1px;
    background: #334155;
    margin: 4px 6px;
}

QTabBar::tab {
    background-color: #1e1e24;
    color: #aaaaaa;
    padding: 6px 8px;
    font-size: 12px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
}

QTabBar::tab:selected {
    background-color: #252530;
    color: #4da6ff;
    font-weight: bold;
}

/* スリム・ダークスクロールバー (幅ブレ完全防止・省スペース) */
QScrollBar:vertical {
    background: #0f172a;
    width: 8px;
    margin: 0px 0px 0px 0px;
}
QScrollBar::handle:vertical {
    background: #334155;
    min-height: 25px;
    border-radius: 4px;
}
QScrollBar::handle:vertical:hover {
    background: #475569;
}
QScrollBar::handle:vertical:pressed {
    background: #0284c7;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
    background: none;
}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
    background: none;
}
"""
