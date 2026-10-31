# -*- coding: utf-8 -*-
"""
デザイントークン（新UI）と QSS 生成

色・文字サイズ・角丸・間隔の唯一の定義場所。QSS はここから生成する（gui/styles.py 経由で取得）。
設計方針は docs/DESIGN_STYLE_PLAN.md を参照。

ルール:
  - 色は「役割」で選ぶ（primary / danger / warning / success / info）。機能ごとの色分けはしない。
  - 文字サイズは 12 / 13 / 15 / 18 の 4 段のみ（10〜11px は使わない）。
  - 角丸は 4 / 6 / 10 の 3 種、間隔は 4 の倍数。
  - セレクタ無しのインライン QSS は書かない（子孫へ伝播して事故になる）。役割は setProperty("role", ...) で表現する。
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Dict, Tuple


def assets_dir() -> str:
    """assets/ フォルダの場所（PyInstaller 実行時は _internal 配下、通常はリポジトリ直下）"""
    frozen_base = getattr(sys, "_MEIPASS", None)
    if frozen_base:
        return os.path.join(frozen_base, "assets")
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")


def icon_url(name: str) -> str:
    """QSS の url() に渡す SVG アイコンのパス（Windows でも通る forward slash 表記）"""
    path = os.path.join(assets_dir(), "icons", name).replace("\\", "/")
    return f'url("{path}")'


@dataclass(frozen=True)
class Tokens:
    # 背景
    bg_canvas: str = "#0f172a"
    bg_surface: str = "#172033"
    bg_elevated: str = "#1e293b"
    bg_input: str = "#0b1220"
    # 枠線
    border_subtle: str = "#263247"
    border_strong: str = "#3b4a63"
    # 文字
    text_primary: str = "#e2e8f0"
    text_secondary: str = "#a3b1c6"
    text_disabled: str = "#6b7a90"
    text_on_accent: str = "#ffffff"
    # アクセント
    accent_base: str = "#0ea5e9"
    accent_strong: str = "#0369a1"   # 白文字で 4.5:1 以上を満たす濃さ（#0284c7 は 4.1 で不足）
    accent_soft: str = "#38bdf8"
    accent_tint: str = "rgba(14, 165, 233, 0.14)"
    # 意味色
    danger_base: str = "#ef4444"
    danger_strong: str = "#b91c1c"
    warning_base: str = "#f59e0b"
    success_base: str = "#22c55e"
    success_text: str = "#052e16"
    info_base: str = "#38bdf8"
    # タイポグラフィ
    font_ui: str = '"Segoe UI", "Yu Gothic UI", "Meiryo UI", sans-serif'
    font_mono: str = '"Cascadia Mono", "Consolas", monospace'
    size_xs: int = 12
    size_sm: int = 13
    size_md: int = 15
    size_lg: int = 18
    # 形状・サイズ
    radius_sm: int = 4
    radius_md: int = 6
    radius_lg: int = 10
    control_h_sm: int = 28
    control_h_md: int = 32
    control_h_lg: int = 36
    sidebar_min: int = 320
    sidebar_default: int = 380


TOKENS = Tokens()


# ----------------------------------------------------------------------------- コントラスト計算
def _lin(c: float) -> float:
    c = c / 255.0
    return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4


def luminance(hex_color: str) -> float:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)


def contrast_ratio(fg: str, bg: str) -> float:
    l1, l2 = luminance(fg), luminance(bg)
    hi, lo = max(l1, l2), min(l1, l2)
    return (hi + 0.05) / (lo + 0.05)


def text_background_pairs(t: Tokens = TOKENS) -> Dict[str, Tuple[str, str, float]]:
    """
    テストで検証する「文字色 / 背景色 / 必要コントラスト比」の組。
    4.5 = 通常文字 (AA)、3.0 = 大きな文字・無効状態・UI 部品。
    """
    return {
        "本文 / canvas": (t.text_primary, t.bg_canvas, 4.5),
        "本文 / surface": (t.text_primary, t.bg_surface, 4.5),
        "本文 / elevated": (t.text_primary, t.bg_elevated, 4.5),
        "補足 / surface": (t.text_secondary, t.bg_surface, 4.5),
        "補足 / elevated": (t.text_secondary, t.bg_elevated, 4.5),
        "無効 / surface": (t.text_disabled, t.bg_surface, 3.0),
        "無効 / elevated": (t.text_disabled, t.bg_elevated, 3.0),
        "見出し / surface": (t.accent_soft, t.bg_surface, 4.5),
        "primary ボタン": (t.text_on_accent, t.accent_strong, 4.5),
        "danger ボタン": (t.text_on_accent, t.danger_strong, 4.5),
        "success 実行ボタン": (t.success_text, t.success_base, 4.5),
        "入力欄": (t.text_primary, t.bg_input, 4.5),
        "警告 HUD 文字": (t.warning_base, t.bg_elevated, 3.0),
    }


# ----------------------------------------------------------------------------- QSS 生成
def build_modern_qss(t: Tokens = TOKENS) -> str:
    """新UIの QSS を生成する（QApplication 全体に適用する想定。ダイアログ・ツールチップも対象）"""
    return f"""
/* ===== 基本 ===== */
QMainWindow, QDialog, QMessageBox, QFileDialog {{
    background-color: {t.bg_canvas};
    color: {t.text_primary};
}}
QWidget {{
    background-color: {t.bg_canvas};
    color: {t.text_primary};
    font-family: {t.font_ui};
    font-size: {t.size_sm}px;
}}
QLabel, QCheckBox, QRadioButton {{
    background: transparent;
}}
QLabel {{
    color: {t.text_primary};
}}
QLabel[hint="true"], QLabel#hint {{
    color: {t.text_secondary};
    font-size: {t.size_xs}px;
}}
QToolTip {{
    background-color: {t.bg_elevated};
    color: {t.text_primary};
    border: 1px solid {t.border_strong};
    border-radius: {t.radius_sm}px;
    padding: 6px 8px;
    font-size: {t.size_xs}px;
}}
QSplitter::handle {{
    background-color: {t.border_subtle};
    width: 4px;
}}
QSplitter::handle:hover {{
    background-color: {t.accent_strong};
}}

/* ===== サイドバー・パネル ===== */
QScrollArea {{
    border: none;
    background: transparent;
}}
QScrollArea > QWidget > QWidget {{
    background: transparent;
}}
QWidget#sidebar_widget {{
    background-color: {t.bg_canvas};
}}
QGroupBox#collapsible_group {{
    background-color: {t.bg_surface};
    border: 1px solid {t.border_subtle};
    border-radius: {t.radius_md}px;
    margin-top: 2px;
    padding: 2px;
}}
QWidget#collapsible_content, QWidget#collapsible_header {{
    background: transparent;
}}
QLabel#collapsible_title {{
    color: {t.accent_soft};
    font-size: {t.size_md}px;
    font-weight: 700;
}}
QPushButton#collapsible_toggle {{
    background: transparent;
    border: none;
    color: {t.text_secondary};
    font-size: {t.size_xs}px;
    min-width: 22px;
    max-width: 22px;
    min-height: 20px;
    padding: 0;
}}
QPushButton#collapsible_toggle:hover {{
    color: {t.text_primary};
    background-color: {t.accent_tint};
    border-radius: {t.radius_sm}px;
}}
QGroupBox {{
    border: 1px solid {t.border_subtle};
    border-radius: {t.radius_md}px;
    margin-top: 10px;
    padding: 12px 8px 8px 8px;
    font-weight: 600;
    color: {t.accent_soft};
    background-color: {t.bg_surface};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 5px;
    background-color: {t.bg_surface};
}}
QFrame#height_range_frame {{
    background-color: {t.bg_elevated};
    border: 1px solid {t.border_subtle};
    border-radius: {t.radius_md}px;
}}

/* ===== ボタン（役割で統一） ===== */
QPushButton {{
    background-color: {t.bg_elevated};
    color: {t.text_primary};
    border: 1px solid {t.border_strong};
    border-radius: {t.radius_md}px;
    padding: 4px 10px;
    min-height: {t.control_h_sm - 10}px;
    font-weight: 600;
}}
QPushButton:hover {{
    background-color: #263247;
    border-color: #4b5b78;
}}
QPushButton:pressed {{
    background-color: #0f1a2e;
}}
QPushButton:focus {{
    border: 1px solid {t.accent_soft};
}}
QPushButton:disabled {{
    color: {t.text_disabled};
    background-color: {t.bg_surface};
    border-color: {t.border_subtle};
}}
QPushButton:checked {{
    background-color: {t.accent_strong};
    color: {t.text_on_accent};
    border-color: {t.accent_base};
}}

QPushButton[role="primary"] {{
    background-color: {t.accent_strong};
    color: {t.text_on_accent};
    border: 1px solid {t.accent_base};
    min-height: {t.control_h_md - 10}px;
}}
QPushButton[role="primary"]:hover {{
    background-color: {t.accent_base};
}}
QPushButton[role="primary"]:pressed {{
    background-color: #075985;
}}
QPushButton[role="primary"]:disabled {{
    background-color: {t.bg_elevated};
    color: {t.text_disabled};
    border-color: {t.border_subtle};
}}

QPushButton#execute_button {{
    background-color: {t.success_base};
    color: {t.success_text};
    border: 1px solid #16a34a;
    font-size: {t.size_md}px;
    font-weight: 700;
    min-height: {t.control_h_lg - 10}px;
    padding: 8px 16px;
}}
QPushButton#execute_button:hover {{
    background-color: #4ade80;
}}
QPushButton#execute_button:pressed {{
    background-color: #16a34a;
}}
QPushButton#execute_button:disabled {{
    background-color: {t.bg_elevated};
    color: {t.text_disabled};
    border-color: {t.border_subtle};
}}

QPushButton[role="outline"] {{
    background-color: transparent;
    color: {t.text_primary};
    border: 1px solid {t.border_strong};
    font-weight: 500;
}}
QPushButton[role="outline"]:hover {{
    background-color: {t.accent_tint};
    border-color: {t.accent_soft};
}}
QPushButton[role="outline"]:disabled {{
    color: {t.text_disabled};
    border-color: {t.border_subtle};
    background-color: transparent;
}}

QPushButton[role="ghost"] {{
    background-color: transparent;
    border: 1px solid transparent;
    color: {t.text_primary};
    font-weight: 500;
}}
QPushButton[role="ghost"]:hover {{
    background-color: {t.accent_tint};
    border-color: {t.border_subtle};
}}

QPushButton[role="danger"] {{
    background-color: {t.danger_strong};
    color: {t.text_on_accent};
    border: 1px solid {t.danger_base};
}}
QPushButton[role="danger"]:hover {{
    background-color: #dc2626;
}}
QPushButton[role="danger"]:disabled {{
    background-color: {t.bg_elevated};
    color: {t.text_disabled};
    border-color: {t.border_subtle};
}}

QPushButton[role="danger-toggle"] {{
    background-color: transparent;
    color: {t.text_primary};
    border: 1px solid {t.border_strong};
}}
QPushButton[role="danger-toggle"]:checked {{
    background-color: {t.danger_strong};
    color: {t.text_on_accent};
    border-color: {t.danger_base};
}}

QPushButton[role="segment"] {{
    background-color: {t.bg_elevated};
    color: {t.text_secondary};
    border: 1px solid {t.border_subtle};
    border-radius: {t.radius_md}px;
    padding: 4px 8px;
    font-size: {t.size_xs}px;
    font-weight: 600;
}}
QPushButton[role="segment"]:hover {{
    background-color: #263247;
    color: {t.text_primary};
}}
QPushButton[role="segment"]:checked {{
    background-color: {t.accent_strong};
    color: {t.text_on_accent};
    border: 1px solid {t.accent_base};
}}

/* ===== 入力系 ===== */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background-color: {t.bg_input};
    border: 1px solid {t.border_strong};
    border-radius: {t.radius_sm}px;
    padding: 4px 8px;
    color: {t.text_primary};
    min-height: {t.control_h_sm - 10}px;
    selection-background-color: {t.accent_strong};
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border: 1px solid {t.accent_soft};
}}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {{
    color: {t.text_disabled};
    border-color: {t.border_subtle};
}}
QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 22px;
    border-left: 1px solid {t.border_subtle};
    border-top-right-radius: {t.radius_sm}px;
    border-bottom-right-radius: {t.radius_sm}px;
}}
QComboBox::down-arrow {{
    image: {icon_url("chevron-down.svg")};
    width: 12px;
    height: 12px;
}}
QComboBox QAbstractItemView {{
    background-color: {t.bg_elevated};
    color: {t.text_primary};
    border: 1px solid {t.border_strong};
    selection-background-color: {t.accent_strong};
    selection-color: {t.text_on_accent};
    outline: none;
}}
QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    width: 18px;
    background-color: {t.bg_elevated};
    border-left: 1px solid {t.border_subtle};
}}
QSpinBox::up-button, QDoubleSpinBox::up-button {{
    subcontrol-position: top right;
    border-top-right-radius: {t.radius_sm}px;
}}
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    subcontrol-position: bottom right;
    border-bottom-right-radius: {t.radius_sm}px;
}}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
    background-color: #263247;
}}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
    image: {icon_url("chevron-up.svg")};
    width: 10px; height: 10px;
}}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
    image: {icon_url("chevron-down.svg")};
    width: 10px; height: 10px;
}}

QCheckBox {{
    color: {t.text_primary};
    spacing: 8px;
}}
QCheckBox::indicator {{
    width: 18px;
    height: 18px;
    border-radius: {t.radius_sm}px;
    border: 1px solid {t.border_strong};
    background-color: {t.bg_input};
}}
QCheckBox::indicator:hover {{
    border-color: {t.accent_soft};
}}
QCheckBox::indicator:checked {{
    background-color: {t.accent_strong};
    border-color: {t.accent_base};
    image: {icon_url("check.svg")};
}}
QCheckBox::indicator:disabled {{
    border-color: {t.border_subtle};
    background-color: {t.bg_surface};
}}
QCheckBox:disabled {{
    color: {t.text_disabled};
}}

QSlider::groove:horizontal {{
    height: 4px;
    background: {t.border_strong};
    border-radius: 2px;
}}
QSlider::sub-page:horizontal {{
    background: {t.accent_strong};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    width: 16px;
    height: 16px;
    margin: -6px 0;
    border-radius: 8px;
    background: {t.text_primary};
    border: 2px solid {t.accent_strong};
}}
QSlider::handle:horizontal:hover {{
    background: #ffffff;
    border-color: {t.accent_base};
}}
QSlider::handle:horizontal:disabled {{
    background: {t.text_disabled};
    border-color: {t.border_subtle};
}}

/* ===== タブ・メニュー・ステータス ===== */
QTabWidget::pane {{
    border: 1px solid {t.border_subtle};
    border-radius: {t.radius_md}px;
    background-color: {t.bg_elevated};
    top: -1px;
}}
QTabBar::tab {{
    background-color: {t.bg_surface};
    color: {t.text_secondary};
    padding: 5px 6px;
    font-size: {t.size_xs}px;
    border: 1px solid {t.border_subtle};
    border-bottom: none;
    border-top-left-radius: {t.radius_md}px;
    border-top-right-radius: {t.radius_md}px;
    margin-right: 2px;
}}
QTabBar::tab:selected {{
    background-color: {t.bg_elevated};
    color: {t.accent_soft};
    font-weight: 700;
}}
QTabBar::tab:hover:!selected {{
    color: {t.text_primary};
}}

QMenuBar {{
    background-color: {t.bg_canvas};
    color: {t.text_primary};
    border-bottom: 1px solid {t.border_subtle};
    padding: 2px 4px;
}}
QMenuBar::item {{
    background: transparent;
    padding: 4px 10px;
    border-radius: {t.radius_sm}px;
}}
QMenuBar::item:selected {{
    background-color: {t.accent_tint};
    color: {t.accent_soft};
}}
QMenu {{
    background-color: {t.bg_elevated};
    color: {t.text_primary};
    border: 1px solid {t.border_strong};
    border-radius: {t.radius_md}px;
    padding: 4px;
}}
QMenu::item {{
    padding: 6px 24px 6px 12px;
    border-radius: {t.radius_sm}px;
}}
QMenu::item:selected {{
    background-color: {t.accent_strong};
    color: {t.text_on_accent};
}}
QMenu::item:disabled {{
    color: {t.text_disabled};
}}
QMenu::separator {{
    height: 1px;
    background: {t.border_subtle};
    margin: 4px 6px;
}}

QStatusBar {{
    background-color: {t.bg_surface};
    color: {t.text_primary};
    border-top: 1px solid {t.border_subtle};
    font-size: {t.size_xs}px;
}}
QStatusBar::item {{
    border: none;
}}
QStatusBar QLabel {{
    color: {t.text_primary};
    font-size: {t.size_xs}px;
    padding: 0 6px;
}}

QProgressBar {{
    border: 1px solid {t.border_subtle};
    border-radius: {t.radius_sm}px;
    text-align: center;
    background-color: {t.bg_input};
    color: {t.text_primary};
    font-size: {t.size_xs}px;
    font-weight: 600;
    min-height: 16px;
    max-height: 18px;
}}
QProgressBar::chunk {{
    background-color: {t.accent_strong};
    border-radius: 3px;
}}

QTextEdit, QPlainTextEdit {{
    background-color: {t.bg_input};
    border: 1px solid {t.border_subtle};
    border-radius: {t.radius_md}px;
    padding: 6px;
    color: {t.text_secondary};
    font-family: {t.font_mono};
    font-size: {t.size_xs}px;
}}

/* ===== スクロールバー ===== */
QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {t.border_strong};
    min-height: 28px;
    border-radius: 5px;
}}
QScrollBar::handle:vertical:hover {{
    background: #4b5b78;
}}
QScrollBar::handle:vertical:pressed {{
    background: {t.accent_strong};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
    background: none;
    height: 0;
}}
QScrollBar:horizontal {{
    background: transparent;
    height: 10px;
    margin: 0;
}}
QScrollBar::handle:horizontal {{
    background: {t.border_strong};
    min-width: 28px;
    border-radius: 5px;
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal,
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
    background: none;
    width: 0;
}}

/* ===== 上部・下部バー、バッジ、トースト ===== */
QWidget#nav_bar {{
    background-color: {t.bg_surface};
    border-bottom: 1px solid {t.border_subtle};
}}
QWidget#bottom_bar {{
    background-color: {t.bg_surface};
    border-top: 1px solid {t.border_subtle};
}}
QLabel#point_count_pill {{
    color: {t.accent_soft};
    font-weight: 700;
    font-size: {t.size_xs}px;
    background-color: {t.bg_elevated};
    border: 1px solid {t.border_strong};
    border-radius: {t.radius_lg}px;
    padding: 2px 10px;
}}
QLabel#floating_sel_badge {{
    color: {t.text_secondary};
    font-size: {t.size_xs}px;
    font-weight: 600;
    padding: 2px 6px;
}}
QLabel#floating_sel_badge[active="true"] {{
    color: #fde68a;
    background-color: #3b2a06;
    border: 1px solid {t.warning_base};
    border-radius: {t.radius_md}px;
    padding: 2px 10px;
}}
QLabel#toast_label {{
    background-color: {t.bg_elevated};
    color: {t.text_primary};
    border: 1px solid {t.border_strong};
    border-radius: {t.radius_lg}px;
    padding: 3px 12px;
    font-size: {t.size_xs}px;
    font-weight: 600;
}}
QLabel#toast_label[kind="success"] {{
    background-color: #0b3b22;
    color: #bbf7d0;
    border-color: {t.success_base};
}}
QLabel#toast_label[kind="warning"] {{
    background-color: #3b2a06;
    color: #fde68a;
    border-color: {t.warning_base};
}}
QLabel#toast_label[kind="error"] {{
    background-color: #450a0a;
    color: #fecaca;
    border-color: {t.danger_base};
}}
QLabel#toast_label[kind="info"] {{
    background-color: #0c2a3f;
    color: #bae6fd;
    border-color: {t.info_base};
}}

/* ===== 空状態カード（3Dビュー中央） ===== */
QFrame#empty_state_card {{
    background-color: rgba(23, 32, 51, 0.92);
    border: 1px dashed {t.border_strong};
    border-radius: {t.radius_lg}px;
}}
QLabel#empty_state_title {{
    color: {t.text_primary};
    font-size: {t.size_lg}px;
    font-weight: 700;
    background: transparent;
}}
QLabel#empty_state_hint {{
    color: {t.text_secondary};
    font-size: {t.size_sm}px;
    background: transparent;
}}
QLabel#empty_state_recent_title {{
    color: {t.text_secondary};
    font-size: {t.size_xs}px;
    background: transparent;
}}

/* ===== 進捗ダイアログ ===== */
QWidget#progressContainer {{
    background-color: {t.bg_surface};
    border: 1px solid {t.border_strong};
    border-radius: {t.radius_lg}px;
}}
QLabel#progressTag {{
    color: {t.accent_soft};
    font-size: {t.size_xs}px;
    font-weight: 700;
    background: transparent;
}}
QLabel#progressPercent {{
    color: {t.text_primary};
    font-size: {t.size_lg}px;
    font-weight: 700;
    font-family: {t.font_mono};
    background: transparent;
}}
QLabel#progressTitle {{
    color: {t.text_primary};
    font-size: {t.size_sm}px;
    font-weight: 600;
    background: transparent;
}}
QLabel#progressSub {{
    color: {t.text_secondary};
    font-size: {t.size_xs}px;
    background: transparent;
}}
QPushButton#progressCancel {{
    background-color: transparent;
    color: {t.text_secondary};
    border: 1px solid {t.border_strong};
    border-radius: {t.radius_sm}px;
    padding: 2px 10px;
    font-size: {t.size_xs}px;
}}
QPushButton#progressCancel:hover {{
    background-color: {t.danger_strong};
    color: {t.text_on_accent};
    border-color: {t.danger_base};
}}
QProgressBar#progressBarModern {{
    background-color: {t.bg_input};
    border: none;
    border-radius: 6px;
    min-height: 12px;
    max-height: 12px;
}}
QProgressBar#progressBarModern::chunk {{
    background-color: {t.accent_base};
    border-radius: 6px;
}}

/* ===== メッセージボックス ===== */
QMessageBox QLabel {{
    color: {t.text_primary};
    font-size: {t.size_sm}px;
}}
QMessageBox QPushButton {{
    min-width: 84px;
}}
"""
