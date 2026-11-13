# -*- coding: utf-8 -*-
"""
デザイントークン（新UI）と QSS 生成

色・文字サイズ・角丸・間隔の唯一の定義場所。QSS はここから生成する（gui/styles.py 経由で取得）。
設計方針は docs/DESIGN_STYLE_PLAN.md を参照。

テーマ:
  - "dark"   : 既定。slate / sky 系のダーク（Web 版と同系統）
  - "kiseki" : 「Stand 軌跡」デザインシステム（https://claude.ai/artifact/8jT43JDYDHk9hRCaZqHFR7）の見た目を
               本ツールの役割に写したライトテーマ。明るい地に墨の文字、箱の頭は黒い帯、主ボタンは深緑、
               選択中は信号赤、角丸 3px・影なし、数字は等幅。

ルール:
  - 色は「役割」で選ぶ（primary / danger / warning / success / info / tone）。機能ごとの色分けは tone で控えめに。
  - 文字サイズは 12 / 13 / 15 / 18 の 4 段のみ（10〜11px は使わない）。
  - 角丸・間隔はトークンから。セレクタ無しのインライン QSS は書かない。役割は setProperty("role"/"tone", ...) で表現する。
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


# 機能別の控えめな色分け（tone）。値: (背景, 文字, 枠, hover 背景, ON/pressed の塗り)
# 意味: violet=塊の選択・壁合わせ, teal=精密除去, amber=粗取り・地面, green=復元・平面・粗取りプリセット, blue=標準・移動
TONES_DARK: Dict[str, Tuple[str, str, str, str, str]] = {
    "violet": ("#2e1065", "#e9d5ff", "#7c3aed", "#3b1a7a", "#7c3aed"),
    "teal":   ("#042f2e", "#99f6e4", "#0d9488", "#064e4a", "#0d9488"),
    "amber":  ("#3b2a06", "#fde68a", "#d97706", "#4d3608", "#d97706"),
    "green":  ("#052e16", "#bbf7d0", "#16a34a", "#0a3d1f", "#16a34a"),
    "blue":   ("#0c2a3f", "#bae6fd", "#0284c7", "#10364f", "#0284c7"),
    "rose":   ("#4c0519", "#fecdd3", "#e11d48", "#5e0a22", "#e11d48"),
}
# 軌跡（明）: 薄い地に濃い文字・濃い枠。ON は濃い塗りに白字
TONES_KISEKI: Dict[str, Tuple[str, str, str, str, str]] = {
    "violet": ("#ece6f7", "#4a2a8f", "#5e35b1", "#e0d6f2", "#5e35b1"),
    "teal":   ("#d9edde", "#0a5c42", "#007a55", "#cbe5d3", "#007a55"),
    "amber":  ("#fdf8d1", "#6a4700", "#7a5200", "#f7efb8", "#7a5200"),
    "green":  ("#d9edde", "#19663f", "#19663f", "#cbe5d3", "#19663f"),
    "blue":   ("#d6e3f8", "#1d4fa8", "#1d5fa8", "#c7d8f3", "#1d5fa8"),
    "rose":   ("#f7dada", "#9a2f22", "#9a2f22", "#f2c9c9", "#9a2f22"),
}
TONES = TONES_DARK  # 後方互換


@dataclass(frozen=True)
class Tokens:
    name: str = "dark"
    is_light: bool = False
    # 背景
    bg_canvas: str = "#0f172a"
    bg_surface: str = "#172033"
    bg_elevated: str = "#1e293b"
    bg_input: str = "#0b1220"
    hover_bg: str = "#263247"
    pressed_bg: str = "#0f1a2e"
    # 枠線
    border_subtle: str = "#263247"
    border_strong: str = "#3b4a63"
    # 文字
    text_primary: str = "#e2e8f0"
    text_secondary: str = "#a3b1c6"
    text_disabled: str = "#6b7a90"
    text_on_accent: str = "#ffffff"
    heading_text: str = "#38bdf8"          # 箱の見出し
    group_header_bg: str = "transparent"   # 箱の頭（kiseki では黒い帯）
    # 帯（箱の頭・上部/下部バー・ステータスバー）
    band_bg: str = "#1e293b"
    band_text: str = "#e2e8f0"
    band_muted: str = "#a3b1c6"
    band_border: str = "#263247"
    # アクセント
    accent_base: str = "#0ea5e9"
    accent_strong: str = "#0369a1"
    accent_soft: str = "#38bdf8"
    accent_tint: str = "rgba(14, 165, 233, 0.14)"
    # 主ボタン
    primary_bg: str = "#0ea5e9"
    primary_bg_hover: str = "#38bdf8"
    primary_bg_pressed: str = "#0284c7"
    primary_text: str = "#06283d"
    primary_border: str = "#bae6fd"
    # 選択中 (ON / checked)
    on_bg: str = "#0ea5e9"
    on_text: str = "#06283d"
    on_border: str = "#bae6fd"
    # 意味色
    danger_base: str = "#ef4444"
    danger_strong: str = "#b91c1c"
    danger_hover: str = "#dc2626"
    warning_base: str = "#f59e0b"
    success_base: str = "#22c55e"
    success_hover: str = "#4ade80"
    success_text: str = "#052e16"
    info_base: str = "#38bdf8"
    # トースト（帯の上の文字色: 種類別）
    toast_bg: str = "#1e293b"
    toast_info: str = "#bae6fd"
    toast_success: str = "#86efac"
    toast_warning: str = "#fde68a"
    toast_error: str = "#fca5a5"
    danger_text: str = "#fca5a5"   # 入力の誤りなど、地の上に書く注意の文字
    # 選択バッジ（選択中の強調）
    badge_bg: str = "#3b2a06"
    badge_text: str = "#fde68a"
    badge_border: str = "#f59e0b"
    # 空状態カード
    card_bg: str = "rgba(23, 32, 51, 0.92)"
    card_border: str = "#3b4a63"
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
    radius_band: int = 6     # 帯（箱の頭）の角
    control_h_sm: int = 28
    control_h_md: int = 32
    control_h_lg: int = 36
    sidebar_min: int = 320
    sidebar_default: int = 400
    tones: Dict[str, Tuple[str, str, str, str, str]] = field(default_factory=lambda: dict(TONES_DARK))


TOKENS = Tokens()

# 「Stand 軌跡」の見た目を本ツールの役割に写したライトテーマ
KISEKI_TOKENS = Tokens(
    name="kiseki",
    is_light=True,
    bg_canvas="#e7efec",        # ground
    bg_surface="#f7fbf9",       # panel
    bg_elevated="#ffffff",      # paper
    bg_input="#ffffff",
    hover_bg="#eef4f1",
    pressed_bg="#dfe6e3",       # well
    border_subtle="#cfd8d4",    # line (不透明化)
    border_strong="#5c6964",    # line-strong
    text_primary="#0f1a17",     # ink
    text_secondary="#4a534e",   # muted
    text_disabled="#6f7a75",
    text_on_accent="#ffffff",
    heading_text="#dfe9e5",     # bar-text（見出しは黒い帯の上）
    group_header_bg="#0b1110",  # 箱の頭は黒い帯（frame）
    band_bg="#0b1110",          # frame
    band_text="#dfe9e5",        # bar-text
    band_muted="#9fb3ac",       # bar-muted
    band_border="#0b1110",
    accent_base="#c0392b",      # signal
    accent_strong="#9a2f22",    # loss
    accent_soft="#1d5fa8",      # rank2（リンク・焦点）
    accent_tint="rgba(192, 57, 43, 0.10)",
    primary_bg="#004c2c",       # cta
    primary_bg_hover="#0a6a40",
    primary_bg_pressed="#003a21",
    primary_text="#ffffff",
    primary_border="#004c2c",
    on_bg="#c0392b",            # signal: 選んだ物
    on_text="#ffffff",
    on_border="#9a2f22",
    danger_base="#c0392b",
    danger_strong="#9a2f22",
    danger_hover="#b3362a",
    warning_base="#7a5200",
    success_base="#19663f",     # gain
    success_hover="#1f7a4c",
    success_text="#ffffff",
    info_base="#1d5fa8",
    toast_bg="#0b1110",
    toast_info="#dfe9e5",
    toast_success="#5cd69c",    # bar-gain
    toast_warning="#eadc62",    # bar-accent
    toast_error="#e27b6c",      # bar-loss
    danger_text="#9a2f22",      # loss
    badge_bg="#0b1110",
    badge_text="#eadc62",
    badge_border="#0b1110",
    card_bg="rgba(247, 251, 249, 0.96)",
    card_border="#5c6964",
    font_ui='"Noto Sans JP", "Yu Gothic UI", "Meiryo UI", sans-serif',
    font_mono='"IBM Plex Mono", "Cascadia Mono", "Consolas", monospace',
    radius_sm=3,
    radius_md=3,
    radius_lg=9,                # r-chip
    radius_band=0,              # 帯は角丸なし
    sidebar_default=430,        # Noto Sans JP は字幅が広いので少し広く
    tones=dict(TONES_KISEKI),
)

THEMES: Dict[str, Tokens] = {"dark": TOKENS, "kiseki": KISEKI_TOKENS}
THEME_LABELS: Dict[str, str] = {"dark": "ダーク（標準）", "kiseki": "軌跡（明）"}


def get_tokens(theme: str = "dark") -> Tokens:
    return THEMES.get(theme or "dark", TOKENS)


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
    pairs = {
        "本文 / canvas": (t.text_primary, t.bg_canvas, 4.5),
        "本文 / surface": (t.text_primary, t.bg_surface, 4.5),
        "本文 / elevated": (t.text_primary, t.bg_elevated, 4.5),
        "補足 / surface": (t.text_secondary, t.bg_surface, 4.5),
        "補足 / elevated": (t.text_secondary, t.bg_elevated, 4.5),
        "無効 / surface": (t.text_disabled, t.bg_surface, 3.0),
        "無効 / elevated": (t.text_disabled, t.bg_elevated, 3.0),
        "見出し / 箱の頭": (t.heading_text, t.bg_surface if t.group_header_bg == "transparent" else t.group_header_bg, 4.5),
        "帯の文字": (t.band_text, t.band_bg, 4.5),
        "帯の補足": (t.band_muted, t.band_bg, 4.5),
        "primary ボタン": (t.primary_text, t.primary_bg, 4.5),
        "primary ボタン hover": (t.primary_text, t.primary_bg_hover, 4.5),
        "選択中(ON)": (t.on_text, t.on_bg, 4.5),
        "danger ボタン": (t.text_on_accent, t.danger_strong, 4.5),
        "success 実行ボタン": (t.success_text, t.success_base, 4.5),
        "入力欄": (t.text_primary, t.bg_input, 4.5),
        "トースト info": (t.toast_info, t.toast_bg, 4.5),
        "トースト success": (t.toast_success, t.toast_bg, 4.5),
        "トースト warning": (t.toast_warning, t.toast_bg, 4.5),
        "トースト error": (t.toast_error, t.toast_bg, 4.5),
        "選択バッジ": (t.badge_text, t.badge_bg, 4.5),
        # ランチャーの選択行・パレットのタイル（hover）・説明書のリンク・パネルドック
        "本文 / hover": (t.text_primary, t.hover_bg, 4.5),
        "補足 / hover": (t.text_secondary, t.hover_bg, 4.5),
        "リンク / surface": (t.accent_soft, t.bg_surface, 4.5),
        "見出し帯の補足（別窓ボタン）": (t.band_muted, t.bg_surface if t.group_header_bg == "transparent" else t.group_header_bg, 4.5),
        # ボタンの役割と状態（文字だけのボタン・通知の「元に戻す」・入力の誤り・完了）
        "リンク / canvas": (t.accent_soft, t.bg_canvas, 4.5),
        "リンク / elevated": (t.accent_soft, t.bg_elevated, 4.5),
        "リンク / hover": (t.accent_soft, t.hover_bg, 4.5),
        "エラー文字 / surface": (t.danger_text, t.bg_surface, 4.5),
        "エラー文字 / elevated": (t.danger_text, t.bg_elevated, 4.5),
        "通知のボタン hover": (t.toast_bg, t.toast_info, 4.5),
        "完了ボタン": (t.success_text, t.success_base, 4.5),
    }
    for name, vals in t.tones.items():
        pairs[f"tone {name}"] = (vals[1], vals[0], 4.5)
        pairs[f"tone {name} ON"] = (t.text_on_accent, vals[4], 3.0)
    return pairs


# ----------------------------------------------------------------------------- QSS 生成
def _tone_qss(t: Tokens) -> str:
    parts = []
    for name, (bg, fg, border, hover, on_fill) in t.tones.items():
        parts.append(
            f"""
QPushButton[tone="{name}"] {{
    background-color: {bg};
    color: {fg};
    border: 1px solid {border};
    font-size: {t.size_xs}px;
    font-weight: 600;
    padding: 4px 7px;
}}
QPushButton[tone="{name}"]:hover {{
    background-color: {hover};
}}
QPushButton[tone="{name}"]:pressed, QPushButton[tone="{name}"]:checked {{
    background-color: {on_fill};
    color: #ffffff;
    border: 1px solid #ffffff;
}}
QPushButton[tone="{name}"]:disabled {{
    background-color: {t.bg_surface};
    color: {t.text_disabled};
    border-color: {t.border_subtle};
}}"""
        )
    return "\n".join(parts)


def _extras_qss(t: Tokens) -> str:
    """コマンドランチャー・右クリックのパレット・操作ガイド・パネルドック・小窓・操作説明書の QSS"""
    return f"""
/* ===== コマンドランチャー / クイックパレット ===== */
QFrame#launcher, QFrame#quick_palette {{
    background-color: {t.bg_elevated};
    border: 1px solid {t.border_strong};
    border-radius: {t.radius_lg}px;
}}
QLineEdit#launcher_search {{
    background-color: {t.bg_input};
    color: {t.text_primary};
    border: 1px solid {t.accent_base};
    border-radius: {t.radius_md}px;
    padding: 8px 10px;
    font-size: {t.size_md}px;
    min-height: 22px;
}}
QListWidget#launcher_list {{
    background-color: {t.bg_elevated};
    border: none;
    outline: none;
}}
QLabel#launcher_footer, QLabel#palette_title {{
    color: {t.text_secondary};
    font-size: {t.size_xs}px;
    background: transparent;
}}
QLabel#palette_title {{
    font-weight: 700;
}}
QPushButton[role="tile"] {{
    background-color: {t.bg_surface};
    color: {t.text_primary};
    border: 1px solid {t.border_strong};
    border-radius: {t.radius_md}px;
    padding: 6px 10px;
    min-width: 84px;
    min-height: 36px;
    font-size: {t.size_sm}px;
    font-weight: 600;
}}
QPushButton[role="tile"]:hover {{
    background-color: {t.hover_bg};
    border-color: {t.accent_base};
}}
QPushButton[role="tile"]:pressed {{
    background-color: {t.pressed_bg};
}}
QPushButton[role="tile"]:disabled {{
    background-color: {t.bg_elevated};
    color: {t.text_disabled};
    border-color: {t.border_subtle};
}}
QPushButton[role="tile"][danger="true"] {{
    border-color: {t.danger_base};
}}
QPushButton[role="tile"][danger="true"]:hover {{
    background-color: {t.danger_strong};
    color: {t.text_on_accent};
}}

/* ===== モード別の操作ガイド（3D 画面の右上） ===== */
QFrame#mode_hint {{
    background-color: {t.toast_bg};
    border: 1px solid {t.border_strong};
    border-radius: {t.radius_md}px;
}}
QLabel#mode_hint_title {{
    color: {t.toast_info};
    font-size: {t.size_sm}px;
    font-weight: 700;
    background: transparent;
}}
QLabel#mode_hint_body {{
    color: {t.toast_info};
    font-size: {t.size_xs}px;
    background: transparent;
}}

/* ===== 状況に応じたヒントのカード（3D 画面の右上）・ヒント集 ===== */
QFrame#hint_card {{
    background-color: {t.bg_elevated};
    border: 1px solid {t.accent_soft};
    border-radius: {t.radius_md}px;
}}
QLabel#hint_card_title, QLabel#tips_counter {{
    color: {t.text_secondary};
    font-size: {t.size_xs}px;
    font-weight: 700;
    background: transparent;
}}
QLabel#hint_card_text {{
    color: {t.text_primary};
    font-size: {t.size_sm}px;
    background: transparent;
}}
QLabel#tips_text {{
    color: {t.text_primary};
    font-size: {t.size_md}px;
    background: transparent;
}}
QLabel#empty_state_tip {{
    color: {t.text_secondary};
    font-size: {t.size_xs}px;
    background: transparent;
}}

/* ===== パネルドック（左パネル格納時）・小窓 ===== */
QFrame#panel_dock {{
    background-color: {t.band_bg};
    border: 1px solid {t.border_strong};
    border-radius: {t.radius_md}px;
}}
QPushButton[role="dock"] {{
    background-color: {t.band_bg};
    color: {t.band_text};
    border: none;
    border-radius: {t.radius_sm}px;
    padding: 6px 8px;
    min-width: 46px;
    font-size: {t.size_xs}px;
    font-weight: 600;
}}
QPushButton[role="dock"]:hover, QPushButton[role="dock"]:checked {{
    background-color: {t.on_bg};
    color: {t.on_text};
}}
QPushButton#collapsible_popout {{
    background: transparent;
    border: 1px solid {t.band_muted};
    border-radius: {t.radius_sm}px;
    color: {t.band_muted};
    font-size: {t.size_xs}px;
    padding: 0px 6px;
    min-height: 18px;
    max-height: 20px;
}}
QPushButton#collapsible_popout:hover {{
    color: {t.band_text};
    border-color: {t.band_text};
}}
QWidget#popout_host {{
    background-color: {t.bg_canvas};
}}

/* ===== 操作説明書 ===== */
QTextBrowser#manual_view {{
    background-color: {t.bg_surface};
    color: {t.text_primary};
    border: 1px solid {t.border_subtle};
    border-radius: {t.radius_md}px;
    padding: 6px;
    font-family: {t.font_ui};
    font-size: {t.size_sm}px;
}}
QPushButton#menu_launcher_button {{
    background-color: {t.bg_elevated};
    color: {t.text_primary};
    border: 1px solid {t.border_strong};
    border-radius: {t.radius_md}px;
    padding: 2px 12px;
    margin: 3px 8px 3px 8px;
    min-height: 18px;
    font-size: {t.size_xs}px;
    font-weight: 600;
}}
QPushButton#menu_launcher_button:hover {{
    background-color: {t.hover_bg};
    border-color: {t.accent_base};
}}
QListWidget#manual_topics {{
    background-color: {t.bg_surface};
    color: {t.text_primary};
    border: 1px solid {t.border_subtle};
    border-radius: {t.radius_md}px;
    outline: none;
    font-size: {t.size_sm}px;
}}
QListWidget#manual_topics::item {{
    padding: 7px 8px;
}}
QListWidget#manual_topics::item:hover {{
    background-color: {t.hover_bg};
}}
QListWidget#manual_topics::item:selected {{
    background-color: {t.on_bg};
    color: {t.on_text};
}}
"""


def _with_alpha(hex_color: str, alpha: float) -> str:
    """#rrggbb を rgba(r, g, b, alpha) にする"""
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)}, {int(h[2:4], 16)}, {int(h[4:6], 16)}, {alpha})"


def _button_states_qss(t: Tokens) -> str:
    """
    ボタンの役割と状態、状態表示の部品（「UIボタン20選」「UIの状態表示20選」を参考にした一覧）。
    QSS の最後に置く（同じ強さの指定は後に書いたものが勝つため、役割ごとの指定を状態で上書きできる）。

    役割 (role): primary=主操作（塗りつぶし）/ outline=補助 / soft=低優先（淡い塗り）/ ghost=最も軽い（文字だけ）
                 / danger=削除など取り返しに注意が要る操作 / pill=検索・絞り込み / segment=切り替え
    状態       : hover / pressed / disabled / checked に加えて、
                 kbfocus="true"（キーボードで移動中の現在位置）・busy="true"（処理中）・done="true"（完了）
    色だけに頼らない: 状態は文字（「保存中…」「保存済み」）と印（アイコン）でも伝える。gui/button_kit.py が付ける
    """
    ring = t.accent_soft
    band_pressed = _with_alpha(t.band_text, 0.16)  # 帯の上で押している間の地（帯の文字色を薄く重ねる）
    return f"""
/* ===== 役割の追加: 淡い塗り（低優先） ===== */
QPushButton[role="soft"] {{
    background-color: {t.hover_bg};
    color: {t.text_primary};
    border: 1px solid {t.border_subtle};
    font-weight: 600;
}}
QPushButton[role="soft"]:hover {{
    border-color: {t.border_strong};
}}
QPushButton[role="soft"]:pressed {{
    background-color: {t.pressed_bg};
}}
QPushButton[role="soft"]:disabled {{
    background-color: {t.bg_surface};
    color: {t.text_disabled};
    border-color: {t.border_subtle};
}}

/* ===== 役割: 文字だけ（最も軽い操作）。リンクの色で書き、枠は出さない ===== */
QPushButton[role="ghost"] {{
    background-color: transparent;
    border: 1px solid transparent;
    color: {t.accent_soft};
    font-weight: 600;
}}
QPushButton[role="ghost"]:hover {{
    background-color: {t.hover_bg};
    border-color: {t.border_subtle};
    text-decoration: underline;
}}
QPushButton[role="ghost"]:pressed {{
    background-color: {t.pressed_bg};
}}
QPushButton[role="ghost"]:disabled {{
    color: {t.text_disabled};
    background-color: transparent;
    border-color: transparent;
}}

/* ===== 役割: ピル型（検索・絞り込み・その場の小さな選択） ===== */
QPushButton[role="pill"] {{
    background-color: {t.bg_elevated};
    color: {t.text_primary};
    border: 1px solid {t.border_strong};
    border-radius: 12px;
    padding: 4px 12px;
    font-size: {t.size_xs}px;
    font-weight: 600;
}}
QPushButton[role="pill"]:hover {{
    background-color: {t.hover_bg};
    border-color: {t.accent_base};
}}
QPushButton[role="pill"]:pressed {{
    background-color: {t.pressed_bg};
}}
QPushButton[role="pill"]:checked {{
    background-color: {t.on_bg};
    color: {t.on_text};
    border-color: {t.on_border};
}}
QPushButton[role="pill"]:disabled {{
    background-color: {t.bg_surface};
    color: {t.text_disabled};
    border-color: {t.border_subtle};
}}
QPushButton#menu_launcher_button {{
    border-radius: 11px;
}}

/* ===== 状態の抜けを埋める（押下・無効・ホバー） ===== */
QPushButton[role="outline"]:pressed {{
    background-color: {t.pressed_bg};
}}
QPushButton[role="danger"]:pressed {{
    background-color: {t.danger_strong};
    border-color: {t.text_on_accent};
}}
QPushButton[role="danger-toggle"]:hover {{
    border-color: {t.danger_base};
}}
QPushButton[role="danger-toggle"]:disabled {{
    color: {t.text_disabled};
    border-color: {t.border_subtle};
}}
QPushButton[role="segment"]:pressed {{
    background-color: {t.pressed_bg};
}}
/* 選択中のものを押している間は、選択中の色のままにする（地だけ押下の色になると、白い文字が薄い地に乗って読めない） */
QPushButton[role="segment"]:checked:pressed, QPushButton[role="pill"]:checked:pressed {{
    background-color: {t.on_bg};
    color: {t.on_text};
}}
QPushButton[role="segment"]:disabled {{
    color: {t.text_disabled};
    border-color: {t.border_subtle};
}}
QPushButton[role="dock"]:pressed {{
    background-color: {t.pressed_bg};
    color: {t.text_primary};
}}

/* ===== 下部バー（帯の上）: 高さを揃え、無効は「文字だけ」にして押せないことを形でも示す ===== */
/* 帯の上のボタンは横の余白を詰める: 3D 画面の幅が 1,000px（横 1,450px の画面で左パネルを開いた状態）でも
   上下のバーが 1 段に収まるようにする。収まらないとボタン 1〜2 個のために 2 段目ができ、3D 画面が狭くなる */
QWidget#nav_bar QPushButton {{
    padding: 4px 5px;
}}
QWidget#nav_bar QPushButton[role="segment"] {{
    padding: 4px 4px;
}}
QWidget#bottom_bar QPushButton {{
    min-height: {t.control_h_sm - 10}px;
    font-size: {t.size_xs}px;
    padding: 4px 6px;
}}
QWidget#bottom_bar QPushButton[role="outline"] {{
    background-color: transparent;
    color: {t.band_text};
    border: 1px solid {t.band_muted};
}}
QWidget#bottom_bar QPushButton[role="outline"]:hover {{
    border-color: {t.band_text};
}}
QWidget#bottom_bar QPushButton:disabled {{
    background-color: transparent;
    color: {t.band_muted};
    border: 1px solid transparent;
}}
/* 帯の上の押下・ホバー（帯用の指定は強いので、役割ごとの :pressed / :hover は帯の上では効かない。ここで別に書く） */
QWidget#nav_bar QPushButton[role="segment"]:pressed, QWidget#nav_bar QPushButton[role="outline"]:pressed,
QWidget#nav_bar QPushButton[role="danger-toggle"]:pressed, QWidget#bottom_bar QPushButton[role="outline"]:pressed {{
    background-color: {band_pressed};
}}
QWidget#nav_bar QPushButton[role="segment"]:checked:pressed {{
    background-color: {t.on_bg};
    color: {t.on_text};
}}
QWidget#nav_bar QPushButton[role="danger-toggle"]:hover {{
    border-color: {t.danger_base};
}}
QWidget#nav_bar QPushButton[role="danger-toggle"]:checked:pressed {{
    background-color: {t.danger_hover};
}}

/* ===== キーボードで移動中の現在位置（Tab / Shift+Tab）。マウスで押しただけでは出さない ===== */
QPushButton[kbfocus="true"], QPushButton[kbfocus="true"]:hover, QPushButton[kbfocus="true"]:checked {{
    border: 2px solid {ring};
    padding: 3px 6px;
}}
QPushButton[role="primary"][kbfocus="true"], QPushButton[role="danger"][kbfocus="true"] {{
    border: 2px solid {t.text_primary};
}}
QPushButton[role="segment"][kbfocus="true"] {{
    padding: 3px 5px;
}}
QPushButton[role="tile"][kbfocus="true"] {{
    padding: 5px 9px;
}}
QPushButton[role="pill"][kbfocus="true"] {{
    padding: 3px 11px;
}}
QPushButton[role="dock"][kbfocus="true"] {{
    border: 2px solid {t.band_text};
    padding: 4px 6px;
}}
QPushButton#execute_button[kbfocus="true"] {{
    border: 2px solid {t.text_primary};
    padding: 7px 15px;
}}
QPushButton#menu_launcher_button[kbfocus="true"] {{
    border: 2px solid {ring};
    padding: 1px 11px;
}}
QWidget#nav_bar QPushButton[kbfocus="true"], QWidget#nav_bar QPushButton[kbfocus="true"]:checked,
QWidget#bottom_bar QPushButton[kbfocus="true"] {{
    border: 2px solid {t.band_text};
}}
QWidget#bottom_bar QPushButton[kbfocus="true"] {{
    padding: 3px 5px;
}}
QWidget#nav_bar QPushButton[kbfocus="true"] {{
    padding: 3px 4px;
}}
QWidget#nav_bar QPushButton[role="segment"][kbfocus="true"] {{
    padding: 3px 3px;
}}

/* ===== 処理中・完了（文字とアイコンは gui/button_kit.py が差し替える） ===== */
QPushButton[busy="true"], QPushButton[busy="true"]:disabled, QPushButton#execute_button[busy="true"]:disabled {{
    background-color: {t.hover_bg};
    color: {t.text_primary};
    border: 1px solid {t.border_strong};
}}
QPushButton[done="true"], QPushButton[done="true"]:hover, QPushButton[done="true"]:disabled {{
    background-color: {t.success_base};
    color: {t.success_text};
    border: 1px solid {t.success_base};
}}

/* ===== 分割ボタン（主操作 + 関連する操作のメニュー） ===== */
QPushButton[split="main"] {{
    border-top-right-radius: 0px;
    border-bottom-right-radius: 0px;
}}
QPushButton[split="arrow"] {{
    border-top-left-radius: 0px;
    border-bottom-left-radius: 0px;
    border-left: 1px solid {t.primary_text};
    padding: 4px 5px;
    min-width: 16px;
}}
QPushButton[split="arrow"]::menu-indicator {{
    image: none;
    width: 0px;
}}
QPushButton[split="arrow"][kbfocus="true"] {{
    padding: 3px 4px;
}}

/* ===== 通知（下部バーの右）: 印 + 文 + 次の一手（「元に戻す」など） ===== */
QFrame#toast_host {{
    background-color: {t.toast_bg};
    border: 1px solid {t.toast_info};
    border-radius: {t.radius_lg}px;
}}
QFrame#toast_host[kind="success"] {{
    border-color: {t.toast_success};
}}
QFrame#toast_host[kind="warning"] {{
    border-color: {t.toast_warning};
}}
QFrame#toast_host[kind="error"] {{
    border-color: {t.toast_error};
}}
QFrame#toast_host QLabel#toast_label {{
    background-color: transparent;
    border: none;
    padding: 2px 2px;
}}
QLabel#toast_icon {{
    background: transparent;
    border: none;
}}
QPushButton#toast_action {{
    background-color: transparent;
    color: {t.toast_info};
    border: 1px solid {t.toast_info};
    border-radius: 8px;
    padding: 1px 10px;
    min-height: 14px;
    font-size: {t.size_xs}px;
    font-weight: 700;
}}
QPushButton#toast_action:hover {{
    background-color: {t.toast_info};
    color: {t.toast_bg};
}}

/* ===== 未保存の変更（ステータスバー）。押すと保存 ===== */
QPushButton#unsaved_pill {{
    background-color: {t.badge_bg};
    color: {t.badge_text};
    border: 1px solid {t.badge_border};
    border-radius: 9px;
    padding: 1px 10px;
    min-height: 16px;
    font-size: {t.size_xs}px;
    font-weight: 700;
}}
QPushButton#unsaved_pill:hover {{
    border-color: {t.badge_text};
}}

/* ===== 入力の誤り（直す場所を枠で、理由を文で示す） ===== */
QLineEdit[invalid="true"] {{
    border: 2px solid {t.danger_base};
    padding: 3px 7px;
}}
QLabel#field_error {{
    color: {t.danger_text};
    font-size: {t.size_xs}px;
    font-weight: 600;
    background: transparent;
}}
QLabel#field_error_icon, QLabel#state_icon {{
    background: transparent;
    border: none;
}}
"""


def build_modern_qss(t: Tokens = TOKENS) -> str:
    """新UIの QSS を生成する（QApplication 全体に適用する想定。ダイアログ・ツールチップも対象）"""
    gl_bg_note = "/* light theme */" if t.is_light else "/* dark theme */"
    return f"""
/* ===== 基本 ({t.name}) ===== */ {gl_bg_note}
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
    background-color: {t.band_bg};
    color: {t.band_text};
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
    background-color: {t.accent_base};
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
QWidget#collapsible_content {{
    background: transparent;
}}
QWidget#collapsible_header {{
    background-color: {t.group_header_bg};
    border-radius: {t.radius_band}px;
}}
QLabel#collapsible_title {{
    color: {t.heading_text};
    font-size: {t.size_md}px;
    font-weight: 700;
    background: transparent;
}}
QPushButton#collapsible_toggle {{
    background: transparent;
    border: none;
    color: {t.band_muted};
    font-size: {t.size_xs}px;
    min-width: 22px;
    max-width: 22px;
    min-height: 20px;
    padding: 0;
}}
QPushButton#collapsible_toggle:hover {{
    color: {t.band_text};
}}
QGroupBox {{
    border: 1px solid {t.border_subtle};
    border-radius: {t.radius_md}px;
    margin-top: 10px;
    padding: 12px 8px 8px 8px;
    font-weight: 600;
    color: {t.text_primary};
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
    padding: 4px 7px;
    min-height: {t.control_h_sm - 10}px;
    font-weight: 600;
}}
QPushButton:hover {{
    background-color: {t.hover_bg};
}}
QPushButton:pressed {{
    background-color: {t.pressed_bg};
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
    background-color: {t.on_bg};
    color: {t.on_text};
    border: 1px solid {t.on_border};
    font-weight: 700;
}}

QPushButton[role="primary"] {{
    background-color: {t.primary_bg};
    color: {t.primary_text};
    border: 1px solid {t.primary_border};
    min-height: {t.control_h_md - 10}px;
    font-size: {t.size_sm + 1}px;
    font-weight: 700;
}}
QPushButton[role="primary"]:hover {{
    background-color: {t.primary_bg_hover};
}}
QPushButton[role="primary"]:pressed {{
    background-color: {t.primary_bg_pressed};
}}
QPushButton[role="primary"]:disabled {{
    background-color: {t.bg_elevated};
    color: {t.text_disabled};
    border-color: {t.border_subtle};
}}

QPushButton#execute_button {{
    background-color: {t.success_base};
    color: {t.success_text};
    border: 1px solid {t.success_base};
    font-size: {t.size_md}px;
    font-weight: 700;
    min-height: {t.control_h_lg - 10}px;
    padding: 8px 16px;
}}
QPushButton#execute_button:hover {{
    background-color: {t.success_hover};
}}
QPushButton#execute_button:pressed {{
    background-color: {t.success_base};
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
    border-color: {t.accent_base};
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
    background-color: {t.danger_hover};
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
    background-color: {t.danger_hover};
    color: {t.text_on_accent};
    border: 1px solid {t.text_on_accent};
    font-weight: 700;
}}

QPushButton[role="segment"] {{
    background-color: {t.bg_elevated};
    color: {t.text_secondary};
    border: 1px solid {t.border_subtle};
    border-radius: {t.radius_md}px;
    padding: 4px 6px;
    font-size: {t.size_xs}px;
    font-weight: 600;
}}
QPushButton[role="segment"]:hover {{
    background-color: {t.hover_bg};
    color: {t.text_primary};
}}
QPushButton[role="segment"]:checked {{
    background-color: {t.on_bg};
    color: {t.on_text};
    border: 1px solid {t.on_border};
    font-weight: 700;
}}

/* ===== 機能別の控えめな色分け（tone） ===== */
{_tone_qss(t)}

/* ===== 入力系 ===== */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background-color: {t.bg_input};
    border: 1px solid {t.border_strong};
    border-radius: {t.radius_sm}px;
    padding: 4px 8px;
    color: {t.text_primary};
    min-height: {t.control_h_sm - 10}px;
    selection-background-color: {t.accent_base};
    selection-color: {t.text_on_accent};
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
    image: {icon_url("chevron-down-light.svg" if t.is_light else "chevron-down.svg")};
    width: 12px;
    height: 12px;
}}
QComboBox QAbstractItemView {{
    background-color: {t.bg_elevated};
    color: {t.text_primary};
    border: 1px solid {t.border_strong};
    selection-background-color: {t.on_bg};
    selection-color: {t.on_text};
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
    background-color: {t.hover_bg};
}}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
    image: {icon_url("chevron-up-light.svg" if t.is_light else "chevron-up.svg")};
    width: 10px; height: 10px;
}}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
    image: {icon_url("chevron-down-light.svg" if t.is_light else "chevron-down.svg")};
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
    background-color: {t.on_bg};
    border-color: {t.on_border};
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
    background: {t.on_bg};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    width: 16px;
    height: 16px;
    margin: -6px 0;
    border-radius: 8px;
    background: {t.bg_elevated};
    border: 2px solid {t.on_bg};
}}
QSlider::handle:horizontal:hover {{
    border-color: {t.accent_soft};
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
    color: {t.text_primary};
    font-weight: 700;
    border-bottom: 2px solid {t.on_bg};
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
    color: {t.text_primary};
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
    background-color: {t.on_bg};
    color: {t.on_text};
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
    background-color: {t.band_bg};
    color: {t.band_text};
    border-top: 1px solid {t.band_border};
    font-size: {t.size_xs}px;
}}
QStatusBar::item {{
    border: none;
}}
QStatusBar QLabel {{
    color: {t.band_text};
    font-size: {t.size_xs}px;
    padding: 0 6px;
    background: transparent;
}}

QProgressBar {{
    border: 1px solid {t.border_subtle};
    border-radius: {t.radius_sm}px;
    text-align: center;
    background-color: {t.bg_input};
    color: {t.text_primary};
    font-size: {t.size_xs}px;
    font-weight: 600;
    font-family: {t.font_mono};
    min-height: 16px;
    max-height: 18px;
}}
QProgressBar::chunk {{
    background-color: {t.on_bg};
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
    background: {t.text_secondary};
}}
QScrollBar::handle:vertical:pressed {{
    background: {t.on_bg};
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
    background-color: {t.band_bg};
    border-bottom: 1px solid {t.band_border};
}}
QWidget#nav_bar QPushButton {{
    color: {t.band_text};
}}
QWidget#nav_bar QPushButton[role="segment"] {{
    background-color: transparent;
    color: {t.band_muted};
    border: 1px solid {t.band_muted};
}}
QWidget#nav_bar QPushButton[role="segment"]:hover {{
    color: {t.band_text};
    border-color: {t.band_text};
}}
QWidget#nav_bar QPushButton[role="segment"]:checked {{
    background-color: {t.on_bg};
    color: {t.on_text};
    border: 1px solid {t.on_border};
}}
QWidget#nav_bar QPushButton[role="outline"], QWidget#nav_bar QPushButton[role="danger-toggle"] {{
    color: {t.band_text};
    border: 1px solid {t.band_muted};
    background-color: transparent;
}}
QWidget#nav_bar QPushButton[role="danger-toggle"]:checked {{
    background-color: {t.danger_hover};
    color: {t.text_on_accent};
    border-color: {t.text_on_accent};
}}
QWidget#bottom_bar {{
    background-color: {t.band_bg};
    border-top: 1px solid {t.band_border};
}}
QLabel#point_count_pill {{
    color: {t.band_text};
    font-weight: 700;
    font-size: {t.size_xs}px;
    font-family: {t.font_mono};
    background-color: transparent;
    border: 1px solid {t.band_muted};
    border-radius: {t.radius_lg}px;
    padding: 2px 10px;
}}
QLabel#floating_sel_badge {{
    color: {t.band_muted};
    font-size: {t.size_xs}px;
    font-weight: 600;
    padding: 2px 6px;
}}
QLabel#floating_sel_badge[active="true"] {{
    color: {t.badge_text};
    background-color: {t.badge_bg};
    border: 1px solid {t.badge_border};
    border-radius: {t.radius_md}px;
    padding: 2px 10px;
    font-family: {t.font_mono};
}}
QLabel#toast_label {{
    background-color: {t.toast_bg};
    color: {t.toast_info};
    border: 1px solid {t.toast_info};
    border-radius: {t.radius_lg}px;
    padding: 3px 12px;
    font-size: {t.size_xs}px;
    font-weight: 600;
}}
QLabel#toast_label[kind="success"] {{
    color: {t.toast_success};
    border-color: {t.toast_success};
}}
QLabel#toast_label[kind="warning"] {{
    color: {t.toast_warning};
    border-color: {t.toast_warning};
}}
QLabel#toast_label[kind="error"] {{
    color: {t.toast_error};
    border-color: {t.toast_error};
}}
QLabel#toast_label[kind="info"] {{
    color: {t.toast_info};
    border-color: {t.toast_info};
}}

/* ===== 空状態カード（3Dビュー中央） ===== */
QFrame#empty_state_card {{
    background-color: {t.card_bg};
    border: 1px dashed {t.card_border};
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
    border: 1px solid {t.border_subtle};
    border-radius: 6px;
    min-height: 12px;
    max-height: 12px;
}}
QProgressBar#progressBarModern::chunk {{
    background-color: {t.on_bg};
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
""" + _extras_qss(t) + _button_states_qss(t)
